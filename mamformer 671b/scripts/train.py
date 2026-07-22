"""
Mamformer Training Script
======================
Training loop with FSDP, mixed precision, gradient checkpointing,
and checkpoint management.

Usage:
    # Single GPU training (debug mode)
    python scripts/train.py --config configs/debug.yaml --data ./data

    # Multi-GPU training with FSDP
    torchrun --nproc_per_node=8 scripts/train.py \\
        --config configs/7b.yaml \\
        --data /path/to/tokenized/data \\
        --batch_size 4 \\
        --gradient_accumulation_steps 8 \\
        --bf16
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Add parent to path for easy imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mamformer.config import MamformerConfig
from mamformer.model import MamformerForCausalLM
from mamformer.tokenizer import MamformerTokenizer

logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────

class TextDataset(IterableDataset):
    """
    Streaming text dataset for language model training.

    Reads pre-tokenized binary files (.bin) containing uint16 token IDs.
    Sequences are packed with EOS separators and chunked to max_seq_len.

    Args:
        data_dir: Directory containing .bin token files
        seq_len: Maximum sequence length
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 8192,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.seed = seed

        # Find all .bin files
        self.files = sorted(self.data_dir.glob("*.bin"))
        if not self.files:
            # Fallback: generate dummy data for testing
            logger.warning(f"No .bin files found in {data_dir}. Using dummy data.")
            self._use_dummy = True
        else:
            self._use_dummy = False
            logger.info(f"Found {len(self.files)} token files")

    def __iter__(self):
        if self._use_dummy:
            return self._dummy_iterator()
        return self._file_iterator()

    def _dummy_iterator(self):
        """Generate random token sequences for testing."""
        while True:
            tokens = torch.randint(0, 32000, (self.seq_len + 1,))
            yield {
                "input_ids": tokens[:-1],
                "labels": tokens[1:],
            }

    def _file_iterator(self):
        """Stream from pre-tokenized files with shuffling."""
        import random
        random.seed(self.seed)

        buffer = []
        for file_path in self.files:
            # Read uint16 binary tokens (matching prepare_data.py output)
            raw_bytes = open(file_path, 'rb').read()
            data = torch.frombuffer(bytearray(raw_bytes), dtype=torch.int16).long()
            buffer.extend(data.tolist())

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                yield {
                    "input_ids": torch.tensor(chunk[:-1]),
                    "labels": torch.tensor(chunk[1:]),
                }


# ── LR Schedule ────────────────────────────────────────────────────────

def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create a cosine LR schedule with linear warmup.

    Schedule: linear warmup → cosine decay to 10% of peak LR
    """
    warmup_steps = config.get("warmup_steps", 2000)
    max_steps = config.get("max_steps", 100000)
    min_lr_ratio = config.get("min_lr_ratio", 0.1)

    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,
        eta_min=config["learning_rate"] * min_lr_ratio,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


# ── Training Loop ─────────────────────────────────────────────────────

def train(config: dict) -> None:
    """Main training loop."""

    # ── Setup ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_distributed = int(os.environ.get("LOCAL_RANK", -1)) != -1

    if is_distributed:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    logger.info(f"Device: {device}, Distributed: {is_distributed}")

    # ── Model ──────────────────────────────────────────────────
    model_config = MamformerConfig.from_yaml(config["config_path"])
    logger.info(f"Loading model: {model_config.name}")
    logger.info(f"Parameters: {model_config.num_parameters_billions:.2f}B")

    model = MamformerForCausalLM(model_config)
    model = model.to(device)

    # Enable gradient checkpointing
    if config.get("gradient_checkpointing", True):
        model.model.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")

    # ── FSDP ────────────────────────────────────────────────────
    if is_distributed and config.get("use_fsdp", True):
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            MixedPrecision,
            BackwardPrefetch,
        )
        from torch.distributed.fsdp.wrap import (
            transformer_auto_wrap_policy,
        )
        from functools import partial

        from mamformer.layers.hybrid import MamformerBlock

        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={MamformerBlock},
        )

        mp_dtype = torch.bfloat16 if config.get("bf16", False) else torch.float16

        mixed_precision = MixedPrecision(
            param_dtype=mp_dtype,
            reduce_dtype=mp_dtype,
            buffer_dtype=mp_dtype,
            cast_forward_inputs=True,
        )

        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            limit_all_gathers=True,
        )
        logger.info("FSDP wrapping applied")

    # ── Optimizer ──────────────────────────────────────────────
    # Separate weight decay params
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name or "gate" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": config.get("weight_decay", 0.1)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.get("learning_rate", 3e-4),
        betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.95)),
        eps=config.get("adam_epsilon", 1e-8),
    )

    scheduler = create_lr_scheduler(optimizer, config)

    # ── Data ───────────────────────────────────────────────────
    dataset = TextDataset(
        data_dir=config.get("data_dir", "./data"),
        seq_len=config.get("max_seq_len", 8192),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 1),
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    # ── Training state ─────────────────────────────────────────
    global_step = 0
    max_steps = config.get("max_steps", 100000)
    grad_accum_steps = config.get("gradient_accumulation_steps", 8)
    log_every = config.get("log_every", 10)
    save_every = config.get("save_every", 5000)
    output_dir = Path(config.get("output_dir", "./checkpoints"))

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume from checkpoint ──────────────────────────────────
    resume_path = config.get("resume")
    if resume_path and Path(resume_path).exists():
        logger.info(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint.get("step", 0)
        logger.info(f"Resumed at step {global_step}")
    elif resume_path:
        logger.warning(f"Resume checkpoint not found: {resume_path}, starting from scratch")

    # Mixed precision
    amp_enabled = config.get("bf16", False) or config.get("fp16", False)
    amp_dtype = torch.bfloat16 if config.get("bf16", False) else torch.float16 if config.get("fp16", False) else None

    # ── WandB ──────────────────────────────────────────────────
    use_wandb = config.get("use_wandb", False)
    if use_wandb and (not is_distributed or torch.distributed.get_rank() == 0):
        try:
            import wandb
            wandb.init(
                project=config.get("wandb_project", "Mamformer"),
                name=config.get("wandb_run_name", model_config.name),
                config=config,
            )
            logger.info("WandB logging enabled")
        except ImportError:
            logger.warning("wandb not installed. Skipping logging.")
            use_wandb = False

    # ── Training Loop ──────────────────────────────────────────
    logger.info(f"Starting training at step {global_step}, target {max_steps}")
    model.train()
    total_loss = 0.0
    start_time = time.time()
    tokens_processed = 0
    nan_count = 0  # Track NaN steps for auto-recovery
    best_loss = float("inf")

    for batch in dataloader:
        if global_step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # ── NaN Detection (before forward) ─────────────────────
        # Check input for NaN (corrupt data)
        if torch.isnan(input_ids.float()).any() or torch.isnan(labels.float()).any():
            logger.warning(f"Step {global_step}: NaN in input data, skipping batch")
            continue

        # Mixed precision forward
        try:
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"] / grad_accum_steps
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM at step {global_step}! Clearing cache and skipping...")
                torch.cuda.empty_cache()
                # Save emergency checkpoint
                save_checkpoint(model, optimizer, scheduler, global_step, output_dir, final=False, tag="oom_recovery")
                continue
            else:
                raise

        # ── NaN Loss Detection ─────────────────────────────────
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            nan_count += 1
            logger.error(f"NaN/Inf loss at step {global_step}! (count={nan_count})")
            if nan_count >= 3:
                logger.critical("3 consecutive NaN steps — auto-resuming from last checkpoint!")
                # Try to recover from last good checkpoint
                last_ckpt = sorted(output_dir.glob("checkpoint_step_*.pt"))
                if last_ckpt:
                    checkpoint = torch.load(str(last_ckpt[-1]), map_location=device)
                    model.load_state_dict(checkpoint["model"], strict=False)
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    scheduler.load_state_dict(checkpoint["scheduler"])
                    global_step = checkpoint.get("step", global_step)
                    logger.info(f"Rolled back to step {global_step}")
                nan_count = 0
            optimizer.zero_grad()
            continue
        else:
            nan_count = 0  # Reset counter on good step

        loss.backward()

        # ── NaN Gradient Detection ──────────────────────────────
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                logger.error(f"NaN/Inf gradient in {name} at step {global_step}")
                has_nan_grad = True
                break
        if has_nan_grad:
            logger.warning("Zeroing gradients and skipping optimizer step")
            optimizer.zero_grad()
            continue

        tokens_processed += input_ids.numel()

        # Collect MTP and MoE statistics for logging
        mtp_loss_val = outputs.get("mtp_loss", None)
        main_loss_val = outputs.get("main_loss", None)
        moe_aux_info = outputs.get("moe_aux_info", None)

        if (global_step + 1) % grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps

        # Logging
        if global_step % log_every == 0 and global_step > 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / log_every
            tokens_per_sec = tokens_processed / elapsed

            # Build log message with optional MTP/MoE info
            log_parts = [
                f"Step {global_step:>6d}",
                f"Loss: {avg_loss:.4f}",
            ]
            wandb_log_dict = {
                "loss": avg_loss,
                "lr": scheduler.get_last_lr()[0],
                "tokens_per_sec": tokens_per_sec,
                "step": global_step,
            }

            # MTP loss breakdown
            if mtp_loss_val is not None and main_loss_val is not None:
                log_parts.append(f"Main: {main_loss_val.item():.4f}")
                log_parts.append(f"MTP: {mtp_loss_val.item():.4f}")
                wandb_log_dict["main_loss"] = main_loss_val.item()
                wandb_log_dict["mtp_loss"] = mtp_loss_val.item()

            # MoE load balance statistics
            if moe_aux_info is not None:
                # Average expert bias across layers
                biases = [
                    info.get("expert_bias_mean", float('nan'))
                    for info in moe_aux_info
                    if isinstance(info, dict)
                ]
                if biases:
                    avg_bias = sum(b for b in biases if not math.isnan(b)) / max(1, sum(1 for b in biases if not math.isnan(b)))
                    wandb_log_dict["moe_expert_bias_mean"] = avg_bias

            log_parts.extend([
                f"LR: {scheduler.get_last_lr()[0]:.2e}",
                f"Tokens/s: {tokens_per_sec:,.0f}",
                f"Elapsed: {elapsed:.0f}s",
            ])
            log_msg = " | ".join(log_parts)
            logger.info(log_msg)

            if use_wandb and (not is_distributed or torch.distributed.get_rank() == 0):
                wandb.log(wandb_log_dict)

            total_loss = 0.0
            tokens_processed = 0
            start_time = time.time()

        # Checkpointing
        if global_step % save_every == 0 and global_step > 0:
            if not is_distributed or torch.distributed.get_rank() == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, output_dir)
                logger.info(f"Checkpoint saved at step {global_step}")

        global_step += 1

    # Final save
    if not is_distributed or torch.distributed.get_rank() == 0:
        save_checkpoint(model, optimizer, scheduler, global_step, output_dir, final=True)
        logger.info(f"Final checkpoint saved at step {global_step}")

    if is_distributed:
        torch.distributed.destroy_process_group()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    output_dir: Path,
    final: bool = False,
    tag: str = "",
) -> str:
    """Save a training checkpoint. Returns the checkpoint path."""
    # Handle FSDP state dict
    if hasattr(model, "_is_fsdp_managed_module"):
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            StateDictType,
        )
        import torch.distributed as dist

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with model.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()

    if tag:
        suffix = f"{tag}_step_{step}"
    elif final:
        suffix = "final"
    else:
        suffix = f"step_{step}"

    ckpt_path = output_dir / f"checkpoint_{suffix}.pt"
    torch.save(
        {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        ckpt_path,
    )
    return str(ckpt_path)


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Mamformer model")
    parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--data", type=str, default="./data", help="Path to tokenized data directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Micro batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Mamformer")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config = {
        "config_path": args.config,
        "data_dir": args.data,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "max_seq_len": args.max_seq_len,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "output_dir": args.output_dir,
        "save_every": args.save_every,
        "log_every": args.log_every,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "num_workers": args.num_workers,
        "resume": args.resume,
    }

    train(config)


if __name__ == "__main__":
    main()
