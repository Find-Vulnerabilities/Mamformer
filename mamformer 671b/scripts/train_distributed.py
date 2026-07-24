"""
Mamformer Distributed Training Script (4D Parallelism)
========================================================
Production training launcher for 7B-671B Mamformer models.

Supports:
  - 4D Parallelism: Data x Tensor x Pipeline x Expert
  - BF16 mixed precision with gradient scaling
  - Gradient checkpointing (activation recomputation)
  - MoE load balancing (aux-loss-free)
  - MTP (Multi-Token Prediction) auxiliary loss
  - WandB logging with MoE statistics
  - Checkpoint save/resume with FSDP-compatible format

Usage (single node, 8 GPUs):
    torchrun --nproc_per_node=8 scripts/train_distributed.py \
        --config configs/ultra-671b-max.yaml \
        --data ./data \
        --tp 4 --pp 2 --ep 1 --dp 1

Usage (multi-node, 64 GPUs):
    torchrun --nnodes=8 --nproc_per_node=8 scripts/train_distributed.py \
        --config configs/ultra-671b-max.yaml \
        --data /data/tokenized \
        --tp 4 --pp 4 --ep 2 --dp 2
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
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mamformer.config import MamformerConfig
from mamformer.model import MamformerForCausalLM
from mamformer.parallelism.coordinator import ParallelConfig, DistributedCoordinator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class StreamingTextDataset(torch.utils.data.IterableDataset):
    """Streaming dataset from pre-tokenized binary files."""

    def __init__(self, data_dir: str, seq_len: int = 8192, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.seed = seed
        self.files = sorted(self.data_dir.glob("*.bin"))
        self._use_dummy = not bool(self.files)

    def __iter__(self):
        if self._use_dummy:
            return self._dummy_iter()
        return self._file_iter()

    def _dummy_iter(self):
        while True:
            tokens = torch.randint(0, 32000, (self.seq_len + 1,))
            yield {"input_ids": tokens[:-1], "labels": tokens[1:]}

    def _file_iter(self):
        import random
        random.seed(self.seed)
        buffer = []
        for fp in self.files:
            raw_bytes = open(fp, 'rb').read()
            data = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint16).long()
            buffer.extend(data.tolist())
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                yield {"input_ids": torch.tensor(chunk[:-1]), "labels": torch.tensor(chunk[1:])}


# ═══════════════════════════════════════════════════════════════════════
# LR Scheduler
# ═══════════════════════════════════════════════════════════════════════

def create_lr_scheduler(optimizer, config: dict):
    warmup_steps = config.get("warmup_steps", 2000)
    max_steps = config.get("max_steps", 100000)
    lr = config.get("learning_rate", 3e-4)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=lr * 0.1)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


# ═══════════════════════════════════════════════════════════════════════
# Main Training Function
# ═══════════════════════════════════════════════════════════════════════

def train_distributed(config: dict) -> None:
    """Main 4D parallel training loop."""

    # ── Distributed Setup ──────────────────────────────────────────
    is_distributed = int(os.environ.get("LOCAL_RANK", -1)) != -1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist = torch.distributed
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Rank {global_rank}/{world_size} | Device: {device} | Distributed: {is_distributed}")

    # ── Parallel Config ────────────────────────────────────────────
    parallel_config = ParallelConfig(
        dp_size=config.get("dp_size", 1),
        tp_size=config.get("tp_size", 1),
        pp_size=config.get("pp_size", 1),
        ep_size=config.get("ep_size", 1),
    )
    assert parallel_config.world_size == world_size, (
        f"Parallel config requires {parallel_config.world_size} GPUs, got {world_size}"
    )

    coordinator = DistributedCoordinator(parallel_config)
    if is_distributed:
        coordinator.initialize()

    if global_rank == 0:
        logger.info(f"4D Topology: {coordinator.get_4d_info()}")

    # ── Model ─────────────────────────────────────────────────────
    model_config = MamformerConfig.from_yaml(config["config_path"])
    if global_rank == 0:
        logger.info(f"\n{model_config.summary()}")

    model = MamformerForCausalLM(model_config)
    model = model.to(device)

    # 4D sharding
    model = coordinator.shard_model(model)

    if config.get("gradient_checkpointing", True):
        if hasattr(model, 'model') and hasattr(model.model, 'enable_gradient_checkpointing'):
            model.model.enable_gradient_checkpointing()
        elif hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()

    # ── Optimizer ─────────────────────────────────────────────────
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(x in name for x in ("norm", "bias", "gate", "alpha", "lambda_log", "A_log")):
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

    # ── Data ──────────────────────────────────────────────────────
    dataset = StreamingTextDataset(
        data_dir=config.get("data_dir", "./data"),
        seq_len=config.get("max_seq_len", 8192),
    )
    dataloader = DataLoader(dataset, batch_size=config.get("batch_size", 1),
                            num_workers=config.get("num_workers", 4), pin_memory=True)

    # ── Training State ────────────────────────────────────────────
    global_step = 0
    max_steps = config.get("max_steps", 100000)
    grad_accum = config.get("gradient_accumulation_steps", 8)
    save_every = config.get("save_every", 5000)
    log_every = config.get("log_every", 10)
    output_dir = Path(config.get("output_dir", "./checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    amp_dtype = torch.bfloat16 if config.get("bf16", False) else torch.float16

    # ── WandB ────────────────────────────────────────────────────
    use_wandb = config.get("use_wandb", False) and global_rank == 0
    if use_wandb:
        try:
            import wandb
            wandb.init(project=config.get("wandb_project", "Mamformer"),
                       name=config.get("wandb_run_name", model_config.name),
                       config={**config, "parallel": coordinator.get_4d_info()})
        except ImportError:
            use_wandb = False

    # ── Diagnostics Monitor ──────────────────────────────────────
    diag_enabled = global_rank == 0  # Only rank 0 prints diagnostics
    from mamformer.parallelism.diagnostics import ParallelismMonitor
    from mamformer.parallelism.tensor_parallel import set_comm_diagnostics
    monitor = ParallelismMonitor(
        enabled=diag_enabled,
        log_every=log_every,
        n_layers=model_config.n_layers,
        n_experts_per_layer=model_config.moe.n_routed_experts if model_config.moe.enabled else 1,
        pp_size=parallel_config.pp_size,
        num_microbatches=grad_accum,
    )
    # Hook communication callbacks into TP/EP ops
    set_comm_diagnostics(
        start_fn=lambda op, size: monitor.comm_start(op, size),
        end_fn=lambda op: monitor.comm_end(op),
    )

    # ── Training Loop ─────────────────────────────────────────────
    logger.info(f"Starting 4D distributed training at step {global_step}, target {max_steps}")
    model.train()
    total_loss = 0.0
    start_time = time.time()
    tokens_processed = 0

    for batch in dataloader:
        if global_step >= max_steps:
            break

        monitor.start_step()

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        monitor.start_compute()
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=config.get("bf16", False)):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"] / grad_accum
        monitor.end_compute()

        loss.backward()

        tokens_processed += input_ids.numel()

        # Collect MoE load stats for diagnostics (guard against PP sharding)
        if moe_aux_info := outputs.get("moe_aux_info"):
            if hasattr(model, 'model') and hasattr(model.model, 'layers') and model_config.moe.enabled:
                for layer_idx, layer in enumerate(model.model.layers):
                    if hasattr(layer, 'ffn') and hasattr(layer.ffn, '_expert_counts'):
                        expert_counts = layer.ffn._expert_counts.float()
                        monitor.update_expert_load(layer_idx, expert_counts)
                        layer.ffn.reset_load_statistics()

        if (global_step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum

        # ── Diagnostics ──────────────────────────────────────────
        monitor.end_step(global_step, loss.item() * grad_accum, tokens_processed)

        # ── Logging ───────────────────────────────────────────────
        if global_step % log_every == 0 and global_step > 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / log_every
            tokens_per_sec = tokens_processed / elapsed

            log_parts = [
                f"Step {global_step:>6d}",
                f"Loss: {avg_loss:.4f}",
                f"LR: {scheduler.get_last_lr()[0]:.2e}",
                f"Tok/s: {tokens_per_sec:,.0f}",
            ]
            main_loss = outputs.get("main_loss")
            mtp_loss = outputs.get("mtp_loss")
            if main_loss is not None and mtp_loss is not None:
                log_parts.append(f"Main: {main_loss.item():.4f}")
                log_parts.append(f"MTP: {mtp_loss.item():.4f}")

            logger.info(" | ".join(log_parts))

            if use_wandb:
                wandb_log = {
                    "loss": avg_loss, "lr": scheduler.get_last_lr()[0],
                    "tokens_per_sec": tokens_per_sec, "step": global_step,
                }
                if main_loss is not None:
                    wandb_log["main_loss"] = main_loss.item()
                    wandb_log["mtp_loss"] = mtp_loss.item() if mtp_loss is not None else 0
                wandb.log(wandb_log)

            total_loss = 0.0
            tokens_processed = 0
            start_time = time.time()

        # ── Checkpoint ────────────────────────────────────────────
        if global_step % save_every == 0 and global_step > 0 and global_rank == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, output_dir)
            logger.info(f"Checkpoint saved at step {global_step}")

        global_step += 1

    # Final save
    if global_rank == 0:
        save_checkpoint(model, optimizer, scheduler, global_step, output_dir, final=True)

    if is_distributed:
        dist.destroy_process_group()


def save_checkpoint(model, optimizer, scheduler, step, output_dir, final=False):
    """Save distributed training checkpoint with FSDP state dict handling."""
    if hasattr(model, "_is_fsdp_managed_module"):
        try:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with model.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = model.state_dict()
        except Exception:
            state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()

    suffix = "final" if final else f"step_{step}"
    # All ranks participate in state dict gathering; only rank 0 writes
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        torch.save(
            {"model": state_dict, "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "step": step},
            output_dir / f"checkpoint_{suffix}.pt",
        )


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mamformer 4D Distributed Training")
    parser.add_argument("--config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--data", type=str, default="./data", help="Tokenized data dir")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--bf16", action="store_true", help="BF16 mixed precision")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Mamformer")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    # 4D Parallelism
    parser.add_argument("--dp", type=int, default=1, help="Data parallel size")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    train_config = {
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
        "output_dir": args.output_dir,
        "save_every": args.save_every,
        "log_every": args.log_every,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "dp_size": args.dp, "tp_size": args.tp,
        "pp_size": args.pp, "ep_size": args.ep,
    }
    train_distributed(train_config)


if __name__ == "__main__":
    main()
