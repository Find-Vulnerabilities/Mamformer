"""
Mamformer GRPO Training (Group Relative Policy Optimization)
==============================================================
Post-training with reinforcement learning to enhance reasoning
capabilities, inspired by DeepSeek-R1's GRPO method.

GRPO eliminates the need for a separate critic/reward model by
using group-relative rewards: for each prompt, G responses are
sampled from the current policy, and their rewards are normalized
within the group to compute advantages.

Loss: L = -E[A * log π(a|s)] + β * KL(π_ref || π)

Where:
  - A = (r - mean(r_group)) / std(r_group)  (group-relative advantage)
  - KL penalty keeps the policy from diverging too far from reference

Usage:
    # Basic GRPO training
    python scripts/train_grpo.py \\
        --config configs/ultra-7b.yaml \\
        --checkpoint ./checkpoints/sft_model.pt \\
        --data ./data/grpo_prompts.jsonl \\
        --reward_type math \\
        --group_size 8 \\
        --kl_beta 0.04 \\
        --bf16

    # Multi-GPU with FSDP
    torchrun --nproc_per_node=8 scripts/train_grpo.py \\
        --config configs/ultra-37b.yaml \\
        --checkpoint ./checkpoints/sft_model.pt \\
        --data ./data/grpo_prompts.jsonl \\
        --reward_type math \\
        --group_size 8 \\
        --kl_beta 0.04 \\
        --bf16

Reference:
  "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
   Reinforcement Learning" (DeepSeek-AI, 2025)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mamformer.config import MamformerConfig
from mamformer.model import MamformerForCausalLM
from mamformer.tokenizer import MamformerTokenizer
from mamformer.rewards import RewardCalculator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Prompt Dataset
# ═══════════════════════════════════════════════════════════════════════

class GRPOPromptDataset(Dataset):
    """
    Dataset of prompts for GRPO training.

    Each line is a JSON object:
    {
        "prompt": "Solve: ...",
        "answer": "42",                       # ground truth (for math reward)
        "reward_type": "math",                 # "math" | "format" | "code" | "combined"
        "test_cases": [                        # for code reward
            {"input": "2,3", "expected_output": "5"}
        ],
        "reward_weights": {"math": 0.7, "format": 0.3}  # for combined reward
    }

    Args:
        data_path: Path to JSONL prompts file
        tokenizer: MamformerTokenizer instance
        max_prompt_len: Maximum prompt length in tokens
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: MamformerTokenizer,
        max_prompt_len: int = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len

        self.items: list[dict] = []
        data_path = Path(data_path)
        if not data_path.exists():
            logger.warning(f"Data file {data_path} not found. Using dummy data.")
            self._use_dummy = True
            # Generate dummy prompts for testing
            for i in range(100):
                self.items.append({
                    "prompt": f"Question {i}: What is 2+2?",
                    "answer": "4",
                    "reward_type": "math",
                })
        else:
            self._use_dummy = False
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        self.items.append(item)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSON line: {line[:80]}...")
            logger.info(f"Loaded {len(self.items)} GRPO prompts from {data_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]

        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(item["prompt"], add_bos=True)
        if len(prompt_ids) > self.max_prompt_len:
            prompt_ids = prompt_ids[:self.max_prompt_len]

        return {
            "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "ground_truth": item.get("answer", ""),
            "reward_type": item.get("reward_type", "format"),
            "test_cases": item.get("test_cases", []),
            "reward_weights": item.get("reward_weights", None),
        }


def collate_prompts(batch: list[dict]) -> dict:
    """
    Custom collate for variable-length prompts.

    Pads prompt_ids to the max length in the batch.
    """
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        L = item["input_ids"].shape[0]
        input_ids[i, :L] = item["input_ids"]

    return {
        "input_ids": input_ids,
        "ground_truth": [item["ground_truth"] for item in batch],
        "reward_type": [item["reward_type"] for item in batch],
        "test_cases": [item["test_cases"] for item in batch],
        "reward_weights": [item["reward_weights"] for item in batch],
    }


# ═══════════════════════════════════════════════════════════════════════
# GRPO Loss
# ═══════════════════════════════════════════════════════════════════════

def compute_grpo_loss(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    kl_beta: float = 0.04,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute GRPO loss with KL penalty.

    GRPO loss = policy_gradient_loss + β * KL_penalty

    Where:
      - policy_gradient_loss = -mean(advantages * policy_log_probs)
      - KL is estimated as: log(π_ref) - log(π)

    The KL estimator is unbiased: E[log π_ref - log π] = KL(π || π_ref)
    when sampling from π (policy). Note that we use the forward KL:
    KL(π_ref || π) in the loss, estimated as (log π_ref - log π) under π.

    Args:
        policy_log_probs: Average log-prob under policy (batch, group_size)
        reference_log_probs: Average log-prob under reference (batch, group_size)
        advantages: Group-relative advantages (batch, group_size)
        kl_beta: KL penalty coefficient (default 0.04 from DeepSeek-R1)

    Returns:
        (total_loss, metrics_dict)
    """
    # Policy gradient: maximize advantage-weighted log-prob
    policy_loss = -(advantages * policy_log_probs).mean()

    # KL divergence: positive when policy diverges from reference
    # kl ≈ log π_ref - log π  (sampled under π)
    kl_div = (reference_log_probs - policy_log_probs).mean()

    # Total loss
    total_loss = policy_loss + kl_beta * kl_div

    metrics = {
        "policy_loss": policy_loss.detach(),
        "kl_div": kl_div.detach(),
        "mean_advantage": advantages.mean().detach(),
        "std_advantage": advantages.std().detach(),
    }

    return total_loss, metrics


# ═══════════════════════════════════════════════════════════════════════
# LR Schedule
# ═══════════════════════════════════════════════════════════════════════

def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create cosine LR schedule with linear warmup."""
    warmup_steps = config.get("warmup_steps", 100)
    max_steps = config.get("max_steps", 10000)
    lr = config.get("learning_rate", 1e-6)

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=lr * 0.1)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


# ═══════════════════════════════════════════════════════════════════════
# Main GRPO Training
# ═══════════════════════════════════════════════════════════════════════

def train_grpo(config: dict) -> None:
    """Main GRPO training loop."""

    # ── Setup ──────────────────────────────────────────────────
    is_distributed = int(os.environ.get("LOCAL_RANK", -1)) != -1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if is_distributed:
        torch.cuda.set_device(local_rank)
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if global_rank == 0:
        logger.info(f"Device: {device}, Distributed: {is_distributed}, World: {world_size}")

    # ── Model Config ───────────────────────────────────────────
    model_config = MamformerConfig.from_yaml(config["config_path"])
    if global_rank == 0:
        logger.info(f"Model: {model_config.name} ({model_config.num_parameters_billions:.1f}B)")

    # ── Policy Model (trainable) ──────────────────────────────
    policy_model = MamformerForCausalLM(model_config)
    policy_model = policy_model.to(device)

    # Load checkpoint
    ckpt_path = config.get("checkpoint")
    if ckpt_path and Path(ckpt_path).exists():
        if global_rank == 0:
            logger.info(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy_model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    elif ckpt_path:
        if global_rank == 0:
            logger.warning(f"Checkpoint not found: {ckpt_path}, using random init")

    # ── Reference Model (frozen) ──────────────────────────────
    reference_model = MamformerForCausalLM(model_config)
    reference_model = reference_model.to(device)
    reference_model.load_state_dict(policy_model.state_dict())
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # Enable gradient checkpointing
    if config.get("gradient_checkpointing", True):
        policy_model.model.enable_gradient_checkpointing()

    # ── FSDP ──────────────────────────────────────────────────
    if is_distributed and config.get("use_fsdp", True):
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                ShardingStrategy,
                MixedPrecision,
                BackwardPrefetch,
            )
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from functools import partial
            from mamformer.layers.hybrid import MamformerBlock

            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={MamformerBlock},
            )
            mp_dtype = torch.bfloat16 if config.get("bf16", False) else torch.float16

            policy_model = FSDP(
                policy_model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=MixedPrecision(
                    param_dtype=mp_dtype,
                    reduce_dtype=mp_dtype,
                    buffer_dtype=mp_dtype,
                    cast_forward_inputs=True,
                ),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                device_id=torch.cuda.current_device(),
                use_orig_params=True,
            )
            if global_rank == 0:
                logger.info("FSDP wrapping applied to policy model")
        except ImportError:
            if global_rank == 0:
                logger.warning("FSDP import failed, running without FSDP")

    # ── Optimizer ─────────────────────────────────────────────
    decay_params, no_decay_params = [], []
    for name, param in policy_model.named_parameters():
        if not param.requires_grad:
            continue
        if any(x in name for x in ("norm", "bias", "gate", "alpha", "lambda_log", "A_log", "comm_strength")):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": config.get("weight_decay", 0.1)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.get("learning_rate", 1e-6),
        betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.95)),
        eps=config.get("adam_epsilon", 1e-8),
    )

    scheduler = create_lr_scheduler(optimizer, config)

    # ── Tokenizer ─────────────────────────────────────────────
    tokenizer = MamformerTokenizer()

    # ── Data ──────────────────────────────────────────────────
    dataset = GRPOPromptDataset(
        data_path=config.get("data", "./data/grpo_prompts.jsonl"),
        tokenizer=tokenizer,
        max_prompt_len=config.get("max_prompt_len", 2048),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        collate_fn=collate_prompts,
        num_workers=config.get("num_workers", 2),
        pin_memory=True,
    )

    # ── Reward Calculator ────────────────────────────────────
    reward_calc = RewardCalculator()

    # ── GRPO Hyperparameters ─────────────────────────────────
    G = config.get("group_size", 8)
    if G < 2 and global_rank == 0:
        logger.warning(
            "GRPO group_size (G=%d) < 2: group-relative advantages will always be 0 "
            "(single sample per group yields zero mean and zero std). "
            "Set --group_size >= 2 for meaningful training.",
            G,
        )
    kl_beta = config.get("kl_beta", 0.04)
    gen_max_tokens = config.get("gen_max_tokens", 1024)
    gen_temperature = config.get("gen_temperature", 1.0)
    gen_top_p = config.get("gen_top_p", 0.95)
    gen_top_k = config.get("gen_top_k", 50)

    # ── Training State ───────────────────────────────────────
    global_step = 0
    max_steps = config.get("max_steps", 10000)
    grad_accum = config.get("gradient_accumulation_steps", 1)
    log_every = config.get("log_every", 10)
    save_every = config.get("save_every", 500)
    eval_every = config.get("eval_every", 100)
    output_dir = Path(config.get("output_dir", "./grpo_checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume ────────────────────────────────────────────────
    resume_path = config.get("resume")
    if resume_path and Path(resume_path).exists():
        if global_rank == 0:
            logger.info(f"Resuming from: {resume_path}")
        r_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        policy_model.load_state_dict(r_ckpt.get("model", r_ckpt), strict=False)
        optimizer.load_state_dict(r_ckpt.get("optimizer", {}))
        scheduler.load_state_dict(r_ckpt.get("scheduler", {}))
        global_step = r_ckpt.get("step", 0)

    # ── WandB ────────────────────────────────────────────────
    use_wandb = config.get("use_wandb", False) and global_rank == 0
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.get("wandb_project", "Mamformer-GRPO"),
                name=config.get("wandb_run_name", f"{model_config.name}-GRPO"),
                config=config,
            )
        except ImportError:
            use_wandb = False

    # Mixed precision
    amp_dtype = torch.bfloat16 if config.get("bf16", False) else torch.float16
    amp_enabled = config.get("bf16", False) or config.get("fp16", False)

    # ── Training Loop ─────────────────────────────────────────
    if global_rank == 0:
        logger.info(f"Starting GRPO training at step {global_step}, target {max_steps}")
        logger.info(f"  Group size G={G}, KL β={kl_beta}, LR={config.get('learning_rate', 1e-6)}")
        logger.info(f"  Gen: max_tokens={gen_max_tokens}, temp={gen_temperature}")

    policy_model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_kl = 0.0
    total_reward = 0.0
    start_time = time.time()

    for batch in dataloader:
        if global_step >= max_steps:
            break

        prompt_ids = batch["input_ids"].to(device)  # (B, prompt_len)
        ground_truths = batch["ground_truth"]
        reward_types = batch["reward_type"]
        test_cases_list = batch["test_cases"]
        reward_weights_list = batch["reward_weights"]

        B = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]

        # ── Step 1: Generate G responses per prompt ──────────
        all_response_ids: List[torch.Tensor] = []
        all_response_texts: List[List[str]] = [[] for _ in range(B)]

        with torch.no_grad():
            for g in range(G):
                # Slightly vary temperature for diversity within group
                temp = gen_temperature * (0.9 + 0.2 * g / max(G - 1, 1))

                generated = policy_model.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=gen_max_tokens,
                    temperature=temp,
                    top_k=gen_top_k,
                    top_p=gen_top_p,
                )  # (B, prompt_len + gen_len)

                # Extract response portion
                response_ids = generated[:, prompt_len:]  # (B, gen_len_actual)
                all_response_ids.append(response_ids)

                # Decode responses for reward computation
                for b_idx in range(B):
                    resp_text = tokenizer.decode(response_ids[b_idx].tolist())
                    all_response_texts[b_idx].append(resp_text)

        # ── Step 2: Compute rewards ───────────────────────────
        rewards = torch.zeros(B, G, device=device)
        for b_idx in range(B):
            rtype = reward_types[b_idx] if isinstance(reward_types, list) else reward_types
            gt = ground_truths[b_idx] if isinstance(ground_truths, list) else ground_truths
            tc = test_cases_list[b_idx] if isinstance(test_cases_list, list) else test_cases_list
            rw = reward_weights_list[b_idx] if isinstance(reward_weights_list, list) else reward_weights_list

            for g_idx in range(G):
                resp = all_response_texts[b_idx][g_idx]

                try:
                    kwargs = {}
                    if rtype == "math":
                        kwargs["ground_truth"] = gt
                    elif rtype == "code":
                        kwargs["test_cases"] = tc
                    elif rtype == "combined" and rw is not None:
                        kwargs["reward_weights"] = rw

                    rewards[b_idx, g_idx] = reward_calc.compute(resp, reward_type=rtype, **kwargs)
                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(f"Reward computation failed for prompt {b_idx}, gen {g_idx}: {e}")
                    rewards[b_idx, g_idx] = 0.0

        # ── Step 3: Compute group-relative advantages ─────────
        reward_mean = rewards.mean(dim=1, keepdim=True)  # (B, 1)
        reward_std = rewards.std(dim=1, keepdim=True) + 1e-8  # (B, 1)
        advantages = (rewards - reward_mean) / reward_std  # (B, G)

        # ── Step 4: Compute log-probs ─────────────────────────
        # Prepare full input sequences: prompt + each response
        prompt_expanded = prompt_ids.unsqueeze(1).expand(B, G, -1)  # (B, G, prompt_len)

        # Pad response ids to uniform length
        max_resp_len = max(r.shape[1] for r in all_response_ids)
        resp_padded = torch.zeros(B, G, max_resp_len, dtype=torch.long, device=device)
        for g_idx in range(G):
            cur_len = all_response_ids[g_idx].shape[1]
            resp_padded[:, g_idx, :cur_len] = all_response_ids[g_idx]

        # Concatenate: (B, G, total_len)
        full_input_ids = torch.cat([prompt_expanded, resp_padded], dim=2)
        full_input_ids = full_input_ids.view(B * G, -1)  # (B*G, total_len)

        # Labels: mask prompt tokens, keep response tokens
        labels = full_input_ids.clone()
        prompt_mask = torch.zeros(B * G, full_input_ids.shape[1], device=device)
        prompt_mask[:, :prompt_len] = 1  # Mask prompt positions
        labels = labels.masked_fill(prompt_mask.bool(), -100)
        # Also mask padding in responses
        for g_idx in range(G):
            cur_len = all_response_ids[g_idx].shape[1]
            for b_idx in range(B):
                if cur_len < max_resp_len:
                    flat_idx = b_idx * G + g_idx
                    labels[flat_idx, prompt_len + cur_len:] = -100

        # ── Forward through policy model ─────────────────────
        policy_log_probs_flat = torch.zeros(B * G, device=device)

        # Process in sub-batches to manage memory
        sub_batch_size = config.get("sub_batch_size") or B * G
        for start in range(0, B * G, sub_batch_size):
            end = min(start + sub_batch_size, B * G)
            sub_input = full_input_ids[start:end]
            sub_labels = labels[start:end]

            with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu",
                                    dtype=amp_dtype, enabled=amp_enabled):
                policy_log_probs_flat[start:end] = policy_model.get_log_probs(
                    input_ids=sub_input,
                    labels=sub_labels,
                )

        # ── Forward through reference model (no grad) ────────
        reference_log_probs_flat = torch.zeros(B * G, device=device)
        with torch.no_grad():
            for start in range(0, B * G, sub_batch_size):
                end = min(start + sub_batch_size, B * G)
                sub_input = full_input_ids[start:end]
                sub_labels = labels[start:end]

                with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu",
                                        dtype=amp_dtype, enabled=amp_enabled):
                    reference_log_probs_flat[start:end] = reference_model.get_log_probs(
                        input_ids=sub_input,
                        labels=sub_labels,
                    )

        # Reshape to (B, G)
        policy_log_probs = policy_log_probs_flat.view(B, G)
        reference_log_probs = reference_log_probs_flat.view(B, G)

        # ── Step 5: Compute GRPO loss ─────────────────────────
        loss, metrics = compute_grpo_loss(
            policy_log_probs=policy_log_probs,
            reference_log_probs=reference_log_probs,
            advantages=advantages,
            kl_beta=kl_beta,
        )

        loss = loss / grad_accum
        loss.backward()

        # Gradient accumulation
        if (global_step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(),
                config.get("max_grad_norm", 1.0),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Track statistics
        step_loss = loss.item() * grad_accum
        total_loss += step_loss
        total_policy_loss += metrics["policy_loss"].item()
        total_kl += metrics["kl_div"].item()
        total_reward += rewards.mean().item()

        # ── Logging ──────────────────────────────────────────
        if global_step % log_every == 0 and global_step > 0:
            elapsed = time.time() - start_time
            n = log_every
            log_msg = (
                f"Step {global_step:>6d} | "
                f"Loss: {total_loss/n:.4f} | "
                f"Policy: {total_policy_loss/n:.4f} | "
                f"KL: {total_kl/n:.4f} | "
                f"Reward: {total_reward/n:.3f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"Time: {elapsed:.0f}s"
            )
            logger.info(log_msg)

            if use_wandb:
                wandb.log({
                    "loss": total_loss / n,
                    "policy_loss": total_policy_loss / n,
                    "kl_div": total_kl / n,
                    "mean_reward": total_reward / n,
                    "lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                    "reward_std": rewards.std().item(),
                })

            total_loss = 0.0
            total_policy_loss = 0.0
            total_kl = 0.0
            total_reward = 0.0
            start_time = time.time()

        # ── Checkpointing ────────────────────────────────────
        if global_step % save_every == 0 and global_step > 0:
            if not is_distributed or global_rank == 0:
                save_checkpoint(
                    policy_model, optimizer, scheduler, global_step, output_dir
                )
                logger.info(f"Checkpoint saved at step {global_step}")

        # ── Evaluation ────────────────────────────────────────
        if global_step % eval_every == 0 and global_step > 0:
            eval_reward = run_evaluation(
                policy_model, dataloader, reward_calc, tokenizer,
                G=G, gen_max_tokens=gen_max_tokens, device=device,
            )
            if global_rank == 0:
                logger.info(f"Eval @ step {global_step}: mean_reward={eval_reward:.3f}")
            if use_wandb:
                wandb.log({"eval_reward": eval_reward, "step": global_step})

        global_step += 1

    # ── Final save ────────────────────────────────────────────
    if not is_distributed or global_rank == 0:
        save_checkpoint(policy_model, optimizer, scheduler, global_step, output_dir, final=True)
        logger.info(f"Final checkpoint saved at step {global_step}")

    if is_distributed:
        import torch.distributed as dist
        dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    reward_calc: RewardCalculator,
    tokenizer: MamformerTokenizer,
    G: int = 4,
    gen_max_tokens: int = 512,
    device: torch.device = None,
    num_batches: int = 5,
) -> float:
    """
    Run evaluation: generate and score on a subset of prompts.

    Returns mean reward across eval prompts.
    """
    model.eval()
    all_rewards = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        prompt_ids = batch["input_ids"].to(device)
        ground_truths = batch["ground_truth"]
        reward_types = batch["reward_type"]
        prompt_len = prompt_ids.shape[1]
        B = prompt_ids.shape[0]

        for b_idx in range(B):
            rtype = reward_types[b_idx]
            gt = ground_truths[b_idx]

            for _ in range(G):
                generated = model.generate(
                    input_ids=prompt_ids[b_idx:b_idx+1],
                    max_new_tokens=gen_max_tokens,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                )
                resp_ids = generated[0, prompt_len:]
                resp_text = tokenizer.decode(resp_ids.tolist())

                kwargs = {}
                if rtype == "math":
                    kwargs["ground_truth"] = gt
                score = reward_calc.compute(resp_text, reward_type=rtype, **kwargs)
                all_rewards.append(score)

    model.train()
    return sum(all_rewards) / max(len(all_rewards), 1)


# ═══════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    output_dir: Path,
    final: bool = False,
) -> str:
    """Save GRPO training checkpoint."""
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
    ckpt_path = output_dir / f"grpo_checkpoint_{suffix}.pt"
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


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mamformer GRPO Training")

    # Model & Data
    parser.add_argument("--config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="SFT checkpoint to load")
    parser.add_argument("--data", type=str, default="./data/grpo_prompts.jsonl", help="JSONL prompts file")

    # GRPO hyperparameters
    parser.add_argument("--reward_type", type=str, default="math",
                       help="Default reward type (math, format, code, combined)")
    parser.add_argument("--group_size", type=int, default=8,
                       help="G = number of samples per prompt group")
    parser.add_argument("--kl_beta", type=float, default=0.04,
                       help="KL penalty coefficient")
    parser.add_argument("--gen_max_tokens", type=int, default=1024,
                       help="Max tokens to generate per response")
    parser.add_argument("--gen_temperature", type=float, default=1.0,
                       help="Sampling temperature for generation")
    parser.add_argument("--gen_top_p", type=float, default=0.95,
                       help="Top-p for generation")
    parser.add_argument("--gen_top_k", type=int, default=50,
                       help="Top-k for generation")

    # Training
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Number of prompts per batch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--max_prompt_len", type=int, default=2048)
    parser.add_argument("--sub_batch_size", type=int, default=0,
                       help="Sub-batch size for log-prob computation (0 = auto = B * G)")

    # Precision
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing (default: enabled)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # I/O
    parser.add_argument("--output_dir", type=str, default="./grpo_checkpoints")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None, help="Resume from GRPO checkpoint")

    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Mamformer-GRPO")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config = {
        "config_path": args.config,
        "checkpoint": args.checkpoint,
        "data": args.data,
        "reward_type": args.reward_type,
        "group_size": args.group_size,
        "kl_beta": args.kl_beta,
        "gen_max_tokens": args.gen_max_tokens,
        "gen_temperature": args.gen_temperature,
        "gen_top_p": args.gen_top_p,
        "gen_top_k": args.gen_top_k,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "max_prompt_len": args.max_prompt_len,
        "sub_batch_size": args.sub_batch_size,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_grad_norm": args.max_grad_norm,
        "output_dir": args.output_dir,
        "save_every": args.save_every,
        "log_every": args.log_every,
        "eval_every": args.eval_every,
        "num_workers": args.num_workers,
        "resume": args.resume,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
    }

    train_grpo(config)


if __name__ == "__main__":
    main()
