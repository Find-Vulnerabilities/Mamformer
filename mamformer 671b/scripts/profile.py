"""
Mamformer Model Profiler
=====================
Memory and parameter profiling utility for Mamformer models.

Usage:
    python scripts/profile.py --config configs/7b.yaml
    python scripts/profile.py --config configs/7b.yaml --seq_len 4096 --batch_size 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mamformer.config import MamformerConfig
from mamformer.model import MamformerForCausalLM


def format_bytes(num_bytes: float) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def profile_model(config: MamformerConfig, seq_len: int, batch_size: int) -> None:
    """
    Profile model parameters and memory usage.
    """
    print("\n" + "=" * 70)
    print("  Mamformer Model Profiler")
    print("=" * 70)

    # -- Configuration --
    print(f"\n  [Configuration]")
    print(f"  Model:           {config.name}")
    print(f"  d_model:         {config.d_model}")
    print(f"  n_layers:        {config.n_layers}")
    print(f"  n_heads (Q):     {config.n_heads}")
    print(f"  n_kv_heads:      {config.n_kv_heads}")
    print(f"  head_dim:        {config.head_dim}")
    print(f"  d_ff:            {config.d_ff}")
    print(f"  vocab_size:      {config.vocab_size}")
    print(f"  max_seq_len:     {config.max_seq_len}")
    print(f"  Mamba d_state:   {config.mamba.d_state}")
    print(f"  Mamba d_conv:    {config.mamba.d_conv}")
    print(f"  Mamba expand:    {config.mamba.expand}")
    print(f"  Tie embeddings:  {config.tie_word_embeddings}")

    # -- Parameter Count (theoretical) --
    print(f"\n  [Parameter Count (Theoretical)]")
    total = config.num_parameters
    print(f"  Total:           {total:>15,}  ({total/1e9:.2f}B)")

    # -- Build model and count actual params --
    print(f"\n  [Parameter Count (Actual)]")
    print(f"  Building model...")

    model = MamformerForCausalLM(config)
    actual_total = model.num_parameters()
    print(f"  Actual total:    {actual_total:>15,}  ({actual_total/1e9:.2f}B)")

    # -- Memory Analysis --
    print(f"\n  [Memory Analysis]")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size:      {batch_size}")

    # Parameter memory
    param_mem_fp32 = actual_total * 4
    param_mem_bf16 = actual_total * 2
    print(f"  Parameters (FP32):  {format_bytes(param_mem_fp32)}")
    print(f"  Parameters (BF16):  {format_bytes(param_mem_bf16)}")

    # Optimizer memory (Adam: 2x params for momentum/variance)
    opt_mem_fp32 = param_mem_fp32 * 2
    print(f"  Optimizer (Adam):   {format_bytes(opt_mem_fp32)}")

    # KV cache memory (for inference)
    kv_cache_bytes = (
        2  # K + V
        * config.n_layers
        * config.n_kv_heads
        * config.head_dim
        * seq_len
        * 2  # BF16
    )
    print(f"  KV Cache (BF16):    {format_bytes(kv_cache_bytes)}")

    # SSM state memory (for inference)
    ssm_state_bytes = (
        config.n_layers
        * config.d_inner
        * config.mamba.d_state
        * 2  # BF16
    )
    print(f"  SSM State (BF16):   {format_bytes(ssm_state_bytes)}")

    # Activation memory estimate
    act_bytes = (
        config.n_layers
        * batch_size
        * seq_len
        * config.d_model
        * 2  # BF16
        * 4  # rough multiplier
    )
    print(f"  Activations est:    {format_bytes(act_bytes)}")

    # Per-GPU with FSDP (8 GPUs)
    if config.n_layers >= 32:
        print(f"\n  [FSDP Estimate (8x A100-80GB)]")
        per_gpu_params = param_mem_bf16 / 8
        per_gpu_opt = opt_mem_fp32 / 8
        per_gpu_act = act_bytes / 8
        per_gpu_total = per_gpu_params + per_gpu_opt + per_gpu_act
        print(f"  Per-GPU params:    {format_bytes(per_gpu_params)}")
        print(f"  Per-GPU optimizer: {format_bytes(per_gpu_opt)}")
        print(f"  Per-GPU activ:     {format_bytes(per_gpu_act)}")
        print(f"  Per-GPU total:     {format_bytes(per_gpu_total)}")
        if per_gpu_total > 0:
            print(f"  GPU mem (80GB):    {per_gpu_total / 80e9 * 100:.1f}% utilized")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Profile Mamformer model")
    parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    config = MamformerConfig.from_yaml(args.config)
    profile_model(config, args.seq_len, args.batch_size)


if __name__ == "__main__":
    main()
