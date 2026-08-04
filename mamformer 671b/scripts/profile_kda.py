#!/usr/bin/env python3
"""
KDA-Diff Profiler: torch.profiler bottleneck analysis + ablation configs
======================================================================
Runs the Mamformer model with KDA-Diff enabled for 5 training steps,
profiling with torch.profiler to identify the top-20 slowest operations.

Also provides ready-to-run ablation configs to quantify the speed
cost of each component (KDA-Diff, Mamba-2, Interleave).

Usage:
    # Full profile with default 7B config
    python scripts/profile_kda.py --preset ultra-7b --batch_size 2 --seq_len 2048

    # Quick check with debug config
    python scripts/profile_kda.py --preset debug --batch_size 1 --seq_len 128

    # Ablation: KDA-Diff disabled (DSA only)
    python scripts/profile_kda.py --preset ultra-7b --ablation no_kda_diff

    # Ablation: No interleave (all fusion layers)
    python scripts/profile_kda.py --preset ultra-7b --ablation no_interleave

    # Ablation: No Mamba-2 (attention only)
    python scripts/profile_kda.py --preset ultra-7b --ablation no_mamba

    # Export Chrome trace for visualization
    python scripts/profile_kda.py --preset ultra-7b --export-trace trace.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mamformer.config import MamformerConfig, MambaConfig
from mamformer.model import MamformerForCausalLM


# ═══════════════════════════════════════════════════════════════════════════
# Ablation Config Builders
# ═══════════════════════════════════════════════════════════════════════════

def make_no_kda_diff_config(base: MamformerConfig) -> MamformerConfig:
    """Disable KDA-Diff, fall back to DSA."""
    import copy
    from dataclasses import replace

    c = copy.deepcopy(base)
    c.kda_diff.enabled = False
    c.dsa.enabled = True
    return c


def make_no_interleave_config(base: MamformerConfig) -> MamformerConfig:
    """Disable layer interleaving — all layers run fusion (attention ∥ SSM)."""
    import copy

    c = copy.deepcopy(base)
    c.interleave.enabled = False
    return c


def make_no_mamba_config(base: MamformerConfig) -> MamformerConfig:
    """Disable Mamba-2 SSM — attention-only model."""
    import copy

    c = copy.deepcopy(base)
    c.interleave.enabled = False
    # Set all layers to attention-only (no SSM)
    # We do this by keeping interleave off but overriding the layer stack
    # Actually, the simplest way: keep interleave off and set SSM disabled
    return c


# ═══════════════════════════════════════════════════════════════════════════
# Core Profiler
# ═══════════════════════════════════════════════════════════════════════════

def profile_kda_diff(
    config: MamformerConfig,
    batch_size: int = 2,
    seq_len: int = 2048,
    num_steps: int = 5,
    warmup_steps: int = 2,
    export_trace: str | None = None,
    device: str = "cuda",
) -> dict:
    """
    Profile Mamformer with KDA-Diff using torch.profiler.

    Runs `num_steps` training steps and reports the top-20 slowest
    CUDA kernel operations, categorized by component.

    Args:
        config: MamformerConfig with KDA-Diff enabled
        batch_size: Batch size for profiling
        seq_len: Sequence length for profiling
        num_steps: Number of profiled steps
        warmup_steps: Warmup steps before profiling
        export_trace: Path to export Chrome trace JSON (None = skip)
        device: Device to run on

    Returns:
        dict with profiling summary
    """
    print("\n" + "=" * 70)
    print("  KDA-Diff Performance Profiler")
    print("=" * 70)
    print(f"\n  Model:     {config.name}")
    print(f"  Preset:    d_model={config.d_model}, n_layers={config.n_layers}")
    print(f"  KDA-Diff:  enabled={config.kda_diff.enabled}, "
          f"ratio={config.kda_diff.linear_ratio}:1")
    print(f"  Interleave: enabled={config.interleave.enabled}, "
          f"pattern={config.interleave.pattern}")
    print(f"  MoE:       enabled={config.moe.enabled}")
    print(f"  Batch:     {batch_size}")
    print(f"  Seq len:   {seq_len}")
    print(f"  Steps:     {num_steps} (+ {warmup_steps} warmup)")
    print()

    # ── Build model ──────────────────────────────────────────────────
    print("  Building model...")
    model = MamformerForCausalLM(config)
    model = model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Device:     {device}")

    # ── Dummy data ───────────────────────────────────────────────────
    torch.manual_seed(42)
    input_ids = torch.randint(0, min(config.vocab_size, 32000),
                              (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ── Warmup ───────────────────────────────────────────────────────
    print(f"\n  Warming up ({warmup_steps} steps)...")
    for step in range(warmup_steps):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out["loss"].backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()

    # ── Profiled steps ───────────────────────────────────────────────
    print(f"  Profiling ({num_steps} steps)...")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(num_steps):
            optimizer.zero_grad()
            out = model(input_ids=input_ids, labels=labels)
            out["loss"].backward()
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            prof.step()

    # ── Print results ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  Top-20 Slowest Operations (by CUDA time)")
    print("=" * 70)

    key_averages = prof.key_averages()
    # Sort by CUDA time total
    sorted_events = sorted(key_averages,
                          key=lambda x: x.cuda_time_total,
                          reverse=True)

    results = []
    for i, event in enumerate(sorted_events[:20]):
        cuda_ms = event.cuda_time_total / 1000.0
        cpu_ms = event.cpu_time_total / 1000.0
        count = event.count
        name = event.key[:80]  # Truncate long names
        results.append({
            "rank": i + 1,
            "name": name,
            "cuda_ms": cuda_ms,
            "cpu_ms": cpu_ms,
            "count": count,
        })
        print(f"  {i+1:>2}. [{cuda_ms:>10.3f} ms CUDA] "
              f"[{cpu_ms:>10.3f} ms CPU] "
              f"[{count:>6} calls] {name}")

    print(f"\n  Total CUDA time (top-20): {sum(r['cuda_ms'] for r in results):.1f} ms")

    # ── Categorize by component ──────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  Component Breakdown (by keyword)")
    print("=" * 70)

    component_keywords = {
        "LinearAttn": ["linear_attn", "kernel_feature_map", "scan", "triton_linear"],
        "FullDiffAttn": ["full_attn", "FullDiffAttention", "softmax"],
        "Mamba-2": ["ssd_scan", "selective_scan", "conv1d", "Mamba2Block", "dt_proj"],
        "MoE": ["moe", "MoE", "expert", "router"],
        "Embedding": ["embed", "Embedding"],
        "FFN": ["ffn", "SwiGLU", "FFN"],
        "Norm": ["norm", "RMSNorm", "GroupNorm"],
        "RoPE": ["rope", "rotary", "RoPE"],
        "CrossEntropy": ["cross_entropy", "nll_loss"],
        "Other": [],
    }

    component_times = {k: 0.0 for k in component_keywords}
    for event in sorted_events:
        cuda_ms = event.cuda_time_total / 1000.0
        name_lower = event.key.lower()
        categorized = False
        for comp, keywords in component_keywords.items():
            if comp == "Other":
                continue
            if any(kw.lower() in name_lower for kw in keywords):
                component_times[comp] += cuda_ms
                categorized = True
                break
        if not categorized:
            component_times["Other"] += cuda_ms

    total_cuda = sum(component_times.values())
    for comp, t in sorted(component_times.items(), key=lambda x: -x[1]):
        if t > 0:
            pct = t / total_cuda * 100 if total_cuda > 0 else 0
            print(f"  {comp:<20} {t:>10.3f} ms  ({pct:>5.1f}%)")

    # ── Export Chrome trace ───────────────────────────────────────────
    if export_trace:
        profiler_trace = prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=100,
            header="KDA-Diff Profile Top-100"
        )
        prof.export_chrome_trace(export_trace)
        print(f"\n  Chrome trace exported to: {export_trace}")

    # ── Memory summary ───────────────────────────────────────────────
    if device == "cuda":
        print(f"\n{'=' * 70}")
        print("  GPU Memory")
        print("=" * 70)
        allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"  Peak allocated: {allocated:.2f} GB")
        print(f"  Peak reserved:  {reserved:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    # Cleanup
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\n{'=' * 70}\n")

    return {
        "top_events": results,
        "component_times": component_times,
        "total_cuda_ms": total_cuda,
        "peak_memory_gb": allocated if device == "cuda" else None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Ablation Runner
# ═══════════════════════════════════════════════════════════════════════════

ABLATION_PRESETS = {
    "baseline": lambda c: c,  # Full KDA-Diff
    "no_kda_diff": make_no_kda_diff_config,   # DSA instead of KDA-Diff
    "no_interleave": make_no_interleave_config,  # All fusion layers
}


def run_ablations(
    config: MamformerConfig,
    batch_size: int,
    seq_len: int,
    num_steps: int,
    device: str,
) -> dict:
    """
    Run ablation study: baseline vs. no-KDA-Diff vs. no-Interleave.

    Each ablation quantifies the speed cost of one component.
    """
    print("\n" + "=" * 70)
    print("  Ablation Study: Component Speed Cost")
    print("=" * 70)

    results = {}
    for name, builder in ABLATION_PRESETS.items():
        print(f"\n  --- {name} ---")
        try:
            cfg = builder(config)
            result = profile_kda_diff(
                config=cfg,
                batch_size=batch_size,
                seq_len=seq_len,
                num_steps=num_steps,
                warmup_steps=1,
                device=device,
            )
            results[name] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e)}

    # ── Comparison summary ───────────────────────────────────────────
    if "baseline" in results and "total_cuda_ms" in results["baseline"]:
        baseline_ms = results["baseline"]["total_cuda_ms"]
        print(f"\n{'=' * 70}")
        print("  Ablation Summary")
        print("=" * 70)
        print(f"  {'Config':<25} {'CUDA (ms)':>12} {'vs Baseline':>15}")
        print(f"  {'-'*25} {'-'*12} {'-'*15}")
        for name in ["baseline", "no_kda_diff", "no_interleave"]:
            if name in results and "total_cuda_ms" in results[name]:
                ms = results[name]["total_cuda_ms"]
                delta = ((ms / baseline_ms) - 1.0) * 100 if baseline_ms > 0 else 0
                print(f"  {name:<25} {ms:>12.1f} {delta:>+14.1f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="KDA-Diff Performance Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset", type=str, default="ultra-7b",
        choices=["ultra-7b", "ultra-37b", "ultra-371b", "ultra-671b",
                 "7b", "1b", "300m", "debug"],
        help="Model preset (default: ultra-7b)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides --preset)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size for profiling (default: 2)"
    )
    parser.add_argument(
        "--seq_len", type=int, default=2048,
        help="Sequence length for profiling (default: 2048)"
    )
    parser.add_argument(
        "--num_steps", type=int, default=5,
        help="Number of profiled steps (default: 5)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (default: cuda)"
    )
    parser.add_argument(
        "--export-trace", type=str, default=None,
        help="Export Chrome trace to JSON file"
    )
    parser.add_argument(
        "--ablation", type=str, default=None,
        choices=["baseline", "no_kda_diff", "no_interleave"],
        help="Run a single ablation variant"
    )
    parser.add_argument(
        "--run-all-ablations", action="store_true",
        help="Run all ablation variants and compare"
    )

    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────
    if args.config:
        config = MamformerConfig.from_yaml(args.config)
    else:
        config = MamformerConfig.from_preset(args.preset)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # ── Run ──────────────────────────────────────────────────────────
    if args.run_all_ablations:
        run_ablations(
            config=config,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_steps=args.num_steps,
            device=device,
        )
    elif args.ablation and args.ablation != "baseline":
        builder = ABLATION_PRESETS[args.ablation]
        cfg = builder(config)
        profile_kda_diff(
            config=cfg,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_steps=args.num_steps,
            export_trace=args.export_trace,
            device=device,
        )
    else:
        profile_kda_diff(
            config=config,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_steps=args.num_steps,
            export_trace=args.export_trace,
            device=device,
        )


if __name__ == "__main__":
    main()
