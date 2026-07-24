"""
Parallelism Diagnostics & Load Balancing
==========================================
Runtime monitoring for communication bottlenecks, load imbalance,
and expert utilization in 4D distributed training.

Key metrics tracked:
  1. Communication time per operation (all-reduce, all-gather, all-to-all)
  2. Compute-to-communication ratio per step
  3. Pipeline bubble ratio (idle time / total time)
  4. Expert load entropy (per-layer expert utilization distribution)
  5. GPU memory utilization per rank
  6. Token throughput per rank (detect stragglers)

Usage:
    from mamformer.parallelism.diagnostics import ParallelismMonitor

    monitor = ParallelismMonitor(enabled=True, log_every=100)
    # ... in training loop ...
    monitor.record_comm_start("tp_all_reduce")
    _tp_all_reduce(tensor)
    monitor.record_comm_end("tp_all_reduce")

    monitor.record_step_end(global_step, loss)
    stats = monitor.get_stats()
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


# ═══════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CommOpStats:
    """Statistics for a single communication operation type."""
    total_time: float = 0.0
    count: int = 0
    total_bytes: int = 0
    current_start: Optional[float] = None

    @property
    def avg_time_ms(self) -> float:
        return (self.total_time / max(self.count, 1)) * 1000

    @property
    def total_bandwidth_gbps(self) -> float:
        """Total effective bandwidth in GB/s."""
        if self.total_time == 0:
            return 0.0
        return (self.total_bytes / self.total_time) / 1e9


@dataclass
class StepStats:
    """Statistics for a single training step."""
    step: int = 0
    compute_time: float = 0.0  # Forward + backward (non-communication)
    comm_time: float = 0.0     # Total communication time
    loss: float = 0.0
    tokens_per_second: float = 0.0
    pipeline_bubble_ratio: float = 0.0

    @property
    def compute_comm_ratio(self) -> float:
        """Compute-to-communication ratio. Higher = better utilization."""
        return self.compute_time / max(self.comm_time, 1e-8)


@dataclass
class ExpertLoadStats:
    """Expert utilization statistics for MoE layers."""
    layer_idx: int = 0
    per_expert_load: List[float] = field(default_factory=list)
    load_entropy: float = 0.0  # 0 = collapsed, 1 = perfectly uniform
    max_load_ratio: float = 0.0  # max_load / avg_load
    idle_experts: int = 0  # Experts with near-zero load


# ═══════════════════════════════════════════════════════════════════════
# Communication Monitor
# ═══════════════════════════════════════════════════════════════════════

class CommMonitor:
    """
    Tracks communication time per operation type.

    Supports nested communication (e.g., TP all-reduce inside EP all-to-all)
    by tracking the compute time separately.
    """

    def __init__(self):
        self._ops: Dict[str, CommOpStats] = defaultdict(CommOpStats)
        self._active_comm: Dict[str, float] = {}  # op_name -> wall_clock_start
        self._total_compute_time: float = 0.0
        self._compute_start: Optional[float] = None

    def start_compute(self):
        """Mark the start of a compute section."""
        self._compute_start = time.perf_counter()

    def end_compute(self):
        """Mark the end of a compute section."""
        if self._compute_start is not None:
            self._total_compute_time += time.perf_counter() - self._compute_start
            self._compute_start = None

    def start_comm(self, op_name: str, tensor_size: Optional[int] = None):
        """
        Record the start of a communication operation.

        Args:
            op_name: Human-readable operation name (e.g., 'tp_all_reduce')
            tensor_size: Size of tensor in bytes (for bandwidth calculation)
        """
        self._active_comm[op_name] = time.perf_counter()
        if tensor_size:
            self._ops[op_name].total_bytes += tensor_size

    def end_comm(self, op_name: str):
        """Record the end of a communication operation."""
        if op_name in self._active_comm:
            elapsed = time.perf_counter() - self._active_comm.pop(op_name)
            self._ops[op_name].total_time += elapsed
            self._ops[op_name].count += 1

    def get_total_comm_time(self) -> float:
        """Total time spent in communication."""
        return sum(op.total_time for op in self._ops.values())

    def get_bottleneck_ops(self, top_k: int = 5) -> List[Tuple[str, CommOpStats]]:
        """Get the most time-consuming communication operations."""
        ranked = sorted(self._ops.items(), key=lambda x: x[1].total_time, reverse=True)
        return ranked[:top_k]

    def get_summary(self) -> dict:
        """Get a human-readable summary of communication statistics."""
        total_comm = self.get_total_comm_time()
        total_time = total_comm + self._total_compute_time

        bottleneck = self.get_bottleneck_ops(3)

        return {
            "total_compute_time_s": f"{self._total_compute_time:.2f}",
            "total_comm_time_s": f"{total_comm:.2f}",
            "comm_overhead_pct": f"{total_comm / max(total_time, 1e-8) * 100:.1f}%",
            "compute_comm_ratio": f"{self._total_compute_time / max(total_comm, 1e-8):.1f}x",
            "top_bottlenecks": [
                {
                    "op": name,
                    "time_s": f"{stats.total_time:.2f}",
                    "count": stats.count,
                    "avg_ms": f"{stats.avg_time_ms:.2f}",
                    "bandwidth_gbps": f"{stats.total_bandwidth_gbps:.2f}",
                }
                for name, stats in bottleneck
            ],
        }

    def reset(self):
        """Reset all statistics."""
        self._ops.clear()
        self._active_comm.clear()
        self._total_compute_time = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Load Balance Analyzer
# ═══════════════════════════════════════════════════════════════════════

class LoadBalanceAnalyzer:
    """
    Analyzes expert load balance across MoE layers.

    Detects:
      - Expert collapse: all tokens go to a few experts
      - Expert starvation: some experts never used
      - Load imbalance: uneven token distribution
    """

    def __init__(self, n_layers: int, n_experts_per_layer: int):
        self.n_layers = n_layers
        self.n_experts = n_experts_per_layer

        # Per-layer expert load EMA
        self._load_ema: Dict[int, torch.Tensor] = {}  # layer_idx -> (n_experts,)
        self._ema_decay = 0.99

        # Warning thresholds
        self.collapse_threshold = 0.5   # Entropy below this = collapse
        self.starvation_pct = 0.01       # Expert with <1% avg load = starving
        self.overload_ratio = 5.0        # Expert with >5x avg load = overloaded

    def update(self, layer_idx: int, expert_counts: torch.Tensor):
        """
        Update load statistics from a training step.

        Args:
            layer_idx: Index of the MoE layer
            expert_counts: Token count per expert (n_experts,)
        """
        if expert_counts.sum() == 0:
            return

        load = expert_counts.float() / expert_counts.sum()

        if layer_idx in self._load_ema:
            self._load_ema[layer_idx] = (
                self._ema_decay * self._load_ema[layer_idx]
                + (1 - self._ema_decay) * load
            )
        else:
            self._load_ema[layer_idx] = load

    def analyze_layer(self, layer_idx: int) -> ExpertLoadStats:
        """Analyze load balance for a specific layer."""
        if layer_idx not in self._load_ema:
            return ExpertLoadStats(layer_idx=layer_idx)

        load = self._load_ema[layer_idx]
        n = len(load)

        # Entropy (normalized): H / log(n)
        entropy = 0.0
        for p in load:
            if p > 0:
                entropy -= p * math.log(p)
        max_entropy = math.log(n) if n > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        avg_load = 1.0 / n
        max_load = load.max().item()

        idle_count = (load < self.starvation_pct * avg_load).sum().item()
        overloaded_count = (load > self.overload_ratio * avg_load).sum().item()

        return ExpertLoadStats(
            layer_idx=layer_idx,
            per_expert_load=load.tolist(),
            load_entropy=normalized_entropy,
            max_load_ratio=max_load / avg_load if avg_load > 0 else 0.0,
            idle_experts=idle_count,
        )

    def get_warnings(self) -> List[str]:
        """Get human-readable warnings about load balance issues."""
        warnings = []
        for layer_idx in range(self.n_layers):
            stats = self.analyze_layer(layer_idx)
            if stats.load_entropy == 0:
                continue  # No data yet

            if stats.load_entropy < self.collapse_threshold:
                warnings.append(
                    f"Layer {layer_idx}: EXPERT COLLAPSE — entropy={stats.load_entropy:.3f}, "
                    f"{stats.idle_experts} idle experts, max load={stats.max_load_ratio:.1f}x avg"
                )
            elif stats.idle_experts > 0:
                warnings.append(
                    f"Layer {layer_idx}: {stats.idle_experts} experts starved "
                    f"(<{self.starvation_pct*100:.0f}% avg load)"
                )

        return warnings

    def get_summary(self) -> dict:
        """Get overall load balance summary."""
        entropies = []
        idle_counts = []
        max_ratios = []

        for layer_idx in range(self.n_layers):
            stats = self.analyze_layer(layer_idx)
            if stats.load_entropy > 0:
                entropies.append(stats.load_entropy)
                idle_counts.append(stats.idle_experts)
                max_ratios.append(stats.max_load_ratio)

        if not entropies:
            return {"status": "no_data"}

        return {
            "avg_load_entropy": f"{sum(entropies)/len(entropies):.4f}",
            "min_load_entropy": f"{min(entropies):.4f}",
            "total_idle_experts": sum(idle_counts),
            "max_load_ratio": f"{max(max_ratios):.1f}x",
            "status": "HEALTHY" if min(entropies) > self.collapse_threshold else "WARNING: collapse detected",
        }


# ═══════════════════════════════════════════════════════════════════════
# Pipeline Bubble Analyzer
# ═══════════════════════════════════════════════════════════════════════

class PipelineBubbleAnalyzer:
    """
    Measures pipeline bubble ratio (idle time waiting for other stages).

    In a pipeline with pp stages and m microbatches:
      Bubble ratio ≈ (pp - 1) / (pp - 1 + m)

    This decreases as m increases (more microbatches = less bubble).
    NOTE: record_step() is for actual measurements but requires integration
    with the training loop (not yet wired up). Currently provides theoretical values.
    """

    def __init__(self, pp_size: int, num_microbatches: int):
        self.pp_size = pp_size
        self.num_microbatches = num_microbatches

        # Theoretical bubble ratio (1F1B schedule)
        self.theoretical_bubble = (pp_size - 1) / (pp_size - 1 + num_microbatches)

        # Actual measurements
        self._total_idle_time: float = 0.0
        self._total_step_time: float = 0.0
        self._step_count: int = 0

    def record_step(self, compute_time: float, idle_time: float):
        """Record actual times for one pipeline step."""
        if not hasattr(self, '_total_compute_time'):
            self._total_compute_time = 0.0
        self._total_compute_time += compute_time
        self._total_idle_time += idle_time
        self._total_step_time += compute_time + idle_time
        self._step_count += 1

    def get_actual_bubble_ratio(self) -> float:
        """Actual measured bubble ratio."""
        if self._total_step_time == 0:
            return 0.0
        return self._total_idle_time / self._total_step_time

    def get_efficiency(self) -> float:
        """Pipeline efficiency (1 - bubble_ratio)."""
        return 1.0 - self.get_actual_bubble_ratio()

    def get_summary(self) -> dict:
        return {
            "pp_size": self.pp_size,
            "num_microbatches": self.num_microbatches,
            "theoretical_bubble_ratio": f"{self.theoretical_bubble:.1%}",
            "actual_bubble_ratio": f"{self.get_actual_bubble_ratio():.1%}",
            "pipeline_efficiency": f"{self.get_efficiency():.1%}",
            "recommendation": self._get_recommendation(),
        }

    def _get_recommendation(self) -> str:
        """Generate recommendation for reducing bubble."""
        actual = self.get_actual_bubble_ratio()
        if actual < 0.05:
            return "Pipeline efficiency excellent (<5% bubble)"
        elif actual < 0.15:
            return f"Increase microbatches from {self.num_microbatches} to {int(self.num_microbatches * 1.5)} for <10% bubble"
        elif actual < 0.30:
            return f"Consider reducing pp_size or increasing microbatches to {self.num_microbatches * 2}"
        else:
            return "High bubble ratio! Reduce pp_size or use GPipe schedule instead"


# ═══════════════════════════════════════════════════════════════════════
# Unified Monitor
# ═══════════════════════════════════════════════════════════════════════

class ParallelismMonitor:
    """
    Unified monitor for communication bottlenecks and load balancing.

    Usage:
        monitor = ParallelismMonitor(
            enabled=True,
            log_every=100,
            n_layers=52,
            n_experts_per_layer=640,
            pp_size=4,
            num_microbatches=16,
        )

        # In training loop:
        monitor.start_step()

        # Wrap communication calls:
        monitor.comm_start("tp_all_reduce")
        dist.all_reduce(tensor)
        monitor.comm_end("tp_all_reduce")

        # After MoE forward:
        monitor.update_expert_load(layer_idx, expert_counts)

        # End of step:
        monitor.end_step(global_step, loss)
    """

    def __init__(
        self,
        enabled: bool = True,
        log_every: int = 100,
        n_layers: int = 32,
        n_experts_per_layer: int = 128,
        pp_size: int = 1,
        num_microbatches: int = 1,
    ):
        self.enabled = enabled
        self.log_every = log_every

        self.comm_monitor = CommMonitor()
        self.load_analyzer = LoadBalanceAnalyzer(n_layers, n_experts_per_layer)
        self.bubble_analyzer = PipelineBubbleAnalyzer(pp_size, num_microbatches)

        self._step_start_time: Optional[float] = None
        self._step_compute_time: float = 0.0
        self._step_losses: List[float] = []
        self._step_tokens: int = 0
        self._global_step: int = 0

    def start_step(self):
        """Call at the beginning of each training step."""
        if not self.enabled:
            return
        self._step_start_time = time.perf_counter()

    def start_compute(self):
        """Mark start of compute section (forward/backward)."""
        if not self.enabled:
            return
        self.comm_monitor.start_compute()

    def end_compute(self):
        """Mark end of compute section."""
        if not self.enabled:
            return
        self.comm_monitor.end_compute()

    def comm_start(self, op_name: str, tensor_size: Optional[int] = None):
        """Record start of communication."""
        if not self.enabled:
            return
        self.comm_monitor.start_comm(op_name, tensor_size)

    def comm_end(self, op_name: str):
        """Record end of communication."""
        if not self.enabled:
            return
        self.comm_monitor.end_comm(op_name)

    def update_expert_load(self, layer_idx: int, expert_counts: torch.Tensor):
        """Update expert load statistics for a MoE layer."""
        if not self.enabled:
            return
        self.load_analyzer.update(layer_idx, expert_counts)

    def end_step(self, global_step: int, loss: float, tokens_processed: int = 0):
        """
        Call at the end of each training step.

        Automatically logs diagnostics every `log_every` steps.
        """
        if not self.enabled:
            return

        self._global_step = global_step
        self._step_losses.append(loss)
        self._step_tokens += tokens_processed

        # Compute per-step timing
        if self._step_start_time is not None:
            total_step_time = time.perf_counter() - self._step_start_time
            comm_time = self.comm_monitor.get_total_comm_time()
            compute_time = max(0, total_step_time - comm_time)
            idle_time = max(0, total_step_time - compute_time - comm_time)

            # Wire actual measurements into bubble analyzer
            self.bubble_analyzer.record_step(compute_time, idle_time)

            self._step_compute_time += compute_time
            self._step_start_time = None

        # Periodic logging
        if global_step % self.log_every == 0 and global_step > 0:
            self._log_diagnostics(global_step)

    def _log_diagnostics(self, step: int):
        """Log comprehensive diagnostics."""
        n_steps = max(len(self._step_losses), 1)
        avg_loss = sum(self._step_losses) / n_steps
        total_comm = self.comm_monitor.get_total_comm_time()
        total_comp = self._step_compute_time

        # Communication summary
        comm_summary = self.comm_monitor.get_summary()

        # Load balance summary
        load_summary = self.load_analyzer.get_summary()
        load_warnings = self.load_analyzer.get_warnings()

        # Pipeline bubble summary
        bubble_summary = self.bubble_analyzer.get_summary()

        # Print concise report
        separator = "-" * 65
        print(f"\n{separator}")
        print(f"  DIAGNOSTICS @ Step {step}")
        print(f"{separator}")
        print(f"  Loss:              {avg_loss:.4f}")
        print(f"  Compute/Comm:      {comm_summary['compute_comm_ratio']}")
        print(f"  Comm overhead:     {comm_summary['comm_overhead_pct']}")
        print(f"  Load entropy:      {load_summary.get('avg_load_entropy', 'N/A')}")
        print(f"  Load status:       {load_summary.get('status', 'N/A')}")
        pp_size_enabled = self.bubble_analyzer.pp_size > 1
        if pp_size_enabled:
            print(f"  Pipeline bubble:   {bubble_summary['actual_bubble_ratio']}")
            print(f"  Pipeline eff:      {bubble_summary['pipeline_efficiency']}")

        # Top bottlenecks
        bottlenecks = self.comm_monitor.get_bottleneck_ops(3)
        if bottlenecks:
            print(f"  --- Top Communication Bottlenecks ---")
            for name, stats in bottlenecks:
                print(f"  {name:<25s} {stats.avg_time_ms:>8.2f}ms  x{stats.count:>6d}  {stats.total_bandwidth_gbps:>6.1f} GB/s")

        # Warnings
        if load_warnings:
            print(f"  --- Load Balance Warnings ---")
            for w in load_warnings[:3]:
                print(f"  ! {w}")
            if len(load_warnings) > 3:
                print(f"  ... and {len(load_warnings) - 3} more warnings")

        print(f"{separator}\n")

        # Reset accumulators for next window
        self._step_losses.clear()
        self._step_compute_time = 0.0
        self._step_tokens = 0

    def get_full_report(self) -> dict:
        """Get a comprehensive diagnostics report as a dict."""
        return {
            "communication": self.comm_monitor.get_summary(),
            "load_balance": self.load_analyzer.get_summary(),
            "load_warnings": self.load_analyzer.get_warnings(),
            "pipeline": self.bubble_analyzer.get_summary(),
        }

    def reset(self):
        """Reset all statistics."""
        self.comm_monitor.reset()
        self._step_losses.clear()
        self._step_compute_time = 0.0
