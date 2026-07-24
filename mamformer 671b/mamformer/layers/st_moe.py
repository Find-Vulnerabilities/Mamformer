"""
Space-Time MoE (ST-MoE): Temporal-State-Guided Expert Routing
===============================================================
A paradigm innovation that couples Mamba-2's temporal hidden state
with Transformer's dynamic spatial representations for expert routing.

Traditional MoE routing is spatially static:
    G(x_t) = Softmax(W_g · x_t)                    [space only]

ST-MoE introduces temporal inertia via Mamba SSM state h_t:
    Logits_t = W_g · x_t + λ · (W_h · h_t)          [space + time]
    G(x_t, h_t) = Top-K(Softmax(Logits_t), K)

Where:
  - W_g: spatial router (existing, d_model → n_experts)
  - W_h: temporal projection (d_state → n_experts)
  - h_t: Mamba-2 SSM hidden state at position t, mean-pooled over d_inner
  - λ:   temporal guidance weight, clamped to [0, λ_max] for safety

Three safety mechanisms:
  1. Residual Decoupling: λ ≤ 0.3 — static routing always dominates
  2. Dynamic Balance Lock: zero temporal bias when expert over-used
  3. Capacity Limiting: hard reset on overload detection

Reference:
  "Space-Time MoE: Temporal-State-Guided Expert Routing for
   Long-Context Language Models" (Research Report, 2025)

Architecture:
  For each token x_t with SSM state h_t:
    1. Shared experts:  h_shared = Σ shared_expert_i(x_t)         [always active]
    2. Spatial logits:  s_t = router(x_t)                          [W_g · x_t]
    3. Temporal bias:   b_t = λ · temporal_proj(h_t)              [W_h · h_t]
    4. Combined:        logits_t = s_t + b_t                       [space+time]
    5. Balance lock:    zero temporal bias for overloaded experts
    6. Gating:          top-k selection + softmax normalization
    7. Routed experts:  h_routed = Σ gate_i · expert_i(x_t)       [sparse]
    8. Output:          h = h_shared + h_routed
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamformer.layers.moe import (
    DeepSeekMoE, _SwiGLUExpert,
    _moe_compute_routed_experts, _moe_update_expert_bias,
)


class SpaceTimeMoE(DeepSeekMoE):
    """
    Space-Time MoE — extends DeepSeekMoE with temporal-state-guided routing.
    Inherits experts, router, load balancing from DeepSeekMoE.

    Args:
        d_model: Input/output hidden dimension
        n_shared_experts: Number of always-active shared experts
        shared_expert_dim: Intermediate dimension per shared expert (SwiGLU)
        n_routed_experts: Total number of sparsely-activated routed experts
        top_k: Number of routed experts activated per token
        routed_expert_dim: Intermediate dimension per routed expert (SwiGLU)
        d_state: SSM state dimension (from Mamba-2, default 128)
        lambda_init: Initial temporal guidance weight λ (default 0.2)
        lambda_max: Safety clamp for λ — residual decoupling (default 0.3)
        learnable_lambda: Whether λ is a learnable parameter
        use_balance_lock: Enable dynamic balance lock
        balance_lock_threshold: Max consecutive expert activations before lock
        router_temperature: Temperature for gating softmax
        aux_loss_free: Use dynamic bias instead of auxiliary loss
        bias_update_speed: β for bias adjustment in aux-loss-free mode
        dropout: Dropout rate after expert computation
    """

    def __init__(
        self,
        d_model: int,
        n_shared_experts: int = 2,
        shared_expert_dim: int = 2304,
        n_routed_experts: int = 64,
        top_k: int = 8,
        routed_expert_dim: int = 576,
        d_state: int = 128,
        lambda_init: float = 0.2,
        lambda_max: float = 0.3,
        learnable_lambda: bool = True,
        use_balance_lock: bool = True,
        balance_lock_threshold: int = 50,
        router_temperature: float = 1.0,
        aux_loss_free: bool = True,
        bias_update_speed: float = 0.001,
        dropout: float = 0.0,
    ) -> None:
        # Shared experts, router, load-balancing: delegated to DeepSeekMoE
        super().__init__(
            d_model=d_model, n_shared_experts=n_shared_experts,
            shared_expert_dim=shared_expert_dim, n_routed_experts=n_routed_experts,
            top_k=top_k, routed_expert_dim=routed_expert_dim,
            router_temperature=router_temperature, aux_loss_free=aux_loss_free,
            bias_update_speed=bias_update_speed, dropout=dropout,
        )

        # ── ST-specific attributes ──────────────────────────────────
        self.d_state = d_state
        self.lambda_max = lambda_max
        self.learnable_lambda = learnable_lambda
        self.use_balance_lock = use_balance_lock
        self.balance_lock_threshold = balance_lock_threshold

        # Temporal guidance λ: λ = λ_max * sigmoid(λ_raw), clamped to [0, λ_max]
        if learnable_lambda:
            init_ratio = lambda_init / lambda_max
            self.lambda_raw = nn.Parameter(
                torch.tensor(math.log(init_ratio / (1.0 - init_ratio)), dtype=torch.float32))
        else:
            self.register_buffer("lambda_raw", torch.tensor(lambda_init, dtype=torch.float32))

        # Temporal projection: d_state → n_experts
        self.temporal_proj = nn.Linear(d_state, n_routed_experts, bias=False)

        # Balance lock state (track consecutive overloaded steps)
        if use_balance_lock:
            self.register_buffer("_consecutive_count", torch.zeros(n_routed_experts, dtype=torch.long))
        self.register_buffer("_lock_trigger_count", torch.zeros(1, dtype=torch.long))

        # Init temporal_proj (router + experts already initialized by DeepSeekMoE)
        nn.init.normal_(self.temporal_proj.weight, mean=0.0, std=0.001)

    @property
    def lambda_value(self) -> float:
        """Get current λ value (clamped for safety)."""
        if self.learnable_lambda:
            lam = self.lambda_max * torch.sigmoid(self.lambda_raw)
        else:
            lam = self.lambda_raw
        return lam.item()

    def _get_lambda_tensor(self) -> torch.Tensor:
        """Get λ as a tensor, clamped to [0, lambda_max]."""
        if self.learnable_lambda:
            return self.lambda_max * torch.sigmoid(self.lambda_raw)
        return self.lambda_raw.clamp(0.0, self.lambda_max)

    def forward(
        self,
        x: torch.Tensor,
        ssm_h_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through Space-Time MoE.

        Args:
            x: Input tensor (batch, seqlen, d_model)
            ssm_h_states: Optional Mamba-2 SSM state summaries.
                         Shape (batch, seqlen, d_state).
                         When None, falls back to pure spatial routing
                         (equivalent to standard DeepSeekMoE).

        Returns:
            (output, aux_info) where:
              - output: (batch, seqlen, d_model)
              - aux_info: dict with routing statistics for logging
        """
        batch_size, seq_len, d_model = x.shape
        aux_info: dict = {}

        # ── 1. Shared experts (always active) ──────────────────────
        shared_output = torch.zeros(
            batch_size, seq_len, d_model, device=x.device, dtype=x.dtype
        )
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x)

        # ── 2. Spatial routing: W_g · x_t ─────────────────────────
        spatial_logits = self.router(x)  # (batch, seqlen, n_routed_experts)

        # ── 3. Temporal routing: λ · W_h · h_t ────────────────────
        lambda_val = self._get_lambda_tensor()
        temporal_bias = torch.zeros_like(spatial_logits)

        if ssm_h_states is not None and lambda_val > 0:
            # Project SSM state → expert logits space
            # ssm_h_states: (batch, seqlen, d_state)
            # temporal_proj: Linear(d_state, n_routed_experts)
            temporal_bias = self.temporal_proj(ssm_h_states)
            temporal_bias = temporal_bias * lambda_val

        # ── 4. Combined logits: space + time ───────────────────────
        router_logits = spatial_logits + temporal_bias

        # Apply temperature scaling
        if self.router_temperature != 1.0:
            router_logits = router_logits / self.router_temperature

        # ── 5. Dynamic Balance Lock ────────────────────────────────
        if self.use_balance_lock and self.training and ssm_h_states is not None:
            router_logits, lock_info = self._apply_balance_lock(
                router_logits, spatial_logits
            )
            aux_info["locks_triggered"] = lock_info["locks_triggered"]

        # ── 6. Expert selection ────────────────────────────────────
        if self.aux_loss_free and self.training:
            gating_scores = router_logits + self.expert_bias
        else:
            gating_scores = router_logits

        # Top-k selection
        top_k_gates, top_k_indices = torch.topk(
            gating_scores, k=self.top_k, dim=-1
        )  # (batch, seqlen, top_k)

        # Softmax normalize gates over selected experts
        top_k_gates = F.softmax(top_k_gates, dim=-1)

        # ── 7. Compute routed expert outputs ──────────────────────
        routed_output = self._compute_routed_experts(
            x, top_k_indices, top_k_gates, batch_size, seq_len
        )

        # ── 8. Update load balancing (aux-loss-free) ──────────────
        if self.aux_loss_free and self.training:
            self._update_expert_bias(top_k_indices, batch_size * seq_len)
            aux_info["expert_bias_mean"] = self.expert_bias.mean().item()
            aux_info["expert_bias_std"] = self.expert_bias.std().item()

        # ── 9. Update balance lock counters ───────────────────────
        if self.use_balance_lock and self.training and ssm_h_states is not None:
            self._update_consecutive_counts(top_k_indices)

        # Track expert usage for monitoring (vectorized)
        if self.training:
            with torch.no_grad():
                self._total_tokens += batch_size * seq_len
                flat_indices = top_k_indices.flatten().long()
                counts = torch.bincount(flat_indices, minlength=self.n_routed_experts)
                self._expert_counts += counts

        # ── ST-MoE specific stats ──────────────────────────────────
        aux_info["routed_expert_count"] = self.n_routed_experts
        aux_info["active_experts"] = self.top_k
        aux_info["lambda"] = lambda_val.item()
        if ssm_h_states is not None:
            aux_info["temporal_bias_mean"] = temporal_bias.mean().item()
            aux_info["temporal_bias_std"] = temporal_bias.std().item()

        # ── 10. Combine shared + routed ───────────────────────────
        output = shared_output + routed_output

        return output, aux_info

    def _apply_balance_lock(
        self,
        combined_logits: torch.Tensor,
        spatial_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Dynamic Balance Lock: prevent temporal bias from causing expert collapse.

        When an expert is activated too many consecutive times, zero out
        the temporal contribution for that expert, falling back to pure
        spatial routing. This prevents the temporal inertia from creating
        pathological over-specialization.

        Safety guarantee: temporal bias is removed (not spatial), so routing
        still works — it just loses the temporal smoothing for locked experts.

        Args:
            combined_logits: spatial + temporal logits (batch, seqlen, n_experts)
            spatial_logits: pure spatial logits (batch, seqlen, n_experts)

        Returns:
            (adjusted_logits, lock_info)
        """
        device = combined_logits.device
        lock_info = {"locks_triggered": 0}

        # Check which experts exceed the consecutive activation threshold
        overloaded = self._consecutive_count >= self.balance_lock_threshold

        if overloaded.any():
            n_locked = overloaded.sum().item()
            lock_info["locks_triggered"] = n_locked

            # For overloaded experts: replace temporal-biased logits
            # with pure spatial logits. This effectively zeros the
            # temporal bias for those experts.
            # Shape: (1, 1, n_experts) → broadcasts over batch and seqlen
            lock_mask = overloaded.float().view(1, 1, self.n_routed_experts)
            combined_logits = (
                lock_mask * spatial_logits + (1.0 - lock_mask) * combined_logits
            )

            # Track cumulative locks triggered
            self._lock_trigger_count += n_locked

        return combined_logits, lock_info

    def _update_consecutive_counts(
        self,
        top_k_indices: torch.Tensor,
    ) -> None:
        """
        Update per-expert consecutive activation counters.

        Uses load-fraction: an expert is "overloaded" only if it receives
        significantly more tokens than expected. This avoids triggering
        the balance lock for all experts in large-batch settings.

        Args:
            top_k_indices: (batch, seqlen, top_k) expert indices per token
        """
        total_tokens = top_k_indices.numel() // self.top_k
        expected_per_expert = total_tokens * self.top_k / self.n_routed_experts
        overload_threshold = max(expected_per_expert * 1.5, 2.0)  # 1.5x expected

        # Count tokens per expert (vectorized with bincount)
        flat_indices = top_k_indices.flatten().long()
        expert_counts = torch.bincount(flat_indices, minlength=self.n_routed_experts).float()

        # An expert is "overloaded" if it receives >1.5x expected load
        overloaded = expert_counts > overload_threshold

        # Increment overloaded experts, reset non-overloaded
        self._consecutive_count = torch.where(
            overloaded,
            self._consecutive_count + 1,
            torch.zeros_like(self._consecutive_count),
        )

    def _compute_routed_experts(
        self, x: torch.Tensor, top_k_indices: torch.Tensor,
        top_k_gates: torch.Tensor, batch_size: int, seq_len: int,
    ) -> torch.Tensor:
        return _moe_compute_routed_experts(
            x, top_k_indices, top_k_gates, batch_size, seq_len,
            self.d_model, self.top_k, self.n_routed_experts, self.routed_experts,
        )

    def _update_expert_bias(self, top_k_indices: torch.Tensor, total_tokens: int) -> None:
        _moe_update_expert_bias(
            top_k_indices, total_tokens, self.n_routed_experts, self.top_k,
            self.bias_update_speed, self.expert_bias, self.expert_load_ema,
        )

    def get_load_statistics(self) -> dict:
        """
        Get expert load balancing statistics for monitoring.
        Extends base class stats with ST-MoE-specific info.
        """
        stats = super().get_load_statistics()
        stats["lambda"] = self.lambda_value
        stats["consecutive_max"] = self._consecutive_count.max().item() if self.use_balance_lock else 0
        stats["locks_triggered"] = self._lock_trigger_count.item() if self.use_balance_lock else 0
        return stats

    def reset_load_statistics(self) -> None:
        """Reset per-expert token counters (call at start of eval)."""
        self._expert_counts.zero_()
        self._total_tokens.zero_()
        if self.use_balance_lock:
            self._consecutive_count.zero_()
            self._lock_trigger_count.zero_()

    @property
    def total_expert_params(self) -> int:
        return super().total_expert_params + sum(p.numel() for p in self.temporal_proj.parameters())

    @property
    def active_params_per_token(self) -> int:
        return super().active_params_per_token + sum(p.numel() for p in self.temporal_proj.parameters())

    def extra_repr(self) -> str:
        lock_info = f", balance_lock(threshold={self.balance_lock_threshold})" if self.use_balance_lock else ""
        return (
            f"d_model={self.d_model}, "
            f"n_shared={self.n_shared_experts} (dim {self.shared_expert_dim}), "
            f"n_routed={self.n_routed_experts} (dim {self.routed_expert_dim}), "
            f"top_k={self.top_k}, "
            f"d_state={self.d_state}, "
            f"lambda={self.lambda_value:.3f} (max={self.lambda_max})"
            f"{lock_info}, "
            f"load_balance={'aux-loss-free' if self.aux_loss_free else 'aux_loss'}"
        )


