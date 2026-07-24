"""
DeepSeekMoE: Mixture of Experts Feed-Forward Network
=====================================================
Implements the DeepSeek V3 style MoE architecture with:

1. **Shared Experts**: Always-active experts that capture common knowledge
   patterns across all tokens. These function like a dense FFN backbone.

2. **Routed Experts**: Fine-grained sparsely-activated experts gated by
   a learned router. Each token selects top-k experts.

3. **Auxiliary-Loss-Free Load Balancing**: Instead of adding an auxiliary
   loss term (which hurts model quality), each expert has a learnable bias
   that is dynamically adjusted based on load:
     - Overloaded expert → bias decreases
     - Underloaded expert → bias increases
   The bias is added to router scores before top-k selection.

Reference:
  "DeepSeek-V3 Technical Report" (DeepSeek-AI, 2024)
  "DeepSeekMoE: Towards Ultimate Expert Specialization" (Dai et al., 2024)

Architecture:
  For each token x:
    1. Shared experts:  h_shared = Σ shared_expert_i(x)     [always active]
    2. Router:          scores = softmax(router(x) / τ)      [per-expert prob]
    3. Gating:          top-k experts selected via scores + bias
    4. Routed experts:  h_routed = Σ gate_i · expert_i(x)   [sparse]
    5. Output:          h = h_shared + h_routed

Parameter scaling for Mamformer Ultra-7B:
  - 2 shared experts × dim 2304:  2 × 3 × 4096 × 2304 ≈ 56.6M params
  - 128 routed experts × dim 576: 128 × 3 × 4096 × 576 ≈ 905M params
  - Router: 4096 × 128 ≈ 0.5M params
  - Per-layer total: ~962M params (vs 113M dense FFN)
  - Active per token: shared(56.6M) + 8×routed(8×7.1M) ≈ 113M ≈ original d_ff
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSeekMoE(nn.Module):
    """
    DeepSeek-style Mixture of Experts FFN with shared and routed experts.

    Replaces the dense SwiGLU FFN in each Mamformer block. Shared experts
    are always active and capture common patterns. Routed experts are
    sparsely activated via top-k gating with aux-loss-free load balancing.

    Args:
        d_model: Input/output hidden dimension
        n_shared_experts: Number of always-active shared experts
        shared_expert_dim: Intermediate dimension per shared expert (SwiGLU)
        n_routed_experts: Total number of sparsely-activated routed experts
        top_k: Number of routed experts activated per token
        routed_expert_dim: Intermediate dimension per routed expert (SwiGLU)
        router_temperature: Temperature for gating softmax (default 1.0)
        aux_loss_free: Use dynamic bias instead of auxiliary loss (default True)
        bias_update_speed: β for bias adjustment in aux-loss-free mode
        dropout: Dropout rate applied after expert computation
    """

    def __init__(
        self,
        d_model: int,
        n_shared_experts: int = 2,
        shared_expert_dim: int = 2304,
        n_routed_experts: int = 64,
        top_k: int = 8,
        routed_expert_dim: int = 576,
        router_temperature: float = 1.0,
        aux_loss_free: bool = True,
        bias_update_speed: float = 0.001,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_shared_experts = n_shared_experts
        self.shared_expert_dim = shared_expert_dim
        self.n_routed_experts = n_routed_experts
        self.top_k = top_k
        self.routed_expert_dim = routed_expert_dim
        self.router_temperature = router_temperature
        self.aux_loss_free = aux_loss_free
        self.bias_update_speed = bias_update_speed
        self.dropout_rate = dropout

        # ── Shared Experts (always active) ──────────────────────────
        # Each is a full SwiGLU FFN: gate_proj, up_proj, down_proj
        self.shared_experts = nn.ModuleList([
            _SwiGLUExpert(d_model, shared_expert_dim, dropout)
            for _ in range(n_shared_experts)
        ])

        # ── Routed Experts (sparsely activated) ──────────────────────
        self.routed_experts = nn.ModuleList([
            _SwiGLUExpert(d_model, routed_expert_dim, dropout)
            for _ in range(n_routed_experts)
        ])

        # ── Router ───────────────────────────────────────────────────
        # Projects d_model → n_routed_experts to produce routing logits
        self.router = nn.Linear(d_model, n_routed_experts, bias=False)

        # ── Aux-Loss-Free Load Balancing ─────────────────────────────
        # Per-expert learnable bias, updated dynamically during training
        if aux_loss_free:
            self.register_buffer(
                "expert_bias",
                torch.zeros(n_routed_experts),
            )
            # Exponential moving average of expert loads
            self.register_buffer(
                "expert_load_ema",
                torch.ones(n_routed_experts) / n_routed_experts,
            )

        # ── Statistics ───────────────────────────────────────────────
        self.register_buffer("_total_tokens", torch.zeros(1, dtype=torch.long))
        # Track expert selection counts (exposed for monitoring)
        self.register_buffer(
            "_expert_counts",
            torch.zeros(n_routed_experts, dtype=torch.long),
        )

        DeepSeekMoE._init_weights(self)  # Explicit dispatch for subclass safety

    def _init_weights(self):
        """Initialize router with small weights for balanced initial routing."""
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02 / self.n_routed_experts ** 0.5)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through DeepSeekMoE.

        Args:
            x: Input tensor (batch, seqlen, d_model)

        Returns:
            (output, aux_info) where:
              - output: (batch, seqlen, d_model)
              - aux_info: dict with routing statistics for logging
        """
        batch_size, seq_len, d_model = x.shape
        aux_info: dict = {}

        # ── 1. Shared experts (always active) ──────────────────────
        shared_output = torch.zeros(batch_size, seq_len, d_model, device=x.device, dtype=x.dtype)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x)

        # ── 2. Router: compute per-expert scores ───────────────────
        router_logits = self.router(x)  # (batch, seqlen, n_routed_experts)

        # Apply temperature scaling
        if self.router_temperature != 1.0:
            router_logits = router_logits / self.router_temperature

        # ── 3. Expert selection with load balancing ────────────────
        if self.aux_loss_free and self.training:
            # Add expert bias to encourage load balancing
            gating_scores = router_logits + self.expert_bias
        else:
            gating_scores = router_logits

        # Top-k selection
        top_k_gates, top_k_indices = torch.topk(
            gating_scores, k=self.top_k, dim=-1
        )  # (batch, seqlen, top_k)

        # Softmax normalize gates over selected experts
        top_k_gates = F.softmax(top_k_gates, dim=-1)

        # ── 4. Compute routed expert outputs ──────────────────────
        routed_output = self._compute_routed_experts(
            x, top_k_indices, top_k_gates, batch_size, seq_len
        )

        # ── 5. Update load balancing (aux-loss-free) ──────────────
        if self.aux_loss_free and self.training:
            self._update_expert_bias(top_k_indices, batch_size * seq_len)
            aux_info["expert_bias_mean"] = self.expert_bias.mean().item()
            aux_info["expert_bias_std"] = self.expert_bias.std().item()

        # Track expert usage for monitoring (vectorized, no per-expert sync)
        if self.training:
            with torch.no_grad():
                self._total_tokens += batch_size * seq_len
                # Use bincount instead of per-expert .item() calls
                flat_indices = top_k_indices.flatten().long()
                counts = torch.bincount(flat_indices, minlength=self.n_routed_experts)
                self._expert_counts += counts

        aux_info["routed_expert_count"] = self.n_routed_experts
        aux_info["active_experts"] = self.top_k

        # ── 6. Combine shared + routed ────────────────────────────
        output = shared_output + routed_output

        return output, aux_info

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

        Returns:
            dict with keys:
              - per_expert_load: fraction of tokens per expert
              - load_entropy: entropy of load distribution (higher = more balanced)
              - max_load_ratio: ratio of most-loaded to average load
              - bias_values: current expert bias values
        """
        if self._total_tokens == 0:
            return {"per_expert_load": None, "load_entropy": 0.0}

        counts = self._expert_counts.float()
        total_assignments = counts.sum()
        if total_assignments == 0:
            return {"per_expert_load": None, "load_entropy": 0.0}

        per_expert_load = counts / total_assignments
        # Entropy (normalized): H / H_max, where H_max = log(n_experts)
        load_entropy = 0.0
        for p in per_expert_load:
            if p > 0:
                load_entropy -= p * math.log(p)
        max_entropy = math.log(self.n_routed_experts)
        normalized_entropy = load_entropy / max_entropy if max_entropy > 0 else 0.0

        max_load = per_expert_load.max().item()
        avg_load = 1.0 / self.n_routed_experts

        return {
            "per_expert_load": per_expert_load.tolist(),
            "load_entropy": normalized_entropy,
            "max_load_ratio": max_load / avg_load if avg_load > 0 else 0.0,
            "bias_values": self.expert_bias.tolist() if self.aux_loss_free else None,
        }

    def reset_load_statistics(self) -> None:
        """Reset per-expert token counters (call at start of eval)."""
        self._expert_counts.zero_()
        self._total_tokens.zero_()

    @property
    def total_expert_params(self) -> int:
        """Total parameters across all experts (shared + routed)."""
        shared_params = sum(
            sum(p.numel() for p in expert.parameters())
            for expert in self.shared_experts
        )
        routed_params = sum(
            sum(p.numel() for p in expert.parameters())
            for expert in self.routed_experts
        )
        router_params = sum(p.numel() for p in self.router.parameters())
        return shared_params + routed_params + router_params

    @property
    def active_params_per_token(self) -> int:
        """Parameters activated per token (shared + top_k routed)."""
        shared_params = sum(
            sum(p.numel() for p in expert.parameters())
            for expert in self.shared_experts
        )
        one_expert_params = sum(
            p.numel() for p in self.routed_experts[0].parameters()
        )
        router_params = sum(p.numel() for p in self.router.parameters())
        return shared_params + self.top_k * one_expert_params + router_params

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"n_shared={self.n_shared_experts} (dim {self.shared_expert_dim}), "
            f"n_routed={self.n_routed_experts} (dim {self.routed_expert_dim}), "
            f"top_k={self.top_k}, "
            f"load_balance={'aux-loss-free' if self.aux_loss_free else 'aux_loss'}"
        )


# ── Shared MoE helpers (used by both DeepSeekMoE and SpaceTimeMoE) ─────────

def _moe_compute_routed_experts(
    x: torch.Tensor, top_k_indices: torch.Tensor, top_k_gates: torch.Tensor,
    batch_size: int, seq_len: int, d_model: int, top_k: int,
    n_routed_experts: int, routed_experts: nn.ModuleList,
) -> torch.Tensor:
    """Compute routed expert outputs efficiently. Shared by DeepSeekMoE and SpaceTimeMoE."""
    output = torch.zeros(batch_size, seq_len, d_model, device=x.device, dtype=x.dtype)
    x_flat = x.view(batch_size * seq_len, d_model)
    idx_flat = top_k_indices.view(batch_size * seq_len, top_k)
    gate_flat = top_k_gates.view(batch_size * seq_len, top_k)
    for expert_idx in range(n_routed_experts):
        expert_mask = (idx_flat == expert_idx)
        token_has_expert = expert_mask.any(dim=-1)
        if not token_has_expert.any():
            continue
        expert_input = x_flat[token_has_expert]
        expert_output = routed_experts[expert_idx](expert_input)
        slot_indices = expert_mask[token_has_expert].float().argmax(dim=-1)
        token_indices = token_has_expert.nonzero(as_tuple=True)[0]
        gates_for_tokens = gate_flat[token_indices, slot_indices]
        expert_output = expert_output * gates_for_tokens.unsqueeze(-1)
        output.view(batch_size * seq_len, d_model)[token_indices] += expert_output
    return output


def _moe_update_expert_bias(
    top_k_indices: torch.Tensor, total_tokens: int,
    n_routed_experts: int, top_k: int, bias_update_speed: float,
    expert_bias: torch.Tensor, expert_load_ema: torch.Tensor,
) -> None:
    """Aux-loss-free expert bias update. Shared by DeepSeekMoE and SpaceTimeMoE."""
    flat_indices = top_k_indices.flatten().long()
    expert_counts = torch.bincount(flat_indices, minlength=n_routed_experts).float()
    actual_load = expert_counts / (total_tokens * top_k + 1e-8)
    expected_load = 1.0 / n_routed_experts
    load_deviation = actual_load - expected_load
    expert_bias -= bias_update_speed * torch.sign(load_deviation)
    expert_load_ema.copy_(0.99 * expert_load_ema + 0.01 * actual_load)


class _SwiGLUExpert(nn.Module):
    """
    A single SwiGLU expert — identical architecture to the dense FFN
    but with a smaller intermediate dimension.

    SwiGLU(x) = (SiLU(x @ W_gate)) * (x @ W_up) @ W_down
    """

    def __init__(
        self,
        d_model: int,
        intermediate_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for proj in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        gated = gate * up
        gated = self.dropout(gated)
        return self.down_proj(gated)


class MoERouter(nn.Module):
    """
    Standalone MoE router for use when you want to swap routing strategies.

    Currently implements top-k routing. Can be extended with:
      - Top-p routing (stochastic expert selection)
      - Expert choice routing (experts choose tokens)
      - Hash-based routing (deterministic, no learned parameters)
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 8,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.temperature = temperature

        self.weight = nn.Parameter(torch.empty(n_experts, d_model))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02 / self.n_experts ** 0.5)

    def forward(
        self, x: torch.Tensor, expert_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seqlen, d_model)
            expert_bias: Optional (n_experts,) bias for load balancing

        Returns:
            (top_k_gates, top_k_indices) — gates are softmax-normalized
        """
        logits = F.linear(x, self.weight)  # (batch, seqlen, n_experts)

        if self.temperature != 1.0:
            logits = logits / self.temperature

        if expert_bias is not None:
            logits = logits + expert_bias

        top_k_gates, top_k_indices = torch.topk(logits, k=self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_gates, dim=-1)

        return top_k_gates, top_k_indices
