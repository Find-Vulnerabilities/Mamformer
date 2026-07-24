"""
Communicative MoE: Cross-Expert Communication Layer
=====================================================
Enables selected experts to communicate via cross-attention before
gate-weighted combination, inspired by Kimi K3's expert collaboration.

In standard MoE, each expert computes independently:
    output = Σ_i gate_i * expert_i(x)

In CommunicativeMoE, experts share information before combination:
    [e_1, ..., e_k] = [expert_1(x), ..., expert_k(x)]          # independent compute
    [c_1, ..., c_k] = CrossAttention([e_1, ..., e_k])           # communicate!
    output = Σ_i gate_i * c_i                                    # combine

This allows experts to condition their outputs on what other experts
have found — enabling specialization patterns like:
  - Expert A detects code → Expert B adds documentation context
  - Expert C finds math → Expert D verifies the calculation
  - Expert E handles English → Expert F adds translation hints

The communication is lightweight: a small multi-head cross-attention
among the top-k expert outputs for each token, with residual connections
for stable training.

Architecture:
  Token x → Router → select top-k experts
                  → compute each expert independently → (N, k, d_model)
                  → ExpertCommunicationLayer (cross-attention + FFN)
                  → gate-weighted combination → output
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamformer.layers.norm import RMSNorm


# ═══════════════════════════════════════════════════════════════════════
# Expert Communication Layer
# ═══════════════════════════════════════════════════════════════════════

class ExpertCommunicationLayer(nn.Module):
    """
    Multi-head cross-attention among top-k expert outputs.

    For each token, its k selected expert outputs attend to each other
    before being combined by gate weights. This allows experts to
    condition their contributions on what other experts are producing.

    Uses a Pre-Norm transformer block architecture:
        x = x + MHA(RMSNorm(x))     # cross-attention among experts
        x = x + FFN(RMSNorm(x))     # per-expert feed-forward

    Args:
        d_model: Hidden dimension (same as model)
        n_heads: Number of attention heads for expert communication
        depth: Number of communication layers (default 1 for efficiency)
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        depth: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = depth
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"

        # ── Per-layer components ──────────────────────────────────
        self.attn_norms = nn.ModuleList([
            RMSNorm(d_model) for _ in range(depth)
        ])
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout if dropout > 0 else 0.0,
                batch_first=True,  # (N, k, d_model)
            )
            for _ in range(depth)
        ])
        self.ffn_norms = nn.ModuleList([
            RMSNorm(d_model) for _ in range(depth)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4, bias=False),
                nn.SiLU(),
                nn.Linear(d_model * 4, d_model, bias=False),
            )
            for _ in range(depth)
        ])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialize communication layer weights."""
        std = 0.02
        for ffn in self.ffns:
            for layer in ffn:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=std)

    def forward(
        self,
        expert_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply cross-expert communication.

        Args:
            expert_outputs: (N, k, d_model) — per-token expert outputs
                           N = batch_size * seq_len
                           k = top_k (selected experts per token)

        Returns:
            Communicated expert outputs, same shape (N, k, d_model)
        """
        # expert_outputs: (N, k, d_model) where N = batch*seqlen, k = top_k
        # We want cross-attention along the k dimension:
        # Each expert (query) attends to all k experts (key/value)

        x = expert_outputs

        for i in range(self.depth):
            # ── Cross-attention among experts ──────────────────
            residual = x
            x_norm = self.attn_norms[i](x)

            # Self-attention on the expert dimension (dim=1)
            # Query, Key, Value are all the expert outputs
            attn_out, _ = self.cross_attns[i](
                query=x_norm,
                key=x_norm,
                value=x_norm,
                need_weights=False,
            )
            x = residual + self.dropout(attn_out)

            # ── Per-expert FFN ──────────────────────────────────
            residual = x
            x_norm = self.ffn_norms[i](x)
            ffn_out = self.ffns[i](x_norm)
            x = residual + self.dropout(ffn_out)

        return x

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"depth={self.depth}, head_dim={self.head_dim}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Communicative MoE Wrapper
# ═══════════════════════════════════════════════════════════════════════

class CommunicativeMoE(nn.Module):
    """
    MoE wrapper that adds cross-expert communication.

    Wraps a standard MoE (DeepSeekMoE or SpaceTimeMoE) and inserts an
    ExpertCommunicationLayer between independent expert computation
    and gate-weighted combination.

    This works by:
    1. Using the base MoE's router for expert selection (shared logic)
    2. Computing each selected expert independently on its tokens
    3. Collecting per-token expert outputs into (N, k, d_model) tensors
    4. Passing through ExpertCommunicationLayer for cross-expert info sharing
    5. Combining communicated outputs with gate weights
    6. Adding shared expert contributions from the base MoE

    Compatible with:
      - DeepSeekMoE: standard spatial routing
      - SpaceTimeMoE: temporal-state-guided routing (ssm_h_states passed through)

    Args:
        base_moe: Existing MoE module (DeepSeekMoE or SpaceTimeMoE)
        d_model: Hidden dimension
        n_comm_heads: Number of attention heads for expert communication
        comm_depth: Number of communication layers (1 recommended)
        comm_dropout: Dropout in communication layer
    """

    def __init__(
        self,
        base_moe: nn.Module,
        d_model: int,
        n_comm_heads: int = 4,
        comm_depth: int = 1,
        comm_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.base_moe = base_moe
        self.d_model = d_model

        # ── Communication layer ────────────────────────────────────
        self.comm_layer = ExpertCommunicationLayer(
            d_model=d_model,
            n_heads=n_comm_heads,
            depth=comm_depth,
            dropout=comm_dropout,
        )

        # ── Learnable gate modifier ────────────────────────────────
        # After communication, expert outputs may need reweighting.
        # Per-slot adjustment: different experts may benefit differently
        # from communication. Initialized to 0 → sigmoid(0+1) ≈ 0.73 (mild boost).
        self.gate_adjustment = nn.Parameter(torch.zeros(1, base_moe.top_k))

        # ── Communication strength (learnable) ────────────────────
        # Controls how much of the communicated signal to use vs original.
        # Initialized to 0.5 → equal mix of original and communicated.
        self.comm_strength = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5

    def forward(
        self,
        x: torch.Tensor,
        ssm_h_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with cross-expert communication.

        Args:
            x: Input tensor (batch, seqlen, d_model)
            ssm_h_states: Optional SSM state summaries for SpaceTimeMoE

        Returns:
            (output, aux_info)
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        dtype = x.dtype
        flat_N = batch_size * seq_len

        base = self.base_moe

        # ── 1. Shared experts (from base MoE) ──────────────────────
        shared_output = torch.zeros(batch_size, seq_len, d_model, device=device, dtype=dtype)
        for expert in base.shared_experts:
            shared_output = shared_output + expert(x)

        # ── 2. Router: standard or temporal ────────────────────────
        # SpaceTimeMoE: router_logits = spatial + temporal
        # DeepSeekMoE: router_logits = spatial only
        if hasattr(base, 'temporal_proj') and ssm_h_states is not None:
            # SpaceTimeMoE routing
            spatial_logits = base.router(x)
            lambda_val = base._get_lambda_tensor()
            temporal_bias = base.temporal_proj(ssm_h_states) * lambda_val
            router_logits = spatial_logits + temporal_bias
        else:
            router_logits = base.router(x)

        # ── Temperature scaling (mirrors ST-MoE and DeepSeekMoE) ───
        if base.router_temperature != 1.0:
            router_logits = router_logits / base.router_temperature

        # ── 3. Aux-loss-free bias ──────────────────────────────────
        if base.aux_loss_free and self.training:
            gating_scores = router_logits + base.expert_bias
        else:
            gating_scores = router_logits

        # ── 4. Top-k selection ────────────────────────────────────
        top_k_gates, top_k_indices = torch.topk(
            gating_scores, k=base.top_k, dim=-1
        )  # (batch, seqlen, top_k)
        top_k_gates = F.softmax(top_k_gates, dim=-1)

        # ── 5. Per-expert computation ─────────────────────────────
        # Flatten: (batch, seqlen) → N
        x_flat = x.view(flat_N, d_model)
        idx_flat = top_k_indices.view(flat_N, base.top_k)  # (N, top_k)
        gate_flat = top_k_gates.view(flat_N, base.top_k)    # (N, top_k)

        # Collect per-token expert outputs: (N, top_k, d_model)
        expert_outputs = self._compute_expert_outputs(
            x_flat, idx_flat, flat_N
        )

        # ── 6. Cross-expert communication ─────────────────────────
        communicated = self.comm_layer(expert_outputs)  # (N, top_k, d_model)

        # Mix original and communicated outputs
        mix_weight = torch.sigmoid(self.comm_strength)  # in [0, 1]
        expert_outputs = (1.0 - mix_weight) * expert_outputs + mix_weight * communicated

        # ── 7. Gate-weighted combination ───────────────────────────
        # Adjust gates after communication: per-slot learnable modulation
        # shape: gate_flat (N, top_k), gate_adjustment (1, top_k)
        adjusted_gates = gate_flat * torch.sigmoid(self.gate_adjustment + 1.0)
        adjusted_gates = adjusted_gates / (adjusted_gates.sum(dim=-1, keepdim=True) + 1e-8)

        # Weighted sum: (N, top_k, d_model) * (N, top_k, 1) → (N, d_model)
        routed_output_flat = (expert_outputs * adjusted_gates.unsqueeze(-1)).sum(dim=1)
        routed_output = routed_output_flat.view(batch_size, seq_len, d_model)

        # ── 8. Combine shared + routed ────────────────────────────
        output = shared_output + routed_output

        # ── 9. Update load balancing (delegated to base MoE) ─────
        aux_info: dict = {
            "routed_expert_count": base.n_routed_experts,
            "active_experts": base.top_k,
            "comm_strength": mix_weight.item(),
            "gate_adjustment": torch.sigmoid(self.gate_adjustment + 1.0).mean().item(),
        }
        if base.aux_loss_free and self.training:
            base._update_expert_bias(top_k_indices, flat_N)
            aux_info["expert_bias_mean"] = base.expert_bias.mean().item()
            aux_info["expert_bias_std"] = base.expert_bias.std().item()

        # Forward ST-MoE specific info
        if hasattr(base, 'lambda_value'):
            aux_info["lambda"] = base.lambda_value

        return output, aux_info

    def _compute_expert_outputs(
        self,
        x_flat: torch.Tensor,
        idx_flat: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """
        Compute each selected expert's output for its assigned tokens.

        For each token position n and each of its k selected experts,
        compute expert_{idx}(x_n) and store in expert_outputs[n, k, :].

        Uses batched computation: each expert is called once with all
        tokens assigned to it, then outputs are scattered into per-token slots.

        Args:
            x_flat: Flattened input (N, d_model)
            idx_flat: Expert indices (N, top_k) — which experts per token
            N: Total number of tokens (batch * seqlen)

        Returns:
            expert_outputs: (N, top_k, d_model)
        """
        d_model = self.d_model
        top_k = idx_flat.shape[1]
        base = self.base_moe

        # Initialize output buffer
        expert_outputs = torch.zeros(N, top_k, d_model, device=x_flat.device, dtype=x_flat.dtype)

        # For each expert, compute output and scatter to token positions
        for global_expert_idx in range(base.n_routed_experts):
            # Find (token, slot) pairs that use this expert
            expert_mask = (idx_flat == global_expert_idx)  # (N, top_k)

            if not expert_mask.any():
                continue

            # Get token indices and slot indices
            token_indices, slot_indices = expert_mask.nonzero(as_tuple=True)
            # token_indices: which tokens use this expert
            # slot_indices: which slot (0..top_k-1) the expert occupies

            # Compute expert output for these tokens (batched)
            expert_input = x_flat[token_indices]  # (n_assigned, d_model)
            expert_output = base.routed_experts[global_expert_idx](expert_input)

            # Scatter into output buffer
            expert_outputs[token_indices, slot_indices] = expert_output

        return expert_outputs

    # ── Delegated properties ───────────────────────────────────────

    @property
    def n_routed_experts(self) -> int:
        return self.base_moe.n_routed_experts

    @property
    def top_k(self) -> int:
        return self.base_moe.top_k

    def get_load_statistics(self) -> dict:
        """Get load balancing statistics from base MoE."""
        if hasattr(self.base_moe, 'get_load_statistics'):
            stats = self.base_moe.get_load_statistics()
            stats["comm_strength"] = torch.sigmoid(self.comm_strength).item()
            return stats
        return {"comm_strength": torch.sigmoid(self.comm_strength).item()}

    def reset_load_statistics(self) -> None:
        """Reset load statistics."""
        if hasattr(self.base_moe, 'reset_load_statistics'):
            self.base_moe.reset_load_statistics()

    @property
    def total_expert_params(self) -> int:
        """Total parameters across experts + communication layer."""
        base_params = self.base_moe.total_expert_params if hasattr(self.base_moe, 'total_expert_params') else sum(
            p.numel() for p in self.base_moe.parameters()
        )
        comm_params = sum(p.numel() for p in self.comm_layer.parameters())
        return base_params + comm_params

    @property
    def active_params_per_token(self) -> int:
        """Active parameters per token including communication."""
        base_active = self.base_moe.active_params_per_token if hasattr(self.base_moe, 'active_params_per_token') else self.base_moe.total_expert_params
        comm_params = sum(p.numel() for p in self.comm_layer.parameters())
        return base_active + comm_params

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"n_routed={self.n_routed_experts}, top_k={self.top_k}, "
            f"comm_heads={self.comm_layer.n_heads}, comm_depth={self.comm_layer.depth}, "
            f"base_moe={type(self.base_moe).__name__}"
        )
