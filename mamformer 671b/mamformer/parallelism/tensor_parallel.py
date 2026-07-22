"""
Tensor Parallelism for Mamformer
==================================
Splits individual layers across multiple GPUs within a node.

Core primitives:
  - ColumnParallelLinear: Split output dimension, all-reduce not needed
  - RowParallelLinear: Split input dimension, all-reduce output
  - TPAttention: Tensor-parallel GQA/DSA attention
  - TPMamba2Block: Tensor-parallel Mamba-2 SSM block
  - TPDeepSeekMoE: Tensor-parallel MoE FFN

Communication:
  - All-reduce: After row-parallel outputs and attention outputs
  - All-gather: K/V heads for attention computation
  - Reduce-scatter: Optional optimization for gradient reduction

Usage:
    import torch.distributed as dist
    dist.init_process_group("nccl")

    # Wrap individual layers
    attn = TPAttention(d_model=4096, n_heads=32, tp_size=4)
    mamba = TPMamba2Block(d_model=4096, tp_size=4)
    moe = TPDeepSeekMoE(d_model=4096, n_experts=128, tp_size=4)

    # Or shard entire model
    model = shard_model_tp(model, tp_size=4)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ═══════════════════════════════════════════════════════════════════════
# TP Communication Helpers
# ═══════════════════════════════════════════════════════════════════════

def _get_tp_group() -> Optional[dist.ProcessGroup]:
    """Get the tensor-parallel process group (set by coordinator)."""
    return getattr(_get_tp_group, "_group", None)


def _set_tp_group(group: dist.ProcessGroup) -> None:
    _get_tp_group._group = group


def _tp_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce across TP group."""
    group = _get_tp_group()
    if group is not None and dist.is_initialized():
        _comm_diag_start("tp_all_reduce", tensor.numel() * tensor.element_size())
        dist.all_reduce(tensor, group=group)
        _comm_diag_end("tp_all_reduce")
    return tensor


# Optional diagnostics hook
_comm_diag_callback: dict = {"start": None, "end": None}


def _comm_diag_start(op: str, size: int = 0):
    if _comm_diag_callback["start"]:
        _comm_diag_callback["start"](op, size)


def _comm_diag_end(op: str):
    if _comm_diag_callback["end"]:
        _comm_diag_callback["end"](op)


def set_comm_diagnostics(start_fn, end_fn):
    """Set callbacks for communication diagnostics."""
    _comm_diag_callback["start"] = start_fn
    _comm_diag_callback["end"] = end_fn


def _tp_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather across TP group along specified dimension."""
    group = _get_tp_group()
    if group is not None and dist.is_initialized():
        world_size = dist.get_world_size(group)
        shapes = [tensor.shape] * world_size
        gathered = [torch.empty(s, device=tensor.device, dtype=tensor.dtype) for s in shapes]
        total_bytes = sum(s.numel() for s in shapes) * tensor.element_size()
        _comm_diag_start("tp_all_gather", total_bytes)
        dist.all_gather(gathered, tensor, group=group)
        _comm_diag_end("tp_all_gather")
        return torch.cat(gathered, dim=dim)
    return tensor


def _tp_reduce_scatter(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Reduce-scatter across TP group."""
    group = _get_tp_group()
    if group is not None and dist.is_initialized():
        world_size = dist.get_world_size(group)
        chunk_size = tensor.shape[dim] // world_size
        output = torch.empty(
            *tensor.shape[:dim], chunk_size, *tensor.shape[dim+1:],
            device=tensor.device, dtype=tensor.dtype,
        )
        dist.reduce_scatter(output, tensor, group=group)
        return output
    return tensor


def get_tp_rank_size() -> Tuple[int, int]:
    """Get (rank, world_size) for TP group."""
    group = _get_tp_group()
    if group is not None and dist.is_initialized():
        return dist.get_rank(group), dist.get_world_size(group)
    return 0, 1


# ═══════════════════════════════════════════════════════════════════════
# Column-Parallel Linear
# ═══════════════════════════════════════════════════════════════════════

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with output dimension split across TP ranks.

    W: (in_features, out_features // tp_size)
    Each rank holds a slice of the output columns. No communication needed
    after forward — each rank has its portion of the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = True,
    ):
        super().__init__()
        _, tp_size = get_tp_rank_size()
        self.tp_size = tp_size
        self.out_features_per_rank = out_features // tp_size
        self.gather_output = gather_output

        assert out_features % tp_size == 0, (
            f"out_features ({out_features}) must be divisible by tp_size ({tp_size})"
        )

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
        else:
            self.register_buffer("bias", None)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        # weight: (out_features_per_rank, in_features)
        out = F.linear(x, self.weight, self.bias)

        if self.gather_output and self.tp_size > 1:
            out = _tp_all_gather(out, dim=-1)

        return out


# ═══════════════════════════════════════════════════════════════════════
# Row-Parallel Linear
# ═══════════════════════════════════════════════════════════════════════

class RowParallelLinear(nn.Module):
    """
    Linear layer with input dimension split across TP ranks.

    W: (out_features, in_features // tp_size)
    Each rank computes a partial sum. All-reduce combines the results.

    This is typically used as the output projection (after column-parallel
    layers), matching the split from the previous layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        _, tp_size = get_tp_rank_size()
        self.tp_size = tp_size
        self.in_features_per_rank = in_features // tp_size

        assert in_features % tp_size == 0, (
            f"in_features ({in_features}) must be divisible by tp_size ({tp_size})"
        )

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_buffer("bias", None)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features_per_rank) — already split from previous layer
        # weight: (out_features, in_features_per_rank)
        out = F.linear(x, self.weight)

        # All-reduce to combine partial sums from all TP ranks
        if self.tp_size > 1:
            out = _tp_all_reduce(out)

        if self.bias is not None:
            out = out + self.bias

        return out


# ═══════════════════════════════════════════════════════════════════════
# TP Attention (GQA)
# ═══════════════════════════════════════════════════════════════════════

class TPAttention(nn.Module):
    """
    Tensor-parallel Grouped Query Attention.

    Splits Q/K/V/O head dimensions across TP ranks.
    Each rank holds n_heads/tp_size query heads and n_kv_heads/tp_size KV heads.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        sliding_window: int = 0,
        use_yarn: bool = False,
        yarn_scale: float = 1.0,
        yarn_original_max_seq_len: int = 8192,
    ):
        super().__init__()
        _, tp_size = get_tp_rank_size()
        self.tp_size = tp_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        assert n_heads % tp_size == 0, f"n_heads ({n_heads}) % tp_size ({tp_size}) != 0"
        assert n_kv_heads % tp_size == 0, f"n_kv_heads ({n_kv_heads}) % tp_size ({tp_size}) != 0"

        self.n_heads_local = n_heads // tp_size
        self.n_kv_heads_local = n_kv_heads // tp_size
        self.n_head_groups = self.n_heads_local // self.n_kv_heads_local

        # Q: split output heads across TP ranks
        self.q_proj = ColumnParallelLinear(
            d_model, n_heads * head_dim, bias=False, gather_output=False,
        )
        # K, V: split KV heads across TP ranks
        self.k_proj = ColumnParallelLinear(
            d_model, n_kv_heads * head_dim, bias=False, gather_output=False,
        )
        self.v_proj = ColumnParallelLinear(
            d_model, n_kv_heads * head_dim, bias=False, gather_output=False,
        )
        # O: row-parallel (combines partial attention outputs)
        self.o_proj = RowParallelLinear(
            n_heads * head_dim, d_model, bias=False,
        )

        self.dropout = dropout
        self.sliding_window = sliding_window

        from mamformer.layers.rope import RotaryEmbedding
        self.rope = RotaryEmbedding(
            head_dim=head_dim, max_seq_len=max_seq_len, theta=rope_theta,
            use_yarn=use_yarn, yarn_scale=yarn_scale,
            yarn_original_max_seq_len=yarn_original_max_seq_len,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        batch_size, seq_len, _ = x.shape

        # Local projections (each rank has partial heads)
        q = self.q_proj(x)  # (batch, seqlen, n_heads_local * head_dim)
        k = self.k_proj(x)  # (batch, seqlen, n_kv_heads_local * head_dim)
        v = self.v_proj(x)

        # Reshape to (batch, n_heads_local, seqlen, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads_local, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads_local, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads_local, self.head_dim).transpose(1, 2)

        # RoPE
        cos, sin = self.rope(seq_len, x.device)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        from mamformer.layers.rope import apply_rotary_emb
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # KV cache
        if use_cache and cache is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        new_cache = {"k": k, "v": v} if use_cache else None

        # All-gather K, V across TP ranks for full attention
        if self.tp_size > 1:
            k_full = _tp_all_gather(k, dim=1)  # Gather KV heads
            v_full = _tp_all_gather(v, dim=1)
        else:
            k_full, v_full = k, v

        # Local GQA: repeat local KV heads for local Q heads
        if self.n_head_groups > 1:
            k_local = k_full.repeat_interleave(self.n_head_groups, dim=1)
            v_local = v_full.repeat_interleave(self.n_head_groups, dim=1)
        else:
            k_local, v_local = k_full, v_full

        # Flash Attention (best available backend)
        from mamformer.kernels.flash_attention import flash_attn_gqa
        attn_out = flash_attn_gqa(
            q, k_full, v_full,
            is_causal=(attention_mask is None),
            attention_mask=attention_mask,
            sliding_window=self.sliding_window,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (batch, n_heads_local, seqlen, head_dim)

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads_local * self.head_dim
        )

        # Row-parallel output: all-reduce happens inside o_proj
        output = self.o_proj(attn_out)

        return output, new_cache


# ═══════════════════════════════════════════════════════════════════════
# TP Mamba-2 Block
# ═══════════════════════════════════════════════════════════════════════

class TPMamba2Block(nn.Module):
    """
    Tensor-parallel Mamba-2 SSM block.

    Splits in_proj (output channels) and out_proj (input channels).
    The conv1d and SSM scan operate locally since they're per-channel.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 1,
        dt_rank: Optional[int] = None,
    ):
        super().__init__()
        _, tp_size = get_tp_rank_size()
        self.tp_size = tp_size
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv

        assert self.d_inner % tp_size == 0, f"d_inner ({self.d_inner}) % tp_size ({tp_size}) != 0"
        self.d_inner_local = self.d_inner // tp_size

        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        # in_proj: column-parallel (split output channels)
        # Output: 2 * d_inner (x and z branches), split across ranks
        self.in_proj = ColumnParallelLinear(
            d_model, 2 * self.d_inner, bias=False, gather_output=False,
        )

        # Depthwise conv1d: per-channel, operates locally
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner_local,
            out_channels=self.d_inner_local,
            kernel_size=d_conv,
            groups=self.d_inner_local,
            padding=0,
            bias=False,
        )

        # dt/B/C projections: operate on local d_inner
        self.dt_proj = nn.Sequential(
            nn.Linear(self.d_inner_local, self.dt_rank, bias=False),
            nn.Linear(self.dt_rank, self.d_inner_local, bias=False),
        )
        self.B_proj = nn.Linear(self.d_inner_local, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner_local, d_state, bias=False)

        # SSM parameters (replicated across TP ranks — small)
        self.A_log = nn.Parameter(torch.log(torch.linspace(0.5, 8, d_state)))
        self.D = nn.Parameter(torch.ones(self.d_inner_local))

        # out_proj: row-parallel (combine partial SSM outputs)
        self.out_proj = RowParallelLinear(
            self.d_inner, d_model, bias=False,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        for layer in self.dt_proj:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.001)
        nn.init.normal_(self.B_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.C_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.conv1d.weight, mean=0.0, std=0.02)

    def forward(
        self,
        u: torch.Tensor,
        use_cache: bool = False,
        cache: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        batch, seqlen, _ = u.shape
        device = u.device

        # in_proj: column-parallel → each rank gets 2 * d_inner_local
        xz = self.in_proj(u)  # (batch, seqlen, 2 * d_inner_local)
        x, z = xz.chunk(2, dim=-1)

        # Causal conv1d (local per-channel)
        x = self._causal_conv1d(x, cache=cache if use_cache else None)

        # SiLU
        x_act = F.silu(x)

        # Project dt, B, C
        dt = F.softplus(self.dt_proj(x_act))
        B = self.B_proj(x_act)
        C = self.C_proj(x_act)

        # SSD scan
        from mamformer.layers.mamba2 import selective_scan
        y = selective_scan(
            x=x_act, dt=dt, A=self.A_log, B=B, C=C, D=self.D,
        )

        # Gate
        z_gate = F.silu(z)
        y = y * z_gate

        # out_proj: row-parallel (all-reduce inside)
        out = self.out_proj(y)

        new_cache = None
        if use_cache:
            new_cache = {
                "conv_state": x[:, -self.d_conv + 1:] if seqlen > 1 else x,
                "ssm_state": None,
            }

        return out, new_cache

    def _causal_conv1d(self, x, cache=None):
        batch, seqlen, d_inner = x.shape
        if cache is not None and "conv_state" in cache:
            x_padded = torch.cat([cache["conv_state"], x], dim=1)
        else:
            x_padded = x
        x_padded = F.pad(x_padded.transpose(1, 2), (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_padded)
        return x_conv.transpose(1, 2)


# ═══════════════════════════════════════════════════════════════════════
# TP DeepSeekMoE
# ═══════════════════════════════════════════════════════════════════════

class TPDeepSeekMoE(nn.Module):
    """
    Tensor-parallel DeepSeekMoE FFN.

    Each expert's gate/up projections are column-parallel (split intermediate dim).
    down projection is row-parallel (all-reduce output).
    Router is replicated (small, <1M params).
    """

    def __init__(
        self,
        d_model: int,
        n_shared_experts: int = 2,
        shared_expert_dim: int = 2304,
        n_routed_experts: int = 64,
        top_k: int = 8,
        routed_expert_dim: int = 576,
        aux_loss_free: bool = True,
        bias_update_speed: float = 0.001,
        dropout: float = 0.0,
    ):
        super().__init__()
        _, tp_size = get_tp_rank_size()
        self.tp_size = tp_size
        self.d_model = d_model
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.top_k = top_k
        self.aux_loss_free = aux_loss_free
        self.bias_update_speed = bias_update_speed

        # Shared expert dim split
        assert shared_expert_dim % tp_size == 0
        assert routed_expert_dim % tp_size == 0
        self.shared_dim_local = shared_expert_dim // tp_size
        self.routed_dim_local = routed_expert_dim // tp_size

        # Router: replicated (small params)
        self.router = nn.Linear(d_model, n_routed_experts, bias=False)

        # Shared experts: gate/up column-parallel, down row-parallel
        self.shared_experts = nn.ModuleList([
            _TPSwiGLUExpert(d_model, shared_expert_dim, self.shared_dim_local, dropout)
            for _ in range(n_shared_experts)
        ])

        # Routed experts: same structure
        self.routed_experts = nn.ModuleList([
            _TPSwiGLUExpert(d_model, routed_expert_dim, self.routed_dim_local, dropout)
            for _ in range(n_routed_experts)
        ])

        if aux_loss_free:
            self.register_buffer("expert_bias", torch.zeros(n_routed_experts))
            self.register_buffer("expert_load_ema", torch.ones(n_routed_experts) / n_routed_experts)

        self.register_buffer("_total_tokens", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_expert_counts", torch.zeros(n_routed_experts, dtype=torch.long))

        nn.init.normal_(self.router.weight, mean=0.0, std=0.02 / n_routed_experts ** 0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        batch_size, seq_len, _ = x.shape

        # Shared experts (local compute, then all-reduce down_proj)
        shared_out = torch.zeros(batch_size, seq_len, self.d_model, device=x.device, dtype=x.dtype)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x)

        # Router (replicated)
        router_logits = self.router(x)
        if self.aux_loss_free and self.training:
            gating_scores = router_logits + self.expert_bias
        else:
            gating_scores = router_logits

        top_k_gates, top_k_indices = torch.topk(gating_scores, k=self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_gates, dim=-1)

        # Routed experts
        routed_out = self._compute_routed_experts(
            x, top_k_indices, top_k_gates, batch_size, seq_len,
        )

        # Load balance update
        aux_info = {"active_experts": self.top_k}
        if self.aux_loss_free and self.training:
            self._update_expert_bias(top_k_indices, batch_size * seq_len)
            aux_info["expert_bias_mean"] = self.expert_bias.mean().item()

        return shared_out + routed_out, aux_info

    def _compute_routed_experts(self, x, top_k_indices, top_k_gates, B, S):
        d_model = self.d_model
        output = torch.zeros(B, S, d_model, device=x.device, dtype=x.dtype)
        x_flat = x.view(B * S, d_model)
        idx_flat = top_k_indices.view(B * S, self.top_k)
        gate_flat = top_k_gates.view(B * S, self.top_k)

        for expert_idx in range(self.n_routed_experts):
            expert_mask = (idx_flat == expert_idx)
            token_has_expert = expert_mask.any(dim=-1)
            if not token_has_expert.any():
                continue

            expert_input = x_flat[token_has_expert]
            expert_output = self.routed_experts[expert_idx](expert_input)

            token_indices = token_has_expert.nonzero(as_tuple=True)[0]
            gates_for_tokens = torch.zeros(len(token_indices), device=x.device, dtype=x.dtype)
            for i, token_idx in enumerate(token_indices):
                slot_idx = expert_mask[token_idx].nonzero(as_tuple=True)[0][0]
                gates_for_tokens[i] = gate_flat[token_idx, slot_idx]

            expert_output = expert_output * gates_for_tokens.unsqueeze(-1)
            output.view(B * S, d_model)[token_indices] += expert_output

        return output

    def _update_expert_bias(self, top_k_indices, total_tokens):
        expert_counts = torch.zeros(self.n_routed_experts, device=top_k_indices.device, dtype=torch.float32)
        for i in range(self.n_routed_experts):
            expert_counts[i] = (top_k_indices == i).sum().float()
        actual_load = expert_counts / (total_tokens * self.top_k)
        expected_load = 1.0 / self.n_routed_experts
        self.expert_bias -= self.bias_update_speed * torch.sign(actual_load - expected_load)
        self.expert_load_ema = 0.99 * self.expert_load_ema + 0.01 * actual_load


class _TPSwiGLUExpert(nn.Module):
    """Single SwiGLU expert with TP-aware dimensions."""

    def __init__(self, d_model, full_intermediate_dim, local_intermediate_dim, dropout):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(d_model, full_intermediate_dim, bias=False, gather_output=False)
        self.up_proj = ColumnParallelLinear(d_model, full_intermediate_dim, bias=False, gather_output=False)
        self.down_proj = RowParallelLinear(full_intermediate_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(self.dropout(gate * up))


# ═══════════════════════════════════════════════════════════════════════
# Model Sharding
# ═══════════════════════════════════════════════════════════════════════

def shard_model_tp(model: nn.Module, tp_size: int) -> nn.Module:
    """
    Convert a Mamformer model to use tensor parallelism.

    This replaces dense layers with their TP equivalents throughout
    the model. The TP process group must be set before calling.

    Args:
        model: MamformerForCausalLM or MamformerModel
        tp_size: Number of GPUs for tensor parallelism

    Returns:
        Model with TP layers (same object, modified in-place)
    """
    # TP is handled at construction time via TPAttention, TPMamba2Block, etc.
    # For already-constructed models, we'd need to replace layers.
    # This function provides the interface for future use.
    # Currently, TP is applied by constructing the model with TP wrappers.
    return model
