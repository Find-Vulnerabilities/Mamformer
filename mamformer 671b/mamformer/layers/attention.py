"""
Grouped Query Attention (GQA) with RoPE
========================================
Implements multi-head attention with grouped query heads.

GQA reduces the KV cache size: instead of having one K/V per Q head,
multiple Q heads share the same K/V head. This provides most of MHA's
modeling capacity with MQA-like inference efficiency.

Reference: "GQA: Training Generalized Multi-Query Transformer Models
from Multi-Head Checkpoints" (Ainslie et al., 2023)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamformer.layers.rope import RotaryEmbedding, apply_rotary_emb


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention with RoPE and causal masking.

    Architecture:
        Q: Linear(d_model → n_heads * head_dim)
        K: Linear(d_model → n_kv_heads * head_dim)  # fewer KV heads
        V: Linear(d_model → n_kv_heads * head_dim)
        O: Linear(n_heads * head_dim → d_model)

    The K/V heads are repeated to match the number of Q heads.
    Flash Attention is used automatically via PyTorch's
    F.scaled_dot_product_attention when available.

    Args:
        d_model: Model hidden dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key/value heads (must divide n_heads)
        head_dim: Dimension per head
        max_seq_len: Maximum sequence length (for RoPE precomputation)
        rope_theta: RoPE base frequency
        dropout: Attention dropout rate
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
        sliding_window: int = 0,  # 0 = disabled, >0 = window size (like Mistral's SWA)
        use_yarn: bool = False,
        yarn_scale: float = 1.0,
        yarn_original_max_seq_len: int = 8192,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_head_groups = n_heads // n_kv_heads  # Q heads per KV head
        self.dropout = dropout
        self.sliding_window = sliding_window

        # Projections
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # RoPE (dynamic for long context, cached for short)
        self.rope = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
            use_yarn=use_yarn,
            yarn_scale=yarn_scale,
            yarn_original_max_seq_len=yarn_original_max_seq_len,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights with normal distribution."""
        std = 0.02  # Standard initialization range
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache: Optional[dict] = None,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass for GQA.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            attention_mask: Boolean mask (True = attend, False = masked)
                           Shape: (batch, 1, seq_len, seq_len) or None for causal
            use_cache: If True, return updated KV cache
            cache: Optional KV cache dict with 'k' and 'v' keys

        Returns:
            (output, cache) — output shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, n_heads * head_dim)
        k = self.k_proj(x)  # (batch, seq_len, n_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, seq_len, n_kv_heads * head_dim)

        # Reshape to (batch, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rope(seq_len, x.device)

        # Cast to match precision
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Handle KV cache
        if use_cache and cache is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)

        new_cache = {"k": k, "v": v} if use_cache else None

        # Repeat KV heads to match Q heads (GQA)
        # (batch, n_kv_heads, seq_len, head_dim) → (batch, n_heads, seq_len, head_dim)
        k = k.repeat_interleave(self.n_head_groups, dim=1)
        v = v.repeat_interleave(self.n_head_groups, dim=1)

        # Build combined mask: causal + sliding window + user-provided
        combined_mask = attention_mask
        if self.sliding_window > 0:
            sw_mask = self._build_sliding_window_mask(seq_len, q.device, q.dtype)
            if combined_mask is not None:
                combined_mask = combined_mask + sw_mask  # Combine additive masks
            else:
                combined_mask = sw_mask

        # Dispatch to attention backend
        if combined_mask is not None:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=combined_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

        # Reshape back: (batch, n_heads, seq_len, head_dim) → (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1
        )

        # Output projection
        output = self.o_proj(attn_output)

        return output, new_cache

    def _build_sliding_window_mask(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build a causal + sliding window attention mask.

        Position i can attend to positions in [max(0, i-W+1), i].
        Shape: (1, 1, seq_len, seq_len), additive mask (0=attend, -inf=mask)
        """
        # Distance matrix: distances[i,j] = i - j
        indices = torch.arange(seq_len, device=device)
        distances = indices.unsqueeze(1) - indices.unsqueeze(0)  # (seq_len, seq_len)

        # Valid: upper bound = causal (i >= j → distance >= 0)
        #         lower bound = window (i - j < sliding_window → distance < W)
        valid = (distances >= 0) & (distances < self.sliding_window)

        # Convert to additive mask: 0 = attend, -inf = mask
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        mask = mask.masked_fill(~valid, float("-inf"))

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def extra_repr(self) -> str:
        sw_info = f", sliding_window={self.sliding_window}" if self.sliding_window > 0 else ""
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, head_dim={self.head_dim}, "
            f"n_head_groups={self.n_head_groups}{sw_info}"
        )
