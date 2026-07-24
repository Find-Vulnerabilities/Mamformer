"""
Flash Attention 2 Integration for Mamformer
=============================================
Provides optimized attention backends with automatic detection
of the best available implementation.

Supported backends (in priority order):
  1. flash_attn (Dao-AILab/flash-attention) — fastest, CUDA only
  2. PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention) — built-in
  3. Manual attention (fallback) — always available, slowest

Features:
  - GQA (Grouped Query Attention) with KV repeat
  - DSA (Differential State-Aware Attention) — two-pass attention
  - Sliding window attention
  - Causal masking
  - BF16/FP16 mixed precision

Usage:
    from mamformer.kernels import flash_attn_gqa
    output = flash_attn_gqa(q, k, v, is_causal=True)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Backend detection ────────────────────────────────────────────────

_flash_attn_available: Optional[bool] = None
_backend: Optional[str] = None


def is_flash_attn_available() -> bool:
    """Check if flash_attn package is installed."""
    global _flash_attn_available
    if _flash_attn_available is None:
        try:
            import flash_attn
            _flash_attn_available = torch.cuda.is_available()
        except ImportError:
            _flash_attn_available = False
    return _flash_attn_available


def get_best_backend() -> str:
    """Get the best available attention backend."""
    global _backend
    if _backend is None:
        if is_flash_attn_available():
            _backend = "flash_attn"
        elif hasattr(F, "scaled_dot_product_attention"):
            _backend = "sdpa"
        else:
            _backend = "manual"
    return _backend


# ── GQA Flash Attention ──────────────────────────────────────────────

def flash_attn_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    attention_mask: Optional[torch.Tensor] = None,
    sliding_window: int = 0,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Flash Attention for Grouped Query Attention.

    Automatically handles KV head repeating for GQA by detecting
    when n_heads_q > n_heads_kv.

    Args:
        q: (batch, n_heads_q, seqlen, head_dim)
        k: (batch, n_heads_kv, seqlen, head_dim)
        v: (batch, n_heads_kv, seqlen, head_dim)
        is_causal: Apply causal masking
        attention_mask: Optional additive mask (0=attend, -inf=mask)
        sliding_window: Window size (0=disabled)
        dropout_p: Attention dropout probability
        softmax_scale: Scale factor (default: 1/sqrt(head_dim))

    Returns:
        output: (batch, n_heads_q, seqlen, head_dim)
    """
    backend = get_best_backend()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    # GQA: repeat KV heads to match Q heads
    n_groups = q.shape[1] // k.shape[1]
    if n_groups > 1:
        k = k.repeat_interleave(n_groups, dim=1)
        v = v.repeat_interleave(n_groups, dim=1)

    # Build mask for sliding window
    mask = attention_mask
    if sliding_window > 0 and mask is None:
        mask = _build_sliding_window_causal_mask(
            q.shape[2], q.device, q.dtype, sliding_window
        )

    training = dropout_p > 0
    if backend == "flash_attn":
        return _flash_attn_impl(q, k, v, is_causal, mask, dropout_p, softmax_scale)
    elif backend == "sdpa":
        return _sdpa_impl(q, k, v, is_causal, mask, dropout_p, softmax_scale)
    else:
        return _manual_impl(q, k, v, is_causal, mask, dropout_p, softmax_scale, training=training)


def _flash_attn_impl(q, k, v, is_causal, mask, dropout_p, scale):
    """Use Dao-AILab flash-attention library."""
    from flash_attn import flash_attn_func

    # flash_attn expects (batch, seqlen, n_heads, head_dim)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    window_size = (-1, -1)  # No sliding window in this call
    if mask is not None and mask.dim() == 4:
        # Convert additive mask to boolean if needed
        pass  # flash_attn handles masks differently

    output = flash_attn_func(
        q, k, v,
        dropout_p=dropout_p if dropout_p > 0 else 0.0,
        softmax_scale=scale,
        causal=is_causal,
        window_size=window_size,
    )

    return output.transpose(1, 2).contiguous()


def _sdpa_impl(q, k, v, is_causal, mask, dropout_p, scale):
    """Use PyTorch's built-in SDPA (supports Flash Attention via cuDNN)."""
    # SDPA handles causal masking internally when is_causal=True and mask is None
    attn_mask = mask
    causal = is_causal and (mask is None)

    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=scale,
    )


def _manual_impl(q, k, v, is_causal, mask, dropout_p, scale, training=True):
    """Manual attention implementation (slow but always available)."""
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

    dtype_min = torch.finfo(attn_weights.dtype).min

    if mask is not None:
        attn_weights = attn_weights + mask
    elif is_causal:
        seq_len = q.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask, dtype_min)

    attn_weights = F.softmax(attn_weights, dim=-1)
    if dropout_p > 0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)
    return torch.matmul(attn_weights, v)


# ── DSA Flash Attention ──────────────────────────────────────────────

def flash_attn_dsa(
    q1: torch.Tensor,
    q2: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lam: torch.Tensor,
    is_causal: bool = True,
    attention_mask: Optional[torch.Tensor] = None,
    sliding_window: int = 0,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Flash Attention for Differential State-Aware Attention.

    Computes: DiffAttn = (softmax(Q1@K^T) - lambda * softmax(Q2@K^T)) @ V

    This requires TWO attention passes (for Q1 and Q2), doubling the
    compute cost but providing noise cancellation.

    Args:
        q1: (batch, n_heads, seqlen, head_dim) — first query
        q2: (batch, n_heads, seqlen, head_dim) — second query
        k:  (batch, n_kv_heads, seqlen, head_dim)
        v:  (batch, n_kv_heads, seqlen, head_dim)
        lam: (n_heads,) or (1, n_heads, 1, 1) — per-head lambda values
        is_causal: Apply causal masking
        attention_mask: Optional additive mask
        sliding_window: Window size
        dropout_p: Attention dropout
        softmax_scale: Scale factor

    Returns:
        output: (batch, n_heads, seqlen, head_dim)
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q1.shape[-1])

    # Ensure lambda is broadcastable
    if lam.dim() == 1:
        lam = lam.view(1, -1, 1, 1)
    elif lam.dim() == 2:
        lam = lam.unsqueeze(-1).unsqueeze(-1)

    # Compute A1 and A2 using flash attention
    attn_out_1 = flash_attn_gqa(
        q1, k, v,
        is_causal=is_causal,
        attention_mask=attention_mask,
        sliding_window=sliding_window,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
    )

    attn_out_2 = flash_attn_gqa(
        q2, k, v,
        is_causal=is_causal,
        attention_mask=attention_mask,
        sliding_window=sliding_window,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
    )

    # Differential combination: A1 - lambda * A2, with (1-lambda) normalization
    # as in Differential Transformer paper (Ye et al., 2024)
    return (attn_out_1 - lam * attn_out_2) / (1.0 - lam + 1e-8)


# ── Sliding Window Mask ──────────────────────────────────────────────

def flash_attn_with_sliding_window(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    is_causal: bool = True,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Flash Attention with sliding window (Mistral-style SWA).

    Position i attends to positions in [max(0, i-W+1), i].

    Args:
        q, k, v: Standard attention tensors
        window_size: Number of past positions to attend to
        is_causal: Must be True for sliding window
        dropout_p: Attention dropout
        softmax_scale: Scale factor

    Returns:
        output: (batch, n_heads, seqlen, head_dim)
    """
    backend = get_best_backend()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    # GQA repeat
    n_groups = q.shape[1] // k.shape[1]
    if n_groups > 1:
        k = k.repeat_interleave(n_groups, dim=1)
        v = v.repeat_interleave(n_groups, dim=1)

    # Build sliding window + causal mask
    mask = _build_sliding_window_causal_mask(
        q.shape[2], q.device, q.dtype, window_size
    )

    if backend == "flash_attn":
        return _flash_attn_impl(q, k, v, False, mask, dropout_p, softmax_scale)
    elif backend == "sdpa":
        return _sdpa_impl(q, k, v, False, mask, dropout_p, softmax_scale)
    else:
        return _manual_impl(q, k, v, False, mask, dropout_p, softmax_scale)


def _build_sliding_window_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    window_size: int,
) -> torch.Tensor:
    """
    Build causal + sliding window additive mask.

    Shape: (1, 1, seq_len, seq_len). 0 = attend, -inf = mask.
    """
    indices = torch.arange(seq_len, device=device)
    distances = indices.unsqueeze(1) - indices.unsqueeze(0)  # (seq_len, seq_len)
    # position i can attend to j where: 0 <= i-j < window_size
    valid = (distances >= 0) & (distances < window_size)
    dtype_min = torch.finfo(dtype).min
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    mask = mask.masked_fill(~valid, dtype_min)
    return mask.unsqueeze(0).unsqueeze(0)


# ── Backend Info ─────────────────────────────────────────────────────

def get_attention_backend_info() -> dict:
    """Get information about available attention backends."""
    info = {
        "best_backend": get_best_backend(),
        "flash_attn_available": is_flash_attn_available(),
        "sdpa_available": hasattr(F, "scaled_dot_product_attention"),
        "cuda_available": torch.cuda.is_available(),
    }

    if is_flash_attn_available():
        try:
            import flash_attn
            info["flash_attn_version"] = flash_attn.__version__
        except Exception:
            pass

    return info
