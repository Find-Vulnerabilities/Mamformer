"""
Rotary Position Embeddings (RoPE) — Dynamic Edition
=====================================================
Implements rotary position embeddings with:

1. **Dynamic on-the-fly computation**: Instead of precomputing tables for
   max_seq_len (which would be ~256MB for 1M context), cos/sin values are
   computed dynamically for the actual sequence length needed.

2. **YaRN extension** for extreme long-context extrapolation:
   Supports scaling factors up to 128x (8K → 1M tokens).
   Uses NTK-aware frequency scaling with ramp interpolation.

3. **Caching**: Recently computed (seq_len, offset) pairs are cached to
   avoid recomputation during autoregressive generation.

Reference:
  "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
  "YaRN: Efficient Context Window Extension of Large Language Models" (Peng et al., 2023)

Usage:
    rope = RotaryEmbedding(head_dim=128, max_seq_len=1048576, theta=50000000.0)
    cos, sin = rope(seq_len=4096, device=cuda_device)  # Dynamic compute
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Dynamic Rotary Position Embedding with YaRN support.

    Computes RoPE cos/sin tables on-the-fly rather than precomputing for
    the full max_seq_len. This is critical for 1M context support.

    Args:
        head_dim: Dimension per attention head
        max_seq_len: Maximum sequence length (for frequency initialization)
        theta: Base frequency (default: 10000.0)
               - 10000.0: Standard Llama-style
               - 1000000.0: Extended context (Mistral)
               - 50000000.0: 1M context with YaRN
        use_yarn: Enable YaNR extension (default: False)
        yarn_scale: YaRN scaling factor. 1.0 = no scaling, 128.0 = 8K→1M
        yarn_original_max_seq_len: Original pre-training max seq len (for YaRN)
        yarn_beta_fast: YaRN fast beta (affects frequency ramping sharpness)
        yarn_beta_slow: YaRN slow beta (affects low-frequency behavior)
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 8192,
        theta: float = 10000.0,
        use_yarn: bool = False,
        yarn_scale: float = 1.0,
        yarn_original_max_seq_len: Optional[int] = None,
        yarn_beta_fast: int = 32,
        yarn_beta_slow: int = 1,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.use_yarn = use_yarn
        self.yarn_scale = yarn_scale
        self.yarn_original_max_seq_len = yarn_original_max_seq_len or max_seq_len
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow

        # Compute base frequencies (half head_dim, one per pair)
        # These are the "inverse frequencies" — never change after init
        freqs = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freqs", freqs, persistent=False)

        # Cache for dynamic computation (seq_len, offset) -> (cos, sin)
        self._cache: dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_max_size = 16  # Keep last 16 entries

    def _compute_freqs(
        self, seq_len: int, device: torch.device, offset: int = 0
    ) -> torch.Tensor:
        """
        Compute position-frequency angles dynamically.

        Args:
            seq_len: Number of positions needed
            device: Target device
            offset: Position offset (for cached generation)

        Returns:
            angles: (seq_len, head_dim/2) — position × frequency
        """
        inv_freqs = self.inv_freqs.to(device)

        if self.use_yarn and self.yarn_scale > 1.0:
            # Apply YaRN NTK-aware frequency scaling
            scaled_freqs = self._apply_yarn_scaling(inv_freqs)
        else:
            scaled_freqs = inv_freqs

        # Position indices
        positions = torch.arange(
            offset, offset + seq_len,
            dtype=torch.float32, device=device,
        )

        # Outer product: (seq_len, head_dim/2)
        angles = torch.outer(positions, scaled_freqs)

        return angles

    def _apply_yarn_scaling(self, freqs: torch.Tensor) -> torch.Tensor:
        """
        Apply YaRN NTK-aware frequency scaling.

        YaRN scales different frequencies by different amounts:
        - Low frequencies (important for long-range): scaled more
        - High frequencies (important for local): scaled less

        Uses the "NTK-by-parts" method with ramp interpolation
        between beta_slow and beta_fast thresholds.

        Reference: Peng et al., 2023 — Equation 3-5
        """
        scale = self.yarn_scale
        n = len(freqs)
        original_context = self.yarn_original_max_seq_len

        # Compute wavelength for each frequency: λ = 2π / freq
        # Actually: λ = 2π * theta^(2d/D) — this grows exponentially
        # The wavelength in tokens is approximately 2π / freq_i
        wavelengths = 2.0 * math.pi / freqs

        # YaRN thresholds based on wavelength ratios
        # beta_fast: wavelengths shorter than this get no scaling
        # beta_slow: wavelengths longer than this get full scaling
        # Between: linear ramp interpolation
        fast_threshold = self.yarn_beta_fast
        slow_threshold = self.yarn_beta_slow

        # Compute ramp: 0 for high freq (short wavelength), 1 for low freq (long wavelength)
        # Smooth ramp between the two thresholds
        ramp_mask = (wavelengths - slow_threshold) / max(
            1e-6, fast_threshold - slow_threshold
        )
        ramp_mask = torch.clamp(ramp_mask, 0.0, 1.0)

        # Scale factor per frequency: from 1 (no scaling) to scale (full NTK scaling)
        scale_per_freq = 1.0 + (scale - 1.0) * ramp_mask

        return freqs * scale_per_freq

    def forward(
        self, seq_len: int, device: torch.device, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin tables for the given sequence length.

        Computes dynamically — no precomputed O(seq_len) storage.
        Results are cached for reuse during generation.

        Args:
            seq_len: Number of positions needed (typically 1 during generation)
            device: Target device
            offset: Position offset (for cached generation, offset = current_pos)

        Returns:
            (cos, sin) each of shape (1, 1, seq_len, head_dim)
        """
        # Check cache
        cache_key = (seq_len, offset)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Compute dynamically
        angles = self._compute_freqs(seq_len, device, offset)  # (seq_len, head_dim/2)

        # Duplicate: (seq_len, head_dim/2) → (seq_len, head_dim)
        angles = torch.cat([angles, angles], dim=-1)

        cos = angles.cos()
        sin = angles.sin()

        # Reshape to standard format: (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Cache the result (LRU-style: remove oldest if full)
        if len(self._cache) >= self._cache_max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = (cos, sin)

        return cos, sin

    def clear_cache(self) -> None:
        """Clear the computation cache (call when switching tasks)."""
        self._cache.clear()

    def extra_repr(self) -> str:
        yarn_info = f", YaRN(scale={self.yarn_scale}, orig={self.yarn_original_max_seq_len})" if self.use_yarn else ""
        return (
            f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, "
            f"theta={self.theta}{yarn_info}"
        )


class CachedRotaryEmbedding(RotaryEmbedding):
    """
    Backward-compatible RotaryEmbedding that precomputes tables for
    the full max_seq_len. Use only for short-context models (<32K).
    For long context (128K+), use RotaryEmbedding (dynamic) instead.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 8192,
        theta: float = 10000.0,
        use_yarn: bool = False,
        yarn_scale: float = 1.0,
        yarn_original_max_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__(
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            theta=theta,
            use_yarn=use_yarn,
            yarn_scale=yarn_scale,
            yarn_original_max_seq_len=yarn_original_max_seq_len,
        )

        # Precompute for full length (backward compatible but memory-heavy)
        self._precompute_full_table()

    def _precompute_full_table(self):
        """Precompute cos/sin for max_seq_len (backward compat)."""
        angles = self._compute_freqs(self.max_seq_len, torch.device("cpu"))  # (max_seq_len, head_dim/2)
        angles = torch.cat([angles, angles], dim=-1)

        cos = angles.cos().unsqueeze(0).unsqueeze(0)
        sin = angles.sin().unsqueeze(0).unsqueeze(0)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self, seq_len: int, device: torch.device, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Slice from precomputed table (fast but memory-heavy)."""
        return (
            self.cos[:, :, offset:offset + seq_len].to(device),
            self.sin[:, :, offset:offset + seq_len].to(device),
        )

    def extra_repr(self) -> str:
        yarn_info = f", YaRN(scale={self.yarn_scale})" if self.use_yarn else ""
        return (
            f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, "
            f"theta={self.theta}, cached{yarn_info}"
        )


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to a tensor.

    This is the core RoPE operation. It rotates pairs of dimensions
    by position-dependent angles.

    Args:
        x: Input tensor of shape (batch, n_heads, seq_len, head_dim)
        cos: Cosine values, shape (1, 1, seq_len, head_dim) or (1, 1, seq_len, head_dim/2)
        sin: Sine values, shape (1, 1, seq_len, head_dim) or (1, 1, seq_len, head_dim/2)
        interleaved: If True, pairs are (0,1), (2,3), ... (GPT-NeoX style)
                     If False, pairs are (0,d/2), (1,d/2+1), ... (LLaMA style)

    Returns:
        Tensor of same shape as input with RoPE applied
    """
    head_dim = x.shape[-1]

    # Handle cos/sin being half-dimension (only head_dim/2 entries)
    if cos.shape[-1] == head_dim // 2:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

    if interleaved:
        # GPT-NeoX style: pairs are adjacent (even, odd)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos_half = cos[..., 0::2]
        sin_half = sin[..., 0::2]

        rotated_even = x_even * cos_half - x_odd * sin_half
        rotated_odd = x_odd * cos_half + x_even * sin_half

        result = torch.empty_like(x)
        result[..., 0::2] = rotated_even
        result[..., 1::2] = rotated_odd
    else:
        # LLaMA style: pairs are (first_half_i, second_half_i)
        half_dim = head_dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]

        cos_half = cos[..., :half_dim]
        sin_half = sin[..., :half_dim]

        rotated_x1 = x1 * cos_half - x2 * sin_half
        rotated_x2 = x2 * cos_half + x1 * sin_half

        result = torch.cat([rotated_x1, rotated_x2], dim=-1)

    return result.to(x.dtype)


def precompute_freqs_cis(
    head_dim: int,
    seq_len: int,
    theta: float = 10000.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Precompute frequency cis (complex representation) for RoPE.

    This is a utility for the complex-number formulation of RoPE:
    RoPE(x) = x * e^{i * pos * freq}

    Returns:
        Complex tensor of shape (seq_len, head_dim/2) with cis values
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(positions, freqs)  # (seq_len, head_dim/2)
    return torch.polar(torch.ones_like(angles), angles)  # e^{i*angle}


def apply_rotary_emb_complex(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE using complex number multiplication.

    Args:
        x: (batch, n_heads, seq_len, head_dim) — real-valued
        freqs_cis: (seq_len, head_dim/2) — complex cis values

    Returns:
        (batch, n_heads, seq_len, head_dim) with RoPE applied
    """
    # Reshape x as complex: last dim split into real/imag pairs
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )  # (..., head_dim/2) complex

    # Broadcast freqs_cis: (1, 1, seq_len, head_dim/2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)

    # Complex multiplication = rotation
    rotated = x_complex * freqs_cis

    # Back to real
    x_out = torch.view_as_real(rotated).flatten(-2).to(x.dtype)
    return x_out
