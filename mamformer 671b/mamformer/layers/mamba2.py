"""
Mamba-2 SSM Block (Structured State Space Duality)
====================================================
Implements the Mamba-2 selective SSM using the SSD formulation.

Mamba-2 reformulates SSMs through Structured State Space Duality:
the SSM recurrence can be expressed as a matrix multiplication with
a 1-semi-separable matrix, enabling efficient training.

Reference: "Transformers are SSMs: Generalized Models and Efficient
Algorithms Through Structured State Space Duality" (Dao & Gu, 2024)

Key algorithm (SSD selective scan):
  Input:  X (batch, seqlen, d_inner)
  State:  h_t = A_t * h_{t-1} + B_t * x_t   (recurrent)
  Output: y_t = C_t^T * h_t + D * x_t

Where A_t, B_t, C_t are input-dependent (selective).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    return_h_states: bool = False,
):
    """
    Mamba-2 selective scan (SSD kernel) — pure PyTorch sequential implementation.

    Uses the recurrent formulation for numerical stability:
        h_t = A_disc[t] * h_{t-1} + B[t] * dt[t] * x[t]
        y_t = C[t]^T * h_t + D * x[t]

    Optimized with vectorized operations over (d_inner * d_state) per timestep.

    Args:
        x: Input tensor (batch, seqlen, d_inner)
        dt: Delta values (batch, seqlen, d_inner), after softplus (positive)
        A: State matrix log parameters (d_state,), A_disc = exp(-exp(A) * dt)
        B: Input-dependent state projection (batch, seqlen, d_state)
        C: Input-dependent output projection (batch, seqlen, d_state)
        D: Skip connection weight (d_inner,)
        return_h_states: If True, also return per-timestep h-state summaries.
                         h_states[t] = mean-pool over d_inner of h at step t.
                         Shape: (batch, seqlen, d_state)

    Returns:
        y: Output tensor (batch, seqlen, d_inner)
        h_states: (only if return_h_states=True)
                  Per-timestep SSM state summaries (batch, seqlen, d_state)
    """
    batch, seqlen, d_inner = x.shape
    d_state = A.shape[0]

    # Compute A_disc for all timesteps: exp(-exp(A) * dt)
    # Shape: (batch, seqlen, d_inner, d_state)
    A_param = torch.exp(A)  # (d_state,) — exp(A_log), always positive
    dt_expanded = dt.unsqueeze(-1)  # (batch, seqlen, d_inner, 1)
    A_disc = torch.exp(-A_param.view(1, 1, 1, d_state) * dt_expanded)
    # A_disc is in (0, 1], numerically stable since A_param >= 0

    # B term: B * dt, shape (batch, seqlen, d_inner, d_state)
    B_disc = B.unsqueeze(2) * dt_expanded

    # Modulated input: B_disc * x
    Bx = B_disc * x.unsqueeze(-1)  # (batch, seqlen, d_inner, d_state)

    # Sequential scan over time — numerically stable because A_disc <= 1
    # Vectorized over (batch, d_inner, d_state)
    h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
    y = torch.zeros(batch, seqlen, d_inner, device=x.device, dtype=x.dtype)

    # Optional: collect per-timestep h summaries for Space-Time MoE
    h_states = None
    if return_h_states:
        h_states = torch.zeros(batch, seqlen, d_state, device=x.device, dtype=x.dtype)

    for t in range(seqlen):
        # h_t = A_disc[t] * h_{t-1} + Bx[t]
        h = A_disc[:, t] * h + Bx[:, t]  # (batch, d_inner, d_state)
        # y_t = C[t]^T * h_t (dot product over d_state)
        y[:, t] = torch.sum(C[:, t].unsqueeze(1) * h, dim=-1)  # (batch, d_inner)
        # Collect h-state summary: mean-pool over d_inner channels
        if return_h_states:
            h_states[:, t] = h.mean(dim=1)  # (batch, d_state)

    # Add skip connection
    y = y + D.view(1, 1, d_inner) * x

    if return_h_states:
        return y, h_states
    return y


def selective_scan_sequential(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    return_h_states: bool = False,
):
    """
    Sequential selective scan (for testing/validation of the SSD kernel).
    Numerically equivalent to selective_scan() — O(seqlen), verifies correctness.

    Returns:
        y: (batch, seqlen, d_inner)
        h_states: (batch, seqlen, d_state) if return_h_states=True
    """
    batch, seqlen, d_inner = x.shape
    d_state = A.shape[0]

    A_bc = A.view(1, 1, 1, d_state)
    dt_expanded = dt.unsqueeze(-1)
    A_disc = torch.exp(-torch.exp(A_bc) * dt_expanded)

    B_expanded = B.unsqueeze(2)
    B_disc = B_expanded * dt_expanded

    h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
    y = torch.zeros(batch, seqlen, d_inner, device=x.device, dtype=x.dtype)
    h_states = torch.zeros(batch, seqlen, d_state, device=x.device, dtype=x.dtype) if return_h_states else None

    for t in range(seqlen):
        A_t = A_disc[:, t]
        Bx_t = B_disc[:, t] * x[:, t].unsqueeze(-1)
        C_t = C[:, t].unsqueeze(1)

        h = A_t * h + Bx_t
        y[:, t] = (C_t * h).sum(dim=-1)
        if return_h_states:
            h_states[:, t] = h.mean(dim=1)

    D_expanded = D.view(1, 1, d_inner)
    y = y + D_expanded * x

    if return_h_states:
        return y, h_states
    return y


class Mamba2Block(nn.Module):
    """
    Mamba-2 SSM block for use within the hybrid Mamformer architecture.

    Structure:
        u → in_proj → [x, z]
        x → causal_conv1d → SiLU → dt/B/C projections → SSD scan → out_proj
        z → SiLU → (gate) → ⊙
        Output: gate * out_proj(SSD_output)

    Args:
        d_model: Input/output dimension (also d_inner when expand=1)
        d_state: SSM state dimension
        d_conv: 1D convolution kernel size
        expand: Channel expansion factor (default: 1)
        dt_rank: Rank of delta projection
        bias: Whether to use bias in linear layers (default: False)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 1,
        dt_rank: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv

        # dt_rank: by default ceil(d_model / 16)
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        # Input projection: d_model → 2 * d_inner (x and z branches)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # 1D causal convolution (operates on x branch)
        # conv1d weight: (d_inner, 1, d_conv) — groups=d_inner for depthwise conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,  # Depthwise: each channel convolved independently
            padding=0,  # We pad manually for causal
            bias=bias,
        )

        # Delta projection: d_inner → dt_rank → d_inner
        self.dt_proj = nn.Sequential(
            nn.Linear(self.d_inner, self.dt_rank, bias=bias),
            nn.Linear(self.dt_rank, self.d_inner, bias=bias),
        )

        # B and C projections: d_inner → d_state (each)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=bias)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=bias)

        # Learnable SSM parameters
        # A_log: log of diagonal state matrix values (d_state,)
        self.A_log = nn.Parameter(
            torch.log(torch.linspace(0.5, 8, d_state))  # Range [0.5, 8]
        )

        # D: skip connection weight (d_inner,)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection: d_inner → d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with appropriate schemes."""
        # Input/output projections: small normal
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

        # dt projection: smaller init (first layer of 2-layer projection)
        for layer in self.dt_proj:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.001)

        # B and C projections
        nn.init.normal_(self.B_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.C_proj.weight, mean=0.0, std=0.02)

        # conv1d: identity-like init
        nn.init.normal_(self.conv1d.weight, mean=0.0, std=0.02)

    def forward(
        self,
        u: torch.Tensor,
        use_cache: bool = False,
        cache: Optional[dict] = None,
        return_h_states: bool = False,
    ):
        """
        Forward pass for the Mamba-2 block.

        Args:
            u: Input tensor (batch, seqlen, d_model)
            use_cache: If True, returns SSM state cache
            cache: Optional dict with 'conv_state' and 'ssm_state' for recurrent inference
            return_h_states: If True, also return per-timestep SSM state summaries
                             for Space-Time MoE routing. Shape: (batch, seqlen, d_state)

        Returns:
            (output, cache) — output shape (batch, seqlen, d_model)
            (output, cache, h_states) — if return_h_states=True
        """
        batch, seqlen, _ = u.shape

        # Input projection: split into x (SSM) and z (gate)
        xz = self.in_proj(u)  # (batch, seqlen, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # each (batch, seqlen, d_inner)

        # Causal 1D convolution
        x = self._causal_conv1d(x, cache=cache if use_cache else None)

        # SiLU activation
        x_act = F.silu(x)

        # Project dt, B, C
        dt = F.softplus(self.dt_proj(x_act))  # (batch, seqlen, d_inner), positive
        B = self.B_proj(x_act)  # (batch, seqlen, d_state)
        C = self.C_proj(x_act)  # (batch, seqlen, d_state)

        # Selective scan (SSD kernel) with proper state caching
        h_states = None
        new_ssm_state = None
        if use_cache and cache is not None and cache.get("ssm_state") is not None:
            y, new_ssm_state = self._recurrent_step(x_act, dt, B, C, cache["ssm_state"])
        else:
            scan_result = selective_scan(
                x=x_act, dt=dt, A=self.A_log, B=B, C=C, D=self.D,
                return_h_states=return_h_states,
            )
            if return_h_states:
                y, h_states = scan_result
            else:
                y = scan_result

        # Gate: z after SiLU
        z_gate = F.silu(z)
        y = y * z_gate

        # Output projection
        out = self.out_proj(y)

        # Build cache with correct conv_state and ssm_state preservation
        new_cache = None
        if use_cache:
            # Compute SSM state at the last position for next step
            # Always compute it, regardless of seqlen
            dt_last = dt[:, -1:]  # (batch, 1, d_inner)
            B_last = B[:, -1:]    # (batch, 1, d_state)
            x_last = x_act[:, -1:]  # (batch, 1, d_inner)
            A_param = torch.exp(self.A_log)
            A_disc_last = torch.exp(-A_param.view(1, 1, 1, self.d_state) * dt_last.unsqueeze(-1))
            B_disc_last = B_last.unsqueeze(2) * dt_last.unsqueeze(-1)
            new_ssm_state = (B_disc_last * x_last.unsqueeze(-1))[:, 0]  # (batch, d_inner, d_state)

            # Compute conv state (last d_conv-1 positions of x before activation)
            conv_x = x[:, -(self.d_conv - 1):] if x.shape[1] >= self.d_conv else x
            if cache is not None and "conv_state" in cache:
                # Extend existing conv state for single-token inference
                conv_x = torch.cat([cache["conv_state"], conv_x], dim=1)[:, -(self.d_conv - 1):]
            new_conv_state = conv_x

            new_cache = {
                "conv_state": new_conv_state,
                "ssm_state": new_ssm_state,
            }

        if return_h_states:
            return out, new_cache, h_states
        return out, new_cache

    def _causal_conv1d(
        self,
        x: torch.Tensor,
        cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Apply causal 1D convolution.

        For causal convolution, we need to:
        1. Pad the left side with (kernel_size - 1) zeros
        2. Apply standard conv1d
        3. Remove the last (kernel_size - 1) outputs

        This ensures position t only sees positions <= t.
        """
        batch, seqlen, d_inner = x.shape

        if cache is not None and "conv_state" in cache:
            # Incremental: concatenate cached state and current input
            x_padded = torch.cat([cache["conv_state"], x], dim=1)
        else:
            x_padded = x

        # Pad left: (batch, seqlen + d_conv - 1, d_inner)
        x_padded = F.pad(x_padded.transpose(1, 2), (self.d_conv - 1, 0))
        # Shape: (batch, d_inner, seqlen + d_conv - 1)

        # Apply conv1d
        x_conv = self.conv1d(x_padded)  # (batch, d_inner, seqlen)

        # When using cache, conv1d outputs more positions than we need
        # Take only the last `seqlen` outputs (matching the original input length)
        if cache is not None and "conv_state" in cache:
            x_conv = x_conv[:, :, -seqlen:]

        return x_conv.transpose(1, 2).contiguous()  # (batch, seqlen, d_inner)

    def _recurrent_step(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        h_prev: Optional[torch.Tensor],
    ):
        """
        Single recurrent step (for autoregressive inference).
        Only processes the last token.

        Args:
            x: Input (batch, seqlen, d_inner) — only last position used
            dt: Delta (batch, seqlen, d_inner)
            B: State projection (batch, seqlen, d_state)
            C: Output projection (batch, seqlen, d_state)
            h_prev: Previous state (batch, d_inner, d_state) or None for first step

        Returns:
            (y_t, h_t) tuple:
              - y_t: Output (batch, 1, d_inner)
              - h_t: Updated hidden state (batch, d_inner, d_state)
        """
        batch, seqlen, d_inner = x.shape
        d_state = self.d_state

        # Take last token
        x_t = x[:, -1]  # (batch, d_inner)
        dt_t = dt[:, -1]  # (batch, d_inner)
        B_t = B[:, -1]  # (batch, d_state)
        C_t = C[:, -1]  # (batch, d_state)

        # Discretize
        A_disc = torch.exp(
            -torch.exp(self.A_log).unsqueeze(0) * dt_t.unsqueeze(-1)
        )  # (batch, d_inner, d_state)

        B_disc = B_t.unsqueeze(1) * dt_t.unsqueeze(-1)  # (batch, d_inner, d_state)

        # State update — init zero state on first step
        if h_prev is None:
            h_prev = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        h_t = A_disc * h_prev + B_disc * x_t.unsqueeze(-1)  # (batch, d_inner, d_state)

        # Output
        y_t = (C_t.unsqueeze(1) * h_t).sum(dim=-1)  # (batch, d_inner)

        # Skip connection
        y_t = y_t + self.D * x_t

        return y_t.unsqueeze(1), h_t  # (batch, 1, d_inner), (batch, d_inner, d_state)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_inner={self.d_inner}, "
            f"d_state={self.d_state}, d_conv={self.d_conv}"
        )
