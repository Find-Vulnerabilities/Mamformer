"""
SwiGLU Feed-Forward Network
============================
Implements the SwiGLU activation FFN used in modern LLMs (Llama, Mistral, etc.).

SwiGLU(x) = Swish(xW_1 + b_1) ⊙ (xW_2 + b_2)
Output = (SwiGLU_output) W_3 + b_3

Where Swish(x) = x * sigmoid(x) = SiLU(x)

Reference: "GLU Variants Improve Transformer" (Shazeer, 2020)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Standard transformer FFN with SwiGLU activation:
        gate = SiLU(Linear_gate(x))
        up   = Linear_up(x)
        down = Linear_down(gate * up)

    Uses intermediate dimension d_ff, which is typically larger than d_model.
    For SwiGLU, the effective parameter count is 3 * d_model * d_ff
    (vs 2 * d_model * d_ff for standard ReLU FFN).

    Args:
        d_model: Input/output hidden dimension
        d_ff: Intermediate dimension (SwiGLU hidden size)
        dropout: Dropout rate after activation (default: 0.0)
        bias: Whether to use bias in linear layers (default: False)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        # Three projections for SwiGLU
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)  # W_gate
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)  # W_up
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)  # W_down

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with a normal distribution."""
        std = 0.02
        for proj in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seqlen, d_model)

        Returns:
            Output tensor (batch, seqlen, d_model)
        """
        gate = F.silu(self.gate_proj(x))  # (batch, seqlen, d_ff)
        up = self.up_proj(x)  # (batch, seqlen, d_ff)

        # Element-wise gating
        gated = gate * up  # (batch, seqlen, d_ff)
        gated = self.dropout_layer(gated)

        out = self.down_proj(gated)  # (batch, seqlen, d_model)

        return out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}, dropout={self.dropout}"


class GEGLUFFN(nn.Module):
    """
    GEGLU (GELU-gated) FFN — for comparison experiments.

    GEGLU uses GELU instead of SiLU for the gating function.
    Included for research ablation studies.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(self.dropout(gate * up))

    def _init_weights(self):
        std = 0.02
        for proj in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)


class StandardFFN(nn.Module):
    """
    Standard ReLU FFN — for comparison experiments.

    FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

    def _init_weights(self):
        std = 0.02
        for proj in [self.fc1, self.fc2]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
