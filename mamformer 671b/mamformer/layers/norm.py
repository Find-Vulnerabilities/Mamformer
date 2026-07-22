"""
RMSNorm: Root Mean Square Layer Normalization
==============================================
Used throughout Mamformer instead of LayerNorm. More computationally
efficient while achieving similar training stability.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Computes: output = x * rsqrt(mean_square(x) + eps) * weight

    Args:
        d_model: Feature dimension
        eps: Small constant for numerical stability
        bias: If True, includes a learnable bias term (default: False)
    """

    def __init__(self, d_model: int, eps: float = 1e-6, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # Compute in float32 for numerical stability
        x_fp32 = x.float()
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps)
        normalized = (x_fp32 / rms).to(dtype)
        output = normalized * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return f"d_model={self.weight.shape[0]}, eps={self.eps}, bias={self.bias is not None}"


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization (for comparison experiments).

    Included as a reference baseline; Mamformer uses RMSNorm by default.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_fp32 = x.float()
        mean = x_fp32.mean(dim=-1, keepdim=True)
        var = x_fp32.var(dim=-1, keepdim=True, unbiased=False)
        normalized = ((x_fp32 - mean) / torch.sqrt(var + self.eps)).to(dtype)
        return normalized * self.weight + self.bias

    def extra_repr(self) -> str:
        return f"d_model={self.weight.shape[0]}, eps={self.eps}"
