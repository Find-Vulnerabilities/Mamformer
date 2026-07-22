"""
Mamformer High-Performance Kernels
===================================
Optimized CUDA/Triton kernels for Mamformer training and inference.

- triton_ssd: Fused selective scan kernel for Mamba-2 (10-50x speedup)
- flash_attention: Flash Attention 2 integration for GQA/DSA
"""

from mamformer.kernels.triton_ssd import (
    triton_ssd_scan,
    triton_selective_scan_fused,
    is_triton_available,
)
from mamformer.kernels.flash_attention import (
    flash_attn_gqa,
    flash_attn_dsa,
    flash_attn_with_sliding_window,
    is_flash_attn_available,
)

__all__ = [
    "triton_ssd_scan",
    "triton_selective_scan_fused",
    "is_triton_available",
    "flash_attn_gqa",
    "flash_attn_dsa",
    "flash_attn_with_sliding_window",
    "is_flash_attn_available",
]
