"""
Mamformer Layers Package
========================
Core building blocks for the Mamformer hybrid LLM architecture.

Standard:
  - RMSNorm: Root Mean Square Layer Normalization
  - GroupedQueryAttention: GQA with RoPE and optional sliding window
  - Mamba2Block: Mamba-2 SSM block (SSD formulation)
  - SwiGLUFFN: SwiGLU feed-forward network
  - MamformerBlock: Hybrid block (GQA + Mamba-2 + FFN)

Ultra (Mamformer Ultra architecture — MoE + DSA + MTP):
  - DeepSeekMoE: Mixture of Experts FFN with shared + routed experts
  - DifferentialStateAttention: Noise-cancelling attention with SSM state injection
  - MultiTokenPredictor: Multi-token prediction for denser training signal
"""

from mamformer.layers.norm import RMSNorm, LayerNorm
from mamformer.layers.rope import RotaryEmbedding, apply_rotary_emb
from mamformer.layers.attention import GroupedQueryAttention
from mamformer.layers.mamba2 import Mamba2Block, selective_scan
from mamformer.layers.ffn import SwiGLUFFN, GEGLUFFN, StandardFFN
from mamformer.layers.hybrid import MamformerBlock
from mamformer.layers.moe import DeepSeekMoE
from mamformer.layers.dsa import DifferentialStateAttention
from mamformer.layers.mtp import MultiTokenPredictor
from mamformer.layers.math_opt import (
    DynamicGate, EntropyRouter, AdaptiveLambda, DeepNorm, compute_moe_entropy_loss,
)

__all__ = [
    # Norms
    "RMSNorm",
    "LayerNorm",
    # RoPE
    "RotaryEmbedding",
    "apply_rotary_emb",
    # Attention
    "GroupedQueryAttention",
    "DifferentialStateAttention",
    # SSM
    "Mamba2Block",
    "selective_scan",
    # FFN
    "SwiGLUFFN",
    "GEGLUFFN",
    "StandardFFN",
    "DeepSeekMoE",
    # Hybrid block
    "MamformerBlock",
    # MTP
    "MultiTokenPredictor",
    # Math optimizations
    "DynamicGate",
    "EntropyRouter",
    "AdaptiveLambda",
    "DeepNorm",
    "compute_moe_entropy_loss",
]
