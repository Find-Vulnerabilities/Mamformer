"""
Mamformer: Mamba-2 + Transformer Hybrid LLM
========================================
A research platform for exploring the fusion of State Space Models
(Mamba-2) with Transformer attention mechanisms.

Architecture (7B):
  - 32 hybrid blocks, each combining GQA attention + Mamba-2 SSM
  - Learnable per-dimension gating between pathways
  - SwiGLU feed-forward with RMSNorm
  - Tied embeddings, RoPE position encoding, no bias terms
"""

from mamformer.config import MamformerConfig
from mamformer.model import MamformerModel, MamformerForCausalLM
from mamformer.tokenizer import MamformerTokenizer
from mamformer.generation import GenerationMixin
from mamformer.reflection import ReflectionModule, SelfReflectiveGenerator, add_reflection_to_model

__version__ = "0.2.0"
__all__ = [
    "MamformerConfig",
    "MamformerModel",
    "MamformerForCausalLM",
    "MamformerTokenizer",
    "GenerationMixin",
    "ReflectionModule",
    "SelfReflectiveGenerator",
    "add_reflection_to_model",
]
