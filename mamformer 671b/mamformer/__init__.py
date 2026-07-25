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

Features:
  - Toggleable thinking mode (NoThink / FastThink / CoreThink / DeepThink)
  - Stochastic GRPO (S-GRPO) for sparse token sampling
"""

from mamformer.config import MamformerConfig
from mamformer.model import MamformerModel, MamformerForCausalLM
from mamformer.tokenizer import MamformerTokenizer
from mamformer.generation import GenerationMixin
from mamformer.reflection import ReflectionModule, SelfReflectiveGenerator, add_reflection_to_model
from mamformer.layers.communicative_moe import CommunicativeMoE
from mamformer.thinking import ThinkingConfig, ThinkingMode, MultiPathController
from mamformer.thinking_data import format_sequence_with_thinking, format_labels_with_thinking, format_batch_with_thinking
from mamformer.sgrpo import StochasticGRPOConfig, sample_token_mask

__version__ = "0.3.0"
__all__ = [
    "MamformerConfig",
    "MamformerModel",
    "MamformerForCausalLM",
    "MamformerTokenizer",
    "GenerationMixin",
    "ReflectionModule",
    "SelfReflectiveGenerator",
    "add_reflection_to_model",
    "CommunicativeMoE",
    # Thinking mode
    "ThinkingConfig",
    "ThinkingMode",
    "MultiPathController",
    # S-GRPO
    "StochasticGRPOConfig",
    "sample_token_mask",
]
