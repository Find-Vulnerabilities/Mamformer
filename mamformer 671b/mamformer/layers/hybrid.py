"""
Mamformer Hybrid Block (Ultra Edition)
========================================
The core innovation of the Mamformer architecture: each transformer block
combines Attention and Mamba-2 SSM in parallel, with a learnable
per-dimension gate controlling the fusion.

Mamformer Ultra supports three attention variants:
  - GQA (Grouped Query Attention) — default
  - DSA (Differential State-Aware Attention) — enhanced noise-cancelling
  - Any custom attention module

And two FFN variants:
  - SwiGLU FFN (dense) — default
  - DeepSeekMoE (sparse mixture of experts) — massive capacity upgrade

Architecture (per block):
    Input
    ├── RMSNorm
    ├── [Attention Pathway] ─┐
    │   GQA/DSA + RoPE       ├── Gate Combine ──→ Residual
    ├── [SSM Pathway] ──────┘
    │   Mamba-2 Block
    ├── RMSNorm
    ├── SwiGLU / MoE FFN ──→ Residual
    Output

The learnable gate allows the model to discover the optimal mix of
attention-based and SSM-based processing for each feature dimension.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from mamformer.layers.norm import RMSNorm
from mamformer.layers.attention import GroupedQueryAttention
from mamformer.layers.dsa import DifferentialStateAttention
from mamformer.layers.kda_diff import KDADiffAttention
from mamformer.layers.mamba2 import Mamba2Block
from mamformer.layers.ffn import SwiGLUFFN
from mamformer.layers.moe import DeepSeekMoE
from mamformer.layers.math_opt import DynamicGate, DeepNorm


class MamformerBlock(nn.Module):
    """
    A single hybrid block combining Attention and Mamba-2 SSM.

    Supports:
      - GQA or DSA for the attention pathway
      - SwiGLU FFN or DeepSeekMoE for the feed-forward pathway

    The two pathways (attention and SSM) process the same normalized
    input in parallel. Their outputs are fused via a learnable
    per-dimension sigmoid gate:

        out = σ(α) ⊙ attn(x) + (1 - σ(α)) ⊙ ssm(x)

    where α is a learnable parameter of shape (d_model,).

    After fusion, a FFN (dense SwiGLU or MoE) with residual completes the block.

    Args:
        d_model: Hidden dimension
        n_heads: Number of query heads for attention
        n_kv_heads: Number of key/value heads (GQA)
        head_dim: Dimension per attention head
        d_ff: SwiGLU intermediate dimension (used if MoE disabled)
        d_state: Mamba-2 SSM state dimension
        d_conv: Mamba-2 convolution kernel size
        mamba_expand: Mamba-2 channel expansion
        max_seq_len: Maximum sequence length for RoPE
        rope_theta: RoPE base frequency
        dropout: Dropout rate
        rms_norm_eps: Epsilon for RMSNorm
        sliding_window: Sliding window attention size (0 = disabled)

        # DSA options
        use_dsa: Use Differential State-Aware Attention instead of GQA
        dsa_lambda_init: Initial λ value for DSA
        dsa_state_injection: Inject Mamba state into DSA K/V

        # MoE options
        use_moe: Use DeepSeekMoE instead of dense SwiGLU FFN
        moe_n_shared: Number of shared experts
        moe_n_routed: Number of routed experts
        moe_top_k: Number of active routed experts per token
        moe_shared_dim: Hidden dim per shared expert
        moe_routed_dim: Hidden dim per routed expert
        moe_aux_loss_free: Use aux-loss-free load balancing
        moe_bias_speed: Bias update speed for load balancing
    """

    def __init__(
        self,
        d_model: int = 4096,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        d_ff: int = 9216,
        d_state: int = 128,
        d_conv: int = 4,
        mamba_expand: int = 1,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        sliding_window: int = 0,
        # DSA
        use_dsa: bool = False,
        dsa_lambda_init: float = 0.8,
        dsa_state_injection: bool = True,
        # RoPE / YaRN
        rope_use_yarn: bool = False,
        rope_yarn_scale: float = 1.0,
        rope_yarn_original_max_seq_len: int = 8192,
        # MoE
        use_moe: bool = False,
        moe_n_shared: int = 2,
        moe_n_routed: int = 64,
        moe_top_k: int = 8,
        moe_shared_dim: int = 2304,
        moe_routed_dim: int = 576,
        moe_aux_loss_free: bool = True,
        moe_bias_speed: float = 0.001,
        # ST-MoE (Space-Time MoE)
        use_st_moe: bool = False,
        st_moe_lambda_init: float = 0.2,
        st_moe_lambda_max: float = 0.3,
        st_moe_learnable_lambda: bool = True,
        st_moe_use_balance_lock: bool = True,
        st_moe_balance_lock_threshold: int = 50,
        # Communicative MoE
        use_communicative_moe: bool = False,
        comm_moe_n_heads: int = 4,
        comm_moe_depth: int = 1,
        comm_moe_dropout: float = 0.0,
        # KDA-Diff
        use_kda_diff: bool = False,
        kda_linear_ratio: int = 3,
        kda_kernel_dim: int = 128,
        kda_latent_dim: int = 512,
        kda_use_dynamic_ratio: bool = True,
        # DynamicGate (math_opt integration)
        use_dynamic_gate: bool = False,
        dynamic_gate_bottleneck: int = 0,
        # DeepNorm (math_opt integration)
        use_deepnorm: bool = False,
        deepnorm_n_layers: int = 52,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.sliding_window = sliding_window
        self.use_dsa = use_dsa
        self.use_kda_diff = use_kda_diff
        self.use_moe = use_moe
        self.use_st_moe = use_st_moe
        self.use_dynamic_gate = use_dynamic_gate

        # Common RoPE config for attention modules
        rope_kwargs = dict(
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            use_yarn=rope_use_yarn,
            yarn_scale=rope_yarn_scale,
            yarn_original_max_seq_len=rope_yarn_original_max_seq_len,
        )

        # Pre-attention normalization
        self.input_norm = RMSNorm(d_model, eps=rms_norm_eps)

        # Attention pathway: GQA, DSA, or KDA-Diff
        if use_kda_diff:
            self.attention = KDADiffAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                linear_ratio=kda_linear_ratio,
                kernel_dim=kda_kernel_dim,
                latent_dim=kda_latent_dim,
                lambda_init=dsa_lambda_init,
                use_state_injection=dsa_state_injection,
                state_injection_dim=64,
                use_dynamic_ratio=kda_use_dynamic_ratio,
                d_state=d_state,
                dropout=dropout,
                sliding_window=sliding_window,
                **rope_kwargs,
            )
        elif use_dsa:
            self.attention = DifferentialStateAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                lambda_init=dsa_lambda_init,
                use_state_injection=dsa_state_injection,
                dropout=dropout,
                sliding_window=sliding_window,
                **rope_kwargs,
            )
        else:
            self.attention = GroupedQueryAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                dropout=dropout,
                sliding_window=sliding_window,
                **rope_kwargs,
            )

        # SSM pathway
        self.ssm = Mamba2Block(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=mamba_expand,
        )

        # Learnable per-dimension gate
        # Initialized to 0 → sigmoid(0) = 0.5 → equal weight to both pathways
        # When use_dynamic_gate=True, replaced by a context-dependent MLP gate
        if use_dynamic_gate:
            self.gate_alpha = None
            self.dynamic_gate = DynamicGate(d_model=d_model, bottleneck=dynamic_gate_bottleneck)
        else:
            self.gate_alpha = nn.Parameter(torch.zeros(d_model))
            self.dynamic_gate = None

        # Post-fusion normalization
        self.post_norm = RMSNorm(d_model, eps=rms_norm_eps)

        # Feed-forward network: ST-MoE, MoE, or dense SwiGLU
        # Build base MoE first, then optionally wrap with CommunicativeMoE
        base_moe = None
        if use_st_moe:
            from mamformer.layers.st_moe import SpaceTimeMoE
            base_moe = SpaceTimeMoE(
                d_model=d_model,
                n_shared_experts=moe_n_shared,
                shared_expert_dim=moe_shared_dim,
                n_routed_experts=moe_n_routed,
                top_k=moe_top_k,
                routed_expert_dim=moe_routed_dim,
                d_state=d_state,
                lambda_init=st_moe_lambda_init,
                lambda_max=st_moe_lambda_max,
                learnable_lambda=st_moe_learnable_lambda,
                use_balance_lock=st_moe_use_balance_lock,
                balance_lock_threshold=st_moe_balance_lock_threshold,
                aux_loss_free=moe_aux_loss_free,
                bias_update_speed=moe_bias_speed,
                dropout=dropout,
            )
        elif use_moe:
            base_moe = DeepSeekMoE(
                d_model=d_model,
                n_shared_experts=moe_n_shared,
                shared_expert_dim=moe_shared_dim,
                n_routed_experts=moe_n_routed,
                top_k=moe_top_k,
                routed_expert_dim=moe_routed_dim,
                aux_loss_free=moe_aux_loss_free,
                bias_update_speed=moe_bias_speed,
                dropout=dropout,
            )

        # Wrap with CommunicativeMoE if enabled
        if use_communicative_moe and base_moe is not None:
            from mamformer.layers.communicative_moe import CommunicativeMoE
            self.ffn = CommunicativeMoE(
                base_moe=base_moe,
                d_model=d_model,
                n_comm_heads=comm_moe_n_heads,
                comm_depth=comm_moe_depth,
                comm_dropout=comm_moe_dropout,
            )
        elif base_moe is not None:
            self.ffn = base_moe
        else:
            self.ffn = SwiGLUFFN(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache: Optional[dict] = None,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass for a single Mamformer block.

        Args:
            hidden_states: (batch, seqlen, d_model)
            attention_mask: Optional attention mask
            use_cache: If True, return updated cache for autoregressive generation
            cache: Optional cache dict from previous step

        Returns:
            (output, cache) — output shape (batch, seqlen, d_model)
            cache dict includes optional 'moe_aux_info' for logging
        """
        # First residual
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)

        # Parallel pathways
        attn_cache = cache.get("attn") if cache is not None else None
        ssm_cache = cache.get("ssm") if cache is not None else None

        # SSM runs first so DSA/KDA-Diff can access its state for cross-pollination
        # ST-MoE needs per-timestep h-state summaries for temporal routing
        ssm_h_states = None
        need_h_states = self.use_st_moe or self.use_dsa or self.use_kda_diff
        if need_h_states:
            ssm_result = self.ssm(
                hidden_states,
                use_cache=use_cache,
                cache=ssm_cache,
                return_h_states=True,
            )
            ssm_out, ssm_new_cache, ssm_h_states = ssm_result
        else:
            ssm_out, ssm_new_cache = self.ssm(
                hidden_states,
                use_cache=use_cache,
                cache=ssm_cache,
            )

        # DSA/KDA-Diff receives per-timestep SSM state summaries for cross-pollination
        ssm_kwargs = {}
        if (self.use_dsa or self.use_kda_diff) and ssm_h_states is not None:
            ssm_kwargs = {"h_states": ssm_h_states}

        attn_out, attn_new_cache = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            cache=attn_cache,
            **ssm_kwargs,
        )

        # Learnable gated fusion (static or dynamic)
        if self.dynamic_gate is not None:
            gate = self.dynamic_gate(hidden_states)  # (d_model,)
        else:
            gate = torch.sigmoid(self.gate_alpha)  # (d_model,)
        combined = gate * attn_out + (1.0 - gate) * ssm_out

        # First residual connection
        hidden_states = residual + combined

        # Second residual block: Norm → FFN (MoE or dense) → Residual
        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)

        moe_aux_info = None
        if self.use_st_moe:
            ffn_out, moe_aux_info = self.ffn(
                hidden_states, ssm_h_states=ssm_h_states
            )
        elif self.use_moe:
            ffn_out, moe_aux_info = self.ffn(hidden_states)
        else:
            ffn_out = self.ffn(hidden_states)

        hidden_states = residual + ffn_out

        # Build cache
        new_cache = None
        if use_cache:
            new_cache = {
                "attn": attn_new_cache,
                "ssm": ssm_new_cache,
            }
            if moe_aux_info is not None:
                new_cache["moe_aux_info"] = moe_aux_info

        return hidden_states, new_cache

    def get_gate_values(self) -> torch.Tensor:
        """
        Return the current gate values (after sigmoid) for analysis.

        Values near 1.0 = attention-dominant
        Values near 0.0 = SSM-dominant
        Values near 0.5 = balanced

        Returns:
            Tensor of shape (d_model,) with values in [0, 1]
        """
        return torch.sigmoid(self.gate_alpha).detach()

    def get_moe_load_statistics(self) -> Optional[dict]:
        """Get MoE load balancing statistics (if MoE enabled)."""
        if self.use_moe and hasattr(self.ffn, 'get_load_statistics'):
            return self.ffn.get_load_statistics()
        return None

    def extra_repr(self) -> str:
        attn_type = "KDA-Diff" if self.use_kda_diff else ("DSA" if self.use_dsa else "GQA")
        if self.use_st_moe:
            ffn_type = "ST-MoE"
        elif self.use_moe:
            ffn_type = "MoE"
        else:
            ffn_type = "SwiGLU"
        if hasattr(self, 'ffn') and hasattr(self.ffn, 'comm_layer'):
            ffn_type = f"Communicative{ffn_type}"
        n_heads = getattr(self.attention, 'n_heads', '?')
        n_kv_heads = getattr(self.attention, 'n_kv_heads', '?')
        head_dim = getattr(self.attention, 'head_dim', '?')
        return (
            f"d_model={self.d_model}, "
            f"attn={attn_type}, ffn={ffn_type}, "
            f"n_heads={n_heads}, "
            f"n_kv_heads={n_kv_heads}, "
            f"head_dim={head_dim}, "
            f"d_state={self.ssm.d_state}, "
            f"d_conv={self.ssm.d_conv}"
        )
