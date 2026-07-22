"""
Differential State-Aware Attention (DSA)
==========================================
A novel attention mechanism combining two innovations:

1. **Differential Attention** (from Microsoft's Differential Transformer):
   The attention output is computed as the difference between two
   independent softmax attention distributions:
       DiffAttn(X) = (softmax(Q₁K₁ᵀ/√d) - λ·softmax(Q₂K₂ᵀ/√d)) · V

   Why this works: Standard attention computes a weighted average over
   values. But attention distributions are often noisy — they assign
   non-trivial probability to irrelevant tokens. By subtracting a second
   attention distribution, we cancel the common-mode noise, leaving
   only the truly salient attention patterns.

   λ is a learnable per-head scalar, initialized so that exp(λ_init)
   gives a reasonable starting point. During training, the model learns
   how much noise cancellation each head needs.

2. **Mamba State Injection**:
   The Mamba-2 SSM maintains a recurrent state h_t that captures
   long-range dependencies. We inject this state information into
   the attention K and V projections, creating a deeper interaction
   between the two pathways.

   h_t → small projection → added to K, V

   This goes beyond the simple per-dimension gate fusion in the
   original Mamformer, allowing the attention mechanism to directly
   leverage the SSM's sequential state.

Reference:
  "Differential Transformer" (Ye et al., Microsoft, 2024)
  "DeepSeek-V3 Technical Report" (DeepSeek-AI, 2024) — MLA concept

Compatibility:
  - Works with Grouped Query Attention (GQA): Q₁ and Q₂ share KV heads
  - Drop-in replacement for GroupedQueryAttention in MamformerBlock
  - Same output shape and caching interface
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamformer.layers.rope import RotaryEmbedding, apply_rotary_emb


class DifferentialStateAttention(nn.Module):
    """
    Differential State-Aware Attention with Mamba state injection.

    Architecture:
        Q₁: Linear(d_model → n_heads * head_dim)
        Q₂: Linear(d_model → n_heads * head_dim)
        K:  Linear(d_model → n_kv_heads * head_dim) [+ state_injection]
        V:  Linear(d_model → n_kv_heads * head_dim) [+ state_injection]
        O:  Linear(n_heads * head_dim → d_model)

        DiffAttn = (A₁ - λ·A₂) @ V
        where A_i = softmax(Q_i @ K^T / √d + mask)

        Output = RMSNorm(DiffAttn) @ O  (with GroupNorm for stability)

    Args:
        d_model: Model hidden dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key/value heads (for GQA)
        head_dim: Dimension per head
        max_seq_len: Maximum sequence length for RoPE precomputation
        rope_theta: RoPE base frequency
        lambda_init: Initial value for λ_log (λ = exp(λ_log))
        use_state_injection: Inject Mamba SSM state into K/V
        state_injection_dim: Bottleneck dimension for state injection
        dropout: Attention dropout rate
        sliding_window: Sliding window attention size (0 = disabled)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        lambda_init: float = 0.8,
        use_state_injection: bool = True,
        state_injection_dim: int = 64,
        dropout: float = 0.0,
        sliding_window: int = 0,
        use_yarn: bool = False,
        yarn_scale: float = 1.0,
        yarn_original_max_seq_len: int = 8192,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_head_groups = n_heads // n_kv_heads
        self.dropout = dropout
        self.sliding_window = sliding_window
        self.use_state_injection = use_state_injection
        self.state_injection_dim = state_injection_dim

        # ── Q projections (two for differential attention) ─────────
        self.q1_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.q2_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)

        # ── KV projections (shared across Q₁, Q₂ — GQA compatible) ──
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)

        # ── Output projection ───────────────────────────────────────
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # ── Differential attention lambda ───────────────────────────
        # λ = exp(λ_log), initialized to exp(lambda_init)
        # Per-head learnable parameter
        self.lambda_log = nn.Parameter(
            torch.ones(n_heads) * lambda_init
        )

        # ── State injection (Mamba SSM → Attention K/V) ─────────────
        if use_state_injection:
            # Project SSM state to a bottleneck, then expand to K/V space
            self.state_k_proj = nn.Sequential(
                nn.Linear(d_model, state_injection_dim, bias=False),  # Compress
                nn.SiLU(),
                nn.Linear(state_injection_dim, n_kv_heads * head_dim, bias=False),  # Expand to K
            )
            self.state_v_proj = nn.Sequential(
                nn.Linear(d_model, state_injection_dim, bias=False),  # Compress
                nn.SiLU(),
                nn.Linear(state_injection_dim, n_kv_heads * head_dim, bias=False),  # Expand to V
            )

        # ── GroupNorm for stability (as in Differential Transformer) ─
        # Applied per-head after differential combination
        self.group_norm = nn.GroupNorm(
            num_groups=n_heads,
            num_channels=n_heads * head_dim,
            eps=1e-6,
        )

        # ── RoPE (dynamic for long context) ────────────────────────
        self.rope = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
            use_yarn=use_yarn,
            yarn_scale=yarn_scale,
            yarn_original_max_seq_len=yarn_original_max_seq_len,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        std = 0.02
        for proj in [self.q1_proj, self.q2_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std)

        if self.use_state_injection:
            # Smaller init for state injection (start near identity)
            for module in [self.state_k_proj, self.state_v_proj]:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, mean=0.0, std=0.01)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache: Optional[dict] = None,
        ssm_state: Optional[torch.Tensor] = None,  # NEW: Mamba SSM state
    ) -> tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass for Differential State-Aware Attention.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            attention_mask: Optional attention mask (additive, 0=attend, -inf=mask)
            use_cache: If True, return updated KV cache
            cache: Optional KV cache dict with 'k' and 'v' keys
            ssm_state: Optional Mamba-2 SSM state (batch, d_inner, d_state)
                       When provided, injects state info into K and V

        Returns:
            (output, cache) — output shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # ── Project Q₁, Q₂, K, V ──────────────────────────────────
        q1 = self.q1_proj(x)  # (batch, seqlen, n_heads * head_dim)
        q2 = self.q2_proj(x)
        k = self.k_proj(x)    # (batch, seqlen, n_kv_heads * head_dim)
        v = self.v_proj(x)

        # ── Mamba State Injection ─────────────────────────────────
        if self.use_state_injection and ssm_state is not None:
            # ssm_state: (batch, d_inner, d_state)
            # We use the mean over d_state as a summary
            state_summary = ssm_state.mean(dim=-1)  # (batch, d_inner)
            # If d_inner != d_model, project (should match in Mamformer)
            if state_summary.shape[-1] != self.d_model:
                # Pad or truncate
                if state_summary.shape[-1] < self.d_model:
                    pad = torch.zeros(
                        batch_size, self.d_model - state_summary.shape[-1],
                        device=x.device, dtype=x.dtype,
                    )
                    state_summary = torch.cat([state_summary, pad], dim=-1)
                else:
                    state_summary = state_summary[..., : self.d_model]

            # Need to replicate state across sequence positions
            # For training: use a learned projection
            k_state = self.state_k_proj(state_summary)  # (batch, n_kv_heads * head_dim)
            v_state = self.state_v_proj(state_summary)

            # Add to K and V (broadcast across seq_len)
            k = k + k_state.unsqueeze(1)
            v = v + v_state.unsqueeze(1)

        # ── Reshape to (batch, n_heads, seq_len, head_dim) ─────────
        q1 = q1.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q2 = q2.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # ── RoPE ───────────────────────────────────────────────────
        cos, sin = self.rope(seq_len, x.device)
        cos = cos.to(q1.dtype)
        sin = sin.to(q1.dtype)

        q1 = apply_rotary_emb(q1, cos, sin)
        q2 = apply_rotary_emb(q2, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # ── KV Cache ──────────────────────────────────────────────
        if use_cache and cache is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)

        new_cache = {"k": k, "v": v} if use_cache else None

        # ── Repeat KV for GQA ─────────────────────────────────────
        k = k.repeat_interleave(self.n_head_groups, dim=1)
        v = v.repeat_interleave(self.n_head_groups, dim=1)

        # ── Differential Attention ─────────────────────────────────
        scale = 1.0 / math.sqrt(self.head_dim)

        # Compute A₁ = softmax(Q₁Kᵀ/√d)
        attn_logits_1 = torch.matmul(q1, k.transpose(-2, -1)) * scale  # (B, H, S, S)

        # Compute A₂ = softmax(Q₂Kᵀ/√d)
        attn_logits_2 = torch.matmul(q2, k.transpose(-2, -1)) * scale

        # Apply causal mask (DSA does manual softmax, must apply mask explicitly)
        if attention_mask is not None:
            attn_logits_1 = attn_logits_1 + attention_mask
            attn_logits_2 = attn_logits_2 + attention_mask
        else:
            kv_len = k.shape[2]
            q_len = q1.shape[2]
            causal_mask = torch.triu(
                torch.full((q_len, kv_len), float("-inf"), device=x.device, dtype=attn_logits_1.dtype),
                diagonal=kv_len - q_len + 1,
            ).unsqueeze(0).unsqueeze(0)
            attn_logits_1 = attn_logits_1 + causal_mask
            attn_logits_2 = attn_logits_2 + causal_mask

        # Compute lambda (per head), clamped for stability
        lam_raw = torch.exp(self.lambda_log).view(1, self.n_heads, 1, 1)
        lam = torch.clamp(lam_raw, max=0.99)

        # Differential softmax combination with (1-lambda) normalization
        attn_weights_1 = F.softmax(attn_logits_1, dim=-1)
        attn_weights_2 = F.softmax(attn_logits_2, dim=-1)
        diff_attn_weights = attn_weights_1 - lam * attn_weights_2
        diff_attn_weights = diff_attn_weights / (1.0 - lam + 1e-8)

        # Apply differential attention
        attn_output = torch.matmul(diff_attn_weights, v)  # (B, H, S, head_dim)

        # ── Reshape back ───────────────────────────────────────────
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1
        )

        # ── GroupNorm stabilization ────────────────────────────────
        # Permute to (B, C, S) for GroupNorm, then back
        attn_output = attn_output.transpose(1, 2)  # (B, d_model, S)
        attn_output = self.group_norm(attn_output)
        attn_output = attn_output.transpose(1, 2)  # (B, S, d_model)

        # ── Output projection ──────────────────────────────────────
        output = self.o_proj(attn_output)

        return output, new_cache

    def get_lambda_values(self) -> torch.Tensor:
        """
        Return current λ values for analysis.

        Returns:
            Tensor of shape (n_heads,) with positive λ values
        """
        return torch.exp(self.lambda_log).detach()

    def extra_repr(self) -> str:
        sw_info = f", sliding_window={self.sliding_window}" if self.sliding_window > 0 else ""
        si_info = ", state_injection" if self.use_state_injection else ""
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, head_dim={self.head_dim}, "
            f"n_head_groups={self.n_head_groups}, "
            f"lambda_init={self.lambda_log[0].item():.2f}"
            f"{sw_info}{si_info}"
        )
