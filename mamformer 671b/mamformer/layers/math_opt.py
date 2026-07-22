"""
Mathematical Optimizations for Mamformer
==========================================
Four key improvements to the mathematical foundations:

1. DynamicGate — context-dependent attention/SSM fusion
2. EntropyRouter — entropy-regularized MoE routing
3. AdaptiveLambda — context-dependent DSA noise cancellation
4. DeepNorm — stabilized residual scaling for deep networks

Reference:
  "DeepNet: Scaling Transformers to 1,000 Layers" (Wang et al., 2022)
  "Mixtral of Experts" (Mistral AI, 2024)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# 1. Dynamic Gate — Context-Dependent Fusion
# ═══════════════════════════════════════════════════════════════════════

class DynamicGate(nn.Module):
    """
    Context-dependent gating for attention/SSM fusion.

    Replaces the static per-dimension learnable α with a small MLP
    that produces gates based on the current hidden states:

        gate(x) = σ(W₂ · SiLU(W₁ · mean(x, dim=1)) + b)

    Why this helps:
      - Static gate treats all tokens identically regardless of context
      - Dynamic gate can route attention-heavy tokens to GQA and
        sequential-pattern tokens to Mamba-2
      - The gate is computed per-sequence (not per-token) for stability

    Args:
        d_model: Hidden dimension
        bottleneck: Bottleneck dimension for gate MLP (default: d_model // 16)
    """

    def __init__(self, d_model: int, bottleneck: int = 0):
        super().__init__()
        if bottleneck <= 0:
            bottleneck = max(d_model // 16, 16)

        # Small MLP: mean_pool(x) → bottleneck → d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, d_model, bias=True),
        )
        # Initialize for near-0.5 output (balanced start)
        nn.init.normal_(self.mlp[0].weight, std=0.02 / bottleneck ** 0.5)
        nn.init.normal_(self.mlp[2].weight, std=0.02 / d_model ** 0.5)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic gate values from hidden states.

        Args:
            x: (batch, seqlen, d_model) — pre-norm hidden states

        Returns:
            gate: (d_model,) — values in (0, 1) for per-dimension fusion
        """
        # Pool over batch and sequence for global context
        pooled = x.mean(dim=(0, 1))  # (d_model,)
        # Normalize to unit norm for stability
        pooled = F.normalize(pooled.unsqueeze(0), dim=-1).squeeze(0)
        # MLP → sigmoid
        gate = torch.sigmoid(self.mlp(pooled))
        return gate


# ═══════════════════════════════════════════════════════════════════════
# 2. Entropy-Regularized MoE Router
# ═══════════════════════════════════════════════════════════════════════

class EntropyRouter(nn.Module):
    """
    MoE router with entropy regularization bonus.

    Adds a small entropy bonus to the routing scores to prevent
    expert collapse. Unlike auxiliary loss methods, this does not
    add a separate loss term — the bonus is applied directly to
    router logits during forward pass.

    entropy_bonus = γ * H(softmax(scores)) per token

    Where H(p) = -Σ p_i log(p_i). This encourages the router to
    produce more uniform (higher entropy) distributions, preventing
    premature convergence to a few experts.

    Args:
        d_model: Input dimension
        n_experts: Number of experts
        entropy_weight: γ — weight of entropy bonus (default 0.01)
    """

    def __init__(self, d_model: int, n_experts: int, entropy_weight: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.entropy_weight = entropy_weight

        self.weight = nn.Parameter(torch.empty(n_experts, d_model))
        nn.init.normal_(self.weight, std=0.02 / n_experts ** 0.5)

    def forward(
        self,
        x: torch.Tensor,
        expert_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute entropy-regularized routing.

        Args:
            x: (batch, seqlen, d_model)
            expert_bias: Optional (n_experts,) bias for load balancing

        Returns:
            (scores, entropy, aux_info) where:
              - scores: (batch*seqlen, n_experts) raw routing scores
              - entropy: per-token entropy values
              - aux_info: dict with entropy statistics
        """
        N = x.shape[0] * x.shape[1]
        x_flat = x.view(N, self.d_model)

        # Raw scores
        logits = F.linear(x_flat, self.weight)
        if expert_bias is not None:
            logits = logits + expert_bias

        # Softmax for probability distribution
        probs = F.softmax(logits, dim=-1)

        # Entropy: H = -Σ p_i log(p_i)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (N,)
        max_entropy = math.log(self.n_experts)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else entropy

        # Entropy-regularized logits: mix original logits with uniform noise
        # When entropy is low (near-collapse), add more noise to encourage exploration
        # When entropy is high (diverse), keep logits mostly unchanged
        # Formula: logits_mixed = (1 - γ·(1-H)) * logits + γ·(1-H) * noise
        noise = torch.randn_like(logits) * 0.1  # Small Gaussian noise
        entropy_deficit = 1.0 - normalized_entropy  # High when routing is too deterministic
        mix_ratio = self.entropy_weight * entropy_deficit.unsqueeze(-1)
        mix_ratio = torch.clamp(mix_ratio, 0.0, 0.3)  # Max 30% noise
        scores_with_bonus = (1.0 - mix_ratio) * logits + mix_ratio * noise

        aux_info = {
            "entropy_mean": entropy.mean().item(),
            "entropy_normalized": normalized_entropy.mean().item(),
            "entropy_weight": self.entropy_weight,
        }

        return scores_with_bonus, entropy, aux_info


# ═══════════════════════════════════════════════════════════════════════
# 3. Adaptive Lambda — Context-Dependent DSA
# ═══════════════════════════════════════════════════════════════════════

class AdaptiveLambda(nn.Module):
    """
    Context-dependent lambda for Differential State-Aware Attention.

    Instead of a static per-head lambda, computes lambda based on
    the attention query statistics:

        lambda = σ(W · std(Q, dim=-1) + b) * max_lambda

    High query variance → more noise cancellation needed → higher lambda.
    Low query variance → clean attention → lower lambda.

    Args:
        n_heads: Number of attention heads
        max_lambda: Maximum lambda value (default 0.99 for stability)
    """

    def __init__(self, n_heads: int, max_lambda: float = 0.99):
        super().__init__()
        self.n_heads = n_heads
        self.max_lambda = max_lambda

        # Small projection: per-head query statistics → lambda
        self.scale = nn.Parameter(torch.ones(n_heads) * 0.5)
        self.bias = nn.Parameter(torch.zeros(n_heads))

    def forward(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive lambda from query statistics.

        Args:
            q1: (batch, n_heads, seqlen, head_dim) — first query
            q2: (batch, n_heads, seqlen, head_dim) — second query

        Returns:
            lambda: (n_heads,) — adaptive lambda per head
        """
        # Query variance per head: how spread out are the attention queries?
        # Higher variance = more uncertainty = need more noise cancellation
        q1_var = q1.std(dim=(0, 2, 3))  # (n_heads,) — variance across batch + seq + head_dim
        q2_var = q2.std(dim=(0, 2, 3))

        # Average query "uncertainty"
        uncertainty = (q1_var + q2_var) / 2.0

        # Scale and clamp
        lam_raw = torch.sigmoid(self.scale * uncertainty + self.bias)
        lam = lam_raw * self.max_lambda

        return lam  # (n_heads,)


# ═══════════════════════════════════════════════════════════════════════
# 4. DeepNorm — Stabilized Residual Scaling
# ═══════════════════════════════════════════════════════════════════════

class DeepNorm(nn.Module):
    """
    DeepNorm residual scaling for stable deep network training.

    Standard residual:  y = x + f(Norm(x))
    DeepNorm residual: y = α * x + f(Norm(x))

    Where α is a small constant that scales down the residual branch.
    This prevents activation explosion in deep networks (50+ layers).

    For Mamformer-671B with 52-60 layers, DeepNorm improves training
    stability by keeping activations bounded.

    Reference: "DeepNet: Scaling Transformers to 1,000 Layers"
              (Wang et al., 2022)

    Args:
        d_model: Hidden dimension
        n_layers: Total number of layers (for automatic α computation)
        alpha: Manual override for residual scale (auto-computed if None)
    """

    def __init__(self, d_model: int, n_layers: int = 52, alpha: Optional[float] = None):
        super().__init__()
        if alpha is not None:
            self.alpha = alpha
        else:
            # DeepNet formula: α = (2 * N)^(-1/4) where N = n_layers
            self.alpha = (2.0 * n_layers) ** (-0.25)

        # Learnable per-dimension scaling (optional fine-tuning)
        self.gamma = nn.Parameter(torch.ones(d_model) * self.alpha)

    def forward(self, residual: torch.Tensor, updated: torch.Tensor) -> torch.Tensor:
        """
        Apply DeepNorm-scaled residual connection.

        Args:
            residual: Original input x
            updated: Output of sub-layer f(Norm(x))

        Returns:
            α * residual + updated
        """
        return self.gamma * residual + updated

    def extra_repr(self) -> str:
        return f"alpha={self.alpha:.4f}"


# ═══════════════════════════════════════════════════════════════════════
# Utility: Combine all optimizations
# ═══════════════════════════════════════════════════════════════════════

def compute_moe_entropy_loss(
    entropies: list[torch.Tensor],
    target_entropy: float = 0.5,
    weight: float = 0.01,
) -> torch.Tensor:
    """
    Compute entropy-based auxiliary loss for MoE routing.

    Penalizes routing entropy that is too far from target,
    encouraging healthy expert diversity.

    Args:
        entropies: List of per-layer normalized entropy tensors
        target_entropy: Desired entropy level (0 = collapse, 1 = uniform)
        weight: Loss weight

    Returns:
        Scalar entropy loss
    """
    if not entropies:
        return torch.tensor(0.0)

    losses = []
    for h in entropies:
        if h.numel() > 0:
            loss = (h.mean() - target_entropy) ** 2
            losses.append(loss)

    if losses:
        return weight * torch.stack(losses).mean()
    return torch.tensor(0.0)
