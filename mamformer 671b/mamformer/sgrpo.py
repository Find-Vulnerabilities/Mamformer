"""
Stochastic GRPO (S-GRPO): Sparse Token Sampling for GRPO Training
===================================================================
Modifies GRPO loss computation to use only a sampled subset of tokens
per trajectory instead of all tokens. This provides:

1. **Memory efficiency**: Fewer tokens → smaller backward graph
2. **Implicit regularization**: Acts like dropout on the loss
3. **Faster training**: Reduced compute per optimization step

Key parameters:
  - P (sampling probability): 0.3-0.5, fraction of response tokens to include
  - alpha (cutoff index): First α tokens always included
  - k (max tokens cap): Upper bound on tokens included in loss

Algorithm (from "Token-Efficient RL for LLM Reasoning", arXiv:2504.20834):
  1. For each trajectory of length L:
     - Always include tokens in [0, alpha] (typically 0)
     - Sample remaining tokens [alpha, L) with Bernoulli(P)
     - If total sampled > k, randomly select k tokens
  2. Compute log-probs only on the selected subset
  3. Average over selected tokens for response-level log-prob
  4. Standard GRPO loss on top: -mean(advantages * avg_log_probs) + KL

Usage:
    from mamformer.sgrpo import StochasticGRPOConfig, sample_token_mask

    cfg = StochasticGRPOConfig(enabled=True, p=0.4, alpha=0)
    mask = sample_token_mask(seq_len=512, config=cfg)
    # Use mask when computing get_log_probs(token_mask=mask)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StochasticGRPOConfig:
    """
    Configuration for Stochastic GRPO sparse token sampling.

    Args:
        enabled: Master toggle — False = standard GRPO (all tokens)
        p: Sampling probability for tokens beyond alpha (0.3-0.5 recommended)
        alpha: Number of leading tokens always included in loss
        k: Maximum tokens to include (0 = no cap)
        seed: Optional seed for reproducible sampling
    """

    enabled: bool = False
    p: float = 0.4
    alpha: int = 0
    k: int = 0
    seed: Optional[int] = None

    def validate(self) -> None:
        """Validate config consistency."""
        if self.enabled:
            if not (0.0 < self.p <= 1.0):
                raise ValueError(f"p must be in (0, 1], got {self.p}")
            if self.alpha < 0:
                raise ValueError(f"alpha must be >= 0, got {self.alpha}")
            if self.k < 0:
                raise ValueError(f"k must be >= 0, got {self.k}")


# ═══════════════════════════════════════════════════════════════════════
# Token Mask Generation
# ═══════════════════════════════════════════════════════════════════════

def sample_token_mask(
    seq_len: int,
    config: StochasticGRPOConfig,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Generate a binary mask for token-level loss computation.

    Sampling strategy:
      1. First `alpha` tokens: always included (mask = True)
      2. Remaining tokens: sampled with Bernoulli(p)
      3. If k > 0 and total selected > k: randomly subsample to exactly k
         (with alpha tokens always preserved)

    Args:
        seq_len: Number of response tokens
        config: StochasticGRPOConfig with sampling parameters
        device: Target device for the mask tensor

    Returns:
        Boolean tensor of shape (seq_len,) where True = include in loss

    Example:
        >>> cfg = StochasticGRPOConfig(enabled=True, p=0.4, alpha=10, k=200)
        >>> mask = sample_token_mask(512, cfg)
        >>> mask.sum().item()  # ~210 tokens selected (10 alpha + ~200 sampled)
    """
    if not config.enabled or seq_len == 0:
        return torch.ones(seq_len, dtype=torch.bool, device=device)

    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

    # Step 1: First alpha tokens always included
    alpha = min(config.alpha, seq_len)
    if alpha > 0:
        mask[:alpha] = True

    # Step 2: Remaining tokens sampled with Bernoulli(p)
    remaining = seq_len - alpha
    if remaining > 0:
        if config.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(config.seed)
            sampled = torch.rand(remaining, device=device, generator=generator) < config.p
        else:
            sampled = torch.rand(remaining, device=device) < config.p
        mask[alpha:] = sampled

    # Step 3: Apply max tokens cap
    if config.k > 0:
        n_selected = mask.sum().item()
        if n_selected > config.k:
            # Keep all alpha tokens, randomly select from the rest
            selected_indices = mask.nonzero(as_tuple=False).squeeze(-1)
            # Separate alpha-protected from the rest
            is_alpha = selected_indices < alpha
            alpha_indices = selected_indices[is_alpha]
            non_alpha_indices = selected_indices[~is_alpha]

            # How many non-alpha slots remain
            remaining_slots = max(0, config.k - len(alpha_indices))
            if remaining_slots < len(non_alpha_indices):
                # Randomly select from non-alpha
                perm = torch.randperm(len(non_alpha_indices), device=device)[:remaining_slots]
                non_alpha_indices = non_alpha_indices[perm]

            # Rebuild mask
            mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            mask[alpha_indices] = True
            if len(non_alpha_indices) > 0:
                mask[non_alpha_indices] = True

    return mask
