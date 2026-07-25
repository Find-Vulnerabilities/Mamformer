"""
Shared Token Sampling Utilities
================================
Consolidated temperature / top-k / top-p / greedy decoding logic
used by model.generate(), GenerationMixin, and ChatSession.

This eliminates duplicate implementations across the codebase.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_one_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Sample a single token from logits with temperature, top-k, top-p.

    This is the canonical implementation shared by all generation paths.

    Args:
        logits: (batch, vocab_size) raw logits
        temperature: >0 for sampling, <=0 for greedy
        top_k: Top-k filter (>0 to enable, 0 to disable)
        top_p: Nucleus threshold (<1.0 to enable, 1.0 to disable)

    Returns:
        (batch, 1) token indices
    """
    # Greedy short-circuit
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    vocab_size = logits.shape[-1]

    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        k = min(top_k, vocab_size)
        top_k_values, _ = torch.topk(logits, k, dim=-1)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < min_top_k, float("-inf"))

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Shift: keep the first token that exceeds threshold
        sorted_mask = cumulative_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False
        # Scatter back to original ordering
        mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Handle all-zero rows (can happen with aggressive filtering)
    row_sums = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(
        row_sums > 0,
        probs,
        torch.ones_like(probs) / vocab_size,
    )
    # Re-normalize after zero-row fix
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    return torch.multinomial(probs, num_samples=1)


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """
    Apply repetition penalty: penalize already-generated tokens.

    For penalty > 1.0: positive logits are divided, negative are multiplied.
    This pushes the model away from repeating tokens.

    Args:
        logits: (batch, vocab_size)
        generated_ids: (batch, seq_len) all tokens generated so far
        penalty: Penalty factor (>1 = penalize, <1 = encourage, 1.0 = no-op)

    Returns:
        Modified logits
    """
    if penalty == 1.0:
        return logits

    for i in range(logits.shape[0]):
        unique_ids = set(generated_ids[i].tolist())
        for token_id in unique_ids:
            if logits[i, token_id] > 0:
                logits[i, token_id] /= penalty
            else:
                logits[i, token_id] *= penalty

    return logits
