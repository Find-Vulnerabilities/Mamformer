"""
Thinking Data Formatting for SFT Training
============================================
Prepares training data with thinking control tokens so the model
learns to use multi-path parallel reasoning during SFT.

When --thinking_format is enabled during training:
  - Each example is wrapped with thinking markers
  - The model learns: prompt → <think> → reasoning → </think>
                     → <answer> → final answer → </answer>

This ensures that at inference time, when thinking mode is enabled
with the same control tokens, the model has been trained to use them.

Usage (in train.py):
    python scripts/train.py --config ... --thinking_format

Format applied to each (prompt, response) pair:
    {prompt}<|think_start|>{response}<|think_end|>
    <|summary_start|>{response}<|answer_start|>{response}<|answer_end|>

During SFT, the loss is computed on ALL tokens (thinking + answer),
so the model learns both how to think and how to answer.
"""

from __future__ import annotations

from typing import List

import torch


# ── Token ID constants (mirrors MamformerTokenizer) ────────────────
THINK_START: int = 3
THINK_END: int = 4
ANSWER_START: int = 5
ANSWER_END: int = 6
SUMMARY_START: int = 7


def format_sequence_with_thinking(
    prompt_ids: List[int],
    response_ids: List[int],
) -> List[int]:
    """
    Wrap a prompt-response pair with thinking control tokens for SFT.

    The model learns to:
      1. Generate reasoning after <|think_start|>
      2. End reasoning with <|think_end|>
      3. Synthesize after <|summary_start|>
      4. Answer after <|answer_start|>
      5. End with <|answer_end|>

    NOTE: The current format repeats response_ids three times (as thinking
    content, summary content, and answer content). This is a BOOTSTRAP
    placeholder — it teaches the model the token format and basic structure
    but does NOT teach genuine parallel reasoning or multi-path synthesis.

    TODO: Replace with real multi-path training data where:
      - Thinking section contains distinct reasoning chains per path
      - Summary section synthesizes insights across paths
      - Answer section is the final condensed answer

    Args:
        prompt_ids: Tokenized prompt
        response_ids: Tokenized target response

    Returns:
        Full sequence with thinking markers inserted

    Format:
        prompt + <|think_start|> + response + <|think_end|>
        + <|summary_start|> + response + <|answer_start|> + response + <|answer_end|>
    """
    return (
        prompt_ids
        + [THINK_START]
        + response_ids
        + [THINK_END]
        + [SUMMARY_START]
        + response_ids
        + [ANSWER_START]
        + response_ids
        + [ANSWER_END]
    )


def format_labels_with_thinking(
    prompt_ids: List[int],
    response_ids: List[int],
) -> List[int]:
    """
    Create labels for thinking-formatted SFT training.

    Prompt tokens are masked (-100), thinking/answer tokens are kept.
    The control tokens themselves are also kept (model learns to emit them).

    Args:
        prompt_ids: Tokenized prompt
        response_ids: Tokenized target response

    Returns:
        Labels with prompt positions set to -100 (ignored in loss)
    """
    full = format_sequence_with_thinking(prompt_ids, response_ids)
    labels = full.copy()
    # Mask prompt tokens — model shouldn't learn to predict the prompt
    for i in range(len(prompt_ids)):
        labels[i] = -100
    return labels


def format_batch_with_thinking(
    prompt_ids_batch: torch.Tensor,
    response_ids_batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batch version: format multiple prompt-response pairs with thinking tokens.

    Args:
        prompt_ids_batch: (batch, prompt_len) padded prompts
        response_ids_batch: (batch, response_len) padded responses

    Returns:
        (input_ids, labels) — both (batch, total_len) with thinking markers
    """
    batch_size = prompt_ids_batch.shape[0]
    results_inputs = []
    results_labels = []

    for i in range(batch_size):
        # Remove padding
        prompt = [t for t in prompt_ids_batch[i].tolist() if t != 0]
        response = [t for t in response_ids_batch[i].tolist() if t != 0]

        inputs = format_sequence_with_thinking(prompt, response)
        labels = format_labels_with_thinking(prompt, response)

        results_inputs.append(inputs)
        results_labels.append(labels)

    # Pad to max length
    max_len = max(len(seq) for seq in results_inputs)
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels_out = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i in range(batch_size):
        L = len(results_inputs[i])
        input_ids[i, :L] = torch.tensor(results_inputs[i], dtype=torch.long)
        labels_out[i, :L] = torch.tensor(results_labels[i], dtype=torch.long)

    return input_ids, labels_out


def is_thinking_token(token_id: int) -> bool:
    """Check if a token ID is a thinking control token."""
    return token_id in {THINK_START, THINK_END, ANSWER_START, ANSWER_END, SUMMARY_START}
