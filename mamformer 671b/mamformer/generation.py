"""
Mamformer Generation Utilities
===========================
Text generation utilities with various decoding strategies.

Supports:
- Greedy decoding (temperature=0)
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Beam search (basic implementation)
- Streaming generation

The generation loop leverages the recurrent-mode Mamba-2 and KV-cache
attention for O(1)-per-token state updates during autoregressive generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Generator

import torch
import torch.nn.functional as F


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    # Decoding strategy
    max_new_tokens: int = 256
    temperature: float = 1.0  # 0 = greedy, >0 = sampling
    top_k: int = 50  # 0 = disabled
    top_p: float = 0.9  # 1.0 = disabled
    repetition_penalty: float = 1.0  # >1.0 penalizes repetition

    # Beam search (when num_beams > 1)
    num_beams: int = 1
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    length_penalty: float = 1.0
    early_stopping: bool = False

    # Stopping criteria
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    stop_strings: List[str] = field(default_factory=list)

    # Output
    return_full_text: bool = True
    skip_special_tokens: bool = True

    def validate(self) -> None:
        """Validate generation config consistency."""
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {self.num_beams}")
        if self.num_beams > 1 and self.temperature > 0:
            raise ValueError("Beam search and sampling cannot be used together")


class GenerationMixin:
    """
    Mixin class that adds generation capabilities to MamformerForCausalLM.

    Usage:
        model = MamformerForCausalLM(config)
        generated = model.generate_ids(
            input_ids,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )
    """

    @torch.no_grad()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate token IDs autoregressively.

        Args:
            input_ids: (batch, seqlen) prompt token IDs
            generation_config: GenerationConfig or keyword overrides

        Keyword args override GenerationConfig fields:
            max_new_tokens, temperature, top_k, top_p, etc.

        Returns:
            Generated token IDs (batch, prompt_len + new_tokens)
        """
        if generation_config is None:
            generation_config = GenerationConfig()

        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(generation_config, key):
                setattr(generation_config, key, value)

        generation_config.validate()

        if generation_config.num_beams > 1:
            return self._beam_search(input_ids, generation_config)
        else:
            return self._sample_loop(input_ids, generation_config)

    def _sample_loop(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """
        Autoregressive sampling loop with caching.

        Uses KV-cache for attention and recurrent state for Mamba-2,
        making each step O(1) in sequence length.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        generated = input_ids
        cache = None

        # Track which sequences are still generating (for batch EOS)
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

        for step in range(config.max_new_tokens):
            if not unfinished.any():
                break

            # Only process last token when cache is available
            if cache is not None:
                current_input = generated[:, -1:]
            else:
                current_input = generated

            # Forward pass
            outputs = self.forward(
                input_ids=current_input,
                use_cache=True,
                cache=cache,
            )

            logits = outputs["logits"][:, -1, :]  # (batch, vocab_size)
            cache = outputs.get("cache")

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated, config.repetition_penalty
                )

            # Sample next token
            next_token = self._sample_token(logits, config)

            # Mask finished sequences
            next_token = next_token * unfinished.long().unsqueeze(-1)
            next_token = next_token.masked_fill(~unfinished.unsqueeze(-1), config.pad_token_id or 0)

            generated = torch.cat([generated, next_token], dim=-1)

            # Check EOS
            if config.eos_token_id is not None:
                unfinished = unfinished & (next_token.squeeze(-1) != config.eos_token_id)

        return generated

    def _sample_token(
        self,
        logits: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """
        Sample a single token from logits.

        Args:
            logits: (batch, vocab_size) raw logits
            config: GenerationConfig

        Returns:
            (batch, 1) token indices
        """
        vocab_size = logits.shape[-1]

        if config.temperature == 0:
            # Greedy decoding
            return logits.argmax(dim=-1, keepdim=True)

        # Temperature scaling
        logits = logits / config.temperature

        # Top-k filtering
        if config.top_k > 0:
            k = min(config.top_k, vocab_size)
            top_k_values, _ = torch.topk(logits, k, dim=-1)
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < min_top_k, float("-inf"))

        # Top-p (nucleus) filtering
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            # Shift: keep first token that exceeds threshold
            sorted_mask = cumulative_probs > config.top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            # Scatter back
            mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            logits = logits.masked_fill(mask, float("-inf"))

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Handle all-zero rows (can happen with aggressive filtering)
        probs = torch.where(
            probs.sum(dim=-1, keepdim=True) > 0,
            probs,
            torch.ones_like(probs) / vocab_size,
        )

        return torch.multinomial(probs, num_samples=1)

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """
        Apply repetition penalty: reduce logits of previously generated tokens.

        Args:
            logits: (batch, vocab_size)
            generated_ids: (batch, gen_len) all tokens generated so far
            penalty: Penalty factor (>1 = penalize, <1 = encourage)

        Returns:
            Modified logits
        """
        if penalty == 1.0:
            return logits

        # Get unique tokens per batch item
        for i in range(logits.shape[0]):
            unique_ids = set(generated_ids[i].tolist())
            for token_id in unique_ids:
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= penalty
                else:
                    logits[i, token_id] *= penalty

        return logits

    def _beam_search(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """
        Basic beam search decoding.

        Args:
            input_ids: (batch, seqlen) prompt
            config: GenerationConfig (must have num_beams > 1)

        Returns:
            Best generated sequences
        """
        batch_size = input_ids.shape[0]
        num_beams = config.num_beams
        device = input_ids.device
        vocab_size = self.config.vocab_size

        # Expand input for beam search
        # Each batch item is repeated num_beams times
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1)
        input_ids = input_ids.reshape(batch_size * num_beams, -1)

        # Beam scores: (batch_size * num_beams,)
        beam_scores = torch.zeros(batch_size * num_beams, device=device)
        beam_scores[1::num_beams] = -1e9  # Disable redundant beams initially

        generated = input_ids
        cache = None

        for step in range(config.max_new_tokens):
            # Forward pass
            if cache is not None:
                current_input = generated[:, -1:]
            else:
                current_input = generated

            outputs = self.forward(
                input_ids=current_input,
                use_cache=True,
                cache=cache,
            )

            logits = outputs["logits"][:, -1, :]  # (batch*num_beams, vocab_size)
            cache = outputs.get("cache")

            # Apply length penalty
            if step + 1 > 1:
                scores = F.log_softmax(logits, dim=-1) / (step + 1) ** config.length_penalty
            else:
                scores = F.log_softmax(logits, dim=-1)

            scores = scores + beam_scores.unsqueeze(-1)  # (batch*num_beams, vocab_size)

            # Reshape to (batch_size, num_beams * vocab_size)
            scores = scores.view(batch_size, num_beams * vocab_size)
            top_scores, top_indices = torch.topk(scores, 2 * num_beams, dim=-1)

            # Decode: which beam and which token
            next_beam_indices = top_indices // vocab_size  # (batch_size, 2*num_beams)
            next_token_indices = top_indices % vocab_size  # (batch_size, 2*num_beams)

            # Update beam scores
            beam_scores = top_scores.view(-1)
            next_beam_indices = next_beam_indices.view(-1)
            next_token_indices = next_token_indices.view(-1)

            # Reorder beams AND their KV caches
            generated = generated[next_beam_indices]
            cache = self._reorder_cache(cache, next_beam_indices)
            generated = torch.cat([generated, next_token_indices.unsqueeze(-1)], dim=-1)

            # Check EOS
            if config.eos_token_id is not None:
                eos_mask = next_token_indices == config.eos_token_id
                beam_scores = beam_scores.masked_fill(eos_mask, float("-inf"))

            # Select top num_beams AND reorder caches
            beam_scores = beam_scores.view(batch_size, -1)
            _, best_indices = torch.topk(beam_scores, num_beams, dim=-1)
            # Add batch offsets to convert from per-batch column indices to global row indices
            batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * (2 * num_beams)
            best_indices_global = (best_indices + batch_offsets).view(-1)
            beam_scores = beam_scores.view(-1)[best_indices_global]
            generated = generated[best_indices_global]
            cache = self._reorder_cache(cache, best_indices_global)

            # End if all beams hit EOS
            if config.eos_token_id is not None:
                if (next_token_indices[best_indices] == config.eos_token_id).all():
                    break

        # Return best sequence for each batch item
        best_generated = generated.view(batch_size, num_beams, -1)[:, 0, :]
        return best_generated

    @staticmethod
    def _reorder_cache(cache, indices):
        """Reorder cached KV+SSM states to match beam reordering."""
        if cache is None:
            return None
        new_cache = []
        for layer_cache in cache:
            if layer_cache is None:
                new_cache.append(None)
                continue
            new_layer = {}
            for key, val in layer_cache.items():
                if isinstance(val, dict):
                    new_layer[key] = {k: v[indices] if v is not None else None
                                     for k, v in val.items()}
                elif isinstance(val, torch.Tensor):
                    new_layer[key] = val[indices]
                else:
                    new_layer[key] = val
            new_cache.append(new_layer)
        return new_cache

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Generator[int, None, torch.Tensor]:
        """
        Stream generated tokens one at a time.

        Yields token IDs as they are generated. Useful for
        interactive applications where tokens are displayed in real-time.

        Yields:
            Individual token IDs (int)
        """
        if generation_config is None:
            generation_config = GenerationConfig()

        for key, value in kwargs.items():
            if hasattr(generation_config, key):
                setattr(generation_config, key, value)

        config = generation_config
        device = input_ids.device
        generated = input_ids
        cache = None

        # Yield prompt tokens first
        if config.return_full_text:
            for token_id in input_ids[0].tolist():
                yield token_id

        for _ in range(config.max_new_tokens):
            if cache is not None:
                current_input = generated[:, -1:]
            else:
                current_input = generated

            outputs = self.forward(
                input_ids=current_input,
                use_cache=True,
                cache=cache,
            )

            logits = outputs["logits"][:, -1, :]
            cache = outputs.get("cache")

            # Apply repetition penalty (mirrors _sample_loop behavior)
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated, config.repetition_penalty
                )

            next_token = self._sample_token(logits, config)
            generated = torch.cat([generated, next_token], dim=-1)
            token_id = next_token[0, 0].item()

            yield token_id

            if config.eos_token_id is not None and token_id == config.eos_token_id:
                break
