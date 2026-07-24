"""
Self-Reflection Module for Mamformer
======================================
Enables the model to review and refine its own outputs through
a dedicated reflection mechanism.

Architecture:
  1. Generate initial response (standard forward pass)
  2. Reflection: analyze the response for errors/improvements
  3. Refinement: generate improved response based on reflection

This implements a form of "test-time compute" — the model can
spend extra computation to improve output quality through
self-critique and revision.

Two modes:
  - Fast: standard generation (no reflection)
  - Reflect: generate → critique → refine (1 extra forward pass)

Training:
  - During SFT, add reflection loss on top of standard LM loss
  - Model learns to generate useful critiques and improvements

Reference:
  "Self-Refine: Iterative Refinement with Self-Feedback" (Madaan et al., 2023)
  "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022)
  "Chain-of-Thought Prompting Elicits Reasoning" (Wei et al., 2022)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamformer.layers.norm import RMSNorm


# ═══════════════════════════════════════════════════════════════════════
# Reflection Module
# ═══════════════════════════════════════════════════════════════════════

class ReflectionModule(nn.Module):
    """
    Self-reflection module that critiques and refines model outputs.

    The module takes generated tokens and their hidden states,
    produces a critique, and generates improved output.

    Args:
        d_model: Hidden dimension
        vocab_size: Vocabulary size
        max_reflection_tokens: Max tokens for reflection output
        embedding_weight: Optional tied embedding for output projection
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_reflection_tokens: int = 128,
        embedding_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_reflection_tokens = max_reflection_tokens

        # ── Reflection Encoder ──────────────────────────────────
        # Processes the generated output + its hidden states
        # to produce a reflection summary
        self.reflection_norm = RMSNorm(d_model)
        self.reflection_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model, bias=False),
        )

        # ── Reflection Token Generator ───────────────────────────
        # Generates critique tokens that describe issues found
        if embedding_weight is not None:
            self.critique_weight = nn.Parameter(embedding_weight.data.clone())
        else:
            self.critique_weight = nn.Parameter(torch.empty(vocab_size, d_model))
            nn.init.normal_(self.critique_weight, std=0.02)

        # ── Refinement Adapter ──────────────────────────────────
        # Takes the reflection summary and original hidden states
        # to produce refined hidden states
        self.refine_adapter = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )

        # ── Refinement Head ─────────────────────────────────────
        if embedding_weight is not None:
            self.refine_weight = nn.Parameter(embedding_weight.data.clone())
        else:
            self.refine_weight = nn.Parameter(torch.empty(vocab_size, d_model))
            nn.init.normal_(self.refine_weight, std=0.02)

        # ── Confidence Scorer ───────────────────────────────────
        # Predicts whether reflection actually improved quality
        self.confidence = nn.Sequential(
            nn.Linear(d_model, 1, bias=False),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.reflection_proj, self.refine_adapter]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
        nn.init.normal_(self.confidence[0].weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        generated_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Reflect on generated output and produce refined version.

        Args:
            hidden_states: (batch, seqlen, d_model) — model's hidden states
            generated_tokens: (batch, genlen) — generated token IDs
            attention_mask: Optional attention mask

        Returns:
            dict with:
              - "refined_logits": (batch, seqlen, vocab_size)
              - "critique_logits": (batch, reflen, vocab_size)
              - "confidence": (batch, 1) — how confident the refinement is
              - "reflection_summary": (batch, d_model)
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # ── 1. Compute reflection summary ──────────────────────
        # Pool over sequence to get a global representation
        # Weight by token position (later tokens more important)
        pooled = hidden_states.mean(dim=1)  # (batch, d_model)

        # Embed generated tokens to capture surface-form information
        gen_embeds = F.embedding(generated_tokens, self.critique_weight)  # (batch, genlen, d_model)
        pooled_tokens = gen_embeds.mean(dim=1)  # (batch, d_model)

        # Combine hidden-state representation with token-level information
        pooled = pooled + pooled_tokens

        # Normalize and project
        pooled_norm = self.reflection_norm(pooled)
        reflection_summary = self.reflection_proj(pooled_norm)  # (batch, d_model)

        # ── 2. Generate critique ────────────────────────────────
        # Project to vocabulary to get critique tokens
        critique_logits = F.linear(reflection_summary.unsqueeze(1), self.critique_weight)  # (batch, 1, vocab_size)

        # ── 3. Refine hidden states ─────────────────────────────
        # Concatenate reflection summary with original hidden states
        reflection_expanded = reflection_summary.unsqueeze(1).expand(-1, seq_len, -1)
        concat_states = torch.cat([hidden_states, reflection_expanded], dim=-1)  # (batch, seqlen, 2*d_model)

        refined_states = self.refine_adapter(concat_states)  # (batch, seqlen, d_model)

        # ── 4. Generate refined output ──────────────────────────
        refined_logits = F.linear(refined_states, self.refine_weight)  # (batch, seqlen, vocab_size)

        # ── 5. Confidence scoring ───────────────────────────────
        conf = self.confidence(reflection_summary)  # (batch, 1)

        return {
            "refined_logits": refined_logits,
            "critique_logits": critique_logits,
            "confidence": conf,
            "reflection_summary": reflection_summary,
        }

    def compute_reflection_loss(
        self,
        original_logits: torch.Tensor,
        refined_logits: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute reflection training loss.

        Loss = CE(refined_logits, labels) + β * (1 - confidence)
        The confidence term encourages the model to be honest about
        when reflection actually helps.

        Args:
            original_logits: (batch, seqlen, vocab_size)
            refined_logits: (batch, seqlen, vocab_size)
            labels: (batch, seqlen) target tokens
            confidence: (batch, 1) predicted improvement confidence

        Returns:
            (loss, aux_info)
        """
        shift_refined = refined_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Main refinement loss
        refine_loss = F.cross_entropy(
            shift_refined.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Confidence calibration: encourage confidence when refined is better
        # If refined output matches labels better, confidence should be high
        with torch.no_grad():
            shift_original = original_logits[..., :-1, :].contiguous()
            original_loss = F.cross_entropy(
                shift_original.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='none',
            )
            refined_loss_per_token = F.cross_entropy(
                shift_refined.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='none',
            )
            # Did refinement help? (1 if yes, 0 if no)
            # Mask ignored positions (-100) to avoid biasing the improvement rate
            valid_mask = (shift_labels.view(-1) != -100).float()
            improved = (refined_loss_per_token < original_loss).float() * valid_mask
            valid_count = valid_mask.view(bs, -1).sum(dim=1).clamp(min=1)
            improved_per_sample = improved.view(bs, -1).sum(dim=1) / valid_count

        # Confidence loss: push confidence toward actual per-sample improvement
        conf_loss = F.mse_loss(confidence.squeeze(-1), improved_per_sample)

        total_loss = refine_loss + 0.1 * conf_loss

        aux_info = {
            "refine_loss": refine_loss.item(),
            "conf_loss": conf_loss.item(),
            "avg_confidence": confidence.mean().item(),
            "improvement_rate": improved_per_sample.mean().item(),
        }

        return total_loss, aux_info


# ═══════════════════════════════════════════════════════════════════════
# Self-Reflection Generator (Inference)
# ═══════════════════════════════════════════════════════════════════════

class SelfReflectiveGenerator:
    """
    Wrapper that adds self-reflection to the generation pipeline.

    Usage:
        reflector = SelfReflectiveGenerator(model, reflection_module, tokenizer)
        output = reflector.generate_with_reflection(prompt_ids)
    """

    def __init__(self, model, reflection_module: ReflectionModule, tokenizer):
        self.model = model
        self.reflection = reflection_module
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate_with_reflection(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        reflection_threshold: float = 0.3,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate with self-reflection: generate → critique → refine.

        1. Generate initial response
        2. Run reflection module on the output
        3. If confidence > threshold, use refined output
        4. Otherwise, keep original

        Args:
            input_ids: Prompt token IDs (batch, seqlen)
            max_new_tokens: Max tokens for initial generation
            temperature, top_k, top_p: Generation parameters
            reflection_threshold: Min confidence to use refined output

        Returns:
            (final_token_ids, reflection_info)
        """
        # ── Step 1: Initial generation ─────────────────────────
        initial_output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Get hidden states for the generated portion
        outputs = self.model.model(input_ids=initial_output, use_cache=False)
        hidden_states = outputs["last_hidden_state"]  # (batch, seqlen, d_model)

        # ── Step 2: Reflection ──────────────────────────────────
        # For efficiency: only reflect on the generated part
        gen_start = input_ids.shape[1]
        gen_hidden = hidden_states[:, gen_start:, :] if hidden_states.shape[1] > gen_start else hidden_states
        gen_tokens = initial_output[:, gen_start:]

        reflection_output = self.reflection(
            hidden_states=gen_hidden,
            generated_tokens=gen_tokens,
        )

        confidence = reflection_output["confidence"].mean().item()
        refined_logits = reflection_output["refined_logits"]

        # ── Step 3: Decide whether to use refinement ───────────
        use_refined = confidence > reflection_threshold

        if use_refined:
            # Sample from refined logits
            refined_ids = self._sample_from_logits(
                refined_logits, temperature, top_k, top_p
            )
            # Concatenate prompt + refined generation
            final_output = torch.cat([input_ids, refined_ids], dim=-1)
        else:
            final_output = initial_output

        reflection_info = {
            "used_reflection": use_refined,
            "confidence": confidence,
            "threshold": reflection_threshold,
            "initial_length": initial_output.shape[1] - input_ids.shape[1],
            "final_length": final_output.shape[1] - input_ids.shape[1],
        }

        return final_output, reflection_info

    def _sample_from_logits(
        self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float
    ) -> torch.Tensor:
        """Sample tokens from logits tensor."""
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            top_k_vals, _ = torch.topk(logits, k, dim=-1)
            logits = logits.masked_fill(logits < top_k_vals[:, :, -1:], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs.view(-1, logits.size(-1)), num_samples=1).view(logits.shape[0], -1)


# ═══════════════════════════════════════════════════════════════════════
# Integration with MamformerForCausalLM
# ═══════════════════════════════════════════════════════════════════════

def add_reflection_to_model(model, vocab_size: int, max_reflection_tokens: int = 128):
    """
    Attach a ReflectionModule to a MamformerForCausalLM model.

    Args:
        model: MamformerForCausalLM instance
        vocab_size: Vocabulary size
        max_reflection_tokens: Max tokens for critique

    Returns:
        model with self.reflection attribute
    """
    embedding_weight = model.model.embed_tokens.weight if model.config.tie_word_embeddings else None
    model.reflection = ReflectionModule(
        d_model=model.config.d_model,
        vocab_size=vocab_size,
        max_reflection_tokens=max_reflection_tokens,
        embedding_weight=embedding_weight,
    )
    return model
