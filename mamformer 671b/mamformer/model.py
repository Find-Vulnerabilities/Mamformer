"""
Mamformer Model: Full Mamba-2 + Transformer Hybrid LLM (Ultra Edition)
======================================================================
Top-level model classes for the Mamformer architecture.

- MamformerModel: Stack of MamformerBlocks → hidden states
- MamformerForCausalLM: MamformerModel + LM head + optional MTP → logits + loss

Ultra features:
  - DeepSeekMoE: Sparse mixture of experts for massive capacity
  - KDA-Diff: Kernelized differential attention with interleaving
  - MTP: Multi-Token Prediction for denser training signal

─── THINKING / REFLECTION ────────────────────────────────────────────
This file provides token-based multi-path parallel thinking (the primary
mechanism). Two other thinking/reflection systems exist in the codebase:

  1. model.py (THIS FILE): Token-based multi-path parallel thinking
     - ThinkingConfig + control tokens (IDs 3-7)
     - N parallel reasoning paths → summary synthesis → answer
     - Most feature-complete; preferred for new development

  2. chat.py: Language-level reflection via XML tags
     - think/critique modes using <thinking>/<draft>/<critique> tags
     - Prompt-level only — no architecture changes needed

  3. reflection.py: MLP-based ReflectionModule
     - Trainable parameters for critique + refinement
     - Attached via add_reflection_to_model()

See mamformer.chat and mamformer.reflection for the other two systems.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from mamformer.config import MamformerConfig
from mamformer.layers.norm import RMSNorm
from mamformer.layers.hybrid import MamformerBlock
from mamformer.layers.mtp import MultiTokenPredictor
from mamformer.generation import GenerationMixin


def _is_full_attention_checkpointable(layer: "MamformerBlock") -> bool:
    """
    Check if a layer's attention module should be gradient-checkpointed.

    Only FullDiffAttention (O(N²)) layers benefit from checkpointing —
    they materialize large attention matrices that dominate VRAM.
    LinearDiffAttention (O(N)) layers are cheap and skipping them
    avoids unnecessary recomputation overhead.

    For non-KDA-Diff attention (GQA/DSA), checkpoint all attention layers
    since they are all O(N²).
    """
    attn = getattr(layer, 'attention', None)
    if attn is None:
        return False

    # KDA-Diff: only checkpoint FullDiffAttention layers (every 4th)
    if hasattr(attn, '_is_full_attention_layer'):
        return attn._is_full_attention_layer()

    # DSA / GQA: checkpoint all attention layers (all are O(N²))
    return True


class MamformerModel(nn.Module):
    """
    The core Mamformer transformer model — a stack of hybrid blocks.

    Takes token IDs and returns hidden states. No LM head.
    This is the base model that can be used for fine-tuning,
    feature extraction, or as a backbone for other tasks.

    Args:
        config: MamformerConfig instance with model hyperparameters
    """

    def __init__(self, config: MamformerConfig) -> None:
        super().__init__()

        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.d_model
        )

        # Resolve layer types for interleaving
        if config.interleave.enabled:
            layer_types = config.interleave.resolve_layer_types(config.n_layers)
        else:
            # All layers are fusion (has_attention=True, has_ssm=True)
            layer_types = [
                {"has_attention": True, "has_ssm": True, "is_fusion": True}
                for _ in range(config.n_layers)
            ]

        # Stack of Mamformer blocks with mode-specific configuration.
        # Track attention-layer index separately for KDA-Diff's fixed interleaving.
        attn_layer_counter = 0
        self.layers = nn.ModuleList()
        for i, lt in enumerate(layer_types):
            is_attn = lt["has_attention"]
            layer = MamformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                head_dim=config.head_dim,
                d_ff=config.d_ff,
                d_state=config.mamba.d_state,
                d_conv=config.mamba.d_conv,
                mamba_expand=config.mamba.expand,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope.theta,
                dropout=config.dropout,
                rms_norm_eps=config.rms_norm_eps,
                sliding_window=config.sliding_window if config.use_sliding_window else 0,
                # Mode control
                has_attention=lt["has_attention"],
                has_ssm=lt["has_ssm"],
                layer_idx=i,
                # DSA options
                use_dsa=config.dsa.enabled and not config.kda_diff.enabled,
                dsa_lambda_init=config.dsa.lambda_init,
                dsa_state_injection=config.dsa.use_state_injection,
                # KDA-Diff options
                use_kda_diff=config.kda_diff.enabled,
                kda_linear_ratio=config.kda_diff.linear_ratio,
                kda_kernel_dim=config.kda_diff.kernel_dim,
                kda_latent_dim=config.kda_diff.latent_dim,
                kda_use_dynamic_ratio=config.kda_diff.use_dynamic_ratio,
                # YaRN options
                rope_use_yarn=config.rope.use_yarn,
                rope_yarn_scale=config.rope.yarn_scale,
                rope_yarn_original_max_seq_len=config.rope.yarn_original_max_seq_len,
                # MoE options
                use_moe=config.moe.enabled and not config.st_moe.enabled,
                moe_n_shared=config.moe.n_shared_experts,
                moe_n_routed=config.moe.n_routed_experts,
                moe_top_k=config.moe.top_k,
                moe_shared_dim=config.moe.shared_expert_intermediate_dim,
                moe_routed_dim=config.moe.routed_expert_intermediate_dim,
                moe_aux_loss_free=config.moe.aux_loss_free,
                moe_bias_speed=config.moe.bias_update_speed,
                # ST-MoE options
                use_st_moe=config.st_moe.enabled,
                st_moe_lambda_init=config.st_moe.lambda_init,
                st_moe_lambda_max=config.st_moe.lambda_max,
                st_moe_learnable_lambda=config.st_moe.learnable_lambda,
                st_moe_use_balance_lock=config.st_moe.use_balance_lock,
                st_moe_balance_lock_threshold=config.st_moe.balance_lock_threshold,
                # Communicative MoE options
                use_communicative_moe=config.communicative_moe.enabled,
                comm_moe_n_heads=config.communicative_moe.n_comm_heads,
                comm_moe_depth=config.communicative_moe.comm_depth,
                comm_moe_dropout=config.communicative_moe.comm_dropout,
            )
            self.layers.append(layer)
            # Set KDA-Diff attention-layer index for correct fixed interleaving
            if is_attn and config.kda_diff.enabled:
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'set_layer_idx'):
                    layer.attention.set_layer_idx(i, attn_layer_counter)
                attn_layer_counter += 1

        # Final normalization
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Gradient checkpointing flag (set externally during training)
        self.gradient_checkpointing = False

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following LLM best practices."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache: Optional[List[dict]] = None,
    ) -> dict:
        """
        Forward pass through the Mamformer model.

        Args:
            input_ids: Token indices (batch, seqlen)
            attention_mask: Optional attention mask. Shape (batch, seqlen)
                           or (batch, 1, seqlen, seqlen). 1 = attend, 0 = mask.
                           If None, causal masking is used automatically.
            use_cache: If True, returns per-layer caches for autoregressive generation
            cache: Optional list of per-layer cache dicts from previous step

        Returns:
            dict with keys:
                - "last_hidden_state": (batch, seqlen, d_model)
                - "cache": List of per-layer caches (if use_cache=True)
                - "moe_aux_info": List of MoE aux info dicts (if MoE enabled)
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)  # (batch, seqlen, d_model)

        # Process attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Convert (batch, seqlen) to (batch, 1, 1, seqlen)
                attention_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                # Convert to additive mask: 0 → -inf, 1 → 0
                attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Initialize cache
        new_caches: List[dict] = [] if use_cache else []
        per_layer_cache = cache if cache is not None else [None] * len(self.layers)
        moe_aux_info_list: List[dict] = []

        # Cross-layer SSM state tracking
        # When an SSM-only layer produces h_states, we hold them here
        # and inject them into the next attention-only layer.
        pending_ssm_h: Optional[torch.Tensor] = None

        # Stack through layers
        for idx, layer in enumerate(self.layers):
            layer_cache = per_layer_cache[idx] if idx < len(per_layer_cache) else None

            # Prepare cross-layer SSM state for attention-only layers
            layer_kwargs: dict = {}
            if layer.has_attention and not layer.has_ssm and pending_ssm_h is not None:
                layer_kwargs["ssm_h_states"] = pending_ssm_h

            # Only checkpoint FullDiffAttention layers (O(N²), every 4th attn layer).
            # LinearDiffAttention (O(N)) and SSM layers are cheap and don't need
            # checkpointing — checkpointing SSM layers also breaks cross-layer
            # SSM state injection.
            _needs_checkpoint = (
                self.gradient_checkpointing
                and self.training
                and layer.has_attention
                and _is_full_attention_checkpointable(layer)
            )
            if _needs_checkpoint:
                def make_custom_forward(layer, kwargs):
                    def custom_forward(hidden_states, attention_mask):
                        outputs = layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            use_cache=False,
                            cache=None,
                            **kwargs,
                        )
                        return outputs[0]
                    return custom_forward

                hidden_states = activation_checkpoint(
                    make_custom_forward(layer, layer_kwargs),
                    hidden_states,
                    attention_mask,
                    use_reentrant=False,
                )
                new_cache_entry = None
            else:
                hidden_states, new_cache_entry = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    cache=layer_cache,
                    **layer_kwargs,
                )

            # Update cross-layer SSM state from SSM-only layers.
            # Works during both training (use_cache=False) and inference —
            # SSM layers always return h_states in their cache dict.
            if not layer.has_attention and layer.has_ssm:
                if new_cache_entry is not None and "ssm_h_states" in new_cache_entry:
                    pending_ssm_h = new_cache_entry["ssm_h_states"]

            # Clear SSM state after it's been consumed by an attention-only
            # layer to prevent stale state reuse across consecutive attention layers.
            if layer.has_attention and not layer.has_ssm and layer_kwargs.get("ssm_h_states") is not None:
                pending_ssm_h = None

            if use_cache:
                new_caches.append(new_cache_entry)

            # Collect MoE aux info
            if new_cache_entry is not None and "moe_aux_info" in new_cache_entry:
                moe_aux_info_list.append(new_cache_entry["moe_aux_info"])

        # Final norm
        hidden_states = self.norm(hidden_states)

        output = {"last_hidden_state": hidden_states}
        if use_cache:
            output["cache"] = new_caches
        if moe_aux_info_list:
            output["moe_aux_info"] = moe_aux_info_list

        return output

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory-efficient training."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def print_parameter_summary(self) -> None:
        """Print detailed parameter count breakdown by component."""
        total = self.num_parameters()
        embedding_params = self.embed_tokens.weight.numel()
        layer_params = sum(
            p.numel() for layer in self.layers for p in layer.parameters()
        )
        norm_params = sum(p.numel() for p in self.norm.parameters())

        # Per-component breakdown
        attn_params = sum(
            p.numel()
            for layer in self.layers
            if layer.has_attention and layer.attention is not None
            for name, p in layer.attention.named_parameters()
        )
        ssm_params = sum(
            p.numel()
            for layer in self.layers
            if layer.has_ssm and layer.ssm is not None
            for name, p in layer.ssm.named_parameters()
        )
        ffn_params = sum(
            p.numel()
            for layer in self.layers
            for name, p in layer.ffn.named_parameters()
        )

        n_fusion = sum(1 for layer in self.layers if getattr(layer, 'is_fusion', layer.has_attention and layer.has_ssm))
        n_attn_only = sum(1 for layer in self.layers if layer.has_attention and not layer.has_ssm)
        n_ssm_only = sum(1 for layer in self.layers if not layer.has_attention and layer.has_ssm)

        sep = "=" * 55
        print(sep)
        print("  Mamformer Model Parameter Summary")
        print(sep)
        print(f"  Embedding:     {embedding_params:>15,}")
        layer_desc_parts = []
        if n_fusion > 0:
            layer_desc_parts.append(f"{n_fusion} fusion")
        if n_attn_only > 0:
            layer_desc_parts.append(f"{n_attn_only} attn-only")
        if n_ssm_only > 0:
            layer_desc_parts.append(f"{n_ssm_only} SSM-only")
        print(f"  Layers: {', '.join(layer_desc_parts)} (x{len(self.layers)} total)")
        print(f"  Per-layer:")
        print(f"    Attention:   {attn_params // max(1, n_fusion + n_attn_only):>15,}")
        print(f"    SSM:         {ssm_params // max(1, n_fusion + n_ssm_only):>15,}")
        print(f"    FFN:         {ffn_params // len(self.layers):>15,}")
        print(f"    Total/layer: {layer_params // len(self.layers):>15,}")
        print(f"  Layers (x{len(self.layers)}):  {layer_params:>15,}")
        print(f"  Final Norm:    {norm_params:>15,}")
        print(sep)
        print(f"  Total:         {total:>15,}")
        print(f"  Total (B):     {total / 1e9:>14.2f}B")
        print(sep)


class MamformerForCausalLM(GenerationMixin, nn.Module):
    """
    Mamformer model with a language modeling head (causal LM).

    Inherits from GenerationMixin for beam search, streaming generation,
    and repetition penalty helpers. The model's own generate() takes
    precedence for the primary API.

    Architecture:
        Token IDs -> MamformerModel -> RMSNorm -> lm_head -> logits
        (Optional) MTP heads for multi-token prediction

    The lm_head weight is tied with the token embedding weight
    when config.tie_word_embeddings is True (saving ~524M params for 7B).

    When MTP is enabled, additional prediction heads compute logits
    for future tokens (t+1, t+2, ...) during training.

    Args:
        config: MamformerConfig instance
    """

    def __init__(self, config: MamformerConfig) -> None:
        super().__init__()

        self.config = config
        self.model = MamformerModel(config)

        # LM head — shares weights with embedding if tie_word_embeddings=True
        if config.tie_word_embeddings:
            self.lm_head = None  # Use model.embed_tokens.weight directly
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.initializer_range)

        # Multi-Token Prediction heads
        self.mtp = None
        if config.mtp.enabled:
            mtp_dim = config.mtp.mtp_d_model if config.mtp.mtp_d_model > 0 else config.d_model
            self.mtp = MultiTokenPredictor(
                d_model=config.d_model,
                vocab_size=config.vocab_size,
                depth=config.mtp.depth,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                head_dim=config.head_dim,
                d_ff=config.d_ff // 8,  # Smaller FFN for MTP blocks
                d_state=config.mamba.d_state,
                d_conv=config.mamba.d_conv,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope.theta,
                dropout=config.dropout,
                rms_norm_eps=config.rms_norm_eps,
                embedding_weight=self.model.embed_tokens.weight if config.tie_word_embeddings else None,
            )

        # ── Track if embeddings have been extended for thinking tokens ──
        self._embeddings_extended: bool = False

    def _extend_embeddings(self, new_vocab_size: int) -> None:
        """
        Extend token embeddings and LM head to accommodate new tokens
        (e.g., thinking control tokens). Initializes new rows with the
        mean of existing embeddings so they start with reasonable values.

        This is a minimal structural change — no retraining required.

        Args:
            new_vocab_size: Target vocabulary size (must be > current)
        """
        current_vocab = self.config.vocab_size
        if new_vocab_size <= current_vocab:
            return  # Already large enough

        n_new = new_vocab_size - current_vocab
        d_model = self.config.d_model

        # ── Extend embed_tokens ────────────────────────────────────
        old_embed = self.model.embed_tokens.weight.data  # (old_vocab, d_model)
        old_mean = old_embed.mean(dim=0, keepdim=True)
        new_embed_rows = old_mean.repeat(n_new, 1) + torch.randn(
            n_new, d_model, device=old_embed.device, dtype=old_embed.dtype
        ) * 0.02
        new_embed = torch.cat([old_embed, new_embed_rows], dim=0)
        self.model.embed_tokens = nn.Embedding(new_vocab_size, d_model)
        self.model.embed_tokens.weight.data.copy_(new_embed)

        # ── Extend lm_head if not tied ─────────────────────────────
        if self.lm_head is not None:
            old_head = self.lm_head.weight.data  # (old_vocab, d_model)
            new_head_rows = old_mean.repeat(n_new, 1) + torch.randn(
                n_new, d_model, device=old_head.device, dtype=old_head.dtype
            ) * 0.02
            new_head = torch.cat([old_head, new_head_rows], dim=0)
            self.lm_head = nn.Linear(d_model, new_vocab_size, bias=False)
            self.lm_head.weight.data.copy_(new_head)

        # ── Update config ──────────────────────────────────────────
        self.config.vocab_size = new_vocab_size
        self._embeddings_extended = True

    def _ensure_thinking_tokens(self) -> None:
        """
        Ensure the model's embedding matrix can represent thinking control tokens.
        Extends embeddings if vocab_size < summary_start_token_id + 1.
        """
        from mamformer.tokenizer import MamformerTokenizer
        min_vocab = MamformerTokenizer.SUMMARY_START_ID + 1  # token IDs 0-7
        if self.config.vocab_size < min_vocab:
            self._extend_embeddings(min_vocab)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache: Optional[List[dict]] = None,
    ) -> dict:
        """
        Forward pass with optional loss computation and MTP.

        Args:
            input_ids: Token indices (batch, seqlen)
            attention_mask: Optional attention mask
            labels: Target token indices for next-token prediction loss.
                   Shape (batch, seqlen). Positions with value -100 are ignored.
            use_cache: If True, returns KV/SSM caches
            cache: Optional list of per-layer caches

        Returns:
            dict with keys:
                - "logits": (batch, seqlen, vocab_size)
                - "loss": scalar cross-entropy loss (if labels provided)
                - "cache": List of per-layer caches (if use_cache=True)
                - "mtp_logits": List of MTP logits (if MTP enabled)
                - "mtp_loss": MTP auxiliary loss (if MTP enabled + labels)
                - "moe_aux_info": MoE routing statistics (if MoE enabled)
        """
        # Get hidden states from backbone
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            cache=cache,
        )

        hidden_states = outputs["last_hidden_state"]  # (batch, seqlen, d_model)

        # Compute logits
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied embeddings: reuse embedding weight
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)

        result = {"logits": logits}
        if use_cache:
            result["cache"] = outputs.get("cache")
        if "moe_aux_info" in outputs:
            result["moe_aux_info"] = outputs["moe_aux_info"]

        # Compute main loss if labels are provided
        main_loss = None
        if labels is not None:
            # Shift: predict next token
            # logits:  (batch, seqlen, vocab_size) → (batch, seqlen-1, vocab_size)
            # labels:  (batch, seqlen) → (batch, seqlen-1)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            main_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # ── Multi-Token Prediction ──────────────────────────────────
        mtp_loss = None
        mtp_logits_list = None
        if self.mtp is not None and self.training:
            mtp_logits_list, mtp_loss = self.mtp(
                hidden_states=hidden_states,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            if mtp_logits_list is not None:
                result["mtp_logits"] = mtp_logits_list

        # Combine losses: L = L_main + α * L_mtp
        if main_loss is not None:
            if mtp_loss is not None:
                total_loss = main_loss + self.config.mtp.loss_weight * mtp_loss
                result["loss"] = total_loss
                result["main_loss"] = main_loss.detach()
                result["mtp_loss"] = mtp_loss.detach()
            else:
                result["loss"] = main_loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        thinking_config: Optional[object] = None,
    ) -> Union[torch.Tensor, dict]:
        """
        Autoregressive text generation with config-driven defaults.

        Supports toggleable thinking mode: when thinking_config is
        provided and enabled, the model first generates internal
        reasoning tokens before producing the final answer.

        If parameters are None, defaults are taken from config.generation.
        This allows each model tier to have appropriate default generation settings.

        Args:
            input_ids: Prompt token indices (batch, seqlen)
            max_new_tokens: Max tokens to generate (default: from config.generation)
            temperature: Sampling temperature (default: from config.generation)
            top_k: Top-k filter (default: from config.generation)
            top_p: Nucleus threshold (default: from config.generation)
            eos_token_id: Stop generation when this token is produced
            pad_token_id: Token ID for padding
            thinking_config: Optional ThinkingConfig for toggleable reasoning mode.
                           When enabled, returns a dict with separated thinking/answer tokens.
                           When None/disabled, returns a tensor (backward compatible).

        Returns:
            - Tensor (batch, prompt_len + generated_len) when thinking is disabled
            - Dict with keys when thinking is enabled:
                "generated_ids": full sequence with thinking markers
                "thinking_ids": thinking/reasoning tokens only
                "answer_ids": answer tokens only
                "thinking_config": snapshot of the thinking config used
        """
        # ── Determine if thinking mode is active ──────────────────
        use_thinking = (
            thinking_config is not None
            and getattr(thinking_config, 'enabled', False)
            and getattr(thinking_config, 'is_active', False)
        )

        if use_thinking:
            # Ensure model embeddings can represent thinking tokens
            self._ensure_thinking_tokens()
            return self._generate_with_thinking(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                thinking_config=thinking_config,
            )

        # ── Standard generation (original path) ──────────────────
        gen_cfg = self.config.generation
        if max_new_tokens is None:
            max_new_tokens = gen_cfg.max_output_tokens
        if temperature is None:
            temperature = gen_cfg.default_temperature
        if top_k is None:
            top_k = gen_cfg.default_top_k
        if top_p is None:
            top_p = gen_cfg.default_top_p
        batch_size = input_ids.shape[0]
        device = input_ids.device
        rep_penalty = gen_cfg.repetition_penalty
        generated = input_ids.clone()
        cache = None
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            if not unfinished.any():
                break

            # Forward pass with cache (only last token when cache available)
            if cache is not None:
                current_input = generated[:, -1:]
            else:
                current_input = generated

            outputs = self.forward(
                input_ids=current_input,
                use_cache=True,
                cache=cache,
            )

            logits = outputs["logits"][:, -1, :]  # (batch, vocab_size)
            cache = outputs.get("cache")

            # Repetition penalty
            if rep_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        if logits[i, token_id] > 0:
                            logits[i, token_id] /= rep_penalty
                        else:
                            logits[i, token_id] *= rep_penalty

            # Sample next token
            next_token = self._sample_from_logits(
                logits, temperature, top_k, top_p
            )

            # Mask finished sequences
            next_token = next_token.masked_fill(
                ~unfinished.unsqueeze(-1),
                pad_token_id if pad_token_id is not None else 0,
            )

            generated = torch.cat([generated, next_token], dim=-1)

            # Track which sequences have hit EOS
            if eos_token_id is not None:
                unfinished = unfinished & (next_token.squeeze(-1) != eos_token_id)

        return generated

    @torch.no_grad()
    def _expand_cache_batch(self, cache: Optional[List[dict]], n: int) -> Optional[List[dict]]:
        """Expand cache batch dimension from 1 to n for parallel path generation."""
        if cache is None:
            return None
        expanded = []
        for layer_cache in cache:
            if layer_cache is None:
                expanded.append(None)
                continue
            layer_exp = {}
            for key, val in layer_cache.items():
                if isinstance(val, torch.Tensor) and val.shape[0] == 1:
                    layer_exp[key] = val.expand(n, *val.shape[1:]).clone()
                elif isinstance(val, dict):
                    layer_exp[key] = {
                        k: v.expand(n, *v.shape[1:]).clone() if isinstance(v, torch.Tensor) and v.shape[0] == 1 else v
                        for k, v in val.items()
                    }
                else:
                    layer_exp[key] = val
            expanded.append(layer_exp)
        return expanded

    def _generate_thinking_paths_batched(
        self,
        prompt_cache: List[dict],
        base_sequence: torch.Tensor,
        think_start_id: int,
        think_end_id: int,
        think_budget: int,
        temperature: float,
        top_k: int,
        top_p: float,
        rep_penalty: float,
        num_paths: int,
        device: torch.device,
    ) -> tuple[List[List[int]], torch.Tensor]:
        """
        Generate N independent thinking paths simultaneously (batched).

        All paths branch from the same prompt KV cache (expanded to batch=N).
        Paths are isolated via cloned caches — they don't see each other.
        Generation is step-interleaved: token 1 for all paths, token 2 for all, etc.

        Returns:
            (all_path_tokens, full_sequence_with_all_paths)
        """
        # Expand prompt cache from batch=1 to batch=num_paths
        path_caches = self._expand_cache_batch(prompt_cache, num_paths)

        # Track per-path state
        path_tokens: List[List[int]] = [[] for _ in range(num_paths)]
        path_done = [False] * num_paths
        path_hit_think_end = [False] * num_paths
        n_done = 0

        # Each path gets its own sequence starting from base, but we batch
        # the generation step-by-step
        # Start: inject <|think_start|> for all paths
        think_start_tensor = torch.full((num_paths, 1), think_start_id, dtype=torch.long, device=device)

        # Step-interleaved generation: one token per active path per step
        for step in range(think_budget):
            if n_done >= num_paths:
                break

            # Forward all paths together
            if step == 0:
                current_input = think_start_tensor
            else:
                current_input = torch.tensor(
                    [[path_tokens[p][-1]] for p in range(num_paths)],
                    dtype=torch.long, device=device,
                )

            outputs = self.forward(
                input_ids=current_input,
                use_cache=True,
                cache=path_caches,
            )
            logits = outputs["logits"][:, -1, :]  # (num_paths, vocab)
            path_caches = outputs.get("cache")

            # Repetition penalty per path
            if rep_penalty != 1.0:
                for p in range(num_paths):
                    if path_done[p]:
                        continue
                    seen = set(path_tokens[p])
                    for tid in seen:
                        if logits[p, tid] > 0:
                            logits[p, tid] /= rep_penalty
                        else:
                            logits[p, tid] *= rep_penalty

            # Sample one token for each path
            next_tokens = self._sample_from_logits(logits, temperature, top_k, top_p)  # (N, 1)

            for p in range(num_paths):
                if path_done[p]:
                    continue
                tid = next_tokens[p, 0].item()
                path_tokens[p].append(tid)

                if tid == think_end_id:
                    path_hit_think_end[p] = True
                    path_done[p] = True
                    n_done += 1
                elif step >= think_budget - 1:
                    # Budget exceeded for all remaining
                    path_done[p] = True
                    n_done += 1

        # Budget forcing: inject think_end for paths that didn't emit it
        for p in range(num_paths):
            if not path_hit_think_end[p]:
                path_tokens[p].append(think_end_id)

        # Build full sequence: prompt + [path_1 + path_2 + ... + path_N]
        full_seq = base_sequence.clone()
        for p in range(num_paths):
            # Add <|think_start|> + path tokens
            full_seq = torch.cat([
                full_seq,
                torch.tensor([[think_start_id]], dtype=torch.long, device=device),
                torch.tensor([path_tokens[p]], dtype=torch.long, device=device),
            ], dim=-1)

        return path_tokens, full_seq

    @torch.no_grad()
    def _generate_with_thinking(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        top_k: Optional[int],
        top_p: Optional[float],
        eos_token_id: Optional[int],
        pad_token_id: Optional[int],
        thinking_config: object,
    ) -> dict:
        """
        Multi-path parallel thinking with batched generation.

        Phases:
          1. Prompt: forward, save KV cache
          2. Paths: N paths generated simultaneously (batched step-interleaved)
          3. Summary: synthesize across all paths
          4. Answer: final answer

        Paths are generated batched — all paths advance one token per step,
        sharing a single forward pass. This is ~N times faster than sequential.
        """
        from mamformer.thinking import MultiPathController

        gen_cfg = self.config.generation
        if max_new_tokens is None:
            max_new_tokens = gen_cfg.max_output_tokens
        if temperature is None:
            temperature = gen_cfg.default_temperature
        if top_k is None:
            top_k = gen_cfg.default_top_k
        if top_p is None:
            top_p = gen_cfg.default_top_p

        batch_size = input_ids.shape[0]
        device = input_ids.device
        rep_penalty = gen_cfg.repetition_penalty
        think_budget = thinking_config.effective_budget
        summary_budget = thinking_config.effective_summary_budget
        num_paths = thinking_config.effective_num_paths
        think_start_id = thinking_config.think_start_token_id
        think_end_id = thinking_config.think_end_token_id
        summary_start_id = thinking_config.summary_start_token_id
        answer_start_id = thinking_config.answer_start_token_id

        controller = MultiPathController(thinking_config)

        # ── Phase 1: Encode prompt ───────────────────────────────
        generated = input_ids.clone()
        outputs = self.forward(input_ids=generated, use_cache=True, cache=None)
        prompt_cache = outputs.get("cache")

        # ── Phase 2: Batched parallel path generation ─────────────
        all_path_tokens, generated = self._generate_thinking_paths_batched(
            prompt_cache=prompt_cache,
            base_sequence=generated,
            think_start_id=think_start_id,
            think_end_id=think_end_id,
            think_budget=think_budget,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rep_penalty=rep_penalty,
            num_paths=num_paths,
            device=device,
        )

        total_think_tokens = sum(len(t) for t in all_path_tokens)
        for p in range(num_paths):
            controller.start_path(p)
            for tid in all_path_tokens[p]:
                controller.record_path_token(p, tid)

        # ── Phase 3: Summary ─────────────────────────────────────
        summary_start = torch.full((batch_size, 1), summary_start_id, dtype=torch.long, device=device)
        generated = torch.cat([generated, summary_start], dim=-1)
        controller.start_summary()

        # Forward through all path tokens to build context, then generate summary
        # Re-forward the last portion to get cache for summary
        recent = generated[:, -(num_paths * 2 + 1):] if generated.shape[1] > num_paths * 2 else generated
        outputs = self.forward(input_ids=recent, use_cache=True, cache=prompt_cache)
        cache = outputs.get("cache")

        for _ in range(summary_budget):
            current_input = generated[:, -1:]
            outputs = self.forward(input_ids=current_input, use_cache=True, cache=cache)
            logits = outputs["logits"][:, -1, :]
            cache = outputs.get("cache")

            if rep_penalty != 1.0:
                for i in range(batch_size):
                    for tid in set(generated[i].tolist()):
                        if logits[i, tid] > 0:
                            logits[i, tid] /= rep_penalty
                        else:
                            logits[i, tid] *= rep_penalty

            next_token = self._sample_from_logits(logits, temperature, top_k, top_p)
            token_id = next_token[0, 0].item()
            event = controller.record_summary_token(token_id)
            generated = torch.cat([generated, next_token], dim=-1)
            if event == "end_summary":
                break

        # ── Phase 4: Answer ──────────────────────────────────────
        answer_start = torch.full((batch_size, 1), answer_start_id, dtype=torch.long, device=device)
        generated = torch.cat([generated, answer_start], dim=-1)
        controller.start_answer()

        marker_overhead = num_paths * 2 + 3  # think_start/end per path + summary + answer
        remaining = max_new_tokens - total_think_tokens - summary_budget - marker_overhead
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

        for _ in range(max(remaining, 1)):
            if not unfinished.any():
                break
            current_input = generated[:, -1:]
            outputs = self.forward(input_ids=current_input, use_cache=True, cache=cache)
            logits = outputs["logits"][:, -1, :]
            cache = outputs.get("cache")

            if rep_penalty != 1.0:
                for i in range(batch_size):
                    for tid in set(generated[i].tolist()):
                        if logits[i, tid] > 0:
                            logits[i, tid] /= rep_penalty
                        else:
                            logits[i, tid] *= rep_penalty

            next_token = self._sample_from_logits(logits, temperature, top_k, top_p)
            token_id = next_token[0, 0].item()
            event = controller.record_answer_token(token_id)
            next_token = next_token.masked_fill(~unfinished.unsqueeze(-1), pad_token_id if pad_token_id is not None else 0)
            generated = torch.cat([generated, next_token], dim=-1)
            if eos_token_id is not None:
                unfinished = unfinished & (next_token.squeeze(-1) != eos_token_id)
            if event == "end_answer":
                break

        # ── Finalize ─────────────────────────────────────────────
        info = controller.finalize()
        all_paths_tensors = [
            torch.tensor([tokens], dtype=torch.long, device=device)
            for tokens in info["all_paths"]
        ]
        summary_tensor = torch.tensor([info["summary_tokens"]], dtype=torch.long, device=device)
        answer_tensor = torch.tensor([info["answer_tokens"]], dtype=torch.long, device=device)

        return {
            "generated_ids": generated,
            "all_paths": all_paths_tensors,
            "path_counts": info["path_counts"],
            "summary_ids": summary_tensor,
            "answer_ids": answer_tensor,
            "total_think_tokens": info["total_think_tokens"],
            "num_paths": num_paths,
            "budget_forced": any(info["path_budget_forced"]),
        }

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """
        Sample a single token from logits with temperature, top-k, top-p.
        Delegates to the shared sampling utility.

        Args:
            logits: (batch, vocab_size)
            temperature: Sampling temperature (>0 for sampling, <=0 for greedy)
            top_k: Top-k filter (>0 to enable)
            top_p: Nucleus threshold (<1.0 to enable)

        Returns:
            (batch, 1) token indices
        """
        from mamformer.sampling import sample_one_token
        return sample_one_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)

    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities. Used by GRPO training.

        Returns average log-probability over non-masked response tokens.
        If labels is None, returns log-probs for all tokens (for generation scoring).

        Supports S-GRPO sparse token sampling via token_mask.

        Args:
            input_ids: Token indices (batch, seqlen)
            labels: Target labels, -100 for ignored positions
            attention_mask: Optional attention mask
            token_mask: Optional (batch, seqlen-1) bool mask for S-GRPO.
                       True = include this token in the average.
                       When None, all response tokens are included.

        Returns:
            If labels provided: (batch,) average log-prob per response token
            If no labels: (batch, seqlen-1) token-level log-probs
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs["logits"]  # (batch, seqlen, vocab_size)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()  # (batch, seqlen-1, vocab)
        shift_input_ids = input_ids[:, 1:].contiguous()  # (batch, seqlen-1)

        # Token-level log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)  # (batch, seqlen-1, vocab)
        # Safe gather: replace -100 (ignore_index) with 0 for index lookup,
        # then zero out those positions in the result to avoid token-0 leakage
        safe_indices = shift_input_ids.clamp(min=0)
        token_log_probs = log_probs.gather(
            dim=-1, index=safe_indices.unsqueeze(-1)
        ).squeeze(-1)  # (batch, seqlen-1)
        # Zero out ignored positions (shift_input_ids == -100)
        ignore_mask = (shift_input_ids == -100)
        token_log_probs = token_log_probs.masked_fill(ignore_mask, 0.0)

        if labels is not None:
            # Mask: only consider response tokens
            shift_labels = labels[:, 1:].contiguous()  # (batch, seqlen-1)
            response_mask = (shift_labels != -100).float()  # (batch, seqlen-1)

            # Apply S-GRPO token mask if provided
            if token_mask is not None:
                response_mask = response_mask * token_mask.float()

            total_tokens = response_mask.sum(dim=1).clamp(min=1)  # (batch,)
            avg_log_probs = (token_log_probs * response_mask).sum(dim=1) / total_tokens
            return avg_log_probs  # (batch,)

        return token_log_probs  # (batch, seqlen-1)

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def num_parameters_billions(self) -> float:
        """Total parameters in billions."""
        return self.num_parameters() / 1e9

    def print_parameter_summary(self) -> None:
        """Print detailed parameter count breakdown."""
        total = self.num_parameters()

        # Count by component
        embedding_params = self.model.embed_tokens.weight.numel()
        layer_params = sum(
            p.numel() for layer in self.model.layers for p in layer.parameters()
        )
        norm_params = sum(p.numel() for p in self.model.norm.parameters())
        lm_head_params = 0 if self.lm_head is None else sum(
            p.numel() for p in self.lm_head.parameters()
        )
        mtp_params = 0 if self.mtp is None else sum(
            p.numel() for p in self.mtp.parameters()
        )

        sep = "=" * 55
        print(sep)
        print("  Mamformer Model Parameter Summary")
        print(sep)
        print(f"  Embedding:     {embedding_params:>15,}")
        print(f"  Layers (x{self.config.n_layers}): {layer_params:>15,}")
        print(f"    Per-layer:   {layer_params // self.config.n_layers:>15,}")
        print(f"  Final Norm:    {norm_params:>15,}")
        if lm_head_params > 0:
            print(f"  LM Head:       {lm_head_params:>15,}")
        if mtp_params > 0:
            print(f"  MTP Heads:     {mtp_params:>15,}")
        print(sep)
        print(f"  Total:         {total:>15,}")
        print(f"  Total (B):     {total / 1e9:>14.2f}B")
        print(sep)
