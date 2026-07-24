"""
Mamformer Model: Full Mamba-2 + Transformer Hybrid LLM (Ultra Edition)
======================================================================
Top-level model classes for the Mamformer architecture.

- MamformerModel: Stack of MamformerBlocks → hidden states
- MamformerForCausalLM: MamformerModel + LM head + optional MTP → logits + loss

Ultra features:
  - DeepSeekMoE: Sparse mixture of experts for massive capacity (50B+ total, 7B active)
  - DSA: Differential State-Aware Attention for noise-cancelling
  - MTP: Multi-Token Prediction for denser training signal
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from mamformer.config import MamformerConfig
from mamformer.layers.norm import RMSNorm
from mamformer.layers.hybrid import MamformerBlock
from mamformer.layers.mtp import MultiTokenPredictor
from mamformer.generation import GenerationMixin


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

        # Stack of hybrid Mamformer blocks
        self.layers = nn.ModuleList([
            MamformerBlock(
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
            for _ in range(config.n_layers)
        ])

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

        # Stack through layers
        for idx, layer in enumerate(self.layers):
            layer_cache = per_layer_cache[idx] if idx < len(per_layer_cache) else None

            if self.gradient_checkpointing and self.training:
                # Use activation checkpointing to save memory
                # NOTE: gradient checkpointing is incompatible with KV caching.
                # During training, caching is not needed, so we disable it.
                def make_custom_forward(layer):
                    def custom_forward(hidden_states, attention_mask):
                        outputs = layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            use_cache=False,
                            cache=None,
                        )
                        return outputs[0]  # Return only hidden_states
                    return custom_forward

                hidden_states = activation_checkpoint(
                    make_custom_forward(layer),
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
                )

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
            for name, p in layer.attention.named_parameters()
        )
        ssm_params = sum(
            p.numel()
            for layer in self.layers
            for name, p in layer.ssm.named_parameters()
        )
        ffn_params = sum(
            p.numel()
            for layer in self.layers
            for name, p in layer.ffn.named_parameters()
        )

        sep = "=" * 55
        print(sep)
        print("  Mamformer Model Parameter Summary")
        print(sep)
        print(f"  Embedding:     {embedding_params:>15,}")
        print(f"  Per-layer:")
        print(f"    Attention:   {attn_params // len(self.layers):>15,}")
        print(f"    SSM:         {ssm_params // len(self.layers):>15,}")
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
    ) -> torch.Tensor:
        """
        Autoregressive text generation with config-driven defaults.

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

        Returns:
            Generated token IDs (batch, prompt_len + generated_len)
        """
        # Apply config-driven defaults
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
        dtype_min = torch.finfo(torch.float32).min
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

            # Repetition penalty: penalize already-generated tokens
            if rep_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        if logits[i, token_id] > 0:
                            logits[i, token_id] /= rep_penalty
                        else:
                            logits[i, token_id] *= rep_penalty

            # Temperature scaling
            if temperature > 0 and temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                top_k_values, _ = torch.topk(logits, k, dim=-1)
                logits = logits.masked_fill(logits < top_k_values[:, -1:], dtype_min)

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits = logits.masked_fill(mask, dtype_min)

            # Sample or greedy
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                next_token = torch.multinomial(probs, num_samples=1)

            # Mask finished sequences (keep generating pad for uniformity)
            next_token = next_token.masked_fill(~unfinished.unsqueeze(-1), pad_token_id or 0)

            generated = torch.cat([generated, next_token], dim=-1)

            # Track which sequences have hit EOS
            if eos_token_id is not None:
                unfinished = unfinished & (next_token.squeeze(-1) != eos_token_id)

        return generated

    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities. Used by GRPO training.

        Returns average log-probability over non-masked response tokens.
        If labels is None, returns log-probs for all tokens (for generation scoring).

        Args:
            input_ids: Token indices (batch, seqlen)
            labels: Target labels, -100 for ignored positions
            attention_mask: Optional attention mask

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
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_input_ids.unsqueeze(-1).clamp(min=0)
        ).squeeze(-1)  # (batch, seqlen-1)

        if labels is not None:
            # Mask: only consider response tokens
            shift_labels = labels[:, 1:].contiguous()  # (batch, seqlen-1)
            response_mask = (shift_labels != -100).float()  # (batch, seqlen-1)
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
