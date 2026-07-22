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
                use_dsa=config.dsa.enabled,
                dsa_lambda_init=config.dsa.lambda_init,
                dsa_state_injection=config.dsa.use_state_injection,
                # YaRN options
                rope_use_yarn=config.rope.use_yarn,
                rope_yarn_scale=config.rope.yarn_scale,
                rope_yarn_original_max_seq_len=config.rope.yarn_original_max_seq_len,
                # MoE options
                use_moe=config.moe.enabled,
                moe_n_shared=config.moe.n_shared_experts,
                moe_n_routed=config.moe.n_routed_experts,
                moe_top_k=config.moe.top_k,
                moe_shared_dim=config.moe.shared_expert_intermediate_dim,
                moe_routed_dim=config.moe.routed_expert_intermediate_dim,
                moe_aux_loss_free=config.moe.aux_loss_free,
                moe_bias_speed=config.moe.bias_update_speed,
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
                def make_custom_forward(layer):
                    def custom_forward(*inputs):
                        return layer(inputs[0], attention_mask=inputs[1] if len(inputs) > 1 else None)
                    return custom_forward

                hidden_states, _ = activation_checkpoint(
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


class MamformerForCausalLM(nn.Module):
    """
    Mamformer model with a language modeling head (causal LM).

    Architecture:
        Token IDs → MamformerModel → RMSNorm → lm_head → logits
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
        generated = input_ids.clone()
        cache = None

        for _ in range(max_new_tokens):
            # Forward pass with cache
            outputs = self.forward(
                input_ids=generated[:, -1:] if cache is not None else generated,
                use_cache=True,
                cache=cache,
            )

            logits = outputs["logits"][:, -1, :]  # (batch, vocab_size)
            cache = outputs.get("cache")

            # Temperature scaling
            if temperature > 0 and temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < min_top_k, float("-inf"))

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample or greedy
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

        return generated

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
