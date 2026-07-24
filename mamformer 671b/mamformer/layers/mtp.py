"""
Multi-Token Prediction (MTP)
==============================
Implements DeepSeek V3's Multi-Token Prediction mechanism.

MTP trains the model to predict multiple future tokens at each position,
rather than just the immediate next token. This provides:

1. **Denser training signal**: Each position supervises N future tokens,
   giving the model more learning signal per training example.

2. **Better representations**: Predicting further-ahead tokens forces the
   model to build more predictive, planning-oriented representations.

3. **Speculative decoding**: At inference, MTP predictions can be used
   for draft-then-verify speculative decoding, speeding up generation
   without quality loss.

Architecture:
  Given main model hidden states h at position t:

  For depth k = 1..N:
    1. Fuse: h_k = h + emb(token_{t+k-1})
    2. Process: h_k = RMSNorm(h_k)
                 h_k = h_k + Attn(h_k) + SSM(h_k)  [shared Mamformer block]
    3. Output: logits_k = OutputHead(h_k)

  Loss: L = L_main + α * Σ_{k=1}^{N} L_k

At inference, only the main model output is used (MTP heads can optionally
be used for speculative decoding).

Reference:
  "DeepSeek-V3 Technical Report" (DeepSeek-AI, 2024)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamformer.layers.norm import RMSNorm
from mamformer.layers.attention import GroupedQueryAttention
from mamformer.layers.mamba2 import Mamba2Block
from mamformer.layers.ffn import SwiGLUFFN


class MultiTokenPredictor(nn.Module):
    """
    Multi-Token Prediction module.

    Predicts N future tokens from the main model's hidden states.
    Each prediction depth uses a shared small transformer block
    to process the fused hidden state before projecting to logits.

    Args:
        d_model: Main model hidden dimension
        vocab_size: Vocabulary size
        depth: Number of future tokens to predict (N)
        n_heads: Number of attention heads in MTP blocks
        n_kv_heads: Number of KV heads (GQA) in MTP blocks
        head_dim: Dimension per attention head
        d_ff: FFN intermediate dimension in MTP blocks
        d_state: SSM state dimension
        d_conv: SSM convolution kernel size
        max_seq_len: Maximum sequence length
        rope_theta: RoPE base frequency
        dropout: Dropout rate
        rms_norm_eps: RMSNorm epsilon
        embedding_weight: Shared embedding weight for output projection
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        depth: int = 2,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        d_ff: int = 1152,
        d_state: int = 128,
        d_conv: int = 4,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        embedding_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.depth = depth

        # ── Token embedding for MTP input ───────────────────────────
        # Each MTP depth takes the previous predicted token as input
        # We use a separate embedding to avoid interfering with the main embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # ── Per-depth processing blocks ────────────────────────────
        # Each depth k processes: fuse(h_main, emb(token_{t+k-1}))
        # then passes through a small hybrid block
        self.mtp_norms_1 = nn.ModuleList([
            RMSNorm(d_model, eps=rms_norm_eps) for _ in range(depth)
        ])
        self.mtp_attentions = nn.ModuleList([
            GroupedQueryAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.mtp_ssms = nn.ModuleList([
            Mamba2Block(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=1,
            )
            for _ in range(depth)
        ])
        self.mtp_norms_2 = nn.ModuleList([
            RMSNorm(d_model, eps=rms_norm_eps) for _ in range(depth)
        ])
        self.mtp_ffns = nn.ModuleList([
            SwiGLUFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
            for _ in range(depth)
        ])

        # ── Output heads ────────────────────────────────────────────
        # Each depth has its own output projection
        # Uses tied embedding weight if provided
        if embedding_weight is not None:
            # Tie with main embedding
            self.output_heads = None  # Use shared weight
            self.register_buffer("_shared_embedding", embedding_weight, persistent=False)
        else:
            self.output_heads = nn.ModuleList([
                nn.Linear(d_model, vocab_size, bias=False)
                for _ in range(depth)
            ])

        # ── Fusion projections ──────────────────────────────────────
        # Fuse h_main with token embedding at each depth
        self.fusion_projections = nn.ModuleList([
            nn.Linear(d_model * 2, d_model, bias=False)
            for _ in range(depth)
        ])

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        std = 0.02
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=std)
        for proj in self.fusion_projections:
            nn.init.normal_(proj.weight, mean=0.0, std=std)
        if self.output_heads is not None:
            for head in self.output_heads:
                nn.init.normal_(head.weight, mean=0.0, std=std)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for MTP.

        Args:
            hidden_states: Main model final hidden states (batch, seqlen, d_model)
            input_ids: Token IDs (batch, seqlen) — used to get target tokens
            labels: Target labels (batch, seqlen) — for loss computation
            attention_mask: Optional attention mask

        Returns:
            (mtp_logits_list, mtp_loss) where:
              - mtp_logits_list: List of logits tensors, each (batch, seqlen, vocab_size)
              - mtp_loss: Combined MTP loss (scalar) or None if labels not provided
        """
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        mtp_logits_list: List[torch.Tensor] = []
        mtp_losses: List[torch.Tensor] = []

        # Current representation, initialized from main model output
        h_current = hidden_states

        for k in range(self.depth):
            # ── 1. Get the token to predict for this depth ──────────
            # Depth 0: use input_ids shifted right by 0 → predict next token
            # Depth k: use input_ids shifted right by k → predict k-th future token
            if labels is not None:
                # During training: depth k conditions on token at position t+k (the k-th future token)
                # prev_token[t] = label at position (t+k), i.e., the k-th next token
                if k == 0:
                    prev_token = input_ids
                else:
                    # Shift labels LEFT by k: prev_token[t] = labels[t+k]
                    # Pad end with zeros
                    prev_token = torch.cat([
                        labels[:, k:],
                        torch.zeros(batch_size, k, dtype=labels.dtype, device=labels.device),
                    ], dim=1)
            else:
                # During inference: use input_ids (only meaningful for depth 0)
                prev_token = input_ids

            # ── 2. Embed the previous token ─────────────────────────
            # Replace -100 (ignore_index) with 0 for safe embedding lookup
            # The embed at position 0 will be multiplied by 0 in subsequent loss
            # computation via labels masking, avoiding sentinel leakage
            prev_token_safe = prev_token.clamp(min=0)
            # Zero out embeddings where original was -100 (ignore_index)
            pad_mask = (prev_token == -100).unsqueeze(-1).float()
            token_emb = self.token_embedding(prev_token_safe) * (1.0 - pad_mask)  # (batch, seqlen, d_model)

            # ── 3. Fuse with main hidden states ──────────────────────
            fused = torch.cat([h_current, token_emb], dim=-1)  # (batch, seqlen, 2*d_model)
            h_k = self.fusion_projections[k](fused)  # (batch, seqlen, d_model)

            # ── 4. Process through small hybrid block ────────────────
            # Pre-norm + Attention + SSM (parallel) + residual
            residual = h_k
            h_k = self.mtp_norms_1[k](h_k)

            attn_out, _ = self.mtp_attentions[k](
                h_k, attention_mask=attention_mask, use_cache=False
            )
            ssm_out, _ = self.mtp_ssms[k](h_k, use_cache=False)

            # Simple addition fusion (no learned gate for MTP blocks)
            h_k = residual + attn_out + ssm_out

            # Pre-norm + FFN + residual
            residual = h_k
            h_k = self.mtp_norms_2[k](h_k)
            h_k = residual + self.mtp_ffns[k](h_k)

            # ── 5. Project to logits ────────────────────────────────
            if self.output_heads is not None:
                logits_k = self.output_heads[k](h_k)
            else:
                logits_k = F.linear(h_k, self._shared_embedding)

            mtp_logits_list.append(logits_k)

            # ── 6. Compute loss for this depth ───────────────────────
            if labels is not None:
                # Each depth predicts a different future token:
                # Depth 0: predict token_{t+1} from position t
                # Depth k: predict token_{t+k+1} from position t
                # So shift logits by 1, shift labels by (k+1)
                shift = k + 1
                if logits_k.shape[1] > shift and labels.shape[1] > shift:
                    shift_logits = logits_k[..., :-shift, :].contiguous()
                    shift_labels = labels[..., shift:].contiguous()

                    loss_k = F.cross_entropy(
                        shift_logits.view(-1, self.vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    mtp_losses.append(loss_k)

            # Update current representation for next depth
            h_current = h_k

        # Combine MTP losses
        mtp_loss = None
        if mtp_losses:
            mtp_loss = torch.stack(mtp_losses).mean()

        return mtp_logits_list, mtp_loss

    def generate_mtp_tokens(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> List[torch.Tensor]:
        """
        Generate MTP predictions for speculative decoding.

        Returns the predicted tokens at each MTP depth.

        Args:
            hidden_states: Main model hidden states (batch, seqlen, d_model)
            input_ids: Token IDs (batch, seqlen)
            attention_mask: Optional attention mask
            temperature: Sampling temperature

        Returns:
            List of token tensors, one per MTP depth, each (batch, 1)
        """
        mtp_logits_list, _ = self.forward(
            hidden_states=hidden_states,
            input_ids=input_ids,
            labels=None,
            attention_mask=attention_mask,
        )

        predicted_tokens = []
        for logits in mtp_logits_list:
            last_logits = logits[:, -1:, :]  # (batch, 1, vocab_size)
            if temperature == 0:
                next_token = last_logits.argmax(dim=-1)  # (batch, 1)
            else:
                last_logits = last_logits / temperature
                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(
                    probs.view(-1, self.vocab_size), num_samples=1
                ).view(-1, 1)  # (batch, 1)
            predicted_tokens.append(next_token)

        return predicted_tokens

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, vocab_size={self.vocab_size}, "
            f"depth={self.depth}"
        )
