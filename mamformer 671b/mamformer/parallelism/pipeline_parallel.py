"""
Pipeline Parallelism for Mamformer
=====================================
Splits model layers across GPU groups with micro-batch scheduling.

Supports two scheduling strategies:
  1. GPipe: All forward passes → all backward passes (simple, high memory)
  2. 1F1B: Interleave forward/backward to reduce activation memory (default)

The model's layers are partitioned into `pp_size` stages. Each stage
runs on a separate GPU (or TP group). Communication between stages
uses point-to-point send/recv.

Reference: "GPipe: Efficient Training of Large Neural Networks
           using Pipeline Parallelism" (Huang et al., 2019)
           "Memory-Efficient Pipeline-Parallel DNN Training" (Narayanan et al., 2020)

Usage:
    stages = shard_model_pp(model, pp_size=4, num_layers=52)
    scheduler = PipelineScheduler1F1B(stages, num_microbatches=8)
    loss = scheduler.run_forward_backward(input_ids, labels)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from collections import deque


# ═══════════════════════════════════════════════════════════════════════
# Autograd Communication Primitives
# ═══════════════════════════════════════════════════════════════════════


class _SendForward(torch.autograd.Function):
    """Autograd function for pipeline forward send. In backward, receives gradients."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dst: int, group) -> torch.Tensor:
        ctx.dst = dst
        ctx.group = group
        ctx.shape = tensor.shape
        if group is not None and dist.is_initialized():
            dist.send(tensor.contiguous(), dst, group=group)
        return tensor  # Pass through on sender side

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.group is not None and dist.is_initialized():
            grad_input = torch.empty(ctx.shape, device=grad_output.device, dtype=grad_output.dtype)
            dist.recv(grad_input, ctx.dst, group=ctx.group)
            return grad_input, None, None
        return grad_output, None, None


class _RecvForward(torch.autograd.Function):
    """Autograd function for pipeline forward recv. In backward, sends gradients upstream.

    Accepts a dummy tensor (grad_trigger) that requires grad. This ensures the autograd
    engine calls backward() even though the actual received tensor has no grad history.
    """

    @staticmethod
    def forward(ctx, shape: tuple, src: int, group, device, dtype, grad_trigger: torch.Tensor) -> torch.Tensor:
        ctx.src = src
        ctx.group = group
        if group is not None and dist.is_initialized():
            tensor = torch.empty(shape, device=device, dtype=dtype)
            dist.recv(tensor, src, group=group)
            return tensor
        return torch.empty(shape, device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.group is not None and dist.is_initialized():
            dist.send(grad_output.contiguous(), ctx.src, group=ctx.group)
        return None, None, None, None, None, None  # shape, src, group, device, dtype, grad_trigger


# ═══════════════════════════════════════════════════════════════════════
# Pipeline Stage
# ═══════════════════════════════════════════════════════════════════════

class PipelineStage(nn.Module):
    """
    A pipeline stage holding a contiguous range of model layers.

    Args:
        layers: List of MamformerBlock layers assigned to this stage
        stage_id: This stage's position in the pipeline (0 = first, pp-1 = last)
        pp_size: Total number of pipeline stages
        embedding: Shared embedding layer (only on first stage)
        final_norm: Final RMSNorm (only on last stage)
        lm_head_weight: LM head weight for tied embeddings (only on last stage)
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        stage_id: int,
        pp_size: int,
        embedding: Optional[nn.Embedding] = None,
        final_norm: Optional[nn.Module] = None,
        lm_head_weight: Optional[torch.Tensor] = None,
        vocab_size: int = 128000,
        d_model: int = 4096,
    ):
        super().__init__()
        self.stage_id = stage_id
        self.pp_size = pp_size
        self.is_first = stage_id == 0
        self.is_last = stage_id == pp_size - 1

        self.layers = layers
        self.embedding = embedding
        self.final_norm = final_norm
        self.lm_head_weight = lm_head_weight  # For tied embeddings
        self.d_model = d_model

        # For loss computation on last stage
        self.vocab_size = vocab_size
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache: Optional[List[dict]] = None,
    ) -> dict:
        """
        Forward pass for this pipeline stage.

        First stage: embeds input_ids → runs its layers → returns hidden_states
        Middle stages: receives hidden_states → runs its layers → returns hidden_states
        Last stage: receives hidden_states → runs its layers + norm → returns hidden_states
        """
        # First stage: embed tokens
        if self.is_first and input_ids is not None:
            assert self.embedding is not None
            hidden_states = self.embedding(input_ids)

        if hidden_states is None:
            raise ValueError(
                f"Stage {self.stage_id}: no hidden_states provided and not first stage"
            )

        # Process attention mask
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Run layers
        cache_list = [] if use_cache else None
        for layer in self.layers:
            hidden_states, layer_cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                cache=None,
            )
            if use_cache:
                cache_list.append(layer_cache)

        # Last stage: final norm
        if self.is_last and self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        result = {"hidden_states": hidden_states}
        if use_cache:
            result["cache"] = cache_list

        return result

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss on the last stage.

        Uses tied embedding weight if available.
        """
        if not self.is_last:
            raise RuntimeError("Only the last pipeline stage can compute loss")

        if self.lm_head_weight is not None:
            logits = torch.nn.functional.linear(hidden_states, self.lm_head_weight)
        else:
            raise RuntimeError("No lm_head_weight available for loss computation")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False


# ═══════════════════════════════════════════════════════════════════════
# 1F1B Pipeline Scheduler
# ═══════════════════════════════════════════════════════════════════════

class PipelineScheduler1F1B:
    """
    One-Forward-One-Backward pipeline scheduler.

    The 1F1B schedule interleaves forward and backward passes to minimize
    the number of in-flight microbatches (and thus activation memory).

    Schedule (pp_size=4, num_mb=8):
      Time →
      S0: F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7
      S1:    F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7
      S2:       F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7
      S3:          F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7

    Warmup: first stage does `pp_size - stage_id - 1` warmup microbatches
    Steady: each stage alternates 1 forward, 1 backward
    Cooldown: last stage finishes all backwards

    Args:
        stages: List of PipelineStage (one per pipeline rank)
        num_microbatches: Number of microbatches per batch
        pp_group: torch.distributed process group for pipeline
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        num_microbatches: int = 8,
        pp_group: Optional[dist.ProcessGroup] = None,
    ):
        self.stages = stages
        self.pp_size = len(stages)
        self.num_microbatches = num_microbatches
        self.pp_group = pp_group
        self.rank = dist.get_rank(pp_group) if pp_group and dist.is_initialized() else 0
        self.stage = stages[self.rank]

    def run_forward_backward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Execute 1F1B schedule for one training step.

        Args:
            input_ids: (batch, seqlen) — full batch
            labels: (batch, seqlen) — for loss computation
            attention_mask: Optional mask

        Returns:
            loss: Scalar loss tensor (only on last rank), None on other ranks
        """
        # Split batch into microbatches
        batch_size = input_ids.shape[0]
        mb_size = (batch_size + self.num_microbatches - 1) // self.num_microbatches

        mbs_input_ids = list(torch.split(input_ids, mb_size, dim=0))
        mbs_labels = list(torch.split(labels, mb_size, dim=0)) if labels is not None else None
        mbs_mask = list(torch.split(attention_mask, mb_size, dim=0)) if attention_mask is not None else [None] * len(mbs_input_ids)

        num_mb = len(mbs_input_ids)
        total_loss = 0.0
        total_tokens = 0

        # ── Warmup: forward passes ─────────────────────────────────
        fwd_queue: deque = deque()

        for mb_idx in range(min(num_mb, self.pp_size - self.rank)):
            # Receive from previous stage (if not first)
            if not self.stage.is_first:
                hidden_states = self._recv_forward(
                    mb_size=mbs_input_ids[mb_idx].shape[0],
                    seq_len=mbs_input_ids[mb_idx].shape[1],
                    d_model=self.stage.d_model,
                )
            else:
                hidden_states = None

            # Forward
            result = self.stage(
                hidden_states=hidden_states,
                input_ids=mbs_input_ids[mb_idx] if self.stage.is_first else None,
                attention_mask=mbs_mask[mb_idx],
            )

            # Send to next stage (if not last)
            if not self.stage.is_last:
                self._send_forward(result["hidden_states"], mb_idx)
            elif mbs_labels is not None:
                loss = self.stage.compute_loss(result["hidden_states"], mbs_labels[mb_idx])
                fwd_queue.append((mb_idx, loss.detach(), result["hidden_states"]))
                total_loss += loss.detach() * mbs_input_ids[mb_idx].numel()
                total_tokens += mbs_input_ids[mb_idx].numel()

        # ── Steady state: 1F1B ─────────────────────────────────────
        for mb_idx in range(self.pp_size - self.rank, num_mb):
            # One backward (if we have completed forwards)
            if fwd_queue:
                mb_id, _, hs = fwd_queue.popleft()
                # Recompute loss with autograd for backward
                loss_with_grad = self.stage.compute_loss(hs, mbs_labels[mb_id])
                loss_with_grad.backward()
                if not self.stage.is_first:
                    self._send_backward(mb_id)

            # One forward
            if not self.stage.is_first:
                hidden_states = self._recv_forward(
                    mb_size=mbs_input_ids[mb_idx].shape[0],
                    seq_len=mbs_input_ids[mb_idx].shape[1],
                    d_model=self.stage.d_model,
                )
            else:
                hidden_states = None

            result = self.stage(
                hidden_states=hidden_states,
                input_ids=mbs_input_ids[mb_idx] if self.stage.is_first else None,
                attention_mask=mbs_mask[mb_idx],
            )

            if not self.stage.is_last:
                self._send_forward(result["hidden_states"], mb_idx)
            elif mbs_labels is not None:
                loss = self.stage.compute_loss(result["hidden_states"], mbs_labels[mb_idx])
                fwd_queue.append((mb_idx, loss.detach(), result["hidden_states"]))
                total_loss += loss.detach() * mbs_input_ids[mb_idx].numel()
                total_tokens += mbs_input_ids[mb_idx].numel()

        # ── Cooldown: finish remaining backwards ───────────────────
        while fwd_queue:
            mb_id, _, hs = fwd_queue.popleft()
            # Non-last stages: wait for backward signal from next stage
            if not self.stage.is_last:
                self._recv_backward(mb_id)
            loss_with_grad = self.stage.compute_loss(hs, mbs_labels[mb_id])
            loss_with_grad.backward()
            if not self.stage.is_first:
                self._send_backward(mb_id)

        # Return average loss from last stage
        if self.stage.is_last and total_tokens > 0:
            return total_loss / total_tokens
        return None

    def _send_forward(self, tensor: torch.Tensor, mb_idx: int):
        """Send hidden states to next stage with autograd tracking for backward gradient flow."""
        if self.pp_group is not None and dist.is_initialized():
            dst = self.rank + 1
            return _SendForward.apply(tensor, dst, self.pp_group)
        return tensor

    def _recv_forward(self, mb_size: int, seq_len: int, d_model: int) -> torch.Tensor:
        """Receive hidden states from previous stage with autograd tracking for backward."""
        if self.pp_group is not None and dist.is_initialized():
            src = self.rank - 1
            device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
            # Dummy tensor requiring grad ensures autograd engine calls backward()
            grad_trigger = torch.zeros(1, device=device, requires_grad=True)
            return _RecvForward.apply(
                (mb_size, seq_len, d_model), src, self.pp_group, device, torch.float32, grad_trigger
            )
        return torch.empty(mb_size, seq_len, d_model)

    def _send_backward(self, mb_idx: int):
        """Send gradient completion signal to previous stage for 1F1B synchronization."""
        if self.pp_group is not None and dist.is_initialized():
            dst = self.rank - 1
            signal = torch.tensor([mb_idx], dtype=torch.long, device=torch.cuda.current_device())
            dist.send(signal, dst, group=self.pp_group)

    def _recv_backward(self, mb_idx: int):
        """Receive gradient completion signal from next stage."""
        if self.pp_group is not None and dist.is_initialized():
            src = self.rank + 1
            signal = torch.empty(1, dtype=torch.long, device=torch.cuda.current_device())
            dist.recv(signal, src, group=self.pp_group)


# ═══════════════════════════════════════════════════════════════════════
# Model Sharding
# ═══════════════════════════════════════════════════════════════════════

def shard_model_pp(
    model: nn.Module,
    pp_size: int,
    pp_rank: int = 0,
) -> PipelineStage:
    """
    Shard a Mamformer model into pipeline stages.

    Layers are distributed: rank r gets layers [r*L/s, (r+1)*L/s).

    Args:
        model: MamformerForCausalLM instance
        pp_size: Number of pipeline stages
        pp_rank: This rank's stage index

    Returns:
        PipelineStage for this rank
    """
    n_layers = len(model.model.layers)
    layers_per_stage = (n_layers + pp_size - 1) // pp_size
    start = pp_rank * layers_per_stage
    end = min(start + layers_per_stage, n_layers)

    layers = nn.ModuleList([model.model.layers[i] for i in range(start, end)])

    embedding = model.model.embed_tokens if pp_rank == 0 else None
    final_norm = model.model.norm if pp_rank == pp_size - 1 else None
    lm_head_weight = (
        model.model.embed_tokens.weight if pp_rank == pp_size - 1 else None
    )

    return PipelineStage(
        layers=layers,
        stage_id=pp_rank,
        pp_size=pp_size,
        embedding=embedding,
        final_norm=final_norm,
        lm_head_weight=lm_head_weight,
        vocab_size=model.config.vocab_size,
        d_model=getattr(model.config, "d_model", 4096),
    )
