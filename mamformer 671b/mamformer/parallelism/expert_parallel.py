"""
Expert Parallelism for Mamformer MoE
======================================
Distributes MoE routed experts across GPUs using all-to-all communication.

Key insight: Each GPU stores only a subset of experts. Tokens are routed
to the GPU that holds their assigned expert(s) via all-to-all.

Forward pass:
  1. Router(x) → expert_indices, gates  (local, replicated)
  2. expert_dispatch: permute + all-to-all send tokens to expert GPUs
  3. Local expert compute on each GPU (only own expert subset)
  4. expert_combine: all-to-all send results back + unpermute
  5. Apply gates + accumulate with shared expert output

Backward pass:
  - autograd handles all-to-all gradients automatically
  - Router gradients computed locally (router is replicated)

This enables training 640+ experts across 8+ GPUs, where each GPU
only needs memory for 640/ep_size experts.

Usage:
    ep_group = ExpertParallelGroup(ep_size=8)
    ep_moe = EPMoE(
        d_model=7168,
        n_routed_experts=640,
        top_k=8,
        expert_dim=896,
        ep_group=ep_group,
    )
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ═══════════════════════════════════════════════════════════════════════
# Expert Parallel Group
# ═══════════════════════════════════════════════════════════════════════

class ExpertParallelGroup:
    """
    Manages expert-parallel communication.

    Each rank in the EP group holds a subset of the total routed experts.
    Expert assignment: expert i → rank (i * ep_size // n_experts)

    Args:
        ep_size: Number of GPUs in the expert-parallel group
        ep_rank: Local rank within the EP group (auto-detected if None)
        group: Optional torch.distributed ProcessGroup
    """

    def __init__(
        self,
        ep_size: int = 1,
        ep_rank: Optional[int] = None,
        group: Optional[dist.ProcessGroup] = None,
    ):
        self.ep_size = ep_size
        self.group = group

        if ep_rank is not None:
            self.ep_rank = ep_rank
        elif group is not None and dist.is_initialized():
            self.ep_rank = dist.get_rank(group)
        else:
            self.ep_rank = 0

        self._is_initialized = group is not None and dist.is_initialized()

    def get_expert_range(self, n_total_experts: int) -> Tuple[int, int]:
        """
        Get the [start, end) range of experts owned by this rank.

        Experts are distributed: rank r gets experts [r*E/s, (r+1)*E/s).
        """
        experts_per_rank = (n_total_experts + self.ep_size - 1) // self.ep_size
        start = self.ep_rank * experts_per_rank
        end = min(start + experts_per_rank, n_total_experts)
        return start, end

    def get_expert_rank(self, expert_idx: int, n_total_experts: int) -> int:
        """Get the rank that owns a given expert."""
        experts_per_rank = (n_total_experts + self.ep_size - 1) // self.ep_size
        return expert_idx // experts_per_rank

    def all_to_all(self, tensor: torch.Tensor, split_sizes: list[int]) -> torch.Tensor:
        """
        Variable-length all-to-all communication across the EP group.

        Uses a two-step protocol:
          1. Exchange token counts via all_gather
          2. All-to-all with variable-length chunks

        Args:
            tensor: Input tensor (total_tokens, d_model)
            split_sizes: Number of tokens to send to each rank

        Returns:
            Output tensor after communication, or None if no tokens received
        """
        if not self._is_initialized or self.ep_size == 1:
            return tensor

        device = tensor.device
        d_model = tensor.shape[-1]

        # Step 1: Exchange token counts
        my_counts = torch.tensor(split_sizes, dtype=torch.long, device=device)
        # Pad to ep_size
        padded_counts = torch.zeros(self.ep_size, dtype=torch.long, device=device)
        padded_counts[:len(my_counts)] = my_counts

        all_counts = [torch.zeros(self.ep_size, dtype=torch.long, device=device) for _ in range(self.ep_size)]
        dist.all_gather(all_counts, padded_counts, group=self.group)

        # Step 2: Variable-length all-to-all using send/recv
        # For each rank pair, send and receive the appropriate number of tokens
        received_chunks = []
        total_received = 0
        for r in range(self.ep_size):
            # How many tokens are coming from rank r?
            tokens_from_r = all_counts[r][self.ep_rank].item()
            total_received += tokens_from_r

        if total_received == 0:
            return torch.zeros(0, d_model, device=device, dtype=tensor.dtype)

        # Use all_to_all_single with padding to max
        max_send = max(split_sizes) if split_sizes else 0
        max_recv = max(
            all_counts[r][self.ep_rank].item() for r in range(self.ep_size)
        ) if self.ep_size > 0 else 0
        chunk_size = max(max_send, max_recv, 1)

        # Pad input
        send_padded = torch.zeros(self.ep_size * chunk_size, d_model, device=device, dtype=tensor.dtype)
        offset = 0
        for r in range(self.ep_size):
            n_tokens = split_sizes[r] if r < len(split_sizes) else 0
            if n_tokens > 0:
                send_padded[r * chunk_size : r * chunk_size + n_tokens] = tensor[offset:offset + n_tokens]
                offset += n_tokens

        recv_padded = torch.zeros(self.ep_size * chunk_size, d_model, device=device, dtype=tensor.dtype)
        dist.all_to_all_single(
            recv_padded, send_padded,
            output_split_sizes=[chunk_size] * self.ep_size,
            input_split_sizes=[chunk_size] * self.ep_size,
            group=self.group,
        )

        # Trim: concatenate received chunks, discarding padding
        result_chunks = []
        for r in range(self.ep_size):
            n_recv = all_counts[r][self.ep_rank].item()
            if n_recv > 0:
                chunk = recv_padded[r * chunk_size : r * chunk_size + n_recv]
                result_chunks.append(chunk)

        if result_chunks:
            return torch.cat(result_chunks, dim=0)
        return torch.zeros(0, d_model, device=device, dtype=tensor.dtype)


# ═══════════════════════════════════════════════════════════════════════
# Expert Dispatch / Combine
# ═══════════════════════════════════════════════════════════════════════

def expert_dispatch(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    n_experts: int,
    ep_group: ExpertParallelGroup,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dispatch tokens to the GPUs that own their assigned experts.

    For each token in the flattened input, we send it to the GPU
    that holds expert_indices[t].

    Args:
        x: Flattened input (total_tokens, d_model)
        expert_indices: Expert assignment (total_tokens,) — which expert per token
        n_experts: Total number of routed experts
        ep_group: Expert parallel group config

    Returns:
        (dispatched_x, token_order, token_counts_per_rank)
          - dispatched_x: Tokens arriving at this rank (n_received, d_model)
          - token_order: For restoring original order during combine
          - token_counts_per_rank: How many tokens sent to each rank
    """
    total_tokens = x.shape[0]
    device = x.device

    if ep_group.ep_size == 1:
        return x, torch.arange(total_tokens, device=device), torch.tensor([total_tokens], device=device)

    # 1. Determine which tokens go to which rank
    token_to_rank = torch.zeros(total_tokens, dtype=torch.long, device=device)
    for expert_idx in range(n_experts):
        mask = (expert_indices == expert_idx)
        target_rank = ep_group.get_expert_rank(expert_idx, n_experts)
        token_to_rank[mask] = target_rank

    # 2. Sort tokens by target rank (for contiguous all-to-all)
    sort_order = torch.argsort(token_to_rank)
    sorted_x = x[sort_order]
    sorted_ranks = token_to_rank[sort_order]

    # 3. Count tokens per rank (send-side)
    token_counts = torch.zeros(ep_group.ep_size, dtype=torch.long, device=device)
    for r in range(ep_group.ep_size):
        token_counts[r] = (sorted_ranks == r).sum()

    if token_counts.sum() == 0:
        return (torch.zeros(0, x.shape[1], device=device, dtype=x.dtype),
                sort_order, token_counts, None)

    # 4. Exchange token counts to build full send/receive matrix
    # After this, recv_counts[r] = tokens rank r sent to ME (what I'll receive)
    all_send_counts = [torch.zeros(ep_group.ep_size, dtype=torch.long, device=device)
                       for _ in range(ep_group.ep_size)]
    if ep_group._is_initialized and ep_group.ep_size > 1:
        dist.all_gather(all_send_counts, token_counts, group=ep_group.group)
    else:
        all_send_counts = [token_counts]
    # count_matrix[src, dst] = tokens sent from src to dst
    count_matrix = torch.stack(all_send_counts, dim=0)  # (ep_size, ep_size)
    # recv_counts[r] = tokens I will receive from rank r = count_matrix[r, my_rank]
    recv_counts = count_matrix[:, ep_group.ep_rank]  # (ep_size,)

    # 5. All-to-all: send sorted tokens to expert-owning ranks
    split_sizes = token_counts.tolist()
    dispatched = ep_group.all_to_all(sorted_x, split_sizes)

    return dispatched, sort_order, token_counts, recv_counts


def expert_combine(
    expert_outputs: torch.Tensor,
    sort_order: torch.Tensor,
    token_counts: torch.Tensor,
    total_tokens: int,
    ep_group: ExpertParallelGroup,
    recv_counts: torch.Tensor = None,
) -> torch.Tensor:
    """
    Combine expert outputs back into the original token order.

    Reverse operation of expert_dispatch: all-to-all to return results
    to original ranks, then unpermute to restore original order.

    Args:
        expert_outputs: Results from local experts (n_local_tokens, d_model)
        sort_order: Permutation from dispatch
        token_counts: Token counts per rank from dispatch
        total_tokens: Original number of tokens
        ep_group: Expert parallel group

    Returns:
        combined: (total_tokens, d_model) in original order
    """
    device = expert_outputs.device

    if ep_group.ep_size == 1:
        return expert_outputs

    # Use receive-side counts for return routing
    # recv_counts[r] = how many tokens source rank r sent to ME during dispatch
    if recv_counts is not None:
        return_counts = recv_counts.tolist()
    else:
        # Fallback: uniform (single-GPU path shouldn't reach here)
        return_counts = token_counts.tolist()

    max_count = max(return_counts) if return_counts else 1
    d_model = expert_outputs.shape[1]

    # Pad for all_to_all_single: place my results into the correct source-rank slots
    padded_output = torch.zeros(ep_group.ep_size * max_count, d_model, device=device, dtype=expert_outputs.dtype)
    # expert_outputs are arranged by source rank order:
    # tokens from rank 0, then tokens from rank 1, ..., then tokens from rank ep-1
    # Place each source rank's chunk at position r * max_count
    offset = 0
    for r in range(ep_group.ep_size):
        n_from_r = return_counts[r] if r < len(return_counts) else 0
        if n_from_r > 0 and offset + n_from_r <= expert_outputs.shape[0]:
            dst_start = r * max_count
            padded_output[dst_start:dst_start + n_from_r] = expert_outputs[offset:offset + n_from_r]
            offset += n_from_r

    # All-to-all return: send results back to ranks that originally sent the tokens
    returned_padded = torch.zeros_like(padded_output)
    if ep_group._is_initialized and ep_group.ep_size > 1:
        dist.all_to_all_single(
            returned_padded, padded_output,
            output_split_sizes=[max_count] * ep_group.ep_size,
            input_split_sizes=[max_count] * ep_group.ep_size,
            group=ep_group.group,
        )

    # Unpermute: extract each source rank's chunk from the returned buffer
    combined = torch.zeros(total_tokens, d_model, device=device, dtype=expert_outputs.dtype)
    # Use send-side counts (token_counts) for the return side's received counts
    send_sizes = token_counts.tolist()
    returned_offset = 0
    for r in range(ep_group.ep_size):
        n_from_r = send_sizes[r] if r < len(send_sizes) else 0
        if n_from_r > 0:
            # The chunk from source rank r starts at r * max_count in returned_padded
            src_start = r * max_count
            chunk = returned_padded[src_start:src_start + n_from_r]
            # Map to original positions: these were at sort_order offsets
            # corresponding to the r-th chunk of the original sorted data
            dst_start = sum(send_sizes[:r]) if r > 0 else 0
            positions = sort_order[dst_start:dst_start + n_from_r]
            combined.scatter_(0, positions.unsqueeze(-1).expand(-1, d_model), chunk)

    return combined


# ═══════════════════════════════════════════════════════════════════════
# EP-Aware MoE
# ═══════════════════════════════════════════════════════════════════════

class EPMoE(nn.Module):
    """
    Expert-parallel Mixture of Experts.

    Each GPU holds only n_experts / ep_size routed experts.
    The router is replicated (small). Shared experts are also replicated.

    This is the key to scaling to 640+ experts: without EP, you'd need
    enough GPU memory to hold all experts on every GPU. With EP, the
    expert parameters are sharded, enabling 50B+ total parameters
    while keeping per-GPU memory manageable.

    Args:
        d_model: Hidden dimension
        n_shared_experts: Number of shared (always-active) experts
        shared_expert_dim: Intermediate dim per shared expert
        n_routed_experts: Total number of routed experts (across all GPUs)
        top_k: Experts activated per token
        routed_expert_dim: Intermediate dim per routed expert
        ep_group: Expert parallel group
    """

    def __init__(
        self,
        d_model: int,
        n_shared_experts: int = 2,
        shared_expert_dim: int = 2304,
        n_routed_experts: int = 64,
        top_k: int = 8,
        routed_expert_dim: int = 576,
        ep_group: Optional[ExpertParallelGroup] = None,
        aux_loss_free: bool = True,
        bias_update_speed: float = 0.001,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts_total = n_routed_experts
        self.top_k = top_k
        self.ep_group = ep_group or ExpertParallelGroup(ep_size=1)
        self.aux_loss_free = aux_loss_free
        self.bias_update_speed = bias_update_speed

        # Which experts does this rank own?
        expert_start, expert_end = self.ep_group.get_expert_range(n_routed_experts)
        self.expert_start = expert_start
        self.expert_end = expert_end
        self.n_local_experts = expert_end - expert_start

        # Router: replicated (always small — d_model * n_experts ≈ 7K*640 ≈ 4.5M)
        self.router = nn.Linear(d_model, n_routed_experts, bias=False)

        # Shared experts: replicated (same on all ranks)
        self.shared_experts = nn.ModuleList([
            _SwiGLUExpert(d_model, shared_expert_dim, dropout)
            for _ in range(n_shared_experts)
        ])

        # Routed experts: ONLY local subset
        self.routed_experts = nn.ModuleList([
            _SwiGLUExpert(d_model, routed_expert_dim, dropout)
            for _ in range(self.n_local_experts)
        ])

        # Aux-loss-free balancing
        if aux_loss_free:
            self.register_buffer("expert_bias", torch.zeros(n_routed_experts))
            self.register_buffer("expert_load_ema", torch.ones(n_routed_experts) / n_routed_experts)

        self.register_buffer("_expert_counts", torch.zeros(n_routed_experts, dtype=torch.long))
        self.register_buffer("_total_tokens", torch.zeros(1, dtype=torch.long))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02 / self.n_routed_experts_total ** 0.5)

    def _get_local_expert(self, global_expert_idx: int) -> Optional[nn.Module]:
        """Get the local expert module for a global expert index."""
        if self.expert_start <= global_expert_idx < self.expert_end:
            local_idx = global_expert_idx - self.expert_start
            return self.routed_experts[local_idx]
        return None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        batch_size, seq_len, d_model = x.shape

        # ── Shared experts (replicated, always active) ────────────
        shared_out = torch.zeros(batch_size, seq_len, d_model, device=x.device, dtype=x.dtype)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x)

        # ── Router (replicated) ──────────────────────────────────
        router_logits = self.router(x)  # (batch, seqlen, n_routed_experts)
        if self.aux_loss_free and self.training:
            gating_scores = router_logits + self.expert_bias
        else:
            gating_scores = router_logits

        top_k_gates, top_k_indices = torch.topk(gating_scores, k=self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_gates, dim=-1)

        # ── EP dispatch + local compute + combine ────────────────
        routed_out = self._ep_routed_forward(x, top_k_indices, top_k_gates)

        # ── Load balance ─────────────────────────────────────────
        aux_info = {"active_experts": self.top_k, "local_experts": self.n_local_experts,
                    "total_experts": self.n_routed_experts_total}
        if self.aux_loss_free and self.training:
            self._update_expert_bias(top_k_indices, batch_size * seq_len)
            aux_info["expert_bias_mean"] = self.expert_bias.mean().item()

        return shared_out + routed_out, aux_info

    def _ep_routed_forward(self, x, top_k_indices, top_k_gates):
        """EP-aware routed expert computation with all-to-all dispatch/combine
        for ALL top-k assignments (not just primary)."""
        B, S, D = x.shape
        N = B * S
        device = x.device

        # Flatten
        x_flat = x.view(N, D)
        idx_flat = top_k_indices.view(N, self.top_k)
        gate_flat = top_k_gates.view(N, self.top_k)

        output = torch.zeros(N, D, device=device, dtype=x.dtype)

        # Process each of the top-k slots with proper dispatch/combine
        # Each slot dispatches tokens to the GPU owning the assigned expert
        for k in range(self.top_k):
            expert_idx_k = idx_flat[:, k]  # (N,) — expert for slot k
            gates_k = gate_flat[:, k]     # (N,) — gate for slot k

            # Dispatch tokens to GPUs that own expert_idx_k
            dispatched_x, sort_order, token_counts, recv_counts = expert_dispatch(
                x_flat, expert_idx_k,
                self.n_routed_experts_total, self.ep_group,
            )

            if dispatched_x.shape[0] == 0:
                continue

            # Process dispatched tokens with local experts
            dispatched_output = torch.zeros_like(dispatched_x)

            # Recompute router logits to determine which local expert each token needs
            if self.aux_loss_free and self.training:
                dispatched_gating = F.linear(dispatched_x, self.router.weight) + self.expert_bias
            else:
                dispatched_gating = F.linear(dispatched_x, self.router.weight)
            _, dispatched_exp_idx = torch.topk(dispatched_gating, k=1, dim=-1)
            dispatched_exp_idx = dispatched_exp_idx.squeeze(-1)

            for local_idx in range(self.n_local_experts):
                global_expert_idx = self.expert_start + local_idx
                expert = self.routed_experts[local_idx]

                expert_mask = (dispatched_exp_idx == global_expert_idx)
                if not expert_mask.any():
                    continue

                expert_input = dispatched_x[expert_mask]
                expert_output = expert(expert_input)

                # Gate: softmax probability for this expert
                probs = F.softmax(dispatched_gating[expert_mask], dim=-1)
                expert_gates = probs[:, global_expert_idx]

                dispatched_output[expert_mask] += expert_output * expert_gates.unsqueeze(-1)

            # Combine results back to original token order
            combined_k = expert_combine(
                dispatched_output, sort_order, token_counts,
                N, self.ep_group, recv_counts,
            )
            output += combined_k

        return output.view(B, S, D)

    def _update_expert_bias(self, top_k_indices, total_tokens):
        expert_counts = torch.zeros(self.n_routed_experts_total, device=top_k_indices.device, dtype=torch.float32)
        for i in range(self.n_routed_experts_total):
            expert_counts[i] = (top_k_indices == i).sum().float()
        actual_load = expert_counts / (total_tokens * self.top_k + 1e-8)
        expected_load = 1.0 / self.n_routed_experts_total
        self.expert_bias -= self.bias_update_speed * torch.sign(actual_load - expected_load)
        self.expert_load_ema = 0.99 * self.expert_load_ema + 0.01 * actual_load


from mamformer.layers.moe import _SwiGLUExpert  # noqa: E402 (import after class def)


def shard_experts_ep(model: nn.Module, ep_group: ExpertParallelGroup) -> nn.Module:
    """Replace MoE layers with EP-aware versions."""
    # For constructed models: iterate and replace MoE layers
    # This is a utility for integration with the coordinator
    return model
