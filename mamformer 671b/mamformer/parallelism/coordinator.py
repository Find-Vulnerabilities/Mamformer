"""
4D Parallelism Coordinator
=============================
Orchestrates Data, Tensor, Pipeline, and Expert parallelism.

This is the top-level module that:
  1. Creates process groups for each parallelism dimension
  2. Maps GPUs to a 4D grid: DP x TP x PP x EP
  3. Shards the model accordingly
  4. Provides a unified training interface

GPU Layout (4D grid):
  The total number of GPUs = DP * TP * PP * EP
  Each GPU has coordinates (dp_rank, tp_rank, pp_rank, ep_rank).

  For 64 GPUs training Mamformer-671B:
    DP = 2  (2 data replicas)
    TP = 4  (within-node, NVLink)
    PP = 4  (52 layers => 13 per stage)
    EP = 2  (640 experts => 320 per EP rank)
    Total: 2 x 4 x 4 x 2 = 64 GPUs

Usage:
    config = ParallelConfig(dp=2, tp=4, pp=4, ep=2)
    coordinator = DistributedCoordinator(config)
    model = coordinator.shard_model(MamformerForCausalLM(model_config))
    loss = model(input_ids, labels=labels)["loss"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist


@dataclass
class ParallelConfig:
    """
    Configuration for 4D parallelism.

    Args:
        dp_size: Data-parallel replicas (default: 1)
        tp_size: Tensor-parallel size (default: 1)
        pp_size: Pipeline-parallel stages (default: 1)
        ep_size: Expert-parallel size (default: 1)
    """

    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1

    @property
    def world_size(self) -> int:
        return self.dp_size * self.tp_size * self.pp_size * self.ep_size

    def validate(self):
        """Validate the parallel configuration."""
        if self.world_size == 1:
            return  # Single GPU, no parallelism needed

        assert self.dp_size >= 1, f"dp_size must be >= 1, got {self.dp_size}"
        assert self.tp_size >= 1, f"tp_size must be >= 1, got {self.tp_size}"
        assert self.pp_size >= 1, f"pp_size must be >= 1, got {self.pp_size}"
        assert self.ep_size >= 1, f"ep_size must be >= 1, got {self.ep_size}"

        # TP should fit within a single node (typically 8 GPUs)
        assert self.tp_size <= 8, (
            f"tp_size ({self.tp_size}) > 8 requires cross-node NVLINK (expensive)"
        )

        # Check total GPU count
        total_gpus = self.world_size
        if torch.distributed.is_initialized():
            actual_gpus = torch.distributed.get_world_size()
            assert total_gpus == actual_gpus, (
                f"ParallelConfig world_size ({total_gpus}) != actual GPUs ({actual_gpus})"
            )

        # TP size must divide model dimensions
        # (checked at model construction time, but warn here)
        if self.tp_size > 1:
            pass  # n_heads, n_kv_heads, d_model divisibility checked at layer init

        # EP must have at least as many experts as ep_size
        # (checked at MoE layer construction)

        # PP should divide layers evenly
        if self.pp_size > 1:
            pass  # Checked at shard_model_pp time

    def get_4d_rank(self, global_rank: int) -> tuple:
        """
        Map a global rank to 4D coordinates (dp, tp, pp, ep).

        Layout: [DP][TP][PP][EP] — DP outermost, EP innermost.
        """
        ep_rank = global_rank % self.ep_size
        pp_rank = (global_rank // self.ep_size) % self.pp_size
        tp_rank = (global_rank // (self.ep_size * self.pp_size)) % self.tp_size
        dp_rank = global_rank // (self.ep_size * self.pp_size * self.tp_size)
        return dp_rank, tp_rank, pp_rank, ep_rank

    def get_global_rank(self, dp: int, tp: int, pp: int, ep: int) -> int:
        """Map 4D coordinates to a global rank."""
        return (dp * self.tp_size * self.pp_size * self.ep_size
                + tp * self.pp_size * self.ep_size
                + pp * self.ep_size
                + ep)


class DistributedCoordinator:
    """
    Manages 4D parallelism setup, process groups, and model sharding.

    Usage:
        config = ParallelConfig(dp=2, tp=4, pp=4, ep=2)
        coord = DistributedCoordinator(config)
        coord.initialize()

        model = MamformerForCausalLM(model_config)
        model = coord.shard_model(model)

        # Training
        for batch in dataloader:
            loss = model(input_ids, labels=labels)["loss"]
            loss.backward()
            optimizer.step()
    """

    def __init__(self, config: ParallelConfig):
        self.config = config
        config.validate()

        self._groups: dict[str, Optional[dist.ProcessGroup]] = {
            "dp": None, "tp": None, "pp": None, "ep": None,
        }
        self._ranks: dict[str, int] = {"dp": 0, "tp": 0, "pp": 0, "ep": 0}
        self._sizes: dict[str, int] = {
            "dp": config.dp_size, "tp": config.tp_size,
            "pp": config.pp_size, "ep": config.ep_size,
        }

    @property
    def is_initialized(self) -> bool:
        return dist.is_initialized()

    def initialize(self):
        """
        Initialize all process groups for 4D parallelism.

        Must be called after dist.init_process_group().
        """
        if not dist.is_initialized():
            return  # Single GPU mode

        global_rank = dist.get_rank()
        dp_rank, tp_rank, pp_rank, ep_rank = self.config.get_4d_rank(global_rank)

        self._ranks = {"dp": dp_rank, "tp": tp_rank, "pp": pp_rank, "ep": ep_rank}
        # Map dimension names to tuple indices for get_4d_rank
        _dim_idx = {"dp": 0, "tp": 1, "pp": 2, "ep": 3}

        # Build process groups: group GPUs that share the same 3 coordinates
        # TP group: same DP, PP, EP — different TP
        for dim, fixed_dims in [
            ("tp", ["dp", "pp", "ep"]),
            ("pp", ["dp", "tp", "ep"]),
            ("ep", ["dp", "tp", "pp"]),
            ("dp", ["tp", "pp", "ep"]),
        ]:
            group_ranks = []
            for r in range(self.config.world_size):
                r_coords = self.config.get_4d_rank(r)
                same = all(
                    r_coords[_dim_idx[d]] == self._ranks[d]
                    for d in fixed_dims
                )
                if same:
                    group_ranks.append(r)

            if len(group_ranks) > 1:
                self._groups[dim] = dist.new_group(group_ranks)
            else:
                self._groups[dim] = None

        self._setup_tp_group()

    def _setup_tp_group(self):
        """Configure TP group in the tensor_parallel module."""
        from mamformer.parallelism.tensor_parallel import _set_tp_group
        if self._groups["tp"] is not None:
            _set_tp_group(self._groups["tp"])

    def get_group(self, dim: str) -> Optional[dist.ProcessGroup]:
        """Get the process group for a parallelism dimension."""
        return self._groups.get(dim)

    def get_rank(self, dim: str) -> int:
        """Get local rank within a parallelism dimension."""
        return self._ranks.get(dim, 0)

    def get_size(self, dim: str) -> int:
        """Get world size for a parallelism dimension."""
        return self._sizes.get(dim, 1)

    def shard_model(self, model: nn.Module) -> nn.Module:
        """
        Apply 4D parallelism sharding to the model.

        Order matters:
          1. Pipeline Parallel: split layers across PP ranks
          2. Tensor Parallel: split individual layers across TP ranks
          3. Expert Parallel: split experts across EP ranks
          4. Data Parallel: replicate weights across DP ranks (handled by DDP/FSDP)

        Args:
            model: The full MamformerForCausalLM model

        Returns:
            Sharded model for this rank
        """
        if not self.is_initialized or self.config.world_size == 1:
            return model

        pp_rank = self._ranks["pp"]
        pp_size = self._sizes["pp"]

        # Step 1: Pipeline sharding (select layers for this rank)
        if pp_size > 1:
            from mamformer.parallelism.pipeline_parallel import shard_model_pp
            model = shard_model_pp(model, pp_size, pp_rank)

        # Step 2 & 3: TP and EP are applied at layer construction time
        # when using TPAttention, TPMamba2Block, TPDeepSeekMoE, EPMoE

        return model

    def get_4d_info(self) -> dict:
        """Get human-readable 4D topology information."""
        return {
            "world_size": self.config.world_size,
            "dp": {"size": self._sizes["dp"], "rank": self._ranks["dp"]},
            "tp": {"size": self._sizes["tp"], "rank": self._ranks["tp"]},
            "pp": {"size": self._sizes["pp"], "rank": self._ranks["pp"]},
            "ep": {"size": self._sizes["ep"], "rank": self._ranks["ep"]},
            "groups": {k: (v is not None) for k, v in self._groups.items()},
        }


def shard_model_4d(
    model: nn.Module,
    parallel_config: ParallelConfig,
) -> nn.Module:
    """
    Convenience function to shard a model with 4D parallelism.

    Args:
        model: MamformerForCausalLM instance
        parallel_config: ParallelConfig specifying the 4D layout

    Returns:
        Sharded model
    """
    coordinator = DistributedCoordinator(parallel_config)
    if coordinator.is_initialized:
        coordinator.initialize()
    return coordinator.shard_model(model)
