"""
Mamformer Distributed Parallelism
==================================
4D parallelism infrastructure for training 7B-671B models.

- tensor_parallel: Split layers across GPUs (attention heads, FFN dims)
- expert_parallel: Distribute MoE experts across GPUs via all-to-all
- pipeline_parallel: Split layers across GPU groups with 1F1B scheduling
- coordinator: Orchestrate DP x TP x PP x EP for 4D parallelism
"""

from mamformer.parallelism.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TPAttention,
    TPMamba2Block,
    TPDeepSeekMoE,
    shard_model_tp,
)
from mamformer.parallelism.expert_parallel import (
    ExpertParallelGroup,
    EPMoE,
    expert_dispatch,
    expert_combine,
    shard_experts_ep,
)
from mamformer.parallelism.pipeline_parallel import (
    PipelineStage,
    PipelineScheduler1F1B,
    shard_model_pp,
)
from mamformer.parallelism.coordinator import (
    DistributedCoordinator,
    ParallelConfig,
    shard_model_4d,
)
from mamformer.parallelism.diagnostics import (
    CommMonitor,
    LoadBalanceAnalyzer,
    PipelineBubbleAnalyzer,
    ParallelismMonitor,
)

__all__ = [
    # Tensor Parallel
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TPAttention",
    "TPMamba2Block",
    "TPDeepSeekMoE",
    "shard_model_tp",
    # Expert Parallel
    "ExpertParallelGroup",
    "EPMoE",
    "expert_dispatch",
    "expert_combine",
    "shard_experts_ep",
    # Pipeline Parallel
    "PipelineStage",
    "PipelineScheduler1F1B",
    "shard_model_pp",
    # Coordinator
    "DistributedCoordinator",
    "ParallelConfig",
    "shard_model_4d",
    # Diagnostics
    "CommMonitor",
    "LoadBalanceAnalyzer",
    "PipelineBubbleAnalyzer",
    "ParallelismMonitor",
]
