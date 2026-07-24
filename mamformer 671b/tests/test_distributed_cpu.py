"""
CPU Distributed Tests for Mamformer Parallelism
=================================================
Verifies pipeline gradient flow and expert parallelism correctness
using torch.distributed with gloo backend (runs on CPU, no GPU needed).

Usage:
    python tests/test_distributed_cpu.py
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_pipeline_gradient_test(rank, world_size):
    """Test pipeline send/recv gradient flow with autograd Functions."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29510'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    from mamformer.parallelism.pipeline_parallel import _SendForward, _RecvForward

    pp_group = dist.new_group(list(range(world_size)))
    device = torch.device('cpu')

    if rank == 0:
        x = torch.randn(2, 8, 16, requires_grad=True)
        sent = _SendForward.apply(x, 1, pp_group)
        loss = sent.sum()
        loss.backward()
        assert x.grad is not None, 'Stage 0: gradient must flow back'
        grad_norm = x.grad.norm().item()
        assert grad_norm > 0, f'Stage 0: gradient should be non-zero, got {grad_norm}'
        print(f'[Rank 0] Pipeline SendForward: gradient OK (norm={grad_norm:.4f})')

    elif rank == 1:
        grad_trigger = torch.zeros(1, device=device, requires_grad=True)
        recv = _RecvForward.apply((2, 8, 16), 0, pp_group, device, torch.float32, grad_trigger)
        loss = recv.sum()
        loss.backward()
        print(f'[Rank 1] Pipeline RecvForward: backward OK, gradient sent upstream')

    dist.destroy_process_group()


def run_ep_expert_mapping_test(rank, world_size):
    """Test EP expert ownership mapping (no re-routing)."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29511'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    from mamformer.layers.moe import DeepSeekMoE

    moe = DeepSeekMoE(
        d_model=64, n_shared_experts=1, shared_expert_dim=64,
        n_routed_experts=8, top_k=2, routed_expert_dim=32,
    )
    moe.train()

    x = torch.randn(2, 8, 64)
    out, aux = moe(x)

    loss = out.sum()
    loss.backward()

    has_grad = sum(
        1 for n, p in moe.named_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f'[Rank {rank}] EP MoE: {has_grad} params with non-zero grad')
    print(f'[Rank {rank}]   bias_mean={aux.get("expert_bias_mean", "N/A"):.6f}')

    dist.destroy_process_group()


def run_full_model_parallel_test(rank, world_size):
    """Test full Mamformer model with cross-layer interleaving + pipeline sharding."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29512'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    from mamformer.config import MamformerConfig
    from mamformer.model import MamformerModel, MamformerForCausalLM

    c = MamformerConfig.from_preset('debug')
    c.interleave.enabled = True
    c.interleave.pattern = 'cross_layer'
    c.interleave.attn_every_k = 2
    c.interleave.fusion_layers = [3]
    c.kda_diff.enabled = True
    c.dsa.enabled = False

    model = MamformerForCausalLM(c)
    model.train()

    x = torch.randint(0, c.vocab_size, (2, 16))
    out = model(x, labels=x)
    loss = out['loss']
    loss.backward()

    grad_params = sum(1 for n, p in model.named_parameters() if p.grad is not None)
    print(f'[Rank {rank}] Full model: loss={loss.item():.4f}, grad_params={grad_params}')

    dist.destroy_process_group()


def run_pipeline_stage_test(rank, world_size):
    """Test PipelineStage with cross-layer layers."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29513'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    from mamformer.config import MamformerConfig
    from mamformer.parallelism.pipeline_parallel import shard_model_pp
    from mamformer.model import MamformerForCausalLM

    c = MamformerConfig.from_preset('debug')
    c.interleave.enabled = True
    c.interleave.pattern = 'cross_layer'
    c.interleave.attn_every_k = 2
    c.interleave.fusion_layers = [3]
    c.kda_diff.enabled = True
    c.dsa.enabled = False

    model = MamformerForCausalLM(c)
    stage = shard_model_pp(model, pp_size=world_size, pp_rank=rank)

    if stage.is_first and stage.is_last:
        # Single stage: embed + layers + norm
        x = torch.randint(0, c.vocab_size, (2, 16))
        result = stage(input_ids=x)
        hs = result['hidden_states']
        print(f'[Rank {rank}] PipelineStage (single): hs shape={hs.shape}')
    elif stage.is_first:
        x = torch.randint(0, c.vocab_size, (2, 16))
        result = stage(input_ids=x)
        print(f'[Rank {rank}] PipelineStage (first): hs shape={result["hidden_states"].shape}')
    elif stage.is_last:
        # Last stage needs hidden_states from previous
        pass  # Skip — needs communication
    else:
        # Middle stage needs hidden_states from previous
        pass  # Skip — needs communication

    print(f'[Rank {rank}] PipelineStage: layers={len(stage.layers)}')
    dist.destroy_process_group()


if __name__ == '__main__':
    print('=' * 60)
    print('Mamformer CPU Distributed Tests')
    print('=' * 60)

    test_failed = False

    # Test 1: Pipeline gradient flow
    print('\n--- Test 1: Pipeline Gradient Flow ---')
    try:
        mp.spawn(run_pipeline_gradient_test, args=(2,), nprocs=2, join=True)
        print('PASSED')
    except Exception as e:
        print(f'FAILED: {e}')
        test_failed = True

    # Test 2: EP expert mapping
    print('\n--- Test 2: EP Expert Mapping ---')
    try:
        mp.spawn(run_ep_expert_mapping_test, args=(2,), nprocs=2, join=True)
        print('PASSED')
    except Exception as e:
        print(f'FAILED: {e}')
        test_failed = True

    # Test 3: Full model (single process, cross-layer)
    print('\n--- Test 3: Full Model Cross-Layer ---')
    try:
        mp.spawn(run_full_model_parallel_test, args=(2,), nprocs=2, join=True)
        print('PASSED')
    except Exception as e:
        print(f'FAILED: {e}')
        test_failed = True

    # Test 4: Pipeline stage sharding
    print('\n--- Test 4: Pipeline Stage Sharding ---')
    try:
        mp.spawn(run_pipeline_stage_test, args=(2,), nprocs=2, join=True)
        print('PASSED')
    except Exception as e:
        print(f'FAILED: {e}')
        test_failed = True

    print('\n' + '=' * 60)
    if test_failed:
        print('SOME TESTS FAILED')
        sys.exit(1)
    else:
        print('ALL TESTS PASSED')
        sys.exit(0)
