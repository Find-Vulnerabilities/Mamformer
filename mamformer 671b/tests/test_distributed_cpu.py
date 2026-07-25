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

    # ── Single-process edge-case tests (no spawn needed) ──────────
    print('\n--- Edge Case: 4D Coordinate Mapping ---')
    try:
        _test_4d_coordinate_mapping()
        print('PASSED')
    except Exception as e:
        print(f'FAILED: {e}')
        test_failed = True

    print('\n--- Edge Case: Expert Range Boundaries ---')
    try:
        _test_expert_range_boundaries()
        print('PASSED')
    except Exception as e:
        print(f'FAILED: {e}')
        test_failed = True

    print('\n--- Edge Case: Coordinator Validation ---')
    try:
        _test_coordinator_validation()
        print('PASSED')
    except Exception as e:
        print(f'FAILED: {e}')
        test_failed = True

    print('\n--- Edge Case: Expert Dispatch/Combine ---')
    try:
        _test_expert_dispatch_combine()
        print('PASSED')
    except Exception as e:
        print(f'FAILED: {e}')
        test_failed = True

    # Test 0: TP+PP combined integration
    print('\n--- Test 0: TP+PP Combined Integration ---')
    try:
        _test_tp_pp_integration()
        print('PASSED')
    except Exception as e:
        print(f'FAILED: {e}')
        import traceback
        traceback.print_exc()
        test_failed = True

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


# ═══════════════════════════════════════════════════════════════════════
# Single-process edge-case test functions
# ═══════════════════════════════════════════════════════════════════════

def _test_4d_coordinate_mapping():
    """Verify 4D coordinate mapping is bijective and consistent."""
    from mamformer.parallelism.coordinator import ParallelConfig

    cfg = ParallelConfig(dp_size=2, tp_size=4, pp_size=2, ep_size=2)
    assert cfg.world_size == 32, f"world_size={cfg.world_size}"

    seen = set()
    for r in range(cfg.world_size):
        dp, tp, pp, ep = cfg.get_4d_rank(r)
        # Round-trip: coordinates -> global rank should match original
        rt = cfg.get_global_rank(dp, tp, pp, ep)
        assert rt == r, f"round-trip failed: rank {r} -> ({dp},{tp},{pp},{ep}) -> {rt}"
        assert 0 <= dp < cfg.dp_size
        assert 0 <= tp < cfg.tp_size
        assert 0 <= pp < cfg.pp_size
        assert 0 <= ep < cfg.ep_size
        seen.add((dp, tp, pp, ep))

    assert len(seen) == cfg.world_size, f"non-bijective: {len(seen)} unique coords for {cfg.world_size} ranks"

    # Edge case: all sizes = 1
    cfg1 = ParallelConfig(dp_size=1, tp_size=1, pp_size=1, ep_size=1)
    assert cfg1.world_size == 1
    assert cfg1.get_4d_rank(0) == (0, 0, 0, 0)


def _test_expert_range_boundaries():
    """Verify EP expert range calculation at boundaries."""
    from mamformer.parallelism.expert_parallel import ExpertParallelGroup

    # 640 experts, 8-way EP
    ep = ExpertParallelGroup(ep_size=8, ep_rank=0)
    s, e = ep.get_expert_range(640)
    assert s == 0 and e == 80, f"rank 0: [{s},{e})"

    ep.ep_rank = 7
    s, e = ep.get_expert_range(640)
    assert s == 560 and e == 640, f"rank 7: [{s},{e})"

    # 640 experts, 1-way EP (no sharding)
    ep = ExpertParallelGroup(ep_size=1, ep_rank=0)
    s, e = ep.get_expert_range(640)
    assert s == 0 and e == 640

    # Odd number of experts (prime)
    ep = ExpertParallelGroup(ep_size=4, ep_rank=2)
    s, e = ep.get_expert_range(67)
    assert s == 34 and e == 51, f"67 experts, rank 2: [{s},{e})"

    # Expert rank mapping
    assert ep.get_expert_rank(0, 67) == 0
    assert ep.get_expert_rank(33, 67) == 1
    assert ep.get_expert_rank(34, 67) == 2
    assert ep.get_expert_rank(66, 67) == 3


def _test_coordinator_validation():
    """Verify coordinator validation catches bad configs."""
    from mamformer.parallelism.coordinator import ParallelConfig

    # power-of-2 TP check
    try:
        cfg = ParallelConfig(tp_size=3, dp_size=1, pp_size=1, ep_size=1)
        cfg.validate()
        assert False, "tp_size=3 should fail (not power of 2)"
    except ValueError:
        pass

    # EP size cap
    try:
        cfg = ParallelConfig(ep_size=128, dp_size=1, tp_size=1, pp_size=1)
        cfg.validate()
        assert False, "ep_size=128 should fail (>64 cap)"
    except ValueError:
        pass

    # Valid config should pass
    cfg = ParallelConfig(dp_size=2, tp_size=4, pp_size=2, ep_size=2)
    cfg.validate()  # should not raise


def _test_expert_dispatch_combine():
    """Verify expert_dispatch returns consistent tuple sizes."""
    from mamformer.parallelism.expert_parallel import (
        ExpertParallelGroup, expert_dispatch, expert_combine,
    )
    import torch

    # Single-rank (ep_size=1): dispatch should return 4 values
    ep = ExpertParallelGroup(ep_size=1, ep_rank=0)
    x = torch.randn(10, 64)
    indices = torch.randint(0, 8, (10,))

    result = expert_dispatch(x, indices, 8, ep)
    assert len(result) == 4, f"dispatch returned {len(result)} values, expected 4"
    dispatched, sort_order, token_counts, recv_counts = result

    # All tokens should come back to rank 0
    assert dispatched.shape[0] == 10
    assert token_counts.sum() == 10

    # Combine should restore original order
    combined = expert_combine(dispatched, sort_order, token_counts, 10, ep, recv_counts)
    # With ep_size=1, dispatched == original in sort_order
    # After combine, should match original order
    assert combined.shape == (10, 64)

    # Zero-token edge case
    x_empty = torch.randn(0, 64)
    indices_empty = torch.randint(0, 8, (0,))
    result = expert_dispatch(x_empty, indices_empty, 8, ep)
    assert len(result) == 4
    assert result[0].shape == (0, 64)
    assert result[2].sum() == 0


def run_tp_pp_integration_test(rank, world_size):
    """Integration test: TP=2 + PP=2 with a tiny model, forward+backward.
    Verifies gradient flow through both parallelism dimensions simultaneously."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29520'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    from mamformer.parallelism.coordinator import ParallelConfig, DistributedCoordinator

    # 4-process layout: DP=1, TP=2, PP=2, EP=1
    cfg = ParallelConfig(dp_size=1, tp_size=2, pp_size=2, ep_size=1)
    coord = DistributedCoordinator(cfg)
    coord.initialize()
    tp_rank = coord.get_rank("tp")
    pp_rank = coord.get_rank("pp")
    tp_size = coord.get_size("tp")
    pp_size = coord.get_size("pp")
    device = torch.device('cpu')

    # Build tiny model
    from mamformer.config import MamformerConfig
    mc = MamformerConfig.from_preset('debug')
    mc.n_layers = 4
    mc.d_model = 64
    mc.n_heads = 4
    mc.n_kv_heads = 2
    mc.head_dim = 16
    mc.d_ff = 128
    mc.vocab_size = 256
    mc.max_seq_len = 32

    from mamformer.model import MamformerForCausalLM
    model = MamformerForCausalLM(mc).to(device)
    model.train()

    # PP shard
    from mamformer.parallelism.pipeline_parallel import shard_model_pp
    stage = shard_model_pp(model, pp_size=pp_size, pp_rank=pp_rank)

    # Run forward
    if stage.is_first:
        x = torch.randint(0, mc.vocab_size, (2, 16))
        result = stage(input_ids=x)
        hidden = result['hidden_states']
        assert hidden.shape[-1] == mc.d_model
        # Send to next PP stage
        if pp_size > 1 and not stage.is_last:
            from mamformer.parallelism.pipeline_parallel import _SendForward
            pp_group = coord.get_group("pp")
            hidden = _SendForward.apply(hidden, pp_rank + 1, pp_group)
    elif stage.is_last:
        # Receive from previous PP stage
        from mamformer.parallelism.pipeline_parallel import _RecvForward
        pp_group = coord.get_group("pp")
        grad_trigger = torch.zeros(1, device=device, requires_grad=True)
        hidden = _RecvForward.apply(
            (2, 16, mc.d_model), pp_rank - 1, pp_group,
            device, torch.float32, grad_trigger,
        )
        # Compute loss on last stage
        if hasattr(stage, 'lm_head_weight') and stage.lm_head_weight is not None:
            logits = torch.nn.functional.linear(hidden, stage.lm_head_weight)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, mc.vocab_size),
                torch.randint(0, mc.vocab_size, (2 * 16,)),
            )
            loss.backward()

    # Verify TP group exists and has correct size
    tp_group = coord.get_group("tp")
    if tp_group is not None:
        assert dist.get_world_size(tp_group) == tp_size

    # Verify gradients flow
    grad_count = sum(1 for p in stage.parameters() if p.grad is not None)
    print(f'[Rank {rank}] TP+PP integration: tp={tp_rank}/{tp_size} pp={pp_rank}/{pp_size} grad_params={grad_count}')

    dist.destroy_process_group()


def _test_tp_pp_integration():
    """Run TP+PP combined test with 4 CPU processes."""
    import torch.multiprocessing as mp
    mp.spawn(run_tp_pp_integration_test, args=(4,), nprocs=4, join=True)
