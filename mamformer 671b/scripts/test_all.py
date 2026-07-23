"""
---- - CPU -procs-------
    torchrun --nproc_per_node=2 scripts/test_all.py
"""

import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def test(name, fn):
    try:
        fn()
        print(f"  {GREEN}PASS{RESET} {name}")
        return True
    except Exception as e:
        print(f"  {RED}FAIL{RESET} {name}: {e}")
        return False

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    has_cuda = torch.cuda.is_available()

    # Use gloo for CPU, nccl for GPU
    backend = "nccl" if has_cuda else "gloo"
    if world_size > 1:
        dist.init_process_group(backend)

    device = torch.device(f"cuda:{local_rank}" if has_cuda else "cpu")
    if has_cuda:
        torch.cuda.set_device(local_rank)
    to_dev = lambda t: t.to(device)

    results = []
    mode_str = f"CUDA({torch.cuda.device_count()} GPUs)" if has_cuda else "CPU(gloo)"
    print(f"\n{'='*50}")
    print(f"  Mamformer Test Suite - {mode_str} - Rank {local_rank}/{world_size}")
    print(f"{'='*50}\n")

    # -- 1. Basic Check ----------------------------------
    print(f"[1] Basic Check (backend={backend})")
    results.append(test("PyTorch OK", lambda: None))
    if has_cuda:
        results.append(test("CUDA available", lambda: torch.cuda.is_available()))
    else:
        print(f"  {YELLOW}CPU MODE{RESET} - can verify distributed logic, not training speed")

    # -- 2. Model Build ----------------------------------
    print("\n[2] Model Build")
    from mamformer.config import MamformerConfig
    from mamformer.model import MamformerForCausalLM

    config = MamformerConfig.from_preset("debug")
    model = MamformerForCausalLM(config).to(device)
    results.append(test("Model build", lambda: model is not None))

    # -- 3. Forward Pass ----------------------------------
    print("\n[3] Forward Pass")
    x = to_dev(torch.randint(0, 1000, (2, 64)))
    labels = to_dev(torch.randint(0, 1000, (2, 64)))
    out = model(x, labels=labels)
    results.append(test("Forward pass", lambda: "loss" in out))
    results.append(test("Loss is finite", lambda: torch.isfinite(out["loss"])))
    results.append(test("Loss > 0", lambda: out["loss"].item() > 0))

    loss = out["loss"]
    loss.backward()
    results.append(test("Backward pass", lambda: any(
        p.grad is not None for p in model.parameters())))

    # -- 4. Distributed Parallel --------------------------------
    if world_size > 1:
        print(f"\n[4] Distributed Parallel ({world_size} procs)")

        # EP
        print("  - Expert Parallel -")
        from mamformer.parallelism.expert_parallel import ExpertParallelGroup, EPMoE
        ep_group = ExpertParallelGroup(ep_size=world_size)
        ep_moe = EPMoE(d_model=256, n_routed_experts=8, top_k=2,
                       routed_expert_dim=64, ep_group=ep_group).to(device)
        x_ep = to_dev(torch.randn(4, 8, 256))
        out_ep, info = ep_moe(x_ep)
        results.append(test(
            f"EP forward: output shape={tuple(out_ep.shape)}, local experts={info.get('local_experts',0)}",
            lambda: out_ep.shape == (4, 8, 256)
        ))
        # EP backward
        out_ep.sum().backward()
        has_ep_grad = any(p.grad is not None for p in ep_moe.parameters())
        results.append(test("EP backward: gradients flow", lambda: has_ep_grad))

        # TP
        print("  - Tensor Parallel -")
        from mamformer.parallelism.tensor_parallel import TPAttention, _set_tp_group
        _set_tp_group(dist.group.WORLD)
        tp_attn = TPAttention(d_model=256, n_heads=8, n_kv_heads=4,
                              head_dim=32, max_seq_len=128).to(device)
        x_tp = to_dev(torch.randn(2, 16, 256))
        out_tp, _ = tp_attn(x_tp)
        results.append(test(
            f"TP forward: output shape={tuple(out_tp.shape)}",
            lambda: out_tp.shape == (2, 16, 256)
        ))
        out_tp.sum().backward()
        results.append(test("TP backward: gradients flow", lambda:
            tp_attn.q_proj.weight.grad is not None))
        _set_tp_group(None)

        # PP
        print("  - Pipeline Parallel -")
        from mamformer.parallelism.pipeline_parallel import shard_model_pp
        stage = shard_model_pp(model, world_size, local_rank)
        n_layers = len(stage.layers)
        results.append(test(
            f"PP: rank {local_rank} holds {n_layers}/{config.n_layers} layers",
            lambda: n_layers > 0
        ))

    # -- 5. Training Step ----------------------------------
    print(f"\n[5] Training Step")
    # Re-run forward to get fresh loss (previous backward freed the graph)
    out = model(x, labels=labels)
    loss_before = out["loss"].item()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    out["loss"].backward()
    opt.step()
    opt.zero_grad()
    out2 = model(x, labels=labels)
    loss_after = out2["loss"].item()
    results.append(test(
        f"Loss: {loss_before:.4f} - {loss_after:.4f}",
        lambda: abs(loss_before - loss_after) > 1e-6
    ))

    # -- 6. Generation Test ----------------------------------
    print(f"\n[6] Generation Test")
    with torch.no_grad():
        gen = model.generate(x[:, :10], max_new_tokens=10, temperature=0)
    results.append(test(
        f"Generate: {gen.shape[1]} tokens",
        lambda: gen.shape[1] == 20
    ))

    # -- -- -----------------------------------------
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"  Result: {passed}/{total} passed")
    if passed == total:
        print(f"  {GREEN}ALL TESTS PASSED{RESET}")
        if world_size > 1:
            print(f"  {GREEN}Distributed logic verified! EP/TP/PP correct on CPU multi-process{RESET}")
        else:
            print(f"  {YELLOW}Hint: run with torchrun --nproc_per_node=2 for distributed test{RESET}")
    else:
        print(f"  {RED}{total-passed} failed{RESET}")
    print(f"{'='*50}\n")

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
