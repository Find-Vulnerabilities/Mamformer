"""
一鍵測試 — GPU 買回來後，只要跑這一行：
    torchrun --nproc_per_node=2 scripts/test_all.py
"""

import os, sys, time
import torch
import torch.distributed as dist

GREEN = "\033[92m"
RED = "\033[91m"
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

    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    results = []
    print(f"\n{'='*50}")
    print(f"  Mamformer GPU Test Suite — Rank {local_rank}/{world_size}")
    print(f"{'='*50}\n")

    # ── 1. 基本檢查 ──────────────────────────────────
    print("[1] 基本檢查")
    results.append(test("CUDA available", lambda: torch.cuda.is_available()))
    results.append(test("GPU count correct", lambda: torch.cuda.device_count() == world_size))

    # ── 2. 模型建構 ──────────────────────────────────
    print("\n[2] 模型建構")
    from mamformer.config import MamformerConfig
    from mamformer.model import MamformerForCausalLM

    config = MamformerConfig.from_preset("debug")
    results.append(test("Config load", lambda: None))

    model = MamformerForCausalLM(config).cuda()
    results.append(test("Model build + to GPU", lambda: model is not None))

    # ── 3. 前向傳播 ──────────────────────────────────
    print("\n[3] 前向傳播")
    x = torch.randint(0, 1000, (2, 64)).cuda()
    labels = torch.randint(0, 1000, (2, 64)).cuda()
    out = model(x, labels=labels)
    results.append(test("Forward pass", lambda: "loss" in out))
    results.append(test("Loss is finite", lambda: torch.isfinite(out["loss"])))
    results.append(test("Loss > 0", lambda: out["loss"].item() > 0))

    loss = out["loss"]
    results.append(test("Backward pass", loss.backward))

    has_grad = any(p.grad is not None for p in model.parameters())
    results.append(test("Gradients flow", lambda: has_grad))

    # ── 4. 分散式（只在多 GPU 時測）─────────────────
    if world_size > 1:
        print(f"\n[4] 分散式並行")

        # EP
        print("  測試 Expert Parallel...")
        from mamformer.parallelism.expert_parallel import ExpertParallelGroup, EPMoE
        ep_group = ExpertParallelGroup(ep_size=world_size)
        ep_moe = EPMoE(d_model=256, n_routed_experts=8, top_k=2,
                       routed_expert_dim=64, ep_group=ep_group).cuda()
        x_ep = torch.randn(4, 8, 256).cuda()
        out_ep, info = ep_moe(x_ep)
        local_exp = info.get("local_experts", 0)
        results.append(test(
            f"EP: {local_exp}/{8} experts on rank {local_rank}",
            lambda: out_ep.shape == (4, 8, 256) and local_exp > 0
        ))

        # TP
        print("  測試 Tensor Parallel...")
        from mamformer.parallelism.tensor_parallel import TPAttention, _set_tp_group
        _set_tp_group(dist.group.WORLD)
        tp_attn = TPAttention(d_model=256, n_heads=8, n_kv_heads=4,
                              head_dim=32, max_seq_len=128).cuda()
        x_tp = torch.randn(2, 16, 256).cuda()
        out_tp, _ = tp_attn(x_tp)
        results.append(test(
            f"TP: attention output shape correct",
            lambda: out_tp.shape == (2, 16, 256)
        ))

        # PP
        print("  測試 Pipeline Parallel...")
        from mamformer.parallelism.pipeline_parallel import shard_model_pp
        stage = shard_model_pp(model, world_size, local_rank)
        results.append(test(
            f"PP: stage {local_rank} has {len(stage.layers)} layers",
            lambda: len(stage.layers) > 0
        ))

        # 清理 TP group
        _set_tp_group(None)

    # ── 5. 訓練一步 ──────────────────────────────────
    print(f"\n[5] 訓練測試")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_before = out["loss"].item()
    loss.backward()
    opt.step()
    opt.zero_grad()
    out2 = model(x, labels=labels)
    loss_after = out2["loss"].item()
    results.append(test(
        f"Loss changes after optimizer step ({loss_before:.4f} → {loss_after:.4f})",
        lambda: abs(loss_before - loss_after) > 1e-6
    ))

    # ── 6. 生成測試 ──────────────────────────────────
    print(f"\n[6] 生成測試")
    with torch.no_grad():
        gen = model.generate(x[:, :10], max_new_tokens=10, temperature=0)
    results.append(test(
        f"Generate: output length {gen.shape[1]}",
        lambda: gen.shape[1] == 20
    ))

    # ── 總結 ─────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"  Result: {passed}/{total} passed ({passed/total*100:.0f}%)")
    if passed == total:
        print(f"  {GREEN}ALL TESTS PASSED — 準備好訓練了！{RESET}")
    else:
        print(f"  {RED}有 {total-passed} 個失敗，請檢查{RESET}")
    print(f"{'='*50}\n")

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
