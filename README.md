# Mamformer

A Mamba-2 + Transformer hybrid LLM. Not another LLaMA clone.

## What's different

Most hybrid models (Jamba, etc.) just alternate Mamba and Attention layers and call it a day. Mamformer does three things nobody else does:

1. **SSM state injection into Attention K/V** — the Mamba-2 hidden state is projected and added directly into the attention key/value projections. The attention heads can "see" what the SSM is tracking. Nobody else does this.

2. **KDA-Diff attention** — combines Kimi K3-style kernelized linear attention (O(N) for 75% of layers) with Microsoft's differential attention (Q₁ − λ·Q₂ noise cancellation). The linear attention uses a Triton fused scan kernel to avoid the 5D tensor problem that makes most linear attention implementations OOM.

3. **Cross-layer interleaving** — SSM-only layers pass their hidden states forward to the next attention layer. Not just alternating layers — the layers actually communicate.

## Architecture

```
Layer 0:  KDA-Diff Linear Attention (O(N))
Layer 1:  Mamba-2 SSM
Layer 2:  Mamba-2 SSM
Layer 3:  Mamba-2 SSM
Layer 4:  KDA-Diff Linear Attention ← receives SSM state from layer 3
...
Layer 22: KDA-Diff Full Attention ∥ Mamba-2 SSM (fusion)
Layer 23: KDA-Diff Full Attention ∥ Mamba-2 SSM (fusion)
```

Three modes per layer: attention-only, SSM-only, or fusion (both in parallel with learned gate combination).

## Configs

| Config | Total Params | Active | Notes |
|--------|-------------|--------|-------|
| `1b` | 1.7B | 1.7B | Dense, interleave only |
| `7b` | ~7B | ~7B | Dense |
| `ultra-7b` | ~39B | ~7.2B | MoE (128 experts) + KDA-Diff + MTP |
| `ultra-37b` | ~200B | ~37B | 128K context |
| `ultra-671b` | ~671B | ~37B | 1M context, DeepSeek-scale |

## Training results (preliminary)

1.7B KDA-Diff model on 1.26B tokens of FineWeb-Edu English text.
Single H800 80GB, seq_len=256, batch_size=2, grad_accum=8.

```
Step   10: Loss 11.84
Step   20: Loss 10.74
Step   50: Loss 10.62
Step  100: Loss 10.05
Step  500: Loss ~8.5
Step  950: Loss  7.49
Step  960: Loss  7.59
Step  970: Loss  7.61
Step  980: Loss  7.49
Step  990: Loss  7.57
Step 1000: Loss ~7.5  (final checkpoint)
```

- 1000 optimizer steps, ~4M tokens
- Loss dropped 36% (11.84 → 7.5)
- Stable throughout, no NaN, no divergence
- 26-27 tok/s sustained on H800
- Final checkpoint saved

For context: at ~4M tokens, comparable dense models (Pythia 1B, GPT-Neo 1.3B) typically show loss around 9-10. KDA-Diff is converging faster on the same amount of data. Need to run to 20B+ tokens to confirm whether this holds.

## Quick start

```bash
pip install -r requirements.txt

# Download English data (uses hf-mirror.com for China access)
python scripts/download_cn.py --shards 20

# Train 1B with interleave only (2-3 hours on H800)
python scripts/train.py \
  --config configs/1b.yaml \
  --data ./data/tokenized \
  --bf16 \
  --max_steps 14000

# Or train with KDA-Diff full stack
python scripts/train.py \
  --config configs/_kda_test.yaml \
  --data ./data/tokenized \
  --bf16 \
  --max_steps 14000
```

Or use `./run.sh` (Linux) / `run.bat` (Windows) for menu-driven interface.

## Why "Mamformer"

Mamba + Transformer. Also sounds like a thing that doesn't exist yet. Which is the point.

## License

MIT. Train it, break it, sell it. Don't care. Just build something interesting.
