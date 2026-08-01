#!/bin/bash
# ======================================================================
# Mamformer 1B Dense Training — 明天一鍵腳本
# Usage: bash tomorrow.sh
# ======================================================================
set -e

echo "============================================================"
echo "  Mamformer 1B Dense + KDA-Diff + MTP + Interleave"
echo "============================================================"
echo

# Step 1: 下載數據（~1GB, 5分鐘）
echo "[1/3] Downloading FineWeb-Edu (~1B tokens)..."
python scripts/download_data.py \
    --preset debug \
    --output ./data \
    --tokenizer huggyllama/llama-7b \
    --seq_len 2048
echo "[OK] Data downloaded."
echo

# Step 2: 生成 dense 1B config（無 MoE，不會 OOM）
echo "[2/3] Generating dense 1B config with KDA-Diff + MTP + Interleave..."
python -c "
from mamformer.config import MamformerConfig
c = MamformerConfig.from_preset('1b')
c.kda_diff.enabled = True
c.mtp.enabled = True
c.interleave.enabled = True
c.interleave.pattern = 'cross_layer'
c.interleave.attn_every_k = 4
c.interleave.fusion_layers = [22, 23]
c.max_seq_len = 2048
c.to_yaml('configs/_1b_dense_kda.yaml')
print(c.summary())
"
echo "[OK] Config generated."
echo

# Step 3: 訓練（~14000 steps, ~8小時）
echo "[3/3] Starting training..."
echo "  Model:  1B dense + KDA-Diff + MTP + Interleave"
echo "  Steps:  14000"
echo "  Seq:    2048"
echo "  GPU:    H800 80GB"
echo
python scripts/train.py \
    --config configs/_1b_dense_kda.yaml \
    --data ./data/tokenized \
    --bf16 \
    --max_steps 14000 \
    --max_seq_len 2048 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-4 \
    --warmup_steps 500 \
    --save_every 2000 \
    --log_every 50 \
    --output_dir ./checkpoints

echo
echo "============================================================"
echo "  [DONE] Training complete!"
echo "  Checkpoints: ./checkpoints/"
echo "============================================================"
