"""
Mamformer — 中国境内可靠数据下载脚本
=======================================
使用 hf-mirror.com 镜像 + huggingface-cli download
数据源: karpathy/fineweb-edu-100b-shuffle (~100B tokens 英文高质量)

每个 shard ≈ 55M tokens, 下载 20 个 shard ≈ 1.1B tokens, ~2GB

Usage:
    python scripts/download_cn.py                    # 下载 + tokenize
    python scripts/download_cn.py --shards 10        # 只下载 10 个 shard
    python scripts/download_cn.py --shards 50        # 下载 50 个 shard (~2.7B tokens)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── 配置 ──────────────────────────────────────────────────────────
DATASET = "karpathy/fineweb-edu-100b-shuffle"
MIRROR = "https://hf-mirror.com"
TOKENIZER = "huggyllama/llama-7b"
SEQ_LEN = 2048

# ── 下载 ──────────────────────────────────────────────────────────
def download_shards(target_dir: Path, num_shards: int = 20):
    """用 huggingface-cli download 从镜像下载 parquet shards"""
    target_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_ENDPOINT"] = MIRROR

    print(f"Downloading {num_shards} shards from {DATASET}")
    print(f"Mirror: {MIRROR}")

    for i in range(num_shards):
        filename = f"shard_{i:05d}.parquet"
        url = f"{MIRROR}/datasets/{DATASET}/resolve/main/{filename}"
        out_path = target_dir / filename

        if out_path.exists():
            size_mb = out_path.stat().st_size / (1024 * 1024)
            if size_mb > 10:  # >10MB, assume complete
                print(f"  [{i+1}/{num_shards}] {filename} already exists ({size_mb:.0f}MB), skip")
                continue
            else:
                out_path.unlink()  # corrupted, re-download

        # Use wget for simplicity (available on all AutoDL instances)
        print(f"  [{i+1}/{num_shards}] Downloading {filename}...")
        ret = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(out_path), url],
            env=env, timeout=300,
        )

        if ret.returncode != 0:
            # Fallback: try curl
            print(f"  wget failed, trying curl...")
            ret = subprocess.run(
                ["curl", "-L", "-o", str(out_path), url],
                env=env, timeout=300,
            )

        if ret.returncode != 0:
            print(f"  [WARN] Failed to download {filename}, skipping...")
            if out_path.exists():
                out_path.unlink()

    # Count successfully downloaded
    downloaded = list(target_dir.glob("*.parquet"))
    total_mb = sum(f.stat().st_size for f in downloaded) / (1024 * 1024)
    print(f"\nDownloaded {len(downloaded)}/{num_shards} shards ({total_mb:.0f}MB)")
    return downloaded


# ── Tokenize ───────────────────────────────────────────────────────
def tokenize_shards(parquet_files: list, output_dir: Path, seq_len: int = 2048):
    """读取 parquet 文件, tokenize, 存成 .bin shards"""
    import pandas as pd
    from transformers import AutoTokenizer

    print(f"\nLoading tokenizer: {TOKENIZER}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Delete old .bin files
    for old in output_dir.glob("*.bin"):
        old.unlink()

    buffer = []
    shard_idx = 0
    total_tokens = 0

    print(f"Tokenizing {len(parquet_files)} parquet files...")

    for pf in sorted(parquet_files):
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            text = str(row.get("text", ""))
            if len(text) < 100:
                continue

            tokens = tokenizer.encode(text, add_special_tokens=True)
            total_tokens += len(tokens)
            buffer.extend(tokens)

            while len(buffer) >= seq_len + 1:
                chunk = buffer[:seq_len + 1]
                buffer = buffer[seq_len:]
                arr = np.array(chunk, dtype=np.uint16)
                arr.tofile(output_dir / f"train_{shard_idx:04d}.bin")
                shard_idx += 1

    # Flush remaining
    if len(buffer) >= 10:
        chunk = list(buffer) + [0] * (seq_len + 1 - len(buffer))
        arr = np.array(chunk, dtype=np.uint16)
        arr.tofile(output_dir / f"train_{shard_idx:04d}.bin")
        shard_idx += 1

    print(f"Tokenized: {total_tokens / 1e6:.0f}M tokens → {shard_idx} shards")
    print(f"Output: {output_dir}/")
    return shard_idx


# ── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download English data via hf-mirror")
    parser.add_argument("--shards", type=int, default=20,
                       help="Number of parquet shards to download (default: 20, ~1.1B tokens)")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--output", type=str, default="./data")
    parser.add_argument("--skip_download", action="store_true",
                       help="Skip download, only tokenize existing parquet files")
    args = parser.parse_args()

    output_dir = Path(args.output)
    parquet_dir = output_dir / "parquet"
    tokenized_dir = output_dir / "tokenized"

    print("=" * 60)
    print("  Mamformer Data Downloader (China Mirror)")
    print("=" * 60)
    print(f"  Dataset:  {DATASET}")
    print(f"  Mirror:   {MIRROR}")
    print(f"  Shards:   {args.shards} (~{args.shards * 55}M tokens)")
    print(f"  Seq len:  {args.seq_len}")
    print(f"  Output:   {tokenized_dir.resolve()}")
    print("=" * 60)
    print()

    # Step 1: Install dependencies
    print("[1/3] Checking dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow", "transformers", "tqdm"],
                   check=False)

    # Step 2: Download
    if not args.skip_download:
        print("\n[2/3] Downloading parquet files...")
        parquet_files = download_shards(parquet_dir, args.shards)
        if not parquet_files:
            print("[ERR] No files downloaded. Check network or try again.")
            sys.exit(1)
    else:
        parquet_files = sorted(parquet_dir.glob("*.parquet"))
        print(f"\n[2/3] Using {len(parquet_files)} existing parquet files...")

    # Step 3: Tokenize
    print("\n[3/3] Tokenizing...")
    n = tokenize_shards(parquet_files, tokenized_dir, args.seq_len)

    if n > 0:
        print(f"\n[DONE] {n} training shards ready in {tokenized_dir}/")
        print(f"  Ready for training:")
        print(f"  python scripts/train.py --config configs/_1b_dense_kda.yaml --data {tokenized_dir} --bf16 ...")
    else:
        print("[ERR] Tokenization produced no output!")


if __name__ == "__main__":
    main()
