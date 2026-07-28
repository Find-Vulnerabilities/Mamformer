"""
Mamformer Auto Data Downloader
==============================
One-click download + tokenize for cloud training instances.

Usage:
    # Download ~10GB for quick start (FineWeb-Edu sample)
    python scripts/download_data.py --preset quick --output ./data

    # Download ~100GB for decent 7B pretraining
    python scripts/download_data.py --preset standard --output ./data

    # Download ~500GB+ for competitive pretraining
    python scripts/download_data.py --preset full --output ./data

    # Custom: specify sources and token budget
    python scripts/download_data.py --sources fineweb,c4,slimpajama --budget 50B --output ./data

Sources (all free, no auth required):
    fineweb    - FineWeb-Edu (high-quality filtered web, CC-main)
    c4         - C4 (cleaned Common Crawl)
    slimpajama - SlimPajama-627B (cleaned, deduped)
    dclm       - DCLM (DataComp-LM, research-grade)
    pile       - The Pile (academic standard)
    starcoder  - The Stack (code)
    wikipedia  - Wikipedia (multilingual)
    books      - Project Gutenberg + BookCorpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ── Presets: estimated raw text → tokenized size ──────────────────────
# Tokens ≈ raw_bytes / 4 (roughly, depends on tokenizer)
# Chinchilla optimal: ~20× params for dense, ~20× active_params for MoE
PRESETS = {
    # ── Debug / Verify ──────────────────────────────────────────
    "debug": {
        "desc": "~1B tokens — debug/arch verify (configs: debug, arch-verify)",
        "for_models": "debug, arch-verify (0.01B~0.1B)",
        "sources": {"fineweb": {"split": "train", "max_rows": 50_000}},
        "approx_tokens": 1_000_000_000,
        "approx_raw_gb": 1,
    },
    # ── 1B models ───────────────────────────────────────────────
    "1b": {
        "desc": "~20B tokens — Chinchilla for 1B dense",
        "for_models": "1b preset",
        "sources": {
            "fineweb": {"split": "train", "max_rows": 500_000},
            "c4": {"split": "train", "max_rows": 300_000},
            "slimpajama": {"split": "train", "max_rows": 300_000},
        },
        "approx_tokens": 20_000_000_000,
        "approx_raw_gb": 16,
    },
    # ── 7B Dense ────────────────────────────────────────────────
    "7b": {
        "desc": "~140B tokens — Chinchilla for 7B dense (~6.5B params)",
        "for_models": "7b.yaml, pro-7b.yaml, ultra-7b-dense.yaml",
        "sources": {
            "fineweb": {"split": "train", "max_rows": 3_000_000},
            "c4": {"split": "train", "max_rows": 2_000_000},
            "dclm": {"split": "train", "max_rows": 1_500_000},
            "slimpajama": {"split": "train", "max_rows": 2_500_000},
            "starcoder": {"split": "train", "max_rows": 500_000},
            "wikipedia": {"split": "train", "max_rows": 500_000},
        },
        "approx_tokens": 140_000_000_000,
        "approx_raw_gb": 110,
    },
    # ── Ultra 7B (MoE: ~39B total / ~7.2B active) ───────────────
    "ultra-7b": {
        "desc": "~140B tokens — Chinchilla for 7.2B active MoE (~39B total)",
        "for_models": "ultra-7b.yaml",
        "sources": {
            "fineweb": {"split": "train", "max_rows": 3_000_000},
            "c4": {"split": "train", "max_rows": 2_000_000},
            "dclm": {"split": "train", "max_rows": 1_500_000},
            "slimpajama": {"split": "train", "max_rows": 2_500_000},
            "starcoder": {"split": "train", "max_rows": 500_000},
            "wikipedia": {"split": "train", "max_rows": 500_000},
        },
        "approx_tokens": 140_000_000_000,
        "approx_raw_gb": 110,
    },
    # ── Ultra 37B (MoE: ~200B total / ~37B active) ──────────────
    "ultra-37b": {
        "desc": "~740B tokens — Chinchilla for 37B active MoE (~200B total)",
        "for_models": "ultra-37b.yaml",
        "sources": {
            "fineweb": {"split": "train", "max_rows": 0},        # 0 = all
            "c4": {"split": "train", "max_rows": 0},
            "dclm": {"split": "train", "max_rows": 0},
            "slimpajama": {"split": "train", "max_rows": 0},
            "starcoder": {"split": "train", "max_rows": 0},
            "wikipedia": {"split": "train", "max_rows": 0},
        },
        "approx_tokens": 740_000_000_000,
        "approx_raw_gb": 600,
    },
    # ── Ultra 671B (MoE: ~671B total / ~37B active) ─────────────
    "ultra-671b": {
        "desc": "~1T tokens — DeepSeek-V3 scale for 37B active MoE",
        "for_models": "ultra-671b-max.yaml (also ultra-371b.yaml)",
        "sources": {
            "fineweb": {"split": "train", "max_rows": 0},
            "c4": {"split": "train", "max_rows": 0},
            "dclm": {"split": "train", "max_rows": 0},
            "slimpajama": {"split": "train", "max_rows": 0},
            "starcoder": {"split": "train", "max_rows": 0},
            "wikipedia": {"split": "train", "max_rows": 0},
        },
        "approx_tokens": 1_000_000_000_000,
        "approx_raw_gb": 800,
    },
}

# ── Dataset registry ──────────────────────────────────────────────────
DATASETS = {
    "fineweb": {
        "name": "HuggingFaceFW/fineweb-edu",
        "text_field": "text",
        "desc": "FineWeb-Edu (high-quality filtered CC)",
    },
    "c4": {
        "name": "allenai/c4",
        "text_field": "text",
        "desc": "C4 (Colossal Clean Crawled Corpus)",
    },
    "slimpajama": {
        "name": "cerebras/SlimPajama-627B",
        "text_field": "text",
        "desc": "SlimPajama (cleaned + deduped RedPajama)",
    },
    "dclm": {
        "name": "mlfoundations/dclm-baseline-1.0",
        "text_field": "text",
        "desc": "DCLM (DataComp for Language Models)",
    },
    "pile": {
        "name": "EleutherAI/pile",
        "text_field": "text",
        "desc": "The Pile (825GB academic dataset)",
    },
    "starcoder": {
        "name": "bigcode/the-stack-dedup",
        "text_field": "content",
        "desc": "The Stack v2 (permissively licensed code)",
    },
    "wikipedia": {
        "name": "wikimedia/wikipedia",
        "text_field": "text",
        "desc": "Wikipedia (multilingual)",
    },
    "books": {
        "name": "bookcorpusopen/bookcorpusopen",
        "text_field": "text",
        "desc": "BookCorpusOpen",
    },
}


def check_disk_space(path: str, required_gb: float) -> bool:
    """Check if enough disk space is available."""
    try:
        import shutil
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < required_gb:
            print(f"  ⚠ WARNING: only {free_gb:.0f}GB free, need ~{required_gb:.0f}GB")
            return False
        print(f"  ✅ Disk: {free_gb:.0f}GB free (need ~{required_gb:.0f}GB)")
        return True
    except Exception:
        return True  # can't check, assume ok


def check_deps() -> bool:
    """Verify required packages are installed."""
    missing = []
    for pkg in ["datasets", "tqdm"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  ❌ Missing packages: {', '.join(missing)}")
        print(f"  Run: pip install {' '.join(missing)}")
        return False

    # Check tokenizer dependency
    try:
        __import__("transformers")
    except ImportError:
        print("  ⚠ 'transformers' not installed (needed for tokenizer)")
        print("  Run: pip install transformers")
        return False
    return True


def download_dataset(
    dataset_id: str,
    config: dict,
    output_dir: Path,
    seq_len: int,
    tokenizer_name: str,
) -> dict:
    """
    Download one dataset, tokenize on-the-fly, save as .bin shards.

    Returns stats dict with token count, file count, etc.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import numpy as np
    from tqdm import tqdm

    ds_info = DATASETS[dataset_id]
    ds_name = ds_info["name"]
    text_field = ds_info["text_field"]
    split = config.get("split", "train")
    max_rows = config.get("max_rows", 0)
    max_rows = max_rows if max_rows > 0 else None

    print(f"\n  📥 Downloading: {ds_name}")
    print(f"     Source: {ds_info['desc']}")
    print(f"     Split: {split}" + (f", max rows: {max_rows:,}" if max_rows else " (all)"))

    # Load tokenizer
    print(f"     Loading tokenizer: {tokenizer_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset (streaming to avoid disk usage)
    print(f"     Streaming from HuggingFace ...")
    try:
        ds = load_dataset(ds_name, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        # Try with default config
        try:
            ds = load_dataset(ds_name, split=split, streaming=True)
        except Exception:
            print(f"     ⚠ Failed to load {ds_name}: {e}")
            print(f"     Skipping this source...")
            return {"tokens": 0, "docs": 0, "shards": 0, "skipped": True}

    # Prepare output shards
    dataset_output = output_dir / dataset_id
    dataset_output.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    shard_tokens = 0
    tokens_per_shard = 1_000_000  # ~1M tokens per .bin shard
    shard_buffer = []
    total_tokens = 0
    total_docs = 0
    shard_files = []

    shard_path = dataset_output / f"shard_{shard_idx:04d}.bin"
    shard_fh = open(shard_path, "wb")

    pbar = tqdm(desc=f"     Tokenizing", unit="docs")

    try:
        for i, row in enumerate(ds):
            if max_rows and i >= max_rows:
                break

            text = str(row.get(text_field, ""))
            if not text or len(text) < 50:
                continue

            total_docs += 1

            # Tokenize with BOS/EOS
            try:
                tokens = tokenizer.encode(text, add_special_tokens=True)
            except Exception:
                continue

            if len(tokens) < 10:
                continue

            total_tokens += len(tokens)
            shard_buffer.extend(tokens)

            # Flush chunks of seq_len+1
            while len(shard_buffer) >= seq_len + 1:
                chunk = shard_buffer[: seq_len + 1]
                shard_buffer = shard_buffer[seq_len:]

                arr = np.array(chunk, dtype=np.uint16 if tokenizer.vocab_size < 65536 else np.uint32)
                arr.tofile(shard_fh)
                shard_tokens += seq_len

                # Rotate shard
                if shard_tokens >= tokens_per_shard:
                    shard_fh.close()
                    shard_files.append(str(shard_path))
                    shard_idx += 1
                    shard_tokens = 0
                    shard_path = dataset_output / f"shard_{shard_idx:04d}.bin"
                    shard_fh = open(shard_path, "wb")

            pbar.update(1)

            # Progress report
            if total_docs % 5000 == 0:
                pbar.set_postfix({
                    "tokens": f"{total_tokens / 1e6:.0f}M",
                    "shards": shard_idx + 1,
                })

    except KeyboardInterrupt:
        print("\n     ⚠ Interrupted — saving progress...")
    except Exception as e:
        print(f"\n     ⚠ Error: {e}")
    finally:
        pbar.close()
        # Flush remaining buffer
        if shard_buffer:
            # Pad final chunk
            chunk = list(shard_buffer) + [0] * (seq_len + 1 - len(shard_buffer))
            arr = np.array(chunk, dtype=np.uint16 if tokenizer.vocab_size < 65536 else np.uint32)
            arr.tofile(shard_fh)
        shard_fh.close()
        if shard_tokens > 0 or not shard_files:
            shard_files.append(str(shard_path))

    # Remove empty shards
    for sf in shard_files:
        if os.path.getsize(sf) < 100:
            os.remove(sf)
    shard_files = [sf for sf in shard_files if os.path.exists(sf)]

    # Write metadata
    meta = {
        "dataset": ds_name,
        "docs": total_docs,
        "tokens": total_tokens,
        "shards": len(shard_files),
        "seq_len": seq_len,
        "dtype": "uint16" if tokenizer.vocab_size < 65536 else "uint32",
    }
    with open(dataset_output / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"     ✅ Done: {total_docs:,} docs → {total_tokens / 1e6:.0f}M tokens → {len(shard_files)} shards")
    return {"tokens": total_tokens, "docs": total_docs, "shards": len(shard_files)}


def merge_shards(output_dir: Path, delete_sources: bool = False) -> Path:
    """Merge all dataset shards into a single tokenized/ directory for training."""
    merged = output_dir / "tokenized"
    merged.mkdir(parents=True, exist_ok=True)

    # Symlink all .bin files into one flat directory with unique names
    all_bins = sorted(output_dir.rglob("*.bin"))
    # Filter: only keep actual data shards (not inside tokenized/)
    all_bins = [b for b in all_bins if "tokenized" not in str(b)]

    if not all_bins:
        print("  ⚠ No .bin shards found to merge!")
        return merged

    print(f"\n  🔗 Merging {len(all_bins)} shards from {len(set(b.parent for b in all_bins))} sources...")

    for i, bin_path in enumerate(all_bins):
        # Create symlink: tokenized/train_XXXX.bin → source shard
        link_name = merged / f"train_{i:04d}.bin"
        if link_name.exists():
            link_name.unlink()
        try:
            os.symlink(bin_path.resolve(), link_name)
        except OSError:
            # Fallback: copy
            import shutil
            shutil.copy2(bin_path, link_name)

    # Write merged metadata
    total_tokens = 0
    total_docs = 0
    for meta_path in sorted(output_dir.rglob("metadata.json")):
        if "tokenized" in str(meta_path):
            continue
        try:
            with open(meta_path) as f:
                m = json.load(f)
            total_tokens += m.get("tokens", 0)
            total_docs += m.get("docs", 0)
        except Exception:
            pass

    meta = {
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "num_shards": len(all_bins),
        "sources": [b.parent.name for b in all_bins],
    }
    with open(merged / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✅ Merged: {total_tokens / 1e9:.1f}B tokens in {len(all_bins)} shards → {merged}/")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Mamformer Auto Data Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_data.py --preset quick
  python scripts/download_data.py --preset standard --output /data/mamformer
  python scripts/download_data.py --sources fineweb,c4 --budget 50B
  python scripts/download_data.py --preset full --tokenizer meta-llama/Llama-2-7b-hf
        """,
    )
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()),
                       help="Download preset (quick/standard/full/max)")
    parser.add_argument("--sources", type=str,
                       help="Comma-separated dataset names (fineweb,c4,slimpajama,dclm,pile,starcoder,wikipedia,books)")
    parser.add_argument("--budget", type=str, default="",
                       help="Token budget (e.g. 10B, 50B, 1T). Overrides preset max_rows.")
    parser.add_argument("--output", type=str, default="./data",
                       help="Output directory (default: ./data)")
    parser.add_argument("--tokenizer", type=str, default="huggyllama/llama-7b",
                       help="Tokenizer name/path (default: huggyllama/llama-7b)")
    parser.add_argument("--seq_len", type=int, default=8192,
                       help="Sequence length for chunking (default: 8192)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show plan without downloading")
    parser.add_argument("--list_sources", action="store_true",
                       help="List available data sources and exit")

    args = parser.parse_args()

    # ── List sources ──────────────────────────────────────────────
    if args.list_sources:
        print("\n  Available Data Sources:\n")
        for key, info in DATASETS.items():
            print(f"  {key:15s}  {info['desc']}")
            print(f"  {'':15s}  HF: {info['name']}")
            print()
        print("  Presets:")
        for key, preset in PRESETS.items():
            print(f"  {key:15s}  {preset['desc']}")
        return

    # ── Resolve what to download ──────────────────────────────────
    if args.preset:
        preset = PRESETS[args.preset]
        sources = preset["sources"]
        desc = preset["desc"]
        approx_raw_gb = preset["approx_raw_gb"]
        approx_tokens = preset["approx_tokens"]
    elif args.sources:
        source_list = [s.strip() for s in args.sources.split(",")]
        sources = {s: {"split": "train", "max_rows": 0} for s in source_list}
        desc = f"Custom: {', '.join(source_list)}"
        approx_raw_gb = 20  # guess
        approx_tokens = 25_000_000_000
    else:
        parser.error("Must specify --preset or --sources")

    # Override with budget if given
    if args.budget:
        budget_str = args.budget.upper().replace(" ", "")
        multipliers = {"B": 1e9, "T": 1e12, "M": 1e6}
        for suffix, mult in multipliers.items():
            if budget_str.endswith(suffix):
                budget_tokens = int(float(budget_str[:-1]) * mult)
                break
        else:
            budget_tokens = int(budget_str)
        approx_tokens = budget_tokens
        # Adjust: set all sources to "all" (max_rows=0)
        for s in sources:
            sources[s]["max_rows"] = 0

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Show plan ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Mamformer Auto Data Downloader")
    print("=" * 60)
    print(f"  Plan:     {desc}")
    print(f"  Target:   ~{approx_tokens / 1e9:.0f}B tokens")
    print(f"  Est disk: ~{approx_raw_gb:.0f}GB (raw) + ~{approx_tokens / 1e9 * 0.4:.0f}GB (tokenized)")
    print(f"  Output:   {output_dir.resolve()}")
    print(f"  Tokenizer:{args.tokenizer}")
    print(f"  Seq len:  {args.seq_len}")
    print()
    print("  Sources:")
    for s, cfg in sources.items():
        info = DATASETS.get(s, {})
        max_r = cfg.get("max_rows", 0)
        limit_str = f"{max_r:,} rows" if max_r else "ALL rows"
        print(f"    - {s}: {info.get('desc', '?')} ({limit_str})")
    print("=" * 60)

    if args.dry_run:
        print("\n  Dry run — nothing downloaded.")
        return

    # ── Check prerequisites ───────────────────────────────────────
    print("\n  🔍 Checking environment...")
    if not check_deps():
        sys.exit(1)
    check_disk_space(str(output_dir), approx_raw_gb * 1.5)

    # ── Confirm ───────────────────────────────────────────────────
    print()
    resp = input(f"  Proceed with download? [Y/n]: ").strip().lower()
    if resp and resp != "y":
        print("  Cancelled.")
        return

    # ── Download each source ──────────────────────────────────────
    total_tokens = 0
    total_docs = 0
    all_ok = True

    for dataset_id, cfg in sources.items():
        if dataset_id not in DATASETS:
            print(f"\n  ⚠ Unknown source '{dataset_id}' — skipping")
            continue

        try:
            stats = download_dataset(
                dataset_id=dataset_id,
                config=cfg,
                output_dir=output_dir,
                seq_len=args.seq_len,
                tokenizer_name=args.tokenizer,
            )
            if not stats.get("skipped"):
                total_tokens += stats["tokens"]
                total_docs += stats["docs"]
        except Exception as e:
            print(f"\n  ❌ Failed to download '{dataset_id}': {e}")
            print(f"     Continuing with remaining sources...")
            all_ok = False

    # ── Merge into flat tokenized/ directory ─────────────────────
    print("\n" + "=" * 60)
    print("  Merging all shards...")
    merged_dir = merge_shards(output_dir)

    # ── Summary ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  ✅ Download Complete!")
    print("=" * 60)
    print(f"  Total docs:    {total_docs:,}")
    print(f"  Total tokens:  {total_tokens / 1e9:.2f}B")
    print(f"  Tokenized dir: {merged_dir.resolve()}")
    print()
    print(f"  Ready for training:")
    print(f"    python scripts/train.py --config configs/7b.yaml --data {merged_dir} --bf16")
    print("=" * 60)


if __name__ == "__main__":
    main()
