"""
Mamformer Data Preparation Pipeline
================================
Convert raw text data to tokenized binary format for efficient training.

Supports:
- JSONL files: {"text": "..."}
- Plain text files: one document per line or free-form
- HuggingFace datasets streaming

Usage:
    # Convert JSONL data
    python scripts/prepare_data.py \
        --input data/raw/train.jsonl \
        --output data/tokenized/ \
        --tokenizer llama \
        --seq_len 8192 \
        --num_shards 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mamformer.tokenizer import MamformerTokenizer, load_tokenizer


def read_jsonl(path: str) -> Iterator[str]:
    """Read JSONL file, yielding text fields."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                text = doc.get("text", "")
                if text.strip():
                    yield text
            except json.JSONDecodeError:
                continue


def read_txt(path: str) -> Iterator[str]:
    """Read plain text file. Each non-empty line is a document."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def read_hf_dataset(dataset_name: str, split: str = "train", text_field: str = "text") -> Iterator[str]:
    """Stream from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split=split, streaming=True)
        for item in ds:
            text = item.get(text_field, "")
            if text and str(text).strip():
                yield str(text)
    except ImportError:
        print("Error: 'datasets' package not installed. pip install datasets")
        sys.exit(1)


def tokenize_and_save(
    texts: Iterator[str],
    tokenizer: MamformerTokenizer,
    output_dir: str,
    seq_len: int,
    num_shards: int = 64,
    eos_token: Optional[str] = None,
) -> None:
    """
    Tokenize texts and save as uint16 binary shards.

    Documents are concatenated with EOS tokens as separators,
    then chunked into seq_len+1 segments (for input/label offset).

    Args:
        texts: Iterator yielding text strings
        tokenizer: MamformerTokenizer instance
        output_dir: Directory to save .bin shards
        seq_len: Maximum sequence length for chunks
        num_shards: Number of shard files to create
        eos_token: Optional EOS separator between documents
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_size = 0
    shard_idx = 0
    buffer: List[int] = []

    total_tokens = 0
    total_docs = 0

    # Open first shard
    current_shard_path = output_dir / f"train_{shard_idx:04d}.bin"

    print(f"Tokenizing to {output_dir}/ (seq_len={seq_len}, shards={num_shards})...")

    for text in texts:
        total_docs += 1

        # Tokenize with BOS and EOS
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        total_tokens += len(ids)
        buffer.extend(ids)

        # When buffer has enough tokens, flush chunks
        while len(buffer) >= seq_len + 1:
            chunk = buffer[: seq_len + 1]
            buffer = buffer[seq_len:]

            # Write as uint16 (max vocab 65536) or uint32 for larger vocab
            arr = np.array(chunk, dtype=np.uint16 if tokenizer.vocab_size <= 65536 else np.uint32)
            arr.tofile(current_shard_path)

            shard_size += 1

            # Rotate shards
            if shard_size >= 10000:  # ~10K chunks per shard
                shard_idx = (shard_idx + 1) % num_shards
                current_shard_path = output_dir / f"train_{shard_idx:04d}.bin"
                shard_size = 0

        if total_docs % 10000 == 0:
            print(f"  Processed {total_docs:,} docs, {total_tokens:,} tokens")

    # Flush remaining tokens (pad if needed)
    if len(buffer) > 0:
        if len(buffer) < seq_len + 1:
            buffer += [tokenizer.pad_token_id] * (seq_len + 1 - len(buffer))
        arr = np.array(buffer[: seq_len + 1], dtype=np.uint16 if tokenizer.vocab_size <= 65536 else np.uint32)
        arr.tofile(current_shard_path)
        shard_size += 1

    print(f"\nDone! {total_docs:,} documents -> {total_tokens:,} tokens")
    print(f"Saved {shard_idx + 1} shards to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Prepare Mamformer training data")
    parser.add_argument("--input", type=str, required=True, help="Input file (JSONL/TXT) or HF dataset name")
    parser.add_argument("--input_type", type=str, default="jsonl",
                        choices=["jsonl", "txt", "hf"], help="Input format")
    parser.add_argument("--hf_split", type=str, default="train", help="HF dataset split")
    parser.add_argument("--hf_text_field", type=str, default="text", help="HF text field name")
    parser.add_argument("--output", type=str, required=True, help="Output directory for .bin shards")
    parser.add_argument("--tokenizer", type=str, default="simple",
                        help="Tokenizer name (HuggingFace model name, or 'simple' for fallback)")
    parser.add_argument("--seq_len", type=int, default=8192, help="Sequence length for chunks")
    parser.add_argument("--num_shards", type=int, default=64, help="Number of shard files")
    parser.add_argument("--max_docs", type=int, default=0, help="Max documents to process (0 = all)")

    args = parser.parse_args()

    # Load tokenizer
    if args.tokenizer == "simple":
        print("Using simple character-level tokenizer (for testing only!)")
        print("For real training, use: --tokenizer huggyllama/llama-7b or path to tokenizer.json")
        tokenizer = MamformerTokenizer()
    else:
        print(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = load_tokenizer(args.tokenizer)

    # Read input
    if args.input_type == "jsonl":
        texts = read_jsonl(args.input)
    elif args.input_type == "txt":
        texts = read_txt(args.input)
    elif args.input_type == "hf":
        texts = read_hf_dataset(args.input, args.hf_split, args.hf_text_field)
    else:
        raise ValueError(f"Unknown input type: {args.input_type}")

    # Limit documents if requested
    if args.max_docs > 0:
        from itertools import islice
        texts = islice(texts, args.max_docs)

    # Tokenize and save
    tokenize_and_save(
        texts=texts,
        tokenizer=tokenizer,
        output_dir=args.output,
        seq_len=args.seq_len,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
