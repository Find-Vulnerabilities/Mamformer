"""
Mamformer Interactive Generation
==============================
Interactive inference script for generating text with Mamformer models.

Usage:
    # Interactive mode
    python scripts/generate.py --config configs/debug.yaml --checkpoint model.pt

    # Single prompt
    python scripts/generate.py --config configs/7b.yaml --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mamformer.config import MamformerConfig
from mamformer.model import MamformerForCausalLM
from mamformer.tokenizer import MamformerTokenizer, load_tokenizer
from mamformer.generation import GenerationConfig


def load_model(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> tuple[MamformerForCausalLM, MamformerTokenizer]:
    """
    Load model and tokenizer from config and checkpoint.

    Args:
        config_path: Path to YAML config file
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model on

    Returns:
        (model, tokenizer)
    """
    print(f"Loading config from {config_path}...")
    config = MamformerConfig.from_yaml(config_path)
    print(config.summary())

    print(f"Initializing model on {device}...")
    model = MamformerForCausalLM(config)
    model = model.to(device)

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}...")
        # weights_only=False: checkpoints may contain optimizer state dicts
        # which are not safe to deserialize with weights_only=True in PyTorch 2.6+
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint (step {checkpoint.get('step', 'unknown')})")

    model.eval()

    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = load_tokenizer()
    except Exception:
        tokenizer = MamformerTokenizer()

    return model, tokenizer


def generate_text(
    model: MamformerForCausalLM,
    tokenizer: MamformerTokenizer,
    prompt: str,
    config: Optional[GenerationConfig] = None,
    device: str = "cpu",
) -> str:
    """
    Generate text from a prompt.

    Args:
        model: MamformerForCausalLM model
        tokenizer: MamformerTokenizer
        prompt: Input text prompt
        config: Generation configuration
        device: Device for tensor operations

    Returns:
        Generated text (including prompt)
    """
    if config is None:
        config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_bos=True)
    input_tensor = torch.tensor([input_ids], device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_tensor,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            eos_token_id=config.eos_token_id,
        )

    # Decode
    generated = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    return generated


def interactive_loop(
    model: MamformerForCausalLM,
    tokenizer: MamformerTokenizer,
    config: GenerationConfig,
    device: str = "cpu",
) -> None:
    """
    Interactive chat-like generation loop.

    Type 'quit' or 'exit' to stop.
    """
    print("\n" + "=" * 60)
    print("  Mamformer Interactive Generation")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        print()
        generated = generate_text(model, tokenizer, prompt, config, device)
        print(generated)
        print()


def main():
    parser = argparse.ArgumentParser(description="Generate text with Mamformer")
    parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)

    args = parser.parse_args()

    # Load model
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    model, tokenizer = load_model(args.config, args.checkpoint, device)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    if args.prompt:
        # Single prompt mode
        generated = generate_text(model, tokenizer, args.prompt, gen_config, device)
        print(generated)
    else:
        # Interactive mode
        interactive_loop(model, tokenizer, gen_config, device)


if __name__ == "__main__":
    main()
