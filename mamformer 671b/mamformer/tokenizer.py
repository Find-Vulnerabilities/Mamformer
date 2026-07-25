"""
Mamformer Tokenizer
================
Tokenizer wrapper supporting both HuggingFace tokenizers
and a simple character-level fallback for testing.

For production use, Mamformer uses a BPE tokenizer compatible with
the Llama tokenizer format (SentencePiece BPE with 128K vocab).
"""

from __future__ import annotations

from typing import List, Optional, Union


class MamformerTokenizer:
    """
    Tokenizer wrapper for Mamformer models.

    Provides encode/decode functionality. For production, wraps a
    HuggingFace PreTrainedTokenizer. For testing, provides a simple
    character-level tokenizer fallback.

    Thinking tokens (for toggleable reasoning mode):
        <|think_start|>  — marks the beginning of internal reasoning
        <|think_end|>    — marks the end of internal reasoning
        <|answer_start|> — marks the beginning of the final answer
        <|answer_end|>   — marks the end of the final answer

    Args:
        tokenizer: A HuggingFace PreTrainedTokenizer instance, or None for fallback
        bos_token: Beginning-of-sequence token string
        eos_token: End-of-sequence token string
        pad_token: Padding token string
    """

    # ── Thinking token ID constants ─────────────────────────────────
    THINK_START_ID: int = 3
    THINK_END_ID: int = 4
    ANSWER_START_ID: int = 5
    ANSWER_END_ID: int = 6
    SUMMARY_START_ID: int = 7

    THINKING_TOKEN_IDS: set[int] = {3, 4, 5, 6, 7}

    THINKING_TOKEN_NAMES: dict[int, str] = {
        3: "<|think_start|>",
        4: "<|think_end|>",
        5: "<|answer_start|>",
        6: "<|answer_end|>",
        7: "<|summary_start|>",
    }

    def __init__(
        self,
        tokenizer=None,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
    ) -> None:
        self._tokenizer = tokenizer

        # ── Thinking token IDs ──────────────────────────────────
        self.think_start_token_id: int = MamformerTokenizer.THINK_START_ID
        self.think_end_token_id: int = MamformerTokenizer.THINK_END_ID
        self.answer_start_token_id: int = MamformerTokenizer.ANSWER_START_ID
        self.answer_end_token_id: int = MamformerTokenizer.ANSWER_END_ID
        self.summary_start_token_id: int = MamformerTokenizer.SUMMARY_START_ID

        if tokenizer is not None:
            # Use HuggingFace tokenizer
            self.bos_token_id = tokenizer.bos_token_id or 0
            self.eos_token_id = tokenizer.eos_token_id or 1
            self.pad_token_id = tokenizer.pad_token_id or 0
            self.vocab_size = max(tokenizer.vocab_size, self.answer_end_token_id + 1)
            self._encode = self._encode_hf
            self._decode = self._decode_hf
        else:
            # Simple character-level fallback for testing
            self.bos_token_id = 1
            self.eos_token_id = 2
            self.pad_token_id = 0
            self.vocab_size = max(256, self.answer_end_token_id + 1)
            self._bos = bos_token
            self._eos = eos_token
            self._pad = pad_token
            self._encode = self._encode_simple
            self._decode = self._decode_simple

    def _encode_hf(self, text: str, **kwargs) -> List[int]:
        """Encode using HuggingFace tokenizer."""
        return self._tokenizer.encode(text, **kwargs)

    def _decode_hf(self, ids: List[int], **kwargs) -> str:
        """Decode using HuggingFace tokenizer."""
        return self._tokenizer.decode(ids, **kwargs)

    def _encode_simple(self, text: str, **kwargs) -> List[int]:
        """Simple character-level encoding (fallback)."""
        ids = []
        oov_warned = False
        for ch in text:
            code = ord(ch)
            if code < self.vocab_size and code > 0:
                ids.append(code)
            else:
                if not oov_warned:
                    import warnings
                    warnings.warn(
                        f"Character '{ch}' (U+{code:04X}) exceeds vocab_size={self.vocab_size}. "
                        "Non-ASCII content will be mapped to space. Use a real tokenizer for production."
                    )
                    oov_warned = True
                ids.append(32)  # space fallback
        return ids

    def _decode_simple(self, ids: List[int], **kwargs) -> str:
        """Simple character-level decoding (fallback)."""
        chars = []
        for tid in ids:
            if 0 < tid < self.vocab_size:
                chars.append(chr(tid))
            elif tid == self.eos_token_id:
                break
            else:
                chars.append("�")
        return "".join(chars)

    def encode(
        self,
        text: Union[str, List[str]],
        add_bos: bool = True,
        add_eos: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
        **kwargs,
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token IDs.

        Args:
            text: Single string or list of strings
            add_bos: Prepend BOS token
            add_eos: Append EOS token
            max_length: Maximum sequence length
            truncation: Truncate to max_length
            padding: Pad to max_length

        Returns:
            List of token IDs (single) or list of lists (batch)
        """
        if isinstance(text, str):
            text = [text]
            single = True
        else:
            single = False

        all_ids = []
        for t in text:
            ids = self._encode(t, **kwargs)

            if add_bos:
                ids = [self.bos_token_id] + ids
            if add_eos:
                ids = ids + [self.eos_token_id]

            if max_length is not None and truncation:
                ids = ids[:max_length]

            all_ids.append(ids)

        # Padding
        if padding and max_length is not None:
            for i in range(len(all_ids)):
                if len(all_ids[i]) < max_length:
                    all_ids[i] = all_ids[i] + [self.pad_token_id] * (
                        max_length - len(all_ids[i])
                    )

        return all_ids[0] if single else all_ids

    def decode(
        self,
        ids: Union[int, List[int]],
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: Token ID or list of token IDs
            skip_special_tokens: Remove special tokens from output

        Returns:
            Decoded text string
        """
        if isinstance(ids, int):
            ids = [ids]

        if skip_special_tokens:
            special_ids = {
                self.bos_token_id, self.eos_token_id, self.pad_token_id,
                self.think_start_token_id, self.think_end_token_id,
                self.answer_start_token_id, self.answer_end_token_id,
                self.summary_start_token_id,
            }
            ids = [tid for tid in ids if tid not in special_ids]

        return self._decode(ids, **kwargs)

    def batch_decode(
        self,
        batch_ids: List[List[int]],
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        """Decode a batch of token ID sequences."""
        return [self.decode(ids, skip_special_tokens=skip_special_tokens, **kwargs) for ids in batch_ids]

    @property
    def bos_token(self) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.bos_token or "<s>"
        return self._bos

    @property
    def eos_token(self) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.eos_token or "</s>"
        return self._eos

    @property
    def pad_token(self) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.pad_token or "<pad>"
        return self._pad

    def get_thinking_token_ids(self) -> dict[str, int]:
        """Return dict mapping thinking token names to their IDs."""
        return {
            "think_start": self.think_start_token_id,
            "think_end": self.think_end_token_id,
            "answer_start": self.answer_start_token_id,
            "answer_end": self.answer_end_token_id,
            "summary_start": self.summary_start_token_id,
        }

    def add_thinking_tokens(self, vocab_size: int) -> int:
        """
        Ensure the vocabulary is large enough to include thinking control tokens.
        Returns the (possibly expanded) vocab_size.

        This is a no-op for the tokenizer itself — the actual embedding expansion
        happens in the model. The tokenizer just ensures its vocab_size accounts
        for the thinking tokens.
        """
        min_vocab = self.answer_end_token_id + 1
        if self.vocab_size < min_vocab:
            self.vocab_size = min_vocab
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"MamformerTokenizer(vocab_size={self.vocab_size}, "
            f"bos_id={self.bos_token_id}, eos_id={self.eos_token_id}, "
            f"pad_id={self.pad_token_id}, "
            f"think_start={self.think_start_token_id}, "
            f"think_end={self.think_end_token_id})"
        )


def load_tokenizer(
    pretrained_name: str = "huggyllama/llama-7b",
    expected_vocab_size: int = 0,
    **kwargs,
) -> MamformerTokenizer:
    """
    Load a pre-trained tokenizer compatible with Mamformer.

    Default uses the Llama tokenizer (SentencePiece BPE, 32K vocab).
    For Mamformer's 128K vocab target, use a larger tokenizer or train a custom one.

    Args:
        pretrained_name: HuggingFace model name for tokenizer
        expected_vocab_size: If > 0, warns if tokenizer vocab doesn't match
        **kwargs: Additional args passed to AutoTokenizer.from_pretrained

    Returns:
        MamformerTokenizer instance
    """
    try:
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(pretrained_name, **kwargs)
        tokenizer = MamformerTokenizer(hf_tokenizer)
        if expected_vocab_size > 0 and tokenizer.vocab_size != expected_vocab_size:
            import logging
            logging.warning(
                f"Tokenizer vocab ({tokenizer.vocab_size}) does not match "
                f"model vocab ({expected_vocab_size}). Token IDs above "
                f"{tokenizer.vocab_size} will never be generated."
            )
        return tokenizer
    except ImportError:
        print("Warning: transformers not installed. Using simple fallback tokenizer.")
        return MamformerTokenizer()
    except Exception as e:
        print(f"Warning: Failed to load tokenizer '{pretrained_name}': {e}")
        print("Using simple fallback tokenizer.")
        return MamformerTokenizer()
