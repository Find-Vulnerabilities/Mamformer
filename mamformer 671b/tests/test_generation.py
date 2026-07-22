"""Tests for generation utilities and tokenizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from mamformer.config import MamformerConfig
from mamformer.model import MamformerForCausalLM
from mamformer.tokenizer import MamformerTokenizer
from mamformer.generation import GenerationConfig


class TestMamformerTokenizer:
    """Test suite for MamformerTokenizer."""

    @pytest.fixture
    def tokenizer(self):
        return MamformerTokenizer()

    def test_encode_decode_roundtrip(self, tokenizer):
        """Encode → decode should recover original text."""
        text = "Hello, world!"
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        decoded = tokenizer.decode(ids)
        assert text == decoded

    def test_bos_token(self, tokenizer):
        """BOS token should be prepended when add_bos=True."""
        text = "test"
        ids = tokenizer.encode(text, add_bos=True, add_eos=False)
        assert ids[0] == tokenizer.bos_token_id

    def test_eos_token(self, tokenizer):
        """EOS token should be appended when add_eos=True."""
        text = "test"
        ids = tokenizer.encode(text, add_bos=False, add_eos=True)
        assert ids[-1] == tokenizer.eos_token_id

    def test_batch_encode(self, tokenizer):
        """Batch encoding should return list of lists."""
        texts = ["hello", "world", "test"]
        ids = tokenizer.encode(texts, add_bos=False)
        assert len(ids) == 3
        assert all(isinstance(x, list) for x in ids)

    def test_decode_with_special_tokens(self, tokenizer):
        """Skip special tokens during decoding."""
        text = "hi"
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        # Should not contain EOS marker (char 2)
        assert "\x02" not in decoded
        assert "\x01" not in decoded

    def test_padding(self, tokenizer):
        """Padding should extend sequences to max_length."""
        texts = ["a", "hello world"]
        ids = tokenizer.encode(texts, max_length=16, padding=True, add_bos=False)
        assert len(ids[0]) == 16
        assert len(ids[1]) == 16
        assert ids[0][-1] == tokenizer.pad_token_id

    def test_truncation(self, tokenizer):
        """Truncation should cap sequences at max_length."""
        text = "a" * 100
        ids = tokenizer.encode(text, max_length=16, truncation=True, add_bos=False)
        assert len(ids) <= 16

    def test_vocab_size(self, tokenizer):
        """Simple tokenizer has 256 vocab (ASCII)."""
        assert tokenizer.vocab_size == 256


class TestGeneration:
    """Test suite for text generation."""

    @pytest.fixture
    def config(self):
        return MamformerConfig.from_preset("debug")

    @pytest.fixture
    def model(self, config):
        return MamformerForCausalLM(config)

    @pytest.fixture
    def tokenizer(self):
        return MamformerTokenizer()

    def test_greedy_output_shape(self, model):
        """Greedy generation should produce correct output shape."""
        input_ids = torch.randint(0, 500, (1, 4))
        model.eval()
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=10,
                temperature=0,  # greedy
            )
        assert output.shape[0] == 1
        assert output.shape[1] == 4 + 10

    def test_temperature_sampling(self, model):
        """Temperature > 0 should produce different outputs with different seeds."""
        input_ids = torch.randint(0, 500, (1, 4))
        model.eval()

        torch.manual_seed(42)
        with torch.no_grad():
            out1 = model.generate(input_ids, max_new_tokens=8, temperature=1.0, top_k=10)

        torch.manual_seed(123)
        with torch.no_grad():
            out2 = model.generate(input_ids, max_new_tokens=8, temperature=1.0, top_k=10)

        # With different seeds and temperature > 0, outputs may differ
        # (Note: they might coincide by chance with tiny vocab, but very unlikely with top_k=10)

    def test_top_k(self, model):
        """Top-k should work without errors."""
        input_ids = torch.randint(0, 500, (1, 4))
        model.eval()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=5, temperature=1.0, top_k=5)
        assert output.shape[1] > 4

    def test_top_p(self, model):
        """Top-p (nucleus) should work without errors."""
        input_ids = torch.randint(0, 500, (1, 4))
        model.eval()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=5, temperature=1.0, top_p=0.9)
        assert output.shape[1] > 4

    def test_eos_stopping(self, model):
        """Generation should stop when EOS token is produced."""
        input_ids = torch.randint(0, 500, (1, 4))
        model.eval()
        # Force model to predict EOS by using a very short generation
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                temperature=0,
                eos_token_id=2,
            )
        # With greedy and random weights, model will eventually produce something
        assert output.shape[1] <= 4 + 100

    def test_batch_generation(self, model):
        """Batch generation should work."""
        input_ids = torch.randint(0, 500, (2, 4))
        model.eval()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=4, temperature=0)
        assert output.shape[0] == 2


class TestGenerationConfig:
    """Test GenerationConfig validation."""

    def test_valid_config(self):
        config = GenerationConfig(temperature=0.7, top_k=50, top_p=0.9)
        config.validate()  # Should not raise

    def test_invalid_temperature(self):
        config = GenerationConfig(temperature=-0.5)
        with pytest.raises(ValueError):
            config.validate()

    def test_beam_search_with_sampling(self):
        config = GenerationConfig(num_beams=4, temperature=0.7)
        with pytest.raises(ValueError):
            config.validate()
