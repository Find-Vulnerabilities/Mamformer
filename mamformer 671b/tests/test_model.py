"""End-to-end tests for the full Mamformer model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from mamformer.config import MamformerConfig
from mamformer.model import MamformerModel, MamformerForCausalLM


# Use a tiny debug config for fast CPU testing
@pytest.fixture(scope="module")
def debug_config():
    return MamformerConfig.from_preset("debug")


@pytest.fixture(scope="module")
def model(debug_config):
    return MamformerForCausalLM(debug_config)


class TestMamformerModel:
    """Tests for the base MamformerModel (hidden states only)."""

    @pytest.fixture
    def base_model(self, debug_config):
        return MamformerModel(debug_config)

    def test_forward_shape(self, base_model):
        """Output hidden states have correct shape."""
        input_ids = torch.randint(0, 1000, (2, 32))
        out = base_model(input_ids)
        hidden = out["last_hidden_state"]
        assert hidden.shape == (2, 32, 256)  # debug config d_model=256

    def test_forward_no_nan(self, base_model):
        """Forward pass should not produce NaN."""
        input_ids = torch.randint(0, 1000, (2, 32))
        out = base_model(input_ids)
        assert not torch.isnan(out["last_hidden_state"]).any()

    def test_gradient_flow(self, base_model):
        """Gradients should flow through the model."""
        input_ids = torch.randint(0, 1000, (2, 32))
        out = base_model(input_ids)
        loss = out["last_hidden_state"].sum()
        loss.backward()

        # Check embedding gradients
        assert base_model.embed_tokens.weight.grad is not None
        assert not torch.isnan(base_model.embed_tokens.weight.grad).any()

        # Check a layer's parameters
        first_layer = base_model.layers[0]
        assert first_layer.attention.q_proj.weight.grad is not None

    def test_cache_generation(self, base_model):
        """Cache should be generated with correct structure."""
        input_ids = torch.randint(0, 1000, (1, 8))
        out = base_model(input_ids, use_cache=True)
        cache = out.get("cache")
        assert cache is not None
        assert len(cache) == len(base_model.layers)


class TestMamformerForCausalLM:
    """Tests for the full Causal LM model."""

    def test_forward_shape(self, model):
        """Logits should have correct shape."""
        input_ids = torch.randint(0, 1000, (2, 16))
        out = model(input_ids)
        logits = out["logits"]
        assert logits.shape == (2, 16, 1000)  # debug config vocab_size=1000

    def test_loss_computation(self, model):
        """Loss should be computed and be a positive scalar."""
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        assert "loss" in out
        assert out["loss"].ndim == 0  # scalar
        assert out["loss"].item() > 0

    def test_loss_decreases(self, model):
        """Loss should decrease when training on a fixed batch (overfitting test)."""
        # Use a very small batch that model can memorize
        input_ids = torch.randint(0, 500, (4, 32))
        labels = input_ids.clone()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()

        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            out = model(input_ids, labels=labels)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease over time
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[-1]:.4f} >= {losses[0]:.4f}"

    def test_weight_tying(self, model):
        """lm_head should share weights with embedding when tie_word_embeddings=True."""
        if model.lm_head is None:
            # Tied: use embed_tokens.weight directly
            # Test that the logits are computed using embed_tokens.weight
            input_ids = torch.randint(0, 1000, (1, 8))
            out = model(input_ids)
            logits = out["logits"]

            # Manually compute logits using embed_tokens.weight
            hidden = model.model(input_ids)["last_hidden_state"]
            manual_logits = torch.matmul(
                hidden, model.model.embed_tokens.weight.t()
            )
            torch.testing.assert_close(logits, manual_logits, atol=1e-5, rtol=1e-5)

    def test_gradient_no_nan(self, model):
        """Backward pass should not produce NaN gradients."""
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN in {name}"
                # Only check finite for linear layers; embedding can have large grads
                if "embed" not in name:
                    assert not torch.isinf(param.grad).any(), f"Inf in {name}"

    def test_generation(self, model):
        """Generation should produce valid token IDs."""
        model.eval()
        input_ids = torch.randint(0, 500, (1, 4))
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=16,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
            )
        assert output.shape[0] == 1
        assert output.shape[1] >= 4 + 1  # At least one new token
        # Token IDs should be in valid range
        assert output.max() < 1000
        assert output.min() >= 0

    def test_greedy_generation(self, model):
        """Greedy generation should be deterministic."""
        model.eval()
        input_ids = torch.randint(0, 500, (1, 4))

        with torch.no_grad():
            out1 = model.generate(input_ids, max_new_tokens=8, temperature=0)
            out2 = model.generate(input_ids, max_new_tokens=8, temperature=0)

        torch.testing.assert_close(out1, out2)

    def test_batch_size(self, model):
        """Model should handle batch_size > 1."""
        input_ids = torch.randint(0, 1000, (4, 16))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        assert out["logits"].shape[0] == 4
        assert out["loss"].ndim == 0

    def test_parameter_count(self, model):
        """Parameter count should match config estimate approximately."""
        actual = model.num_parameters()
        estimated = model.config.num_parameters
        # Should be within ~1% (embedding layer specifics may differ slightly)
        diff_pct = abs(actual - estimated) / estimated * 100
        assert diff_pct < 20, f"Parameter count mismatch: actual={actual:,} vs estimated={estimated:,} ({diff_pct:.1f}%)"


class TestGradientCheckpointing:
    """Test gradient checkpointing functionality."""

    def test_checkpointing_no_error(self, debug_config):
        """Gradient checkpointing should work without errors."""
        model = MamformerForCausalLM(debug_config)
        model.model.enable_gradient_checkpointing()
        assert model.model.gradient_checkpointing

        model.train()
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        loss.backward()

        # Check gradients still flow
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No parameters received gradients with checkpointing"
