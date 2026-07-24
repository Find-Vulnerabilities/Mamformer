"""Tests for the Mamformer Hybrid Block — all three modes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from mamformer.layers.hybrid import MamformerBlock


# Shared block args for test fixtures
BLOCK_ARGS = dict(
    d_model=256, n_heads=4, n_kv_heads=2, head_dim=64,
    d_ff=512, d_state=32, d_conv=4, mamba_expand=1,
    max_seq_len=128, rope_theta=10000.0,
)


class TestFusionBlock:
    """Tests for fusion blocks (has_attention=True, has_ssm=True)."""

    @pytest.fixture
    def block(self):
        return MamformerBlock(**BLOCK_ARGS, has_attention=True, has_ssm=True)

    def test_output_shape(self, block):
        x = torch.randn(2, 16, 256)
        out, _ = block(x)
        assert out.shape == x.shape

    def test_no_nan(self, block):
        x = torch.randn(2, 16, 256)
        out, _ = block(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flow(self, block):
        x = torch.randn(2, 16, 256)
        out, _ = block(x)
        loss = out.sum()
        loss.backward()
        params_with_grad = sum(1 for n, p in block.named_parameters() if p.grad is not None)
        assert params_with_grad > 0

    def test_gate_init(self, block):
        gate_values = block.get_gate_values()
        assert gate_values is not None, "Fusion block must have gate"
        assert torch.allclose(gate_values, torch.full_like(gate_values, 0.5), atol=1e-5)

    def test_gate_modification(self, block):
        x = torch.randn(1, 8, 256)
        out_balanced, _ = block(x)
        block.gate_alpha.data[:] = 10.0
        out_attn, _ = block(x)
        block.gate_alpha.data[:] = -10.0
        out_ssm, _ = block(x)
        assert not torch.allclose(out_balanced, out_attn, atol=1e-4)
        assert not torch.allclose(out_balanced, out_ssm, atol=1e-4)

    def test_deterministic(self, block):
        x = torch.randn(2, 16, 256)
        out1, _ = block(x)
        out2, _ = block(x)
        torch.testing.assert_close(out1, out2)

    def test_cache_structure(self, block):
        x = torch.randn(1, 8, 256)
        out, cache = block(x, use_cache=True)
        assert cache is not None
        assert "attn" in cache, "Fusion cache must have attn"
        assert "ssm" in cache, "Fusion cache must have ssm"


class TestAttnOnlyBlock:
    """Tests for attention-only blocks (cross-layer mode)."""

    @pytest.fixture
    def block(self):
        return MamformerBlock(**BLOCK_ARGS, has_attention=True, has_ssm=False)

    def test_output_shape(self, block):
        x = torch.randn(2, 16, 256)
        out, _ = block(x)
        assert out.shape == x.shape

    def test_no_nan(self, block):
        x = torch.randn(2, 16, 256)
        out, _ = block(x)
        assert not torch.isnan(out).any()

    def test_no_gate(self, block):
        assert block.gate_alpha is None, "Attn-only block must not have gate"
        gate = block.get_gate_values()
        assert gate is None

    def test_no_ssm(self, block):
        assert block.ssm is None, "Attn-only block must not have SSM module"

    def test_cross_layer_state_injection(self, block):
        x = torch.randn(1, 8, 256)
        # Cross-layer SSM state from a previous SSM-only layer
        fake_ssm_h = torch.randn(1, 8, 32)  # (batch, seqlen, d_state)
        out, cache = block(x, ssm_h_states=fake_ssm_h)
        assert out.shape == x.shape

    def test_cache_structure(self, block):
        x = torch.randn(1, 8, 256)
        out, cache = block(x, use_cache=True)
        assert cache is not None
        assert "attn" in cache, "Attn-only cache must have attn"
        assert "ssm" not in cache, "Attn-only cache must NOT have ssm"


class TestSSMOnlyBlock:
    """Tests for SSM-only blocks (cross-layer mode)."""

    @pytest.fixture
    def block(self):
        return MamformerBlock(**BLOCK_ARGS, has_attention=False, has_ssm=True)

    def test_output_shape(self, block):
        x = torch.randn(2, 16, 256)
        out, cache = block(x)
        assert out.shape == x.shape

    def test_no_nan(self, block):
        x = torch.randn(2, 16, 256)
        out, _ = block(x)
        assert not torch.isnan(out).any()

    def test_no_attention(self, block):
        assert block.attention is None

    def test_no_gate(self, block):
        assert block.gate_alpha is None
        assert block.get_gate_values() is None

    def test_returns_h_states(self, block):
        """SSM-only block must return ssm_h_states for cross-layer injection."""
        x = torch.randn(1, 8, 256)
        out, cache = block(x)
        assert cache is not None
        assert "ssm_h_states" in cache, "SSM-only must return h_states"
        h = cache["ssm_h_states"]
        assert h.shape == (1, 8, 32)  # (batch, seqlen, d_state)

    def test_h_states_during_training(self, block):
        """SSM-only must return h_states even when use_cache=False (training)."""
        block.train()
        x = torch.randn(1, 8, 256)
        out, cache = block(x, use_cache=False)
        assert cache is not None
        assert "ssm_h_states" in cache, "SSM-only must return h_states during training"
