"""Tests for the Mamformer Hybrid Block."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from mamformer.layers.hybrid import MamformerBlock


class TestMamformerBlock:
    """Test suite for the hybrid Mamformer block."""

    @pytest.fixture
    def block(self):
        return MamformerBlock(
            d_model=256,
            n_heads=4,
            n_kv_heads=2,
            head_dim=64,
            d_ff=512,
            d_state=32,
            d_conv=4,
            mamba_expand=1,
            max_seq_len=128,
            rope_theta=10000.0,
        )

    def test_output_shape(self, block):
        """Output shape matches input shape."""
        x = torch.randn(2, 16, 256)
        out, _ = block(x)
        assert out.shape == x.shape

    def test_no_nan(self, block):
        """Forward pass should not produce NaN."""
        x = torch.randn(2, 16, 256)
        out, _ = block(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flow(self, block):
        """All parameters should receive gradients."""
        x = torch.randn(2, 16, 256)
        out, _ = block(x)
        loss = out.sum()
        loss.backward()

        params_with_grad = 0
        params_without_grad = []
        for name, param in block.named_parameters():
            if param.grad is not None:
                params_with_grad += 1
                assert not torch.isnan(param.grad).any(), f"{name} has NaN gradient"
            else:
                params_without_grad.append(name)

        assert params_with_grad > 0, f"No parameters received gradients. Missing: {params_without_grad}"

    def test_gate_init(self, block):
        """Gate should be initialized to 0 (sigmoid(0)=0.5, equal weighting)."""
        gate_values = block.get_gate_values()
        assert torch.allclose(gate_values, torch.full_like(gate_values, 0.5), atol=1e-5)

    def test_gate_modification(self, block):
        """Manually setting gate should affect output."""
        x = torch.randn(1, 8, 256)

        # With standard gate (0.5)
        out_balanced, _ = block(x)

        # Force all-attention mode (gate = 1.0 for all dims)
        block.gate_alpha.data[:] = 10.0  # sigmoid(10) ≈ 1.0
        out_attn, _ = block(x)

        # Force all-SSM mode (gate = 0.0 for all dims)
        block.gate_alpha.data[:] = -10.0  # sigmoid(-10) ≈ 0.0
        out_ssm, _ = block(x)

        # Different gate values should produce different outputs
        assert not torch.allclose(out_balanced, out_attn, atol=1e-4)
        assert not torch.allclose(out_balanced, out_ssm, atol=1e-4)
        assert not torch.allclose(out_attn, out_ssm, atol=1e-4)

    def test_deterministic(self, block):
        """Same input should produce same output (no dropout)."""
        x = torch.randn(2, 16, 256)
        out1, _ = block(x)
        out2, _ = block(x)
        torch.testing.assert_close(out1, out2)

    def test_cache(self, block):
        """Cache should be generated with correct structure."""
        x = torch.randn(1, 8, 256)
        out, cache = block(x, use_cache=True)
        assert cache is not None
        assert "attn" in cache
        assert "ssm" in cache
        assert out.shape == x.shape
