"""Tests for RMSNorm and LayerNorm implementations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from mamformer.layers.norm import RMSNorm, LayerNorm


class TestRMSNorm:
    """Test suite for RMSNorm."""

    def test_output_shape(self):
        """RMSNorm preserves input shape."""
        norm = RMSNorm(d_model=256)
        x = torch.randn(2, 10, 256)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization_effect(self):
        """After RMSNorm, the RMS should be approximately 1."""
        norm = RMSNorm(d_model=256, eps=1e-6)
        x = torch.randn(2, 10, 256)
        out = norm(x)

        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(out.float() ** 2, dim=-1))
        # Should be close to 1.0 (weight is initialized to 1)
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    def test_zero_mean_not_required(self):
        """RMSNorm does not center the distribution (no mean subtraction)."""
        norm = RMSNorm(d_model=256, eps=1e-6)
        # Use nonzero mean input
        x = torch.randn(2, 10, 256) + 5.0
        out = norm(x)
        # Mean should NOT be zero (unlike LayerNorm)
        assert not torch.allclose(
            out.mean(dim=-1), torch.zeros(2, 10), atol=1e-5
        )

    def test_gradient_flow(self):
        """Gradients should flow through RMSNorm."""
        norm = RMSNorm(d_model=256)
        x = torch.randn(2, 10, 256, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert norm.weight.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(norm.weight.grad).any()

    def test_with_bias(self):
        """RMSNorm with bias enabled."""
        norm = RMSNorm(d_model=256, bias=True)
        x = torch.randn(2, 10, 256)
        out = norm(x)
        assert out.shape == x.shape
        assert norm.bias is not None

    def test_batch_independence(self):
        """Each batch item is normalized independently."""
        norm = RMSNorm(d_model=256, eps=1e-6)
        x1 = torch.ones(1, 5, 256)
        x2 = torch.ones(1, 5, 256) * 100
        out1 = norm(x1)
        out2 = norm(x2)
        # Both should have RMS ~ 1 since normalization is per-sample
        rms1 = torch.sqrt(torch.mean(out1.float() ** 2, dim=-1))
        rms2 = torch.sqrt(torch.mean(out2.float() ** 2, dim=-1))
        assert torch.allclose(rms1, torch.ones_like(rms1), atol=1e-5)
        assert torch.allclose(rms2, torch.ones_like(rms2), atol=1e-5)


class TestLayerNorm:
    """Test suite for LayerNorm (comparison baseline)."""

    def test_output_shape(self):
        norm = LayerNorm(d_model=256)
        x = torch.randn(2, 10, 256)
        out = norm(x)
        assert out.shape == x.shape

    def test_standardization(self):
        """LayerNorm should produce mean=0 and var=1."""
        norm = LayerNorm(d_model=256, eps=1e-6)
        x = torch.randn(2, 10, 256) + 5.0
        out = norm(x)
        mean = out.float().mean(dim=-1)
        var = out.float().var(dim=-1, unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5)
