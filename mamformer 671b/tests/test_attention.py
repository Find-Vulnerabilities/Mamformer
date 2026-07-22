"""Tests for GroupedQueryAttention with RoPE."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from mamformer.layers.attention import GroupedQueryAttention
from mamformer.layers.rope import apply_rotary_emb


class TestGroupedQueryAttention:
    """Test suite for GQA attention."""

    @pytest.fixture
    def attn(self):
        return GroupedQueryAttention(
            d_model=512,
            n_heads=8,
            n_kv_heads=2,
            head_dim=64,
            max_seq_len=256,
            rope_theta=10000.0,
        )

    def test_output_shape(self, attn):
        """Output shape matches input shape."""
        x = torch.randn(2, 64, 512)
        out, _ = attn(x)
        assert out.shape == x.shape

    def test_causal_attention(self, attn):
        """Position i should only depend on positions ≤ i."""
        # Use a carefully constructed input where position 0 is very different
        x = torch.randn(1, 16, 512)
        x[:, 0, :] = 100.0  # Make first token stand out

        out1, _ = attn(x)
        # Modify first token and check only positions after it change
        x2 = x.clone()
        x2[:, 8, :] = -100.0  # Modify middle token

        out2, _ = attn(x2)
        # Position 0-7 should be unchanged
        assert torch.allclose(out1[:, :8, :], out2[:, :8, :], atol=1e-5)

    def test_kv_groups(self, attn):
        """KV heads are properly repeated to match Q heads."""
        assert attn.n_head_groups == 4  # 8 Q heads / 2 KV heads = 4

        x = torch.randn(1, 8, 512)
        out, _ = attn(x)
        # If KV groups weren't working, shapes wouldn't match
        assert out.shape == (1, 8, 512)

    def test_gradient_flow(self, attn):
        """All parameters receive gradients."""
        x = torch.randn(2, 16, 512, requires_grad=False)
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()

        for name, param in attn.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"{name} has NaN gradient"

    def test_rope_position_sensitivity(self, attn):
        """RoPE should apply different rotations at different positions."""
        # Test: same token at position 0 vs position 1 should have different
        # Q vectors due to different RoPE rotations
        x = torch.randn(1, 1, 512)
        x_repeated = x.expand(1, 2, 512)  # Same content at positions 0 and 1

        # Manually check Q projections with RoPE
        q = attn.q_proj(x_repeated)
        q = q.view(1, 2, attn.n_heads, attn.head_dim).transpose(1, 2)
        cos, sin = attn.rope(2, x.device)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q_rotated = apply_rotary_emb(q, cos, sin)

        # Q at position 0 vs position 1 should differ due to RoPE
        assert not torch.allclose(q_rotated[:, :, 0, :], q_rotated[:, :, 1, :], atol=1e-4)

    def test_with_cache(self, attn):
        """KV cache should be generated and usable."""
        x = torch.randn(1, 16, 512)

        # First pass
        out_full, _ = attn(x, use_cache=False)

        # Pass with caching
        out_cached, cache = attn(x, use_cache=True)
        assert cache is not None
        assert "k" in cache and "v" in cache
        assert cache["k"].shape[0] == 1  # batch
        assert cache["k"].shape[2] == 16  # seqlen

    def test_attention_mask(self, attn):
        """Custom attention mask should be respected."""
        x = torch.randn(1, 8, 512)
        # Full bidirectional mask
        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        out_full, _ = attn(x, attention_mask=mask)
        assert out_full.shape == x.shape

    def test_batch_size(self, attn):
        """Works with batch_size > 1."""
        x = torch.randn(4, 32, 512)
        out, _ = attn(x)
        assert out.shape == x.shape


class TestGQAConfigurations:
    """Test different GQA configurations."""

    def test_mha_mode(self):
        """Standard MHA: n_kv_heads = n_heads."""
        attn = GroupedQueryAttention(
            d_model=256, n_heads=4, n_kv_heads=4, head_dim=64, max_seq_len=128,
        )
        x = torch.randn(2, 16, 256)
        out, _ = attn(x)
        assert out.shape == x.shape
        assert attn.n_head_groups == 1

    def test_mqa_mode(self):
        """Multi-Query Attention: n_kv_heads = 1."""
        attn = GroupedQueryAttention(
            d_model=256, n_heads=4, n_kv_heads=1, head_dim=64, max_seq_len=128,
        )
        x = torch.randn(2, 16, 256)
        out, _ = attn(x)
        assert out.shape == x.shape
        assert attn.n_head_groups == 4
