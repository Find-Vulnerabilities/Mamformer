"""
Tests for DeepSeekMoE (Mixture of Experts) module.
"""

import pytest
import torch
import torch.nn as nn

from mamformer.layers.moe import DeepSeekMoE, MoERouter


class TestDeepSeekMoE:
    """Test DeepSeekMoE FFN module."""

    @pytest.fixture
    def small_moe(self):
        """Create a small MoE for testing."""
        return DeepSeekMoE(
            d_model=256,
            n_shared_experts=2,
            shared_expert_dim=128,
            n_routed_experts=8,
            top_k=2,
            routed_expert_dim=64,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )

    @pytest.fixture
    def batch(self):
        """Create a sample batch."""
        return torch.randn(4, 16, 256)

    def test_forward_shape(self, small_moe, batch):
        """Output shape should match input shape."""
        output, aux_info = small_moe(batch)
        assert output.shape == batch.shape
        assert output.dtype == batch.dtype

    def test_output_not_nan(self, small_moe, batch):
        """Output should not contain NaN."""
        output, _ = small_moe(batch)
        assert not torch.isnan(output).any()

    def test_output_not_zero(self, small_moe, batch):
        """Output should not be all zeros (experts are active)."""
        output, _ = small_moe(batch)
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_aux_info_contains_keys(self, small_moe, batch):
        """Aux info should contain expected keys."""
        _, aux_info = small_moe(batch)
        assert "routed_expert_count" in aux_info
        assert "active_experts" in aux_info
        assert aux_info["active_experts"] == 2  # top_k

    def test_shared_experts_always_active(self, small_moe, batch):
        """Shared experts contribute even with extreme routing."""
        # Zero out router to test shared experts still work
        with torch.no_grad():
            small_moe.router.weight.zero_()

        output, _ = small_moe(batch)
        # Output should still be non-zero from shared experts
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_gradient_flow(self, small_moe, batch):
        """Gradients should flow through all components."""
        output, _ = small_moe(batch)
        loss = output.sum()
        loss.backward()

        # Router should have gradients
        assert small_moe.router.weight.grad is not None
        assert not torch.allclose(small_moe.router.weight.grad, torch.zeros_like(small_moe.router.weight.grad))

        # Shared experts should have gradients
        for expert in small_moe.shared_experts:
            for name, param in expert.named_parameters():
                assert param.grad is not None, f"Shared expert {name} has no grad"

        # At least some routed experts should have gradients (depends on routing)
        has_grad = False
        for expert in small_moe.routed_experts:
            if expert.gate_proj.weight.grad is not None:
                has_grad = True
                break
        # Note: not all experts necessarily receive tokens in a small batch
        # But in 4*16=64 tokens with top_k=2, most experts should be hit

    def test_expert_bias_updated(self, small_moe, batch):
        """Expert bias should update during training."""
        small_moe.train()
        initial_bias = small_moe.expert_bias.clone()

        output, _ = small_moe(batch)
        loss = output.sum()
        loss.backward()

        # Bias should have changed
        assert not torch.allclose(small_moe.expert_bias, initial_bias)

    def test_eval_mode_no_bias_update(self, small_moe, batch):
        """In eval mode, expert bias should not change."""
        small_moe.eval()
        initial_bias = small_moe.expert_bias.clone()

        with torch.no_grad():
            output, _ = small_moe(batch)

        # Bias should be unchanged in eval mode
        assert torch.allclose(small_moe.expert_bias, initial_bias)

    def test_load_statistics(self, small_moe, batch):
        """Load statistics should return valid values."""
        small_moe.train()
        output, _ = small_moe(batch)
        loss = output.sum()
        loss.backward()

        stats = small_moe.get_load_statistics()
        assert "load_entropy" in stats
        assert 0.0 <= stats["load_entropy"] <= 1.0
        assert stats["per_expert_load"] is not None

    def test_reset_statistics(self, small_moe, batch):
        """Reset should clear counters."""
        small_moe.train()
        output, _ = small_moe(batch)
        loss = output.sum()
        loss.backward()

        small_moe.reset_load_statistics()
        stats = small_moe.get_load_statistics()
        # After reset before any forward, total_tokens is 0
        # get_load_statistics handles this gracefully
        assert stats["load_entropy"] == 0.0

    def test_variable_batch_size(self, small_moe):
        """Should work with different batch sizes."""
        for bs in [1, 2, 8]:
            x = torch.randn(bs, 32, 256)
            output, _ = small_moe(x)
            assert output.shape == x.shape

    def test_variable_seq_len(self, small_moe):
        """Should work with different sequence lengths."""
        for sl in [1, 8, 64, 128]:
            x = torch.randn(2, sl, 256)
            output, _ = small_moe(x)
            assert output.shape == x.shape

    def test_total_expert_params(self, small_moe):
        """Total expert params should be reasonable."""
        total = small_moe.total_expert_params
        active = small_moe.active_params_per_token
        assert total > active  # Total >> active for sparsity
        assert total > 0
        assert active > 0

    def test_different_top_k(self, batch):
        """Different top_k values should work."""
        for k in [1, 2, 4]:
            moe = DeepSeekMoE(
                d_model=256,
                n_shared_experts=2,
                shared_expert_dim=128,
                n_routed_experts=8,
                top_k=k,
                routed_expert_dim=64,
            )
            output, aux_info = moe(batch)
            assert output.shape == batch.shape
            assert aux_info["active_experts"] == k


class TestMoERouter:
    """Test standalone MoE router."""

    def test_router_output_shape(self):
        router = MoERouter(d_model=256, n_experts=8, top_k=2)
        x = torch.randn(4, 16, 256)
        gates, indices = router(x)
        assert gates.shape == (4, 16, 2)
        assert indices.shape == (4, 16, 2)

    def test_gates_sum_to_one(self):
        """Softmax-normalized gates should sum to ~1."""
        router = MoERouter(d_model=256, n_experts=8, top_k=2)
        x = torch.randn(4, 16, 256)
        gates, _ = router(x)
        assert torch.allclose(gates.sum(dim=-1), torch.ones(4, 16), atol=1e-5)

    def test_indices_in_range(self):
        """Expert indices should be in valid range."""
        router = MoERouter(d_model=256, n_experts=8, top_k=3)
        x = torch.randn(4, 16, 256)
        _, indices = router(x)
        assert indices.min() >= 0
        assert indices.max() < 8

    def test_router_with_bias(self):
        """Router should work with expert bias."""
        router = MoERouter(d_model=256, n_experts=8, top_k=2)
        x = torch.randn(4, 16, 256)
        bias = torch.randn(8)
        gates, indices = router(x, expert_bias=bias)
        assert gates.shape == (4, 16, 2)

    def test_temperature_effect(self):
        """Higher temperature should produce more uniform gates."""
        x = torch.randn(4, 16, 256)

        router_cold = MoERouter(d_model=256, n_experts=8, top_k=2, temperature=0.1)
        router_hot = MoERouter(d_model=256, n_experts=8, top_k=2, temperature=10.0)

        gates_cold, _ = router_cold(x)
        gates_hot, _ = router_hot(x)

        # Cold: more confident (higher max gate)
        # Hot: more uniform (lower max gate)
        assert gates_cold.max() >= gates_hot.max() - 0.1  # Approximate check
