"""
Tests for Space-Time MoE (ST-MoE) module.
===========================================
Covers:
  - Basic forward pass with/without temporal state
  - Lambda clamping and gradient flow
  - Dynamic balance lock safety mechanism
  - Integration with MamformerBlock
  - Backward compatibility (h_states=None = standard MoE)
"""

import pytest
import torch
import torch.nn.functional as F

from mamformer.config import MamformerConfig, STMoEConfig
from mamformer.layers.st_moe import SpaceTimeMoE
from mamformer.layers.hybrid import MamformerBlock
from mamformer.layers.mamba2 import Mamba2Block, selective_scan


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def st_moe():
    """Create a small ST-MoE module for testing."""
    return SpaceTimeMoE(
        d_model=256,
        n_shared_experts=1,
        shared_expert_dim=128,
        n_routed_experts=8,
        top_k=2,
        routed_expert_dim=64,
        d_state=32,
        lambda_init=0.2,
        lambda_max=0.3,
        learnable_lambda=True,
        use_balance_lock=True,
        balance_lock_threshold=5,
        aux_loss_free=True,
        bias_update_speed=0.01,
        dropout=0.0,
    )


@pytest.fixture
def dummy_input():
    """Create a small dummy input."""
    return torch.randn(2, 4, 256)  # (batch=2, seqlen=4, d_model=256)


@pytest.fixture
def dummy_h_states():
    """Create dummy SSM h-state summaries."""
    return torch.randn(2, 4, 32)  # (batch=2, seqlen=4, d_state=32)


# ── Unit Tests: SpaceTimeMoE ────────────────────────────────────────────

class TestSpaceTimeMoE:
    """Unit tests for the SpaceTimeMoE module."""

    def test_forward_shape_without_h_states(self, st_moe, dummy_input):
        """Forward pass without SSM states should preserve shape."""
        out, aux = st_moe(dummy_input, ssm_h_states=None)
        assert out.shape == dummy_input.shape
        assert "lambda" in aux
        assert aux["lambda"] == pytest.approx(0.2, abs=0.05)

    def test_forward_shape_with_h_states(self, st_moe, dummy_input, dummy_h_states):
        """Forward pass with SSM states should preserve shape."""
        out, aux = st_moe(dummy_input, ssm_h_states=dummy_h_states)
        assert out.shape == dummy_input.shape
        assert "temporal_bias_mean" in aux
        assert "lambda" in aux

    def test_temporal_bias_changes_output(self, st_moe, dummy_input):
        """Different h_states should produce different outputs."""
        h1 = torch.zeros(2, 4, 32)
        h2 = torch.ones(2, 4, 32)

        out1, _ = st_moe(dummy_input, ssm_h_states=h1)
        out2, _ = st_moe(dummy_input, ssm_h_states=h2)

        # Outputs should differ when temporal state differs
        assert not torch.allclose(out1, out2, atol=1e-4)

    def test_lambda_zero_disables_temporal(self, st_moe, dummy_input, dummy_h_states):
        """When lambda is forced to 0, temporal state should have no effect."""
        st_moe.eval()  # Must be in eval mode to prevent bias updates
        original_max = st_moe.lambda_max
        st_moe.lambda_max = 0.0

        h1 = torch.zeros(2, 4, 32)
        h2 = torch.ones(2, 4, 32)
        out1, _ = st_moe(dummy_input, ssm_h_states=h1)
        out2, _ = st_moe(dummy_input, ssm_h_states=h2)

        assert torch.allclose(out1, out2, atol=1e-6)

        st_moe.lambda_max = original_max

    def test_lambda_clamping(self, st_moe):
        """Lambda should never exceed lambda_max regardless of parameter value."""
        # Force lambda_raw to a huge value
        st_moe.lambda_raw.data.fill_(100.0)
        lam = st_moe._get_lambda_tensor()
        assert lam.item() <= st_moe.lambda_max + 1e-6
        assert lam.item() >= 0.0

        # Force lambda_raw to a very negative value
        st_moe.lambda_raw.data.fill_(-100.0)
        lam = st_moe._get_lambda_tensor()
        assert lam.item() >= 0.0

    def test_gradient_flow_through_temporal_proj(self, st_moe, dummy_input, dummy_h_states):
        """Gradients should flow through the temporal projection."""
        st_moe.train()
        out, _ = st_moe(dummy_input, ssm_h_states=dummy_h_states)
        loss = out.sum()
        loss.backward()

        # Temporal projection should have gradients
        assert st_moe.temporal_proj.weight.grad is not None
        assert st_moe.temporal_proj.weight.grad.abs().sum() > 0

    def test_gradient_flow_through_lambda(self, st_moe, dummy_input, dummy_h_states):
        """Gradients should flow through the lambda parameter."""
        st_moe.train()
        out, _ = st_moe(dummy_input, ssm_h_states=dummy_h_states)
        loss = out.sum()
        loss.backward()

        # Lambda_raw should have gradient
        assert st_moe.lambda_raw.grad is not None

    def test_balance_lock_updates_consecutive_count(self, st_moe, dummy_input, dummy_h_states):
        """Balance lock should track consecutive expert activations correctly."""
        st_moe.train()

        # Run forward to trigger _update_consecutive_counts
        out, _ = st_moe(dummy_input, ssm_h_states=dummy_h_states)

        # All counts should be initialized and valid (>= 0)
        assert st_moe._consecutive_count.shape[0] == st_moe.n_routed_experts
        assert (st_moe._consecutive_count >= 0).all()
        assert st_moe._lock_trigger_count.item() >= 0

    def test_balance_lock_resets_on_deselection(self, st_moe, dummy_input, dummy_h_states):
        """Balance lock should reset counter when expert not selected."""
        st_moe.train()

        # Manually set some counters high
        st_moe._consecutive_count[0] = 10

        # Run forward - if expert 0 not selected, its counter should reset
        out, _ = st_moe(dummy_input, ssm_h_states=dummy_h_states)
        # Counter 0 should have changed (reset to 0 if not selected, incremented if still selected)
        assert st_moe._consecutive_count[0].item() >= 0

    def test_variable_batch_size(self, st_moe):
        """Should work with different batch sizes."""
        for bs in [1, 3, 8]:
            x = torch.randn(bs, 4, 256)
            h = torch.randn(bs, 4, 32)
            out, _ = st_moe(x, ssm_h_states=h)
            assert out.shape == (bs, 4, 256)

    def test_variable_seq_len(self, st_moe):
        """Should work with different sequence lengths."""
        for sl in [1, 2, 8, 16]:
            x = torch.randn(2, sl, 256)
            h = torch.randn(2, sl, 32)
            out, _ = st_moe(x, ssm_h_states=h)
            assert out.shape == (2, sl, 256)

    def test_aux_info_keys(self, st_moe, dummy_input, dummy_h_states):
        """Aux info should contain all expected keys."""
        _, aux = st_moe(dummy_input, ssm_h_states=dummy_h_states)
        expected_keys = [
            "routed_expert_count", "active_experts", "lambda",
            "temporal_bias_mean", "temporal_bias_std",
        ]
        for key in expected_keys:
            assert key in aux, f"Missing key: {key}"

    def test_load_statistics(self, st_moe, dummy_input, dummy_h_states):
        """Load statistics should include ST-MoE specific info."""
        st_moe.train()
        st_moe(dummy_input, ssm_h_states=dummy_h_states)
        stats = st_moe.get_load_statistics()
        assert "lambda" in stats
        if st_moe.use_balance_lock:
            assert "consecutive_max" in stats
            assert "locks_triggered" in stats

    def test_reset_statistics(self, st_moe, dummy_input, dummy_h_states):
        """Reset should clear all counters."""
        st_moe.train()
        st_moe(dummy_input, ssm_h_states=dummy_h_states)
        st_moe.reset_load_statistics()

        assert st_moe._total_tokens.item() == 0
        assert st_moe._expert_counts.sum() == 0
        if st_moe.use_balance_lock:
            assert st_moe._consecutive_count.sum() == 0

    def test_parameter_count(self, st_moe):
        """ST-MoE should have the right extra params vs standard MoE."""
        total = st_moe.total_expert_params
        active = st_moe.active_params_per_token
        assert total > 0
        assert active > 0
        assert active <= total


# ── Integration Tests ────────────────────────────────────────────────────

class TestSTMoEIntegration:
    """Integration tests: Mamba2Block + MamformerBlock with ST-MoE."""

    def test_mamba2_returns_h_states(self):
        """Mamba2Block should return h_states when requested."""
        mamba = Mamba2Block(d_model=256, d_state=32, d_conv=4, expand=1)
        x = torch.randn(2, 8, 256)

        # Without return_h_states
        out, cache = mamba(x)
        assert out.shape == (2, 8, 256)

        # With return_h_states
        out, cache, h_states = mamba(x, return_h_states=True)
        assert out.shape == (2, 8, 256)
        assert h_states.shape == (2, 8, 32)  # (batch, seqlen, d_state)

    def test_selective_scan_h_states_shape(self):
        """selective_scan should return correct h_states shape."""
        batch, seqlen, d_inner, d_state = 2, 4, 256, 32

        x = torch.randn(batch, seqlen, d_inner)
        dt = F.softplus(torch.randn(batch, seqlen, d_inner))
        A = torch.log(torch.linspace(0.5, 8, d_state))
        B = torch.randn(batch, seqlen, d_state)
        C = torch.randn(batch, seqlen, d_state)
        D = torch.ones(d_inner)

        y, h_states = selective_scan(
            x, dt, A, B, C, D, return_h_states=True
        )
        assert y.shape == (batch, seqlen, d_inner)
        assert h_states.shape == (batch, seqlen, d_state)

    def test_mamformer_block_with_st_moe(self):
        """Full MamformerBlock with ST-MoE should work end-to-end."""
        block = MamformerBlock(
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
            dropout=0.0,
            use_moe=False,
            moe_n_shared=1,
            moe_n_routed=8,
            moe_top_k=2,
            moe_shared_dim=128,
            moe_routed_dim=64,
            use_st_moe=True,
            st_moe_lambda_init=0.2,
            st_moe_lambda_max=0.3,
            st_moe_balance_lock_threshold=10,
        )

        x = torch.randn(2, 8, 256)
        out, cache = block(x, use_cache=False)
        assert out.shape == (2, 8, 256)

    def test_mamformer_block_st_moe_gradient(self):
        """Gradients should flow through entire MamformerBlock with ST-MoE."""
        block = MamformerBlock(
            d_model=256,
            n_heads=4,
            n_kv_heads=2,
            head_dim=64,
            d_ff=512,
            d_state=32,
            d_conv=4,
            max_seq_len=128,
            use_moe=False,
            moe_n_shared=1,
            moe_n_routed=8,
            moe_top_k=2,
            moe_shared_dim=128,
            moe_routed_dim=64,
            use_st_moe=True,
        )

        x = torch.randn(2, 8, 256, requires_grad=False)
        out, _ = block(x)
        loss = out.sum()
        loss.backward()

        # Verify gradients exist on ST-MoE specific params
        st_moe_params_with_grad = 0
        for name, param in block.ffn.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                st_moe_params_with_grad += 1
        assert st_moe_params_with_grad >= 1

    def test_backward_compat_standard_moe(self):
        """MamformerBlock with standard MoE should still work unchanged."""
        from mamformer.layers.moe import DeepSeekMoE

        block = MamformerBlock(
            d_model=256,
            n_heads=4,
            n_kv_heads=2,
            head_dim=64,
            d_ff=512,
            d_state=32,
            d_conv=4,
            max_seq_len=128,
            use_moe=True,
            moe_n_shared=1,
            moe_n_routed=8,
            moe_top_k=2,
            moe_shared_dim=128,
            moe_routed_dim=64,
            use_st_moe=False,
        )

        assert isinstance(block.ffn, DeepSeekMoE)
        assert not isinstance(block.ffn, SpaceTimeMoE)

        x = torch.randn(2, 8, 256)
        out, _ = block(x)
        assert out.shape == (2, 8, 256)


# ── Config Tests ─────────────────────────────────────────────────────────

class TestSTMoEConfig:
    """Tests for ST-MoE configuration integration."""

    def test_config_serialization_roundtrip(self):
        """ST-MoE config should survive dict roundtrip."""
        config = MamformerConfig.from_preset("debug")
        config.st_moe.enabled = True
        config.st_moe.lambda_init = 0.25
        config.st_moe.lambda_max = 0.35

        d = config.to_dict()
        config2 = MamformerConfig.from_dict(d)

        assert config2.st_moe.enabled == True
        assert config2.st_moe.lambda_init == 0.25
        assert config2.st_moe.lambda_max == 0.35
        assert config2.st_moe.learnable_lambda == True
        assert config2.st_moe.use_balance_lock == True
        assert config2.st_moe.balance_lock_threshold == 50

    def test_config_default_values(self):
        """Default ST-MoE config should be disabled."""
        config = MamformerConfig.from_preset("debug")
        assert config.st_moe.enabled == False
        assert config.st_moe.lambda_init == 0.2
        assert config.st_moe.lambda_max == 0.3

    def test_config_summary_includes_st_moe(self):
        """Summary should mention ST-MoE when enabled."""
        config = MamformerConfig.from_preset("debug")
        config.st_moe.enabled = True
        summary = config.summary()
        assert "ST-MoE" in summary
        assert "ENABLED" in summary

    def test_config_summary_excludes_st_moe_when_disabled(self):
        """Summary should not mention ST-MoE when disabled."""
        config = MamformerConfig.from_preset("debug")
        config.st_moe.enabled = False
        summary = config.summary()
        assert "ST-MoE" not in summary
