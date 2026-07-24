"""
Tests for CommunicativeMoE (Cross-Expert Communication).
=========================================================
Covers:
  - Wrapping DeepSeekMoE and SpaceTimeMoE
  - Forward pass shape preservation
  - Communication layer gradient flow
  - Integration with MamformerBlock
  - Config serialization roundtrip
  - Communication strength learning
"""

import pytest
import torch

from mamformer.config import MamformerConfig, CommunicativeMoEConfig
from mamformer.layers.moe import DeepSeekMoE
from mamformer.layers.st_moe import SpaceTimeMoE
from mamformer.layers.communicative_moe import (
    CommunicativeMoE,
    ExpertCommunicationLayer,
)
from mamformer.layers.hybrid import MamformerBlock


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def small_base_moe():
    """Create a small DeepSeekMoE for wrapping."""
    return DeepSeekMoE(
        d_model=256,
        n_shared_experts=1,
        shared_expert_dim=128,
        n_routed_experts=8,
        top_k=2,
        routed_expert_dim=64,
        aux_loss_free=True,
        bias_update_speed=0.01,
    )


@pytest.fixture
def small_comm_moe(small_base_moe):
    """Create a CommunicativeMoE wrapping DeepSeekMoE."""
    return CommunicativeMoE(
        base_moe=small_base_moe,
        d_model=256,
        n_comm_heads=4,
        comm_depth=1,
    )


@pytest.fixture
def small_st_moe_base():
    """Create a small SpaceTimeMoE for wrapping."""
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
        balance_lock_threshold=10,
        aux_loss_free=True,
        bias_update_speed=0.01,
    )


@pytest.fixture
def small_comm_st_moe(small_st_moe_base):
    """Create a CommunicativeMoE wrapping SpaceTimeMoE."""
    return CommunicativeMoE(
        base_moe=small_st_moe_base,
        d_model=256,
        n_comm_heads=4,
        comm_depth=1,
    )


@pytest.fixture
def dummy_input():
    """Small dummy input tensor."""
    return torch.randn(2, 4, 256)


@pytest.fixture
def dummy_h_states():
    """Dummy SSM h-state summaries for ST-MoE."""
    return torch.randn(2, 4, 32)


# ── Unit Tests: ExpertCommunicationLayer ────────────────────────────────

class TestExpertCommunicationLayer:
    """Tests for the ExpertCommunicationLayer."""

    def test_forward_shape(self):
        """Output shape should match input shape."""
        layer = ExpertCommunicationLayer(d_model=256, n_heads=4, depth=1)
        x = torch.randn(16, 3, 256)  # (N=16 tokens, k=3 experts, d_model=256)
        out = layer(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_communication_changes_output(self):
        """Communication should modify the expert outputs."""
        layer = ExpertCommunicationLayer(d_model=256, n_heads=4, depth=1)
        x = torch.randn(16, 3, 256)
        out = layer(x)
        # Output should differ from input (communication happened)
        assert not torch.allclose(out, x, atol=1e-4)

    def test_residual_ensures_stability(self):
        """With depth > 1 and residuals, output should be stable."""
        layer = ExpertCommunicationLayer(d_model=256, n_heads=4, depth=3)
        x = torch.randn(16, 3, 256)
        out = layer(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flow(self):
        """Gradients should flow through communication layer."""
        layer = ExpertCommunicationLayer(d_model=256, n_heads=4, depth=1)
        x = torch.randn(16, 3, 256, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_variable_num_experts(self):
        """Should work with different numbers of selected experts (k)."""
        layer = ExpertCommunicationLayer(d_model=256, n_heads=4, depth=1)
        for k in [1, 2, 4, 8]:
            x = torch.randn(4, k, 256)
            out = layer(x)
            assert out.shape == (4, k, 256)

    def test_batch_independence(self):
        """Communication should be per-token (N dimension is batch)."""
        layer = ExpertCommunicationLayer(d_model=256, n_heads=4, depth=1)
        x = torch.randn(4, 3, 256)
        # Permute batch order
        perm = torch.randperm(4)
        x_perm = x[perm]
        out = layer(x)
        out_perm = layer(x_perm)
        # Output at position i should match regardless of batch order
        for i in range(4):
            assert torch.allclose(out[i], out_perm[(perm == i).nonzero()[0, 0]], atol=1e-5)


# ── Unit Tests: CommunicativeMoE ───────────────────────────────────────

class TestCommunicativeMoE:
    """Tests for CommunicativeMoE wrapping DeepSeekMoE."""

    def test_forward_shape(self, small_comm_moe, dummy_input):
        """Output shape should match input shape."""
        out, aux = small_comm_moe(dummy_input)
        assert out.shape == dummy_input.shape
        assert out.dtype == dummy_input.dtype

    def test_output_not_nan(self, small_comm_moe, dummy_input):
        """Output should not contain NaN."""
        out, _ = small_comm_moe(dummy_input)
        assert not torch.isnan(out).any()

    def test_output_not_zero(self, small_comm_moe, dummy_input):
        """Output should not be all zeros."""
        out, _ = small_comm_moe(dummy_input)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_aux_info_keys(self, small_comm_moe, dummy_input):
        """Aux info should contain expected keys."""
        _, aux = small_comm_moe(dummy_input)
        assert "routed_expert_count" in aux
        assert "active_experts" in aux
        assert "comm_strength" in aux
        assert "gate_adjustment" in aux

    def test_gradient_flow_through_comm_layer(self, small_comm_moe, dummy_input):
        """Gradients should flow through communication layer."""
        small_comm_moe.train()
        out, _ = small_comm_moe(dummy_input)
        loss = out.sum()
        loss.backward()

        # Communication layer should have gradients
        has_grad = False
        for name, param in small_comm_moe.comm_layer.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients in communication layer"

    def test_gradient_flow_through_comm_strength(self, small_comm_moe, dummy_input):
        """comm_strength should receive gradients."""
        small_comm_moe.train()
        out, _ = small_comm_moe(dummy_input)
        loss = out.sum()
        loss.backward()
        assert small_comm_moe.comm_strength.grad is not None

    def test_gradient_flow_through_gate_adjustment(self, small_comm_moe, dummy_input):
        """gate_adjustment should receive gradients."""
        small_comm_moe.train()
        out, _ = small_comm_moe(dummy_input)
        loss = out.sum()
        loss.backward()
        assert small_comm_moe.gate_adjustment.grad is not None

    def test_comm_strength_effect(self, small_comm_moe, dummy_input):
        """Different comm_strength values should affect output."""
        # Force low communication
        small_comm_moe.comm_strength.data.fill_(-10.0)  # sigmoid(-10) ≈ 0
        out_low, _ = small_comm_moe(dummy_input)

        # Force high communication
        small_comm_moe.comm_strength.data.fill_(10.0)  # sigmoid(10) ≈ 1
        out_high, _ = small_comm_moe(dummy_input)

        # Outputs should differ
        assert not torch.allclose(out_low, out_high, atol=1e-3)

    def test_variable_batch_size(self, small_comm_moe):
        """Should work with different batch sizes."""
        for bs in [1, 3, 8]:
            x = torch.randn(bs, 4, 256)
            out, _ = small_comm_moe(x)
            assert out.shape == (bs, 4, 256)

    def test_variable_seq_len(self, small_comm_moe):
        """Should work with different sequence lengths."""
        for sl in [1, 2, 8, 16]:
            x = torch.randn(2, sl, 256)
            out, _ = small_comm_moe(x)
            assert out.shape == (2, sl, 256)

    def test_expert_bias_updated(self, small_comm_moe, dummy_input):
        """Expert bias should update during training."""
        small_comm_moe.train()
        initial_bias = small_comm_moe.base_moe.expert_bias.clone()
        out, _ = small_comm_moe(dummy_input)
        loss = out.sum()
        loss.backward()
        assert not torch.allclose(small_comm_moe.base_moe.expert_bias, initial_bias)

    def test_eval_mode_no_bias_update(self, small_comm_moe, dummy_input):
        """In eval mode, expert bias should not change."""
        small_comm_moe.eval()
        initial_bias = small_comm_moe.base_moe.expert_bias.clone()
        with torch.no_grad():
            small_comm_moe(dummy_input)
        assert torch.allclose(small_comm_moe.base_moe.expert_bias, initial_bias)

    def test_load_statistics(self, small_comm_moe, dummy_input):
        """Load statistics should include comm info."""
        small_comm_moe.train()
        out, _ = small_comm_moe(dummy_input)
        loss = out.sum()
        loss.backward()
        stats = small_comm_moe.get_load_statistics()
        assert "comm_strength" in stats

    def test_parameter_count(self, small_comm_moe):
        """Parameter count should include communication layer."""
        total = small_comm_moe.total_expert_params
        active = small_comm_moe.active_params_per_token
        base_total = small_comm_moe.base_moe.total_expert_params
        assert total > base_total  # Communication adds params
        assert active > 0

    def test_n_routed_experts_delegated(self, small_comm_moe):
        """n_routed_experts should match base MoE."""
        assert small_comm_moe.n_routed_experts == small_comm_moe.base_moe.n_routed_experts

    def test_top_k_delegated(self, small_comm_moe):
        """top_k should match base MoE."""
        assert small_comm_moe.top_k == small_comm_moe.base_moe.top_k


# ── ST-MoE Integration Tests ────────────────────────────────────────────

class TestCommunicativeSTMoe:
    """Tests for CommunicativeMoE wrapping SpaceTimeMoE."""

    def test_forward_with_h_states(self, small_comm_st_moe, dummy_input, dummy_h_states):
        """Forward pass with SSM states should work."""
        out, aux = small_comm_st_moe(dummy_input, ssm_h_states=dummy_h_states)
        assert out.shape == dummy_input.shape
        assert "lambda" in aux

    def test_forward_without_h_states(self, small_comm_st_moe, dummy_input):
        """Forward pass without SSM states should work (spatial-only fallback)."""
        out, aux = small_comm_st_moe(dummy_input, ssm_h_states=None)
        assert out.shape == dummy_input.shape

    def test_temporal_state_changes_output(self, small_comm_st_moe, dummy_input):
        """Different h_states should produce different outputs."""
        h1 = torch.zeros(2, 4, 32)
        h2 = torch.ones(2, 4, 32)
        out1, _ = small_comm_st_moe(dummy_input, ssm_h_states=h1)
        out2, _ = small_comm_st_moe(dummy_input, ssm_h_states=h2)
        assert not torch.allclose(out1, out2, atol=1e-4)

    def test_gradient_through_temporal_proj(self, small_comm_st_moe, dummy_input, dummy_h_states):
        """Gradients should flow through temporal projection."""
        small_comm_st_moe.train()
        out, _ = small_comm_st_moe(dummy_input, ssm_h_states=dummy_h_states)
        loss = out.sum()
        loss.backward()
        assert small_comm_st_moe.base_moe.temporal_proj.weight.grad is not None


# ── Integration Tests: MamformerBlock ──────────────────────────────────

class TestCommunicativeMoEIntegration:
    """Integration tests with MamformerBlock."""

    def test_mamformer_block_with_comm_moe(self):
        """Full MamformerBlock with CommunicativeMoE should work."""
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
            use_moe=True,
            moe_n_shared=1,
            moe_n_routed=8,
            moe_top_k=2,
            moe_shared_dim=128,
            moe_routed_dim=64,
            use_communicative_moe=True,
            comm_moe_n_heads=4,
            comm_moe_depth=1,
        )

        x = torch.randn(2, 8, 256)
        out, cache = block(x, use_cache=False)
        assert out.shape == (2, 8, 256)

    def test_mamformer_block_comm_moe_gradient(self):
        """Gradients should flow through entire block with CommunicativeMoE."""
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
            use_communicative_moe=True,
        )

        x = torch.randn(2, 8, 256)
        out, _ = block(x)
        loss = out.sum()
        loss.backward()

        # Verify gradients exist on communication params
        comm_grad_count = 0
        for name, param in block.ffn.comm_layer.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                comm_grad_count += 1
        assert comm_grad_count > 0, "No gradients in communication layer"

    def test_mamformer_block_comm_st_moe(self):
        """MamformerBlock with CommunicativeMoE wrapping ST-MoE."""
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
            use_communicative_moe=True,
            comm_moe_n_heads=4,
            comm_moe_depth=1,
        )

        x = torch.randn(2, 8, 256)
        out, cache = block(x, use_cache=False)
        assert out.shape == (2, 8, 256)

    def test_mamformer_block_comm_moe_extra_repr(self):
        """Extra repr should include CommunicativeMoE info."""
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
            use_communicative_moe=True,
        )
        assert isinstance(block.ffn, CommunicativeMoE)

    def test_block_without_comm_still_works(self):
        """MamformerBlock without CommunicativeMoE should still work."""
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
            moe_n_routed=4,
            moe_top_k=2,
            moe_shared_dim=128,
            moe_routed_dim=64,
            use_communicative_moe=False,
        )
        x = torch.randn(2, 8, 256)
        out, _ = block(x)
        assert out.shape == (2, 8, 256)
        assert isinstance(block.ffn, DeepSeekMoE)

    def test_comm_moe_requires_moe_enabled(self):
        """CommunicativeMoE should work only when MoE is enabled."""
        # This should work: MoE enabled + CommunicativeMoE
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
            moe_n_routed=4,
            moe_top_k=2,
            moe_shared_dim=128,
            moe_routed_dim=64,
            use_communicative_moe=True,
        )
        assert isinstance(block.ffn, CommunicativeMoE)

        # Without MoE, CommunicativeMoE flag should be ignored, fall back to SwiGLU
        block2 = MamformerBlock(
            d_model=256,
            n_heads=4,
            n_kv_heads=2,
            head_dim=64,
            d_ff=512,
            d_state=32,
            d_conv=4,
            max_seq_len=128,
            use_moe=False,
            use_communicative_moe=True,  # Will be ignored
        )
        # Should still work (falls back to SwiGLU)
        x = torch.randn(2, 8, 256)
        out, _ = block2(x)
        assert out.shape == (2, 8, 256)


# ── Config Tests ─────────────────────────────────────────────────────────

class TestCommunicativeMoEConfig:
    """Tests for CommunicativeMoEConfig serialization."""

    def test_default_values(self):
        """Default config should be disabled."""
        cfg = CommunicativeMoEConfig()
        assert cfg.enabled == False
        assert cfg.n_comm_heads == 4
        assert cfg.comm_depth == 1
        assert cfg.comm_dropout == 0.0

    def test_config_roundtrip(self):
        """CommunicativeMoE config should survive dict roundtrip."""
        config = MamformerConfig.from_preset("debug")
        config.moe.enabled = True
        config.communicative_moe.enabled = True
        config.communicative_moe.n_comm_heads = 4
        config.communicative_moe.comm_depth = 2

        d = config.to_dict()
        config2 = MamformerConfig.from_dict(d)

        assert config2.communicative_moe.enabled == True
        assert config2.communicative_moe.n_comm_heads == 4
        assert config2.communicative_moe.comm_depth == 2

    def test_validation_requires_moe(self):
        """Should raise if CommunicativeMoE enabled without MoE."""
        config = MamformerConfig.from_preset("debug")
        # Neither MoE nor ST-MoE enabled
        config.moe.enabled = False
        config.st_moe.enabled = False
        config.communicative_moe.enabled = True

        with pytest.raises(ValueError, match="CommunicativeMoE requires MoE"):
            MamformerConfig(**{k: v for k, v in config.__dict__.items() if not k.startswith('_')})  # trigger validation

    def test_summary_includes_comm_moe(self):
        """Summary should mention CommunicativeMoE when enabled."""
        config = MamformerConfig.from_preset("debug")
        config.moe.enabled = True
        config.communicative_moe.enabled = True
        summary = config.summary()
        assert "CommunicativeMoE" in summary
        assert "ENABLED" in summary

    def test_summary_excludes_comm_moe_when_disabled(self):
        """Summary should not mention CommunicativeMoE when disabled."""
        config = MamformerConfig.from_preset("debug")
        config.moe.enabled = True
        config.communicative_moe.enabled = False
        summary = config.summary()
        assert "CommunicativeMoE" not in summary

    def test_comm_moe_in_preset(self):
        """Ultra presets should have comm_moe disabled by default."""
        config = MamformerConfig.from_preset("ultra-7b")
        assert not config.communicative_moe.enabled
