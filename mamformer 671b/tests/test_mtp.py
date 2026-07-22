"""
Tests for Multi-Token Prediction (MTP) module.
"""

import pytest
import torch

from mamformer.layers.mtp import MultiTokenPredictor


class TestMultiTokenPredictor:
    """Test MTP module."""

    @pytest.fixture
    def mtp(self):
        """Create an MTP module for testing."""
        return MultiTokenPredictor(
            d_model=256,
            vocab_size=1000,
            depth=2,
            n_heads=8,
            n_kv_heads=2,
            head_dim=32,
            d_ff=512,
            d_state=64,
            d_conv=4,
            max_seq_len=512,
            rope_theta=10000.0,
            dropout=0.0,
            rms_norm_eps=1e-6,
        )

    @pytest.fixture
    def mtp_tied(self):
        """Create an MTP with tied embeddings."""
        embedding_weight = torch.randn(1000, 256)
        return MultiTokenPredictor(
            d_model=256,
            vocab_size=1000,
            depth=2,
            n_heads=8,
            n_kv_heads=2,
            head_dim=32,
            d_ff=512,
            d_state=64,
            max_seq_len=512,
            embedding_weight=embedding_weight,
        )

    @pytest.fixture
    def batch_data(self):
        """Create sample hidden states and labels."""
        hidden_states = torch.randn(4, 16, 256)
        input_ids = torch.randint(0, 1000, (4, 16))
        labels = torch.randint(0, 1000, (4, 16))
        labels[:, 0] = -100  # Mask first position
        return hidden_states, input_ids, labels

    def test_forward_logits_shape(self, mtp, batch_data):
        """MTP should produce logits with correct shape."""
        hidden_states, input_ids, _ = batch_data
        mtp_logits_list, _ = mtp(hidden_states, input_ids, labels=None)

        assert len(mtp_logits_list) == 2  # depth=2
        for logits in mtp_logits_list:
            assert logits.shape == (4, 16, 1000)

    def test_loss_computation(self, mtp, batch_data):
        """MTP should compute non-zero loss."""
        hidden_states, input_ids, labels = batch_data
        _, mtp_loss = mtp(hidden_states, input_ids, labels=labels)

        assert mtp_loss is not None
        assert mtp_loss.item() > 0
        assert torch.isfinite(mtp_loss).all()

    def test_no_loss_without_labels(self, mtp, batch_data):
        """MTP should return None loss when no labels provided."""
        hidden_states, input_ids, _ = batch_data
        _, mtp_loss = mtp(hidden_states, input_ids, labels=None)

        assert mtp_loss is None

    def test_gradient_flow(self, mtp, batch_data):
        """Gradients should flow through MTP components."""
        hidden_states, input_ids, labels = batch_data
        _, mtp_loss = mtp(hidden_states, input_ids, labels=labels)

        mtp_loss.backward()

        # Check that token embedding gets gradients
        assert mtp.token_embedding.weight.grad is not None

        # Check that fusion projections get gradients
        for proj in mtp.fusion_projections:
            assert proj.weight.grad is not None

        # Check that MTP block components get gradients
        for attn in mtp.mtp_attentions:
            assert attn.q_proj.weight.grad is not None
        for ssm in mtp.mtp_ssms:
            assert ssm.in_proj.weight.grad is not None

    def test_tied_embeddings(self, mtp_tied, batch_data):
        """MTP with tied embeddings should work without output_heads."""
        hidden_states, input_ids, labels = batch_data
        mtp_logits_list, mtp_loss = mtp_tied(hidden_states, input_ids, labels=labels)

        assert len(mtp_logits_list) == 2
        assert mtp_loss is not None
        assert mtp_tied.output_heads is None  # Uses tied weight

    def test_generate_mtp_tokens(self, mtp, batch_data):
        """Should generate predicted tokens for speculative decoding."""
        hidden_states, input_ids, _ = batch_data
        tokens = mtp.generate_mtp_tokens(
            hidden_states, input_ids, temperature=1.0
        )

        assert len(tokens) == 2  # depth=2
        for t in tokens:
            assert t.shape == (4, 1)  # (batch, 1) token per depth

    def test_greedy_generation(self, mtp, batch_data):
        """Greedy generation (temperature=0) should be deterministic."""
        hidden_states, input_ids, _ = batch_data

        tokens1 = mtp.generate_mtp_tokens(hidden_states, input_ids, temperature=0)
        tokens2 = mtp.generate_mtp_tokens(hidden_states, input_ids, temperature=0)

        for t1, t2 in zip(tokens1, tokens2):
            assert torch.equal(t1, t2)

    def test_different_depths(self, batch_data):
        """MTP with different depths should work."""
        hidden_states, input_ids, labels = batch_data

        for depth in [1, 2, 3, 4]:
            mtp = MultiTokenPredictor(
                d_model=256, vocab_size=1000, depth=depth,
                n_heads=8, n_kv_heads=2, head_dim=32,
                d_ff=512, d_state=64, max_seq_len=512,
            )
            mtp_logits_list, mtp_loss = mtp(hidden_states, input_ids, labels=labels)
            assert len(mtp_logits_list) == depth
            assert mtp_loss is not None

    def test_variable_batch_size(self, mtp):
        """Should work with different batch sizes."""
        for bs in [1, 2, 8]:
            hidden_states = torch.randn(bs, 16, 256)
            input_ids = torch.randint(0, 1000, (bs, 16))
            labels = torch.randint(0, 1000, (bs, 16))

            mtp_logits_list, mtp_loss = mtp(hidden_states, input_ids, labels=labels)
            for logits in mtp_logits_list:
                assert logits.shape[0] == bs
            assert mtp_loss is not None

    def test_variable_seq_len(self, mtp):
        """Should work with different sequence lengths."""
        for sl in [4, 8, 32]:
            hidden_states = torch.randn(2, sl, 256)
            input_ids = torch.randint(0, 1000, (2, sl))
            labels = torch.randint(0, 1000, (2, sl))

            mtp_logits_list, _ = mtp(hidden_states, input_ids, labels=labels)
            for logits in mtp_logits_list:
                assert logits.shape[1] == sl
