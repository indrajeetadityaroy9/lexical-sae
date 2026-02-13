"""Tests for splade.circuits module — core, geco, losses, activation."""

import math

import pytest
import torch

from splade.circuits.core import CircuitState, circuit_mask
from splade.circuits.geco import GECOController


class TestCircuitState:
    def test_unpack_as_4_tuple(self):
        logits = torch.randn(2, 3)
        sparse = torch.randn(2, 10)
        w_eff = torch.randn(2, 3, 10)
        b_eff = torch.randn(2, 3)
        state = CircuitState(logits, sparse, w_eff, b_eff)

        l, s, w, b = state
        assert torch.equal(l, logits)
        assert torch.equal(s, sparse)
        assert torch.equal(w, w_eff)
        assert torch.equal(b, b_eff)

    def test_named_access(self):
        state = CircuitState(
            logits=torch.randn(1, 2),
            sparse_vector=torch.randn(1, 5),
            W_eff=torch.randn(1, 2, 5),
            b_eff=torch.randn(1, 2),
        )
        assert state.logits.shape == (1, 2)
        assert state.sparse_vector.shape == (1, 5)
        assert state.W_eff.shape == (1, 2, 5)

    def test_is_tuple_subclass(self):
        state = CircuitState(
            torch.randn(1, 2), torch.randn(1, 5),
            torch.randn(1, 2, 5), torch.randn(1, 2),
        )
        assert isinstance(state, tuple)
        assert len(state) == 4


class TestCircuitMask:
    def test_hard_mask_is_nearly_binary(self):
        attr = torch.rand(4, 100)
        mask = circuit_mask(attr, circuit_fraction=0.1, temperature=1e6)
        # At temperature 1e6, mask values should be 0, 0.5 (at threshold), or 1
        assert ((mask < 0.01) | ((mask > 0.49) & (mask < 0.51)) | (mask > 0.99)).all()

    def test_soft_mask_is_smooth(self):
        attr = torch.rand(4, 100)
        mask = circuit_mask(attr, circuit_fraction=0.1, temperature=10.0)
        # At temperature 10, there should be some intermediate values
        intermediate = (mask > 0.01) & (mask < 0.99)
        assert intermediate.any()

    def test_correct_fraction_retained(self):
        attr = torch.rand(4, 1000)
        mask = circuit_mask(attr, circuit_fraction=0.1, temperature=1e6)
        # ~10% of dims should be active (allowing some tolerance)
        active_frac = (mask > 0.5).float().mean(dim=-1)
        assert (active_frac > 0.08).all() and (active_frac < 0.12).all()

    def test_output_shape_matches_input(self):
        attr = torch.rand(8, 200)
        mask = circuit_mask(attr, circuit_fraction=0.2, temperature=50.0)
        assert mask.shape == attr.shape

    def test_fraction_one_retains_all(self):
        attr = torch.rand(2, 50)
        mask = circuit_mask(attr, circuit_fraction=1.0, temperature=1e6)
        # With fraction=1.0, all values should be >= 0.5 (min element sits at threshold)
        assert (mask >= 0.5).all()


class TestGECOController:
    def test_warmup_records_ce(self):
        geco = GECOController()
        for i in range(10):
            geco.record_warmup_ce(1.0 - i * 0.05)
        assert len(geco._warmup_ces) == 10

    def test_finalize_warmup_sets_tau_from_percentile(self):
        """tau_ce should be set from 25th percentile of warmup CE values."""
        geco = GECOController()
        # Feed 100 values; finalize uses last 50: [0.55, 0.559, ..., 0.991]
        for i in range(100):
            geco.record_warmup_ce(0.1 + 0.009 * i)
        tau = geco.finalize_warmup()
        # 25th percentile of last 50 values ≈ 0.66
        assert 0.5 < tau < 0.8

    def test_finalize_warmup_constant_input(self):
        """When all warmup CE values are identical, tau = that value."""
        geco = GECOController()
        for _ in range(60):
            geco.record_warmup_ce(0.5)
        tau = geco.finalize_warmup()
        assert abs(tau - 0.5) < 1e-6

    def test_lambda_increases_when_ce_above_tau(self):
        geco = GECOController()
        for _ in range(10):
            geco.record_warmup_ce(0.5)
        geco.finalize_warmup()

        initial_lambda = geco.lambda_ce
        # CE = 2.0 >> tau, so lambda should increase
        ce = torch.tensor(2.0)
        obj = torch.tensor(0.1)
        geco.compute_loss(ce, obj)
        assert geco.lambda_ce > initial_lambda

    def test_lambda_decreases_when_ce_below_tau(self):
        geco = GECOController()
        for _ in range(10):
            geco.record_warmup_ce(0.5)
        geco.finalize_warmup()

        initial_lambda = geco.lambda_ce
        # CE = 0.01 << tau, so lambda should decrease
        ce = torch.tensor(0.01)
        obj = torch.tensor(0.1)
        geco.compute_loss(ce, obj)
        assert geco.lambda_ce < initial_lambda

    def test_compute_loss_returns_lagrangian(self):
        geco = GECOController()
        for _ in range(10):
            geco.record_warmup_ce(0.5)
        geco.finalize_warmup()

        ce = torch.tensor(2.0, requires_grad=True)
        obj = torch.tensor(3.0, requires_grad=True)
        loss = geco.compute_loss(ce, obj)
        # Loss = obj + lambda_updated * ce. Lambda increases because CE > tau.
        # The returned value should be > obj (3.0) because lambda * ce > 0
        assert loss.item() > 3.0
        # And the loss should have gradients flowing through both inputs
        loss.backward()
        assert ce.grad is not None
        assert obj.grad is not None

    def test_lambda_always_positive(self):
        geco = GECOController()
        for _ in range(10):
            geco.record_warmup_ce(0.5)
        geco.finalize_warmup()

        # Drive lambda down aggressively
        for _ in range(100):
            geco.compute_loss(torch.tensor(0.01), torch.tensor(0.1))
        assert geco.lambda_ce > 0

    def test_no_constructor_args_needed(self):
        """GECOController should work with zero arguments (fully adaptive)."""
        geco = GECOController()
        assert geco.lambda_ce == 1.0
        assert geco.tau_ce is None

    def test_lambda_bounded_by_clamp(self):
        """log_lambda should be clamped to [-5, 5], bounding lambda."""
        from splade.circuits.geco import _LOG_LAMBDA_CLAMP
        geco = GECOController()
        for _ in range(10):
            geco.record_warmup_ce(0.5)
        geco.finalize_warmup()

        # Drive lambda up with many high-CE steps
        for _ in range(1000):
            geco.compute_loss(torch.tensor(10.0), torch.tensor(0.1))
        # Lambda should be bounded by exp(5) ≈ 148.4
        assert geco.lambda_ce <= math.exp(_LOG_LAMBDA_CLAMP) + 1e-6
        assert geco._log_lambda <= _LOG_LAMBDA_CLAMP + 1e-6

        # Drive lambda down with many low-CE steps
        for _ in range(1000):
            geco.compute_loss(torch.tensor(0.001), torch.tensor(0.1))
        # Lambda should be bounded by exp(-5) ≈ 0.007
        assert geco.lambda_ce >= math.exp(-_LOG_LAMBDA_CLAMP) - 1e-6
        assert geco._log_lambda >= -_LOG_LAMBDA_CLAMP - 1e-6

    def test_lambda_stable_moderate_violation(self):
        """With small positive constraint, lambda should grow moderately."""
        geco = GECOController()
        for _ in range(10):
            geco.record_warmup_ce(0.5)
        geco.finalize_warmup()

        # CE slightly above tau (constraint ≈ 0.1)
        for _ in range(50):
            geco.compute_loss(torch.tensor(0.6), torch.tensor(0.1))
        # Lambda should increase but stay reasonable (not millions)
        assert geco.lambda_ce > 1.0  # increased
        assert geco.lambda_ce < 200  # bounded


class TestGradientCentralization:
    def test_centralizes_weight_gradients(self):
        from splade.training.optim import _gradient_centralization

        model = torch.nn.Linear(10, 5)
        # Create a dummy loss and backward
        x = torch.randn(3, 10)
        loss = model(x).sum()
        loss.backward()

        _gradient_centralization(model)

        # After centralization, gradient mean across non-output dims should be ~0
        grad = model.weight.grad
        mean_per_output = grad.mean(dim=1)
        assert torch.allclose(mean_per_output, torch.zeros_like(mean_per_output), atol=1e-6)

    def test_skips_1d_params(self):
        from splade.training.optim import _gradient_centralization

        model = torch.nn.Linear(10, 5)
        x = torch.randn(3, 10)
        loss = model(x).sum()
        loss.backward()

        bias_grad_before = model.bias.grad.clone()
        _gradient_centralization(model)
        # Bias (1D) should be unchanged
        assert torch.equal(model.bias.grad, bias_grad_before)


class TestJumpReLUGate:
    def test_eval_produces_exact_zeros(self):
        from splade.models.layers.activation import JumpReLUGate
        gate = JumpReLUGate(dim=100, init_log_threshold=2.0)  # high threshold
        gate.eval()
        x = torch.randn(4, 100) * 0.1  # small values, below threshold
        out, _, _ = gate(x)
        assert (out == 0.0).sum() > 0

    def test_train_produces_exact_binary_gates(self):
        """JumpReLU should produce exactly {0, 1} gates during training."""
        from splade.models.layers.activation import JumpReLUGate
        gate = JumpReLUGate(dim=100, init_log_threshold=0.0)
        gate.train()
        x = torch.randn(4, 100) * 3.0
        out, gate_mask, _ = gate(x)
        # Gate values should be exactly 0.0 or 1.0
        assert ((gate_mask == 0.0) | (gate_mask == 1.0)).all(), \
            f"Gate has non-binary values: {gate_mask.unique()}"

    def test_train_produces_values(self):
        from splade.models.layers.activation import JumpReLUGate
        gate = JumpReLUGate(dim=100, init_log_threshold=-2.0)  # low threshold
        gate.train()
        x = torch.ones(4, 100) * 2.0
        out, _, _ = gate(x)
        assert out.shape == (4, 100)
        # With low threshold and large input, most should be active
        assert (out > 0).float().mean() > 0.5

    def test_l0_probs_between_0_and_1(self):
        from splade.models.layers.activation import JumpReLUGate
        gate = JumpReLUGate(dim=100)
        x = torch.randn(4, 100)
        _, _, l0_probs = gate(x)
        assert (l0_probs >= 0).all() and (l0_probs <= 1).all()

    def test_3_tuple_return(self):
        from splade.models.layers.activation import JumpReLUGate
        gate = JumpReLUGate(dim=50)
        x = torch.randn(4, 50)
        result = gate(x)
        assert len(result) == 3

    def test_log_threshold_in_state_dict(self):
        from splade.models.layers.activation import JumpReLUGate
        gate = JumpReLUGate(dim=50)
        sd = gate.state_dict()
        assert "log_threshold" in sd

    def test_ste_gradient_flows_to_threshold(self):
        """Verify STE provides gradient to log_threshold."""
        from splade.models.layers.activation import JumpReLUGate
        gate = JumpReLUGate(dim=10, init_log_threshold=0.0)
        gate.train()
        x = torch.randn(8, 10) * 2.0
        out, _, l0_probs = gate(x)
        loss = out.sum() + l0_probs.sum()
        loss.backward()
        assert gate.log_threshold.grad is not None
        assert gate.log_threshold.grad.abs().sum() > 0

    def test_dla_identity_exact_during_training(self):
        """Gate outputs should be exactly binary, making DLA exact in train mode."""
        from splade.models.layers.activation import JumpReLUGate
        gate = JumpReLUGate(dim=50, init_log_threshold=0.0)
        gate.train()
        x = torch.randn(16, 50) * 2.0
        out, gate_mask, _ = gate(x)

        # Verify: output == relu(x) * gate_mask exactly
        z = torch.relu(x)
        reconstructed = z * gate_mask
        assert torch.allclose(out, reconstructed, atol=1e-6)


class TestFeatureFrequencyTracker:
    def test_frequency_updates(self):
        from splade.circuits.losses import FeatureFrequencyTracker
        tracker = FeatureFrequencyTracker(num_features=100, target_sparsity=0.1)
        gate = torch.zeros(4, 100)
        gate[:, :50] = 1.0
        for _ in range(10):
            tracker.update(gate)
        # Features 0-49 should have high frequency, 50-99 should be ~0
        assert tracker.freq_ema[:50].mean() > 0.1
        assert tracker.freq_ema[50:].mean() < 0.01

    def test_step_counter(self):
        from splade.circuits.losses import FeatureFrequencyTracker
        tracker = FeatureFrequencyTracker(num_features=50, target_sparsity=0.1)
        gate = torch.ones(2, 50)
        for _ in range(5):
            tracker.update(gate)
        assert tracker._step == 5


class TestComputeFrequencyPenalty:
    def test_returns_zero_before_warmup(self):
        from splade.circuits.losses import FeatureFrequencyTracker, compute_frequency_penalty
        from splade.models.layers.activation import JumpReLUGate
        tracker = FeatureFrequencyTracker(num_features=50, target_sparsity=0.1)
        activation = JumpReLUGate(dim=50)
        # Before 50 steps, should return 0
        loss = compute_frequency_penalty(tracker, activation)
        assert loss.item() == 0.0

    def test_penalizes_under_active_features(self):
        from splade.circuits.losses import FeatureFrequencyTracker, compute_frequency_penalty
        from splade.models.layers.activation import JumpReLUGate
        tracker = FeatureFrequencyTracker(num_features=10, target_sparsity=0.1)
        # Simulate enough steps with only features 0-4 active
        for _ in range(60):
            gate = torch.zeros(2, 10)
            gate[:, :5] = 1.0
            tracker.update(gate)
        activation = JumpReLUGate(dim=10)
        with torch.no_grad():
            activation.log_threshold[5:] = 5.0
        loss = compute_frequency_penalty(tracker, activation)
        assert loss.item() > 0.0, "Should penalize under-active features"
        loss.backward()
        assert activation.log_threshold.grad is not None
        assert activation.log_threshold.grad[5:].abs().sum() > 0.0


class TestLossNormalizer:
    def test_first_call_returns_one(self):
        from splade.circuits.losses import LossNormalizer
        norm = LossNormalizer()
        loss = torch.tensor(5.0, requires_grad=True)
        result = norm(loss)
        # First call: EMA = 5.0, so result = 5.0 / 5.0 = 1.0
        assert abs(result.item() - 1.0) < 1e-6

    def test_normalizes_to_unit_scale(self):
        from splade.circuits.losses import LossNormalizer
        norm = LossNormalizer()
        # Feed constant value; after convergence, output should be ~1.0
        for _ in range(200):
            result = norm(torch.tensor(42.0))
        assert abs(result.item() - 1.0) < 1e-4

    def test_gradient_flows_through(self):
        from splade.circuits.losses import LossNormalizer
        norm = LossNormalizer()
        x = torch.tensor(3.0, requires_grad=True)
        result = norm(x)
        result.backward()
        assert x.grad is not None
        assert x.grad.item() != 0.0

    def test_different_scales_produce_similar_output(self):
        from splade.circuits.losses import LossNormalizer
        norm_small = LossNormalizer()
        norm_large = LossNormalizer()
        # After convergence, both should output ~1.0 despite 1000x scale difference
        for _ in range(200):
            r_small = norm_small(torch.tensor(0.01))
            r_large = norm_large(torch.tensor(10.0))
        assert abs(r_small.item() - 1.0) < 1e-3
        assert abs(r_large.item() - 1.0) < 1e-3

    def test_adapts_to_changing_scale(self):
        from splade.circuits.losses import LossNormalizer
        norm = LossNormalizer()
        # Converge on scale=1.0
        for _ in range(200):
            norm(torch.tensor(1.0))
        # Switch to scale=100.0; after 500 steps (~5 time constants) should adapt
        for _ in range(500):
            result = norm(torch.tensor(100.0))
        assert abs(result.item() - 1.0) < 0.05


class TestContrastiveSeparationLoss:
    def test_zero_loss_when_well_separated(self):
        from splade.circuits.losses import AttributionCentroidTracker, compute_separation_loss
        tracker = AttributionCentroidTracker(num_classes=2, vocab_size=10)
        # Set up well-separated centroids
        tracker.centroids[0] = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        tracker.centroids[1] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float)
        tracker._initialized[:] = True
        # Without per-sample data, falls back to centroid cosine
        loss = compute_separation_loss(tracker)
        # Orthogonal centroids → cosine = 0
        assert loss.item() < 0.01

    def test_high_loss_when_overlapping(self):
        from splade.circuits.losses import AttributionCentroidTracker, compute_separation_loss
        tracker = AttributionCentroidTracker(num_classes=2, vocab_size=10)
        # Nearly identical centroids
        tracker.centroids[0] = torch.ones(10)
        tracker.centroids[1] = torch.ones(10) * 0.99
        tracker._initialized[:] = True
        loss = compute_separation_loss(tracker)
        assert loss.item() > 0.9

    def test_returns_zero_with_single_class(self):
        from splade.circuits.losses import AttributionCentroidTracker, compute_separation_loss
        tracker = AttributionCentroidTracker(num_classes=3, vocab_size=10)
        tracker.centroids[0] = torch.ones(10)
        tracker._initialized[0] = True
        loss = compute_separation_loss(tracker)
        assert loss.item() == 0.0


class TestKLCompletenessLoss:
    def test_kl_loss_is_non_negative(self):
        from splade.circuits.losses import compute_completeness_loss
        sparse = torch.randn(4, 100).abs()
        W_eff = torch.randn(4, 3, 100)
        labels = torch.tensor([0, 1, 2, 0])
        classifier_fn = torch.nn.Linear(100, 3)
        loss = compute_completeness_loss(
            sparse, W_eff, labels, classifier_fn,
            circuit_fraction=0.1,
        )
        assert loss.item() >= -1e-6  # KL divergence is non-negative

    def test_kl_loss_with_precomputed_logits(self):
        from splade.circuits.losses import compute_completeness_loss
        sparse = torch.randn(4, 100).abs()
        W_eff = torch.randn(4, 3, 100)
        labels = torch.tensor([0, 1, 2, 0])
        classifier_fn = torch.nn.Linear(100, 3)
        full_logits = classifier_fn(sparse)
        loss = compute_completeness_loss(
            sparse, W_eff, labels, classifier_fn,
            circuit_fraction=0.1,
            full_logits=full_logits,
        )
        assert loss.item() >= -1e-6


class TestRemovedConstants:
    """Verify that manually-tuned constants have been removed from constants.py."""

    def test_no_agc_constants(self):
        from splade.training import constants
        assert not hasattr(constants, "AGC_CLIP_FACTOR")
        assert not hasattr(constants, "AGC_EPS")

    def test_no_df_alpha_beta(self):
        from splade.training import constants
        assert not hasattr(constants, "DF_ALPHA")
        assert not hasattr(constants, "DF_BETA")

    def test_no_circuit_loss_weights(self):
        from splade.training import constants
        assert not hasattr(constants, "CC_WEIGHT")
        assert not hasattr(constants, "SEP_WEIGHT")
        assert not hasattr(constants, "SHARP_WEIGHT")

    def test_no_geco_tunable_params(self):
        from splade.training import constants
        assert not hasattr(constants, "GECO_ETA")
        assert not hasattr(constants, "GECO_EMA_DECAY")
        assert not hasattr(constants, "GECO_TAU_MARGIN")

    def test_no_centroid_momentum(self):
        from splade.training import constants
        assert not hasattr(constants, "CENTROID_MOMENTUM")

    def test_no_df_momentum(self):
        from splade.training import constants
        assert not hasattr(constants, "DF_MOMENTUM")

    def test_no_circuit_fraction_in_constants(self):
        from splade.training import constants
        assert not hasattr(constants, "CIRCUIT_FRACTION")
        assert not hasattr(constants, "CIRCUIT_WARMUP_FRACTION")

    def test_jumprelu_bandwidth_exists(self):
        from splade.training import constants
        assert hasattr(constants, "JUMPRELU_BANDWIDTH")
        assert constants.JUMPRELU_BANDWIDTH > 0

    def test_removed_adaptive_constants(self):
        """Verify constants eliminated by adaptive mechanisms are gone."""
        from splade.training import constants
        assert not hasattr(constants, "DEAD_FEATURE_WINDOW")
        assert not hasattr(constants, "AUXK_COEFF")
        assert not hasattr(constants, "WEIGHT_DECAY")
        assert not hasattr(constants, "EMA_DECAY")
        assert not hasattr(constants, "WARMUP_RATIO")
        assert not hasattr(constants, "LR_FIND_STEPS")
        assert not hasattr(constants, "LR_FIND_END")
        assert not hasattr(constants, "LR_FIND_DIVERGE_FACTOR")
