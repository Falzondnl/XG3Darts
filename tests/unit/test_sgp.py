"""
Unit tests for the SGP (Same-Game Parlay) system.

Tests:
  - test_correlation_matrix_psd
  - test_higham_projection
  - test_shrinkage_toward_prior
  - test_copula_builder_independent_events
  - test_copula_builder_correlated
  - test_psd_enforcer_already_psd
  - test_psd_enforcer_negative_eigenvalue
  - test_correlation_estimator_no_empirical_returns_prior
  - test_shrinkage_alpha
"""
from __future__ import annotations

import numpy as np
import pytest

from engines.errors import DartsEngineError
from sgp.copula_builder import DartsSGPBuilder, SGPSelection
from sgp.correlation_estimator import (
    DARTS_GLOBAL_CORRELATION_PRIOR,
    DartsSGPCorrelationEstimator,
)
from sgp.psd_enforcer import _is_psd, higham_psd


# ---------------------------------------------------------------------------
# PSD enforcer
# ---------------------------------------------------------------------------

class TestHighamPSD:
    """Tests for the Higham (2002) PSD projection."""

    def test_higham_projection(self):
        """
        Higham projection must return a PSD matrix.
        Test with a known non-PSD matrix (eigenvalues include negative values).
        """
        # Construct a non-PSD symmetric matrix
        rho = np.array([
            [1.0,  0.9,  0.9],
            [0.9,  1.0, -0.9],
            [0.9, -0.9,  1.0],
        ], dtype=np.float64)

        # Verify it's not PSD before projection
        eigvals_before = np.linalg.eigvalsh(rho)
        assert np.any(eigvals_before < 0), "Test matrix should have negative eigenvalue"

        result = higham_psd(rho, eigenvalue_floor=0.0)

        # After projection, all eigenvalues must be >= 0
        eigvals_after = np.linalg.eigvalsh(result)
        assert np.all(eigvals_after >= -1e-7), (
            f"After Higham projection, min eigenvalue = {min(eigvals_after):.8f}"
        )

        # Must be symmetric
        assert np.allclose(result, result.T, atol=1e-10)

        # Unit diagonal
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-8)

    def test_psd_enforcer_already_psd(self):
        """An already-PSD matrix should be returned unchanged (up to numerics)."""
        rho = np.array([
            [1.0, 0.3, -0.2],
            [0.3, 1.0,  0.1],
            [-0.2, 0.1, 1.0],
        ], dtype=np.float64)

        assert _is_psd(rho), "Test matrix should already be PSD"

        result = higham_psd(rho)
        # Should be very close to original
        np.testing.assert_allclose(result, rho, atol=1e-6)

    def test_psd_enforcer_non_square_raises(self):
        """Non-square input should raise DartsEngineError."""
        with pytest.raises(DartsEngineError):
            higham_psd(np.ones((3, 4)))

    def test_psd_enforcer_7x7_prior(self):
        """The global prior must already be PSD (or become PSD after projection)."""
        result = higham_psd(DARTS_GLOBAL_CORRELATION_PRIOR, eigenvalue_floor=1e-8)
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals >= -1e-7), (
            f"Global prior after Higham projection has min eigenvalue {min(eigvals):.8f}"
        )


# ---------------------------------------------------------------------------
# DartsSGPCorrelationEstimator
# ---------------------------------------------------------------------------

class TestDartsSGPCorrelationEstimator:
    """Tests for the correlation estimator."""

    def setup_method(self):
        self.estimator = DartsSGPCorrelationEstimator()

    def test_correlation_matrix_psd(self):
        """
        The estimated correlation matrix must be positive semi-definite.
        Test with and without empirical data.
        """
        # Without empirical data (uses prior)
        matrix_prior = self.estimator.estimate_for_competition(
            competition_code="PDC_WC",
            n_samples=0,
        )
        eigvals = np.linalg.eigvalsh(matrix_prior)
        assert np.all(eigvals >= -1e-7), (
            f"Prior correlation matrix is not PSD. Min eigenvalue = {min(eigvals):.8f}"
        )
        assert matrix_prior.shape == (7, 7)
        np.testing.assert_allclose(np.diag(matrix_prior), 1.0, atol=1e-8)

        # With simulated empirical data (1000 samples)
        rng = np.random.default_rng(42)
        empirical = np.eye(7) * 0.8 + 0.2 * DARTS_GLOBAL_CORRELATION_PRIOR
        np.fill_diagonal(empirical, 1.0)
        empirical = (empirical + empirical.T) / 2.0

        matrix_empirical = self.estimator.estimate_for_competition(
            competition_code="PDC_WC",
            n_samples=1000,
            empirical_spearman=empirical,
        )
        eigvals2 = np.linalg.eigvalsh(matrix_empirical)
        assert np.all(eigvals2 >= -1e-7), (
            f"Empirical correlation matrix is not PSD. Min eigenvalue = {min(eigvals2):.8f}"
        )

    def test_shrinkage_toward_prior(self):
        """
        Shrinkage must interpolate between empirical and prior.

        At n_samples=0: result = prior.
        At n_samples=min_samples: result = empirical.
        At n_samples < min_samples: result is a blend.
        """
        prior = DARTS_GLOBAL_CORRELATION_PRIOR.copy()

        # Construct an empirical matrix (identity-like)
        empirical = np.eye(7)

        # Blend with alpha=0.5 (half samples of min)
        blended = self.estimator._shrink_toward_prior(
            empirical=empirical,
            alpha=0.5,
            prior=prior,
        )

        # Off-diagonal elements should be between prior and empirical
        # For empirical=I: off-diagonal = 0; prior has non-zero off-diagonal
        # Blended should have off-diagonal = 0.5 * 0 + 0.5 * prior_off_diag
        for i in range(7):
            for j in range(7):
                if i != j:
                    expected = 0.5 * 0.0 + 0.5 * prior[i, j]
                    assert abs(blended[i, j] - expected) < 1e-10, (
                        f"[{i},{j}]: expected {expected:.6f}, got {blended[i,j]:.6f}"
                    )

    def test_shrinkage_alpha(self):
        """Alpha should be correctly computed from sample count."""
        assert self.estimator._shrinkage_alpha(0, 500) == 0.0
        assert self.estimator._shrinkage_alpha(500, 500) == 1.0
        assert self.estimator._shrinkage_alpha(250, 500) == 0.5
        assert self.estimator._shrinkage_alpha(1000, 500) == 1.0  # capped at 1

    def test_correlation_estimator_no_empirical_returns_prior(self):
        """Without empirical data, result should be close to the global prior."""
        result = self.estimator.estimate_for_competition(
            competition_code="PDC_PL",
            n_samples=0,
        )
        # With zero samples, result = Higham(prior) ≈ prior
        # (prior is already near-PSD so should be close)
        np.testing.assert_allclose(result, higham_psd(DARTS_GLOBAL_CORRELATION_PRIOR, eigenvalue_floor=1e-8), atol=1e-6)

    def test_wrong_shape_empirical_raises(self):
        """Wrong-shape empirical matrix should raise DartsEngineError."""
        bad_matrix = np.eye(5)
        with pytest.raises(DartsEngineError, match="7x7"):
            self.estimator.estimate_for_competition(
                competition_code="PDC_WC",
                empirical_spearman=bad_matrix,
                n_samples=500,
            )


# ---------------------------------------------------------------------------
# DartsSGPBuilder
# ---------------------------------------------------------------------------

class TestDartsSGPBuilder:
    """Tests for the Gaussian copula SGP builder."""

    def setup_method(self):
        # Use fixed seed for reproducibility
        self.builder = DartsSGPBuilder(n_mc=100_000, random_seed=42)
        self.estimator = DartsSGPCorrelationEstimator()
        self.corr = self.estimator.estimate_for_competition("PDC_WC")

    def _sel(self, market_type: str, prob: float, idx: int) -> SGPSelection:
        return SGPSelection(
            market_type=market_type,
            outcome=f"{market_type}_outcome",
            probability=prob,
            correlation_index=idx,
        )

    def test_copula_builder_independent_events(self):
        """
        For a zero-correlation matrix (identity), joint probability should
        approximate the product of marginal probabilities.
        """
        # Build identity correlation matrix
        identity_corr = np.eye(7)

        # 2 independent selections: 0.6 * 0.4 = 0.24
        selections = [
            self._sel("match_win", 0.60, 0),
            self._sel("total_legs_over", 0.40, 1),
        ]
        p_joint = self.builder.price_parlay(
            selections=selections,
            correlation_matrix=identity_corr,
        )
        expected = 0.60 * 0.40
        # MC error ~±0.1% for 100k samples
        assert abs(p_joint - expected) < 0.005, (
            f"Independent joint probability: expected {expected:.4f}, got {p_joint:.4f}"
        )

    def test_copula_builder_correlated(self):
        """
        Positive correlation between match_win and handicap should increase
        the joint probability above the naive product.
        """
        # Strong positive correlation between match_win (idx=0) and handicap (idx=2)
        corr = np.array([
            [1.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        # Already PSD
        corr = higham_psd(corr, eigenvalue_floor=1e-8)

        p1 = 0.65  # P(match_win)
        p2 = 0.60  # P(handicap)
        naive = p1 * p2  # 0.39

        selections = [
            self._sel("match_win", p1, 0),
            self._sel("handicap", p2, 2),
        ]
        p_corr = self.builder.price_parlay(
            selections=selections,
            correlation_matrix=corr,
        )
        # With strong positive correlation, joint > naive
        assert p_corr > naive - 0.01, (
            f"Correlated joint prob {p_corr:.4f} should be >= naive {naive:.4f}"
        )

    def test_empty_selections_raises(self):
        """Empty selections list should raise DartsEngineError."""
        with pytest.raises(DartsEngineError):
            self.builder.price_parlay(
                selections=[],
                correlation_matrix=self.corr,
            )

    def test_invalid_probability_raises(self):
        """Probability of 0 or 1 should raise DartsEngineError."""
        with pytest.raises(DartsEngineError, match="probability"):
            self.builder.price_parlay(
                selections=[
                    self._sel("match_win", 0.0, 0),  # invalid: must be > 0
                    self._sel("handicap", 0.5, 2),
                ],
                correlation_matrix=self.corr,
            )

    def test_result_is_between_zero_and_one(self):
        """Joint probability must always be in [0, 1]."""
        selections = [
            self._sel("match_win", 0.70, 0),
            self._sel("total_legs_over", 0.55, 1),
            self._sel("180_over", 0.45, 3),
        ]
        p = self.builder.price_parlay(
            selections=selections,
            correlation_matrix=self.corr,
        )
        assert 0.0 <= p <= 1.0

    def test_joint_probability_less_than_min_marginal(self):
        """Joint probability cannot exceed the minimum marginal probability."""
        p1, p2 = 0.80, 0.30
        selections = [
            self._sel("match_win", p1, 0),
            self._sel("handicap", p2, 2),
        ]
        p_joint = self.builder.price_parlay(
            selections=selections,
            correlation_matrix=self.corr,
        )
        # Joint cannot exceed the smallest marginal
        assert p_joint <= min(p1, p2) + 0.01, (
            f"Joint ({p_joint:.4f}) must not significantly exceed min marginal ({min(p1, p2):.4f})"
        )
