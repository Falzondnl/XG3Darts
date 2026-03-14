"""
Tests for calibration infrastructure.

Tests that 7 calibrators are independent, market lookup works,
and fit/predict pipeline is correct.
"""
from __future__ import annotations

import numpy as np
import pytest

from calibration.beta_calibrator import BetaCalibrator, BetaCalibrationError
from calibration.market_calibrators import (
    MARKET_FAMILIES,
    DartsCalibrationError,
    MarketCalibrationRegistry,
)
from calibration.calibration_monitor import (
    CalibrationMonitor,
    DartsCalibrationMonitorError,
)


def _make_calibration_data(
    n: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate calibration test data with good class separation."""
    rng = np.random.default_rng(seed)
    proba = np.concatenate([
        rng.beta(2, 5, n // 2),
        rng.beta(5, 2, n // 2),
    ])
    y_true = np.concatenate([
        np.zeros(n // 2),
        np.ones(n // 2),
    ])
    return proba, y_true


class TestSevenCalibratorsIndependent:
    """Verify that all 7 market calibrators are independently fitted."""

    def test_7_calibrators_independent(self) -> None:
        """Fitting one calibrator must not affect another."""
        registry = MarketCalibrationRegistry()
        proba, y_true = _make_calibration_data()

        # Fit only match_winner
        registry.fit("match_winner", proba, y_true)

        # match_winner is fitted
        assert registry.is_fitted("match_winner")

        # All others are NOT fitted
        for family in MARKET_FAMILIES:
            if family != "match_winner":
                assert not registry.is_fitted(family), (
                    f"{family} should NOT be fitted"
                )

    def test_7_calibrators_count(self) -> None:
        registry = MarketCalibrationRegistry()
        assert len(registry) == 7

    def test_all_market_families_present(self) -> None:
        assert len(MARKET_FAMILIES) == 7
        expected = {
            "match_winner", "leg_handicap", "totals", "exact_score",
            "props_180", "props_checkout", "outright",
        }
        assert set(MARKET_FAMILIES) == expected

    def test_independent_parameters(self) -> None:
        """Each calibrator must have its own parameters after fitting."""
        registry = MarketCalibrationRegistry()

        # Fit two calibrators with different data
        rng1 = np.random.default_rng(10)
        proba1 = np.concatenate([rng1.beta(1, 5, 250), rng1.beta(5, 1, 250)])
        y1 = np.concatenate([np.zeros(250), np.ones(250)])

        rng2 = np.random.default_rng(20)
        proba2 = np.concatenate([rng2.beta(3, 3, 250), rng2.beta(3, 3, 250)])
        y2 = np.concatenate([np.zeros(250), np.ones(250)])

        registry.fit("match_winner", proba1, y1)
        registry.fit("leg_handicap", proba2, y2)

        cal1 = registry.get_calibrator("match_winner")
        cal2 = registry.get_calibrator("leg_handicap")

        # Parameters should differ (different training data)
        assert cal1.a_ != cal2.a_ or cal1.b_ != cal2.b_


class TestMarketCalibratorLookup:
    """Test market calibrator registry lookup."""

    def test_market_calibrator_lookup(self) -> None:
        registry = MarketCalibrationRegistry()
        for family in MARKET_FAMILIES:
            cal = registry.get_calibrator(family)
            assert isinstance(cal, BetaCalibrator)
            assert cal.market_family == family

    def test_unknown_family_raises(self) -> None:
        registry = MarketCalibrationRegistry()
        with pytest.raises(DartsCalibrationError, match="Unknown market family"):
            registry.get_calibrator("nonexistent_market")

    def test_calibrate_unfitted_raises(self) -> None:
        registry = MarketCalibrationRegistry()
        with pytest.raises(DartsCalibrationError, match="not been fitted"):
            registry.calibrate("match_winner", np.array([0.5]))


class TestCalibratorFitPredict:
    """Test end-to-end fit and predict."""

    def test_calibrator_fit_predict(self) -> None:
        proba, y_true = _make_calibration_data()
        cal = BetaCalibrator(market_family="test")
        cal.fit(proba, y_true)

        test_input = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = cal.predict_proba(test_input)

        assert result.shape == (5,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_calibrator_2d_input(self) -> None:
        """Calibrator must handle (n, 2) probability arrays."""
        proba, y_true = _make_calibration_data()
        cal = BetaCalibrator()
        cal.fit(proba, y_true)

        test_input = np.column_stack([1.0 - proba[:10], proba[:10]])
        result = cal.predict_proba(test_input)
        assert result.shape == (10, 2)
        assert np.allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_calibrator_preserves_ordering(self) -> None:
        """Higher raw probabilities should generally map to higher calibrated."""
        proba, y_true = _make_calibration_data(n=1000)
        cal = BetaCalibrator()
        cal.fit(proba, y_true)

        test_input = np.linspace(0.05, 0.95, 50)
        result = cal.predict_proba(test_input)

        # Check monotonicity (with small tolerance for edge effects)
        diffs = np.diff(result)
        assert np.sum(diffs < -0.01) < 3, "Calibration should be approximately monotone"

    def test_fit_all_markets(self) -> None:
        """fit_all should handle all families."""
        registry = MarketCalibrationRegistry()
        proba, y_true = _make_calibration_data()

        data = {family: (proba, y_true) for family in MARKET_FAMILIES}
        results = registry.fit_all(data)

        assert len(results) == 7
        assert all(results.values())
        assert registry.all_fitted()

    def test_status_report(self) -> None:
        registry = MarketCalibrationRegistry()
        proba, y_true = _make_calibration_data()
        registry.fit("match_winner", proba, y_true)

        status = registry.status()
        assert "match_winner" in status
        assert status["match_winner"]["fitted"] == 1.0


class TestCalibrationMonitor:
    """Test calibration monitoring and validation gates."""

    def test_monitor_evaluate(self) -> None:
        monitor = CalibrationMonitor(n_bins=10)
        rng = np.random.default_rng(42)

        # Well-calibrated predictions
        n = 1000
        true_proba = rng.uniform(0.1, 0.9, n)
        y_true = (rng.random(n) < true_proba).astype(float)

        report = monitor.evaluate("match_winner", true_proba, y_true)
        assert report.market_family == "match_winner"
        assert report.n_samples == n
        assert 0.0 <= report.ece <= 1.0
        assert 0.0 <= report.brier_score <= 1.0

    def test_monitor_gate_failure(self) -> None:
        """Deliberately miscalibrated predictions should fail the gate."""
        monitor = CalibrationMonitor(ece_threshold=0.02)
        rng = np.random.default_rng(42)

        n = 500
        # Badly calibrated: predict all 0.9 but 50% actually positive
        predicted = np.full(n, 0.9)
        y_true = rng.choice([0.0, 1.0], n)

        report = monitor.evaluate("match_winner", predicted, y_true)
        assert not report.passes_gate
        assert len(report.gate_failures) > 0

    def test_monitor_empty_raises(self) -> None:
        monitor = CalibrationMonitor()
        with pytest.raises(DartsCalibrationMonitorError, match="Empty"):
            monitor.evaluate("match_winner", np.array([]), np.array([]))
