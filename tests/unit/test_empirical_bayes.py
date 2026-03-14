"""
Tests for Empirical Bayes skill estimation.

Tests EB shrinkage application and segment accuracy ranges.
"""
from __future__ import annotations

import numpy as np
import pytest

from models import DartsMLError
from models.empirical_bayes import EmpiricalBayesSkillModel


def _make_throw_data(
    n: int = 200,
    sigma: float = 15.0,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic throw data for testing."""
    rng = np.random.default_rng(seed)
    throws = []
    for _ in range(n):
        # Target: treble 20 (approximately at (103, 0) in board coords)
        target_x = 103.0
        target_y = 0.0
        actual_x = target_x + rng.normal(0, sigma)
        actual_y = target_y + rng.normal(0, sigma)
        throws.append({
            "target_x": target_x,
            "target_y": target_y,
            "actual_x": actual_x,
            "actual_y": actual_y,
            "segment": "T20",
        })
    return throws


class TestEBModelShrinkageApplied:
    """Test that EB shrinkage is applied correctly."""

    def test_eb_model_shrinkage_applied(self) -> None:
        """Shrinkage factor should be between 0 and 1."""
        model = EmpiricalBayesSkillModel()

        throws = _make_throw_data(n=200, sigma=12.0)
        result = model.fit("player_001", throws)

        assert 0.0 <= result["shrinkage_factor"] <= 1.0
        assert result["n_throws"] == 200

    def test_low_data_high_shrinkage(self) -> None:
        """With very few throws, shrinkage should be high (factor near 0)."""
        model = EmpiricalBayesSkillModel(min_throws_for_individual=100)

        throws = _make_throw_data(n=5, sigma=15.0)
        result = model.fit("player_few", throws)

        # With only 5 throws vs minimum 100, shrinkage should be very low
        assert result["shrinkage_factor"] < 0.5

    def test_high_data_low_shrinkage(self) -> None:
        """With many throws, individual estimate should dominate."""
        model = EmpiricalBayesSkillModel(min_throws_for_individual=50)

        throws = _make_throw_data(n=500, sigma=10.0)
        result = model.fit("player_pro", throws)

        # With 500 throws vs minimum 50, shrinkage should be high (near 1.0)
        # meaning individual estimate dominates
        assert result["shrinkage_factor"] > 0.3

    def test_shrinkage_pulls_toward_prior(self) -> None:
        """With full shrinkage, parameters should match the prior."""
        model = EmpiricalBayesSkillModel(min_throws_for_individual=10000)

        # Very few throws -> should shrink heavily toward prior
        throws = _make_throw_data(n=3, sigma=15.0)
        result = model.fit("player_tiny", throws)

        # sigma should be close to prior sigma (~15.0)
        assert abs(result["sigma_x"] - 15.0) < 5.0
        assert abs(result["sigma_y"] - 18.0) < 5.0

    def test_empty_throws_raises(self) -> None:
        model = EmpiricalBayesSkillModel()
        with pytest.raises(DartsMLError, match="no throw data"):
            model.fit("player_empty", [])

    def test_malformed_throw_raises(self) -> None:
        model = EmpiricalBayesSkillModel()
        throws = [{"target_x": 100.0}]  # missing keys
        with pytest.raises(DartsMLError, match="missing keys"):
            model.fit("player_bad", throws)


class TestEBSegmentAccuracyInRange:
    """Test that segment accuracy estimates are in valid range."""

    def test_eb_segment_accuracy_in_range(self) -> None:
        """All segment accuracies must be in [0, 1]."""
        model = EmpiricalBayesSkillModel()

        throws = _make_throw_data(n=200, sigma=12.0)
        result = model.fit("player_001", throws)

        for segment, accuracy in result["segment_accuracies"].items():
            assert 0.0 <= accuracy <= 1.0, (
                f"Segment {segment} accuracy {accuracy} out of range"
            )

    def test_t20_accuracy_elite(self) -> None:
        """Elite player (low sigma) should have higher T20 accuracy."""
        model = EmpiricalBayesSkillModel()

        elite_throws = _make_throw_data(n=500, sigma=8.0)
        elite_result = model.fit("elite", elite_throws)

        avg_throws = _make_throw_data(n=500, sigma=20.0, seed=99)
        avg_result = model.fit("average", avg_throws)

        elite_t20 = elite_result["segment_accuracies"]["T20"]
        avg_t20 = avg_result["segment_accuracies"]["T20"]

        assert elite_t20 > avg_t20, (
            f"Elite T20={elite_t20} should be > average T20={avg_t20}"
        )

    def test_bull_accuracy_very_small(self) -> None:
        """Bull accuracy should be small (tiny target)."""
        model = EmpiricalBayesSkillModel()

        throws = _make_throw_data(n=200, sigma=15.0)
        result = model.fit("player_001", throws)

        d25 = result["segment_accuracies"].get("D25", 0.0)
        # Bull is only 6.35mm radius — very hard to hit
        assert d25 < 0.3, f"D25 accuracy {d25} seems too high"

    def test_p_hit_segment_method(self) -> None:
        """p_hit_segment should return same values as fit output."""
        model = EmpiricalBayesSkillModel()

        throws = _make_throw_data(n=200, sigma=12.0)
        result = model.fit("player_001", throws)

        for segment in result["segment_accuracies"]:
            p = model.p_hit_segment("player_001", segment)
            assert p == result["segment_accuracies"][segment]

    def test_p_hit_unfitted_raises(self) -> None:
        model = EmpiricalBayesSkillModel()
        with pytest.raises(DartsMLError, match="No EB posterior"):
            model.p_hit_segment("nobody", "T20")


class TestEBPredictProba:
    """Test EB as base learner in R2 ensemble."""

    def test_predict_proba_output_range(self) -> None:
        model = EmpiricalBayesSkillModel()
        # Simulate R2 feature array with EB columns at the end
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 0.9, (10, 6))

        proba = model.predict_proba(X)
        assert proba.shape == (10,)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_proba_equal_skill(self) -> None:
        model = EmpiricalBayesSkillModel()
        # Equal skills for both players
        X = np.array([[0.4, 0.4, 0.35, 0.35, 0.2, 0.2]])
        proba = model.predict_proba(X)
        assert abs(proba[0] - 0.5) < 0.01

    def test_predict_proba_dominant_p1(self) -> None:
        model = EmpiricalBayesSkillModel()
        # P1 much better
        X = np.array([[0.8, 0.2, 0.7, 0.1, 0.5, 0.1]])
        proba = model.predict_proba(X)
        assert proba[0] > 0.6
