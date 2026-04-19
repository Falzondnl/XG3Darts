"""
Unit tests for the darts Pinnacle logit blend function.

Tests blend_with_pinnacle() from shared.pricing_layer.

Coverage:
  - Nominal blend: model 0.60, pinnacle_prob 0.55, weight 0.25 → verifiable logit result
  - Pinnacle unavailable (None) → model_prob returned unchanged
  - Pinnacle degenerate (out of (0.001, 0.999)) → model_prob returned unchanged
  - RuntimeError raised for invalid model_prob (0.0 boundary)
  - RuntimeError raised for invalid model_prob (1.0 boundary)
  - RuntimeError raised for invalid weight (0.0)
  - Blend is monotone: larger pinnacle_prob pulls blended result upward
  - Blend result is strictly inside (0, 1) for all valid inputs
  - Weight=0.0 and weight=1.0 are rejected (RuntimeError)
  - Numerical verification against manual logit calculation
"""
from __future__ import annotations

import math
import pytest

from shared.pricing_layer import blend_with_pinnacle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logit(p: float, clamp_lo: float = 0.02, clamp_hi: float = 0.98) -> float:
    p = max(clamp_lo, min(clamp_hi, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x > 20.0:
        return 1.0 - 1e-9
    if x < -20.0:
        return 1e-9
    return 1.0 / (1.0 + math.exp(-x))


def _expected_blend(model_prob: float, pinnacle_prob: float, weight: float) -> float:
    """Manual logit-space blend — mirrors pricing_layer internals."""
    return _sigmoid(weight * _logit(model_prob) + (1.0 - weight) * _logit(pinnacle_prob))


# ---------------------------------------------------------------------------
# Nominal blend
# ---------------------------------------------------------------------------

class TestNominalBlend:
    """Blend is applied correctly when all inputs are valid."""

    def test_blend_matches_manual_logit_calculation(self) -> None:
        """
        Verify numerical result equals manual logit blend calculation.

        Inputs:  model_prob=0.60, pinnacle_prob=0.55, weight=0.25 (default)
        Expected: sigmoid(0.25 * logit(0.60) + 0.75 * logit(0.55))
        """
        model_prob = 0.60
        pinnacle_prob = 0.55
        weight = 0.25

        expected = _expected_blend(model_prob, pinnacle_prob, weight)
        result = blend_with_pinnacle(model_prob, pinnacle_prob, weight)

        assert abs(result - expected) < 1e-9, (
            f"blend_with_pinnacle returned {result:.8f}; expected {expected:.8f}"
        )

    def test_blend_result_is_strictly_between_inputs(self) -> None:
        """
        When model_prob > pinnacle_prob, the blended result lies strictly
        between the two inputs (i.e., Pinnacle pulls the model toward itself).
        """
        model_prob = 0.70
        pinnacle_prob = 0.55
        result = blend_with_pinnacle(model_prob, pinnacle_prob)

        assert pinnacle_prob < result < model_prob, (
            f"Expected {pinnacle_prob} < {result:.6f} < {model_prob}"
        )

    def test_blend_result_inside_open_unit_interval(self) -> None:
        """Result is always strictly in (0, 1) for valid inputs."""
        result = blend_with_pinnacle(0.30, 0.40)
        assert 0.0 < result < 1.0

    def test_pinnacle_anchor_dominates_at_default_weight(self) -> None:
        """
        At default weight=0.25, result should be closer to pinnacle_prob than
        to model_prob (Pinnacle 75% weight, model 25% weight).
        """
        model_prob = 0.90
        pinnacle_prob = 0.50
        result = blend_with_pinnacle(model_prob, pinnacle_prob)

        dist_to_pinnacle = abs(result - pinnacle_prob)
        dist_to_model = abs(result - model_prob)

        assert dist_to_pinnacle < dist_to_model, (
            f"Expected result {result:.4f} to be closer to pinnacle_prob "
            f"{pinnacle_prob} than model_prob {model_prob}. "
            f"dist_pinnacle={dist_to_pinnacle:.4f}, dist_model={dist_to_model:.4f}"
        )

    def test_sample_blend_calculation_explicit(self) -> None:
        """
        Hard-coded numerical sample:
          model_prob = 0.65, pinnacle_prob = 0.58, weight = 0.25
          logit(0.65) ≈ 0.6190, logit(0.58) ≈ 0.3228
          blended_logit = 0.25 * 0.6190 + 0.75 * 0.3228 = 0.3973
          sigmoid(0.3973) ≈ 0.5980
        """
        model_prob = 0.65
        pinnacle_prob = 0.58
        weight = 0.25

        blended_logit = weight * math.log(0.65 / 0.35) + (1 - weight) * math.log(0.58 / 0.42)
        expected = 1.0 / (1.0 + math.exp(-blended_logit))
        result = blend_with_pinnacle(model_prob, pinnacle_prob, weight)

        assert abs(result - expected) < 1e-9, (
            f"Expected {expected:.8f}, got {result:.8f}"
        )


# ---------------------------------------------------------------------------
# Pinnacle unavailable / degenerate — model_prob pass-through
# ---------------------------------------------------------------------------

class TestPinnacleUnavailable:
    """When pinnacle_prob is None or degenerate, model_prob is returned unchanged."""

    def test_none_pinnacle_returns_model_prob(self) -> None:
        model_prob = 0.62
        result = blend_with_pinnacle(model_prob, pinnacle_prob=None)
        assert result == model_prob, (
            f"Expected model_prob {model_prob} unchanged, got {result}"
        )

    def test_zero_pinnacle_prob_returns_model_prob(self) -> None:
        """pinnacle_prob=0.0 is degenerate → model_prob returned unchanged."""
        model_prob = 0.55
        result = blend_with_pinnacle(model_prob, pinnacle_prob=0.0)
        assert result == model_prob

    def test_one_pinnacle_prob_returns_model_prob(self) -> None:
        """pinnacle_prob=1.0 is degenerate → model_prob returned unchanged."""
        model_prob = 0.55
        result = blend_with_pinnacle(model_prob, pinnacle_prob=1.0)
        assert result == model_prob

    def test_boundary_low_pinnacle_prob_returns_model_prob(self) -> None:
        """pinnacle_prob=0.001 is exactly at boundary → model_prob returned unchanged."""
        model_prob = 0.48
        result = blend_with_pinnacle(model_prob, pinnacle_prob=0.001)
        assert result == model_prob

    def test_boundary_high_pinnacle_prob_returns_model_prob(self) -> None:
        """pinnacle_prob=0.999 is exactly at boundary → model_prob returned unchanged."""
        model_prob = 0.48
        result = blend_with_pinnacle(model_prob, pinnacle_prob=0.999)
        assert result == model_prob

    def test_valid_just_inside_boundary(self) -> None:
        """pinnacle_prob just inside (0.001, 0.999) → blend is applied (result differs from model_prob)."""
        model_prob = 0.60
        pinnacle_prob = 0.0011  # just above 0.001
        result = blend_with_pinnacle(model_prob, pinnacle_prob)
        # Should not equal model_prob because blend was applied
        assert result != model_prob


# ---------------------------------------------------------------------------
# RuntimeError on invalid inputs
# ---------------------------------------------------------------------------

class TestInvalidInputs:
    """RuntimeError is raised for invalid model_prob or weight values."""

    def test_model_prob_zero_raises(self) -> None:
        with pytest.raises(RuntimeError, match="model_prob"):
            blend_with_pinnacle(0.0, pinnacle_prob=0.50)

    def test_model_prob_one_raises(self) -> None:
        with pytest.raises(RuntimeError, match="model_prob"):
            blend_with_pinnacle(1.0, pinnacle_prob=0.50)

    def test_model_prob_negative_raises(self) -> None:
        with pytest.raises(RuntimeError, match="model_prob"):
            blend_with_pinnacle(-0.1, pinnacle_prob=0.50)

    def test_model_prob_above_one_raises(self) -> None:
        with pytest.raises(RuntimeError, match="model_prob"):
            blend_with_pinnacle(1.1, pinnacle_prob=0.50)

    def test_weight_zero_raises(self) -> None:
        with pytest.raises(RuntimeError, match="weight"):
            blend_with_pinnacle(0.60, pinnacle_prob=0.55, weight=0.0)

    def test_weight_one_raises(self) -> None:
        with pytest.raises(RuntimeError, match="weight"):
            blend_with_pinnacle(0.60, pinnacle_prob=0.55, weight=1.0)

    def test_weight_negative_raises(self) -> None:
        with pytest.raises(RuntimeError, match="weight"):
            blend_with_pinnacle(0.60, pinnacle_prob=0.55, weight=-0.1)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

class TestMonotonicity:
    """
    Larger pinnacle_prob → larger blended result (monotone in pinnacle anchor).
    """

    def test_blend_monotone_increasing_in_pinnacle_prob(self) -> None:
        """
        Holding model_prob constant, increasing pinnacle_prob should
        increase the blended result.
        """
        model_prob = 0.50
        pinnacle_probs = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
        results = [blend_with_pinnacle(model_prob, pp) for pp in pinnacle_probs]

        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], (
                f"Non-monotone at index {i}: result[{i}]={results[i]:.6f} "
                f"vs result[{i+1}]={results[i+1]:.6f} "
                f"(pinnacle_probs={pinnacle_probs[i]}, {pinnacle_probs[i+1]})"
            )

    def test_blend_monotone_increasing_in_model_prob(self) -> None:
        """
        Holding pinnacle_prob constant, increasing model_prob should
        increase the blended result.
        """
        pinnacle_prob = 0.50
        model_probs = [0.30, 0.40, 0.50, 0.60, 0.70]
        results = [blend_with_pinnacle(mp, pinnacle_prob) for mp in model_probs]

        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], (
                f"Non-monotone at index {i}: result[{i}]={results[i]:.6f} "
                f"vs result[{i+1}]={results[i+1]:.6f}"
            )


# ---------------------------------------------------------------------------
# Equal inputs edge case
# ---------------------------------------------------------------------------

class TestEqualInputs:
    """When model_prob == pinnacle_prob, blend result should equal (approximately) both."""

    @pytest.mark.parametrize("p", [0.30, 0.45, 0.50, 0.60, 0.75])
    def test_equal_inputs_idempotent(self, p: float) -> None:
        """blend_with_pinnacle(p, p) should return approximately p."""
        result = blend_with_pinnacle(p, pinnacle_prob=p)
        assert abs(result - p) < 1e-6, (
            f"Expected blend({p}, {p}) ≈ {p}, got {result:.8f}"
        )
