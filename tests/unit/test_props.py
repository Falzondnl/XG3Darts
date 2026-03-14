"""
Unit tests for prop pricing models.

Tests:
  - test_180_poisson_distribution
  - test_nine_darter_probability_in_range
  - test_checkout_range_sum_to_one
  - test_data_gate_blocks_r2_market_in_r1
  - test_data_gate_allows_r1_checkout
  - test_180_over_plus_under_sum_to_one
  - test_nine_darter_zero_rate_gives_zero_prob
  - test_checkout_highest_over_increases_with_line
"""
from __future__ import annotations

import math

import pytest

from engines.errors import DartsDataError, DartsEngineError
from props.data_sufficiency_gate import DATA_SUFFICIENCY_GATES, DataSufficiencyGate
from props.prop_180 import Prop180Model, _poisson_cdf
from props.prop_checkout import PropCheckoutModel
from props.prop_nine_darter import PropNineDarterModel


# ---------------------------------------------------------------------------
# Poisson helper
# ---------------------------------------------------------------------------

class TestPoissonCDF:
    """Verify the Poisson CDF implementation."""

    def test_poisson_cdf_zero_k(self):
        """P(X <= 0 | lambda=1) = exp(-1)."""
        p = _poisson_cdf(0, 1.0)
        assert abs(p - math.exp(-1.0)) < 1e-8

    def test_poisson_cdf_exact_known(self):
        """P(X <= 2 | lambda=1) = exp(-1)(1 + 1 + 0.5)."""
        expected = math.exp(-1.0) * (1.0 + 1.0 + 0.5)
        p = _poisson_cdf(2, 1.0)
        assert abs(p - expected) < 1e-6

    def test_poisson_cdf_zero_lambda(self):
        """P(X <= k | lambda=0) = 1 for any k >= 0."""
        assert _poisson_cdf(0, 0.0) == 1.0
        assert _poisson_cdf(10, 0.0) == 1.0

    def test_poisson_cdf_negative_k(self):
        """P(X <= -1) = 0."""
        assert _poisson_cdf(-1, 2.0) == 0.0

    def test_poisson_cdf_monotone_increasing(self):
        """CDF must be monotone non-decreasing in k."""
        lam = 5.0
        vals = [_poisson_cdf(k, lam) for k in range(20)]
        for i in range(len(vals) - 1):
            assert vals[i + 1] >= vals[i] - 1e-10


# ---------------------------------------------------------------------------
# Prop180Model
# ---------------------------------------------------------------------------

class TestProp180Model:
    """Tests for the 180 Poisson pricing model."""

    def setup_method(self):
        self.model = Prop180Model()
        # Sufficient stats for gate to pass
        self.p1_stats = {
            "legs_observed": 80,
            "pct_180_ewm": 0.04,
            "confidence": 0.75,
        }
        self.p2_stats = {
            "legs_observed": 80,
            "pct_180_ewm": 0.03,
            "confidence": 0.75,
        }

    def test_180_poisson_distribution(self):
        """
        Test that the over+under probabilities sum to 1.0 and that the
        Poisson model responds correctly to changes in rate.
        """
        result = self.model.price_total_180s(
            p1_180_rate=0.04,
            p2_180_rate=0.04,
            expected_legs=10.0,
            expected_visits_per_leg=9.0,
            line=6.5,
            p1_stats=self.p1_stats,
            p2_stats=self.p2_stats,
            regime=1,
        )
        over = result["over_prob"]
        under = result["under_prob"]

        # Over + under must sum to 1 (no push on half-integer line)
        assert abs(over + under - 1.0) < 1e-6, (
            f"over ({over:.6f}) + under ({under:.6f}) should sum to 1"
        )
        # Both should be between 0 and 1
        assert 0.0 < over < 1.0
        assert 0.0 < under < 1.0

    def test_180_over_plus_under_sum_to_one(self):
        """Over + under must sum to exactly 1.0 for any half-integer line."""
        for line in [3.5, 6.5, 9.5, 12.5]:
            result = self.model.price_total_180s(
                p1_180_rate=0.05,
                p2_180_rate=0.05,
                expected_legs=10.0,
                expected_visits_per_leg=8.5,
                line=line,
                regime=1,
            )
            total = result["over_prob"] + result["under_prob"]
            assert abs(total - 1.0) < 1e-6, (
                f"For line={line}: over+under = {total:.8f} (expected 1.0)"
            )

    def test_180_higher_rate_gives_more_180s(self):
        """Higher 180 rate → higher P(over)."""
        low_rate_result = self.model.price_total_180s(
            p1_180_rate=0.02,
            p2_180_rate=0.02,
            expected_legs=10.0,
            expected_visits_per_leg=9.0,
            line=5.5,
            regime=1,
        )
        high_rate_result = self.model.price_total_180s(
            p1_180_rate=0.08,
            p2_180_rate=0.08,
            expected_legs=10.0,
            expected_visits_per_leg=9.0,
            line=5.5,
            regime=1,
        )
        assert high_rate_result["over_prob"] > low_rate_result["over_prob"], (
            "Higher 180 rate should increase P(over)"
        )

    def test_player_180s_pricing(self):
        """Single player 180s pricing should work with valid inputs."""
        result = self.model.price_player_180s(
            player_180_rate=0.04,
            expected_visits=40.0,
            line=2.5,
            regime=1,
        )
        assert abs(result["over_prob"] + result["under_prob"] - 1.0) < 1e-6

    def test_data_gate_blocked_when_insufficient_legs(self):
        """Gate should block pricing when legs_observed is too low."""
        bad_stats = {
            "legs_observed": 10,  # < 50 required
            "pct_180_ewm": 0.04,
            "confidence": 0.75,
        }
        with pytest.raises(DartsDataError, match="insufficient legs"):
            self.model.price_total_180s(
                p1_180_rate=0.04,
                p2_180_rate=0.04,
                expected_legs=10.0,
                expected_visits_per_leg=9.0,
                line=5.5,
                p1_stats=bad_stats,
                p2_stats=self.p2_stats,
                regime=1,
            )

    def test_invalid_rate_raises(self):
        """Rate outside [0, 1] should raise DartsEngineError."""
        with pytest.raises(DartsEngineError, match="p1_180_rate"):
            self.model.price_total_180s(
                p1_180_rate=1.5,  # invalid
                p2_180_rate=0.04,
                expected_legs=10.0,
                expected_visits_per_leg=9.0,
                line=5.5,
                regime=1,
            )


# ---------------------------------------------------------------------------
# PropNineDarterModel
# ---------------------------------------------------------------------------

class TestPropNineDarterModel:
    """Tests for the nine-darter pricing model."""

    def setup_method(self):
        self.model = PropNineDarterModel()

    def test_nine_darter_probability_in_range(self):
        """
        Nine-darter probability per leg must be in a realistic range.

        For a top PDC player (T20 acc ~0.45, double acc ~0.38):
        P(nine-darter per leg) ≈ (0.45^3)^2 * P(141) is very small but > 0.
        """
        # Top PDC player statistics
        p_leg = self.model.p_nine_darter_in_leg(
            player_t20_accuracy=0.45,
            player_double_accuracy=0.38,
        )
        # Must be very small but positive
        assert p_leg > 0.0, "Nine-darter prob must be positive for good player"
        assert p_leg < 0.01, f"Nine-darter prob unrealistically high: {p_leg:.8f}"

        # Professional average
        p_leg_avg = self.model.p_nine_darter_in_leg(
            player_t20_accuracy=0.35,
            player_double_accuracy=0.30,
        )
        assert p_leg_avg > 0.0
        assert p_leg_avg < p_leg, "Better player should have higher nine-dart prob"

    def test_nine_darter_zero_t20_accuracy_gives_zero(self):
        """Zero T20 accuracy → zero nine-darter probability."""
        p = self.model.p_nine_darter_in_leg(
            player_t20_accuracy=0.0,
            player_double_accuracy=0.30,
        )
        assert p == 0.0

    def test_nine_darter_match_probability_in_range(self):
        """Match nine-darter probability must be between per-leg and 1."""
        p_per_leg = self.model.p_nine_darter_in_leg(
            player_t20_accuracy=0.45,
            player_double_accuracy=0.38,
        )
        p_match = self.model.p_nine_darter_in_match(
            p1_nine_dart_per_leg=p_per_leg,
            p2_nine_dart_per_leg=p_per_leg,
            expected_legs=20.0,
        )
        assert 0.0 < p_match <= 1.0
        assert p_match > p_per_leg, (
            "Match probability should be higher than single-leg probability"
        )

    def test_nine_darter_more_legs_increases_probability(self):
        """More expected legs → higher nine-darter probability."""
        p_per_leg = 1e-5

        p_short = self.model.p_nine_darter_in_match(
            p1_nine_dart_per_leg=p_per_leg,
            p2_nine_dart_per_leg=p_per_leg,
            expected_legs=5.0,
        )
        p_long = self.model.p_nine_darter_in_match(
            p1_nine_dart_per_leg=p_per_leg,
            p2_nine_dart_per_leg=p_per_leg,
            expected_legs=20.0,
        )
        assert p_long > p_short

    def test_invalid_accuracy_raises(self):
        """Accuracy outside [0, 1] should raise DartsEngineError."""
        with pytest.raises(DartsEngineError, match="player_t20_accuracy"):
            self.model.p_nine_darter_in_leg(
                player_t20_accuracy=1.5,
                player_double_accuracy=0.35,
            )


# ---------------------------------------------------------------------------
# PropCheckoutModel
# ---------------------------------------------------------------------------

class TestPropCheckoutModel:
    """Tests for checkout prop model."""

    def setup_method(self):
        self.model = PropCheckoutModel()
        self.player_stats = {
            "legs_observed": 60,
            "checkout_ewm": 0.43,
            "checkout_pct_by_band": {
                "sub_40": 0.30,
                "40_60": 0.25,
                "61_80": 0.20,
                "81_100": 0.12,
                "101_120": 0.07,
                "121_140": 0.04,
                "141_170": 0.02,
            },
            "confidence": 0.7,
        }

    def test_checkout_range_sum_to_one(self):
        """
        Non-overlapping checkout ranges that tile [2, 170] should sum to 1.
        """
        # Build ranges that cover [2, 170] completely without overlap
        ranges = [(2, 39), (40, 60), (61, 80), (81, 100), (101, 120), (121, 140), (141, 170)]
        total_prob = 0.0
        for lo, hi in ranges:
            result = self.model.price_checkout_range(
                player_checkout_model=self.player_stats,
                line_low=lo,
                line_high=hi,
                player_id="test_player",
                regime=1,
            )
            total_prob += result["in_range_prob"]

        assert abs(total_prob - 1.0) < 1e-4, (
            f"Checkout range probabilities should sum to 1.0, got {total_prob:.6f}"
        )

    def test_checkout_range_partial(self):
        """Partial range probability should be in (0, 1)."""
        result = self.model.price_checkout_range(
            player_checkout_model=self.player_stats,
            line_low=100,
            line_high=170,
            player_id="test_player",
            regime=1,
        )
        assert 0.0 < result["in_range_prob"] < 1.0
        assert abs(result["in_range_prob"] + result["out_range_prob"] - 1.0) < 1e-6

    def test_checkout_highest_checkout_over_probability(self):
        """Highest checkout probability should be between 0 and 1."""
        result = self.model.price_highest_checkout(
            p1_checkout_model=self.player_stats,
            p2_checkout_model=self.player_stats,
            expected_legs=10.0,
            line=100,
            p1_id="p1",
            p2_id="p2",
            regime=1,
        )
        assert 0.0 < result["over_prob"] < 1.0
        assert abs(result["over_prob"] + result["under_prob"] - 1.0) < 1e-6

    def test_higher_line_lower_over_prob(self):
        """Higher checkout line → lower P(over) for highest checkout."""
        low_line = self.model.price_highest_checkout(
            p1_checkout_model=self.player_stats,
            p2_checkout_model=self.player_stats,
            expected_legs=10.0,
            line=80,
            p1_id="p1",
            p2_id="p2",
            regime=1,
        )
        high_line = self.model.price_highest_checkout(
            p1_checkout_model=self.player_stats,
            p2_checkout_model=self.player_stats,
            expected_legs=10.0,
            line=140,
            p1_id="p1",
            p2_id="p2",
            regime=1,
        )
        assert high_line["over_prob"] < low_line["over_prob"], (
            f"Higher line should produce lower P(over): "
            f"line=80: {low_line['over_prob']:.4f}, line=140: {high_line['over_prob']:.4f}"
        )


# ---------------------------------------------------------------------------
# DataSufficiencyGate
# ---------------------------------------------------------------------------

class TestDataSufficiencyGate:
    """Tests for data sufficiency gate."""

    def setup_method(self):
        self.gate = DataSufficiencyGate()

    def test_data_gate_blocks_r2_market_in_r1(self):
        """
        A market requiring R2 data must be blocked when regime < 2.
        """
        stats = {
            "legs_observed": 200,
            "double_hit_rate": 0.35,
            "eb_double_accuracy": 0.35,
            "confidence": 0.9,
        }
        # props_double_segment requires R2
        ok_r1, reason_r1 = self.gate.can_open_market(
            market_family="props_double_segment",
            player_id="test_player",
            regime=1,
            stats=stats,
        )
        assert not ok_r1, f"Should be blocked at R1 but got ok=True, reason={reason_r1}"
        assert "R2" in reason_r1 or "regime" in reason_r1.lower()

        ok_r0, reason_r0 = self.gate.can_open_market(
            market_family="props_double_segment",
            player_id="test_player",
            regime=0,
            stats=stats,
        )
        assert not ok_r0

        # Should pass at R2
        ok_r2, reason_r2 = self.gate.can_open_market(
            market_family="props_double_segment",
            player_id="test_player",
            regime=2,
            stats=stats,
        )
        assert ok_r2, f"Should pass at R2 but got ok=False, reason={reason_r2}"

    def test_data_gate_allows_r1_checkout(self):
        """
        props_checkout does not require R2, so R1 with sufficient data should pass.
        """
        good_stats = {
            "legs_observed": 40,  # > 30 required
            "checkout_ewm": 0.43,
            "checkout_pct_by_band": {"sub_40": 0.5, "40_60": 0.5},
            "confidence": 0.7,
        }
        ok, reason = self.gate.can_open_market(
            market_family="props_checkout",
            player_id="test_player",
            regime=1,
            stats=good_stats,
        )
        assert ok, f"Expected gate to pass at R1 with sufficient data. reason={reason}"

    def test_data_gate_blocks_insufficient_legs(self):
        """Gate must block when legs_observed is below threshold."""
        stats = {
            "legs_observed": 5,  # way below any threshold
            "pct_180_ewm": 0.04,
            "confidence": 0.8,
        }
        ok, reason = self.gate.can_open_market(
            market_family="props_180",
            player_id="test_player",
            regime=2,  # even R2 can't overcome insufficient legs
            stats=stats,
        )
        assert not ok
        assert "legs" in reason.lower()

    def test_data_gate_blocks_missing_required_field(self):
        """Gate must block when a required coverage field is missing."""
        stats = {
            "legs_observed": 60,
            # pct_180_ewm missing
            "confidence": 0.8,
        }
        ok, reason = self.gate.can_open_market(
            market_family="props_180",
            player_id="test_player",
            regime=2,
            stats=stats,
        )
        assert not ok
        assert "pct_180_ewm" in reason

    def test_data_gate_blocks_low_confidence(self):
        """Gate must block when confidence is below threshold."""
        stats = {
            "legs_observed": 60,
            "pct_180_ewm": 0.04,
            "confidence": 0.5,  # below 0.7 required
        }
        ok, reason = self.gate.can_open_market(
            market_family="props_180",
            player_id="test_player",
            regime=2,
            stats=stats,
        )
        assert not ok
        assert "confidence" in reason.lower()

    def test_unknown_market_family_raises(self):
        """Unknown market family should raise DartsDataError."""
        from engines.errors import DartsDataError
        with pytest.raises(DartsDataError, match="Unknown market family"):
            self.gate.can_open_market(
                market_family="nonexistent_market",
                player_id="test",
                regime=1,
                stats={},
            )
