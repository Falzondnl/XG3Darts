"""
Unit tests for margin/blending_engine.py and margin/shin_margin.py.

Tests:
  - test_5_factor_margin_computation
  - test_margin_caps_at_15pct
  - test_regime_widening_r0_highest
  - test_ecosystem_widening_womens
  - test_shin_margin_sums_to_target
  - test_shin_margin_underdog_gets_more_overround
  - test_shin_margin_zero_margin_unchanged
  - test_shin_margin_invalid_probs_raises
"""
from __future__ import annotations

import pytest

from engines.errors import DartsEngineError
from margin.blending_engine import DartsMarginEngine
from margin.shin_margin import ShinMarginModel


class TestDartsMarginEngine:
    """Tests for the 5-factor margin blending engine."""

    def setup_method(self):
        self.engine = DartsMarginEngine()

    def test_5_factor_margin_computation(self):
        """5 factors applied multiplicatively should produce expected result."""
        # All neutral factors except regime R1 (1.10 multiplier)
        margin = self.engine.compute_margin(
            base_margin=0.05,
            regime=1,
            starter_confidence=1.0,
            source_confidence=1.0,
            model_agreement=1.0,
            market_liquidity="high",
            ecosystem="pdc_mens",
        )
        # Only regime multiplier (1.10) applied: 0.05 * 1.10 = 0.055
        assert abs(margin - 0.055) < 1e-6, f"Expected 0.055, got {margin:.8f}"

    def test_5_factor_all_factors_applied(self):
        """All 5 factors should compound correctly."""
        # Use known values for each factor
        # base=0.05, regime=R1(1.10), starter=0.5(→1.15), source=0.5(→1.10),
        # model=0.5(→1.125), ecosystem=pdc_mens, liquidity=high
        margin = self.engine.compute_margin(
            base_margin=0.05,
            regime=1,
            starter_confidence=0.5,
            source_confidence=0.5,
            model_agreement=0.5,
            market_liquidity="high",
            ecosystem="pdc_mens",
        )
        expected = 0.05 * 1.10 * 1.15 * 1.10 * 1.125
        # should be capped at 15% if exceeds it
        expected = min(expected, 0.15)
        assert abs(margin - expected) < 1e-5, f"Expected {expected:.6f}, got {margin:.6f}"

    def test_margin_caps_at_15pct(self):
        """Margin must never exceed 15%, regardless of factor values."""
        # Use worst-case inputs that should push margin well above 15%
        margin = self.engine.compute_margin(
            base_margin=0.10,     # High base
            regime=0,             # R0: 1.30 multiplier
            starter_confidence=0.0,  # Max starter uncertainty
            source_confidence=0.0,   # Max source uncertainty
            model_agreement=0.0,     # Max disagreement
            market_liquidity="low",  # Low liquidity
            ecosystem="development", # Ecosystem widening
        )
        assert margin == 0.15, f"Expected margin to be capped at 0.15, got {margin:.6f}"

    def test_regime_widening_r0_highest(self):
        """R0 should produce a higher margin than R1 which should be higher than R2."""
        base = 0.05
        margins = {}
        for regime in (0, 1, 2):
            margins[regime] = self.engine.compute_margin(
                base_margin=base,
                regime=regime,
                starter_confidence=1.0,
                source_confidence=1.0,
                model_agreement=1.0,
                market_liquidity="high",
                ecosystem="pdc_mens",
            )

        assert margins[0] > margins[1] > margins[2], (
            f"Expected R0 > R1 > R2, got {margins}"
        )
        assert abs(margins[2] - base) < 1e-9, (
            f"R2 with all neutral factors should equal base_margin, got {margins[2]:.6f}"
        )

    def test_ecosystem_widening_womens(self):
        """Women's / development ecosystems should get 20% extra widening."""
        base = 0.05
        mens_margin = self.engine.compute_margin(
            base_margin=base,
            regime=2,
            starter_confidence=1.0,
            source_confidence=1.0,
            model_agreement=1.0,
            market_liquidity="high",
            ecosystem="pdc_mens",
        )
        womens_margin = self.engine.compute_margin(
            base_margin=base,
            regime=2,
            starter_confidence=1.0,
            source_confidence=1.0,
            model_agreement=1.0,
            market_liquidity="high",
            ecosystem="pdc_womens",
        )
        wdf_margin = self.engine.compute_margin(
            base_margin=base,
            regime=2,
            starter_confidence=1.0,
            source_confidence=1.0,
            model_agreement=1.0,
            market_liquidity="high",
            ecosystem="wdf_open",
        )

        assert womens_margin > mens_margin, (
            f"Womens margin ({womens_margin:.4f}) should be > mens ({mens_margin:.4f})"
        )
        assert wdf_margin > mens_margin, (
            f"WDF margin ({wdf_margin:.4f}) should be > mens ({mens_margin:.4f})"
        )
        # Exact: womens = base * 1.20
        assert abs(womens_margin - base * 1.20) < 1e-9

    def test_low_liquidity_widening(self):
        """Low liquidity should add 15% extra."""
        base = 0.05
        high_liq = self.engine.compute_margin(
            base_margin=base, regime=2,
            starter_confidence=1.0, source_confidence=1.0,
            model_agreement=1.0, market_liquidity="high",
            ecosystem="pdc_mens",
        )
        low_liq = self.engine.compute_margin(
            base_margin=base, regime=2,
            starter_confidence=1.0, source_confidence=1.0,
            model_agreement=1.0, market_liquidity="low",
            ecosystem="pdc_mens",
        )
        assert low_liq > high_liq
        assert abs(low_liq - base * 1.15) < 1e-9

    def test_invalid_regime_raises(self):
        """Invalid regime should raise DartsEngineError."""
        with pytest.raises(DartsEngineError, match="regime"):
            self.engine.compute_margin(
                base_margin=0.05,
                regime=3,
                starter_confidence=1.0,
                source_confidence=1.0,
                model_agreement=1.0,
                market_liquidity="high",
                ecosystem="pdc_mens",
            )

    def test_invalid_base_margin_raises(self):
        """Base margin outside (0, 0.15] should raise."""
        with pytest.raises(DartsEngineError):
            self.engine.compute_margin(
                base_margin=0.0,
                regime=1,
                starter_confidence=1.0,
                source_confidence=1.0,
                model_agreement=1.0,
                market_liquidity="high",
                ecosystem="pdc_mens",
            )

    def test_invalid_liquidity_raises(self):
        """Invalid liquidity string should raise."""
        with pytest.raises(DartsEngineError, match="market_liquidity"):
            self.engine.compute_margin(
                base_margin=0.05,
                regime=1,
                starter_confidence=1.0,
                source_confidence=1.0,
                model_agreement=1.0,
                market_liquidity="extreme",
                ecosystem="pdc_mens",
            )


class TestShinMarginModel:
    """Tests for the Shin (1993) margin model."""

    def setup_method(self):
        self.model = ShinMarginModel()

    def test_shin_margin_sums_to_target(self):
        """Adjusted probabilities must sum to approximately 1 + target_margin."""
        probs = {"p1_win": 0.60, "p2_win": 0.40}
        target = 0.05
        result = self.model.apply_shin_margin(probs, target_margin=target)
        total = sum(result.values())
        assert abs(total - (1.0 + target)) < 1e-3, (
            f"Expected sum = {1 + target:.4f}, got {total:.6f}"
        )

    def test_shin_margin_three_outcomes(self):
        """Three-outcome market (including draw) should also sum correctly."""
        probs = {"p1_win": 0.50, "draw": 0.15, "p2_win": 0.35}
        target = 0.06
        result = self.model.apply_shin_margin(probs, target_margin=target)
        total = sum(result.values())
        assert abs(total - (1.0 + target)) < 1e-3

    def test_shin_margin_underdog_gets_more_overround(self):
        """
        The Shin model allocates more overround to underdogs.
        With z_param > 0, the underdog's adjusted probability increases
        proportionally more than the favourite's.
        """
        probs = {"favourite": 0.80, "underdog": 0.20}
        result = self.model.apply_shin_margin(probs, target_margin=0.05, z_param=0.10)

        # Favourite at 80% true prob: margin scaling
        fav_ratio = result["favourite"] / 0.80
        # Underdog at 20% true prob: margin scaling
        dog_ratio = result["underdog"] / 0.20

        # Underdog should receive proportionally more margin
        assert dog_ratio > fav_ratio, (
            f"Shin model: underdog ratio {dog_ratio:.4f} should be > "
            f"favourite ratio {fav_ratio:.4f}"
        )

    def test_shin_margin_zero_margin_unchanged(self):
        """Zero target margin should return normalised true probabilities."""
        probs = {"a": 0.60, "b": 0.40}
        result = self.model.apply_shin_margin(probs, target_margin=0.0)
        # With zero margin the probs should be unchanged (already sum to 1)
        assert abs(result["a"] - 0.60) < 1e-6
        assert abs(result["b"] - 0.40) < 1e-6

    def test_shin_margin_invalid_negative_prob_raises(self):
        """Negative probabilities should raise DartsEngineError."""
        with pytest.raises(DartsEngineError):
            self.model.apply_shin_margin({"a": -0.1, "b": 1.1}, target_margin=0.05)

    def test_shin_margin_empty_dict_raises(self):
        """Empty probability dict should raise DartsEngineError."""
        with pytest.raises(DartsEngineError):
            self.model.apply_shin_margin({}, target_margin=0.05)

    def test_shin_margin_preserves_ranking(self):
        """After margin application, favourite should remain favourite."""
        probs = {"p1_win": 0.70, "p2_win": 0.30}
        result = self.model.apply_shin_margin(probs, target_margin=0.05)
        assert result["p1_win"] > result["p2_win"], (
            "After margin application, the favourite should still have higher prob"
        )

    def test_shin_margin_prob_not_normalised_raises(self):
        """Probability dict summing far from 1 should raise."""
        with pytest.raises(DartsEngineError):
            self.model.apply_shin_margin({"a": 0.5, "b": 0.1}, target_margin=0.05)
