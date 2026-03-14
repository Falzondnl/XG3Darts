"""
Unit tests for the starter inference engine.

Tests:
  - test_infer_alternating_starts
  - test_first_leg_unknown_confidence
  - test_margin_widening_factor_values
  - test_confirmed_starter_from_feed
  - test_inference_chain_decay
  - test_margin_widening_monotone
"""

from __future__ import annotations

import pytest

from engines.state_layer.starter_inference import LegStarterInfo, StarterInferenceEngine


PLAYERS = ("player_a", "player_b")


class TestInferAlternatingStarts:
    """G: alternating starts inference from a confirmed anchor."""

    def test_leg_1_confirmed_p1_starts(self):
        """Confirmed starter for leg 1 is P1: returned as-is."""
        engine = StarterInferenceEngine()
        info = engine.infer_starter(
            leg_number=1,
            players=PLAYERS,
            alternating_starts=True,
            feed_starter_id="player_a",
            feed_confidence=1.0,
            feed_source="dartconnect",
        )
        assert info.starter_player_id == "player_a"
        assert info.is_confirmed
        assert info.confidence == 1.0
        assert info.source == "dartconnect"

    def test_leg_2_inferred_from_leg_1(self):
        """Leg 2 starter inferred from confirmed leg 1 starter."""
        engine = StarterInferenceEngine()
        leg1 = LegStarterInfo(
            leg_number=1,
            starter_player_id="player_a",
            source="dartconnect",
            confidence=1.0,
            is_confirmed=True,
        )
        info = engine.infer_starter(
            leg_number=2,
            players=PLAYERS,
            alternating_starts=True,
            previous_starters=[leg1],
        )
        assert info.starter_player_id == "player_b"  # alternates
        assert info.source == "inferred"
        assert not info.is_confirmed
        assert info.confidence > 0.8  # high confidence from close anchor

    def test_leg_3_inferred_same_as_leg_1(self):
        """Leg 3 starter is same as leg 1 (2-apart = same player in alternating)."""
        engine = StarterInferenceEngine()
        leg1 = LegStarterInfo(
            leg_number=1,
            starter_player_id="player_a",
            source="dartconnect",
            confidence=1.0,
            is_confirmed=True,
        )
        info = engine.infer_starter(
            leg_number=3,
            players=PLAYERS,
            alternating_starts=True,
            previous_starters=[leg1],
        )
        assert info.starter_player_id == "player_a"  # 2 legs apart = same

    def test_leg_4_inferred_as_player_b(self):
        """Leg 4 = same as leg 2 (odd offset from leg 1)."""
        engine = StarterInferenceEngine()
        leg1 = LegStarterInfo(
            leg_number=1,
            starter_player_id="player_a",
            source="dartconnect",
            confidence=1.0,
            is_confirmed=True,
        )
        info = engine.infer_starter(
            leg_number=4,
            players=PLAYERS,
            alternating_starts=True,
            previous_starters=[leg1],
        )
        assert info.starter_player_id == "player_b"

    def test_chain_of_inferred_starters(self):
        """Full match starter chain alternates correctly."""
        engine = StarterInferenceEngine()
        starters = engine.infer_match_starters(
            num_legs=7,
            players=PLAYERS,
            alternating_starts=True,
            confirmed_starters={1: ("player_a", "dartconnect", 1.0)},
        )
        assert len(starters) == 7
        expected_starters = [
            "player_a", "player_b", "player_a", "player_b",
            "player_a", "player_b", "player_a",
        ]
        for i, (info, expected) in enumerate(zip(starters, expected_starters)):
            assert info.starter_player_id == expected, (
                f"Leg {i+1}: expected {expected}, got {info.starter_player_id}"
            )


class TestFirstLegUnknownConfidence:
    """G: first leg with no data has 0.5 confidence."""

    def test_no_data_gives_unknown_source(self):
        """No anchor, no feed → 'unknown' source, 0.5 confidence."""
        engine = StarterInferenceEngine()
        info = engine.infer_starter(
            leg_number=1,
            players=PLAYERS,
            alternating_starts=True,
            previous_starters=None,
            feed_starter_id=None,
        )
        assert info.source == "unknown"
        assert info.confidence == 0.5
        assert not info.is_confirmed
        assert info.starter_player_id is None

    def test_no_alternating_gives_unknown(self):
        """Non-alternating format with no data → unknown."""
        engine = StarterInferenceEngine()
        info = engine.infer_starter(
            leg_number=5,
            players=PLAYERS,
            alternating_starts=False,  # not alternating
            feed_starter_id=None,
        )
        assert info.source == "unknown"
        assert info.confidence == 0.5


class TestMarginWideningFactorValues:
    """G: margin_widening_factor has correct values at key confidence levels."""

    def test_confidence_1_gives_factor_1(self):
        """Full confidence (1.0) → no widening (factor = 1.0)."""
        factor = StarterInferenceEngine.margin_widening_factor(1.0)
        assert factor == pytest.approx(1.0, abs=1e-10)

    def test_confidence_0_5_gives_factor_1_15(self):
        """Half confidence (0.5) → 15% widening (factor = 1.15)."""
        factor = StarterInferenceEngine.margin_widening_factor(0.5)
        assert factor == pytest.approx(1.15, abs=1e-10)

    def test_confidence_0_gives_factor_1_3(self):
        """Zero confidence (0.0) → 30% widening (factor = 1.30)."""
        factor = StarterInferenceEngine.margin_widening_factor(0.0)
        assert factor == pytest.approx(1.30, abs=1e-10)

    def test_confidence_0_75_gives_factor_1_075(self):
        """75% confidence → 7.5% widening."""
        factor = StarterInferenceEngine.margin_widening_factor(0.75)
        assert factor == pytest.approx(1.075, abs=1e-10)

    def test_factor_always_geq_1(self):
        """Margin widening factor is always >= 1.0."""
        for conf in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            factor = StarterInferenceEngine.margin_widening_factor(conf)
            assert factor >= 1.0, f"Factor {factor} < 1.0 for confidence={conf}"

    def test_factor_is_monotone_decreasing_in_confidence(self):
        """Higher confidence → lower widening factor (monotone decreasing)."""
        confidences = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        factors = [StarterInferenceEngine.margin_widening_factor(c) for c in confidences]
        for i in range(len(factors) - 1):
            assert factors[i] >= factors[i + 1], (
                f"Factor should decrease: f({confidences[i]})={factors[i]:.4f} "
                f">= f({confidences[i+1]})={factors[i+1]:.4f}"
            )

    def test_invalid_confidence_raises(self):
        """Confidence outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError):
            StarterInferenceEngine.margin_widening_factor(-0.1)
        with pytest.raises(ValueError):
            StarterInferenceEngine.margin_widening_factor(1.1)


class TestConfirmedStarterFromFeed:
    """G: confirmed starter from authoritative feed takes precedence."""

    def test_dartconnect_feed_overrides_inference(self):
        """DartConnect feed starter is always used regardless of inference."""
        engine = StarterInferenceEngine()
        # Even if inference would suggest player_b, feed says player_a
        info = engine.infer_starter(
            leg_number=2,
            players=PLAYERS,
            alternating_starts=True,
            feed_starter_id="player_a",  # feed says A
            feed_confidence=1.0,
            feed_source="dartconnect",
        )
        assert info.starter_player_id == "player_a"
        assert info.is_confirmed
        assert info.source == "dartconnect"

    def test_pdc_official_feed_is_confirmed(self):
        """PDC official feed marks starter as confirmed."""
        engine = StarterInferenceEngine()
        info = engine.infer_starter(
            leg_number=3,
            players=PLAYERS,
            alternating_starts=True,
            feed_starter_id="player_b",
            feed_confidence=0.98,
            feed_source="pdc_official",
        )
        assert info.is_confirmed
        assert info.confidence == pytest.approx(0.98)

    def test_inferred_source_not_confirmed(self):
        """Inferred starter is NOT marked as confirmed."""
        engine = StarterInferenceEngine()
        leg1 = LegStarterInfo(
            leg_number=1,
            starter_player_id="player_a",
            source="dartconnect",
            confidence=1.0,
            is_confirmed=True,
        )
        info = engine.infer_starter(
            leg_number=2,
            players=PLAYERS,
            alternating_starts=True,
            previous_starters=[leg1],
        )
        assert not info.is_confirmed
        assert info.source == "inferred"


class TestAggregateConfidence:
    """G: aggregate_match_confidence returns minimum confidence."""

    def test_all_confirmed_gives_1(self):
        """All confirmed starters → aggregate confidence = 1.0."""
        engine = StarterInferenceEngine()
        starters = [
            LegStarterInfo(1, "player_a", "dartconnect", 1.0, True),
            LegStarterInfo(2, "player_b", "dartconnect", 1.0, True),
            LegStarterInfo(3, "player_a", "dartconnect", 1.0, True),
        ]
        conf = engine.aggregate_match_confidence(starters)
        assert conf == 1.0

    def test_mixed_starters_gives_minimum(self):
        """Mixed confidence → minimum is returned."""
        engine = StarterInferenceEngine()
        starters = [
            LegStarterInfo(1, "player_a", "dartconnect", 1.0, True),
            LegStarterInfo(2, "player_b", "inferred", 0.95, False),
            LegStarterInfo(3, "player_a", "inferred", 0.90, False),
        ]
        conf = engine.aggregate_match_confidence(starters)
        assert conf == pytest.approx(0.90)

    def test_empty_list_gives_0_5(self):
        """Empty starter list → 0.5 (uniform prior)."""
        engine = StarterInferenceEngine()
        conf = engine.aggregate_match_confidence([])
        assert conf == 0.5
