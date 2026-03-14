"""
Unit tests for the leg layer.

Tests:
  - test_hold_break_consistency_check
  - test_conditional_visit_distribution_validates
  - test_visit_distribution_sums_to_one
  - test_hierarchical_pooling_thin_data
  - test_markov_chain_produces_valid_probabilities
  - test_hold_probability_between_0_and_1
"""

from __future__ import annotations

import numpy as np
import pytest

from engines.leg_layer.markov_chain import (
    DartsMarkovChain,
    HoldBreakProbabilities,
    MARKOV_TOTAL_TOLERANCE,
)
from engines.leg_layer.visit_distributions import (
    BAND_NAMES,
    ConditionalVisitDistribution,
    HierarchicalVisitDistributionModel,
    score_to_band,
)
from engines.state_layer.score_state import DartsEngineError


# ---------------------------------------------------------------------------
# HoldBreakProbabilities tests
# ---------------------------------------------------------------------------


class TestHoldBreakConsistencyCheck:
    """G3: p1_hold + p2_break = 1.0; p2_hold + p1_break = 1.0."""

    def test_consistent_probabilities_pass(self):
        """Valid hold/break probabilities pass consistency check."""
        hb = HoldBreakProbabilities(
            p1_hold=0.65,
            p1_break=0.40,
            p2_hold=0.60,
            p2_break=0.35,
        )
        assert hb.consistency_check()

    def test_inconsistent_probabilities_fail(self):
        """Invalid hold/break probabilities fail consistency check."""
        hb = HoldBreakProbabilities(
            p1_hold=0.65,
            p1_break=0.40,
            p2_hold=0.60,
            p2_break=0.45,  # should be 0.35 for consistency
        )
        assert not hb.consistency_check()

    def test_exact_boundary_passes(self):
        """Exact sum = 1.0 passes within tolerance."""
        hb = HoldBreakProbabilities(
            p1_hold=0.600000,
            p1_break=0.400000,
            p2_hold=0.600000,
            p2_break=0.400000,
        )
        assert hb.consistency_check()

    def test_symmetric_match_is_consistent(self):
        """Equal players: hold = 0.5+ε, break = 0.5-ε should be consistent."""
        hb = HoldBreakProbabilities(
            p1_hold=0.55,
            p1_break=0.45,
            p2_hold=0.55,
            p2_break=0.45,
        )
        assert hb.consistency_check()

    def test_validate_raises_on_probability_out_of_range(self):
        """validate() raises ValueError when probabilities not in [0, 1]."""
        hb = HoldBreakProbabilities(
            p1_hold=1.05,  # > 1.0
            p1_break=0.40,
            p2_hold=0.60,
            p2_break=-0.05,  # < 0.0
        )
        with pytest.raises(ValueError):
            hb.validate()

    def test_validate_raises_on_consistency_violation(self):
        """validate() raises ValueError on consistency violation."""
        hb = HoldBreakProbabilities(
            p1_hold=0.60,
            p1_break=0.50,
            p2_hold=0.50,
            p2_break=0.50,  # 0.60 + 0.50 != 1.0
        )
        with pytest.raises(ValueError):
            hb.validate()


# ---------------------------------------------------------------------------
# ConditionalVisitDistribution tests
# ---------------------------------------------------------------------------


class TestConditionalVisitDistributionValidates:
    """G: ConditionalVisitDistribution validates correctly."""

    def _make_dist(
        self,
        visit_probs: dict[int, float] | None = None,
        bust_prob: float = 0.05,
        score_band: str = "open",
    ) -> ConditionalVisitDistribution:
        if visit_probs is None:
            visit_probs = {60: 0.50, 100: 0.30, 140: 0.15}
        return ConditionalVisitDistribution(
            player_id="test_player",
            score_band=score_band,
            stage=True,
            short_format=False,
            throw_first=True,
            visit_probs=visit_probs,
            bust_prob=bust_prob,
            data_source="derived",
            n_observations=100,
            confidence=0.8,
        )

    def test_valid_distribution_passes(self):
        """A correctly normalised distribution passes validation."""
        dist = self._make_dist(
            visit_probs={60: 0.475, 100: 0.285, 140: 0.190},
            bust_prob=0.05,
        )
        dist.validate()  # should not raise

    def test_distribution_not_summing_to_one_raises(self):
        """Distribution that does not sum to 1.0 raises DartsEngineError."""
        dist = self._make_dist(
            visit_probs={60: 0.50, 100: 0.30},  # sum = 0.80, + bust=0.05 = 0.85 ≠ 1.0
            bust_prob=0.05,
        )
        with pytest.raises(DartsEngineError, match="does not sum to 1"):
            dist.validate()

    def test_negative_probability_raises(self):
        """Negative probability raises DartsEngineError."""
        dist = self._make_dist(
            visit_probs={60: -0.10, 100: 1.05},
            bust_prob=0.05,
        )
        with pytest.raises(DartsEngineError):
            dist.validate()

    def test_bust_prob_out_of_range_raises(self):
        """bust_prob outside [0, 1] raises DartsEngineError."""
        dist = self._make_dist(
            visit_probs={60: 0.70},
            bust_prob=-0.10,  # negative
        )
        with pytest.raises(DartsEngineError):
            dist.validate()


class TestVisitDistributionSumsToOne:
    """G1 invariant: all visit distributions must sum to 1.0 ± tolerance."""

    def test_hierarchical_model_produces_valid_distributions(self):
        """All distributions from HierarchicalVisitDistributionModel sum to 1.0."""
        model = HierarchicalVisitDistributionModel()
        for three_da in [50.0, 70.0, 90.0, 100.0]:
            for band in BAND_NAMES:
                dist = model.get_distribution(
                    player_id=f"player_{three_da}",
                    score_band=band,
                    stage=True,
                    short_format=False,
                    throw_first=True,
                    three_da=three_da,
                )
                total = sum(dist.visit_probs.values()) + dist.bust_prob
                assert abs(total - 1.0) <= MARKOV_TOTAL_TOLERANCE, (
                    f"Distribution for 3DA={three_da}, band={band} sums to {total:.8f}"
                )

    def test_derived_distribution_all_bands(self):
        """Derived distributions for all bands sum to 1.0."""
        model = HierarchicalVisitDistributionModel()
        for band in BAND_NAMES:
            dist = model._derive_from_3da(
                three_da=75.0,
                score_band=band,
                player_id="test",
                stage=False,
                short_format=False,
                throw_first=True,
            )
            total = sum(dist.visit_probs.values()) + dist.bust_prob
            assert abs(total - 1.0) <= 1e-6, (
                f"Derived dist for band={band} sums to {total:.8f}"
            )

    def test_blended_distribution_sums_to_one(self):
        """Blended (partial pooling) distribution sums to 1.0."""
        model = HierarchicalVisitDistributionModel()
        prior = model._derive_from_3da(
            three_da=70.0,
            score_band="open",
            player_id="test",
            stage=False,
            short_format=False,
            throw_first=True,
        )
        # Fake player-specific distribution
        player_dist = ConditionalVisitDistribution(
            player_id="test",
            score_band="open",
            stage=False,
            short_format=False,
            throw_first=True,
            visit_probs={60: 0.50, 100: 0.30, 140: 0.16},
            bust_prob=0.04,
            data_source="dartconnect",
            n_observations=15,  # below min_obs
            confidence=0.8,
        )
        blended = model._blend(
            player_dist=player_dist,
            prior_dist=prior,
            min_obs=30,
        )
        total = sum(blended.visit_probs.values()) + blended.bust_prob
        assert abs(total - 1.0) <= 1e-6


class TestHierarchicalPoolingThinData:
    """G: hierarchical pooling uses prior for thin-data players."""

    def test_thin_data_player_uses_derived_prior(self):
        """Player with no data uses derived-from-3DA distribution."""
        model = HierarchicalVisitDistributionModel()
        dist = model.get_distribution(
            player_id="unknown_player_xyz",
            score_band="open",
            stage=False,
            short_format=False,
            throw_first=True,
            three_da=72.0,
        )
        assert dist.data_source in ("derived", "pooled")
        assert dist.n_observations == 0
        # Distribution should be valid
        dist.validate()

    def test_expected_score_increases_with_3da(self):
        """Higher 3DA → higher expected visit score in open band."""
        model = HierarchicalVisitDistributionModel()
        dist_low = model.get_distribution(
            player_id="low_player", score_band="open",
            stage=False, short_format=False, throw_first=True, three_da=55.0
        )
        dist_high = model.get_distribution(
            player_id="high_player", score_band="open",
            stage=False, short_format=False, throw_first=True, three_da=95.0
        )
        assert dist_high.expected_score() > dist_low.expected_score(), (
            f"High 3DA={95.0} expected score {dist_high.expected_score():.1f} "
            f"should exceed low 3DA={55.0} expected score {dist_low.expected_score():.1f}"
        )

    def test_bust_prob_increases_in_pressure_band(self):
        """Pressure band has higher bust probability than open band."""
        model = HierarchicalVisitDistributionModel()
        dist_open = model.get_distribution(
            player_id="player_a", score_band="open",
            stage=False, short_format=False, throw_first=True, three_da=75.0
        )
        dist_pressure = model.get_distribution(
            player_id="player_a", score_band="pressure",
            stage=False, short_format=False, throw_first=True, three_da=75.0
        )
        assert dist_pressure.bust_prob > dist_open.bust_prob, (
            f"Pressure bust={dist_pressure.bust_prob:.4f} should exceed open bust={dist_open.bust_prob:.4f}"
        )


# ---------------------------------------------------------------------------
# Markov chain tests
# ---------------------------------------------------------------------------


class TestMarkovChainValidation:
    """G1: Markov chain transitions sum to 1.0."""

    def test_validate_markov_totals_passes_for_valid_dist(self):
        """validate_markov_totals returns True for a valid distribution."""
        markov = DartsMarkovChain()
        dist = ConditionalVisitDistribution(
            player_id="test",
            score_band="open",
            stage=False,
            short_format=False,
            throw_first=True,
            visit_probs={60: 0.70, 100: 0.25},
            bust_prob=0.05,
            data_source="derived",
            n_observations=0,
            confidence=0.5,
        )
        assert markov.validate_markov_totals(dist)

    def test_validate_markov_totals_fails_for_invalid_dist(self):
        """validate_markov_totals returns False for distribution not summing to 1."""
        markov = DartsMarkovChain()
        dist = ConditionalVisitDistribution(
            player_id="test",
            score_band="open",
            stage=False,
            short_format=False,
            throw_first=True,
            visit_probs={60: 0.50},  # sum = 0.50 + bust=0.05 = 0.55 ≠ 1.0
            bust_prob=0.05,
            data_source="derived",
            n_observations=0,
            confidence=0.5,
        )
        assert not markov.validate_markov_totals(dist, tol=MARKOV_TOTAL_TOLERANCE)


class TestScoreToBand:
    """G: score_to_band maps correctly."""

    def test_all_scores_have_valid_band(self):
        """Every score from 2 to 501 maps to a valid band."""
        for s in range(2, 502):
            band = score_to_band(s)
            assert band in BAND_NAMES, f"Score {s} maps to invalid band {band!r}"

    def test_band_boundaries(self):
        """Test key band boundary scores."""
        assert score_to_band(501) == "open"
        assert score_to_band(302) == "open"
        assert score_to_band(301) == "open"
        assert score_to_band(300) == "middle"
        assert score_to_band(171) == "middle"
        assert score_to_band(170) == "setup"
        assert score_to_band(100) == "setup"
        assert score_to_band(99) == "finish"
        assert score_to_band(41) == "finish"
        assert score_to_band(40) == "pressure"
        assert score_to_band(2) == "pressure"
