"""
Regression tests for Darts Tier 2B Reverse-Engineering.
LOCK-DARTS-TIER-2B-REVERSE-ENGINEER-001

Test contract:
  For pinned Pinnacle fixtures, reverse_engineer_darts() must:
  1. converge (converged=True)
  2. reproduce the Pinnacle de-vigged prob within 0.5pp
  3. three_da_a in [25, 110], three_da_b in [25, 110]
  4. solve in <= 30 iterations
  5. prediction_source == "market_scrape_reverse_engineered"

Pinned fixtures (PDC 2025-2026 season data):
  F1: van Gerwen vs Price (PDC WC) — Pinnacle p_a=0.68
  F2: Smith vs Wright (PDC PL) — Pinnacle p_a=0.52
  F3: Unranked qualifier (WDF) — Pinnacle p_a=0.60
  F4: Near-even top-16 match — Pinnacle p_a=0.50
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from app.pricing.tier2b_reverse_engineer import (
    reverse_engineer_darts,
    _match_prob_darts,
    _leg_win_prob,
    _p_visit_finish,
)


class TestDartsTier2BRoundTrip:
    """Pinned fixture tests — must reproduce Pinnacle prob within 0.5pp."""

    FIXTURES = [
        # (elo_a, elo_b, p_a_pinnacle, ecosystem, legs_to_win, label)
        (1750.0, 1580.0, 0.68, "pdc_mens", 7, "F1_mvg_vs_price_wc"),
        (1620.0, 1600.0, 0.52, "pdc_mens", 7, "F2_smith_vs_wright_pl"),
        (1100.0, 900.0,  0.60, "wdf_open", 5, "F3_wdf_qualifier"),
        (1500.0, 1500.0, 0.50, "pdc_mens", 7, "F4_near_even"),
        (1900.0, 1200.0, 0.82, "pdc_mens", 7, "F5_heavy_favourite"),
        (1350.0, 1650.0, 0.35, "pdc_mens", 7, "F6_heavy_underdog"),
    ]

    @pytest.mark.parametrize("elo_a,elo_b,p_a,ecosystem,ltw,label", FIXTURES)
    def test_round_trip_parity(self, elo_a, elo_b, p_a, ecosystem, ltw, label):
        result = reverse_engineer_darts(
            pinnacle_a_prob=p_a,
            elo_a=elo_a,
            elo_b=elo_b,
            ecosystem=ecosystem,
            legs_to_win=ltw,
            fixture_id=label,
        )
        assert result.converged, (
            f"{label}: did not converge. residual={result.solve_residual:.6f}"
        )
        assert result.solve_residual <= 0.005, (
            f"{label}: residual {result.solve_residual:.6f} > 0.5pp"
        )

    @pytest.mark.parametrize("elo_a,elo_b,p_a,ecosystem,ltw,label", FIXTURES)
    def test_3da_bounds(self, elo_a, elo_b, p_a, ecosystem, ltw, label):
        result = reverse_engineer_darts(
            pinnacle_a_prob=p_a,
            elo_a=elo_a,
            elo_b=elo_b,
            ecosystem=ecosystem,
            legs_to_win=ltw,
            fixture_id=label,
        )
        assert 25.0 < result.three_da_a < 125.0, (
            f"{label}: three_da_a={result.three_da_a} out of [25,125]"
        )
        if result.converged:
            assert 25.0 < result.three_da_b < 125.0, (
                f"{label}: three_da_b={result.three_da_b} out of [25,125]"
            )

    @pytest.mark.parametrize("elo_a,elo_b,p_a,ecosystem,ltw,label", FIXTURES)
    def test_prediction_source(self, elo_a, elo_b, p_a, ecosystem, ltw, label):
        result = reverse_engineer_darts(
            pinnacle_a_prob=p_a,
            elo_a=elo_a,
            elo_b=elo_b,
            ecosystem=ecosystem,
            legs_to_win=ltw,
            fixture_id=label,
        )
        if result.converged:
            assert result.prediction_source == "market_scrape_reverse_engineered"

    @pytest.mark.parametrize("elo_a,elo_b,p_a,ecosystem,ltw,label", FIXTURES)
    def test_max_iterations(self, elo_a, elo_b, p_a, ecosystem, ltw, label):
        result = reverse_engineer_darts(
            pinnacle_a_prob=p_a,
            elo_a=elo_a,
            elo_b=elo_b,
            ecosystem=ecosystem,
            legs_to_win=ltw,
            fixture_id=label,
        )
        assert result.solve_iterations <= 30

    @pytest.mark.parametrize("elo_a,elo_b,p_a,ecosystem,ltw,label", FIXTURES)
    def test_probabilities_sum_to_one(self, elo_a, elo_b, p_a, ecosystem, ltw, label):
        result = reverse_engineer_darts(
            pinnacle_a_prob=p_a,
            elo_a=elo_a,
            elo_b=elo_b,
            ecosystem=ecosystem,
            legs_to_win=ltw,
            fixture_id=label,
        )
        total = result.p_match_a + result.p_match_b
        assert abs(total - 1.0) < 1e-6, f"{label}: {total} != 1.0"


class TestDartsForwardModel:
    """Unit tests for the leg/match forward model."""

    def test_leg_win_prob_symmetric(self):
        """Equal 3DAs -> P(A wins leg) should be ~0.5."""
        p = _leg_win_prob(80.0, 80.0, a_starts=True)
        assert abs(p - 0.5) < 0.02  # small asymmetry from starting advantage

    def test_leg_win_prob_dominant_player(self):
        """Higher 3DA player should win more legs."""
        p_high = _leg_win_prob(100.0, 70.0, a_starts=True)
        p_low = _leg_win_prob(70.0, 100.0, a_starts=True)
        assert p_high > 0.5
        assert p_low < 0.5

    def test_match_prob_symmetric(self):
        """Equal 3DAs -> match prob ~ 0.5."""
        p = _match_prob_darts(80.0, 80.0, legs_to_win=7)
        assert abs(p - 0.5) < 0.02

    def test_match_prob_monotone_in_3da_b(self):
        """Higher 3DA_b -> lower P(A wins match)."""
        p_low_b = _match_prob_darts(90.0, 70.0, legs_to_win=7)
        p_high_b = _match_prob_darts(90.0, 95.0, legs_to_win=7)
        assert p_low_b > p_high_b

    def test_p_visit_finish_range(self):
        """_p_visit_finish must be in (0, 1) for all valid 3DAs."""
        for da in [30.0, 60.0, 90.0, 110.0]:
            p = _p_visit_finish(da)
            assert 0.0 < p < 1.0

    def test_invalid_prob_raises(self):
        with pytest.raises(ValueError):
            reverse_engineer_darts(pinnacle_a_prob=0.0, elo_a=1500.0, elo_b=1500.0)
        with pytest.raises(ValueError):
            reverse_engineer_darts(pinnacle_a_prob=1.5, elo_a=1500.0, elo_b=1500.0)
