"""
Unit tests for the hold/break model.

Tests:
  - test_hold_gt_break_for_same_player
  - test_hold_break_consistency
  - test_hold_break_from_3da
  - test_hold_break_monotone_in_3da
  - test_hold_break_symmetry_equal_players
  - test_hold_break_validates_probabilities
"""

from __future__ import annotations

import pytest

from engines.leg_layer.hold_break_model import HoldBreakModel, PlayerMatchupProfile
from engines.leg_layer.markov_chain import HoldBreakProbabilities


class TestHoldGtBreakForSamePlayer:
    """G: For the same player, hold > break (harder to break from behind)."""

    def test_hold_gt_break_equal_players(self):
        """Even against equal opponent, hold > break (server advantage)."""
        model = HoldBreakModel()
        hb = model.compute_from_3da(
            p1_id="p1",
            p2_id="p2",
            p1_three_da=75.0,
            p2_three_da=75.0,
        )
        # With equal players, each player should hold more than break
        assert hb.p1_hold > hb.p1_break, (
            f"p1_hold={hb.p1_hold:.4f} should be > p1_break={hb.p1_break:.4f}"
        )
        assert hb.p2_hold > hb.p2_break, (
            f"p2_hold={hb.p2_hold:.4f} should be > p2_break={hb.p2_break:.4f}"
        )

    def test_hold_gt_break_stronger_player(self):
        """Even stronger player should have hold > break."""
        model = HoldBreakModel()
        hb = model.compute_from_3da(
            p1_id="strong",
            p2_id="weak",
            p1_three_da=95.0,
            p2_three_da=60.0,
        )
        # P1 is stronger: holds more than breaks (service advantage still holds)
        assert hb.p1_hold > hb.p1_break

    def test_hold_break_range(self):
        """All hold/break probabilities are in [0, 1]."""
        model = HoldBreakModel()
        for p1_3da, p2_3da in [(65.0, 65.0), (80.0, 70.0), (100.0, 55.0)]:
            hb = model.compute_from_3da(
                p1_id=f"p1_{p1_3da}",
                p2_id=f"p2_{p2_3da}",
                p1_three_da=p1_3da,
                p2_three_da=p2_3da,
            )
            for name, val in [
                ("p1_hold", hb.p1_hold), ("p1_break", hb.p1_break),
                ("p2_hold", hb.p2_hold), ("p2_break", hb.p2_break),
            ]:
                assert 0.0 <= val <= 1.0, f"{name}={val} out of range for 3DA={p1_3da},{p2_3da}"


class TestHoldBreakConsistency:
    """G3: p1_hold + p2_break = 1.0; p2_hold + p1_break = 1.0."""

    def test_consistency_for_various_3das(self):
        """Consistency constraint holds for a range of 3DA values."""
        model = HoldBreakModel()
        for p1_3da, p2_3da in [
            (60.0, 60.0),
            (75.0, 65.0),
            (90.0, 70.0),
            (95.0, 55.0),
        ]:
            hb = model.compute_from_3da(
                p1_id=f"p1",
                p2_id=f"p2",
                p1_three_da=p1_3da,
                p2_three_da=p2_3da,
            )
            assert hb.consistency_check(), (
                f"G3 violation for 3DA=({p1_3da}, {p2_3da}): "
                f"p1_hold={hb.p1_hold:.6f}, p2_break={hb.p2_break:.6f}, "
                f"p2_hold={hb.p2_hold:.6f}, p1_break={hb.p1_break:.6f}"
            )

    def test_g3_constraint_manually(self):
        """Manual G3 check: p1_hold + p2_break = 1."""
        hb = HoldBreakProbabilities(
            p1_hold=0.63,
            p1_break=0.42,
            p2_hold=0.58,
            p2_break=0.37,
        )
        assert abs(hb.p1_hold + hb.p2_break - 1.0) < 1e-6
        assert abs(hb.p2_hold + hb.p1_break - 1.0) < 1e-6


class TestHoldBreakFrom3DA:
    """G: hold/break computed from 3DA produces meaningful results."""

    def test_higher_3da_gives_higher_hold(self):
        """Player with higher 3DA has higher hold probability."""
        model = HoldBreakModel()

        hb_low = model.compute_from_3da(
            p1_id="low", p2_id="avg",
            p1_three_da=55.0, p2_three_da=70.0,
        )
        hb_high = model.compute_from_3da(
            p1_id="high", p2_id="avg",
            p1_three_da=90.0, p2_three_da=70.0,
        )

        assert hb_high.p1_hold > hb_low.p1_hold, (
            f"High 3DA player should have higher hold: "
            f"high={hb_high.p1_hold:.4f}, low={hb_low.p1_hold:.4f}"
        )

    def test_hold_break_from_profile(self):
        """PlayerMatchupProfile interface produces valid results."""
        model = HoldBreakModel()
        p1 = PlayerMatchupProfile(
            player_id="player_a",
            three_da=82.5,
            stage=True,
            short_format=False,
            throw_first=True,
        )
        p2 = PlayerMatchupProfile(
            player_id="player_b",
            three_da=78.0,
            stage=True,
            short_format=False,
            throw_first=False,
        )
        hb = model.compute(p1, p2)
        hb.validate()  # should not raise

    def test_symmetric_3da_near_symmetric_hold_break(self):
        """Players with same 3DA should have nearly symmetric hold/break."""
        model = HoldBreakModel()
        hb = model.compute_from_3da(
            p1_id="equal_a",
            p2_id="equal_b",
            p1_three_da=75.0,
            p2_three_da=75.0,
        )
        # p1_hold should ≈ p2_hold (symmetric players)
        assert abs(hb.p1_hold - hb.p2_hold) < 0.05, (
            f"Equal players should have similar hold rates: "
            f"p1={hb.p1_hold:.4f}, p2={hb.p2_hold:.4f}"
        )

    def test_hold_probability_between_0_5_and_1(self):
        """Hold probability should be > 0.5 (server advantage in darts)."""
        model = HoldBreakModel()
        for p1_3da, p2_3da in [(60.0, 60.0), (75.0, 70.0), (90.0, 85.0)]:
            hb = model.compute_from_3da(
                p1_id="p1", p2_id="p2",
                p1_three_da=p1_3da, p2_three_da=p2_3da,
            )
            assert hb.p1_hold > 0.50, (
                f"p1_hold={hb.p1_hold:.4f} should be > 0.5 (server advantage), 3DA={p1_3da}"
            )
            assert hb.p2_hold > 0.50, (
                f"p2_hold={hb.p2_hold:.4f} should be > 0.5 (server advantage), 3DA={p2_3da}"
            )


class TestHoldBreakToMatchWin:
    """G: HoldBreakProbabilities.to_match_win_prob integrates correctly."""

    def test_stronger_player_wins_more(self):
        """Stronger player has higher match win probability."""
        hb_strong = HoldBreakProbabilities(
            p1_hold=0.72, p1_break=0.42, p2_hold=0.58, p2_break=0.28
        )
        hb_weak = HoldBreakProbabilities(
            p1_hold=0.56, p1_break=0.44, p2_hold=0.56, p2_break=0.44
        )

        p1_strong = hb_strong.to_match_win_prob(p1_starts_first=True, legs_to_win=5)
        p1_weak = hb_weak.to_match_win_prob(p1_starts_first=True, legs_to_win=5)

        assert p1_strong > p1_weak

    def test_match_win_prob_in_range(self):
        """Match win probability is in [0, 1]."""
        hb = HoldBreakProbabilities(
            p1_hold=0.65, p1_break=0.42, p2_hold=0.58, p2_break=0.35
        )
        for ltw in [2, 3, 5, 7]:
            p = hb.to_match_win_prob(p1_starts_first=True, legs_to_win=ltw)
            assert 0.0 <= p <= 1.0, f"Match win prob {p} out of range for legs_to_win={ltw}"
