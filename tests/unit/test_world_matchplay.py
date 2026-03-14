"""
Unit tests for the World Matchplay two-clear-legs engine.

Tests:
  - test_two_clear_legs_nominal_terminal
  - test_two_clear_legs_extra_legs_trigger
  - test_two_clear_legs_sudden_death
  - test_world_matchplay_prices_sum_to_one
  - test_world_matchplay_stronger_player_wins_more
  - test_two_clear_legs_symmetry
"""

from __future__ import annotations

import pytest

from competition.format_registry import DartsRoundFormat
from engines.leg_layer.markov_chain import HoldBreakProbabilities
from engines.match_layer.world_matchplay_engine import WorldMatchplayEngine


def make_hb(p1_hold: float, p2_hold: float) -> HoldBreakProbabilities:
    """Build HoldBreakProbabilities from hold rates."""
    return HoldBreakProbabilities(
        p1_hold=p1_hold,
        p1_break=1.0 - p2_hold,
        p2_hold=p2_hold,
        p2_break=1.0 - p1_hold,
    )


class TestTwoClearLegsNominalTerminal:
    """G: nominal target states terminate correctly under two-clear rule."""

    def test_18_1_is_terminal_p1_wins(self):
        """State (18, 1) under nominal=18, two-clear rule: P1 wins (17 ahead)."""
        engine = WorldMatchplayEngine()
        is_term, winner = engine._is_terminal(18, 1, nominal=18, max_extra=40)
        assert is_term
        assert winner == 1  # P1 wins

    def test_18_17_not_terminal_without_2_clear(self):
        """State (18, 17) at nominal=18: not terminal (only 1 clear leg)."""
        engine = WorldMatchplayEngine()
        is_term, winner = engine._is_terminal(18, 17, nominal=18, max_extra=40)
        assert not is_term
        assert winner == -1

    def test_19_17_is_terminal_p1_wins(self):
        """State (19, 17) at nominal=18: P1 leads by 2 and has >= nominal → P1 wins."""
        engine = WorldMatchplayEngine()
        is_term, winner = engine._is_terminal(19, 17, nominal=18, max_extra=40)
        assert is_term
        assert winner == 1

    def test_17_19_is_terminal_p2_wins(self):
        """State (17, 19) at nominal=18: P2 leads by 2 and has >= nominal → P2 wins."""
        engine = WorldMatchplayEngine()
        is_term, winner = engine._is_terminal(17, 19, nominal=18, max_extra=40)
        assert is_term
        assert winner == 2

    def test_before_nominal_not_terminal(self):
        """States below nominal are not terminal even if 2 clear."""
        engine = WorldMatchplayEngine()
        # P1 has 5 legs, P2 has 3 — leads by 2 but hasn't reached nominal=18
        is_term, winner = engine._is_terminal(5, 3, nominal=18, max_extra=40)
        assert not is_term


class TestTwoClearLegsExtraLegsHappens:
    """G: extra legs are triggered when scores are level at nominal."""

    def test_18_18_not_terminal(self):
        """Tied at 18-18 (nominal=18): not terminal, extra legs continue."""
        engine = WorldMatchplayEngine()
        is_term, winner = engine._is_terminal(18, 18, nominal=18, max_extra=40)
        assert not is_term  # continues until 2 clear

    def test_extra_legs_cap_triggers_sudden_death(self):
        """When max_extra legs are exhausted, sudden death applies."""
        engine = WorldMatchplayEngine()
        # nominal=10, max_extra=5: if we reach 15 legs (nominal + max_extra)
        # and P1 leads by 1, P1 wins
        is_term, winner = engine._is_terminal(16, 15, nominal=10, max_extra=5)
        # max(16, 15) = 16 >= 10 + 5 = 15 → sudden death, P1 leads → P1 wins
        assert is_term
        assert winner == 1

    def test_tied_at_sudden_death_cap_is_draw(self):
        """Exact tie when max_extra exhausted → winner=0 (coin flip)."""
        engine = WorldMatchplayEngine()
        # nominal=10, max_extra=5: at (15, 15) → sudden death tie
        is_term, winner = engine._is_terminal(15, 15, nominal=10, max_extra=5)
        assert is_term
        assert winner == 0  # coin flip


class TestWorldMatchplaySuddenDeath:
    """G: sudden death rules produce valid probabilities."""

    def test_sudden_death_at_equal_players_50_50(self):
        """Equal players under sudden death give 50/50."""
        engine = WorldMatchplayEngine()
        hb = make_hb(p1_hold=0.60, p2_hold=0.60)
        round_fmt = DartsRoundFormat(
            name="Final",
            legs_to_win=18,
            two_clear_legs=True,
            two_clear_legs_max_extra=None,
        )
        result = engine.price_match(round_fmt=round_fmt, hb=hb, p1_starts=True)
        total = result["p1_win"] + result["p2_win"]
        assert abs(total - 1.0) < 1e-6


class TestWorldMatchplayPricesSumToOne:
    """G: P1_win + P2_win = 1.0 for all World Matchplay configurations."""

    def test_basic_format_sums_to_one(self):
        """Basic World Matchplay format priced correctly."""
        engine = WorldMatchplayEngine()
        hb = make_hb(0.63, 0.60)
        round_fmt = DartsRoundFormat(
            name="Final",
            legs_to_win=18,
            two_clear_legs=True,
            two_clear_legs_max_extra=None,
        )
        result = engine.price_match(round_fmt=round_fmt, hb=hb, p1_starts=True)
        total = result["p1_win"] + result["p2_win"]
        assert abs(total - 1.0) < 1e-6, f"Total={total:.8f}"

    def test_various_formats_sum_to_one(self):
        """Multiple World Matchplay formats all sum to 1."""
        engine = WorldMatchplayEngine()
        hb = make_hb(0.65, 0.58)

        for nominal, max_extra in [(10, None), (13, None), (15, None), (18, None)]:
            round_fmt = DartsRoundFormat(
                name="Round",
                legs_to_win=nominal,
                two_clear_legs=True,
                two_clear_legs_max_extra=max_extra,
            )
            result = engine.price_match(round_fmt=round_fmt, hb=hb, p1_starts=True)
            total = result["p1_win"] + result["p2_win"]
            assert abs(total - 1.0) < 1e-6, (
                f"nominal={nominal}: total={total:.8f}"
            )

    def test_two_clear_legs_stronger_player_wins_more(self):
        """Stronger player (higher hold) wins more often with two-clear-legs rule."""
        engine = WorldMatchplayEngine()
        hb_strong = make_hb(0.72, 0.58)
        hb_weak = make_hb(0.55, 0.58)

        round_fmt = DartsRoundFormat(
            name="Final",
            legs_to_win=18,
            two_clear_legs=True,
            two_clear_legs_max_extra=None,
        )

        result_strong = engine.price_match(round_fmt=round_fmt, hb=hb_strong, p1_starts=True)
        result_weak = engine.price_match(round_fmt=round_fmt, hb=hb_weak, p1_starts=True)

        assert result_strong["p1_win"] > result_weak["p1_win"]


class TestWorldMatchplaySymmetry:
    """G2: P1/P2 symmetry for two-clear-legs engine."""

    def test_swap_players_complementary_probs(self):
        """Swapping players gives complementary probabilities."""
        engine = WorldMatchplayEngine()
        hb = make_hb(p1_hold=0.65, p2_hold=0.60)
        hb_swapped = make_hb(p1_hold=0.60, p2_hold=0.65)

        round_fmt = DartsRoundFormat(
            name="Final",
            legs_to_win=18,
            two_clear_legs=True,
            two_clear_legs_max_extra=None,
        )

        result = engine.price_match(round_fmt=round_fmt, hb=hb, p1_starts=True)
        result_swapped = engine.price_match(round_fmt=round_fmt, hb=hb_swapped, p1_starts=True)

        # After swapping, P1 and P2 probabilities should swap
        assert abs(result["p1_win"] - result_swapped["p2_win"]) < 1e-6, (
            f"Symmetry violation: p1={result['p1_win']:.6f}, swapped_p2={result_swapped['p2_win']:.6f}"
        )
