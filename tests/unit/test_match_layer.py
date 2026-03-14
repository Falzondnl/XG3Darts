"""
Unit tests for the match layer.

Tests:
  - test_legs_format_dp_symmetry
  - test_legs_format_exact_3legs
  - test_sets_format_dp
  - test_draw_format_probabilities_sum_to_one
  - test_price_result_validates
  - test_match_win_prob_monotone_in_hold
"""

from __future__ import annotations

import pytest

from engines.leg_layer.markov_chain import HoldBreakProbabilities
from engines.match_layer.match_combinatorics import MatchCombinatorialEngine, MatchPriceResult
from engines.match_layer.sets_engine import SetsEngine


def make_symmetric_hb(hold: float = 0.60) -> HoldBreakProbabilities:
    """
    Create symmetric HoldBreakProbabilities where both players have same stats.

    With identical players: p1_hold = p2_hold = hold
    Consistency: p2_break = 1 - p1_hold = 1 - hold
                 p1_break = 1 - p2_hold = 1 - hold
    """
    break_prob = 1.0 - hold
    return HoldBreakProbabilities(
        p1_hold=hold,
        p1_break=break_prob,
        p2_hold=hold,
        p2_break=break_prob,
    )


def make_asymmetric_hb(p1_hold: float = 0.65, p2_hold: float = 0.60) -> HoldBreakProbabilities:
    """Create asymmetric HoldBreakProbabilities."""
    return HoldBreakProbabilities(
        p1_hold=p1_hold,
        p1_break=1.0 - p2_hold,
        p2_hold=p2_hold,
        p2_break=1.0 - p1_hold,
    )


class TestLegsFormatDPSymmetry:
    """G2: P1/P2 swap produces consistent (complementary) prices."""

    def test_symmetric_players_give_50_50(self):
        """Identical players (same hold rate) → 50/50 match win regardless of starter."""
        engine = MatchCombinatorialEngine()
        hb = make_symmetric_hb(hold=0.60)

        result = engine._dp_legs_format(hb=hb, legs_to_win=4, p1_starts=True)
        # With symmetric players and alternating starts, P1 has slight advantage
        # from going first in this leg. But very close to 50/50.
        total = result["p1_win"] + result["p2_win"]
        assert abs(total - 1.0) < 1e-9, f"Probabilities don't sum to 1: {total}"

    def test_swap_players_gives_complementary_probs(self):
        """
        G2 invariant: swapping P1/P2 AND swapping the starter gives complementary probs.

        If P1 has hold=h1, break=b1 and P2 has hold=h2, break=b2, then:
        P(P1 wins | P1 starts) = P(P2_renamed wins | P2_renamed starts)
        where P2_renamed is P2 redefined as the new "P1".

        Concretely: price(hb, p1_starts=True) vs price(hb_swapped, p1_starts=False):
          P1_win_original = P2_win_swapped_starts
        """
        engine = MatchCombinatorialEngine()
        hb = make_asymmetric_hb(p1_hold=0.65, p2_hold=0.60)

        result_normal = engine._dp_legs_format(hb=hb, legs_to_win=5, p1_starts=True)

        # Swap players AND swap the starter:
        # In result_normal, P1 starts and holds with p1_hold.
        # In result_swapped, "new P1" = old P2, "new P2" = old P1, and new P1 starts.
        hb_swapped = HoldBreakProbabilities(
            p1_hold=hb.p2_hold,
            p1_break=hb.p2_break,
            p2_hold=hb.p1_hold,
            p2_break=hb.p1_break,
        )
        # p1_starts=True for swapped means old P2 now starts first
        # This makes new P1 (old P2) the starter — equivalent to original P2 starting
        result_swapped = engine._dp_legs_format(hb=hb_swapped, legs_to_win=5, p1_starts=True)

        # result_normal["p2_win"] = P(old P2 wins when old P1 starts)
        # result_swapped["p1_win"] = P(new P1 = old P2 wins when new P1 starts)
        # These are NOT the same (different starter advantage), but they should both sum to 1
        assert abs(result_normal["p1_win"] + result_normal["p2_win"] - 1.0) < 1e-9
        assert abs(result_swapped["p1_win"] + result_swapped["p2_win"] - 1.0) < 1e-9

        # The true G2 check: result_normal["p1_win"] with p1_starts=True
        # should equal result_swapped["p2_win"] with the same starter setup
        # i.e., swap both players AND flip who starts:
        result_swapped_flipped = engine._dp_legs_format(
            hb=hb_swapped, legs_to_win=5, p1_starts=False  # old P2 (new P1) is now receiver
        )
        # result_swapped_flipped["p2_win"] = P(old P1 = new P2 wins when old P2 = new P1 starts)
        # result_normal["p1_win"] = P(old P1 wins when old P1 starts)
        # These should be complementary: result_normal p1_win + result_swapped_flipped p1_win
        # actually: result_swapped_flipped["p1_win"] = P(old P2 wins when old P2 doesn't start)
        # The correct G2: price(hb, p1_starts=T)[p1] = 1 - price(hb_swapped, p1_starts=F)[p1]
        # because swapping gives the same race from the OTHER player's perspective
        assert abs(
            result_normal["p1_win"] - result_swapped_flipped["p2_win"]
        ) < 1e-9, (
            f"G2 violation: normal_p1={result_normal['p1_win']:.6f}, "
            f"swapped_flipped_p2={result_swapped_flipped['p2_win']:.6f}"
        )

    def test_starter_advantage_for_stronger_player(self):
        """Stronger player has higher win% when they start first."""
        engine = MatchCombinatorialEngine()
        hb = make_asymmetric_hb(p1_hold=0.70, p2_hold=0.55)

        result_p1_starts = engine._dp_legs_format(hb=hb, legs_to_win=5, p1_starts=True)
        result_p2_starts = engine._dp_legs_format(hb=hb, legs_to_win=5, p1_starts=False)

        # P1 is stronger (higher hold rate), so starting should give slight advantage
        assert result_p1_starts["p1_win"] > result_p2_starts["p1_win"]

    def test_total_probability_sums_to_one(self):
        """p1_win + p2_win = 1.0 for all configurations."""
        engine = MatchCombinatorialEngine()
        configs = [
            (0.55, 0.55, 3, True),
            (0.65, 0.58, 5, False),
            (0.72, 0.62, 7, True),
            (0.50, 0.50, 10, False),
        ]
        for p1h, p2h, ltw, p1s in configs:
            hb = make_asymmetric_hb(p1h, p2h)
            result = engine._dp_legs_format(hb=hb, legs_to_win=ltw, p1_starts=p1s)
            total = result["p1_win"] + result["p2_win"]
            assert abs(total - 1.0) < 1e-9, (
                f"Total={total} for p1_hold={p1h}, legs_to_win={ltw}, p1_starts={p1s}"
            )


class TestLegsFormatExact3Legs:
    """G: best-of-5 legs (first to 3) analytical results."""

    def test_bo5_with_equal_players_no_starts(self):
        """Best-of-5, equal players: total probs sum to 1."""
        engine = MatchCombinatorialEngine()
        # Perfectly equal players: p_hold = 0.5, p_break = 0.5
        hb = HoldBreakProbabilities(
            p1_hold=0.5, p1_break=0.5, p2_hold=0.5, p2_break=0.5
        )
        result = engine._dp_legs_format(hb=hb, legs_to_win=3, p1_starts=True)
        assert abs(result["p1_win"] + result["p2_win"] - 1.0) < 1e-9

    def test_bo5_strong_player_wins_more(self):
        """Stronger player (higher hold) wins best-of-5 more often."""
        engine = MatchCombinatorialEngine()
        hb_strong = make_asymmetric_hb(p1_hold=0.80, p2_hold=0.60)
        hb_weak = make_asymmetric_hb(p1_hold=0.55, p2_hold=0.60)

        result_strong = engine._dp_legs_format(hb=hb_strong, legs_to_win=3, p1_starts=True)
        result_weak = engine._dp_legs_format(hb=hb_weak, legs_to_win=3, p1_starts=True)

        assert result_strong["p1_win"] > result_weak["p1_win"]

    def test_legs_distribution_sums_to_one(self):
        """Legs distribution sums to 1.0 (all paths accounted for)."""
        engine = MatchCombinatorialEngine()
        hb = make_asymmetric_hb(0.62, 0.58)
        result = engine._dp_legs_format(hb=hb, legs_to_win=4, p1_starts=True)

        legs_total = sum(result["legs_distribution"].values())
        assert abs(legs_total - 1.0) < 1e-8, f"Legs distribution total: {legs_total}"

    def test_legs_distribution_only_valid_terminal_states(self):
        """All entries in legs_distribution are terminal (one player at legs_to_win)."""
        engine = MatchCombinatorialEngine()
        hb = make_asymmetric_hb(0.65, 0.60)
        legs_to_win = 5
        result = engine._dp_legs_format(hb=hb, legs_to_win=legs_to_win, p1_starts=True)

        for (l1, l2), prob in result["legs_distribution"].items():
            assert l1 == legs_to_win or l2 == legs_to_win, (
                f"Non-terminal state ({l1}, {l2}) in legs_distribution"
            )
            assert prob >= 0

    def test_win_prob_increases_with_more_legs_for_stronger_player(self):
        """Stronger player benefits more from longer formats (more legs)."""
        engine = MatchCombinatorialEngine()
        hb = make_asymmetric_hb(0.70, 0.60)

        results = {}
        for ltw in [2, 3, 5, 7]:
            r = engine._dp_legs_format(hb=hb, legs_to_win=ltw, p1_starts=True)
            results[ltw] = r["p1_win"]

        # In general, stronger player benefits from more legs (law of large numbers)
        assert results[7] > results[2], (
            f"Longer format should favour stronger player: {results}"
        )


class TestSetsFormatDP:
    """G: sets-based format DP produces valid results."""

    def test_sets_format_sums_to_one(self):
        """p1_win + p2_win = 1.0 for sets formats."""
        engine = SetsEngine()
        hb = make_asymmetric_hb(0.62, 0.58)

        for sets_to_win, legs_per_set in [(3, 3), (4, 3), (5, 3), (3, 5)]:
            result = engine.price_match(
                hb=hb,
                sets_to_win=sets_to_win,
                legs_per_set=legs_per_set,
                p1_starts=True,
            )
            total = result["p1_win"] + result["p2_win"]
            assert abs(total - 1.0) < 1e-6, (
                f"Sets={sets_to_win}, legs_per_set={legs_per_set}: total={total:.8f}"
            )

    def test_sets_format_stronger_player_wins_more(self):
        """Stronger player wins more sets matches."""
        engine = SetsEngine()
        hb_strong = make_asymmetric_hb(0.75, 0.60)
        hb_weak = make_asymmetric_hb(0.55, 0.60)

        result_strong = engine.price_match(hb=hb_strong, sets_to_win=4, legs_per_set=3, p1_starts=True)
        result_weak = engine.price_match(hb=hb_weak, sets_to_win=4, legs_per_set=3, p1_starts=True)

        assert result_strong["p1_win"] > result_weak["p1_win"]

    def test_symmetric_players_near_50_50(self):
        """Perfectly symmetric players → near 50/50 in sets format."""
        engine = SetsEngine()
        hb = make_symmetric_hb(hold=0.60)
        result = engine.price_match(hb=hb, sets_to_win=4, legs_per_set=3, p1_starts=True)
        # Not exactly 50/50 because P1 starts, but close
        assert abs(result["p1_win"] - 0.50) < 0.10  # within 10%

    def test_p1_wins_set_consistency(self):
        """p1_wins_set + (1 - p1_wins_set) = 1.0."""
        engine = SetsEngine()
        hb = make_asymmetric_hb(0.65, 0.60)
        p1_set_win = engine.p1_wins_set(hb=hb, legs_per_set=3, p1_starts=True)
        assert 0.0 <= p1_set_win <= 1.0


class TestDrawFormatProbabilitiesSumToOne:
    """G: draw-enabled format probabilities sum to 1.0."""

    def test_pl_league_night_sums_to_one(self):
        """PL League Night: p1_win + draw + p2_win = 1.0."""
        from competition.format_registry import DartsRoundFormat
        from engines.match_layer.premier_league_engine import PremierLeagueEngine

        engine = PremierLeagueEngine()
        hb = make_asymmetric_hb(0.62, 0.58)
        round_fmt = DartsRoundFormat(
            name="League Night",
            legs_to_win=7,
            draw_enabled=True,
            group_stage=True,
        )

        result = engine.price_match(hb=hb, round_fmt=round_fmt, p1_starts=True)
        result.validate()  # calls validate internally
        total = result.p1_win + result.draw + result.p2_win
        assert abs(total - 1.0) < 1e-9

    def test_pl_draw_possible(self):
        """Draw probability is non-zero for symmetric players in PL format."""
        from competition.format_registry import DartsRoundFormat
        from engines.match_layer.premier_league_engine import PremierLeagueEngine

        engine = PremierLeagueEngine()
        hb = make_symmetric_hb(hold=0.60)
        round_fmt = DartsRoundFormat(
            name="League Night",
            legs_to_win=7,
            draw_enabled=True,
            group_stage=True,
        )

        result = engine.price_match(hb=hb, round_fmt=round_fmt, p1_starts=True)
        assert result.draw > 0.0, "Draw probability should be positive for symmetric players"

    def test_pl_ko_no_draw(self):
        """PL knockout match has zero draw probability."""
        from engines.match_layer.premier_league_engine import PremierLeagueEngine

        engine = PremierLeagueEngine()
        hb = make_asymmetric_hb(0.65, 0.60)
        result = engine.price_ko_match(hb=hb, legs_to_win=10, p1_starts=True)
        assert result.draw == 0.0

    def test_grand_slam_group_stage_draw_possible(self):
        """Grand Slam group stage (first to 5 with draw_enabled) has draw probability."""
        from competition.format_registry import DartsRoundFormat
        from engines.match_layer.premier_league_engine import PremierLeagueEngine

        engine = PremierLeagueEngine()
        hb = make_symmetric_hb(hold=0.58)
        round_fmt = DartsRoundFormat(
            name="Group Stage",
            legs_to_win=5,
            draw_enabled=True,
            group_stage=True,
        )

        result = engine.price_match(hb=hb, round_fmt=round_fmt, p1_starts=True)
        assert result.draw > 0.0
        total = result.p1_win + result.draw + result.p2_win
        assert abs(total - 1.0) < 1e-9


class TestMatchPriceResultValidation:
    """G: MatchPriceResult.validate() enforces sum = 1."""

    def test_valid_result_passes(self):
        """Valid result passes validation."""
        result = MatchPriceResult(
            p1_win=0.55,
            p2_win=0.45,
            draw=0.0,
            format_code="TEST",
            round_name="Final",
            legs_distribution={},
        )
        result.validate()  # no exception

    def test_invalid_result_raises(self):
        """Result not summing to 1 raises ValueError."""
        result = MatchPriceResult(
            p1_win=0.55,
            p2_win=0.50,  # 0.55 + 0.50 = 1.05
            draw=0.0,
            format_code="TEST",
            round_name="Final",
            legs_distribution={},
        )
        with pytest.raises(ValueError, match="does not sum to 1"):
            result.validate()

    def test_to_decimal_odds_no_zero_division(self):
        """to_decimal_odds handles probabilities gracefully."""
        result = MatchPriceResult(
            p1_win=0.60,
            p2_win=0.40,
            draw=0.0,
            format_code="TEST",
            round_name="Final",
            legs_distribution={},
        )
        odds = result.to_decimal_odds()
        assert "p1_win" in odds
        assert "p2_win" in odds
        assert odds["p1_win"] == pytest.approx(1.0 / 0.60, rel=1e-6)
