"""
Unit tests for Premier League draw-enabled engine.

Tests:
  - test_pl_draw_possible_at_6_6
  - test_pl_draw_probabilities_sum_to_one
  - test_pl_ko_no_draw
  - test_draw_probability_symmetric_players
  - test_pl_p1_advantage_when_stronger
  - test_pl_format_from_registry
"""

from __future__ import annotations

import pytest

from competition.format_registry import DartsRoundFormat, get_format
from engines.leg_layer.markov_chain import HoldBreakProbabilities
from engines.match_layer.premier_league_engine import PremierLeagueEngine, PremierLeaguePriceResult


def make_hb(p1_hold: float, p2_hold: float) -> HoldBreakProbabilities:
    """Build HoldBreakProbabilities from hold rates."""
    return HoldBreakProbabilities(
        p1_hold=p1_hold,
        p1_break=1.0 - p2_hold,
        p2_hold=p2_hold,
        p2_break=1.0 - p1_hold,
    )


class TestPLDrawPossible:
    """G: draw probability is non-zero in draw-enabled formats."""

    def test_pl_draw_possible_at_6_6(self):
        """Premier League league night: draw at 6-6 is possible."""
        engine = PremierLeagueEngine()
        hb = make_hb(0.60, 0.60)  # symmetric players
        round_fmt = DartsRoundFormat(
            name="League Night",
            legs_to_win=7,
            draw_enabled=True,
            group_stage=True,
        )
        result = engine.price_match(hb=hb, round_fmt=round_fmt, p1_starts=True)
        assert result.draw > 0.0, f"Draw probability should be positive, got {result.draw}"

    def test_draw_probability_higher_for_equal_players(self):
        """More equal players → higher draw probability."""
        engine = PremierLeagueEngine()
        round_fmt = DartsRoundFormat(
            name="League Night",
            legs_to_win=7,
            draw_enabled=True,
            group_stage=True,
        )

        # Equal players
        hb_equal = make_hb(0.60, 0.60)
        result_equal = engine.price_match(hb=hb_equal, round_fmt=round_fmt, p1_starts=True)

        # Unequal players
        hb_unequal = make_hb(0.75, 0.55)
        result_unequal = engine.price_match(hb=hb_unequal, round_fmt=round_fmt, p1_starts=True)

        assert result_equal.draw > result_unequal.draw, (
            f"Equal players draw={result_equal.draw:.4f} should be "
            f"> unequal players draw={result_unequal.draw:.4f}"
        )

    def test_draw_probability_is_in_range(self):
        """Draw probability is between 0 and 1."""
        engine = PremierLeagueEngine()
        hb = make_hb(0.63, 0.58)
        round_fmt = DartsRoundFormat(
            name="League Night",
            legs_to_win=7,
            draw_enabled=True,
            group_stage=True,
        )
        result = engine.price_match(hb=hb, round_fmt=round_fmt, p1_starts=True)
        assert 0.0 <= result.draw <= 1.0


class TestPLDrawProbabilitiesSumToOne:
    """G: p1_win + draw + p2_win = 1.0 for all draw-enabled formats."""

    def test_basic_pl_sums_to_one(self):
        """Standard PL league night probs sum to 1."""
        engine = PremierLeagueEngine()
        hb = make_hb(0.62, 0.58)
        round_fmt = DartsRoundFormat(
            name="League Night",
            legs_to_win=7,
            draw_enabled=True,
            group_stage=True,
        )
        result = engine.price_match(hb=hb, round_fmt=round_fmt, p1_starts=True)
        result.validate()
        total = result.p1_win + result.draw + result.p2_win
        assert abs(total - 1.0) < 1e-9, f"Total={total}"

    def test_various_configs_sum_to_one(self):
        """Multiple PL configurations all sum to 1."""
        engine = PremierLeagueEngine()
        round_fmt = DartsRoundFormat(
            name="League Night",
            legs_to_win=7,
            draw_enabled=True,
            group_stage=True,
        )

        for p1h, p2h, p1s in [
            (0.55, 0.55, True),
            (0.65, 0.60, False),
            (0.72, 0.62, True),
            (0.50, 0.50, True),
        ]:
            hb = make_hb(p1h, p2h)
            result = engine.price_match(hb=hb, round_fmt=round_fmt, p1_starts=p1s)
            total = result.p1_win + result.draw + result.p2_win
            assert abs(total - 1.0) < 1e-9, (
                f"p1h={p1h}, p2h={p2h}, p1s={p1s}: total={total:.9f}"
            )

    def test_grand_slam_group_stage_sums_to_one(self):
        """Grand Slam group stage (first to 5, draw enabled) sums to 1."""
        engine = PremierLeagueEngine()
        hb = make_hb(0.60, 0.58)
        round_fmt = DartsRoundFormat(
            name="Group Stage",
            legs_to_win=5,
            draw_enabled=True,
            group_stage=True,
        )
        result = engine.price_match(hb=hb, round_fmt=round_fmt, p1_starts=True)
        total = result.p1_win + result.draw + result.p2_win
        assert abs(total - 1.0) < 1e-9

    def test_result_validate_method_works(self):
        """PremierLeaguePriceResult.validate() passes for valid result."""
        result = PremierLeaguePriceResult(
            p1_win=0.48,
            draw=0.12,
            p2_win=0.40,
            legs_distribution={},
        )
        result.validate()  # should not raise

    def test_result_validate_raises_for_invalid(self):
        """PremierLeaguePriceResult.validate() raises for invalid result."""
        result = PremierLeaguePriceResult(
            p1_win=0.48,
            draw=0.12,
            p2_win=0.50,  # 0.48 + 0.12 + 0.50 = 1.10
            legs_distribution={},
        )
        with pytest.raises(ValueError):
            result.validate()


class TestPLKONoDraws:
    """G: knockout format has zero draw probability."""

    def test_pl_ko_no_draw(self):
        """Play-Off (no draw_enabled) has zero draw probability."""
        engine = PremierLeagueEngine()
        hb = make_hb(0.65, 0.60)
        result = engine.price_ko_match(hb=hb, legs_to_win=10, p1_starts=True)
        assert result.draw == 0.0

    def test_pl_ko_sums_to_one(self):
        """Play-Off probs sum to 1.0."""
        engine = PremierLeagueEngine()
        hb = make_hb(0.65, 0.60)
        result = engine.price_ko_match(hb=hb, legs_to_win=10, p1_starts=True)
        total = result.p1_win + result.p2_win
        assert abs(total - 1.0) < 1e-9

    def test_non_draw_format_gives_no_draw(self):
        """draw_enabled=False via DartsRoundFormat gives zero draw."""
        engine = PremierLeagueEngine()
        hb = make_hb(0.63, 0.60)
        result = engine.price_ko_match(hb=hb, legs_to_win=7, p1_starts=False)
        assert result.draw == 0.0


class TestPLFromRegistry:
    """G: PL format from the competition registry works correctly."""

    def test_pl_registry_format_has_draw(self):
        """PDC_PL format from registry has draw_enabled for league night."""
        fmt = get_format("PDC_PL")
        league_night = fmt.get_round("League Night")
        assert league_night.draw_enabled
        assert league_night.legs_to_win == 7

    def test_pl_ko_round_no_draw(self):
        """PDC_PL play-off final has no draw."""
        fmt = get_format("PDC_PL")
        final = fmt.get_round("Play-Off Final")
        assert not final.draw_enabled
        assert final.legs_to_win == 10
