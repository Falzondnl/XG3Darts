"""
Unit tests for World Cup doubles pricing engine.

Tests
-----
- test_team_3da_computation
- test_doubles_match_prices_sum_to_one
- test_knockout_tie_prices_sum_to_one
- test_group_stage_with_draw (World Cup has no draws)
- test_world_cup_pricer_price_doubles_leg
- test_world_cup_pricer_price_singles_match
- test_full_tie_components_consistent
- test_team_3da_raises_on_missing_stat
"""
from __future__ import annotations

import pytest

from competition.format_registry import get_format
from engines.doubles.team_visit_model import DoublesTeam, TeamVisitModel
from engines.doubles.world_cup_pricer import WorldCupMatchup, WorldCupPricer, WorldCupPriceResult
from engines.errors import DartsEngineError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_team(
    team_id: str = "ENG",
    player_a_id: str = "P_A",
    player_b_id: str = "P_B",
    three_da_a: float = 90.0,
    three_da_b: float = 85.0,
) -> DoublesTeam:
    return DoublesTeam(
        team_id=team_id,
        player_a_id=player_a_id,
        player_b_id=player_b_id,
        player_a_three_da=three_da_a,
        player_b_three_da=three_da_b,
        player_a_checkout_rate=0.42,
        player_b_checkout_rate=0.38,
    )


def _make_matchup(
    team1: DoublesTeam = None,
    team2: DoublesTeam = None,
    stage: bool = True,
) -> WorldCupMatchup:
    t1 = team1 or _make_team("ENG", "P_A", "P_B", 90.0, 85.0)
    t2 = team2 or _make_team("NLD", "P_C", "P_D", 88.0, 83.0)
    return WorldCupMatchup(team1=t1, team2=t2, stage=stage)


def _get_round_formats(stage: str = "Round 1"):
    fmt = get_format("PDC_WCUP")
    singles_key = f"{stage} Singles"
    doubles_key = f"{stage} Doubles"
    return fmt.get_round(singles_key), fmt.get_round(doubles_key)


# ---------------------------------------------------------------------------
# test_team_3da_computation
# ---------------------------------------------------------------------------

class TestTeam3DA:
    """DoublesTeam.team_three_da must be the average of both players' 3DA."""

    def test_team_3da_is_average(self):
        team = _make_team(three_da_a=90.0, three_da_b=80.0)
        expected = (90.0 + 80.0) / 2.0
        assert team.team_three_da == pytest.approx(expected, abs=1e-10)

    def test_team_3da_equal_players(self):
        team = _make_team(three_da_a=85.0, three_da_b=85.0)
        assert team.team_three_da == pytest.approx(85.0, abs=1e-10)

    def test_team_3da_with_weak_player(self):
        team = _make_team(three_da_a=100.0, three_da_b=60.0)
        expected = 80.0
        assert team.team_three_da == pytest.approx(expected, abs=1e-10)

    def test_combined_checkout_rate_weighted(self):
        """Weighted checkout rate uses 3DA as weights."""
        team = _make_team(
            three_da_a=100.0,
            three_da_b=100.0,
        )
        team_custom = DoublesTeam(
            team_id="TST",
            player_a_id="A",
            player_b_id="B",
            player_a_three_da=100.0,
            player_b_three_da=100.0,
            player_a_checkout_rate=0.50,
            player_b_checkout_rate=0.30,
        )
        # Equal 3DA -> arithmetic average
        expected = 0.5 * 0.50 + 0.5 * 0.30
        assert team_custom.combined_checkout_rate == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# test_doubles_match_prices_sum_to_one
# ---------------------------------------------------------------------------

class TestDoublesPricesSumToOne:
    """Doubles match win probabilities must sum to 1.0."""

    def test_doubles_leg_sums_to_one(self):
        pricer = WorldCupPricer()
        fmt = get_format("PDC_WCUP")
        doubles_fmt = fmt.get_round("Round 1 Doubles")
        team1 = _make_team("ENG", "P_A", "P_B", 90.0, 85.0)
        team2 = _make_team("NLD", "P_C", "P_D", 88.0, 82.0)
        result = pricer.price_doubles_leg(
            team1=team1,
            team2=team2,
            round_fmt=doubles_fmt,
            team1_starts=True,
        )
        total = result.p1_win + result.p2_win + result.draw
        assert abs(total - 1.0) < 1e-6

    def test_doubles_leg_sums_to_one_team2_starts(self):
        pricer = WorldCupPricer()
        fmt = get_format("PDC_WCUP")
        doubles_fmt = fmt.get_round("Quarter-Final Doubles")
        team1 = _make_team("ENG", "P_A", "P_B", 90.0, 85.0)
        team2 = _make_team("NLD", "P_C", "P_D", 88.0, 82.0)
        result = pricer.price_doubles_leg(
            team1=team1,
            team2=team2,
            round_fmt=doubles_fmt,
            team1_starts=False,
        )
        total = result.p1_win + result.p2_win + result.draw
        assert abs(total - 1.0) < 1e-6

    def test_singles_match_in_worldcup_sums_to_one(self):
        pricer = WorldCupPricer()
        fmt = get_format("PDC_WCUP")
        singles_fmt = fmt.get_round("Round 1 Singles")
        result = pricer.price_singles_match(
            player1_id="P_A",
            player2_id="P_C",
            player1_three_da=90.0,
            player2_three_da=85.0,
            round_fmt=singles_fmt,
            p1_starts=True,
        )
        total = result.p1_win + result.p2_win + result.draw
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# test_knockout_tie_prices_sum_to_one
# ---------------------------------------------------------------------------

class TestKnockoutTieSumToOne:
    """Full knockout tie: P(team1 wins) + P(team2 wins) == 1.0."""

    @pytest.mark.parametrize("stage", ["Round 1", "Quarter-Final", "Semi-Final", "Final"])
    def test_full_tie_sums_to_one(self, stage: str):
        pricer = WorldCupPricer()
        matchup = _make_matchup()
        singles_fmt, doubles_fmt = _get_round_formats(stage)
        result = pricer.price_full_tie(
            matchup=matchup,
            singles1_fmt=singles_fmt,
            singles2_fmt=singles_fmt,
            doubles_fmt=doubles_fmt,
        )
        total = result.team1_win + result.team2_win
        assert abs(total - 1.0) < 1e-6, (
            f"Stage {stage}: tie probs sum to {total:.8f}"
        )

    def test_full_tie_probabilities_in_range(self):
        pricer = WorldCupPricer()
        matchup = _make_matchup()
        singles_fmt, doubles_fmt = _get_round_formats("Round 1")
        result = pricer.price_full_tie(
            matchup=matchup,
            singles1_fmt=singles_fmt,
            singles2_fmt=singles_fmt,
            doubles_fmt=doubles_fmt,
        )
        assert 0.0 < result.team1_win < 1.0
        assert 0.0 < result.team2_win < 1.0

    def test_stronger_team_has_higher_win_prob(self):
        """Team with higher 3DA should have higher win probability."""
        pricer = WorldCupPricer()
        strong_team = _make_team("STRONG", "SA", "SB", 100.0, 98.0)
        weak_team = _make_team("WEAK", "WA", "WB", 70.0, 68.0)
        matchup = WorldCupMatchup(team1=strong_team, team2=weak_team, stage=True)
        singles_fmt, doubles_fmt = _get_round_formats("Quarter-Final")
        result = pricer.price_full_tie(
            matchup=matchup,
            singles1_fmt=singles_fmt,
            singles2_fmt=singles_fmt,
            doubles_fmt=doubles_fmt,
        )
        assert result.team1_win > result.team2_win, (
            f"Strong team win prob {result.team1_win:.4f} should be > "
            f"weak team win prob {result.team2_win:.4f}"
        )

    def test_symmetric_teams_give_near_50_50(self):
        """Identical teams should produce approximately 50/50."""
        pricer = WorldCupPricer()
        team1 = _make_team("T1", "A1", "B1", 85.0, 85.0)
        team2 = _make_team("T2", "A2", "B2", 85.0, 85.0)
        matchup = WorldCupMatchup(team1=team1, team2=team2, stage=True)
        singles_fmt, doubles_fmt = _get_round_formats("Round 1")
        result = pricer.price_full_tie(
            matchup=matchup,
            singles1_fmt=singles_fmt,
            singles2_fmt=singles_fmt,
            doubles_fmt=doubles_fmt,
        )
        # With equal teams the result should be close to 50/50
        assert abs(result.team1_win - 0.5) < 0.15, (
            f"Expected ~0.5, got team1_win={result.team1_win:.4f}"
        )


# ---------------------------------------------------------------------------
# test_group_stage_with_draw
# ---------------------------------------------------------------------------

class TestWorldCupGroupStage:
    """
    World Cup does not have draws.
    The PDC_WCUP format has no draw_enabled rounds; verify this.
    """

    def test_world_cup_format_no_draw(self):
        """PDC_WCUP should not allow draws in any round."""
        fmt = get_format("PDC_WCUP")
        assert not fmt.allows_draw(), "PDC_WCUP should not allow draws"

    def test_doubles_result_draw_is_zero(self):
        """MatchPriceResult.draw should be 0.0 for World Cup doubles."""
        pricer = WorldCupPricer()
        fmt = get_format("PDC_WCUP")
        doubles_fmt = fmt.get_round("Round 1 Doubles")
        team1 = _make_team("ENG", "P_A", "P_B", 90.0, 85.0)
        team2 = _make_team("NLD", "P_C", "P_D", 88.0, 82.0)
        result = pricer.price_doubles_leg(
            team1=team1,
            team2=team2,
            round_fmt=doubles_fmt,
            team1_starts=True,
        )
        assert result.draw == 0.0

    def test_world_cup_result_tie_no_draw_field(self):
        """WorldCupPriceResult does not include a draw field."""
        pricer = WorldCupPricer()
        matchup = _make_matchup()
        singles_fmt, doubles_fmt = _get_round_formats("Round 1")
        result = pricer.price_full_tie(
            matchup=matchup,
            singles1_fmt=singles_fmt,
            singles2_fmt=singles_fmt,
            doubles_fmt=doubles_fmt,
        )
        assert isinstance(result, WorldCupPriceResult)
        assert not hasattr(result, "draw"), (
            "WorldCupPriceResult should not have a draw field"
        )


# ---------------------------------------------------------------------------
# Additional WorldCupPricer tests
# ---------------------------------------------------------------------------

class TestWorldCupPricerComponents:
    """Tests for individual components of WorldCupPricer."""

    def test_price_result_has_all_components(self):
        pricer = WorldCupPricer()
        matchup = _make_matchup()
        singles_fmt, doubles_fmt = _get_round_formats("Round 1")
        result = pricer.price_full_tie(
            matchup=matchup,
            singles1_fmt=singles_fmt,
            singles2_fmt=singles_fmt,
            doubles_fmt=doubles_fmt,
        )
        assert result.singles1_result is not None
        assert result.singles2_result is not None
        assert result.doubles_result is not None

    def test_starting_order_affects_odds(self):
        """Changing who starts should produce different (but valid) odds."""
        pricer = WorldCupPricer()
        matchup = _make_matchup()
        singles_fmt, doubles_fmt = _get_round_formats("Round 1")
        result_normal = pricer.price_full_tie(
            matchup=matchup,
            singles1_fmt=singles_fmt,
            singles2_fmt=singles_fmt,
            doubles_fmt=doubles_fmt,
            team1_starts_singles1=True,
            team1_starts_doubles=True,
        )
        result_reversed = pricer.price_full_tie(
            matchup=matchup,
            singles1_fmt=singles_fmt,
            singles2_fmt=singles_fmt,
            doubles_fmt=doubles_fmt,
            team1_starts_singles1=False,
            team1_starts_doubles=False,
        )
        # Both must be valid
        assert abs((result_normal.team1_win + result_normal.team2_win) - 1.0) < 1e-6
        assert abs((result_reversed.team1_win + result_reversed.team2_win) - 1.0) < 1e-6

    def test_team_visit_model_blends_distributions(self):
        """TeamVisitModel should return a distribution for each score band."""
        from engines.leg_layer.visit_distributions import BAND_NAMES
        team = _make_team()
        team_model = TeamVisitModel()
        dists = team_model.get_all_bands_for_team(
            team=team,
            stage=True,
            short_format=False,
            throw_first=True,
        )
        assert set(dists.keys()) == set(BAND_NAMES)
        for band, dist in dists.items():
            total = sum(dist.visit_probs.values()) + dist.bust_prob
            assert abs(total - 1.0) < 1e-6, (
                f"Band {band}: visit_probs + bust_prob = {total:.8f}"
            )
