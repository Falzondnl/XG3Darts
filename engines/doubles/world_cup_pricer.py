"""
World Cup doubles pricer.

The PDC World Cup of Darts is a team format where two players
compete as a national pair against another national pair.

Pricing structure:
  - Singles matches: standard 1v1 pricing (same as singles engine)
  - Doubles match: team vs team using TeamVisitModel
  - Tie context: if singles is split 1-1, doubles is the decider

The doubles leg is played with alternating visits within the team:
  Team 1: Player A throws, then Player B throws (alternating visits within team)
  Team 2: Player C throws, then Player D throws (alternating visits within team)
  Full match: Team 1 visits and Team 2 visits alternate (team 1 throws, team 2 throws, etc.)

Checkout: any player on the team can finish the leg on their visit.

This module prices the doubles match component of the World Cup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog

from competition.format_registry import DartsRoundFormat
from engines.doubles.team_visit_model import DoublesTeam, TeamVisitModel
from engines.leg_layer.markov_chain import DartsMarkovChain, HoldBreakProbabilities
from engines.match_layer.match_combinatorics import MatchCombinatorialEngine, MatchPriceResult

logger = structlog.get_logger(__name__)


@dataclass
class WorldCupMatchup:
    """
    Complete World Cup singles + doubles matchup.

    Attributes
    ----------
    team1 / team2:
        The two competing teams.
    player_a1_id / player_b1_id:
        Player IDs for Team 1 (A = higher ranked).
    player_a2_id / player_b2_id:
        Player IDs for Team 2.
    p1_three_da_singles / p2_three_da_singles:
        3DA for each team's singles players.
    stage:
        Whether this is a televised stage event.
    """

    team1: DoublesTeam
    team2: DoublesTeam
    stage: bool = True


@dataclass
class WorldCupPriceResult:
    """
    Full World Cup match price result.

    Attributes
    ----------
    team1_win:
        P(Team 1 wins the overall tie).
    team2_win:
        P(Team 2 wins the overall tie).
    singles1_result:
        Price result for first singles match.
    singles2_result:
        Price result for second singles match.
    doubles_result:
        Price result for doubles match.
    """

    team1_win: float
    team2_win: float
    singles1_result: Optional[MatchPriceResult]
    singles2_result: Optional[MatchPriceResult]
    doubles_result: Optional[MatchPriceResult]


class WorldCupPricer:
    """
    Prices PDC World Cup doubles tie.

    A World Cup tie consists of:
    1. Singles match (Player A vs Player C)
    2. Singles match (Player B vs Player D)
    3. Doubles match (Team 1 vs Team 2) — played if 1-1 after singles, or always

    For early rounds (before QF), format:
    - 2 singles (best-of-N each) + 1 doubles (best-of-M)
    - Team wins the tie by winning 2 of 3 matches.
    """

    def __init__(self) -> None:
        self._team_visit_model = TeamVisitModel()
        self._markov = DartsMarkovChain()
        self._match_engine = MatchCombinatorialEngine()

    def price_doubles_leg(
        self,
        team1: DoublesTeam,
        team2: DoublesTeam,
        round_fmt: DartsRoundFormat,
        team1_starts: bool,
        stage: bool = True,
    ) -> MatchPriceResult:
        """
        Price a single doubles match.

        Computes hold/break probabilities for both teams using the
        team visit model, then applies standard match combinatorics.

        Parameters
        ----------
        team1 / team2:
            The two competing teams.
        round_fmt:
            Format for this doubles match.
        team1_starts:
            Whether Team 1 starts the first leg.
        stage:
            Stage event flag.

        Returns
        -------
        MatchPriceResult
        """
        assert round_fmt.legs_to_win is not None, "Doubles round must have legs_to_win"

        # Get team visit distributions for all bands
        team1_dists = self._team_visit_model.get_all_bands_for_team(
            team=team1,
            stage=stage,
            short_format=(round_fmt.legs_to_win <= 4),
            throw_first=team1_starts,
        )
        team2_dists = self._team_visit_model.get_all_bands_for_team(
            team=team2,
            stage=stage,
            short_format=(round_fmt.legs_to_win <= 4),
            throw_first=not team1_starts,
        )

        # Compute hold/break probabilities via Markov chain
        hb = self._markov.break_probability(
            p1_visit_dists=team1_dists,
            p2_visit_dists=team2_dists,
            p1_id=team1.team_id,
            p2_id=team2.team_id,
            p1_three_da=team1.team_three_da,
            p2_three_da=team2.team_three_da,
            starting_score=501,
            double_start=False,
        )

        # Price the match
        result = self._match_engine._dp_legs_format(
            hb=hb,
            legs_to_win=round_fmt.legs_to_win,
            p1_starts=team1_starts,
        )

        logger.info(
            "doubles_leg_priced",
            team1=team1.team_id,
            team2=team2.team_id,
            legs_to_win=round_fmt.legs_to_win,
            team1_win=round(result["p1_win"], 4),
            team2_win=round(result["p2_win"], 4),
        )

        return MatchPriceResult(
            p1_win=result["p1_win"],
            p2_win=result["p2_win"],
            draw=0.0,
            format_code="PDC_WCUP",
            round_name=round_fmt.name,
            legs_distribution=result.get("legs_distribution", {}),
        )

    def price_singles_match(
        self,
        player1_id: str,
        player2_id: str,
        player1_three_da: float,
        player2_three_da: float,
        round_fmt: DartsRoundFormat,
        p1_starts: bool,
        stage: bool = True,
    ) -> MatchPriceResult:
        """
        Price a singles match within the World Cup tie.

        Uses the standard singles Markov chain (same as prematch pricing).
        """
        from engines.leg_layer.visit_distributions import HierarchicalVisitDistributionModel
        visit_model = HierarchicalVisitDistributionModel()

        p1_dists = visit_model.get_all_bands(
            player_id=player1_id,
            stage=stage,
            short_format=(round_fmt.legs_to_win is not None and round_fmt.legs_to_win <= 6),
            throw_first=p1_starts,
            three_da=player1_three_da,
        )
        p2_dists = visit_model.get_all_bands(
            player_id=player2_id,
            stage=stage,
            short_format=(round_fmt.legs_to_win is not None and round_fmt.legs_to_win <= 6),
            throw_first=not p1_starts,
            three_da=player2_three_da,
        )

        hb = self._markov.break_probability(
            p1_visit_dists=p1_dists,
            p2_visit_dists=p2_dists,
            p1_id=player1_id,
            p2_id=player2_id,
            p1_three_da=player1_three_da,
            p2_three_da=player2_three_da,
        )

        assert round_fmt.legs_to_win is not None
        result = self._match_engine._dp_legs_format(
            hb=hb,
            legs_to_win=round_fmt.legs_to_win,
            p1_starts=p1_starts,
        )

        return MatchPriceResult(
            p1_win=result["p1_win"],
            p2_win=result["p2_win"],
            draw=0.0,
            format_code="PDC_WCUP",
            round_name=round_fmt.name,
            legs_distribution=result.get("legs_distribution", {}),
        )

    def price_full_tie(
        self,
        matchup: WorldCupMatchup,
        singles1_fmt: DartsRoundFormat,
        singles2_fmt: DartsRoundFormat,
        doubles_fmt: DartsRoundFormat,
        team1_starts_singles1: bool = True,
        team1_starts_singles2: bool = False,
        team1_starts_doubles: bool = True,
    ) -> WorldCupPriceResult:
        """
        Price the full World Cup tie (2 singles + 1 doubles).

        The tie winner is determined by the best-of-3 between:
        singles1, singles2, doubles.

        Parameters
        ----------
        matchup:
            The two teams.
        *_fmt:
            Format specifications for each match type.
        team1_starts_*:
            Starting throw assignments for each match.

        Returns
        -------
        WorldCupPriceResult
        """
        # Price each component
        singles1 = self.price_singles_match(
            player1_id=matchup.team1.player_a_id,
            player2_id=matchup.team2.player_a_id,
            player1_three_da=matchup.team1.player_a_three_da,
            player2_three_da=matchup.team2.player_a_three_da,
            round_fmt=singles1_fmt,
            p1_starts=team1_starts_singles1,
            stage=matchup.stage,
        )

        singles2 = self.price_singles_match(
            player1_id=matchup.team1.player_b_id,
            player2_id=matchup.team2.player_b_id,
            player1_three_da=matchup.team1.player_b_three_da,
            player2_three_da=matchup.team2.player_b_three_da,
            round_fmt=singles2_fmt,
            p1_starts=team1_starts_singles2,
            stage=matchup.stage,
        )

        doubles = self.price_doubles_leg(
            team1=matchup.team1,
            team2=matchup.team2,
            round_fmt=doubles_fmt,
            team1_starts=team1_starts_doubles,
            stage=matchup.stage,
        )

        # Compute P(Team 1 wins tie) via combinatorics on the 3-match series
        # Team wins tie by winning 2 of 3 matches.
        # P(T1 wins tie) = P(T1 wins S1)*P(T1 wins S2)   [2-0]
        #                + P(T1 wins S1)*P(T2 wins S2)*P(T1 wins D)  [2-1 via S1+D]
        #                + P(T2 wins S1)*P(T1 wins S2)*P(T1 wins D)  [2-1 via S2+D]

        p1_s1 = singles1.p1_win
        p1_s2 = singles2.p1_win
        p1_d = doubles.p1_win

        # 2-0: T1 wins both singles
        p_2_0 = p1_s1 * p1_s2
        # 2-1 (T1 wins S1 and D, loses S2)
        p_2_1a = p1_s1 * (1.0 - p1_s2) * p1_d
        # 2-1 (T1 wins S2 and D, loses S1)
        p_2_1b = (1.0 - p1_s1) * p1_s2 * p1_d

        team1_win = p_2_0 + p_2_1a + p_2_1b
        team2_win = 1.0 - team1_win

        logger.info(
            "world_cup_tie_priced",
            team1=matchup.team1.team_id,
            team2=matchup.team2.team_id,
            team1_win=round(team1_win, 4),
            team2_win=round(team2_win, 4),
        )

        return WorldCupPriceResult(
            team1_win=team1_win,
            team2_win=team2_win,
            singles1_result=singles1,
            singles2_result=singles2,
            doubles_result=doubles,
        )
