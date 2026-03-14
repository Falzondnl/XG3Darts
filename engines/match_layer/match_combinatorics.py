"""
Exact combinatorics for match win probability from leg win probability.

Handles:
1. First-to-N-legs formats (legs only)
2. First-to-N-sets formats (each set = first to M legs)
3. Draw-enabled formats (Premier League, Grand Slam group stage)
4. Two-clear-legs formats (World Matchplay) — delegates to WorldMatchplayEngine

Algorithm:
  DP over (legs_p1, legs_p2, current_starter) state space.
  P(P1 wins from (n1, n2) with P1 to throw) =
    lwp * P(P1 wins from (n1+1, n2) with P2 to throw)
    + (1-lwp) * P(P1 wins from (n1, n2+1) with P1 to throw)

Note: alternating starts means hold/break probabilities alternate each leg.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import structlog

from competition.format_registry import DartsCompetitionFormat, DartsRoundFormat
from engines.leg_layer.markov_chain import HoldBreakProbabilities

logger = structlog.get_logger(__name__)


@dataclass
class MatchPriceResult:
    """
    Complete match price result from the combinatorial engine.

    Attributes
    ----------
    p1_win:
        Probability P1 wins the match.
    p2_win:
        Probability P2 wins the match.
    draw:
        Probability of a draw (0.0 for non-draw formats).
    format_code:
        The competition format code.
    round_name:
        The round name.
    legs_distribution:
        Distribution over final leg scores (e.g. {(3,0): 0.25, (3,1): 0.35, ...}).
    """

    p1_win: float
    p2_win: float
    draw: float
    format_code: str
    round_name: str
    legs_distribution: dict[tuple[int, int], float]

    def validate(self, tol: float = 1e-6) -> None:
        """Verify probabilities sum to 1.0."""
        total = self.p1_win + self.p2_win + self.draw
        if abs(total - 1.0) > tol:
            raise ValueError(
                f"Match price does not sum to 1: "
                f"p1={self.p1_win:.6f}, p2={self.p2_win:.6f}, draw={self.draw:.6f}, "
                f"total={total:.6f}"
            )

    def to_decimal_odds(self) -> dict[str, float]:
        """Convert to decimal odds (no margin applied)."""
        result = {
            "p1_win": 1.0 / self.p1_win if self.p1_win > 0 else float("inf"),
            "p2_win": 1.0 / self.p2_win if self.p2_win > 0 else float("inf"),
        }
        if self.draw > 0:
            result["draw"] = 1.0 / self.draw
        return result


class MatchCombinatorialEngine:
    """
    Combinatorial engine for computing match win probabilities.

    Delegates to specialised engines for:
    - WorldMatchplayEngine: two-clear-legs formats
    - PremierLeagueEngine: draw-enabled formats
    - SetsEngine: nested sets formats
    """

    def price_match(
        self,
        hold_break: HoldBreakProbabilities,
        fmt: DartsCompetitionFormat,
        round_name: str,
        p1_starts_first: bool,
    ) -> MatchPriceResult:
        """
        Compute all market prices for this match.

        Dispatches to the appropriate sub-engine based on format rules.

        Parameters
        ----------
        hold_break:
            Pre-computed hold/break probabilities from the Markov chain.
        fmt:
            Competition format definition.
        round_name:
            The specific round (to look up per_round format).
        p1_starts_first:
            Whether P1 throws first in the first leg.
        """
        hold_break.validate()
        round_fmt = fmt.get_round(round_name)

        logger.info(
            "pricing_match",
            format=fmt.code,
            round=round_name,
            p1_starts=p1_starts_first,
            p1_hold=round(hold_break.p1_hold, 4),
            p1_break=round(hold_break.p1_break, 4),
        )

        # Dispatch to specialised engine
        if round_fmt.two_clear_legs:
            from engines.match_layer.world_matchplay_engine import WorldMatchplayEngine
            engine = WorldMatchplayEngine()
            result_dict = engine.price_match(
                round_fmt=round_fmt,
                hb=hold_break,
                p1_starts=p1_starts_first,
            )
            return MatchPriceResult(
                p1_win=result_dict["p1_win"],
                p2_win=result_dict["p2_win"],
                draw=0.0,
                format_code=fmt.code,
                round_name=round_name,
                legs_distribution=result_dict.get("legs_distribution", {}),
            )

        if round_fmt.draw_enabled:
            from engines.match_layer.premier_league_engine import PremierLeagueEngine
            engine = PremierLeagueEngine()
            result = engine.price_match(
                hb=hold_break,
                round_fmt=round_fmt,
                p1_starts=p1_starts_first,
            )
            return MatchPriceResult(
                p1_win=result.p1_win,
                p2_win=result.p2_win,
                draw=result.draw,
                format_code=fmt.code,
                round_name=round_name,
                legs_distribution=result.legs_distribution,
            )

        if round_fmt.is_sets_format:
            from engines.match_layer.sets_engine import SetsEngine
            engine = SetsEngine()
            result_dict = engine.price_match(
                hb=hold_break,
                sets_to_win=round_fmt.sets_to_win,
                legs_per_set=round_fmt.legs_per_set,
                p1_starts=p1_starts_first,
            )
            return MatchPriceResult(
                p1_win=result_dict["p1_win"],
                p2_win=result_dict["p2_win"],
                draw=0.0,
                format_code=fmt.code,
                round_name=round_name,
                legs_distribution=result_dict.get("legs_distribution", {}),
            )

        # Standard legs-only format
        assert round_fmt.legs_to_win is not None
        result_dict = self._dp_legs_format(
            hb=hold_break,
            legs_to_win=round_fmt.legs_to_win,
            p1_starts=p1_starts_first,
        )
        match_result = MatchPriceResult(
            p1_win=result_dict["p1_win"],
            p2_win=result_dict["p2_win"],
            draw=0.0,
            format_code=fmt.code,
            round_name=round_name,
            legs_distribution=result_dict.get("legs_distribution", {}),
        )
        match_result.validate()
        return match_result

    def _dp_legs_format(
        self,
        hb: HoldBreakProbabilities,
        legs_to_win: int,
        p1_starts: bool,
    ) -> dict:
        """
        Dynamic programming over (legs_p1, legs_p2, current_starter) state space.

        State: (l1, l2, starter_idx) where:
          l1, l2 ∈ {0..legs_to_win-1}: legs won by each player
          starter_idx ∈ {0=P1, 1=P2}: who starts the next leg

        Terminal states:
          l1 == legs_to_win → P1 wins
          l2 == legs_to_win → P2 wins

        Transition probabilities:
          When P1 starts leg: P1 wins leg with prob p1_hold; P2 wins with p2_break
          When P2 starts leg: P1 wins leg with prob p1_break; P2 wins with p2_hold

        Returns
        -------
        dict with keys: "p1_win", "p2_win", "legs_distribution"
        """
        N = legs_to_win

        # dp[l1][l2][starter] = P(P1 eventually wins | at state (l1, l2, starter))
        # Use nested dict for clarity; for performance could use numpy
        dp: dict[tuple[int, int, int], float] = {}

        def solve(l1: int, l2: int, starter: int) -> float:
            """Recursive DP with memoization."""
            # Terminal states
            if l1 == N:
                return 1.0  # P1 wins
            if l2 == N:
                return 0.0  # P2 wins

            key = (l1, l2, starter)
            if key in dp:
                return dp[key]

            if starter == 0:
                # P1 starts: holds with p1_hold, P2 breaks with p2_break
                p_p1_wins_leg = hb.p1_hold
                p_p2_wins_leg = hb.p2_break
            else:
                # P2 starts: holds with p2_hold, P1 breaks with p1_break
                p_p1_wins_leg = hb.p1_break
                p_p2_wins_leg = hb.p2_hold

            next_starter = 1 - starter  # alternating starts

            result = (
                p_p1_wins_leg * solve(l1 + 1, l2, next_starter)
                + p_p2_wins_leg * solve(l1, l2 + 1, next_starter)
            )
            dp[key] = result
            return result

        initial_starter = 0 if p1_starts else 1
        p1_win = solve(0, 0, initial_starter)
        p2_win = 1.0 - p1_win

        # Compute legs distribution
        legs_dist = self._compute_legs_distribution(hb, legs_to_win, p1_starts)

        return {
            "p1_win": p1_win,
            "p2_win": p2_win,
            "legs_distribution": legs_dist,
        }

    def _compute_legs_distribution(
        self,
        hb: HoldBreakProbabilities,
        legs_to_win: int,
        p1_starts: bool,
    ) -> dict[tuple[int, int], float]:
        """
        Compute the distribution over final leg scorelines.

        Returns dict mapping (p1_legs, p2_legs) → probability.
        All paths where one player reaches legs_to_win are enumerated.
        """
        N = legs_to_win
        # State probabilities: P(reaching state (l1, l2, starter))
        state_prob: dict[tuple[int, int, int], float] = {}
        initial_starter = 0 if p1_starts else 1
        state_prob[(0, 0, initial_starter)] = 1.0

        # Enumerate all reachable states in order
        for l1 in range(N + 1):
            for l2 in range(N + 1):
                for starter in range(2):
                    if l1 == N or l2 == N:
                        continue
                    prob = state_prob.get((l1, l2, starter), 0.0)
                    if prob == 0.0:
                        continue

                    if starter == 0:
                        p_p1 = hb.p1_hold
                        p_p2 = hb.p2_break
                    else:
                        p_p1 = hb.p1_break
                        p_p2 = hb.p2_hold

                    next_starter = 1 - starter

                    # P1 wins this leg
                    new_l1 = l1 + 1
                    key_p1 = (new_l1, l2, next_starter)
                    state_prob[key_p1] = state_prob.get(key_p1, 0.0) + prob * p_p1

                    # P2 wins this leg
                    new_l2 = l2 + 1
                    key_p2 = (l1, new_l2, next_starter)
                    state_prob[key_p2] = state_prob.get(key_p2, 0.0) + prob * p_p2

        # Extract terminal state probabilities
        legs_dist: dict[tuple[int, int], float] = {}
        for (l1, l2, starter), prob in state_prob.items():
            if l1 == N or l2 == N:
                score_key = (l1, l2)
                legs_dist[score_key] = legs_dist.get(score_key, 0.0) + prob

        return legs_dist

    def _dp_sets_format(
        self,
        hb: HoldBreakProbabilities,
        sets_to_win: int,
        legs_per_set: int,
        p1_starts: bool,
    ) -> dict:
        """
        DP over (sets_p1, sets_p2, legs_in_set_p1, legs_in_set_p2, starter) state.

        Delegates to SetsEngine for the full implementation.
        """
        from engines.match_layer.sets_engine import SetsEngine
        engine = SetsEngine()
        return engine.price_match(
            hb=hb,
            sets_to_win=sets_to_win,
            legs_per_set=legs_per_set,
            p1_starts=p1_starts,
        )

    def p1_win_probability(
        self,
        hb: HoldBreakProbabilities,
        legs_to_win: int,
        p1_starts: bool,
    ) -> float:
        """
        Direct access to P1 match win probability for legs-only format.

        Convenience wrapper over _dp_legs_format.
        """
        result = self._dp_legs_format(hb, legs_to_win, p1_starts)
        return result["p1_win"]

    # ------------------------------------------------------------------
    # Derivative markets (Sprint 4)
    # ------------------------------------------------------------------

    def price_handicap(
        self,
        hb: HoldBreakProbabilities,
        fmt: DartsCompetitionFormat,
        round_name: str,
        p1_starts: bool,
        handicap_legs: int,
    ) -> dict:
        """
        P(P1 covers the -handicap_legs handicap).

        A handicap of ``handicap_legs`` means P1 must win by more than
        ``handicap_legs`` legs.  E.g. handicap_legs=-1.5 means P1 wins
        from the perspective of ``(P1 legs - P2 legs) > -1.5``.

        This implementation supports integer handicaps.  The integer
        handicap_legs is the number of legs P1 is "given" (+) or "conceded" (-).

        Algorithm:
          P(P1 covers -h) = sum over (s1, s2) in legs_distribution
                            where (s1 + h) > s2

        Parameters
        ----------
        hb:
            Hold/break probabilities.
        fmt:
            Competition format.
        round_name:
            Round within competition.
        p1_starts:
            Whether P1 starts first.
        handicap_legs:
            Legs handicap for P1 (positive = P1 starts with extra legs,
            negative = P1 concedes legs).  E.g. handicap_legs=-1 means
            P1 concedes 1 leg.

        Returns
        -------
        dict:
            ``p1_covers``, ``p2_covers``, ``handicap_legs``,
            ``legs_distribution``.
        """
        match_result = self.price_match(
            hold_break=hb,
            fmt=fmt,
            round_name=round_name,
            p1_starts_first=p1_starts,
        )

        legs_dist = match_result.legs_distribution
        round_fmt = fmt.get_round(round_name)

        if not legs_dist:
            logger.warning(
                "handicap_empty_legs_distribution",
                format=fmt.code,
                round=round_name,
            )
            return {
                "p1_covers": match_result.p1_win,
                "p2_covers": match_result.p2_win + match_result.draw,
                "handicap_legs": handicap_legs,
                "legs_distribution": {},
            }

        p1_covers = 0.0
        p2_covers = 0.0

        for (s1, s2), prob in legs_dist.items():
            # P1 covers if (s1 - s2) > -handicap_legs
            # i.e. P1 wins by more than |handicap_legs| if conceding legs
            # or by at least 1 if being given legs
            adjusted_s1 = s1 + handicap_legs  # virtual P1 score after handicap
            if adjusted_s1 > s2:
                p1_covers += prob
            elif adjusted_s1 < s2:
                p2_covers += prob
            # Push (tie on adjusted scores): distribute 50/50
            else:
                p1_covers += prob * 0.5
                p2_covers += prob * 0.5

        logger.debug(
            "price_handicap",
            format=fmt.code,
            round=round_name,
            handicap_legs=handicap_legs,
            p1_covers=round(p1_covers, 5),
            p2_covers=round(p2_covers, 5),
        )

        return {
            "p1_covers": round(p1_covers, 6),
            "p2_covers": round(p2_covers, 6),
            "handicap_legs": handicap_legs,
            "legs_distribution": {
                f"{s1}-{s2}": round(p, 8)
                for (s1, s2), p in legs_dist.items()
            },
        }

    def price_totals(
        self,
        hb: HoldBreakProbabilities,
        fmt: DartsCompetitionFormat,
        round_name: str,
        p1_starts: bool,
        total_line: float,
    ) -> dict:
        """
        P(total legs in match > total_line).

        Parameters
        ----------
        hb:
            Hold/break probabilities.
        fmt:
            Competition format.
        round_name:
            Round name.
        p1_starts:
            Whether P1 starts first.
        total_line:
            Over/under line for total legs (e.g. 8.5).

        Returns
        -------
        dict:
            ``over_prob``, ``under_prob``, ``line``,
            ``over_decimal``, ``under_decimal``.
        """
        match_result = self.price_match(
            hold_break=hb,
            fmt=fmt,
            round_name=round_name,
            p1_starts_first=p1_starts,
        )

        legs_dist = match_result.legs_distribution

        if not legs_dist:
            logger.warning(
                "totals_empty_legs_distribution",
                format=fmt.code,
                round=round_name,
            )
            return {
                "over_prob": 0.5,
                "under_prob": 0.5,
                "line": total_line,
                "over_decimal": 2.0,
                "under_decimal": 2.0,
            }

        over_prob = 0.0
        under_prob = 0.0

        for (s1, s2), prob in legs_dist.items():
            total_legs = s1 + s2
            if total_legs > total_line:
                over_prob += prob
            else:
                under_prob += prob

        over_prob = max(0.0, min(1.0, over_prob))
        under_prob = max(0.0, min(1.0, under_prob))

        logger.debug(
            "price_totals",
            format=fmt.code,
            round=round_name,
            line=total_line,
            over_prob=round(over_prob, 5),
            under_prob=round(under_prob, 5),
        )

        return {
            "over_prob": round(over_prob, 6),
            "under_prob": round(under_prob, 6),
            "line": total_line,
            "over_decimal": round(1.0 / over_prob, 4) if over_prob > 1e-9 else None,
            "under_decimal": round(1.0 / under_prob, 4) if under_prob > 1e-9 else None,
        }

    def price_exact_scores(
        self,
        hb: HoldBreakProbabilities,
        fmt: DartsCompetitionFormat,
        round_name: str,
        p1_starts: bool,
    ) -> dict:
        """
        Full distribution over possible exact final leg scores.

        Returns probabilities for each reachable (P1_legs, P2_legs) score.

        Parameters
        ----------
        hb:
            Hold/break probabilities.
        fmt:
            Competition format.
        round_name:
            Round name.
        p1_starts:
            Whether P1 starts first.

        Returns
        -------
        dict:
            ``scores``: dict mapping "X-Y" → probability.
            ``most_likely``: the most probable exact score.
            ``total_prob``: sum of all score probabilities (should be ~1.0).
        """
        match_result = self.price_match(
            hold_break=hb,
            fmt=fmt,
            round_name=round_name,
            p1_starts_first=p1_starts,
        )

        legs_dist = match_result.legs_distribution

        if not legs_dist:
            logger.warning(
                "exact_scores_empty_distribution",
                format=fmt.code,
                round=round_name,
            )
            return {
                "scores": {},
                "most_likely": None,
                "total_prob": 0.0,
            }

        # Format as string keys for JSON compatibility
        scores = {
            f"{s1}-{s2}": round(p, 8)
            for (s1, s2), p in sorted(
                legs_dist.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }

        total_prob = sum(scores.values())
        most_likely = max(scores, key=scores.get) if scores else None

        logger.debug(
            "price_exact_scores",
            format=fmt.code,
            round=round_name,
            n_scores=len(scores),
            most_likely=most_likely,
            most_likely_prob=scores.get(most_likely, 0.0) if most_likely else 0.0,
            total_prob=round(total_prob, 6),
        )

        return {
            "scores": scores,
            "most_likely": most_likely,
            "total_prob": round(total_prob, 6),
            "p1_win": round(match_result.p1_win, 6),
            "p2_win": round(match_result.p2_win, 6),
        }
