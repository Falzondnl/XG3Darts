"""
Two-clear-legs engine for World Matchplay and similar formats.

Rules:
  - Must win by 2 clear legs.
  - If tied at (N-1, N-1) (or at the nominal target), play continues.
  - Sudden death after max_extra_legs (if configured), or play indefinitely.

Used for the PDC World Matchplay Final (first to 18 legs, must win by 2 clear,
no cap on extra legs).

DP state space: (legs_p1, legs_p2, current_thrower)
Terminal conditions:
  |l1 - l2| >= 2 AND max(l1, l2) >= nominal → winner = player with more legs
  OR legs exhausted (max_extra reached) → sudden death rule
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog

from competition.format_registry import DartsRoundFormat
from engines.leg_layer.markov_chain import HoldBreakProbabilities

logger = structlog.get_logger(__name__)


@dataclass
class WorldMatchplayResult:
    """Result from the two-clear-legs engine."""

    p1_win: float
    p2_win: float
    legs_distribution: dict[tuple[int, int], float]


class WorldMatchplayEngine:
    """
    Two-clear-legs engine for World Matchplay format.

    Key rules:
    1. Normal play: first to reach nominal target (legs_to_win) wins.
    2. If scores are tied at (N-1, N-1) where N = legs_to_win, continue.
    3. Continue until one player leads by 2 clear legs (two_clear_legs rule).
    4. If two_clear_legs_max_extra is set, sudden death after that many extra legs.

    The DP enumerates states beyond the nominal target up to a truncation limit.
    """

    # Maximum additional legs to simulate beyond nominal target
    # (in practice, very long ties are astronomically unlikely)
    MAX_EXTRA_LEGS: int = 40

    def price_match(
        self,
        round_fmt: DartsRoundFormat,
        hb: HoldBreakProbabilities,
        p1_starts: bool,
    ) -> dict:
        """
        Compute match win probabilities for a two-clear-legs format.

        Parameters
        ----------
        round_fmt:
            The round format definition. Must have two_clear_legs=True.
        hb:
            Hold/break probabilities.
        p1_starts:
            Whether P1 starts the first leg.

        Returns
        -------
        dict with keys: "p1_win", "p2_win", "legs_distribution"
        """
        assert round_fmt.two_clear_legs, "WorldMatchplayEngine requires two_clear_legs=True"
        assert round_fmt.legs_to_win is not None, "legs_to_win must be set for two-clear-legs format"

        nominal = round_fmt.legs_to_win
        max_extra = (
            round_fmt.two_clear_legs_max_extra
            if round_fmt.two_clear_legs_max_extra is not None
            else self.MAX_EXTRA_LEGS
        )

        result = self._solve_two_clear_dp(
            nominal=nominal,
            max_extra=max_extra,
            p1_hold=hb.p1_hold,
            p1_break=hb.p1_break,
            p2_hold=hb.p2_hold,
            p2_break=hb.p2_break,
            p1_starts=p1_starts,
        )

        logger.info(
            "world_matchplay_priced",
            nominal=nominal,
            max_extra=max_extra,
            p1_win=round(result["p1_win"], 4),
            p2_win=round(result["p2_win"], 4),
        )

        return result

    def _solve_two_clear_dp(
        self,
        nominal: int,
        max_extra: int,
        p1_hold: float,
        p1_break: float,
        p2_hold: float,
        p2_break: float,
        p1_starts: bool,
    ) -> dict:
        """
        Full DP solver for two-clear-legs format.

        States: (l1, l2, thrower_idx)

        Terminal conditions:
          1. l1 >= nominal and l1 - l2 >= 2  → P1 wins
          2. l2 >= nominal and l2 - l1 >= 2  → P2 wins
          3. max(l1, l2) >= nominal + max_extra → sudden death rules apply

        For sudden death (max_extra reached):
          If l1 > l2: P1 wins (they lead, count the advantage)
          If l2 > l1: P2 wins
          If tied: 50/50 (coin flip) — but in practice the problem is
          formulated such that we never reach an exact tie at max_extra
          unless both are playing equally.

        The computation is iterated via forward DP (state-transition enumeration).
        """
        # Maximum legs state we need to track
        max_legs = nominal + max_extra + 2

        # dp[(l1, l2, thrower)] = probability of being in this state
        # We enumerate forward from (0, 0, initial_thrower)
        state_probs: dict[tuple[int, int, int], float] = {}
        initial_thrower = 0 if p1_starts else 1
        state_probs[(0, 0, initial_thrower)] = 1.0

        p1_win_prob = 0.0
        p2_win_prob = 0.0
        legs_dist: dict[tuple[int, int], float] = {}

        # Process states in order of total legs played
        for total_legs in range(max_legs * 2 + 1):
            # Collect states at this total_legs level
            current_states = {
                (l1, l2, t): p
                for (l1, l2, t), p in state_probs.items()
                if l1 + l2 == total_legs and p > 0
            }

            for (l1, l2, thrower), prob in current_states.items():
                if prob < 1e-12:
                    continue

                # Check terminal conditions
                terminated, winner = self._is_terminal(l1, l2, nominal, max_extra)

                if terminated:
                    if winner == 1:
                        p1_win_prob += prob
                        key = (l1, l2)
                        legs_dist[key] = legs_dist.get(key, 0.0) + prob
                    elif winner == 2:
                        p2_win_prob += prob
                        key = (l1, l2)
                        legs_dist[key] = legs_dist.get(key, 0.0) + prob
                    else:
                        # Sudden death tie — 50/50
                        p1_win_prob += prob * 0.5
                        p2_win_prob += prob * 0.5
                        key = (l1, l2)
                        legs_dist[key] = legs_dist.get(key, 0.0) + prob
                    continue

                # Transition: play the next leg
                if thrower == 0:
                    p_p1_wins_leg = p1_hold
                    p_p2_wins_leg = p2_break
                else:
                    p_p1_wins_leg = p1_break
                    p_p2_wins_leg = p2_hold

                next_thrower = 1 - thrower

                # P1 wins this leg
                new_key_p1 = (l1 + 1, l2, next_thrower)
                state_probs[new_key_p1] = state_probs.get(new_key_p1, 0.0) + prob * p_p1_wins_leg

                # P2 wins this leg
                new_key_p2 = (l1, l2 + 1, next_thrower)
                state_probs[new_key_p2] = state_probs.get(new_key_p2, 0.0) + prob * p_p2_wins_leg

        # Normalise in case of floating point drift
        total = p1_win_prob + p2_win_prob
        if total > 0:
            p1_win_prob /= total
            p2_win_prob /= total

        return {
            "p1_win": p1_win_prob,
            "p2_win": p2_win_prob,
            "legs_distribution": legs_dist,
        }

    @staticmethod
    def _is_terminal(
        l1: int,
        l2: int,
        nominal: int,
        max_extra: int,
    ) -> tuple[bool, int]:
        """
        Check if a state is terminal.

        Returns
        -------
        (is_terminal, winner):
            winner = 1 (P1 wins), 2 (P2 wins), 0 (sudden death tie), -1 (not terminal)
        """
        # P1 wins: has nominal or more legs AND leads by at least 2
        if l1 >= nominal and (l1 - l2) >= 2:
            return True, 1

        # P2 wins: has nominal or more legs AND leads by at least 2
        if l2 >= nominal and (l2 - l1) >= 2:
            return True, 2

        # Max extra legs exhausted — sudden death
        max_legs_played = max(l1, l2)
        if max_legs_played >= nominal + max_extra:
            if l1 > l2:
                return True, 1
            elif l2 > l1:
                return True, 2
            else:
                return True, 0  # exact tie → coin flip

        return False, -1
