"""
Sets-based format engine (PDC_WC, PDC_GP, WDF_WC, etc.).

Handles nested structure: match = first-to-N-sets, each set = first-to-M-legs.

State space:
  (sets_p1, sets_p2, legs_in_set_p1, legs_in_set_p2, leg_starter)

Alternating starts:
  - Within a set: leg starter alternates each leg.
  - Between sets: the player who LOST the last leg of the previous set
    (i.e. the winner of the set) does NOT restart — PDC rule is that
    the player who did NOT start the first leg of the previous set
    starts the first leg of the next set. In practice, alternating starts
    means leg_starter alternates strictly regardless of set boundaries
    in most PDC formats.

Implementation note: We propagate the leg_starter through set boundaries
using the same alternating rule (last leg's next_starter becomes first
leg of new set's starter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog

from engines.leg_layer.markov_chain import HoldBreakProbabilities

logger = structlog.get_logger(__name__)


class SetsEngine:
    """
    Sets-based format combinatorial engine.

    Handles all PDC/WDF formats with sets structure:
    - PDC World Championship (sets_to_win=3..7, legs_per_set=3)
    - PDC Grand Prix (sets_to_win=3..4, legs_per_set=3)
    - WDF World Championship (sets_to_win=3..5, legs_per_set=3)
    - PDC World Series Finals (sets_to_win=3..5, legs_per_set=5)

    Algorithm:
      Forward DP enumeration over (s1, s2, l1, l2, starter) state space.
      Terminal: s1 == sets_to_win → P1 wins; s2 == sets_to_win → P2 wins.
      Set terminal: l1 == legs_per_set → P1 wins set; l2 == legs_per_set → P2 wins set.
    """

    def price_match(
        self,
        hb: HoldBreakProbabilities,
        sets_to_win: int,
        legs_per_set: int,
        p1_starts: bool,
    ) -> dict:
        """
        Compute match win probabilities for a sets-based format.

        Parameters
        ----------
        hb:
            Hold/break probabilities.
        sets_to_win:
            Number of sets required to win the match.
        legs_per_set:
            Number of legs required to win one set.
        p1_starts:
            Whether P1 starts the first leg of the match.

        Returns
        -------
        dict with keys: "p1_win", "p2_win", "legs_distribution", "sets_distribution"
        """
        result = self._dp_sets(
            hb=hb,
            sets_to_win=sets_to_win,
            legs_per_set=legs_per_set,
            p1_starts=p1_starts,
        )

        logger.info(
            "sets_format_priced",
            sets_to_win=sets_to_win,
            legs_per_set=legs_per_set,
            p1_win=round(result["p1_win"], 4),
            p2_win=round(result["p2_win"], 4),
        )

        return result

    def _dp_sets(
        self,
        hb: HoldBreakProbabilities,
        sets_to_win: int,
        legs_per_set: int,
        p1_starts: bool,
    ) -> dict:
        """
        Forward DP over (s1, s2, l1, l2, leg_starter) state.

        State meaning:
          s1, s2: sets won by P1 and P2 so far
          l1, l2: legs won by P1 and P2 in the CURRENT set
          leg_starter: who starts the current leg (0=P1, 1=P2)

        Transition:
          Play one leg:
            P(P1 wins leg | starter) = p1_hold (starter=P1) or p1_break (starter=P2)
          If P1 wins leg: l1 += 1
            If l1 == legs_per_set: s1 += 1, reset l1=l2=0, advance starter
          Etc.
        """
        N_sets = sets_to_win
        M_legs = legs_per_set

        # state: (s1, s2, l1, l2, starter) → probability
        state_probs: dict[tuple[int, int, int, int, int], float] = {}
        initial_starter = 0 if p1_starts else 1
        state_probs[(0, 0, 0, 0, initial_starter)] = 1.0

        p1_win_prob = 0.0
        p2_win_prob = 0.0
        sets_dist: dict[tuple[int, int], float] = {}

        # Maximum iterations: (2*N_sets - 1) sets * (2*M_legs - 1) legs per set
        max_total_legs = (2 * N_sets - 1) * (2 * M_legs - 1)

        for iteration in range(max_total_legs + 10):
            # Process all current non-terminal states
            new_state_probs: dict[tuple[int, int, int, int, int], float] = {}
            progress = False

            for (s1, s2, l1, l2, starter), prob in state_probs.items():
                if prob < 1e-14:
                    continue

                # Terminal check
                if s1 == N_sets:
                    p1_win_prob += prob
                    key = (s1, s2)
                    sets_dist[key] = sets_dist.get(key, 0.0) + prob
                    continue

                if s2 == N_sets:
                    p2_win_prob += prob
                    key = (s1, s2)
                    sets_dist[key] = sets_dist.get(key, 0.0) + prob
                    continue

                # Play one leg
                progress = True
                if starter == 0:
                    p_p1 = hb.p1_hold
                    p_p2 = hb.p2_break
                else:
                    p_p1 = hb.p1_break
                    p_p2 = hb.p2_hold

                next_leg_starter = 1 - starter

                # P1 wins leg
                new_l1 = l1 + 1
                if new_l1 == M_legs:
                    # P1 wins set
                    new_s1 = s1 + 1
                    new_key = (new_s1, s2, 0, 0, next_leg_starter)
                else:
                    new_key = (s1, s2, new_l1, l2, next_leg_starter)
                new_state_probs[new_key] = new_state_probs.get(new_key, 0.0) + prob * p_p1

                # P2 wins leg
                new_l2 = l2 + 1
                if new_l2 == M_legs:
                    # P2 wins set
                    new_s2 = s2 + 1
                    new_key = (s1, new_s2, 0, 0, next_leg_starter)
                else:
                    new_key = (s1, s2, l1, new_l2, next_leg_starter)
                new_state_probs[new_key] = new_state_probs.get(new_key, 0.0) + prob * p_p2

            if not progress:
                break

            # Merge terminal absorptions
            state_probs = new_state_probs

        # Any remaining probability in state_probs → enumerate remaining
        # (shouldn't happen with correct max_total_legs)
        for (s1, s2, l1, l2, starter), prob in state_probs.items():
            if prob < 1e-12:
                continue
            if s1 == N_sets:
                p1_win_prob += prob
            elif s2 == N_sets:
                p2_win_prob += prob
            else:
                # Unresolved: allocate proportionally to current leg advantage
                if l1 > l2 or (l1 == l2 and s1 > s2):
                    p1_win_prob += prob
                else:
                    p2_win_prob += prob

        # Normalise
        total = p1_win_prob + p2_win_prob
        if total > 0 and abs(total - 1.0) > 1e-8:
            p1_win_prob /= total
            p2_win_prob /= total

        return {
            "p1_win": p1_win_prob,
            "p2_win": p2_win_prob,
            "sets_distribution": sets_dist,
            "legs_distribution": {},  # full legs breakdown requires additional tracking
        }

    def p1_wins_set(
        self,
        hb: HoldBreakProbabilities,
        legs_per_set: int,
        p1_starts: bool,
    ) -> float:
        """
        P(P1 wins a single set | P1 starts first leg of set).

        Convenience method for set-level probability computation.
        """
        state_probs: dict[tuple[int, int, int], float] = {}
        initial_starter = 0 if p1_starts else 1
        state_probs[(0, 0, initial_starter)] = 1.0

        p1_win = 0.0
        p2_win = 0.0

        for _ in range((legs_per_set * 2) + 5):
            new_states: dict[tuple[int, int, int], float] = {}
            progress = False

            for (l1, l2, starter), prob in state_probs.items():
                if prob < 1e-14:
                    continue
                if l1 == legs_per_set:
                    p1_win += prob
                    continue
                if l2 == legs_per_set:
                    p2_win += prob
                    continue

                progress = True
                if starter == 0:
                    p_p1 = hb.p1_hold
                    p_p2 = hb.p2_break
                else:
                    p_p1 = hb.p1_break
                    p_p2 = hb.p2_hold

                next_starter = 1 - starter
                k_p1 = (l1 + 1, l2, next_starter)
                k_p2 = (l1, l2 + 1, next_starter)
                new_states[k_p1] = new_states.get(k_p1, 0.0) + prob * p_p1
                new_states[k_p2] = new_states.get(k_p2, 0.0) + prob * p_p2

            if not progress:
                break
            state_probs = new_states

        total = p1_win + p2_win
        if total > 0:
            return p1_win / total
        return 0.5
