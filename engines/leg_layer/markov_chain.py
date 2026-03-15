"""
501 Markov chain — the irreducible mathematical core of darts pricing.

States: 0..501
Transitions: P(new_score | current_score, player, context)

The visit distribution at each state is:
  - State-conditional (5 score bands)
  - Player-specific (from DartsOrakel stats or DartConnect)
  - Hierarchically pooled when data sparse

Architecture:
  1. build_transition_matrix(player, dist) → compute P[s', s] for each score state
  2. hold_probability(player_dist, checkout_model) → V[501] from Bellman recursion
  3. break_probability(p1_dist, p2_dist) → full HoldBreakProbabilities
  4. simulate_leg(p1_dist, p2_dist, starter) → LegResult via Markov sampling

Mathematical basis:
  The hold probability is computed via a single-player backward induction.
  The competitive hold (P wins before opponent) is then derived from the
  single-player "time-to-finish" distribution using a convolutional approach:

    P(P1 wins | P1 starts) = sum_k P(P1 finishes on visit k AND P2 hasn't finished yet)

  where "visit k" means k-th 3-dart visit.
  This is exact for independent players and is O(N*V) per player.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

from engines.leg_layer.checkout_model import CheckoutModel
from engines.leg_layer.visit_distributions import (
    ConditionalVisitDistribution,
    HierarchicalVisitDistributionModel,
    score_to_band,
)
from engines.state_layer.score_state import ScoreState

logger = structlog.get_logger(__name__)

# Invariant tolerance for Markov validation (G1)
MARKOV_TOTAL_TOLERANCE: float = 1e-3

# Maximum visits to consider in the finish-time distribution
MAX_VISITS_DISTRIBUTION: int = 120

# ---------------------------------------------------------------------------
# Module-level caches for finish-time distributions and transition matrices.
#
# Keyed by (three_da_rounded_to_1dp, starting_score).
# Valid only for prior-only distributions (data_source in {"derived", "pooled"}).
# Bypassed automatically when DartConnect visit-level data is present.
# ---------------------------------------------------------------------------
_FINISH_DIST_CACHE: dict[tuple[float, int], np.ndarray] = {}
_TRANSITION_MAT_CACHE: dict[tuple[float, int], np.ndarray] = {}


@dataclass
class LegResult:
    """
    Result of one simulated leg.

    Attributes
    ----------
    winner:
        0 = Player 1 won, 1 = Player 2 won.
    total_darts:
        Total darts thrown in the leg (both players combined).
    visits_p1:
        Number of visits player 1 threw.
    visits_p2:
        Number of visits player 2 threw.
    final_score_p1:
        Score remaining for P1 when leg ended (0 if P1 won).
    final_score_p2:
        Score remaining for P2 when leg ended (0 if P2 won).
    starter:
        Which player started (0 or 1).
    """

    winner: int
    total_darts: int
    visits_p1: int
    visits_p2: int
    final_score_p1: int
    final_score_p2: int
    starter: int


@dataclass
class HoldBreakProbabilities:
    """
    Hold/break probabilities for a two-player match.

    All four probabilities are fully derived from the Markov chain.
    Never hardcoded.

    Invariants:
        p1_hold + p2_break = 1.0   (G3)
        p2_hold + p1_break = 1.0   (G3)
    """

    p1_hold: float    # P(P1 wins leg | P1 starts)
    p1_break: float   # P(P1 wins leg | P2 starts)
    p2_hold: float    # P(P2 wins leg | P2 starts)
    p2_break: float   # P(P2 wins leg | P1 starts)

    def consistency_check(self, tol: float = 1e-6) -> bool:
        """
        Verify the fundamental hold/break consistency constraint (G3).

        p1_hold + p2_break = 1.0 (when P1 starts, one of them wins)
        p2_hold + p1_break = 1.0 (when P2 starts, one of them wins)
        """
        c1 = abs(self.p1_hold + self.p2_break - 1.0) <= tol
        c2 = abs(self.p2_hold + self.p1_break - 1.0) <= tol
        if not (c1 and c2):
            logger.warning(
                "hold_break_consistency_violation",
                p1_hold=self.p1_hold,
                p2_break=self.p2_break,
                p2_hold=self.p2_hold,
                p1_break=self.p1_break,
                diff_c1=abs(self.p1_hold + self.p2_break - 1.0),
                diff_c2=abs(self.p2_hold + self.p1_break - 1.0),
            )
        return c1 and c2

    def to_match_win_prob(
        self,
        p1_starts_first: bool,
        legs_to_win: int,
    ) -> float:
        """
        Compute P1 match win probability via DP over leg states.

        Uses the dynamic programming combinatorics from MatchCombinatorialEngine.
        """
        from engines.match_layer.match_combinatorics import MatchCombinatorialEngine
        engine = MatchCombinatorialEngine()
        result = engine._dp_legs_format(
            hb=self,
            legs_to_win=legs_to_win,
            p1_starts=p1_starts_first,
        )
        return result["p1_win"]

    def validate(self) -> None:
        """
        Validate that all probabilities are in [0, 1] and consistency holds.

        Raises
        ------
        ValueError
            If any constraint is violated.
        """
        for name, val in [
            ("p1_hold", self.p1_hold),
            ("p1_break", self.p1_break),
            ("p2_hold", self.p2_hold),
            ("p2_break", self.p2_break),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name}={val} is not in [0, 1]")
        if not self.consistency_check():
            raise ValueError(
                f"Hold/break consistency violated: "
                f"p1_hold={self.p1_hold:.6f}, p2_break={self.p2_break:.6f}, "
                f"p2_hold={self.p2_hold:.6f}, p1_break={self.p1_break:.6f}"
            )


class DartsMarkovChain:
    """
    Markov chain on score states 0..501.

    Core algorithm: backward induction for the single-player absorption problem.

    For a single player starting at score S, define:
      f(s, k) = P(first checkout at exactly visit k | start at score s)

    This is computed via forward convolution starting from V[s]:
      V[s] = P(player finishes from score s on any future visit)
           = 1.0 for s <= 170 (eventually reachable)
           = (varies by player quality)

    For the competitive hold problem, we use the finish-time distribution:
      P(P1 holds | P1 starts) = sum_{k=1}^{inf} f_A(k) * P(B hasn't finished in k-1 visits)

    where f_A(k) = P(P1 finishes on exactly visit k) and P(B not finished in j visits)
    is derived from P2's finish-time distribution.
    """

    def __init__(self) -> None:
        self._checkout_model = CheckoutModel()
        self._visit_model = HierarchicalVisitDistributionModel()
        self._transition_cache: dict[str, np.ndarray] = {}

    def _build_effective_visit_distribution(
        self,
        score: int,
        visit_dists: dict[str, ConditionalVisitDistribution],
        player_id: str,
        three_da: float,
    ) -> tuple[dict[int, float], float]:
        """
        Get the effective visit distribution for a given score.

        Returns (visit_probs, bust_prob) where:
          - For scores <= 170: checkout probability is explicitly handled
          - For scores > 170: use the band distribution directly
          - The checkout model overrides the visit distribution for checkout territory

        Returns
        -------
        (visit_probs, bust_prob):
            visit_probs maps score_reduction → probability
            bust_prob = probability of busting this visit
        """
        band = score_to_band(score)
        if band in visit_dists:
            dist = visit_dists[band]
        else:
            dist = self._visit_model.get_distribution(
                player_id=player_id,
                score_band=band,
                stage=False,
                short_format=False,
                throw_first=True,
                three_da=three_da,
            )

        visit_probs = dict(dist.visit_probs)
        bust_prob = dist.bust_prob

        # For checkout territory: explicitly add checkout probability
        if score <= 170 and score > 1:
            checkout_result = self._checkout_model.p_checkout_visit(
                player_id=player_id,
                score=score,
                darts_available=3,
                three_da=three_da,
                pressure=(band == "pressure"),
            )
            if checkout_result.p_checkout > 1e-6:
                # Add checkout as a transition to score 0
                # Remove any existing score == score entry (can't reduce by own score to 0
                # differently) and replace with explicit checkout prob
                existing_checkout = visit_probs.pop(score, 0.0)
                visit_probs[score] = max(0.0, checkout_result.p_checkout)
                # Adjust bust prob to ensure sum = 1
                new_total = sum(visit_probs.values()) + bust_prob
                if abs(new_total - 1.0) > 1e-6:
                    # Normalise
                    scale = 1.0 / new_total
                    visit_probs = {k: v * scale for k, v in visit_probs.items()}
                    bust_prob *= scale

        # Clean: remove entries where score reduction exceeds current score (bust)
        valid_probs: dict[int, float] = {}
        excess_bust = 0.0
        for reduction, prob in visit_probs.items():
            if reduction > score:
                excess_bust += prob  # over-score → bust
            elif reduction < 0:
                excess_bust += prob
            else:
                valid_probs[reduction] = valid_probs.get(reduction, 0.0) + prob

        bust_prob += excess_bust
        visit_probs = valid_probs

        # Final normalisation
        total = sum(visit_probs.values()) + bust_prob
        if total > 0 and abs(total - 1.0) > 1e-9:
            scale = 1.0 / total
            visit_probs = {k: v * scale for k, v in visit_probs.items()}
            bust_prob *= scale

        return visit_probs, bust_prob

    def _compute_score_transition_probs(
        self,
        score: int,
        visit_dists: dict[str, ConditionalVisitDistribution],
        player_id: str,
        three_da: float,
    ) -> dict[int, float]:
        """
        Compute P(new_score | current_score) for one visit.

        Returns dict mapping new_score → probability.
        Bust = stays at same score.
        Checkout = new_score = 0.
        """
        visit_probs, bust_prob = self._build_effective_visit_distribution(
            score=score,
            visit_dists=visit_dists,
            player_id=player_id,
            three_da=three_da,
        )

        transitions: dict[int, float] = {}

        # Bust: stay at same score
        if bust_prob > 0:
            transitions[score] = transitions.get(score, 0.0) + bust_prob

        # Scoring visits
        for reduction, prob in visit_probs.items():
            new_score = score - reduction
            if new_score < 0:
                # This shouldn't happen after _build_effective_visit_distribution
                # but handle defensively
                transitions[score] = transitions.get(score, 0.0) + prob
            elif new_score == 1:
                # Score of 1: bad leave (can't checkout), opponent gets turn
                # From P's perspective, being stuck at 1 is the same as 0 progress
                transitions[1] = transitions.get(1, 0.0) + prob
            else:
                transitions[new_score] = transitions.get(new_score, 0.0) + prob

        # Normalise
        total = sum(transitions.values())
        if total > 0 and abs(total - 1.0) > 1e-9:
            transitions = {k: v / total for k, v in transitions.items()}

        return transitions

    def _build_transition_matrix(
        self,
        visit_dists: dict[str, ConditionalVisitDistribution],
        player_id: str,
        three_da: float,
        starting_score: int,
    ) -> np.ndarray:
        """
        Build a dense numpy transition matrix T of shape (N, N).

        T[new_s, s] = P(transition from score s to score new_s in one visit).

        State 0 = checkout (absorbing sink — callers zero it out after extracting).
        State 1 = stuck (absorbing: self-loop).
        States 2..starting_score = active play.

        This matrix is the core of the vectorized forward simulation.
        """
        N = starting_score + 1
        T = np.zeros((N, N), dtype=np.float64)

        for s in range(2, N):
            trans = self._compute_score_transition_probs(
                score=s,
                visit_dists=visit_dists,
                player_id=player_id,
                three_da=three_da,
            )
            for new_s, prob in trans.items():
                if 0 <= new_s < N:
                    T[new_s, s] += prob

        # State 1: stuck — self-loop
        T[1, 1] = 1.0
        # State 0: do NOT make absorbing — caller extracts T[0,:] @ state and zeros it

        return T

    def _compute_visit_finish_distribution(
        self,
        visit_dists: dict[str, ConditionalVisitDistribution],
        player_id: str,
        three_da: float,
        starting_score: int = 501,
        max_visits: int = MAX_VISITS_DISTRIBUTION,
    ) -> np.ndarray:
        """
        Compute f[k] = P(player finishes leg on exactly visit k).

        This is the finish-time PMF for a single player starting at starting_score.
        Uses forward simulation of the Markov chain with numpy vectorization:

          state_prob[s] = P(player is at score s after k complete visits)
          f[k] = T[0, :] @ state_prob  (checkout probability this step)

        The transition matrix T is built once per (three_da, starting_score) pair
        and cached at module level for prior-only distributions.

        This is O(max_visits * N^2) with BLAS matrix-vector multiply — typically
        100x faster than the pure-Python dict loop it replaces.

        Caching: bypassed if any distribution uses DartConnect data
        (data_source == "dartconnect"), because those distributions are
        player-specific and cannot be shared across players.

        Returns
        -------
        np.ndarray of shape (max_visits+1,)
            f[0] = 0, f[k] = P(finishes on visit k) for k >= 1.
        """
        N = starting_score + 1

        # Determine whether all distributions are prior-only (cacheable).
        prior_only = all(
            d.data_source in {"derived", "pooled"}
            for d in visit_dists.values()
        )
        cache_key = (round(three_da, 1), starting_score)

        if prior_only and cache_key in _FINISH_DIST_CACHE:
            return _FINISH_DIST_CACHE[cache_key]

        # Build or retrieve the transition matrix.
        if prior_only and cache_key in _TRANSITION_MAT_CACHE:
            T = _TRANSITION_MAT_CACHE[cache_key]
        else:
            T = self._build_transition_matrix(
                visit_dists=visit_dists,
                player_id=player_id,
                three_da=three_da,
                starting_score=starting_score,
            )
            if prior_only:
                _TRANSITION_MAT_CACHE[cache_key] = T

        # Forward simulation using numpy matrix-vector multiply.
        f = np.zeros(max_visits + 1, dtype=np.float64)
        state_prob = np.zeros(N, dtype=np.float64)
        state_prob[starting_score] = 1.0

        # Checkout row: T[0, s] = probability of checking out from state s.
        checkout_row = T[0, :]

        for k in range(1, max_visits + 1):
            # f[k] = probability of finishing exactly on visit k.
            newly_finished = float(checkout_row @ state_prob)
            f[k] = max(0.0, newly_finished)

            # Advance state distribution; do not let probability pool at state 0.
            new_state_prob = T @ state_prob
            new_state_prob[0] = 0.0  # remove absorbed probability

            state_prob = new_state_prob

            # Early termination when virtually all mass is absorbed.
            remaining = float(state_prob[2:].sum()) + float(state_prob[1])
            if remaining < 1e-8:
                break

        f = np.maximum(f, 0.0)

        logger.debug(
            "finish_distribution_computed",
            player=player_id,
            starting_score=starting_score,
            total_prob=round(float(f.sum()), 6),
            expected_visits=round(float(np.sum(np.arange(len(f)) * f)), 2),
        )

        if prior_only:
            _FINISH_DIST_CACHE[cache_key] = f

        return f

    def hold_probability(
        self,
        player_visit_dists: dict[str, ConditionalVisitDistribution],
        player_id: str,
        three_da: float,
        starting_score: int = 501,
        double_start: bool = False,
    ) -> float:
        """
        P(player holds = wins leg when starting from starting_score).

        Computed via finish-time distribution:
        P(holds) = P(player finishes before opponent starts)
        In the single-player formulation:
        P(player eventually finishes) = sum_k f[k]
        """
        f = self._compute_visit_finish_distribution(
            visit_dists=player_visit_dists,
            player_id=player_id,
            three_da=three_da,
            starting_score=starting_score,
        )
        return min(1.0, float(f.sum()))

    def break_probability(
        self,
        p1_visit_dists: dict[str, ConditionalVisitDistribution],
        p2_visit_dists: dict[str, ConditionalVisitDistribution],
        p1_id: str,
        p2_id: str,
        p1_three_da: float,
        p2_three_da: float,
        starting_score: int = 501,
        double_start: bool = False,
    ) -> HoldBreakProbabilities:
        """
        Compute all four hold/break probabilities from visit distributions.

        Uses the finish-time distribution approach:
          P(P1 wins | P1 starts) = P(P1 finishes on visit k) * P(P2 hasn't finished in k-1 visits)
          summed over all k.

        This is exact for independent players and efficient: O(max_visits * N) per player.
        """
        logger.info(
            "computing_hold_break",
            p1=p1_id,
            p2=p2_id,
            p1_three_da=p1_three_da,
            p2_three_da=p2_three_da,
            starting_score=starting_score,
        )

        # Compute finish-time distributions for both players
        f_p1 = self._compute_visit_finish_distribution(
            visit_dists=p1_visit_dists,
            player_id=p1_id,
            three_da=p1_three_da,
            starting_score=starting_score,
        )
        f_p2 = self._compute_visit_finish_distribution(
            visit_dists=p2_visit_dists,
            player_id=p2_id,
            three_da=p2_three_da,
            starting_score=starting_score,
        )

        # P(P1 holds | P1 starts first):
        # P1 finishes on visit k means P1 took k visits.
        # P2 had k-1 visits (P2 throws AFTER P1's first visit).
        # But for k=1: P2 hasn't thrown at all.
        # For k=2: P2 had 1 visit after P1's first.
        # Etc.
        #
        # Structure: P1 throws visit 1, P2 throws visit 1, P1 throws visit 2, ...
        # P1 wins if P1 finishes on THEIR visit k before P2 finishes on their visit j.
        # P1 throws visits: 1, 2, 3, ...
        # P2 throws visits: 1, 2, 3, ... (each P2 visit happens AFTER the corresponding P1 visit)
        #
        # P1 finishes on P1-visit k: P2 has thrown k-1 visits
        # Condition for P1 winning: P2 didn't finish in their k-1 visits
        #
        # P(P1 holds) = sum_{k=1}^{inf} f_p1[k] * G_p2(k-1)
        # where G_p2(j) = P(P2 hasn't finished in j visits) = 1 - sum_{m=1}^{j} f_p2[m]
        #                                                     = 1 - F_p2(j)

        max_k = len(f_p1) - 1  # f has indices 0..max_k

        # Cumulative distribution F_p2[j] = P(P2 finishes in <= j visits)
        F_p2 = np.cumsum(f_p2)  # F_p2[j] = sum_{m=0}^{j} f_p2[m]

        # P(P1 holds | P1 starts)
        p1_hold = 0.0
        for k in range(1, max_k + 1):
            if f_p1[k] < 1e-12:
                continue
            # P(P2 hasn't finished in k-1 visits)
            if k - 1 == 0:
                g_p2 = 1.0  # P2 has had 0 visits → certainly hasn't finished
            else:
                j = min(k - 1, len(F_p2) - 1)
                g_p2 = max(0.0, 1.0 - float(F_p2[j]))
            p1_hold += f_p1[k] * g_p2

        p1_hold = min(1.0, max(0.0, p1_hold))

        # Same calculation for P2 holding (P2 starts first):
        # P(P2 holds) = sum_{k=1}^{inf} f_p2[k] * G_p1(k-1)
        F_p1 = np.cumsum(f_p1)
        p2_hold = 0.0
        max_k2 = len(f_p2) - 1
        for k in range(1, max_k2 + 1):
            if f_p2[k] < 1e-12:
                continue
            if k - 1 == 0:
                g_p1 = 1.0
            else:
                j = min(k - 1, len(F_p1) - 1)
                g_p1 = max(0.0, 1.0 - float(F_p1[j]))
            p2_hold += f_p2[k] * g_p1

        p2_hold = min(1.0, max(0.0, p2_hold))

        # G3 constraints are enforced by construction:
        p2_break = 1.0 - p1_hold
        p1_break = 1.0 - p2_hold

        hb = HoldBreakProbabilities(
            p1_hold=p1_hold,
            p1_break=p1_break,
            p2_hold=p2_hold,
            p2_break=p2_break,
        )
        hb.validate()

        logger.info(
            "hold_break_computed",
            p1=p1_id,
            p2=p2_id,
            p1_hold=round(hb.p1_hold, 4),
            p1_break=round(hb.p1_break, 4),
            p2_hold=round(hb.p2_hold, 4),
            p2_break=round(hb.p2_break, 4),
        )

        return hb

    def simulate_leg(
        self,
        p1_visit_dists: dict[str, ConditionalVisitDistribution],
        p2_visit_dists: dict[str, ConditionalVisitDistribution],
        p1_id: str,
        p2_id: str,
        p1_three_da: float,
        p2_three_da: float,
        starter: int,  # 0=P1, 1=P2
        starting_score: int = 501,
        double_start: bool = False,
        max_visits: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> LegResult:
        """
        Simulate one leg via Markov chain Monte Carlo sampling.

        Returns winner, darts thrown, visit sequence.
        """
        if rng is None:
            rng = np.random.default_rng()

        scores = [starting_score, starting_score]
        visits = [0, 0]
        total_darts = 0
        current_player = starter

        all_dists = [p1_visit_dists, p2_visit_dists]
        ids = [p1_id, p2_id]
        three_das = [p1_three_da, p2_three_da]

        for _visit_num in range(max_visits):
            p = current_player
            score = scores[p]

            if score == 0:
                break

            # Sample from transition distribution
            trans = self._compute_score_transition_probs(
                score=score,
                visit_dists=all_dists[p],
                player_id=ids[p],
                three_da=three_das[p],
            )

            # Sample new score
            r = rng.random()
            cumulative = 0.0
            new_score = score  # default: bust

            trans_scores = list(trans.keys())
            trans_probs = list(trans.values())
            for ns, prob in zip(trans_scores, trans_probs):
                cumulative += prob
                if r < cumulative:
                    new_score = ns
                    break

            darts_this_visit = 3
            if new_score == 0:
                # Checkout - figure out darts used
                checkout_result = self._checkout_model.p_checkout_visit(
                    player_id=ids[p],
                    score=score,
                    darts_available=3,
                    three_da=three_das[p],
                    pressure=(score_to_band(score) == "pressure"),
                )
                if checkout_result.route:
                    darts_this_visit = checkout_result.route.darts_required

            scores[p] = new_score
            total_darts += darts_this_visit
            visits[p] += 1

            if new_score == 0:
                return LegResult(
                    winner=p,
                    total_darts=total_darts,
                    visits_p1=visits[0],
                    visits_p2=visits[1],
                    final_score_p1=scores[0],
                    final_score_p2=scores[1],
                    starter=starter,
                )

            current_player = 1 - p

        logger.warning("simulate_leg_max_visits_exceeded", max_visits=max_visits)
        winner = 0 if scores[0] <= scores[1] else 1
        return LegResult(
            winner=winner,
            total_darts=total_darts,
            visits_p1=visits[0],
            visits_p2=visits[1],
            final_score_p1=scores[0],
            final_score_p2=scores[1],
            starter=starter,
        )

    def validate_markov_totals(
        self,
        visit_dist: ConditionalVisitDistribution,
        tol: float = MARKOV_TOTAL_TOLERANCE,
    ) -> bool:
        """
        G1 invariant: verify that the visit distribution sums to 1.0 ± tol.
        """
        total = sum(visit_dist.visit_probs.values()) + visit_dist.bust_prob
        ok = abs(total - 1.0) <= tol
        if not ok:
            logger.error(
                "markov_total_violation_G1",
                player=visit_dist.player_id,
                band=visit_dist.score_band,
                total=total,
                deviation=abs(total - 1.0),
            )
        return ok
