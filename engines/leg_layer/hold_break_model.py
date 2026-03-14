"""
Hold/Break as primary latent variables.

P(holds) = P(wins leg when starting)
P(breaks) = P(wins leg when receiving)

DERIVED from Markov chain visit distributions — never hardcoded.

This module is the primary consumer of DartsMarkovChain and presents
a clean interface to the match layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog

from engines.leg_layer.markov_chain import DartsMarkovChain, HoldBreakProbabilities, LegResult
from engines.leg_layer.visit_distributions import (
    ConditionalVisitDistribution,
    HierarchicalVisitDistributionModel,
)

logger = structlog.get_logger(__name__)


@dataclass
class PlayerMatchupProfile:
    """
    Complete player profiling data required for hold/break computation.

    Attributes
    ----------
    player_id:
        Canonical player identifier.
    three_da:
        3-dart average from DartsOrakel or DartConnect.
    stage:
        True if the match is a televised stage event.
    short_format:
        True for short formats (e.g. first-to-3 legs).
    throw_first:
        Whether this player throws first (used for visit distribution selection).
    elo_rating:
        Current ELO rating (used for hierarchical pooling tier selection).
    """

    player_id: str
    three_da: float
    stage: bool = False
    short_format: bool = False
    throw_first: bool = True
    elo_rating: Optional[float] = None


class HoldBreakModel:
    """
    High-level interface for computing hold/break probabilities.

    Orchestrates:
      1. Visit distribution retrieval (hierarchical model)
      2. Markov chain computation
      3. Hold/break probability packaging

    All probabilities are derived from the Markov chain.
    Raises NotImplementedError when player-specific data is not wired.
    Uses 3DA-based prior as fallback.
    """

    def __init__(self) -> None:
        self._markov = DartsMarkovChain()
        self._visit_model = HierarchicalVisitDistributionModel()

    def compute(
        self,
        p1: PlayerMatchupProfile,
        p2: PlayerMatchupProfile,
        starting_score: int = 501,
        double_start: bool = False,
    ) -> HoldBreakProbabilities:
        """
        Compute full hold/break probabilities for a P1 vs P2 matchup.

        Parameters
        ----------
        p1:
            Player 1 profile.
        p2:
            Player 2 profile.
        starting_score:
            Starting score for each leg (501 standard, 701 for doubles).
        double_start:
            Whether double-in rule applies (Grand Prix format).

        Returns
        -------
        HoldBreakProbabilities
            All four probabilities derived from Markov chain computation.
        """
        logger.info(
            "computing_hold_break",
            p1=p1.player_id,
            p2=p2.player_id,
            p1_three_da=p1.three_da,
            p2_three_da=p2.three_da,
            starting_score=starting_score,
            double_start=double_start,
        )

        # Get visit distributions for both players
        p1_dists = self._visit_model.get_all_bands(
            player_id=p1.player_id,
            stage=p1.stage,
            short_format=p1.short_format,
            throw_first=p1.throw_first,
            three_da=p1.three_da,
        )
        p2_dists = self._visit_model.get_all_bands(
            player_id=p2.player_id,
            stage=p2.stage,
            short_format=p2.short_format,
            throw_first=p2.throw_first,
            three_da=p2.three_da,
        )

        # Validate all distributions (G1 invariant)
        for band, dist in p1_dists.items():
            if not self._markov.validate_markov_totals(dist):
                logger.warning("g1_violation_p1", band=band, player=p1.player_id)
        for band, dist in p2_dists.items():
            if not self._markov.validate_markov_totals(dist):
                logger.warning("g1_violation_p2", band=band, player=p2.player_id)

        hb = self._markov.break_probability(
            p1_visit_dists=p1_dists,
            p2_visit_dists=p2_dists,
            p1_id=p1.player_id,
            p2_id=p2.player_id,
            p1_three_da=p1.three_da,
            p2_three_da=p2.three_da,
            starting_score=starting_score,
            double_start=double_start,
        )

        logger.info(
            "hold_break_computed",
            p1=p1.player_id,
            p2=p2.player_id,
            p1_hold=round(hb.p1_hold, 4),
            p1_break=round(hb.p1_break, 4),
            p2_hold=round(hb.p2_hold, 4),
            p2_break=round(hb.p2_break, 4),
        )

        return hb

    def compute_from_3da(
        self,
        p1_id: str,
        p2_id: str,
        p1_three_da: float,
        p2_three_da: float,
        stage: bool = False,
        short_format: bool = False,
        starting_score: int = 501,
        double_start: bool = False,
    ) -> HoldBreakProbabilities:
        """
        Convenience method: compute hold/break from 3DA averages only.

        Uses the hierarchical prior when no player-specific data is available.
        All probabilities derived from Markov chain — never hardcoded.

        Parameters
        ----------
        p1_id / p2_id:
            Player IDs.
        p1_three_da / p2_three_da:
            3-dart averages.
        stage:
            Televised stage event flag.
        short_format:
            Short format flag.
        starting_score:
            501 for standard, 701 for doubles.
        double_start:
            Grand Prix double-in rule.
        """
        p1 = PlayerMatchupProfile(
            player_id=p1_id,
            three_da=p1_three_da,
            stage=stage,
            short_format=short_format,
            throw_first=True,
        )
        p2 = PlayerMatchupProfile(
            player_id=p2_id,
            three_da=p2_three_da,
            stage=stage,
            short_format=short_format,
            throw_first=False,
        )
        return self.compute(p1, p2, starting_score=starting_score, double_start=double_start)

    def simulate_legs(
        self,
        p1: PlayerMatchupProfile,
        p2: PlayerMatchupProfile,
        n_simulations: int,
        starter: int = 0,
        starting_score: int = 501,
        rng=None,
    ) -> list[LegResult]:
        """
        Simulate multiple legs via Markov chain Monte Carlo.

        Used for calibration and validation. Returns individual leg results.
        """
        import numpy as np
        if rng is None:
            rng = np.random.default_rng()

        p1_dists = self._visit_model.get_all_bands(
            player_id=p1.player_id,
            stage=p1.stage,
            short_format=p1.short_format,
            throw_first=p1.throw_first,
            three_da=p1.three_da,
        )
        p2_dists = self._visit_model.get_all_bands(
            player_id=p2.player_id,
            stage=p2.stage,
            short_format=p2.short_format,
            throw_first=p2.throw_first,
            three_da=p2.three_da,
        )

        results = []
        for _ in range(n_simulations):
            result = self._markov.simulate_leg(
                p1_visit_dists=p1_dists,
                p2_visit_dists=p2_dists,
                p1_id=p1.player_id,
                p2_id=p2.player_id,
                p1_three_da=p1.three_da,
                p2_three_da=p2.three_da,
                starter=starter,
                starting_score=starting_score,
                rng=rng,
            )
            results.append(result)

        return results
