"""
World Cup doubles team visit model.

In doubles darts (PDC World Cup), two players alternate throwing
within a team. The team visit distribution is derived from
the individual player distributions and the alternating throw order.

Structure:
  - Within each leg, players A and B alternate throwing (A throws, then B throws,
    then A, etc.) — NOT per dart but per VISIT.
  - The team "visit distribution" is the combined distribution of the pair.
  - Hold/break is computed as a modified Markov chain for team play.

Key model:
  Team 3DA ≈ (3DA_A + 3DA_B) / 2 (simplified; full model uses conditional dists).
  The team checkout model accounts for which player ends the leg.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import structlog

from engines.leg_layer.visit_distributions import (
    ConditionalVisitDistribution,
    HierarchicalVisitDistributionModel,
)

logger = structlog.get_logger(__name__)


@dataclass
class DoublesTeam:
    """
    A pair of players forming a doubles team.

    Attributes
    ----------
    team_id:
        Unique team identifier (usually country code for World Cup).
    player_a_id:
        Player A identifier (throws first in the team).
    player_b_id:
        Player B identifier (throws second).
    player_a_three_da:
        Player A's 3-dart average.
    player_b_three_da:
        Player B's 3-dart average.
    player_a_checkout_rate:
        Player A's checkout percentage.
    player_b_checkout_rate:
        Player B's checkout percentage.
    """

    team_id: str
    player_a_id: str
    player_b_id: str
    player_a_three_da: float
    player_b_three_da: float
    player_a_checkout_rate: float = 0.40
    player_b_checkout_rate: float = 0.40

    @property
    def team_three_da(self) -> float:
        """Team 3DA: average of both players."""
        return (self.player_a_three_da + self.player_b_three_da) / 2.0

    @property
    def combined_checkout_rate(self) -> float:
        """Effective team checkout rate."""
        # The player who finishes is typically the better checker
        # Simple approximation: max of both rates weighted by their 3DA
        total_3da = self.player_a_three_da + self.player_b_three_da
        if total_3da == 0:
            return (self.player_a_checkout_rate + self.player_b_checkout_rate) / 2.0
        w_a = self.player_a_three_da / total_3da
        w_b = self.player_b_three_da / total_3da
        return w_a * self.player_a_checkout_rate + w_b * self.player_b_checkout_rate


class TeamVisitModel:
    """
    Doubles team visit distribution model.

    Derives the effective team visit distribution by modelling the
    alternating visit structure between two players.

    The team leg is played as:
      Visit 1: Player A throws
      Visit 2: Player B throws
      Visit 3: Player A throws
      ...

    For the Markov chain, we aggregate this into an "effective team visit
    distribution" by:
    1. Computing the expected score per 2-visit cycle (A then B)
    2. Using the combined distribution as the team's effective distribution

    Checkout coordination:
    Either player can finish the leg on their visit. The effective checkout
    probability accounts for both players' opportunities.
    """

    def __init__(self) -> None:
        self._visit_model = HierarchicalVisitDistributionModel()

    def get_team_visit_distribution(
        self,
        team: DoublesTeam,
        score_band: str,
        stage: bool,
        short_format: bool,
        throw_first: bool,
    ) -> ConditionalVisitDistribution:
        """
        Derive team visit distribution for a given score band.

        The team distribution is computed as the convolution of player A
        and player B visit distributions, weighted by the visit alternation.

        For simplicity, we use the average distribution weighted by
        each player's effective throw share within the band.

        Parameters
        ----------
        team:
            DoublesTeam with player IDs and stats.
        score_band:
            One of "open", "middle", "setup", "finish", "pressure".
        stage:
            Stage event flag.
        short_format:
            Short format flag.
        throw_first:
            Whether this team throws first.

        Returns
        -------
        ConditionalVisitDistribution
            Effective team distribution.
        """
        dist_a = self._visit_model.get_distribution(
            player_id=team.player_a_id,
            score_band=score_band,
            stage=stage,
            short_format=short_format,
            throw_first=throw_first,
            three_da=team.player_a_three_da,
        )
        dist_b = self._visit_model.get_distribution(
            player_id=team.player_b_id,
            score_band=score_band,
            stage=stage,
            short_format=short_format,
            throw_first=not throw_first,  # B alternates
            three_da=team.player_b_three_da,
        )

        # Blend distributions 50/50 for the team (equal throw share)
        all_scores = set(dist_a.visit_probs.keys()) | set(dist_b.visit_probs.keys())
        blended_probs: dict[int, float] = {}
        for s in all_scores:
            p_a = dist_a.visit_probs.get(s, 0.0)
            p_b = dist_b.visit_probs.get(s, 0.0)
            blended_probs[s] = 0.5 * p_a + 0.5 * p_b

        blended_bust = 0.5 * dist_a.bust_prob + 0.5 * dist_b.bust_prob

        # Renormalise
        total = sum(blended_probs.values()) + blended_bust
        if total > 0:
            blended_probs = {s: p / total for s, p in blended_probs.items()}
            blended_bust /= total

        team_dist = ConditionalVisitDistribution(
            player_id=team.team_id,
            score_band=score_band,
            stage=stage,
            short_format=short_format,
            throw_first=throw_first,
            visit_probs=blended_probs,
            bust_prob=blended_bust,
            data_source="derived",
            n_observations=min(dist_a.n_observations, dist_b.n_observations),
            confidence=min(dist_a.confidence, dist_b.confidence),
        )
        team_dist.validate()
        return team_dist

    def get_all_bands_for_team(
        self,
        team: DoublesTeam,
        stage: bool,
        short_format: bool,
        throw_first: bool,
    ) -> dict[str, ConditionalVisitDistribution]:
        """Get team visit distributions for all five score bands."""
        from engines.leg_layer.visit_distributions import BAND_NAMES
        return {
            band: self.get_team_visit_distribution(
                team=team,
                score_band=band,
                stage=stage,
                short_format=short_format,
                throw_first=throw_first,
            )
            for band in BAND_NAMES
        }
