"""
Darts ELO rating pipeline.

Five independent ELO pools are maintained to avoid cross-contaminating ratings
between ecosystems:

- ``pdc_mens``     – PDC Tour Card holders and Order of Merit players
- ``pdc_womens``   – PDC Women's Series competitors
- ``wdf_open``     – WDF-affiliated players (amateur and professional)
- ``development``  – PDC Development/Challenge/Youth Tour players
- ``team_doubles`` – PDC World Cup of Darts pairs

K-factor tiers are calibrated so that top players' ratings are more stable
(more data, lower uncertainty) while newcomers update faster.  Draws are
supported for Premier League and Grand Slam group-stage matches.

ELO formula (standard):
    E_a = 1 / (1 + 10^((R_b - R_a)/400))
    ΔR_a = K * (S_a - E_a)

where S_a ∈ {0.0, 0.5, 1.0} (loss, draw, win).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import structlog

from competition.draw_result import DartsResultError, elo_score_for_result
from competition.format_registry import get_format


logger = structlog.get_logger(__name__)


class DartsEloError(Exception):
    """Raised when an ELO operation is logically inconsistent."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RATING = 1500.0
ELO_SCALE = 400.0  # standard Elo scale factor

# Ecosystem labels
POOL_PDC_MENS = "pdc_mens"
POOL_PDC_WOMENS = "pdc_womens"
POOL_WDF_OPEN = "wdf_open"
POOL_DEVELOPMENT = "development"
POOL_TEAM_DOUBLES = "team_doubles"

VALID_POOLS = frozenset(
    {POOL_PDC_MENS, POOL_PDC_WOMENS, POOL_WDF_OPEN, POOL_DEVELOPMENT, POOL_TEAM_DOUBLES}
)

# K-factor tiers keyed by (pool, rank_tier)
# rank_tier = "elite" | "established" | "challenger" | "newcomer"
_K_FACTORS: dict[str, dict[str, float]] = {
    POOL_PDC_MENS: {
        "elite":       16.0,   # ranked 1–16 (seeded World Championship)
        "established": 24.0,   # ranked 17–64 (Tour Card holders)
        "challenger":  32.0,   # ranked 65–128
        "newcomer":    40.0,   # ranked 129+
    },
    POOL_PDC_WOMENS: {
        "elite":       20.0,
        "established": 28.0,
        "challenger":  36.0,
        "newcomer":    44.0,
    },
    POOL_WDF_OPEN: {
        "elite":       20.0,
        "established": 28.0,
        "challenger":  36.0,
        "newcomer":    44.0,
    },
    POOL_DEVELOPMENT: {
        "elite":       28.0,
        "established": 32.0,
        "challenger":  40.0,
        "newcomer":    48.0,
    },
    POOL_TEAM_DOUBLES: {
        "elite":       20.0,
        "established": 28.0,
        "challenger":  36.0,
        "newcomer":    44.0,
    },
}

# Rank boundaries defining tiers (pdc_mens / pdc_womens)
_TIER_BOUNDARIES: dict[str, list[tuple[int, str]]] = {
    POOL_PDC_MENS: [(16, "elite"), (64, "established"), (128, "challenger")],
    POOL_PDC_WOMENS: [(8, "elite"), (32, "established"), (64, "challenger")],
    POOL_WDF_OPEN: [(8, "elite"), (32, "established"), (64, "challenger")],
    POOL_DEVELOPMENT: [(8, "elite"), (32, "established"), (64, "challenger")],
    POOL_TEAM_DOUBLES: [(8, "elite"), (16, "established"), (32, "challenger")],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EloRating:
    """
    The ELO rating record for a single player within a single pool.

    Attributes
    ----------
    player_id:
        Canonical internal player identifier.
    pool:
        ELO pool this rating belongs to.
    rating:
        Current ELO rating value.
    games_played:
        Total matches used to compute this rating.
    last_updated:
        Date of the last rating-changing match.
    provisional:
        True when ``games_played`` < 20 (high uncertainty).
    """

    player_id: str
    pool: str
    rating: float = DEFAULT_RATING
    games_played: int = 0
    last_updated: Optional[date] = None
    provisional: bool = True

    def __post_init__(self) -> None:
        if self.pool not in VALID_POOLS:
            raise DartsEloError(f"Unknown pool {self.pool!r}.")
        self.provisional = self.games_played < 20


@dataclass
class EloMatchInput:
    """
    Input record for a single ELO update.

    Attributes
    ----------
    player1_id:
        Canonical ID of player 1.
    player2_id:
        Canonical ID of player 2.
    result_type:
        ``"p1_win"`` | ``"p2_win"`` | ``"draw"``.
    pool:
        ELO pool.
    format_code:
        Competition format code.
    round_name:
        Round within the competition.
    match_date:
        Date the match was played.
    player1_rank:
        PDC Order of Merit / WDF ranking at time of match (for K-factor).
        ``None`` when unknown.
    player2_rank:
        Same for player 2.
    weight:
        Optional match weight multiplier (default 1.0). Use < 1.0 for
        exhibition matches or < 1.0 for low-confidence historical records.
    """

    player1_id: str
    player2_id: str
    result_type: str
    pool: str
    format_code: str
    round_name: str
    match_date: date
    player1_rank: Optional[int] = None
    player2_rank: Optional[int] = None
    weight: float = 1.0


@dataclass
class EloUpdateResult:
    """
    Output from a single ELO rating update.

    Attributes
    ----------
    player1_id, player2_id:
        Player identifiers.
    old_rating_p1, old_rating_p2:
        Ratings before the update.
    new_rating_p1, new_rating_p2:
        Ratings after the update.
    delta_p1, delta_p2:
        Rating changes.
    expected_p1, expected_p2:
        Pre-match expected scores in [0, 1].
    k_p1, k_p2:
        K-factors applied.
    result_type:
        Match outcome.
    """

    player1_id: str
    player2_id: str
    old_rating_p1: float
    old_rating_p2: float
    new_rating_p1: float
    new_rating_p2: float
    delta_p1: float
    delta_p2: float
    expected_p1: float
    expected_p2: float
    k_p1: float
    k_p2: float
    result_type: str


# ---------------------------------------------------------------------------
# Core ELO functions
# ---------------------------------------------------------------------------

def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Compute the expected score for player A against player B.

    Parameters
    ----------
    rating_a:
        ELO rating of player A.
    rating_b:
        ELO rating of player B.

    Returns
    -------
    float
        Expected score in [0, 1].
    """
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / ELO_SCALE))


def k_factor_for_rank(pool: str, rank: Optional[int]) -> float:
    """
    Return the K-factor for a player given their current ranking.

    Parameters
    ----------
    pool:
        ELO pool.
    rank:
        Current ranking (1-based, lower is better). ``None`` → newcomer tier.

    Returns
    -------
    float
        K-factor.

    Raises
    ------
    DartsEloError
        If the pool is unknown.
    """
    if pool not in VALID_POOLS:
        raise DartsEloError(f"Unknown pool {pool!r}.")
    tier = _rank_to_tier(pool, rank)
    return _K_FACTORS[pool][tier]


def _rank_to_tier(pool: str, rank: Optional[int]) -> str:
    """Map a numeric rank to a tier label within the given pool."""
    if rank is None:
        return "newcomer"
    boundaries = _TIER_BOUNDARIES.get(pool, [])
    for threshold, tier_label in boundaries:
        if rank <= threshold:
            return tier_label
    return "newcomer"


# ---------------------------------------------------------------------------
# ELO Pool
# ---------------------------------------------------------------------------

class EloPool:
    """
    In-memory ELO rating store for one ecosystem pool.

    This is the stateful component used during batch backfill and by the
    live engine between database persistence calls.

    Parameters
    ----------
    pool:
        The ecosystem pool identifier.
    initial_ratings:
        Optional seed ratings ``{player_id: float}`` from a previous run or
        from the database.
    """

    def __init__(
        self,
        pool: str,
        initial_ratings: Optional[dict[str, EloRating]] = None,
    ) -> None:
        if pool not in VALID_POOLS:
            raise DartsEloError(f"Unknown pool {pool!r}.")
        self.pool = pool
        self._ratings: dict[str, EloRating] = dict(initial_ratings or {})
        self._log = logger.bind(pool=pool)

    def get_or_create(self, player_id: str) -> EloRating:
        """
        Return the rating for *player_id*, creating a default entry if absent.

        Parameters
        ----------
        player_id:
            Canonical player identifier.

        Returns
        -------
        EloRating
        """
        if player_id not in self._ratings:
            self._ratings[player_id] = EloRating(
                player_id=player_id,
                pool=self.pool,
                rating=DEFAULT_RATING,
                games_played=0,
            )
        return self._ratings[player_id]

    def get_rating(self, player_id: str) -> float:
        """Return the current numeric rating for a player."""
        return self.get_or_create(player_id).rating

    def update(self, match_input: EloMatchInput) -> EloUpdateResult:
        """
        Process a single match and update both players' ratings.

        Parameters
        ----------
        match_input:
            Structured match input.

        Returns
        -------
        EloUpdateResult
            Full before/after rating record.

        Raises
        ------
        DartsEloError
            If the match pool doesn't match this pool's identifier.
        DartsResultError
            If the result_type is invalid.
        """
        if match_input.pool != self.pool:
            raise DartsEloError(
                f"Match pool {match_input.pool!r} does not match "
                f"this pool {self.pool!r}."
            )

        # Validate format allows this result
        fmt = get_format(match_input.format_code)
        if match_input.result_type not in fmt.result_types:
            raise DartsEloError(
                f"result_type {match_input.result_type!r} not valid for "
                f"format {match_input.format_code}."
            )

        r1 = self.get_or_create(match_input.player1_id)
        r2 = self.get_or_create(match_input.player2_id)

        old_r1 = r1.rating
        old_r2 = r2.rating

        e1 = expected_score(old_r1, old_r2)
        e2 = 1.0 - e1

        s1, s2 = elo_score_for_result(match_input.result_type)

        k1 = k_factor_for_rank(self.pool, match_input.player1_rank)
        k2 = k_factor_for_rank(self.pool, match_input.player2_rank)

        # Apply weight
        k1 *= match_input.weight
        k2 *= match_input.weight

        delta1 = k1 * (s1 - e1)
        delta2 = k2 * (s2 - e2)

        new_r1 = old_r1 + delta1
        new_r2 = old_r2 + delta2

        r1.rating = new_r1
        r1.games_played += 1
        r1.last_updated = match_input.match_date
        r1.provisional = r1.games_played < 20

        r2.rating = new_r2
        r2.games_played += 1
        r2.last_updated = match_input.match_date
        r2.provisional = r2.games_played < 20

        result = EloUpdateResult(
            player1_id=match_input.player1_id,
            player2_id=match_input.player2_id,
            old_rating_p1=old_r1,
            old_rating_p2=old_r2,
            new_rating_p1=new_r1,
            new_rating_p2=new_r2,
            delta_p1=delta1,
            delta_p2=delta2,
            expected_p1=e1,
            expected_p2=e2,
            k_p1=k1,
            k_p2=k2,
            result_type=match_input.result_type,
        )

        self._log.debug(
            "elo_update",
            player1_id=match_input.player1_id,
            player2_id=match_input.player2_id,
            result=match_input.result_type,
            delta_p1=round(delta1, 2),
            delta_p2=round(delta2, 2),
            new_rating_p1=round(new_r1, 2),
            new_rating_p2=round(new_r2, 2),
        )
        return result

    def snapshot(self) -> dict[str, EloRating]:
        """Return a shallow copy of all current ratings."""
        return dict(self._ratings)

    def player_count(self) -> int:
        """Number of players with ratings in this pool."""
        return len(self._ratings)

    def top_n(self, n: int = 20) -> list[EloRating]:
        """
        Return the top-N rated players, descending by rating.

        Parameters
        ----------
        n:
            Number of players to return.

        Returns
        -------
        list[EloRating]
        """
        sorted_ratings = sorted(
            self._ratings.values(),
            key=lambda r: r.rating,
            reverse=True,
        )
        return sorted_ratings[:n]


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class EloPipeline:
    """
    Orchestrates ELO updates across all five pools.

    Typical usage:
        pipeline = EloPipeline()
        result = pipeline.process_match(match_input)
        snapshot = pipeline.snapshot_pool("pdc_mens")

    Parameters
    ----------
    seed_ratings:
        Optional mapping ``{pool: {player_id: EloRating}}`` for warm-start.
    """

    def __init__(
        self,
        seed_ratings: Optional[dict[str, dict[str, EloRating]]] = None,
    ) -> None:
        self._pools: dict[str, EloPool] = {
            pool: EloPool(
                pool=pool,
                initial_ratings=(seed_ratings or {}).get(pool),
            )
            for pool in VALID_POOLS
        }
        self._log = logger.bind(component="EloPipeline")

    def process_match(self, match_input: EloMatchInput) -> EloUpdateResult:
        """
        Route a match to the correct pool and apply the ELO update.

        Parameters
        ----------
        match_input:
            Structured match input record.

        Returns
        -------
        EloUpdateResult

        Raises
        ------
        DartsEloError
            If the pool in match_input is unknown.
        """
        pool = match_input.pool
        if pool not in self._pools:
            raise DartsEloError(f"Unknown pool {pool!r}.")
        return self._pools[pool].update(match_input)

    def process_batch(
        self,
        matches: list[EloMatchInput],
    ) -> list[EloUpdateResult]:
        """
        Process a chronologically-ordered batch of matches.

        Parameters
        ----------
        matches:
            List of match inputs, ordered by match_date ascending.

        Returns
        -------
        list[EloUpdateResult]
            One result per input match, in the same order.
        """
        self._log.info("batch_start", count=len(matches))
        results: list[EloUpdateResult] = []
        for match in matches:
            result = self.process_match(match)
            results.append(result)
        self._log.info("batch_complete", count=len(results))
        return results

    def get_rating(self, pool: str, player_id: str) -> float:
        """
        Return the current ELO rating for a player in a pool.

        Parameters
        ----------
        pool:
            ELO pool name.
        player_id:
            Canonical player identifier.

        Returns
        -------
        float
            Current rating.

        Raises
        ------
        DartsEloError
            If the pool is unknown.
        """
        if pool not in self._pools:
            raise DartsEloError(f"Unknown pool {pool!r}.")
        return self._pools[pool].get_rating(player_id)

    def snapshot_pool(self, pool: str) -> dict[str, EloRating]:
        """
        Return a snapshot of all ratings in a pool.

        Parameters
        ----------
        pool:
            ELO pool name.

        Returns
        -------
        dict[str, EloRating]

        Raises
        ------
        DartsEloError
            If the pool is unknown.
        """
        if pool not in self._pools:
            raise DartsEloError(f"Unknown pool {pool!r}.")
        return self._pools[pool].snapshot()

    def snapshot_all(self) -> dict[str, dict[str, EloRating]]:
        """Return snapshots of all pools."""
        return {pool: p.snapshot() for pool, p in self._pools.items()}

    def pool_stats(self) -> dict[str, dict[str, int]]:
        """
        Return per-pool statistics.

        Returns
        -------
        dict
            ``{pool: {"player_count": int}}``
        """
        return {
            pool: {"player_count": p.player_count()}
            for pool, p in self._pools.items()
        }
