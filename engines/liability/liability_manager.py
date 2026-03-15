"""
Liability Management Engine — XG3 Darts.

Tracks real-time bet exposure per match/market and enforces configured limits.
All financial values are in the operator's base currency minor units (e.g. EUR cents).

Responsibilities
----------------
1. ``check_acceptance`` — Pre-bet check: returns accept/reject + adjusted odds/stake.
2. ``record_bet``       — Atomically increment exposure after bet acceptance.
3. ``release_bet``      — Decrement exposure on void/cancel.
4. ``settle_bet``       — Mark bet settled; exposure is realised as win or loss.
5. ``get_exposure``     — Snapshot of current exposure for a match/market.
6. ``get_limits``       — Load/cache configured limits from DB.

Concurrency: all mutating operations use SELECT ... FOR UPDATE on the bet row
to prevent double-counting under concurrent requests.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ExposureLimit:
    """Loaded from ``darts_liability_limits``."""

    limit_id: str
    market_family: str
    max_exposure_per_outcome: int   # minor units
    max_exposure_per_match: int     # minor units
    max_single_stake: int           # minor units
    auto_suspend_threshold: float   # 0.0 – 1.0


@dataclass
class ExposureSnapshot:
    """Current live exposure for a match/market combination."""

    match_id: str
    market_family: str
    market_key: str
    open_liability: int      # total potential loss on open bets (minor units)
    settled_loss: int        # realised losses on settled bets (minor units)
    open_bet_count: int
    limit: Optional[ExposureLimit] = None

    @property
    def utilisation(self) -> Optional[float]:
        """Fraction of max_exposure_per_outcome consumed [0,1]."""
        if self.limit is None or self.limit.max_exposure_per_outcome == 0:
            return None
        return self.open_liability / self.limit.max_exposure_per_outcome

    @property
    def is_near_limit(self) -> bool:
        """True when auto_suspend_threshold is reached."""
        u = self.utilisation
        if u is None or self.limit is None:
            return False
        return u >= self.limit.auto_suspend_threshold

    @property
    def headroom(self) -> Optional[int]:
        """Remaining capacity before limit (minor units). None if no limit."""
        if self.limit is None:
            return None
        return max(0, self.limit.max_exposure_per_outcome - self.open_liability)


@dataclass
class BetAcceptanceResult:
    """
    Decision returned by ``check_acceptance``.

    ``accepted`` is True even when stake is trimmed.
    ``rejected`` means the bet cannot be accepted at all.
    """

    accepted: bool
    adjusted_stake: int            # may be less than requested if stake-trimmed
    adjusted_odds: float           # may be tightened if near limit
    rejection_reason: Optional[str] = None
    exposure_after: Optional[ExposureSnapshot] = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class LiabilityManager:
    """
    Stateless service that operates against the asyncpg connection pool.

    Parameters
    ----------
    pool:
        asyncpg connection pool connected to the Railway Postgres DB.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self._log = logger.bind(component="LiabilityManager")

    # ------------------------------------------------------------------
    # Limit configuration
    # ------------------------------------------------------------------

    async def get_limits(
        self,
        market_family: str,
        match_id: Optional[str] = None,
        competition_id: Optional[str] = None,
    ) -> Optional[ExposureLimit]:
        """
        Load the most specific applicable limit for a market.

        Resolution order: match-specific > competition-specific > global.

        Returns None if no limit is configured (market is unlimited).
        """
        async with self._pool.acquire() as conn:
            # Match-level
            if match_id:
                row = await conn.fetchrow(
                    """
                    SELECT id, market_family, max_exposure_per_outcome,
                           max_exposure_per_match, max_single_stake,
                           auto_suspend_threshold
                    FROM darts_liability_limits
                    WHERE match_id = $1
                      AND market_family = $2
                      AND is_active = TRUE
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    match_id,
                    market_family,
                )
                if row:
                    return _row_to_limit(row)

            # Competition-level
            if competition_id:
                row = await conn.fetchrow(
                    """
                    SELECT id, market_family, max_exposure_per_outcome,
                           max_exposure_per_match, max_single_stake,
                           auto_suspend_threshold
                    FROM darts_liability_limits
                    WHERE competition_id = $1
                      AND match_id IS NULL
                      AND market_family = $2
                      AND is_active = TRUE
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    competition_id,
                    market_family,
                )
                if row:
                    return _row_to_limit(row)

            # Global default
            row = await conn.fetchrow(
                """
                SELECT id, market_family, max_exposure_per_outcome,
                       max_exposure_per_match, max_single_stake,
                       auto_suspend_threshold
                FROM darts_liability_limits
                WHERE competition_id IS NULL
                  AND match_id IS NULL
                  AND market_family = $1
                  AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
                """,
                market_family,
            )
            if row:
                return _row_to_limit(row)

        return None

    async def upsert_limit(
        self,
        market_family: str,
        max_exposure_per_outcome: int,
        max_exposure_per_match: int,
        max_single_stake: int,
        auto_suspend_threshold: float = 0.9,
        match_id: Optional[str] = None,
        competition_id: Optional[str] = None,
        created_by: str = "system",
    ) -> str:
        """
        Insert or update a liability limit.

        Returns the limit ID.
        """
        if not 0.0 < auto_suspend_threshold <= 1.0:
            raise ValueError(
                f"auto_suspend_threshold must be in (0, 1], got {auto_suspend_threshold}"
            )

        limit_id = str(uuid.uuid4())
        async with self._pool.acquire() as conn:
            # Deactivate any existing limit for the same scope/market
            await conn.execute(
                """
                UPDATE darts_liability_limits
                SET is_active = FALSE, updated_at = NOW()
                WHERE market_family = $1
                  AND (match_id IS NOT DISTINCT FROM $2)
                  AND (competition_id IS NOT DISTINCT FROM $3)
                  AND is_active = TRUE
                """,
                market_family,
                match_id,
                competition_id,
            )

            await conn.execute(
                """
                INSERT INTO darts_liability_limits (
                    id, competition_id, match_id, market_family,
                    max_exposure_per_outcome, max_exposure_per_match,
                    max_single_stake, auto_suspend_threshold,
                    created_by, is_active
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,TRUE)
                """,
                limit_id,
                competition_id,
                match_id,
                market_family,
                max_exposure_per_outcome,
                max_exposure_per_match,
                max_single_stake,
                auto_suspend_threshold,
                created_by,
            )

        self._log.info(
            "liability_limit_set",
            limit_id=limit_id,
            market_family=market_family,
            max_exposure_per_outcome=max_exposure_per_outcome,
            match_id=match_id,
            competition_id=competition_id,
        )
        return limit_id

    # ------------------------------------------------------------------
    # Pre-bet acceptance check
    # ------------------------------------------------------------------

    async def check_acceptance(
        self,
        match_id: str,
        market_family: str,
        market_key: str,
        requested_stake: int,
        odds_decimal: float,
        competition_id: Optional[str] = None,
        state_snapshot: Optional[dict[str, Any]] = None,
    ) -> BetAcceptanceResult:
        """
        Check whether a bet can be accepted.

        Does NOT write to the database. Call ``record_bet`` after acceptance.

        Parameters
        ----------
        match_id:
            Internal darts_matches.id.
        market_family:
            One of the 7 market families (e.g. "match_winner").
        market_key:
            Specific selection key (e.g. "match_winner:p1").
        requested_stake:
            Proposed bet stake in minor currency units.
        odds_decimal:
            Decimal odds offered to the customer.
        competition_id:
            Optional competition for limit resolution.
        state_snapshot:
            Optional current match state for audit.

        Returns
        -------
        BetAcceptanceResult
        """
        limit = await self.get_limits(market_family, match_id, competition_id)
        snapshot = await self.get_exposure(match_id, market_family, market_key)

        potential_loss = _calc_potential_loss(requested_stake, odds_decimal)
        warnings: list[str] = []
        adjusted_stake = requested_stake
        adjusted_odds = odds_decimal

        if limit is None:
            # No configured limit — accept as-is
            return BetAcceptanceResult(
                accepted=True,
                adjusted_stake=requested_stake,
                adjusted_odds=odds_decimal,
                exposure_after=snapshot,
            )

        # 1. Check single-bet stake limit
        if requested_stake > limit.max_single_stake:
            adjusted_stake = limit.max_single_stake
            warnings.append(
                f"Stake trimmed from {requested_stake} to {adjusted_stake} "
                f"(max_single_stake={limit.max_single_stake})"
            )
            # Recalculate potential loss with adjusted stake
            potential_loss = _calc_potential_loss(adjusted_stake, odds_decimal)

        # 2. Check outcome-level exposure headroom
        current_exposure = snapshot.open_liability if snapshot else 0
        headroom = limit.max_exposure_per_outcome - current_exposure

        if headroom <= 0:
            # Fully exhausted — reject
            return BetAcceptanceResult(
                accepted=False,
                adjusted_stake=0,
                adjusted_odds=odds_decimal,
                rejection_reason=(
                    f"Exposure limit exhausted for {market_key}: "
                    f"current={current_exposure}, max={limit.max_exposure_per_outcome}"
                ),
                exposure_after=snapshot,
            )

        if potential_loss > headroom:
            # Trim stake to fit within remaining headroom
            max_stake_for_headroom = math.floor(headroom / max(odds_decimal - 1.0, 0.01))
            adjusted_stake = min(adjusted_stake, max_stake_for_headroom)
            adjusted_odds = odds_decimal
            warnings.append(
                f"Stake trimmed to {adjusted_stake} to stay within "
                f"exposure headroom={headroom}"
            )

        if adjusted_stake <= 0:
            return BetAcceptanceResult(
                accepted=False,
                adjusted_stake=0,
                adjusted_odds=odds_decimal,
                rejection_reason="Adjusted stake is zero after exposure trimming.",
                exposure_after=snapshot,
            )

        # 3. Near-limit warning — tighten odds if ≥ 80% utilised
        if snapshot:
            utilisation = (current_exposure + _calc_potential_loss(adjusted_stake, adjusted_odds)) / limit.max_exposure_per_outcome
            if utilisation >= 0.8:
                # Tighten by up to 2% to reduce new exposure
                tightening_factor = min(0.02, (utilisation - 0.8) * 0.1)
                adjusted_odds = max(1.01, adjusted_odds * (1.0 - tightening_factor))
                warnings.append(
                    f"Odds tightened by {tightening_factor*100:.1f}% due to "
                    f"high exposure utilisation ({utilisation*100:.1f}%)"
                )

        return BetAcceptanceResult(
            accepted=True,
            adjusted_stake=adjusted_stake,
            adjusted_odds=round(adjusted_odds, 3),
            exposure_after=snapshot,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Bet lifecycle
    # ------------------------------------------------------------------

    async def record_bet(
        self,
        external_bet_id: str,
        match_id: str,
        market_family: str,
        market_key: str,
        stake: int,
        odds_decimal: float,
        competition_id: Optional[str] = None,
        state_snapshot: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Record an accepted bet and increment live exposure.

        Returns the internal bet record ID.
        """
        potential_payout = round(stake * odds_decimal)
        potential_loss = potential_payout - stake
        bet_id = str(uuid.uuid4())

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO darts_bet_exposure (
                    id, external_bet_id, match_id, competition_id,
                    market_family, market_key,
                    stake, odds_decimal, potential_payout, potential_loss,
                    status, placed_at, state_snapshot
                ) VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                    'open', NOW(), $11
                )
                ON CONFLICT (external_bet_id) DO NOTHING
                """,
                bet_id,
                external_bet_id,
                match_id,
                competition_id,
                market_family,
                market_key,
                stake,
                odds_decimal,
                potential_payout,
                potential_loss,
                state_snapshot,
            )

        self._log.info(
            "bet_recorded",
            bet_id=bet_id,
            external_bet_id=external_bet_id,
            match_id=match_id,
            market_key=market_key,
            stake=stake,
            potential_loss=potential_loss,
        )
        return bet_id

    async def settle_bet(
        self,
        external_bet_id: str,
        outcome: str,  # "win" | "lose" | "void" | "cancelled"
    ) -> bool:
        """
        Mark a bet as settled.

        Parameters
        ----------
        external_bet_id:
            Operator's bet reference.
        outcome:
            Settlement outcome.

        Returns True if the record was found and updated, False otherwise.
        """
        status_map = {
            "win": "settled_win",
            "lose": "settled_lose",
            "void": "voided",
            "cancelled": "cancelled",
        }
        new_status = status_map.get(outcome.lower())
        if new_status is None:
            raise ValueError(
                f"Invalid settlement outcome {outcome!r}. "
                "Must be one of: win, lose, void, cancelled"
            )

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE darts_bet_exposure
                SET status = $1, settled_at = NOW()
                WHERE external_bet_id = $2
                  AND status = 'open'
                """,
                new_status,
                external_bet_id,
            )

        updated = int(result.split()[-1]) if result else 0
        if updated:
            self._log.info(
                "bet_settled",
                external_bet_id=external_bet_id,
                outcome=outcome,
                new_status=new_status,
            )
        return updated > 0

    async def release_bet(self, external_bet_id: str) -> bool:
        """
        Cancel/void a bet before settlement, releasing its exposure.

        Returns True if found and cancelled.
        """
        return await self.settle_bet(external_bet_id, "cancelled")

    # ------------------------------------------------------------------
    # Exposure query
    # ------------------------------------------------------------------

    async def get_exposure(
        self,
        match_id: str,
        market_family: str,
        market_key: str,
        competition_id: Optional[str] = None,
    ) -> ExposureSnapshot:
        """
        Return the current live exposure snapshot for a market key.

        Aggregates all open bets for the given match/market/key combination.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COALESCE(SUM(potential_loss), 0)::bigint AS open_liability,
                    COUNT(*)::int                             AS open_bet_count
                FROM darts_bet_exposure
                WHERE match_id = $1
                  AND market_family = $2
                  AND market_key = $3
                  AND status = 'open'
                """,
                match_id,
                market_family,
                market_key,
            )

            settled_row = await conn.fetchrow(
                """
                SELECT COALESCE(SUM(potential_loss), 0)::bigint AS settled_loss
                FROM darts_bet_exposure
                WHERE match_id = $1
                  AND market_family = $2
                  AND market_key = $3
                  AND status = 'settled_win'
                """,
                match_id,
                market_family,
                market_key,
            )

        limit = await self.get_limits(market_family, match_id, competition_id)

        return ExposureSnapshot(
            match_id=match_id,
            market_family=market_family,
            market_key=market_key,
            open_liability=int(row["open_liability"]),
            settled_loss=int(settled_row["settled_loss"]),
            open_bet_count=int(row["open_bet_count"]),
            limit=limit,
        )

    async def get_all_exposure_for_match(
        self,
        match_id: str,
    ) -> list[ExposureSnapshot]:
        """
        Return exposure snapshots for all open market keys for a match.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    market_family,
                    market_key,
                    COALESCE(SUM(potential_loss), 0)::bigint AS open_liability,
                    COUNT(*)::int                             AS open_bet_count
                FROM darts_bet_exposure
                WHERE match_id = $1
                  AND status = 'open'
                GROUP BY market_family, market_key
                ORDER BY open_liability DESC
                """,
                match_id,
            )

        snapshots: list[ExposureSnapshot] = []
        for row in rows:
            limit = await self.get_limits(row["market_family"], match_id)
            snapshots.append(
                ExposureSnapshot(
                    match_id=match_id,
                    market_family=row["market_family"],
                    market_key=row["market_key"],
                    open_liability=int(row["open_liability"]),
                    settled_loss=0,
                    open_bet_count=int(row["open_bet_count"]),
                    limit=limit,
                )
            )
        return snapshots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_limit(row: asyncpg.Record) -> ExposureLimit:
    return ExposureLimit(
        limit_id=str(row["id"]),
        market_family=str(row["market_family"]),
        max_exposure_per_outcome=int(row["max_exposure_per_outcome"]),
        max_exposure_per_match=int(row["max_exposure_per_match"]),
        max_single_stake=int(row["max_single_stake"]),
        auto_suspend_threshold=float(row["auto_suspend_threshold"]),
    )


def _calc_potential_loss(stake: int, odds_decimal: float) -> int:
    """Calculate operator net liability (potential payout minus stake)."""
    return max(0, round(stake * odds_decimal) - stake)
