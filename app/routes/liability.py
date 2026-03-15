"""
Liability management API routes.

Endpoints
---------
GET  /api/v1/darts/liability/exposure/{match_id}
    Full exposure snapshot for a match (all open market keys).

GET  /api/v1/darts/liability/exposure/{match_id}/{market_family}/{market_key}
    Exposure snapshot for a specific market key.

POST /api/v1/darts/liability/check
    Pre-bet acceptance check (does NOT write to DB).

POST /api/v1/darts/liability/bets
    Record an accepted bet and increment exposure.

PATCH /api/v1/darts/liability/bets/{external_bet_id}/settle
    Settle a bet (win/lose/void/cancelled).

GET  /api/v1/darts/liability/limits
    List configured exposure limits.

POST /api/v1/darts/liability/limits
    Create or update an exposure limit.
"""
from __future__ import annotations

from typing import Any, Optional

import asyncpg
import structlog
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from engines.liability.liability_manager import (
    BetAcceptanceResult,
    ExposureSnapshot,
    LiabilityManager,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

_pool: Optional[asyncpg.Pool] = None


async def _get_liability_manager() -> LiabilityManager:
    """Return a LiabilityManager backed by the shared asyncpg pool."""
    global _pool
    if _pool is None or _pool._closed:
        db_url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
        _pool = await asyncpg.create_pool(
            db_url,
            min_size=1,
            max_size=5,
            command_timeout=10,
        )
    return LiabilityManager(_pool)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ExposureSnapshotResponse(BaseModel):
    match_id: str
    market_family: str
    market_key: str
    open_liability: int
    settled_loss: int
    open_bet_count: int
    utilisation: Optional[float]
    headroom: Optional[int]
    is_near_limit: bool
    limit_max_per_outcome: Optional[int]


class BetCheckRequest(BaseModel):
    match_id: str
    market_family: str = Field(..., description="One of: match_winner | leg_handicap | totals | exact_score | props_180 | props_checkout | outright")
    market_key: str = Field(..., description="e.g. 'match_winner:p1' | 'totals:over:8.5'")
    stake: int = Field(..., gt=0, description="Stake in minor currency units (e.g. EUR cents)")
    odds_decimal: float = Field(..., gt=1.0, description="Decimal odds offered to customer")
    competition_id: Optional[str] = None


class BetCheckResponse(BaseModel):
    accepted: bool
    adjusted_stake: int
    adjusted_odds: float
    rejection_reason: Optional[str]
    warnings: list[str]


class RecordBetRequest(BaseModel):
    external_bet_id: str = Field(..., description="Operator's unique bet reference")
    match_id: str
    market_family: str
    market_key: str
    stake: int = Field(..., gt=0)
    odds_decimal: float = Field(..., gt=1.0)
    competition_id: Optional[str] = None
    state_snapshot: Optional[dict[str, Any]] = None


class RecordBetResponse(BaseModel):
    bet_id: str
    external_bet_id: str
    potential_loss: int


class SettleBetRequest(BaseModel):
    outcome: str = Field(
        ...,
        description="Settlement outcome: win | lose | void | cancelled",
    )

    @field_validator("outcome")
    @classmethod
    def validate_outcome(cls, v: str) -> str:
        valid = {"win", "lose", "void", "cancelled"}
        if v.lower() not in valid:
            raise ValueError(f"outcome must be one of {sorted(valid)}, got {v!r}")
        return v.lower()


class LimitSetRequest(BaseModel):
    market_family: str
    max_exposure_per_outcome: int = Field(..., gt=0)
    max_exposure_per_match: int = Field(..., gt=0)
    max_single_stake: int = Field(..., gt=0)
    auto_suspend_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    match_id: Optional[str] = None
    competition_id: Optional[str] = None
    created_by: str = "api"


class LimitSetResponse(BaseModel):
    limit_id: str
    market_family: str
    scope: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/liability/exposure/{match_id}",
    response_model=list[ExposureSnapshotResponse],
    summary="Get exposure snapshots for all open markets on a match",
)
async def get_match_exposure(
    match_id: str = Path(..., description="Internal match UUID"),
    manager: LiabilityManager = Depends(_get_liability_manager),
) -> list[ExposureSnapshotResponse]:
    """
    Return exposure for all open market keys on a match, ordered by liability (descending).
    """
    snapshots = await manager.get_all_exposure_for_match(match_id)
    return [_snapshot_to_response(s) for s in snapshots]


@router.get(
    "/liability/exposure/{match_id}/{market_family}/{market_key:path}",
    response_model=ExposureSnapshotResponse,
    summary="Get exposure for a specific market key",
)
async def get_market_exposure(
    match_id: str = Path(...),
    market_family: str = Path(...),
    market_key: str = Path(...),
    competition_id: Optional[str] = Query(None),
    manager: LiabilityManager = Depends(_get_liability_manager),
) -> ExposureSnapshotResponse:
    snapshot = await manager.get_exposure(
        match_id=match_id,
        market_family=market_family,
        market_key=market_key,
        competition_id=competition_id,
    )
    return _snapshot_to_response(snapshot)


@router.post(
    "/liability/check",
    response_model=BetCheckResponse,
    summary="Pre-bet acceptance check (read-only)",
)
async def check_bet_acceptance(
    req: BetCheckRequest = Body(...),
    manager: LiabilityManager = Depends(_get_liability_manager),
) -> BetCheckResponse:
    """
    Check whether a bet can be accepted at the requested stake/odds.
    Does NOT write to the database — safe to call multiple times.
    """
    result: BetAcceptanceResult = await manager.check_acceptance(
        match_id=req.match_id,
        market_family=req.market_family,
        market_key=req.market_key,
        requested_stake=req.stake,
        odds_decimal=req.odds_decimal,
        competition_id=req.competition_id,
    )
    return BetCheckResponse(
        accepted=result.accepted,
        adjusted_stake=result.adjusted_stake,
        adjusted_odds=result.adjusted_odds,
        rejection_reason=result.rejection_reason,
        warnings=result.warnings,
    )


@router.post(
    "/liability/bets",
    response_model=RecordBetResponse,
    status_code=201,
    summary="Record an accepted bet",
)
async def record_bet(
    req: RecordBetRequest = Body(...),
    manager: LiabilityManager = Depends(_get_liability_manager),
) -> RecordBetResponse:
    """
    Persist an accepted bet and increment live exposure counters.

    The caller must have already run ``/liability/check`` and obtained
    acceptance.  This endpoint is idempotent on ``external_bet_id``.
    """
    from engines.liability.liability_manager import _calc_potential_loss

    bet_id = await manager.record_bet(
        external_bet_id=req.external_bet_id,
        match_id=req.match_id,
        market_family=req.market_family,
        market_key=req.market_key,
        stake=req.stake,
        odds_decimal=req.odds_decimal,
        competition_id=req.competition_id,
        state_snapshot=req.state_snapshot,
    )
    return RecordBetResponse(
        bet_id=bet_id,
        external_bet_id=req.external_bet_id,
        potential_loss=_calc_potential_loss(req.stake, req.odds_decimal),
    )


@router.patch(
    "/liability/bets/{external_bet_id}/settle",
    summary="Settle a bet",
)
async def settle_bet(
    external_bet_id: str = Path(...),
    req: SettleBetRequest = Body(...),
    manager: LiabilityManager = Depends(_get_liability_manager),
) -> dict[str, Any]:
    """
    Mark a bet as settled.  Removes it from open exposure.

    Outcome must be: ``win`` | ``lose`` | ``void`` | ``cancelled``.
    """
    found = await manager.settle_bet(
        external_bet_id=external_bet_id,
        outcome=req.outcome,
    )
    if not found:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "bet_not_found",
                "external_bet_id": external_bet_id,
                "message": "Bet not found or already settled.",
            },
        )
    return {"settled": True, "external_bet_id": external_bet_id, "outcome": req.outcome}


@router.post(
    "/liability/limits",
    response_model=LimitSetResponse,
    status_code=201,
    summary="Create or update an exposure limit",
)
async def set_limit(
    req: LimitSetRequest = Body(...),
    manager: LiabilityManager = Depends(_get_liability_manager),
) -> LimitSetResponse:
    """
    Set an exposure limit for a market family.

    Scope resolution: ``match_id`` → ``competition_id`` → global.
    The most specific limit wins when evaluating a bet.
    """
    limit_id = await manager.upsert_limit(
        market_family=req.market_family,
        max_exposure_per_outcome=req.max_exposure_per_outcome,
        max_exposure_per_match=req.max_exposure_per_match,
        max_single_stake=req.max_single_stake,
        auto_suspend_threshold=req.auto_suspend_threshold,
        match_id=req.match_id,
        competition_id=req.competition_id,
        created_by=req.created_by,
    )
    scope = "match" if req.match_id else ("competition" if req.competition_id else "global")
    return LimitSetResponse(
        limit_id=limit_id,
        market_family=req.market_family,
        scope=scope,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot_to_response(s: ExposureSnapshot) -> ExposureSnapshotResponse:
    return ExposureSnapshotResponse(
        match_id=s.match_id,
        market_family=s.market_family,
        market_key=s.market_key,
        open_liability=s.open_liability,
        settled_loss=s.settled_loss,
        open_bet_count=s.open_bet_count,
        utilisation=round(s.utilisation, 4) if s.utilisation is not None else None,
        headroom=s.headroom,
        is_near_limit=s.is_near_limit,
        limit_max_per_outcome=(
            s.limit.max_exposure_per_outcome if s.limit else None
        ),
    )
