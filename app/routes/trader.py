"""
Live Trader Override API routes.

Allows human traders to intervene in automated pricing during:
  - Player injury / withdrawal
  - Breaking news affecting a match
  - Suspicious betting patterns
  - Technical feed issues

Endpoints
---------
POST   /api/v1/darts/trader/overrides
    Create a trader override (suspend, widen margin, restrict stakes, pull).

GET    /api/v1/darts/trader/overrides
    List active overrides (filterable by match/competition).

GET    /api/v1/darts/trader/overrides/{override_id}
    Retrieve a specific override.

DELETE /api/v1/darts/trader/overrides/{override_id}
    Revoke an override.

GET    /api/v1/darts/trader/overrides/match/{match_id}/effective
    Compute the effective pricing instructions for a match (merged overrides).

The engine checks ``get_effective_override`` before publishing any price.
An active SUSPEND override will block the market; WIDEN_MARGIN adds to the
5-factor margin; RESTRICT_STAKES caps the max accepted stake.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg
import structlog
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field, field_validator

from app.config import settings

logger = structlog.get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ACTIONS = {"SUSPEND", "RESUME", "WIDEN_MARGIN", "PULL", "RESTRICT_STAKES"}
VALID_REASON_CODES = {
    "INJURY", "WITHDRAWAL", "NEWS", "SUSPICIOUS_BETTING", "TECHNICAL", "OTHER"
}
VALID_SCOPES = {"match", "competition", "global"}

# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

_pool: Optional[asyncpg.Pool] = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None or _pool._closed:
        db_url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
        _pool = await asyncpg.create_pool(
            db_url, min_size=1, max_size=5, command_timeout=10,
        )
    return _pool


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CreateOverrideRequest(BaseModel):
    action: str = Field(
        ...,
        description="SUSPEND | RESUME | WIDEN_MARGIN | PULL | RESTRICT_STAKES",
    )
    scope: str = Field(
        ...,
        description="match | competition | global",
    )
    match_id: Optional[str] = Field(None, description="Required if scope='match'")
    competition_id: Optional[str] = Field(None, description="Required if scope='competition'")
    market_family: Optional[str] = Field(
        None,
        description="Target market family. NULL = all markets.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action-specific params. "
            "WIDEN_MARGIN: {'extra_margin_pct': 5.0} | "
            "RESTRICT_STAKES: {'max_stake': 500} | "
            "SUSPEND: {'reason': 'injury report'}"
        ),
    )
    reason: Optional[str] = Field(None, description="Free-text reason for audit log")
    reason_code: Optional[str] = Field(
        None,
        description="INJURY | WITHDRAWAL | NEWS | SUSPICIOUS_BETTING | TECHNICAL | OTHER",
    )
    valid_until: Optional[datetime] = Field(
        None,
        description="Override expiry (ISO-8601). NULL = active until manually revoked.",
    )
    created_by: str = Field(..., description="Trader ID / username")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        v = v.upper()
        if v not in VALID_ACTIONS:
            raise ValueError(f"action must be one of {sorted(VALID_ACTIONS)}, got {v!r}")
        return v

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, v: str) -> str:
        v = v.lower()
        if v not in VALID_SCOPES:
            raise ValueError(f"scope must be one of {sorted(VALID_SCOPES)}, got {v!r}")
        return v

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.upper()
        if v not in VALID_REASON_CODES:
            raise ValueError(
                f"reason_code must be one of {sorted(VALID_REASON_CODES)}, got {v!r}"
            )
        return v


class OverrideResponse(BaseModel):
    id: str
    action: str
    scope: str
    match_id: Optional[str]
    competition_id: Optional[str]
    market_family: Optional[str]
    params: dict[str, Any]
    reason: Optional[str]
    reason_code: Optional[str]
    valid_from: str
    valid_until: Optional[str]
    is_active: bool
    created_by: str
    created_at: str
    revoked_by: Optional[str]
    revoked_at: Optional[str]


class EffectiveOverrideResponse(BaseModel):
    """Merged effective instructions for a match (used by the pricing engine)."""

    match_id: str
    is_suspended: bool
    is_pulled: bool
    extra_margin_pct: float          # sum of all active WIDEN_MARGIN overrides
    max_stake_override: Optional[int] # lowest active RESTRICT_STAKES, if any
    active_overrides: list[OverrideResponse]


class RevokeOverrideRequest(BaseModel):
    revoked_by: str = Field(..., description="Trader ID / username revoking the override")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/trader/overrides",
    response_model=OverrideResponse,
    status_code=201,
    summary="Create a trader override",
)
async def create_override(
    req: CreateOverrideRequest = Body(...),
    pool: asyncpg.Pool = Depends(_get_pool),
) -> OverrideResponse:
    """
    Create a trader override to intervene in automated market pricing.

    Common scenarios
    ----------------
    * Player injury — SUSPEND all markets for a match until withdrawal confirmed.
    * Suspicious pattern — RESTRICT_STAKES to €100 while investigation runs.
    * Big news release — WIDEN_MARGIN by 5% to protect against sharp info asymmetry.
    * Pull the match — PULL removes all markets until situation is resolved.
    """
    if req.scope == "match" and not req.match_id:
        raise HTTPException(
            status_code=422,
            detail={"error": "match_id required when scope='match'"},
        )
    if req.scope == "competition" and not req.competition_id:
        raise HTTPException(
            status_code=422,
            detail={"error": "competition_id required when scope='competition'"},
        )

    # Validate WIDEN_MARGIN params
    if req.action == "WIDEN_MARGIN":
        extra = req.params.get("extra_margin_pct")
        if extra is None or not isinstance(extra, (int, float)) or extra <= 0:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "WIDEN_MARGIN requires params.extra_margin_pct > 0"
                },
            )

    # Validate RESTRICT_STAKES params
    if req.action == "RESTRICT_STAKES":
        max_s = req.params.get("max_stake")
        if max_s is None or not isinstance(max_s, int) or max_s <= 0:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "RESTRICT_STAKES requires params.max_stake (positive int)"
                },
            )

    override_id = str(uuid.uuid4())
    now = datetime.now(tz=timezone.utc)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO darts_trader_overrides (
                id, match_id, competition_id, scope, action, market_family,
                params, reason, reason_code, valid_from, valid_until,
                is_active, created_by, created_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,TRUE,$12,$13)
            """,
            override_id,
            req.match_id,
            req.competition_id,
            req.scope,
            req.action,
            req.market_family,
            req.params,
            req.reason,
            req.reason_code,
            now,
            req.valid_until,
            req.created_by,
            now,
        )

    logger.info(
        "trader_override_created",
        override_id=override_id,
        action=req.action,
        scope=req.scope,
        match_id=req.match_id,
        created_by=req.created_by,
    )

    return OverrideResponse(
        id=override_id,
        action=req.action,
        scope=req.scope,
        match_id=req.match_id,
        competition_id=req.competition_id,
        market_family=req.market_family,
        params=req.params,
        reason=req.reason,
        reason_code=req.reason_code,
        valid_from=now.isoformat(),
        valid_until=req.valid_until.isoformat() if req.valid_until else None,
        is_active=True,
        created_by=req.created_by,
        created_at=now.isoformat(),
        revoked_by=None,
        revoked_at=None,
    )


@router.get(
    "/trader/overrides",
    response_model=list[OverrideResponse],
    summary="List active trader overrides",
)
async def list_overrides(
    match_id: Optional[str] = Query(None),
    competition_id: Optional[str] = Query(None),
    active_only: bool = Query(True),
    pool: asyncpg.Pool = Depends(_get_pool),
) -> list[OverrideResponse]:
    """
    List trader overrides, optionally filtered by match or competition.
    """
    async with pool.acquire() as conn:
        conditions = []
        params: list[Any] = []

        if active_only:
            conditions.append("is_active = TRUE")
            conditions.append(
                "(valid_until IS NULL OR valid_until > NOW())"
            )
        if match_id:
            params.append(match_id)
            conditions.append(f"match_id = ${len(params)}")
        if competition_id:
            params.append(competition_id)
            conditions.append(f"competition_id = ${len(params)}")

        where = " AND ".join(conditions) if conditions else "TRUE"
        rows = await conn.fetch(
            f"""
            SELECT id, match_id, competition_id, scope, action, market_family,
                   params, reason, reason_code, valid_from, valid_until,
                   is_active, created_by, created_at, revoked_by, revoked_at
            FROM darts_trader_overrides
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT 500
            """,
            *params,
        )

    return [_row_to_response(r) for r in rows]


@router.get(
    "/trader/overrides/{override_id}",
    response_model=OverrideResponse,
    summary="Retrieve a specific override",
)
async def get_override(
    override_id: str = Path(...),
    pool: asyncpg.Pool = Depends(_get_pool),
) -> OverrideResponse:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, match_id, competition_id, scope, action, market_family,
                   params, reason, reason_code, valid_from, valid_until,
                   is_active, created_by, created_at, revoked_by, revoked_at
            FROM darts_trader_overrides
            WHERE id = $1
            """,
            override_id,
        )
    if not row:
        raise HTTPException(
            status_code=404,
            detail={"error": "override_not_found", "id": override_id},
        )
    return _row_to_response(row)


@router.delete(
    "/trader/overrides/{override_id}",
    summary="Revoke a trader override",
)
async def revoke_override(
    override_id: str = Path(...),
    req: RevokeOverrideRequest = Body(...),
    pool: asyncpg.Pool = Depends(_get_pool),
) -> dict[str, Any]:
    """
    Revoke an active override. Markets resume normal automated pricing.
    """
    now = datetime.now(tz=timezone.utc)
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE darts_trader_overrides
            SET is_active = FALSE, revoked_by = $1, revoked_at = $2
            WHERE id = $3
              AND is_active = TRUE
            """,
            req.revoked_by,
            now,
            override_id,
        )

    if result == "UPDATE 0":
        raise HTTPException(
            status_code=404,
            detail={
                "error": "override_not_found_or_already_revoked",
                "id": override_id,
            },
        )

    logger.info(
        "trader_override_revoked",
        override_id=override_id,
        revoked_by=req.revoked_by,
    )
    return {
        "revoked": True,
        "override_id": override_id,
        "revoked_by": req.revoked_by,
        "revoked_at": now.isoformat(),
    }


@router.get(
    "/trader/overrides/match/{match_id}/effective",
    response_model=EffectiveOverrideResponse,
    summary="Compute effective pricing instructions for a match",
)
async def get_effective_override(
    match_id: str = Path(...),
    market_family: Optional[str] = Query(None),
    pool: asyncpg.Pool = Depends(_get_pool),
) -> EffectiveOverrideResponse:
    """
    Merge all active overrides for a match into a single pricing instruction set.

    Used by the pricing engine before publishing any market price.
    Returns the combined effect of all applicable active overrides.
    """
    now = datetime.now(tz=timezone.utc)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, match_id, competition_id, scope, action, market_family,
                   params, reason, reason_code, valid_from, valid_until,
                   is_active, created_by, created_at, revoked_by, revoked_at
            FROM darts_trader_overrides
            WHERE is_active = TRUE
              AND (valid_until IS NULL OR valid_until > $1)
              AND (
                  match_id = $2
                  OR competition_id IN (
                      SELECT competition_id FROM darts_matches WHERE id = $2
                  )
                  OR scope = 'global'
              )
              AND (market_family IS NULL OR $3::text IS NULL OR market_family = $3)
            ORDER BY created_at DESC
            """,
            now,
            match_id,
            market_family,
        )

    overrides = [_row_to_response(r) for r in rows]

    # Merge into effective instructions
    is_suspended = any(o.action in ("SUSPEND", "PULL") for o in overrides)
    is_pulled = any(o.action == "PULL" for o in overrides)

    extra_margin = sum(
        float(o.params.get("extra_margin_pct", 0.0))
        for o in overrides
        if o.action == "WIDEN_MARGIN"
    )

    stake_limits = [
        int(o.params["max_stake"])
        for o in overrides
        if o.action == "RESTRICT_STAKES" and "max_stake" in o.params
    ]
    max_stake_override = min(stake_limits) if stake_limits else None

    return EffectiveOverrideResponse(
        match_id=match_id,
        is_suspended=is_suspended,
        is_pulled=is_pulled,
        extra_margin_pct=round(extra_margin, 2),
        max_stake_override=max_stake_override,
        active_overrides=overrides,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _row_to_response(row: asyncpg.Record) -> OverrideResponse:
    def _iso(v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)

    return OverrideResponse(
        id=str(row["id"]),
        action=str(row["action"]),
        scope=str(row["scope"]),
        match_id=str(row["match_id"]) if row["match_id"] else None,
        competition_id=str(row["competition_id"]) if row["competition_id"] else None,
        market_family=str(row["market_family"]) if row["market_family"] else None,
        params=dict(row["params"]) if row["params"] else {},
        reason=str(row["reason"]) if row["reason"] else None,
        reason_code=str(row["reason_code"]) if row["reason_code"] else None,
        valid_from=_iso(row["valid_from"]),
        valid_until=_iso(row["valid_until"]),
        is_active=bool(row["is_active"]),
        created_by=str(row["created_by"]),
        created_at=_iso(row["created_at"]),
        revoked_by=str(row["revoked_by"]) if row["revoked_by"] else None,
        revoked_at=_iso(row["revoked_at"]),
    )
