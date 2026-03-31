"""
Trading controls — manual suspend / resume for darts matches and markets.

Human traders can intervene in automated pricing in response to:
  - Player withdrawal or injury mid-match
  - Breaking news affecting a fixture
  - Suspicious betting patterns detected by risk
  - Technical feed issues requiring a market pull

All state is stored in-process (module-level dicts and sets).
On service restart, state resets to fully open.  For persistent
suspension state, integrate with the existing trader overrides table
in trader.py (SUSPEND/RESUME actions).

Endpoints
---------
POST /api/v1/darts/trading/suspend/match/{match_id}
    Suspend all markets for a match.

POST /api/v1/darts/trading/resume/match/{match_id}
    Resume all markets for a match.

POST /api/v1/darts/trading/suspend/market
    Suspend a specific market key within a match.

POST /api/v1/darts/trading/resume/market
    Resume a specific market key within a match.

GET  /api/v1/darts/trading/status/{match_id}
    Return the suspension status for a match and its known market keys.

GET  /api/v1/darts/trading/status
    Return all suspended matches and markets.

GET  /api/v1/darts/trading/health
    Liveness check for this router.

Public helpers (importable by other route modules)
--------------------------------------------------
is_match_suspended(match_id: str) -> bool
is_market_suspended(match_id: str, market_key: str) -> bool
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional, Set

import structlog
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# In-process suspension state
# ---------------------------------------------------------------------------

# Fully-suspended matches: {match_id}
_SUSPENDED_MATCHES: Set[str] = set()

# Per-market suspension: {match_id: {market_key}}
_SUSPENDED_MARKETS: Dict[str, Set[str]] = {}

# Audit log — most-recent 200 events, ring-buffer style
_AUDIT_LOG: list[Dict[str, Any]] = []
_AUDIT_LOG_MAX = 200


def _audit(action: str, match_id: str, market_key: Optional[str], reason: str) -> None:
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": action,
        "match_id": match_id,
        "market_key": market_key,
        "reason": reason,
    }
    _AUDIT_LOG.append(entry)
    if len(_AUDIT_LOG) > _AUDIT_LOG_MAX:
        _AUDIT_LOG.pop(0)
    logger.info(
        "trading_control_action",
        **entry,
    )


# ---------------------------------------------------------------------------
# Public helpers — imported by pricing engines to gate price publication
# ---------------------------------------------------------------------------


def is_match_suspended(match_id: str) -> bool:
    """Return True if the match is globally suspended."""
    return match_id in _SUSPENDED_MATCHES


def is_market_suspended(match_id: str, market_key: str) -> bool:
    """Return True if the match OR the specific market key is suspended."""
    if match_id in _SUSPENDED_MATCHES:
        return True
    return market_key in _SUSPENDED_MARKETS.get(match_id, set())


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class MatchSuspendRequest(BaseModel):
    reason: str = Field(
        default="Trader override",
        max_length=256,
        description="Human-readable reason for the suspension.",
        examples=["Player injury — awaiting official confirmation"],
    )


class MarketSuspendRequest(BaseModel):
    match_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Match identifier.",
    )
    market_key: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Market key, e.g. 'match_winner', 'handicap_-1.5', 'total_legs_ou_4.5'.",
        examples=["match_winner"],
    )
    reason: str = Field(
        default="Trader override",
        max_length=256,
        description="Human-readable reason for the suspension.",
    )


# ---------------------------------------------------------------------------
# Route: POST /trading/suspend/match/{match_id}
# ---------------------------------------------------------------------------


@router.post(
    "/trading/suspend/match/{match_id}",
    summary="Suspend all markets for a match",
    tags=["Trading Controls"],
)
async def suspend_match(
    match_id: str,
    body: MatchSuspendRequest = Body(default_factory=MatchSuspendRequest),
) -> Dict[str, Any]:
    """
    Globally suspend a match.  All pricing calls for this match_id will
    be blocked until a corresponding resume is issued.
    """
    already = match_id in _SUSPENDED_MATCHES
    _SUSPENDED_MATCHES.add(match_id)
    _audit("SUSPEND_MATCH", match_id, None, body.reason)
    return {
        "success": True,
        "data": {
            "match_id": match_id,
            "action": "suspended",
            "already_suspended": already,
            "reason": body.reason,
        },
        "meta": {"request_id": str(uuid.uuid4())},
    }


# ---------------------------------------------------------------------------
# Route: POST /trading/resume/match/{match_id}
# ---------------------------------------------------------------------------


@router.post(
    "/trading/resume/match/{match_id}",
    summary="Resume all markets for a match",
    tags=["Trading Controls"],
)
async def resume_match(
    match_id: str,
    body: MatchSuspendRequest = Body(default_factory=MatchSuspendRequest),
) -> Dict[str, Any]:
    """Resume a globally suspended match."""
    was_suspended = match_id in _SUSPENDED_MATCHES
    _SUSPENDED_MATCHES.discard(match_id)
    _audit("RESUME_MATCH", match_id, None, body.reason)
    return {
        "success": True,
        "data": {
            "match_id": match_id,
            "action": "resumed",
            "was_suspended": was_suspended,
            "reason": body.reason,
        },
        "meta": {"request_id": str(uuid.uuid4())},
    }


# ---------------------------------------------------------------------------
# Route: POST /trading/suspend/market
# ---------------------------------------------------------------------------


@router.post(
    "/trading/suspend/market",
    summary="Suspend a specific market key within a match",
    tags=["Trading Controls"],
)
async def suspend_market(body: MarketSuspendRequest) -> Dict[str, Any]:
    """
    Suspend a single market key (e.g. 'match_winner', 'handicap_-1.5')
    within a match without affecting other markets.
    """
    if body.match_id not in _SUSPENDED_MARKETS:
        _SUSPENDED_MARKETS[body.match_id] = set()
    already = body.market_key in _SUSPENDED_MARKETS[body.match_id]
    _SUSPENDED_MARKETS[body.match_id].add(body.market_key)
    _audit("SUSPEND_MARKET", body.match_id, body.market_key, body.reason)
    return {
        "success": True,
        "data": {
            "match_id": body.match_id,
            "market_key": body.market_key,
            "action": "suspended",
            "already_suspended": already,
            "reason": body.reason,
        },
        "meta": {"request_id": str(uuid.uuid4())},
    }


# ---------------------------------------------------------------------------
# Route: POST /trading/resume/market
# ---------------------------------------------------------------------------


@router.post(
    "/trading/resume/market",
    summary="Resume a specific market key within a match",
    tags=["Trading Controls"],
)
async def resume_market(body: MarketSuspendRequest) -> Dict[str, Any]:
    """Resume a previously suspended market key."""
    was_suspended = body.market_key in _SUSPENDED_MARKETS.get(body.match_id, set())
    if body.match_id in _SUSPENDED_MARKETS:
        _SUSPENDED_MARKETS[body.match_id].discard(body.market_key)
        if not _SUSPENDED_MARKETS[body.match_id]:
            del _SUSPENDED_MARKETS[body.match_id]
    _audit("RESUME_MARKET", body.match_id, body.market_key, body.reason)
    return {
        "success": True,
        "data": {
            "match_id": body.match_id,
            "market_key": body.market_key,
            "action": "resumed",
            "was_suspended": was_suspended,
            "reason": body.reason,
        },
        "meta": {"request_id": str(uuid.uuid4())},
    }


# ---------------------------------------------------------------------------
# Route: GET /trading/status/{match_id}
# ---------------------------------------------------------------------------


@router.get(
    "/trading/status/{match_id}",
    summary="Suspension status for a match",
    tags=["Trading Controls"],
)
async def match_trading_status(match_id: str) -> Dict[str, Any]:
    """
    Return the current suspension state for a match and all known
    market keys with per-market suspension status.
    """
    match_suspended = match_id in _SUSPENDED_MATCHES
    market_keys = _SUSPENDED_MARKETS.get(match_id, set())
    return {
        "success": True,
        "data": {
            "match_id": match_id,
            "match_suspended": match_suspended,
            "suspended_markets": sorted(market_keys),
            "any_suspended": match_suspended or bool(market_keys),
        },
        "meta": {"request_id": str(uuid.uuid4())},
    }


# ---------------------------------------------------------------------------
# Route: GET /trading/status
# ---------------------------------------------------------------------------


@router.get(
    "/trading/status",
    summary="All active suspensions",
    tags=["Trading Controls"],
)
async def all_trading_status() -> Dict[str, Any]:
    """
    Return the full suspension state: all suspended matches and all
    matches with one or more suspended market keys.
    """
    return {
        "success": True,
        "data": {
            "suspended_matches": sorted(_SUSPENDED_MATCHES),
            "suspended_markets": {
                mid: sorted(keys)
                for mid, keys in _SUSPENDED_MARKETS.items()
                if keys
            },
            "total_suspended_matches": len(_SUSPENDED_MATCHES),
            "total_matches_with_suspended_markets": len(
                [m for m in _SUSPENDED_MARKETS if _SUSPENDED_MARKETS[m]]
            ),
            "recent_audit_log": _AUDIT_LOG[-20:],
        },
        "meta": {"request_id": str(uuid.uuid4())},
    }


# ---------------------------------------------------------------------------
# Route: GET /trading/health
# ---------------------------------------------------------------------------


@router.get(
    "/trading/health",
    summary="Health check for the trading controls router",
    tags=["Trading Controls"],
    include_in_schema=True,
)
async def trading_health() -> Dict[str, Any]:
    """Liveness probe for the darts trading controls subsystem."""
    return {
        "status": "ok",
        "service": "xg3-darts-trading-controls",
        "suspended_matches_count": len(_SUSPENDED_MATCHES),
        "matches_with_suspended_markets": len(
            [m for m in _SUSPENDED_MARKETS if _SUSPENDED_MARKETS[m]]
        ),
    }
