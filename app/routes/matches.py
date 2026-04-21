"""
app/routes/matches.py — Upcoming and active darts fixtures from Optic Odds.

GET /api/v1/darts/matches

Returns paginated upcoming/active fixtures from the Optic Odds REST API.
Data source: Optic Odds /api/v3/fixtures/active?sport=darts

Follows the same response envelope (_ok) and structlog pattern as feeds.py.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Query

from app.config import settings

logger = structlog.get_logger(__name__)
router = APIRouter()

_OPTIC_API_BASE = "https://api.opticodds.com/api/v3"
_HTTP_TIMEOUT_S = 15.0
_OPTIC_SPORT_KEY = "darts"


def _ok(data: Any, request_id: str | None = None) -> dict[str, Any]:
    """Standard XG3 success envelope."""
    return {
        "success": True,
        "data": data,
        "meta": {
            "request_id": request_id or str(uuid.uuid4()),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        },
    }


def _extract_player(fixture: dict, side: str) -> str:
    """Extract player name from an Optic Odds v3 fixture dict.

    Optic Odds v3 uses home_competitors / away_competitors (list of objects)
    and home_team_display / away_team_display strings.
    """
    competitors = fixture.get(f"{side}_competitors")
    if isinstance(competitors, list) and competitors:
        first = competitors[0]
        if isinstance(first, dict):
            return first.get("name", "")
    display = fixture.get(f"{side}_team_display")
    if display and isinstance(display, str):
        return display
    participants = fixture.get("participants") or []
    idx = 0 if side == "home" else 1
    if len(participants) > idx:
        p = participants[idx]
        if isinstance(p, dict):
            return p.get("name", "")
        return str(p) if p else ""
    raw = fixture.get(f"{side}_team", {})
    if isinstance(raw, dict):
        return raw.get("name", "")
    return str(raw) if raw else ""


@router.get(
    "/matches",
    summary="Upcoming and active darts fixtures",
    response_model=None,
    tags=["Matches"],
)
async def darts_matches(
    status: str = Query(
        default="all",
        description="Fixture status filter: unplayed | in_progress | all",
    ),
    page: int = Query(default=1, ge=1, le=50, description="Page number (1-based)"),
    limit: int = Query(default=50, ge=1, le=200, description="Results per page"),
) -> dict[str, Any]:
    """
    Return upcoming and active darts fixtures from Optic Odds.

    Fetches from the Optic Odds /api/v3/fixtures/active endpoint.
    Requires OPTIC_ODDS_API_KEY environment variable.
    """
    request_id = str(uuid.uuid4())
    log = logger.bind(endpoint="darts_matches", request_id=request_id)

    api_key = os.getenv("OPTIC_ODDS_API_KEY", "") or settings.OPTIC_ODDS_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPTIC_ODDS_API_KEY not configured — cannot fetch darts fixtures",
        )

    params: dict[str, Any] = {
        "sport": _OPTIC_SPORT_KEY,
        "page": page,
        "limit": limit,
    }
    if status and status != "all":
        params["status"] = status

    log.info("darts_matches_fetch", params=str(params))

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            resp = await client.get(
                f"{_OPTIC_API_BASE}/fixtures/active",
                headers={"x-api-key": api_key},
                params=params,
            )
    except httpx.HTTPError as exc:
        log.error("optic_odds_http_error", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Optic Odds HTTP error: {exc}") from exc

    if resp.status_code != 200:
        log.error("optic_odds_bad_status", status_code=resp.status_code)
        raise HTTPException(
            status_code=502,
            detail=f"Optic Odds /fixtures/active returned {resp.status_code}: {resp.text[:200]}",
        )

    body = resp.json()
    raw_fixtures: list[dict] = body.get("data", [])
    has_more: bool = body.get("has_more", False)

    fixtures: list[dict[str, Any]] = []
    for raw in raw_fixtures:
        league_obj = raw.get("league") or raw.get("tournament") or {}
        fixtures.append({
            "fixture_id": raw.get("id", ""),
            "league": league_obj.get("name", "") if isinstance(league_obj, dict) else str(league_obj),
            "league_id": league_obj.get("id", "") if isinstance(league_obj, dict) else "",
            "player1_name": _extract_player(raw, "home"),
            "player2_name": _extract_player(raw, "away"),
            "start_date": raw.get("start_date", ""),
            "status": raw.get("status", ""),
        })

    log.info("darts_matches_ok", count=len(fixtures), has_more=has_more)

    return _ok(
        {
            "source": "Optic Odds",
            "sport": _OPTIC_SPORT_KEY,
            "fixtures": fixtures,
            "total": len(fixtures),
            "page": page,
            "limit": limit,
            "has_more": has_more,
        },
        request_id=request_id,
    )
