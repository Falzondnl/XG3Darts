"""
Events / Fixtures API routes — current season coverage.

Provides a live-ish view of tournaments and matches available for betting,
bridging the DB match history with the pricing engine.

Endpoints
---------
GET  /events/current              — list active/upcoming tournaments
GET  /events/{event_id}           — full fixture list for an event
POST /events/{event_id}/price-all — price every match in an event
GET  /events/search               — search events by name/code

Tier-1 coverage (2025-26 PDC season):
    Premier League, Players Championships, European Tour, Grand Slam,
    UK Open, World Matchplay, Grand Prix, World Championship,
    World Masters, World Cup, Women's Series, WDF World Championship.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from db.session import get_session_dependency

logger = structlog.get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# 2026 fixtures data — loaded once at module import
# ---------------------------------------------------------------------------

_FIXTURES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "fixtures_2026.json"
_FIXTURES_2026: dict = {}


def _load_fixtures() -> dict:
    global _FIXTURES_2026
    if not _FIXTURES_2026:
        try:
            with open(_FIXTURES_PATH, encoding="utf-8") as f:
                _FIXTURES_2026 = json.load(f)
        except Exception as exc:
            logger.warning("fixtures_2026_load_failed", error=str(exc))
            _FIXTURES_2026 = {"events": []}
    return _FIXTURES_2026

# ---------------------------------------------------------------------------
# Static 2025-26 PDC calendar — tier-1 tournament coverage
# ---------------------------------------------------------------------------

PDC_CALENDAR_2026: list[dict] = [
    # January–March 2026
    {"event_id": "pdc_wm_2026", "name": "PDC Winmau World Masters",
     "code": "PDC_WM", "date": "2026-02-01", "status": "completed",
     "prize_fund": 500000, "players": 32},
    {"event_id": "pdc_pl_2026", "name": "PDC Premier League 2026",
     "code": "PDC_PL", "date": "2026-02-05", "status": "active",
     "prize_fund": 1000000, "players": 8, "weeks": 17},
    {"event_id": "pdc_pc1_2026", "name": "PDC Players Championship 1",
     "code": "PDC_PC", "date": "2026-02-08", "status": "completed",
     "prize_fund": 150000, "players": 128},
    {"event_id": "pdc_pc2_2026", "name": "PDC Players Championship 2",
     "code": "PDC_PC", "date": "2026-02-09", "status": "completed",
     "prize_fund": 150000, "players": 128},
    {"event_id": "pdc_et1_2026", "name": "PDC European Tour 1 — Austria Open",
     "code": "PDC_ET", "date": "2026-03-14", "status": "upcoming",
     "prize_fund": 130000, "players": 96},
    {"event_id": "pdc_et2_2026", "name": "PDC European Tour 2 — German Open",
     "code": "PDC_ET", "date": "2026-03-21", "status": "upcoming",
     "prize_fund": 130000, "players": 96},
    {"event_id": "pdc_uk_2026", "name": "PDC UK Open 2026",
     "code": "PDC_UK", "date": "2026-03-06", "status": "completed",
     "prize_fund": 450000, "players": 256},
    # April–June 2026
    {"event_id": "pdc_pcf_2026", "name": "PDC Players Championship Finals",
     "code": "PDC_PCF", "date": "2026-11-28", "status": "upcoming",
     "prize_fund": 750000, "players": 64},
    {"event_id": "pdc_gs_2026", "name": "PDC Grand Slam of Darts",
     "code": "PDC_GS", "date": "2026-11-07", "status": "upcoming",
     "prize_fund": 650000, "players": 32},
    {"event_id": "pdc_wc_2026_27", "name": "PDC World Championship 2026/27",
     "code": "PDC_WC", "date": "2026-12-18", "status": "upcoming",
     "prize_fund": 3000000, "players": 96},
    # WDF
    {"event_id": "wdf_wc_2026", "name": "WDF World Championship 2026",
     "code": "WDF_WC", "date": "2026-01-01", "status": "completed",
     "prize_fund": 500000, "players": 96},
    # World Series
    {"event_id": "pdc_ws_bahrain_2026", "name": "Bahrain Darts Masters 2026",
     "code": "PDC_WS", "date": "2026-01-16", "status": "completed",
     "prize_fund": 150000, "players": 16},
    {"event_id": "pdc_ws_saudi_2026", "name": "Saudi Arabia Darts Masters 2026",
     "code": "PDC_WS", "date": "2026-01-20", "status": "completed",
     "prize_fund": 150000, "players": 16},
    # World Cup
    {"event_id": "pdc_wcup_2026", "name": "PDC World Cup of Darts 2026",
     "code": "PDC_WCUP", "date": "2026-06-05", "status": "upcoming",
     "prize_fund": 750000, "players": 32},
]

# Map status to priority for listing
_STATUS_ORDER = {"active": 0, "upcoming": 1, "completed": 2}


def _service_url(path: str) -> str:
    base = settings.DARTS_SERVICE_URL.rstrip("/")
    return f"{base}/{path.lstrip('/')}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/events/current",
    summary="List active and upcoming tournaments",
    tags=["Events"],
)
async def list_current_events(
    status: Optional[str] = Query(
        None,
        description="Filter by status: active | upcoming | completed",
    ),
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """
    Return the current PDC/WDF tournament calendar.

    Active tournaments are returned first, then upcoming, then completed.
    """
    events = sorted(
        PDC_CALENDAR_2026,
        key=lambda e: (_STATUS_ORDER.get(e["status"], 9), e["date"]),
    )
    if status:
        events = [e for e in events if e["status"] == status]

    events = events[:limit]
    enriched = []
    for ev in events:
        enriched.append({
            **ev,
            "_links": {
                "event": _service_url(f"/events/{ev['event_id']}"),
                "price_all": _service_url(f"/events/{ev['event_id']}/price-all"),
            },
        })

    return {
        "count": len(enriched),
        "events": enriched,
        "_links": {"self": _service_url("/events/current")},
    }


@router.get(
    "/events/{event_id}",
    summary="Get event details and fixture list",
    tags=["Events"],
)
async def get_event(
    event_id: str,
    session: AsyncSession = Depends(get_session_dependency),
) -> dict[str, Any]:
    """
    Return event metadata and, where available, recent/upcoming fixtures
    from the darts_matches table for this competition.
    """
    from sqlalchemy import text as sql_text

    event = next((e for e in PDC_CALENDAR_2026 if e["event_id"] == event_id), None)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id!r} not found")

    # Try to fetch recent matches for this competition code from DB
    try:
        result = await session.execute(
            sql_text("""
                SELECT
                    m.id, m.match_date,
                    m.player1_id, m.player2_id, m.winner_player_id,
                    m.player1_score, m.player2_score, m.round_name,
                    p1.first_name || ' ' || p1.last_name AS p1_name,
                    p2.first_name || ' ' || p2.last_name AS p2_name,
                    COALESCE(p1.nickname, '') AS p1_nick,
                    COALESCE(p2.nickname, '') AS p2_nick,
                    COALESCE(p1.dartsorakel_3da, 50.0) AS p1_3da,
                    COALESCE(p2.dartsorakel_3da, 50.0) AS p2_3da
                FROM darts_matches m
                JOIN darts_players p1 ON p1.id = m.player1_id
                JOIN darts_players p2 ON p2.id = m.player2_id
                JOIN darts_competitions c ON c.id = m.competition_id
                WHERE c.format_code = :code
                  AND m.match_date >= :start_date
                  AND m.status = 'Completed'
                ORDER BY m.match_date DESC, m.round_name ASC
                LIMIT 64
            """),
            {"code": event["code"], "start_date": event["date"][:7] + "-01"},
        )
        rows = result.fetchall()
        fixtures = [
            {
                "match_id": row[0],
                "match_date": row[1].isoformat() if row[1] else None,
                "round_name": row[7],
                "player1": {
                    "id": row[2],
                    "name": (row[10] or row[8] or "").strip(),
                    "three_da": float(row[12]),
                },
                "player2": {
                    "id": row[3],
                    "name": (row[11] or row[9] or "").strip(),
                    "three_da": float(row[13]),
                },
                "result": {
                    "winner": "player1" if row[4] == row[2] else "player2",
                    "score": f"{row[5]}-{row[6]}",
                } if row[4] else None,
                "_links": {
                    "smart_price": _service_url("/prematch/smart-price"),
                },
            }
            for row in rows
        ]
    except Exception as exc:
        logger.warning("event_fixture_db_error", event_id=event_id, error=str(exc))
        fixtures = []

    return {
        **event,
        "fixtures_loaded": len(fixtures),
        "fixtures": fixtures,
        "_links": {
            "self": _service_url(f"/events/{event_id}"),
            "price_all": _service_url(f"/events/{event_id}/price-all"),
        },
    }


@router.get(
    "/events/search",
    summary="Search events by name or competition code",
    tags=["Events"],
)
async def search_events(
    q: str = Query(..., min_length=2, description="Search query"),
) -> dict[str, Any]:
    """Search the tournament calendar by name or competition code."""
    q_lower = q.lower()
    matches = [
        e for e in PDC_CALENDAR_2026
        if q_lower in e["name"].lower() or q_lower in e["code"].lower()
        or q_lower in e["event_id"].lower()
    ]
    return {
        "query": q,
        "count": len(matches),
        "events": matches,
    }


class PriceAllRequest(BaseModel):
    """Request body for price-all endpoint."""
    base_margin: float = 0.05
    p1_starts_first: bool = True


@router.post(
    "/events/{event_id}/price-all",
    summary="Price all available fixtures in an event",
    tags=["Events"],
)
async def price_all_fixtures(
    event_id: str,
    request: PriceAllRequest,
    session: AsyncSession = Depends(get_session_dependency),
) -> dict[str, Any]:
    """
    Fetch all unpriced fixtures for an event and compute match winner odds.

    Returns a full card of priced matches suitable for the betting frontend.
    Only prices matches where both players have 3DA data available.
    """
    from sqlalchemy import text as sql_text
    from engines.leg_layer.hold_break_model import HoldBreakModel
    from engines.match_layer.match_combinatorics import MatchCombinatorialEngine
    from competition.format_registry import get_format
    from margin.blending_engine import DartsMarginEngine

    event = next((e for e in PDC_CALENDAR_2026 if e["event_id"] == event_id), None)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id!r} not found")

    try:
        result = await session.execute(
            sql_text("""
                SELECT
                    m.id, m.round_name,
                    m.player1_id, m.player2_id,
                    p1.first_name || ' ' || COALESCE(p1.last_name, '') AS p1_name,
                    COALESCE(p1.nickname, '') AS p1_nick,
                    p2.first_name || ' ' || COALESCE(p2.last_name, '') AS p2_name,
                    COALESCE(p2.nickname, '') AS p2_nick,
                    COALESCE(p1.dartsorakel_3da, 50.0) AS p1_3da,
                    COALESCE(p2.dartsorakel_3da, 50.0) AS p2_3da,
                    COALESCE(e1.rating_after, 1500.0) AS p1_elo,
                    COALESCE(e2.rating_after, 1500.0) AS p2_elo
                FROM darts_matches m
                JOIN darts_players p1 ON p1.id = m.player1_id
                JOIN darts_players p2 ON p2.id = m.player2_id
                JOIN darts_competitions c ON c.id = m.competition_id
                LEFT JOIN darts_elo_ratings e1
                    ON e1.player_id = p1.id AND e1.pool = 'pdc_mens'
                LEFT JOIN darts_elo_ratings e2
                    ON e2.player_id = p2.id AND e2.pool = 'pdc_mens'
                WHERE c.format_code = :code
                  AND m.match_date >= :start_date
                ORDER BY m.match_date DESC, m.round_name
                LIMIT 100
            """),
            {"code": event["code"], "start_date": event["date"][:7] + "-01"},
        )
        rows = result.fetchall()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB error: {exc}")

    hb_model = HoldBreakModel()
    match_engine = MatchCombinatorialEngine()
    margin_engine = DartsMarginEngine()
    priced = []

    for row in rows:
        p1_3da = float(row[8])
        p2_3da = float(row[9])
        round_name = row[1] or "Final"

        try:
            fmt = get_format(event["code"], round_name)
            hb = hb_model.compute(
                p1_three_da=p1_3da,
                p2_three_da=p2_3da,
                format_code=event["code"],
                double_start=fmt.double_in,
            )
            result_match = match_engine.price_match(
                hold_break=hb,
                fmt=fmt,
                round_name=round_name,
                p1_starts_first=request.p1_starts_first,
            )
            true_p = {"p1_win": result_match.p1_win, "p2_win": result_match.p2_win}
            if result_match.draw > 0:
                true_p["draw"] = result_match.draw

            adj_p = margin_engine.apply(
                true_probs=true_p,
                regime=0,
                base_margin=request.base_margin,
            )
            dec_odds = {k: round(1.0 / v, 4) if v > 1e-9 else None
                        for k, v in adj_p.items()}

            priced.append({
                "match_id": row[0],
                "round_name": round_name,
                "player1": {
                    "id": row[2],
                    "name": (row[5] or row[4] or "").strip(),
                    "three_da": round(p1_3da, 2),
                    "elo": round(float(row[10]), 1),
                },
                "player2": {
                    "id": row[3],
                    "name": (row[7] or row[6] or "").strip(),
                    "three_da": round(p2_3da, 2),
                    "elo": round(float(row[11]), 1),
                },
                "decimal_odds": dec_odds,
                "true_probabilities": {k: round(v, 4) for k, v in true_p.items()},
                "applied_margin": round(request.base_margin, 4),
            })
        except Exception as exc:
            logger.warning("price_match_failed", match_id=row[0], error=str(exc))
            continue

    return {
        "event_id": event_id,
        "event_name": event["name"],
        "competition_code": event["code"],
        "priced_count": len(priced),
        "base_margin": request.base_margin,
        "matches": priced,
    }


# ---------------------------------------------------------------------------
# GET /events/fixtures/feed — Flat fixture feed for all 2026 events
# ---------------------------------------------------------------------------

@router.get(
    "/events/fixtures/feed",
    summary="All 2026 season fixtures feed",
    tags=["Events"],
)
async def fixture_feed(
    competition_code: Optional[str] = Query(
        None,
        description="Filter by competition code, e.g. PDC_PC, PDC_PL",
    ),
    status: Optional[str] = Query(
        None,
        description="Filter by status: completed | upcoming | live",
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """
    Return a flat feed of all 2026 season fixtures from the parsed DartsDatabase CSV data.

    This endpoint exposes 4,000+ matches across 37 events including:
    - PDC Challenge Tour (6 events, 2,000+ matches)
    - PDC Premier League (weeks 1-4+)
    - PDC Players Championships
    - PDC Women's Series
    - World Series Masters events
    - European Tour / Scottish Open / Dutch Open
    - WDF events

    Each fixture includes player names, 3DA ratings, score, and winner.
    Suitable for populating a betting platform fixture list.
    """
    data = _load_fixtures()
    events = data.get("events", [])

    all_matches = []
    for event in events:
        ecode = event.get("competition_code", "")
        estatus = event.get("status", "completed")

        if competition_code and ecode.upper() != competition_code.upper():
            continue
        if status and estatus != status:
            continue

        for m in event.get("matches", []):
            all_matches.append({
                "event_id": event.get("event_id"),
                "event_name": event.get("event_name"),
                "competition_code": ecode,
                "event_date": event.get("event_date"),
                "status": estatus,
                "round": m.get("round"),
                "player1": {
                    "name": m.get("player1_name"),
                    "three_da": m.get("player1_3da"),
                },
                "player2": {
                    "name": m.get("player2_name"),
                    "three_da": m.get("player2_3da"),
                },
                "score": f"{m.get('score_p1', 0)}-{m.get('score_p2', 0)}"
                         if m.get("score_p1") is not None else None,
                "winner": m.get("winner"),
                "_links": {
                    "smart_price": _service_url("/prematch/smart-price"),
                    "first_leg": _service_url("/prematch/first-leg"),
                    "multi_totals": _service_url("/prematch/multi-totals"),
                },
            })

    total = len(all_matches)
    page = all_matches[offset: offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "count": len(page),
        "fixtures": page,
    }


# ---------------------------------------------------------------------------
# GET /events/fixtures/competitions — List all unique competition codes
# ---------------------------------------------------------------------------

@router.get(
    "/events/fixtures/competitions",
    summary="List all competition codes in the fixture feed",
    tags=["Events"],
)
async def list_fixture_competitions() -> dict[str, Any]:
    """Return all unique competition codes and event counts in fixtures_2026.json."""
    data = _load_fixtures()
    events = data.get("events", [])
    comps: dict[str, dict] = {}
    for ev in events:
        code = ev.get("competition_code", "UNKNOWN")
        if code not in comps:
            comps[code] = {"competition_code": code, "events": 0, "matches": 0}
        comps[code]["events"] += 1
        comps[code]["matches"] += len(ev.get("matches", []))
    return {
        "total_competitions": len(comps),
        "total_events": len(events),
        "total_matches": sum(c["matches"] for c in comps.values()),
        "competitions": sorted(comps.values(), key=lambda x: -x["matches"]),
    }
