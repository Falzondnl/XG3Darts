"""
Data-feed API routes — Flashscore and Darts24 live/recent match data.

Exposes scraper functionality as REST endpoints so the platform can
trigger data collection and retrieve scraped match records on-demand.

Endpoints
---------
GET  /feeds/status                — Status of all registered data sources
GET  /feeds/flashscore/live       — Currently live matches from Flashscore
GET  /feeds/darts24/live          — Currently live matches from Darts24
POST /feeds/flashscore/scrape     — Background scrape of recent matches (days_back)
POST /feeds/darts24/scrape        — Background scrape of a competition (slug)
GET  /feeds/flashscore/recent     — Recent scraped results (days_back query param)

Authentication
--------------
Applied globally via api_key_middleware in main.py — no per-route dependency
needed here.  In development mode (API_KEYS not set) all requests are open.

Notes
-----
Both scrapers are native asyncio (aiohttp-based), so they are awaited
directly in the async route handlers without run_in_executor.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from db.session import get_session_dependency

try:
    from data.sources.darts24_client import Darts24Client
    from data.sources.flashscore_client import FlashscoreClient
    _SCRAPERS_AVAILABLE = True
except ImportError:
    Darts24Client = None  # type: ignore[assignment,misc]
    FlashscoreClient = None  # type: ignore[assignment,misc]
    _SCRAPERS_AVAILABLE = False

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/feeds", tags=["Feeds"])


# ---------------------------------------------------------------------------
# Shared response envelope helpers
# ---------------------------------------------------------------------------


def _ok(data: Any, request_id: str | None = None) -> dict[str, Any]:
    """Build the standard XG3 success envelope."""
    return {
        "success": True,
        "data": data,
        "meta": {
            "request_id": request_id or str(uuid.uuid4()),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        },
    }


def _service_unavailable(source: str, detail: str, request_id: str) -> HTTPException:
    """Build a 503 HTTPException for scraper/upstream failures."""
    logger.error(
        "feed_source_unavailable",
        source=source,
        detail=detail,
        request_id=request_id,
    )
    raise HTTPException(
        status_code=503,
        detail={
            "error": "feed_source_unavailable",
            "source": source,
            "message": detail,
            "request_id": request_id,
        },
    )


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class MatchRecord(BaseModel):
    """Normalised match record returned by feed endpoints."""

    match_id: str
    competition_name: str
    player1_name: str
    player2_name: str
    score1: Optional[int] = None
    score2: Optional[int] = None
    match_date: Optional[str] = None
    status: str
    source_url: str


class LiveMatchRecord(BaseModel):
    """Live match record from Darts24 (less structured — embedded JSON varies)."""

    player1_name: Optional[str] = None
    player2_name: Optional[str] = None
    score1: Optional[Any] = None
    score2: Optional[Any] = None
    match_date: Optional[str] = None
    status: str = "Live"
    raw_text: Optional[str] = None


class FlashscoreScrapeRequest(BaseModel):
    """Request body for POST /feeds/flashscore/scrape."""

    days_back: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of calendar days to look back for recent matches.",
    )


class Darts24ScrapeRequest(BaseModel):
    """Request body for POST /feeds/darts24/scrape."""

    competition_slug: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "Darts24 competition path slug, "
            "e.g. 'pdc-world-championship-2025'."
        ),
    )


class SourceStatus(BaseModel):
    """Status descriptor for a single data source."""

    name: str
    type: str
    status: str
    description: str
    notes: Optional[str] = None


class FeedsStatusResponse(BaseModel):
    """Response model for GET /feeds/status."""

    sources: list[SourceStatus]
    checked_at: str


class FlashscoreLiveResponse(BaseModel):
    """Response model for GET /feeds/flashscore/live."""

    source: str
    live_count: int
    matches: list[MatchRecord]


class Darts24LiveResponse(BaseModel):
    """Response model for GET /feeds/darts24/live."""

    source: str
    live_count: int
    matches: list[LiveMatchRecord]


class ScrapeAcceptedResponse(BaseModel):
    """Response model for POST /feeds/*/scrape (background task accepted)."""

    accepted: bool
    task: str
    message: str


class FlashscoreRecentResponse(BaseModel):
    """Response model for GET /feeds/flashscore/recent."""

    source: str
    days_back: int
    match_count: int
    matches: list[MatchRecord]


# ---------------------------------------------------------------------------
# Background task functions
# ---------------------------------------------------------------------------


async def _bg_flashscore_scrape(days_back: int, request_id: str) -> None:
    """
    Background task: scrape recent Flashscore matches and persist to disk.

    Errors are logged but not re-raised — the HTTP response has already
    been sent when this function runs.
    """
    log = logger.bind(task="bg_flashscore_scrape", days_back=days_back, request_id=request_id)
    log.info("flashscore_bg_scrape_start")
    try:
        client = FlashscoreClient()
        results = await client.scrape_recent_matches(days_back=days_back)
        log.info("flashscore_bg_scrape_complete", result_count=len(results))
    except Exception as exc:
        log.error("flashscore_bg_scrape_failed", error=str(exc), error_type=type(exc).__name__)


async def _bg_darts24_scrape(competition_slug: str, request_id: str) -> None:
    """
    Background task: scrape a Darts24 competition and persist to disk.

    Errors are logged but not re-raised.
    """
    log = logger.bind(
        task="bg_darts24_scrape",
        competition_slug=competition_slug,
        request_id=request_id,
    )
    log.info("darts24_bg_scrape_start")
    try:
        client = Darts24Client()
        results = await client.scrape_competition(competition_slug=competition_slug)
        log.info("darts24_bg_scrape_complete", result_count=len(results))
    except Exception as exc:
        log.error("darts24_bg_scrape_failed", error=str(exc), error_type=type(exc).__name__)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/status",
    summary="Data source status",
    description=(
        "Returns the current status of all registered data sources "
        "(Optic Odds, Flashscore, Darts24, DartsOrakel, Mastercaller). "
        "Status values reflect configuration availability, not live connectivity."
    ),
)
async def get_feeds_status() -> dict[str, Any]:
    """Return status of all data sources."""
    import os

    request_id = str(uuid.uuid4())
    log = logger.bind(endpoint="feeds_status", request_id=request_id)
    log.info("feeds_status_requested")

    sources: list[dict[str, Any]] = [
        {
            "name": "Optic Odds",
            "type": "live_feed",
            "status": "configured" if os.environ.get("OPTIC_ODDS_API_KEY") else "unconfigured",
            "description": "Primary live odds and match data feed via HTTP + RabbitMQ.",
            "notes": (
                "OPTIC_ODDS_API_KEY environment variable "
                + ("is set." if os.environ.get("OPTIC_ODDS_API_KEY") else "is NOT set.")
            ),
        },
        {
            "name": "Flashscore",
            "type": "scraper",
            "status": "available",
            "description": (
                "HTML scraper for darts match results and live scores. "
                "Rate-limited to 2 req/5s per Flashscore TOS."
            ),
            "notes": "Use /feeds/flashscore/live and /feeds/flashscore/recent.",
        },
        {
            "name": "Darts24",
            "type": "scraper",
            "status": "available",
            "description": (
                "Scraper for throw-by-throw live coverage and leg results. "
                "Extracts embedded JSON from page HTML. Rate-limited to 1 req/3s."
            ),
            "notes": "Use /feeds/darts24/live and /feeds/darts24/scrape.",
        },
        {
            "name": "DartsOrakel",
            "type": "scraper",
            "status": "available",
            "description": (
                "3,358 player career averages and 24k H2H records "
                "scraped and stored. Primary model features for R1 regime."
            ),
            "notes": "Data already scraped; stored in DB / raw JSON files.",
        },
        {
            "name": "Mastercaller",
            "type": "scraper",
            "status": "discovered",
            "description": (
                "836 match links discovered. "
                "180-score and checkout enrichment source."
            ),
            "notes": "Scraping of match data not yet complete.",
        },
    ]

    return _ok(
        {
            "sources": sources,
            "checked_at": datetime.now(tz=timezone.utc).isoformat(),
        },
        request_id=request_id,
    )


@router.get(
    "/flashscore/live",
    response_model=None,
    summary="Flashscore live matches",
    description=(
        "Scrapes and returns currently live darts matches from Flashscore. "
        "Makes a live HTTP request to Flashscore — latency reflects network "
        "conditions. Returns 503 if Flashscore is unreachable."
    ),
)
async def get_flashscore_live() -> dict[str, Any]:
    """Return currently live darts matches from Flashscore."""
    request_id = str(uuid.uuid4())
    log = logger.bind(endpoint="flashscore_live", request_id=request_id)
    log.info("flashscore_live_requested")

    client = FlashscoreClient()
    try:
        raw_results = await client.scrape_live_matches()
    except Exception as exc:
        _service_unavailable(
            source="Flashscore",
            detail=f"Live scrape failed: {type(exc).__name__}: {exc}",
            request_id=request_id,
        )

    matches: list[dict[str, Any]] = [
        {
            "match_id": r.match_id,
            "competition_name": r.competition_name,
            "player1_name": r.player1_name,
            "player2_name": r.player2_name,
            "score1": r.score1,
            "score2": r.score2,
            "match_date": r.match_date.isoformat() if r.match_date else None,
            "status": r.status,
            "source_url": r.source_url,
        }
        for r in raw_results
    ]

    log.info("flashscore_live_ok", match_count=len(matches))
    return _ok(
        {
            "source": "Flashscore",
            "live_count": len(matches),
            "matches": matches,
        },
        request_id=request_id,
    )


@router.get(
    "/darts24/live",
    response_model=None,
    summary="Darts24 live matches",
    description=(
        "Scrapes and returns currently live darts matches from Darts24. "
        "Extracts embedded JSON from the Darts24 live page. "
        "Returns 503 if Darts24 is unreachable."
    ),
)
async def get_darts24_live() -> dict[str, Any]:
    """Return currently live darts matches from Darts24."""
    request_id = str(uuid.uuid4())
    log = logger.bind(endpoint="darts24_live", request_id=request_id)
    log.info("darts24_live_requested")

    client = Darts24Client()
    try:
        raw_matches = await client.scrape_live_matches()
    except Exception as exc:
        _service_unavailable(
            source="Darts24",
            detail=f"Live scrape failed: {type(exc).__name__}: {exc}",
            request_id=request_id,
        )

    log.info("darts24_live_ok", match_count=len(raw_matches))
    return _ok(
        {
            "source": "Darts24",
            "live_count": len(raw_matches),
            "matches": raw_matches,
        },
        request_id=request_id,
    )


@router.post(
    "/flashscore/scrape",
    response_model=None,
    status_code=202,
    summary="Trigger Flashscore background scrape",
    description=(
        "Triggers a non-blocking background scrape of recent Flashscore darts "
        "match results. Returns 202 Accepted immediately. "
        "Results are saved to the configured output directory. "
        "Use GET /feeds/flashscore/recent to retrieve the scraped data."
    ),
)
async def post_flashscore_scrape(
    body: FlashscoreScrapeRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Enqueue a background Flashscore scrape for recent matches."""
    request_id = str(uuid.uuid4())
    log = logger.bind(
        endpoint="flashscore_scrape",
        days_back=body.days_back,
        request_id=request_id,
    )
    log.info("flashscore_scrape_enqueued")

    background_tasks.add_task(_bg_flashscore_scrape, body.days_back, request_id)

    return _ok(
        {
            "accepted": True,
            "task": "flashscore_recent_scrape",
            "message": (
                f"Background scrape enqueued for the last {body.days_back} day(s). "
                "Results will be saved to the Flashscore output directory."
            ),
        },
        request_id=request_id,
    )


@router.post(
    "/darts24/scrape",
    response_model=None,
    status_code=202,
    summary="Trigger Darts24 competition scrape",
    description=(
        "Triggers a non-blocking background scrape of a Darts24 competition. "
        "Returns 202 Accepted immediately. "
        "Provide the competition path slug from the Darts24 URL, "
        "e.g. 'pdc-world-championship-2025'."
    ),
)
async def post_darts24_scrape(
    body: Darts24ScrapeRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Enqueue a background Darts24 competition scrape."""
    request_id = str(uuid.uuid4())
    log = logger.bind(
        endpoint="darts24_scrape",
        competition_slug=body.competition_slug,
        request_id=request_id,
    )
    log.info("darts24_competition_scrape_enqueued")

    background_tasks.add_task(_bg_darts24_scrape, body.competition_slug, request_id)

    return _ok(
        {
            "accepted": True,
            "task": "darts24_competition_scrape",
            "message": (
                f"Background scrape enqueued for competition '{body.competition_slug}'. "
                "Results will be saved to the Darts24 output directory."
            ),
        },
        request_id=request_id,
    )


@router.get(
    "/flashscore/recent",
    response_model=None,
    summary="Recent Flashscore results (live scrape)",
    description=(
        "Scrapes and returns recent darts match results from Flashscore. "
        "Unlike the background POST endpoint this runs the scrape inline "
        "and returns the results immediately. "
        "Latency is proportional to the number of matches found. "
        "Returns 503 if Flashscore is unreachable."
    ),
)
async def get_flashscore_recent(
    days_back: int = Query(
        default=7,
        ge=1,
        le=365,
        description="Number of calendar days to look back for match results.",
    ),
) -> dict[str, Any]:
    """Scrape and return recent Flashscore match results."""
    request_id = str(uuid.uuid4())
    log = logger.bind(
        endpoint="flashscore_recent",
        days_back=days_back,
        request_id=request_id,
    )
    log.info("flashscore_recent_requested")

    client = FlashscoreClient()
    try:
        raw_results = await client.scrape_recent_matches(days_back=days_back)
    except Exception as exc:
        _service_unavailable(
            source="Flashscore",
            detail=f"Recent scrape failed: {type(exc).__name__}: {exc}",
            request_id=request_id,
        )

    matches: list[dict[str, Any]] = [
        {
            "match_id": r.match_id,
            "competition_name": r.competition_name,
            "player1_name": r.player1_name,
            "player2_name": r.player2_name,
            "score1": r.score1,
            "score2": r.score2,
            "match_date": r.match_date.isoformat() if r.match_date else None,
            "status": r.status,
            "source_url": r.source_url,
        }
        for r in raw_results
    ]

    log.info("flashscore_recent_ok", match_count=len(matches))
    return _ok(
        {
            "source": "Flashscore",
            "days_back": days_back,
            "match_count": len(matches),
            "matches": matches,
        },
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Optic Odds fixture discovery + auto-pricing
# ---------------------------------------------------------------------------

# League-to-competition_code mapping for format resolution
_LEAGUE_TO_COMP: dict[str, str] = {
    "premier league": "PDC_PL",
    "world championship": "PDC_WC",
    "world matchplay": "PDC_WM",
    "grand prix": "PDC_GP",
    "uk open": "PDC_UK",
    "grand slam": "PDC_GS",
    "european tour": "PDC_ET",
    "players championship": "PDC_PC",
    "players championship finals": "PDC_PCF",
    "masters": "PDC_MASTERS",
    "world series": "PDC_WS",
    "world cup": "PDC_WCUP",
    "women's series": "PDC_WOM_SERIES",
    "women's world matchplay": "PDC_WWM",
    "development tour": "PDC_DEVTOUR",
    "challenge tour": "PDC_CHALLENGE",
    "world youth championship": "PDC_WYC",
    "wdf world championship": "WDF_WC",
    "wdf europe cup": "WDF_EC",
    "wdf open": "WDF_OPEN",
}


def _map_league_to_comp(league_name: str) -> str:
    """Map an Optic Odds league name to a competition_code.

    Uses substring matching against known PDC/WDF competition names.
    Falls back to PDC_PC (Players Championship format) which has a
    sensible best-of-11 legs default.
    """
    league_lower = league_name.lower()
    for key, comp in _LEAGUE_TO_COMP.items():
        if key in league_lower:
            return comp
    return "PDC_PC"


@router.get(
    "/optic-odds/fixtures",
    summary="Active darts fixtures from Optic Odds with optional auto-pricing",
    tags=["Feeds"],
)
async def optic_odds_fixtures(
    auto_price: bool = Query(
        default=True,
        description="Attempt to price each fixture via Markov engine",
    ),
    max_price: int = Query(
        default=50, ge=1, le=200,
        description="Max fixtures to auto-price per request",
    ),
    session: AsyncSession = Depends(get_session_dependency),
) -> dict[str, Any]:
    """Discover active/scheduled darts fixtures from Optic Odds REST API.

    For each fixture, the endpoint:
    1. Fetches active fixtures from Optic Odds ``/fixtures/active?sport=darts``
    2. Looks up each player in the ``darts_players`` DB by name to get their 3DA
    3. Maps the league to a ``competition_code`` for format resolution
    4. Prices the fixture via the Markov hold-break engine

    Fixtures that cannot be priced (unknown player, no 3DA, format error)
    are returned with ``priced=false`` and an ``error`` field.
    """
    import os
    import httpx
    from sqlalchemy import text as sql_text

    request_id = str(uuid.uuid4())
    log = logger.bind(request_id=request_id)

    api_key = os.getenv("OPTIC_ODDS_API_KEY", "") or settings.OPTIC_ODDS_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPTIC_ODDS_API_KEY not configured — cannot fetch darts fixtures",
        )

    # --- Fetch fixtures from Optic Odds ---
    try:
        all_raw: list[dict] = []
        async with httpx.AsyncClient(timeout=15.0) as client:
            for page in range(1, 6):
                resp = await client.get(
                    "https://api.opticodds.com/api/v3/fixtures/active",
                    headers={"x-api-key": api_key},
                    params={"sport": "darts", "page": page, "limit": 100},
                )
                if resp.status_code != 200:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Optic Odds /fixtures/active returned {resp.status_code}",
                    )
                body = resp.json()
                all_raw.extend(body.get("data", []))
                if not body.get("has_more", False):
                    break
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Optic Odds HTTP error: {exc}") from exc

    log.info("optic_odds_darts_raw_fetched", count=len(all_raw))

    # --- Parse fixtures ---
    # BUG-DARTS-PLAYER-NAME-001: Optic Odds v3 uses
    # "home_competitors"/"away_competitors" (array of Competitor objects)
    # and "home_team_display"/"away_team_display" strings instead of
    # the v2 "participants" array. Extract names with fallback chain.
    def _extract_player(fixture: dict, side: str) -> str:
        """Extract player name from Optic Odds v3 fixture dict."""
        # 1. home_competitors / away_competitors (primary v3 field)
        competitors = fixture.get(f"{side}_competitors")
        if isinstance(competitors, list) and competitors:
            first = competitors[0]
            if isinstance(first, dict):
                return first.get("name", "")
        # 2. home_team_display / away_team_display
        display = fixture.get(f"{side}_team_display")
        if display and isinstance(display, str):
            return display
        # 3. Legacy participants array (v2)
        participants = fixture.get("participants") or []
        idx = 0 if side == "home" else 1
        if len(participants) > idx:
            p = participants[idx]
            if isinstance(p, dict):
                return p.get("name", "")
            return str(p) if p else ""
        # 4. Legacy home_team / away_team (dict or string)
        raw = fixture.get(f"{side}_team", {})
        if isinstance(raw, dict):
            return raw.get("name", "")
        return str(raw) if raw else ""

    fixtures: list[dict[str, Any]] = []
    for raw in all_raw:
        league = raw.get("league") or raw.get("tournament") or {}

        fixtures.append({
            "fixture_id": raw.get("id", ""),
            "league": league.get("name", "") if isinstance(league, dict) else str(league),
            "league_id": league.get("id", "") if isinstance(league, dict) else "",
            "player1_name": _extract_player(raw, "home"),
            "player2_name": _extract_player(raw, "away"),
            "start_date": raw.get("start_date", ""),
            "status": raw.get("status", "not_started"),
        })

    # --- Auto-pricing phase ---
    priced_count = 0
    pricing_errors = 0

    if auto_price and fixtures:
        from competition.format_registry import get_format, DartsFormatError
        from engines.leg_layer.hold_break_model import HoldBreakModel
        from engines.match_layer.match_combinatorics import MatchCombinatorialEngine
        from margin.blending_engine import DartsMarginEngine
        from margin.shin_margin import ShinMarginModel

        hb_model = HoldBreakModel()
        match_engine = MatchCombinatorialEngine()
        margin_engine = DartsMarginEngine()
        shin_model = ShinMarginModel()

        for i, fx in enumerate(fixtures):
            if i >= max_price:
                fx["priced"] = False
                fx["pricing_error"] = "max_price limit reached"
                continue

            p1_name = fx.get("player1_name", "")
            p2_name = fx.get("player2_name", "")
            league_name = fx.get("league", "")

            if not p1_name or not p2_name:
                fx["priced"] = False
                fx["pricing_error"] = "Missing player name(s)"
                pricing_errors += 1
                continue

            # Look up player 3DA from DB by name (fuzzy match on first_name + last_name)
            try:
                p1_row = (await session.execute(
                    sql_text("""
                        SELECT id, dartsorakel_3da FROM darts_players
                        WHERE LOWER(CONCAT(first_name, ' ', last_name)) = LOWER(:name)
                        OR LOWER(nickname) = LOWER(:name)
                        LIMIT 1
                    """),
                    {"name": p1_name.strip()},
                )).fetchone()

                p2_row = (await session.execute(
                    sql_text("""
                        SELECT id, dartsorakel_3da FROM darts_players
                        WHERE LOWER(CONCAT(first_name, ' ', last_name)) = LOWER(:name)
                        OR LOWER(nickname) = LOWER(:name)
                        LIMIT 1
                    """),
                    {"name": p2_name.strip()},
                )).fetchone()
            except Exception as db_err:
                fx["priced"] = False
                fx["pricing_error"] = f"DB lookup error: {type(db_err).__name__}"
                pricing_errors += 1
                continue

            p1_3da = p1_row.dartsorakel_3da if p1_row else None
            p2_3da = p2_row.dartsorakel_3da if p2_row else None
            p1_id = p1_row.id if p1_row else p1_name.lower().replace(" ", "-")
            p2_id = p2_row.id if p2_row else p2_name.lower().replace(" ", "-")

            if not p1_3da or not p2_3da:
                fx["priced"] = False
                fx["pricing_error"] = (
                    f"Missing 3DA: p1={'found' if p1_3da else 'NOT_FOUND'}, "
                    f"p2={'found' if p2_3da else 'NOT_FOUND'}"
                )
                fx["p1_found"] = p1_row is not None
                fx["p2_found"] = p2_row is not None
                pricing_errors += 1
                continue

            # Map league to competition_code and resolve format
            comp_code = _map_league_to_comp(league_name)
            try:
                fmt = get_format(comp_code)
                round_fmt = next(iter(fmt.per_round.values()))  # Default to first round
            except (DartsFormatError, IndexError, AttributeError, StopIteration):
                # Fallback: generic best-of-11 legs
                comp_code = "PDC_PC"
                try:
                    fmt = get_format(comp_code)
                    round_fmt = next(iter(fmt.per_round.values()))
                except Exception:
                    fx["priced"] = False
                    fx["pricing_error"] = f"Format resolution failed for {comp_code}"
                    pricing_errors += 1
                    continue

            # Compute hold/break probabilities from Markov engine
            try:
                legs_to_win = round_fmt.legs_to_win or 6
                hb = hb_model.compute_from_3da(
                    p1_id=p1_id,
                    p2_id=p2_id,
                    p1_three_da=p1_3da,
                    p2_three_da=p2_3da,
                )
                p1_prob = match_engine.p1_win_probability(
                    hb=hb,
                    legs_to_win=legs_to_win,
                    p1_starts=True,
                )
                p2_prob = 1.0 - p1_prob

                # Apply margin via 5-factor engine + Shin model
                base_margin = 0.05
                final_margin = margin_engine.compute_margin(
                    base_margin=base_margin,
                    regime=1,
                    starter_confidence=1.0,
                    source_confidence=1.0,
                    model_agreement=1.0,
                    market_liquidity="high",
                    ecosystem=fmt.ecosystem,
                )
                adjusted = shin_model.apply_shin_margin(
                    true_probs={"p1": p1_prob, "p2": p2_prob},
                    target_margin=final_margin,
                )
                p1_implied = adjusted["p1"]
                p2_implied = adjusted["p2"]
                p1_odds = round(1.0 / p1_implied, 3) if p1_implied > 0 else 999.0
                p2_odds = round(1.0 / p2_implied, 3) if p2_implied > 0 else 999.0
                overround = round(p1_implied + p2_implied - 1.0, 4)

                fx["priced"] = True
                fx["p1_win_prob"] = round(p1_prob, 4)
                fx["p2_win_prob"] = round(p2_prob, 4)
                fx["p1_odds"] = p1_odds
                fx["p2_odds"] = p2_odds
                fx["overround"] = overround
                fx["model_data"] = {
                    "p1_3da": p1_3da,
                    "p2_3da": p2_3da,
                    "p1_hold": round(hb.p1_hold, 4),
                    "p1_break": round(hb.p1_break, 4),
                    "competition_code": comp_code,
                    "legs_to_win": legs_to_win,
                }
                fx["pricing_error"] = None
                priced_count += 1

                log.debug(
                    "auto_priced_darts_fixture",
                    fixture_id=fx["fixture_id"],
                    p1=p1_name,
                    p2=p2_name,
                    p1_odds=p1_odds,
                    p2_odds=p2_odds,
                )

            except Exception as price_err:
                fx["priced"] = False
                fx["pricing_error"] = f"{type(price_err).__name__}: {str(price_err)[:200]}"
                pricing_errors += 1

        log.info(
            "optic_odds_darts_auto_pricing_complete",
            priced=priced_count,
            errors=pricing_errors,
            total=len(fixtures),
        )

    return _ok(
        {
            "source": "Optic Odds",
            "fixtures": fixtures,
            "total": len(fixtures),
            "auto_pricing": {
                "enabled": auto_price,
                "priced": priced_count,
                "errors": pricing_errors,
                "max_price": max_price,
            },
        },
        request_id=request_id,
    )
