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
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

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
    response_model=FeedsStatusResponse,
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
