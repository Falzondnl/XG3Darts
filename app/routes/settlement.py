"""
Settlement / Grading API routes for the XG3 Darts microservice.

Handles post-match market grading for all 15 darts market types.
All grading is performed against real match result data — no mocked outcomes.

Endpoints
---------
POST /api/v1/darts/settlement/grade/{match_id}
    Grade all submitted markets for a completed match.
    Input: MatchResultPayload + list of MarketsToGrade
    Output: SettlementReportResponse

POST /api/v1/darts/settlement/match-completed
    Webhook receiver for match-completion events from AMQP consumer / Enterprise.
    Triggers full market grading and optionally forwards results to the
    Enterprise platform via the configured callback URL.
    Input: MatchCompletedEvent
    Output: SettlementReportResponse

GET /api/v1/darts/settlement/status/{match_id}
    Check whether a match has been settled and retrieve the settlement summary.
    Output: SettlementStatusResponse

Design Notes
------------
- Settlement reports are stored in Redis (key: xg3:settlement:<match_id>)
  with a 30-day TTL.  The DB write is handled asynchronously.
- The /match-completed endpoint is designed to be called by the AMQP consumer
  when it detects a MatchStatus.COMPLETED event from Optic Odds / DartConnect.
- The grading service is stateless; all state is carried in the request payload.
- Missing or ambiguous data always results in VOID, never in a forced grade.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from fastapi import APIRouter, Body, HTTPException, Path
from pydantic import BaseModel, Field, field_validator

from settlement.darts_settlement_service import (
    DartsSettlementService,
    GradeOutcome,
    LegRecord,
    MatchResult,
)

logger = structlog.get_logger(__name__)
router = APIRouter()

# Redis key prefix and TTL for settlement reports
_SETTLEMENT_KEY_PREFIX = "xg3:settlement:"
_SETTLEMENT_TTL_SECONDS = 30 * 24 * 3600  # 30 days


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class LegRecordPayload(BaseModel):
    """Per-leg statistics included with a match result payload."""

    leg_number: int = Field(..., ge=1, description="1-indexed leg number")
    set_number: Optional[int] = Field(None, description="Set number (sets-format only)")
    winner_is_p1: Optional[bool] = Field(None, description="True if P1 won this leg")
    starter_is_p1: Optional[bool] = Field(None, description="True if P1 threw first")
    had_180: bool = Field(default=False)
    nine_darter: bool = Field(default=False)
    winning_checkout: Optional[int] = Field(
        None, ge=2, le=170,
        description="Winning checkout score (2–170)",
    )
    p1_180s: int = Field(default=0, ge=0)
    p2_180s: int = Field(default=0, ge=0)


class MatchResultPayload(BaseModel):
    """
    Complete match result submitted for settlement grading.

    Mirrors the MatchResult dataclass but expressed as a Pydantic model
    for HTTP request validation.
    """

    status: str = Field(
        ...,
        description=(
            "Terminal match status: Completed | Walkover | Retired | "
            "Abandoned | Cancelled | Postponed"
        ),
    )
    winner_is_p1: Optional[bool] = Field(
        None,
        description="True=P1 won, False=P2 won, None=draw or no result",
    )
    p1_legs: int = Field(default=0, ge=0, description="Legs won by P1")
    p2_legs: int = Field(default=0, ge=0, description="Legs won by P2")
    p1_sets: Optional[int] = Field(None, ge=0, description="Sets won by P1 (sets-format)")
    p2_sets: Optional[int] = Field(None, ge=0, description="Sets won by P2 (sets-format)")
    is_sets_format: bool = Field(
        default=False,
        description="True for sets-format competitions (PDC World Championship etc.)",
    )
    is_draw: bool = Field(
        default=False,
        description="True when match ended in a draw (Premier League group stage)",
    )
    total_legs_played: int = Field(
        default=0, ge=0,
        description="Total legs played.  Computed from p1_legs+p2_legs if 0.",
    )
    total_sets_played: Optional[int] = Field(
        None, ge=0,
        description="Total sets played (sets-format only).",
    )
    p1_180s_total: int = Field(default=0, ge=0)
    p2_180s_total: int = Field(default=0, ge=0)
    p1_highest_checkout: Optional[int] = Field(
        None, ge=2, le=170,
        description="P1's highest single-leg finishing checkout (2–170)",
    )
    p2_highest_checkout: Optional[int] = Field(
        None, ge=2, le=170,
        description="P2's highest single-leg finishing checkout (2–170)",
    )
    nine_darter_in_match: bool = Field(
        default=False,
        description="True if any nine-darter was completed in the match",
    )
    legs: list[LegRecordPayload] = Field(
        default_factory=list,
        description="Per-leg records (optional but improves grading accuracy)",
    )
    p1_player_id: str = Field(default="", description="P1 canonical player ID")
    p2_player_id: str = Field(default="", description="P2 canonical player ID")
    format_code: str = Field(default="", description="Competition format code")
    source_name: str = Field(default="", description="Data source name")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {
            "Completed", "Walkover", "Retired",
            "Abandoned", "Cancelled", "Postponed",
        }
        if v not in valid:
            raise ValueError(
                f"status must be one of {sorted(valid)}, got {v!r}"
            )
        return v


class MarketToGrade(BaseModel):
    """A single market submission for grading."""

    market_type: str = Field(
        ...,
        description=(
            "One of: match_win | exact_score | total_legs_over | handicap | "
            "most_180s | 180_over | highest_checkout | first_leg_winner | "
            "race_to_x | nine_dart_finish | player_checkout_over | sets_over | "
            "break_of_throw | leg_winner_next | total_180s_band"
        ),
    )
    selection: str = Field(
        ...,
        description=(
            "Customer selection.  Values depend on market_type: "
            "'p1'|'p2'|'draw'|'over'|'under'|'yes'|'no'|'tie'|'6-4' etc."
        ),
    )
    market_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Market-specific parameters (line, target, player, leg_number, etc.)",
    )
    market_id: Optional[str] = Field(
        None,
        description="External market identifier (passed through to response)",
    )


class GradeResultResponse(BaseModel):
    """Grading result for a single market."""

    outcome: str = Field(description="WIN | LOSE | VOID | PUSH")
    reason: str


class MarketGradeResponse(BaseModel):
    """Full graded market entry."""

    market_type: str
    selection: str
    market_params: dict[str, Any]
    outcome: str
    reason: str
    market_id: Optional[str] = None


class SettlementReportResponse(BaseModel):
    """Complete settlement report for a match."""

    match_id: str
    status: str
    settled_at: str
    n_markets: int
    n_win: int
    n_lose: int
    n_void: int
    n_push: int
    grades: list[MarketGradeResponse]


class GradeMatchRequest(BaseModel):
    """Request body for POST /settlement/grade/{match_id}."""

    result: MatchResultPayload = Field(
        ..., description="Authoritative match result"
    )
    markets: list[MarketToGrade] = Field(
        ...,
        min_length=1,
        description="Markets to grade (at least 1 required)",
    )


class MatchCompletedEvent(BaseModel):
    """
    Match-completed webhook payload from AMQP consumer or Enterprise platform.

    This is the canonical event shape sent when Optic Odds / DartConnect
    report a terminal match state.
    """

    match_id: str = Field(..., description="Match identifier")
    result: MatchResultPayload = Field(
        ..., description="Complete match result data"
    )
    markets: list[MarketToGrade] = Field(
        default_factory=list,
        description=(
            "Markets to grade.  May be empty if this event is used only "
            "to record the result (no open market positions)."
        ),
    )
    source: str = Field(
        default="amqp",
        description="Event source identifier: amqp | enterprise | manual",
    )
    forward_to_enterprise: bool = Field(
        default=False,
        description="If True, the settlement report is POSTed to ENTERPRISE_CALLBACK_URL",
    )
    enterprise_callback_url: Optional[str] = Field(
        None,
        description="Enterprise webhook URL to receive the settlement report",
    )

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        valid = {"amqp", "enterprise", "manual", "optic_odds", "dartconnect"}
        if v not in valid:
            raise ValueError(
                f"source must be one of {sorted(valid)}, got {v!r}"
            )
        return v


class SettlementStatusResponse(BaseModel):
    """Response for GET /settlement/status/{match_id}."""

    match_id: str
    is_settled: bool
    settled_at: Optional[str] = None
    status: Optional[str] = None
    n_markets: Optional[int] = None
    n_win: Optional[int] = None
    n_lose: Optional[int] = None
    n_void: Optional[int] = None
    n_push: Optional[int] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _payload_to_match_result(match_id: str, payload: MatchResultPayload) -> MatchResult:
    """Convert a Pydantic payload into the domain MatchResult dataclass."""
    legs = [
        LegRecord(
            leg_number=lr.leg_number,
            set_number=lr.set_number,
            winner_is_p1=lr.winner_is_p1,
            starter_is_p1=lr.starter_is_p1,
            had_180=lr.had_180,
            nine_darter=lr.nine_darter,
            winning_checkout=lr.winning_checkout,
            p1_180s=lr.p1_180s,
            p2_180s=lr.p2_180s,
        )
        for lr in payload.legs
    ]

    return MatchResult(
        match_id=match_id,
        status=payload.status,
        winner_is_p1=payload.winner_is_p1,
        p1_legs=payload.p1_legs,
        p2_legs=payload.p2_legs,
        p1_sets=payload.p1_sets,
        p2_sets=payload.p2_sets,
        is_sets_format=payload.is_sets_format,
        is_draw=payload.is_draw,
        total_legs_played=payload.total_legs_played,
        total_sets_played=payload.total_sets_played,
        p1_180s_total=payload.p1_180s_total,
        p2_180s_total=payload.p2_180s_total,
        p1_highest_checkout=payload.p1_highest_checkout,
        p2_highest_checkout=payload.p2_highest_checkout,
        nine_darter_in_match=payload.nine_darter_in_match,
        legs=legs,
        p1_player_id=payload.p1_player_id,
        p2_player_id=payload.p2_player_id,
        format_code=payload.format_code,
        completed_at=datetime.now(timezone.utc),
        source_name=payload.source_name,
    )


def _report_to_response(report: Any) -> SettlementReportResponse:
    """Convert a SettlementReport domain object to the HTTP response model."""
    grades_out: list[MarketGradeResponse] = []
    for g in report.grades:
        grades_out.append(
            MarketGradeResponse(
                market_type=g.market_type,
                selection=g.selection,
                market_params=g.market_params,
                outcome=g.grade.outcome.value,
                reason=g.grade.reason,
                market_id=g.market_id,
            )
        )

    return SettlementReportResponse(
        match_id=report.match_id,
        status=report.status,
        settled_at=report.settled_at.isoformat(),
        n_markets=len(report.grades),
        n_win=report.n_win,
        n_lose=report.n_lose,
        n_void=report.n_void,
        n_push=report.n_push,
        grades=grades_out,
    )


async def _persist_settlement_report(
    match_id: str,
    response: SettlementReportResponse,
) -> None:
    """
    Persist a settlement report to Redis with a 30-day TTL.

    Non-fatal: if Redis is unavailable the settlement still succeeds;
    we log a warning and continue.
    """
    from app.config import settings

    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=2,
            decode_responses=True,
        )
        key = f"{_SETTLEMENT_KEY_PREFIX}{match_id}"
        serialised = response.model_dump_json()
        await client.set(key, serialised, ex=_SETTLEMENT_TTL_SECONDS)
        await client.aclose()
        logger.info(
            "settlement_persisted_to_redis",
            match_id=match_id,
            key=key,
            ttl_days=30,
        )
    except Exception as exc:
        logger.warning(
            "settlement_redis_persist_failed",
            match_id=match_id,
            error=str(exc),
        )


async def _forward_to_enterprise(
    callback_url: str,
    match_id: str,
    report: SettlementReportResponse,
) -> None:
    """
    Forward settlement report to the Enterprise platform via HTTP POST.

    Non-fatal: if the forward fails we log the error and continue.
    The Enterprise platform should implement its own retry mechanism.
    """
    try:
        import aiohttp  # type: ignore[import]
        payload = report.model_dump()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                callback_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status not in (200, 201, 202):
                    body = await resp.text()
                    logger.warning(
                        "enterprise_forward_non_2xx",
                        match_id=match_id,
                        callback_url=callback_url,
                        status=resp.status,
                        body=body[:200],
                    )
                else:
                    logger.info(
                        "enterprise_forward_success",
                        match_id=match_id,
                        callback_url=callback_url,
                        status=resp.status,
                    )
    except ImportError:
        logger.warning(
            "enterprise_forward_skipped_no_aiohttp",
            match_id=match_id,
            callback_url=callback_url,
        )
    except Exception as exc:
        logger.error(
            "enterprise_forward_failed",
            match_id=match_id,
            callback_url=callback_url,
            error=str(exc),
        )


async def _get_settlement_from_redis(match_id: str) -> Optional[dict[str, Any]]:
    """
    Retrieve a persisted settlement report from Redis.

    Returns None if not found or Redis is unavailable.
    """
    from app.config import settings

    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=2,
            decode_responses=True,
        )
        key = f"{_SETTLEMENT_KEY_PREFIX}{match_id}"
        raw = await client.get(key)
        await client.aclose()
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.warning(
            "settlement_redis_read_failed",
            match_id=match_id,
            error=str(exc),
        )
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/settlement/grade/{match_id}",
    response_model=SettlementReportResponse,
    status_code=200,
    summary="Grade all markets for a completed match",
    tags=["Settlement"],
)
async def grade_match(
    match_id: str = Path(
        ...,
        description="Unique match identifier",
    ),
    body: GradeMatchRequest = Body(...),
) -> SettlementReportResponse:
    """
    Grade all submitted markets for a completed match.

    Accepts the authoritative match result (scores, 180s, checkouts, etc.)
    and a list of open markets with their customer selections.  Returns
    WIN / LOSE / VOID / PUSH for every market.

    All grading logic runs synchronously and deterministically based on
    the supplied result data.  No database queries are performed during
    grading — the caller is responsible for supplying complete result data.

    The settlement report is cached in Redis (30-day TTL) so that the
    ``GET /settlement/status/{match_id}`` endpoint can return the result
    without re-grading.

    Raises
    ------
    422 Unprocessable Entity
        When the request payload fails validation (invalid status,
        malformed selection, etc.).
    """
    request_id = str(uuid.uuid4())
    logger.info(
        "settlement_grade_requested",
        match_id=match_id,
        n_markets=len(body.markets),
        status=body.result.status,
        request_id=request_id,
    )

    # Convert payload to domain objects
    match_result = _payload_to_match_result(match_id, body.result)
    markets_to_grade = [
        {
            "market_type": m.market_type,
            "selection": m.selection,
            "market_params": m.market_params,
            "market_id": m.market_id,
        }
        for m in body.markets
    ]

    # Run settlement
    service = DartsSettlementService()
    report = service.grade_match(match_result, markets_to_grade)

    response = _report_to_response(report)

    # Persist to Redis (non-fatal)
    await _persist_settlement_report(match_id, response)

    return response


@router.post(
    "/settlement/match-completed",
    response_model=SettlementReportResponse,
    status_code=200,
    summary="Webhook receiver for match-completion events (AMQP / Enterprise)",
    tags=["Settlement"],
)
async def match_completed_webhook(
    event: MatchCompletedEvent = Body(...),
) -> SettlementReportResponse:
    """
    Handle a match-completed event from the AMQP consumer or Enterprise platform.

    This endpoint is the primary settlement trigger in production.  It is
    called by the AMQP consumer (Optic Odds RabbitMQ) or the Enterprise
    platform webhook when a darts match reaches a terminal state.

    Workflow
    --------
    1. Validate the event payload.
    2. Convert to domain MatchResult + markets list.
    3. Run DartsSettlementService.grade_match().
    4. Persist the settlement report to Redis (TTL = 30 days).
    5. If ``forward_to_enterprise=True``, POST the report to
       ``enterprise_callback_url`` (non-blocking, best-effort).
    6. Return the settlement report.

    When ``markets`` is empty (result-only event), the response will
    contain an empty grades list with all counts at 0.

    Raises
    ------
    422 Unprocessable Entity
        When the event payload fails Pydantic validation.
    """
    match_id = event.match_id
    request_id = str(uuid.uuid4())

    logger.info(
        "match_completed_event_received",
        match_id=match_id,
        status=event.result.status,
        source=event.source,
        n_markets=len(event.markets),
        request_id=request_id,
    )

    # Convert to domain objects
    match_result = _payload_to_match_result(match_id, event.result)
    markets_to_grade = [
        {
            "market_type": m.market_type,
            "selection": m.selection,
            "market_params": m.market_params,
            "market_id": m.market_id,
        }
        for m in event.markets
    ]

    # Run settlement
    service = DartsSettlementService()
    report = service.grade_match(match_result, markets_to_grade)
    response = _report_to_response(report)

    # Persist to Redis
    await _persist_settlement_report(match_id, response)

    # Forward to Enterprise if requested
    if event.forward_to_enterprise and event.enterprise_callback_url:
        await _forward_to_enterprise(
            callback_url=event.enterprise_callback_url,
            match_id=match_id,
            report=response,
        )

    logger.info(
        "settlement_complete",
        match_id=match_id,
        status=event.result.status,
        n_win=response.n_win,
        n_lose=response.n_lose,
        n_void=response.n_void,
        n_push=response.n_push,
        request_id=request_id,
    )

    return response


@router.get(
    "/settlement/status/{match_id}",
    response_model=SettlementStatusResponse,
    summary="Check whether a match has been settled",
    tags=["Settlement"],
)
async def get_settlement_status(
    match_id: str = Path(
        ...,
        description="Unique match identifier",
    ),
) -> SettlementStatusResponse:
    """
    Check the settlement status of a match.

    Looks up the settlement report in Redis.  Returns ``is_settled=True``
    with summary counts if the match has been settled, or
    ``is_settled=False`` if no settlement report exists.

    Note: This endpoint returns cached data only.  It does not re-run
    grading.  If no Redis connection is available, ``is_settled`` will
    always be False.

    Raises
    ------
    404 Not Found
        Not raised — a not-found result is returned as is_settled=False.
    """
    cached = await _get_settlement_from_redis(match_id)

    if cached is None:
        return SettlementStatusResponse(
            match_id=match_id,
            is_settled=False,
        )

    return SettlementStatusResponse(
        match_id=match_id,
        is_settled=True,
        settled_at=cached.get("settled_at"),
        status=cached.get("status"),
        n_markets=cached.get("n_markets"),
        n_win=cached.get("n_win"),
        n_lose=cached.get("n_lose"),
        n_void=cached.get("n_void"),
        n_push=cached.get("n_push"),
    )
