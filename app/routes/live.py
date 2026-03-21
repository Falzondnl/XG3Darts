"""
Live in-play pricing API routes.

Endpoints
---------
POST /api/v1/darts/live/seed
    Initialise a new live match state (required before any live pricing).

POST /api/v1/darts/live/update
    Accept a visit-scored event and reprice all live markets.

GET  /api/v1/darts/live/state/{match_id}
    Return the current live state for a match.

POST /api/v1/darts/live/price/{match_id}
    Trigger an on-demand reprice for a match in the current live state.

GET  /api/v1/darts/live/markets/{match_id}
    Return all live market prices for a match.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import structlog
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from engines.errors import DartsDataError, DartsEngineError
from shared.stale_price_guard import StalePriceGuard
from shared.pricing_layer import get_pricing_layer

logger = structlog.get_logger(__name__)
router = APIRouter()

# Stale-price guard — module-level singleton
_stale_guard = StalePriceGuard(sport="darts")
_live_blend = get_pricing_layer("darts")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class VisitScoredEvent(BaseModel):
    """
    Payload for a visit-scored event.

    Sent by the live ingestion layer when a player completes a 3-dart visit.
    """

    match_id: str = Field(..., description="Unique match identifier")
    visit_score: int = Field(
        ..., ge=0, le=180, description="Total score from the visit (0 on bust)"
    )
    is_bust: bool = Field(default=False, description="Whether the visit ended in a bust")
    current_thrower: int = Field(
        ..., ge=0, le=1, description="0=P1, 1=P2"
    )
    dartconnect_feed_lag_ms: int = Field(
        default=0, ge=0, description="Reported DartConnect feed lag in milliseconds"
    )

    @field_validator("visit_score")
    @classmethod
    def bust_must_be_zero(cls, v: int, info: Any) -> int:
        # If is_bust is True, visit_score should be 0
        # We don't hard-fail here because the field order in Pydantic v2
        # makes cross-field validation tricky; the engine handles this.
        return v


class LiveStateResponse(BaseModel):
    """Serialised live state returned to clients."""

    match_id: str
    score_p1: int
    score_p2: int
    current_thrower: int
    legs_p1: int
    legs_p2: int
    sets_p1: int
    sets_p2: int
    lwp_current: float
    regime: int
    double_start: bool
    draw_enabled: bool
    two_clear_legs: bool
    format_code: str
    leg_starter: Optional[str]
    leg_starter_confidence: float
    is_pressure_state: bool
    current_dart_number: int
    dartconnect_feed_lag_ms: int
    last_updated: str
    is_stale: bool
    p1_three_da: float
    p2_three_da: float


class LivePriceResponse(BaseModel):
    """Market prices returned after a reprice."""

    match_id: str
    p1_match_win: float
    p2_match_win: float
    draw_prob: float
    p1_leg_win: float
    p2_leg_win: float
    processing_latency_ms: float
    pricing_authority: str = "live_model_only"


class LiveMarketsResponse(BaseModel):
    """All live market prices for a match."""

    match_id: str
    match_winner: dict[str, float]
    current_leg: dict[str, float]
    draw_available: bool
    state_summary: dict[str, Any]
    pricing_authority: str = "live_model_only"


class VisitUpdateResponse(BaseModel):
    """Response returned after processing a visit-scored event."""

    match_id: str
    state: LiveStateResponse
    prices: LivePriceResponse
    processing_latency_ms: float


# ---------------------------------------------------------------------------
# Dependency: live engine
# ---------------------------------------------------------------------------

def _get_live_engine() -> Any:
    """
    Return the DartsLiveEngine instance.

    In production this should be a singleton attached to the application
    lifespan. For now we create one per request (engine itself is stateless
    for pricing; state is held in Redis).
    """
    from engines.live.live_state_machine import DartsLiveEngine

    # Redis client is optional; engine degrades gracefully without it
    redis_client = None
    try:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=1,
            decode_responses=True,
        )
    except Exception:
        pass

    return DartsLiveEngine(redis_client=redis_client)


def _state_to_response(state: Any, engine: Any = None) -> LiveStateResponse:
    """Convert a DartsLiveState dataclass to the API response model."""
    return LiveStateResponse(
        match_id=state.match_id,
        score_p1=state.score_p1,
        score_p2=state.score_p2,
        current_thrower=state.current_thrower,
        legs_p1=state.legs_p1,
        legs_p2=state.legs_p2,
        sets_p1=state.sets_p1,
        sets_p2=state.sets_p2,
        lwp_current=state.lwp_current,
        regime=state.regime,
        double_start=state.double_start,
        draw_enabled=state.draw_enabled,
        two_clear_legs=state.two_clear_legs,
        format_code=state.format_code,
        leg_starter=state.leg_starter,
        leg_starter_confidence=state.leg_starter_confidence,
        is_pressure_state=state.is_pressure_state,
        current_dart_number=state.current_dart_number,
        dartconnect_feed_lag_ms=state.dartconnect_feed_lag_ms,
        last_updated=state.last_updated.isoformat(),
        is_stale=state.is_stale(),
        p1_three_da=state.p1_three_da,
        p2_three_da=state.p2_three_da,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

class MatchSeedRequest(BaseModel):
    """
    Payload to initialise a new live match state.

    Send this once when a match goes live (before the first visit is scored).
    All subsequent ``/live/update`` calls will apply visit-scored events to
    this state.
    """

    match_id: str = Field(..., description="Unique match identifier")
    format_code: str = Field(
        default="PDC_PL",
        description="Competition format code from the registry (e.g. PDC_PL, PDC_WC)",
    )
    p1_player_id: str = Field(default="", description="Player 1 identifier")
    p2_player_id: str = Field(default="", description="Player 2 identifier")
    p1_three_da: float = Field(
        default=70.0, ge=20.0, le=120.0, description="Player 1 three-dart average"
    )
    p2_three_da: float = Field(
        default=70.0, ge=20.0, le=120.0, description="Player 2 three-dart average"
    )
    regime: int = Field(
        default=1, ge=0, le=2, description="Data regime: 0=R0 (logit), 1=R1 (LightGBM), 2=R2 (stacking)"
    )
    current_thrower: int = Field(
        default=0, ge=0, le=1, description="Who throws first: 0=P1, 1=P2"
    )
    leg_starter_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in starter attribution"
    )


class MatchSeedResponse(BaseModel):
    """Response after successfully seeding a live match state."""

    match_id: str
    format_code: str
    state: LiveStateResponse
    message: str


@router.post(
    "/live/seed",
    response_model=MatchSeedResponse,
    summary="Initialise a live match state",
    status_code=201,
)
async def seed_live_match(
    payload: MatchSeedRequest = Body(...),
) -> MatchSeedResponse:
    """
    Initialise a new live match state in Redis.

    Must be called once before any ``/live/update`` or ``/live/price`` calls.
    Typically invoked by the live ingestion worker when a match transitions to
    in-play status.

    Calling seed on an already-active match_id overwrites the existing state
    (useful for match restarts or corrections).

    Raises
    ------
    422 Unprocessable Entity
        When the format_code is invalid or the payload fails validation.
    500 Internal Server Error
        When Redis is unavailable and the state cannot be persisted.
    """
    from datetime import datetime, timezone

    from engines.live.live_state_machine import DartsLiveState
    from competition.format_registry import get_format

    # Resolve format to get draw_enabled / two_clear_legs flags
    try:
        fmt = get_format(payload.format_code)
        draw_enabled = getattr(fmt, "draw_enabled", False)
        two_clear_legs = getattr(fmt, "two_clear_legs", False)
        double_start = getattr(fmt, "double_start_required", False)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_format_code",
                "format_code": payload.format_code,
                "message": str(exc),
            },
        )

    state = DartsLiveState(
        match_id=payload.match_id,
        score_p1=501,
        score_p2=501,
        current_thrower=payload.current_thrower,
        legs_p1=0,
        legs_p2=0,
        sets_p1=0,
        sets_p2=0,
        lwp_current=0.5,
        regime=payload.regime,
        double_start=double_start,
        draw_enabled=draw_enabled,
        two_clear_legs=two_clear_legs,
        format_code=payload.format_code,
        leg_starter=payload.p1_player_id if payload.current_thrower == 0 else payload.p2_player_id,
        leg_starter_confidence=payload.leg_starter_confidence,
        is_pressure_state=False,
        current_dart_number=1,
        dartconnect_feed_lag_ms=0,
        last_updated=datetime.now(timezone.utc),
        p1_player_id=payload.p1_player_id,
        p2_player_id=payload.p2_player_id,
        p1_three_da=payload.p1_three_da,
        p2_three_da=payload.p2_three_da,
        current_leg_number=1,
    )

    engine = _get_live_engine()
    await engine.save_state(payload.match_id, state)

    logger.info(
        "live_match_seeded",
        match_id=payload.match_id,
        format_code=payload.format_code,
        regime=payload.regime,
    )

    return MatchSeedResponse(
        match_id=payload.match_id,
        format_code=payload.format_code,
        state=_state_to_response(state),
        message=f"Match {payload.match_id!r} seeded successfully.",
    )


@router.post(
    "/live/update",
    response_model=VisitUpdateResponse,
    summary="Process a visit-scored event and reprice all live markets",
    status_code=200,
)
async def update_live_state(
    event: VisitScoredEvent = Body(...),
) -> VisitUpdateResponse:
    """
    Accept a visit-scored event from the live ingestion layer.

    Applies the visit to the current live state, runs Kalman updates,
    reprices all markets via the Markov chain, and persists the new state
    to Redis.

    The match state must have been initialised previously (e.g. via the
    match-start ingestion workflow). If no state exists, a 404 is returned.

    Raises
    ------
    404 Not Found
        When no live state exists for the given match_id.
    422 Unprocessable Entity
        When the event payload is invalid.
    500 Internal Server Error
        When the pricing engine encounters an unrecoverable error.
    """
    t_start = time.perf_counter()
    engine = _get_live_engine()

    # Load current state from Redis
    state = await engine.get_state(event.match_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "match_not_found",
                "match_id": event.match_id,
                "message": (
                    f"No live state found for match {event.match_id!r}. "
                    "The match may not be active or has not been initialised."
                ),
            },
        )

    # Override thrower from event if provided
    import dataclasses
    state = dataclasses.replace(
        state,
        current_thrower=event.current_thrower,
        dartconnect_feed_lag_ms=event.dartconnect_feed_lag_ms,
    )

    try:
        new_state = await engine.on_visit_scored(
            match_id=event.match_id,
            state=state,
            visit_score=event.visit_score,
            is_bust=event.is_bust,
        )
    except DartsDataError as exc:
        raise HTTPException(status_code=422, detail={"error": str(exc)})
    except DartsEngineError as exc:
        logger.error(
            "live_update_engine_error",
            match_id=event.match_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail={"error": "pricing_engine_error", "message": str(exc)},
        )

    price_update = await engine.reprice_match(event.match_id, new_state)
    total_ms = (time.perf_counter() - t_start) * 1000.0

    # Record that live data arrived for freshness tracking
    _stale_guard.record_update(event.match_id)

    # Pinnacle logit-space blend (falls back to model-only when Pinnacle unavailable)
    _blend = await _live_blend.blend_pre_match(event.match_id, price_update.p1_match_win)
    _p1 = _blend.blended_prob
    _p2 = max(0.0, 1.0 - _p1 - price_update.draw_prob)
    _authority = "live_blended" if _blend.source == "pinnacle" else "live_model_only"

    return VisitUpdateResponse(
        match_id=event.match_id,
        state=_state_to_response(new_state),
        prices=LivePriceResponse(
            match_id=event.match_id,
            p1_match_win=_p1,
            p2_match_win=_p2,
            draw_prob=price_update.draw_prob,
            p1_leg_win=price_update.p1_leg_win,
            p2_leg_win=price_update.p2_leg_win,
            processing_latency_ms=price_update.processing_latency_ms,
            pricing_authority=_authority,
        ),
        processing_latency_ms=total_ms,
    )


@router.get(
    "/live/state/{match_id}",
    response_model=LiveStateResponse,
    summary="Get current live state for a match",
)
async def get_live_state(
    match_id: str = Path(..., description="Match identifier"),
) -> LiveStateResponse:
    """
    Return the current live state for an in-progress match.

    The state is loaded from Redis. Includes staleness flag.

    Raises
    ------
    404 Not Found
        When no live state exists for the given match_id.
    """
    engine = _get_live_engine()
    state = await engine.get_state(match_id)

    if state is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "match_not_found",
                "match_id": match_id,
            },
        )

    # Gate: suspend if live data has gone stale
    _freshness = _stale_guard.check(match_id)
    if _freshness.should_suspend:
        return JSONResponse(
            status_code=503,
            content={
                "error": "live_suspended",
                "reason": _freshness.reason,
                "stale_seconds": round(_freshness.stale_seconds, 1),
                "level": _freshness.level.value,
            },
        )

    return _state_to_response(state)


@router.post(
    "/live/price/{match_id}",
    response_model=LivePriceResponse,
    summary="Reprice a match from current live state",
)
async def reprice_match(
    match_id: str = Path(..., description="Match identifier"),
) -> LivePriceResponse:
    """
    Trigger an on-demand reprice for an in-progress match.

    Loads the current live state from Redis and runs a full Markov chain
    reprice without processing a new visit.

    Raises
    ------
    404 Not Found
        When no live state exists for the given match_id.
    """
    engine = _get_live_engine()
    state = await engine.get_state(match_id)

    if state is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "match_not_found", "match_id": match_id},
        )

    try:
        price_update = await engine.reprice_match(match_id, state)
    except (DartsDataError, DartsEngineError) as exc:
        logger.error("live_reprice_error", match_id=match_id, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail={"error": "pricing_error", "message": str(exc)},
        )

    _blend = await _live_blend.blend_pre_match(match_id, price_update.p1_match_win)
    _p1 = _blend.blended_prob
    _p2 = max(0.0, 1.0 - _p1 - price_update.draw_prob)
    _authority = "live_blended" if _blend.source == "pinnacle" else "live_model_only"

    return LivePriceResponse(
        match_id=match_id,
        p1_match_win=_p1,
        p2_match_win=_p2,
        draw_prob=price_update.draw_prob,
        p1_leg_win=price_update.p1_leg_win,
        p2_leg_win=price_update.p2_leg_win,
        processing_latency_ms=price_update.processing_latency_ms,
        pricing_authority=_authority,
    )


@router.get(
    "/live/markets/{match_id}",
    response_model=LiveMarketsResponse,
    summary="Get all live market prices for a match",
)
async def get_live_markets(
    match_id: str = Path(..., description="Match identifier"),
) -> LiveMarketsResponse:
    """
    Return all live market prices for an in-progress match.

    Combines match winner, current leg, and draw markets into a single response.

    Raises
    ------
    404 Not Found
        When no live state exists for the given match_id.
    """
    engine = _get_live_engine()
    state = await engine.get_state(match_id)

    if state is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "match_not_found", "match_id": match_id},
        )

    # Gate: suspend if live data has gone stale
    _freshness = _stale_guard.check(match_id)
    if _freshness.should_suspend:
        return JSONResponse(
            status_code=503,
            content={
                "error": "live_suspended",
                "reason": _freshness.reason,
                "stale_seconds": round(_freshness.stale_seconds, 1),
                "level": _freshness.level.value,
            },
        )

    try:
        price_update = await engine.reprice_match(match_id, state)
    except (DartsDataError, DartsEngineError) as exc:
        logger.error("live_markets_error", match_id=match_id, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail={"error": "pricing_error", "message": str(exc)},
        )

    # Pinnacle logit-space blend for match winner market
    _blend = await _live_blend.blend_pre_match(match_id, price_update.p1_match_win)
    _p1_mw = _blend.blended_prob
    _p2_mw = max(0.0, 1.0 - _p1_mw - price_update.draw_prob)
    _authority = "live_blended" if _blend.source == "pinnacle" else "live_model_only"

    match_winner_market: dict[str, float] = {
        "p1_win": round(_p1_mw, 6),
        "p2_win": round(_p2_mw, 6),
    }
    if state.draw_enabled:
        match_winner_market["draw"] = round(price_update.draw_prob, 6)

    current_leg_market: dict[str, float] = {
        "p1_leg_win": round(price_update.p1_leg_win, 6),
        "p2_leg_win": round(price_update.p2_leg_win, 6),
    }

    state_summary: dict[str, Any] = {
        "score_p1": state.score_p1,
        "score_p2": state.score_p2,
        "legs_p1": state.legs_p1,
        "legs_p2": state.legs_p2,
        "sets_p1": state.sets_p1,
        "sets_p2": state.sets_p2,
        "current_leg": state.current_leg_number,
        "current_thrower": state.current_thrower,
        "is_pressure_state": state.is_pressure_state,
        "feed_lag_ms": state.dartconnect_feed_lag_ms,
        "is_stale": state.is_stale(),
    }

    return LiveMarketsResponse(
        match_id=match_id,
        match_winner=match_winner_market,
        current_leg=current_leg_market,
        draw_available=state.draw_enabled,
        state_summary=state_summary,
        pricing_authority=_authority,
    )


# ---------------------------------------------------------------------------
# In-play proposition markets
# ---------------------------------------------------------------------------

class InPlayPropMarketResponse(BaseModel):
    """A single in-play proposition market."""

    market_key: str
    market_label: str
    outcomes: dict[str, float]
    is_open: bool
    data_gate_reason: Optional[str]
    metadata: dict[str, Any]


class InPlayPropsResponse(BaseModel):
    """All granular in-play proposition markets for a match state."""

    match_id: str
    current_leg: int
    sets_p1: int
    sets_p2: int
    score_p1: int
    score_p2: int
    open_market_count: int
    markets: list[InPlayPropMarketResponse]


@router.get(
    "/live/inplay-props/{match_id}",
    response_model=InPlayPropsResponse,
    summary="Get all granular in-play proposition markets for a match",
)
async def get_inplay_props(
    match_id: str = Path(..., description="Match identifier"),
) -> InPlayPropsResponse:
    """
    Return all per-leg/per-set proposition markets for an in-progress match.

    Markets include:
    - ``next_leg_winner``      — who wins the current leg
    - ``next_set_winner``      — who wins the current set (set-format only)
    - ``leg_180_count``        — over/under 180s remaining in this leg
    - ``first_180_in_leg``     — who hits the first 180
    - ``leg_checkout``         — who finishes first, and checkout score bands
    - ``nine_darter``          — probability of a nine-darter in this leg
    - ``live_leg_handicap``    — updated handicap from current leg state
    - ``set_correct_score``    — correct score for current set (set-format only)

    All prices include the standard in-play overround margin (4%).

    Raises
    ------
    404 Not Found
        When no live state exists for the given match_id.
    """
    from engines.live.inplay_props import InPlayPropsEngine

    engine = _get_live_engine()
    state = await engine.get_state(match_id)

    if state is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "match_not_found", "match_id": match_id},
        )

    try:
        props_engine = InPlayPropsEngine()
        result = props_engine.compute(state)
    except DartsDataError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "data_gate_failed", "message": str(exc)},
        )
    except DartsEngineError as exc:
        logger.error("inplay_props_engine_error", match_id=match_id, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail={"error": "engine_error", "message": str(exc)},
        )

    market_responses = [
        InPlayPropMarketResponse(
            market_key=m.market_key,
            market_label=m.market_label,
            outcomes=m.outcomes,
            is_open=m.is_open,
            data_gate_reason=m.data_gate_reason,
            metadata=m.metadata,
        )
        for m in result.markets
    ]

    return InPlayPropsResponse(
        match_id=match_id,
        current_leg=result.current_leg,
        sets_p1=result.current_set_p1,
        sets_p2=result.current_set_p2,
        score_p1=result.score_p1,
        score_p2=result.score_p2,
        open_market_count=len(result.open_markets()),
        markets=market_responses,
    )
