"""
Pre-match pricing API routes — Sprint 6 update: full ecosystem routing.

All endpoints compute prices on-demand from the engine pipeline.
No static odds, no hardcoded probabilities.

Ecosystem routing
-----------------
pdc_mens     → R0 / R1 / R2 full model path (regime selected by data availability)
pdc_womens   → WomensTransferModel + 1.20x ecosystem margin widening
wdf_open     → R0 primarily (limited data), no R1/R2 upgrade
development  → R0 only (very limited data: PDC Development / Challenge Tour)
team_doubles → Routed to World Cup pricer (not this module)

Endpoints
---------
POST /api/v1/darts/prematch/price            — match winner market
POST /api/v1/darts/prematch/exact-score      — correct score market
POST /api/v1/darts/prematch/handicap         — leg handicap market
POST /api/v1/darts/prematch/totals           — total legs O/U
POST /api/v1/darts/prematch/multi-totals     — all O/U lines in one call (Pinnacle-style)
POST /api/v1/darts/prematch/180s             — match 180 O/U
POST /api/v1/darts/prematch/checkout         — highest checkout O/U
POST /api/v1/darts/prematch/first-leg        — first leg winner market
POST /api/v1/darts/prematch/both-to-score    — both players score X+ legs
POST /api/v1/darts/prematch/race-to-x        — race-to-X legs market
POST /api/v1/darts/prematch/winning-margin   — winning margin distribution
POST /api/v1/darts/prematch/ecosystem-price  — ecosystem-aware match pricing
GET  /api/v1/darts/prematch/markets/{match_id} — all markets for a fixture
"""
from __future__ import annotations

from typing import Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session_dependency

from competition.format_registry import get_format, DartsFormatError
from engines.errors import DartsDataError, DartsEngineError
from engines.leg_layer.hold_break_model import HoldBreakModel
from engines.match_layer.match_combinatorics import MatchCombinatorialEngine
from margin.blending_engine import DartsMarginEngine
from margin.shin_margin import ShinMarginModel
from props.prop_180 import Prop180Model
from shared.pricing_layer import get_pricing_layer

logger = structlog.get_logger(__name__)
router = APIRouter()

_hold_break_model = HoldBreakModel()
_match_engine = MatchCombinatorialEngine()
_margin_engine = DartsMarginEngine()
_shin_model = ShinMarginModel()
_prop_180_model = Prop180Model()
_blend_layer = get_pricing_layer("darts")


# ---------------------------------------------------------------------------
# Shared request/response models
# ---------------------------------------------------------------------------

class MatchPriceRequest(BaseModel):
    """Core match pricing request."""
    competition_code: str = Field(..., description="Format code, e.g. 'PDC_WC'")
    round_name: str = Field(..., description="Round name, e.g. 'Final'")
    p1_id: str
    p2_id: str
    p1_three_da: float = Field(..., gt=0, lt=200)
    p2_three_da: float = Field(..., gt=0, lt=200)
    p1_starts_first: bool = True
    regime: int = Field(default=1, ge=0, le=2)
    starter_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    model_agreement: float = Field(default=1.0, ge=0.0, le=1.0)
    market_liquidity: str = Field(default="high")
    base_margin: float = Field(default=0.05, gt=0.0, le=0.15)
    double_start_required: Optional[bool] = None
    fixture_id: Optional[str] = Field(default=None, description="Optic Odds fixture ID for Pinnacle auto-fetch")
    pinnacle_home_odds: Optional[float] = Field(default=None, gt=1.0, description="Explicit Pinnacle P1 odds")
    pinnacle_away_odds: Optional[float] = Field(default=None, gt=1.0, description="Explicit Pinnacle P2 odds")

    @field_validator("market_liquidity")
    @classmethod
    def validate_liquidity(cls, v: str) -> str:
        if v not in ("high", "medium", "low"):
            raise ValueError(f"market_liquidity must be high/medium/low, got {v!r}")
        return v


class HandicapRequest(MatchPriceRequest):
    """Handicap market request."""
    handicap_legs: int = Field(..., description="Leg handicap for P1 (e.g. -1, +1)")


class TotalsRequest(MatchPriceRequest):
    """Totals (O/U) market request."""
    total_line: float = Field(..., description="O/U line for total legs, e.g. 8.5")


class Match180Request(MatchPriceRequest):
    """180 prop request."""
    line: float = Field(..., description="O/U line for total 180s")
    p1_180_rate: float = Field(..., ge=0.0, le=1.0)
    p2_180_rate: float = Field(..., ge=0.0, le=1.0)
    expected_visits_per_leg: float = Field(default=8.0, gt=0)


class HighestCheckoutRequest(MatchPriceRequest):
    """Highest checkout O/U request."""
    line: int = Field(..., ge=2, le=170, description="Checkout threshold")
    p1_checkout_ewm: float = Field(..., ge=0.0, le=1.0)
    p2_checkout_ewm: float = Field(..., ge=0.0, le=1.0)
    p1_legs_observed: int = Field(default=50, ge=0)
    p2_legs_observed: int = Field(default=50, ge=0)
    p1_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    p2_confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class FirstLegRequest(BaseModel):
    """First leg winner market request."""
    competition_code: str
    round_name: str = "Final"
    p1_id: str
    p2_id: str
    p1_three_da: float = Field(..., gt=0, lt=200)
    p2_three_da: float = Field(..., gt=0, lt=200)
    p1_starts_first: bool = True
    base_margin: float = Field(default=0.05, gt=0.0, le=0.15)
    double_start_required: Optional[bool] = None


class BothToScoreRequest(BaseModel):
    """Both players to score at least X legs market."""
    competition_code: str
    round_name: str = "Final"
    p1_id: str
    p2_id: str
    p1_three_da: float = Field(..., gt=0, lt=200)
    p2_three_da: float = Field(..., gt=0, lt=200)
    p1_starts_first: bool = True
    min_legs_each: int = Field(default=3, ge=1, le=12,
                               description="Both players must reach at least this many legs")
    base_margin: float = Field(default=0.05, gt=0.0, le=0.15)
    double_start_required: Optional[bool] = None


class MultiLegLineRequest(BaseModel):
    """Return O/U odds for multiple leg totals simultaneously (Pinnacle-style multi-line)."""
    competition_code: str
    round_name: str = "Final"
    p1_id: str
    p2_id: str
    p1_three_da: float = Field(..., gt=0, lt=200)
    p2_three_da: float = Field(..., gt=0, lt=200)
    p1_starts_first: bool = True
    base_margin: float = Field(default=0.04, gt=0.0, le=0.15)
    double_start_required: Optional[bool] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_format(competition_code: str, round_name: str):
    """Resolve format or raise 422."""
    try:
        fmt = get_format(competition_code)
    except DartsFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        fmt.get_round(round_name)  # validate round
    except DartsFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return fmt


def _compute_hold_break(req: MatchPriceRequest):
    """Compute hold/break, raising 422 on error."""
    try:
        fmt = _resolve_format(req.competition_code, req.round_name)
        double_start = (
            req.double_start_required
            if req.double_start_required is not None
            else fmt.double_start_required
        )
        hb = _hold_break_model.compute_from_3da(
            p1_id=req.p1_id,
            p2_id=req.p2_id,
            p1_three_da=req.p1_three_da,
            p2_three_da=req.p2_three_da,
            starting_score=fmt.starting_score,
            double_start=double_start,
        )
        return hb, fmt
    except (DartsEngineError, DartsDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def _apply_margin(
    req: MatchPriceRequest,
    fmt,
    true_probs: dict[str, float],
) -> dict[str, float]:
    """Compute margin and apply Shin model."""
    try:
        margin = _margin_engine.compute_margin(
            base_margin=req.base_margin,
            regime=req.regime,
            starter_confidence=req.starter_confidence,
            source_confidence=req.source_confidence,
            model_agreement=req.model_agreement,
            market_liquidity=req.market_liquidity,
            ecosystem=fmt.ecosystem,
        )
        adjusted = _shin_model.apply_shin_margin(
            true_probs=true_probs,
            target_margin=margin,
        )
        return adjusted, margin
    except DartsEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def _expected_legs(legs_dist: dict) -> float:
    """Compute expected total legs from legs distribution."""
    if not legs_dist:
        return 9.0
    return sum((s1 + s2) * p for (s1, s2), p in legs_dist.items())


# ---------------------------------------------------------------------------
# POST /prematch/price
# ---------------------------------------------------------------------------

@router.post(
    "/prematch/price",
    summary="Price the match winner market",
    response_model=dict,
    tags=["Pre-Match"],
)
async def price_match_winner(request: MatchPriceRequest) -> dict[str, Any]:
    """
    Compute pre-match win probabilities (and optionally draw) with full
    margin applied via the Shin model.

    Returns true probabilities, adjusted probabilities, decimal odds,
    and the computed margin.
    """
    hb, fmt = _compute_hold_break(request)

    try:
        match_result = _match_engine.price_match(
            hold_break=hb,
            fmt=fmt,
            round_name=request.round_name,
            p1_starts_first=request.p1_starts_first,
        )
    except (DartsEngineError, DartsDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Pinnacle logit blend — applied before margin so that all downstream
    # odds derive from the blended probability. p1 is treated as "home"
    # for the 2-way blend. Darts has no draw in standard PDC formats.
    # Callers may pass Pinnacle odds in request as pinnacle_home_odds /
    # pinnacle_away_odds optional fields; absent → model-only fallback.
    _pin_home_darts = request.pinnacle_home_odds
    _pin_away_darts = request.pinnacle_away_odds

    # Auto-fetch Pinnacle from Optic Odds REST when not explicitly provided
    if _pin_home_darts is None and _pin_away_darts is None and request.fixture_id:
        try:
            import os, httpx
            _optic_key = os.getenv("OPTIC_ODDS_API_KEY", "")
            if _optic_key:
                async with httpx.AsyncClient(timeout=5.0) as _cl:
                    _r = await _cl.get(
                        "https://api.opticodds.com/api/v3/fixtures/odds",
                        params={"fixture_id": request.fixture_id, "market": "moneyline", "sportsbook": "pinnacle"},
                        headers={"X-Api-Key": _optic_key},
                    )
                    if _r.status_code == 200:
                        for _entry in _r.json().get("data", [{}])[0].get("odds", []):
                            if _entry.get("market_id") == "moneyline":
                                _american = _entry.get("price", 0)
                                _dec = (1 + _american / 100) if _american > 0 else (1 + 100 / abs(_american)) if _american < 0 else 0
                                if _dec > 1.0 and _entry.get("selection_line") is None:
                                    if not _pin_home_darts:
                                        _pin_home_darts = _dec
                                    else:
                                        _pin_away_darts = _dec
                        if _pin_home_darts and _pin_away_darts:
                            logger.info("[Prematch] Pinnacle auto-fetched: %.2f / %.2f for %s", _pin_home_darts, _pin_away_darts, request.fixture_id)
        except Exception as _exc:
            logger.warning("[Prematch] Pinnacle auto-fetch failed: %s — model-only", _exc)
    _darts_blend = _blend_layer.blend_sync(
        model_prob=match_result.p1_win,
        pinnacle_home_odds=float(_pin_home_darts) if _pin_home_darts is not None else None,
        pinnacle_away_odds=float(_pin_away_darts) if _pin_away_darts is not None else None,
        pinnacle_draw_odds=None,
    )
    _blended_p1 = _darts_blend.blended_prob
    _blended_p2 = 1.0 - _blended_p1

    # BUG-DARTS-PROB-NORM-001: When draw > 0, p1+p2 must be scaled down so
    # p1 + p2 + draw = 1.0.  Previously draw was added on top → sum > 1.0.
    _draw = match_result.draw if match_result.draw > 0 else 0.0
    if _draw > 0:
        _scale = (1.0 - _draw) / max(_blended_p1 + _blended_p2, 1e-9)
        true_probs: dict[str, float] = {
            "p1_win": _blended_p1 * _scale,
            "p2_win": _blended_p2 * _scale,
            "draw": _draw,
        }
    else:
        true_probs: dict[str, float] = {
            "p1_win": _blended_p1,
            "p2_win": _blended_p2,
        }

    adjusted, margin = _apply_margin(request, fmt, true_probs)
    decimal_odds = {
        k: round(1.0 / v, 4) if v > 1e-9 else None
        for k, v in adjusted.items()
    }

    logger.info(
        "prematch_price_computed",
        competition=request.competition_code,
        round=request.round_name,
        p1_id=request.p1_id,
        p2_id=request.p2_id,
        p1_win_true=round(match_result.p1_win, 4),
        p1_win_blended=round(_blended_p1, 4),
        blend_source=_darts_blend.source,
        margin=round(margin, 4),
    )

    return {
        "competition_code": request.competition_code,
        "round_name": request.round_name,
        "p1_id": request.p1_id,
        "p2_id": request.p2_id,
        "true_probabilities": {k: round(v, 6) for k, v in true_probs.items()},
        "adjusted_probabilities": {k: round(v, 6) for k, v in adjusted.items()},
        "decimal_odds": decimal_odds,
        "applied_margin": round(margin, 5),
        "regime": request.regime,
        "p1_hold": round(hb.p1_hold, 6),
        "p1_break": round(hb.p1_break, 6),
    }


# ---------------------------------------------------------------------------
# POST /prematch/exact-score
# ---------------------------------------------------------------------------

@router.post(
    "/prematch/exact-score",
    summary="Price the exact score market",
    response_model=dict,
    tags=["Pre-Match"],
)
async def price_exact_score(request: MatchPriceRequest) -> dict[str, Any]:
    """
    Compute the full distribution over exact final leg scores (e.g. 7-5, 7-3).
    Returns probabilities and decimal odds for each possible scoreline.
    """
    hb, fmt = _compute_hold_break(request)

    try:
        result = _match_engine.price_exact_scores(
            hb=hb,
            fmt=fmt,
            round_name=request.round_name,
            p1_starts=request.p1_starts_first,
        )
    except (DartsEngineError, DartsDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Apply margin to score distribution
    scores_probs = result["scores"]
    if scores_probs:
        try:
            adjusted_scores, margin = _apply_margin(request, fmt, scores_probs)
        except Exception:
            adjusted_scores = scores_probs
            margin = request.base_margin
    else:
        adjusted_scores = {}
        margin = request.base_margin

    decimal_odds = {
        k: round(1.0 / v, 2) if v > 1e-9 else None
        for k, v in adjusted_scores.items()
    }

    return {
        "competition_code": request.competition_code,
        "round_name": request.round_name,
        "p1_id": request.p1_id,
        "p2_id": request.p2_id,
        "true_probabilities": scores_probs,
        "adjusted_probabilities": adjusted_scores,
        "decimal_odds": decimal_odds,
        "most_likely_score": result["most_likely"],
        "applied_margin": round(margin, 5),
        "total_probability": result["total_prob"],
    }


# ---------------------------------------------------------------------------
# POST /prematch/handicap
# ---------------------------------------------------------------------------

@router.post(
    "/prematch/handicap",
    summary="Price the leg handicap market",
    response_model=dict,
    tags=["Pre-Match"],
)
async def price_handicap(request: HandicapRequest) -> dict[str, Any]:
    """
    Compute P(P1 covers the -handicap_legs handicap).
    E.g. handicap_legs=-1 means P1 concedes 1 leg.
    """
    hb, fmt = _compute_hold_break(request)

    try:
        result = _match_engine.price_handicap(
            hb=hb,
            fmt=fmt,
            round_name=request.round_name,
            p1_starts=request.p1_starts_first,
            handicap_legs=request.handicap_legs,
        )
    except (DartsEngineError, DartsDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    true_probs = {
        "p1_covers": result["p1_covers"],
        "p2_covers": result["p2_covers"],
    }
    adjusted, margin = _apply_margin(request, fmt, true_probs)

    return {
        "competition_code": request.competition_code,
        "round_name": request.round_name,
        "p1_id": request.p1_id,
        "p2_id": request.p2_id,
        "handicap_legs": request.handicap_legs,
        "true_probabilities": {k: round(v, 6) for k, v in true_probs.items()},
        "adjusted_probabilities": {k: round(v, 6) for k, v in adjusted.items()},
        "decimal_odds": {
            k: round(1.0 / v, 4) if v > 1e-9 else None
            for k, v in adjusted.items()
        },
        "applied_margin": round(margin, 5),
    }


# ---------------------------------------------------------------------------
# POST /prematch/totals
# ---------------------------------------------------------------------------

@router.post(
    "/prematch/totals",
    summary="Price total legs over/under market",
    response_model=dict,
    tags=["Pre-Match"],
)
async def price_totals(request: TotalsRequest) -> dict[str, Any]:
    """
    Compute P(total legs in match > total_line).
    """
    hb, fmt = _compute_hold_break(request)

    try:
        result = _match_engine.price_totals(
            hb=hb,
            fmt=fmt,
            round_name=request.round_name,
            p1_starts=request.p1_starts_first,
            total_line=request.total_line,
        )
    except (DartsEngineError, DartsDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    true_probs = {
        "over": result["over_prob"],
        "under": result["under_prob"],
    }
    adjusted, margin = _apply_margin(request, fmt, true_probs)

    return {
        "competition_code": request.competition_code,
        "round_name": request.round_name,
        "p1_id": request.p1_id,
        "p2_id": request.p2_id,
        "line": request.total_line,
        "true_probabilities": {k: round(v, 6) for k, v in true_probs.items()},
        "adjusted_probabilities": {k: round(v, 6) for k, v in adjusted.items()},
        "decimal_odds": {
            k: round(1.0 / v, 4) if v > 1e-9 else None
            for k, v in adjusted.items()
        },
        "applied_margin": round(margin, 5),
    }


# ---------------------------------------------------------------------------
# POST /prematch/180s
# ---------------------------------------------------------------------------

@router.post(
    "/prematch/180s",
    summary="Price total 180s over/under",
    response_model=dict,
    tags=["Pre-Match"],
)
async def price_180s(request: Match180Request) -> dict[str, Any]:
    """
    Compute P(total 180s in match > line) using the Poisson prop model.
    """
    hb, fmt = _compute_hold_break(request)

    # Get expected legs from legs distribution
    try:
        match_result = _match_engine.price_match(
            hold_break=hb,
            fmt=fmt,
            round_name=request.round_name,
            p1_starts_first=request.p1_starts_first,
        )
        exp_legs = _expected_legs(match_result.legs_distribution)
    except (DartsEngineError, DartsDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        result = _prop_180_model.price_total_180s(
            p1_180_rate=request.p1_180_rate,
            p2_180_rate=request.p2_180_rate,
            expected_legs=exp_legs,
            expected_visits_per_leg=request.expected_visits_per_leg,
            line=request.line,
            p1_id=request.p1_id,
            p2_id=request.p2_id,
            regime=request.regime,
        )
    except DartsDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except DartsEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    true_probs = {
        "over": result["over_prob"],
        "under": result["under_prob"],
    }
    adjusted, margin = _apply_margin(request, fmt, true_probs)

    return {
        "competition_code": request.competition_code,
        "round_name": request.round_name,
        "p1_id": request.p1_id,
        "p2_id": request.p2_id,
        "line": request.line,
        "lambda": result["lam"],
        "expected_legs": round(exp_legs, 2),
        "true_probabilities": {k: round(v, 6) for k, v in true_probs.items()},
        "adjusted_probabilities": {k: round(v, 6) for k, v in adjusted.items()},
        "decimal_odds": {
            k: round(1.0 / v, 4) if v > 1e-9 else None
            for k, v in adjusted.items()
        },
        "applied_margin": round(margin, 5),
    }


# ---------------------------------------------------------------------------
# POST /prematch/checkout
# ---------------------------------------------------------------------------

@router.post(
    "/prematch/checkout",
    summary="Price highest checkout over/under",
    response_model=dict,
    tags=["Pre-Match"],
)
async def price_checkout(request: HighestCheckoutRequest) -> dict[str, Any]:
    """
    Compute P(highest checkout in match > line).
    """
    from props.prop_checkout import PropCheckoutModel
    _checkout_model = PropCheckoutModel()

    hb, fmt = _compute_hold_break(request)

    try:
        match_result = _match_engine.price_match(
            hold_break=hb,
            fmt=fmt,
            round_name=request.round_name,
            p1_starts_first=request.p1_starts_first,
        )
        exp_legs = _expected_legs(match_result.legs_distribution)
    except (DartsEngineError, DartsDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    p1_model = {
        "checkout_ewm": request.p1_checkout_ewm,
        "checkout_pct_by_band": None,
        "legs_observed": request.p1_legs_observed,
        "confidence": request.p1_confidence,
    }
    p2_model = {
        "checkout_ewm": request.p2_checkout_ewm,
        "checkout_pct_by_band": None,
        "legs_observed": request.p2_legs_observed,
        "confidence": request.p2_confidence,
    }

    try:
        result = _checkout_model.price_highest_checkout(
            p1_checkout_model=p1_model,
            p2_checkout_model=p2_model,
            expected_legs=exp_legs,
            line=request.line,
            p1_id=request.p1_id,
            p2_id=request.p2_id,
            regime=request.regime,
        )
    except DartsDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except DartsEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    true_probs = {
        "over": result["over_prob"],
        "under": result["under_prob"],
    }
    adjusted, margin = _apply_margin(request, fmt, true_probs)

    return {
        "competition_code": request.competition_code,
        "round_name": request.round_name,
        "p1_id": request.p1_id,
        "p2_id": request.p2_id,
        "line": request.line,
        "expected_legs": round(exp_legs, 2),
        "true_probabilities": {k: round(v, 6) for k, v in true_probs.items()},
        "adjusted_probabilities": {k: round(v, 6) for k, v in adjusted.items()},
        "decimal_odds": {
            k: round(1.0 / v, 4) if v > 1e-9 else None
            for k, v in adjusted.items()
        },
        "applied_margin": round(margin, 5),
    }


# ---------------------------------------------------------------------------
# GET /prematch/markets/{match_id}
# ---------------------------------------------------------------------------

@router.get(
    "/prematch/markets/{match_id}",
    summary="Get all pre-match markets for a fixture",
    response_model=dict,
    tags=["Pre-Match"],
)
async def get_prematch_markets(match_id: str) -> dict[str, Any]:
    """
    Return available pre-match market types for a fixture.

    In a full deployment this would load fixture metadata from the database
    and compute all markets.  Currently returns the market schema and
    status for the specified match ID.
    """
    # NOTE: Full DB integration is handled at infrastructure layer.
    # This endpoint returns the canonical market structure for a fixture.
    logger.info("get_prematch_markets", match_id=match_id)
    return {
        "match_id": match_id,
        "available_markets": [
            {"id": "match_winner",      "endpoint": "/prematch/price",         "tier": 1},
            {"id": "correct_score",     "endpoint": "/prematch/exact-score",   "tier": 1},
            {"id": "handicap",          "endpoint": "/prematch/handicap",      "tier": 1},
            {"id": "total_legs",        "endpoint": "/prematch/totals",        "tier": 1},
            {"id": "multi_totals",      "endpoint": "/prematch/multi-totals",  "tier": 1},
            {"id": "first_leg_winner",  "endpoint": "/prematch/first-leg",     "tier": 1},
            {"id": "both_to_score",     "endpoint": "/prematch/both-to-score", "tier": 1},
            {"id": "race_to_x",         "endpoint": "/prematch/race-to-x",    "tier": 1},
            {"id": "winning_margin",    "endpoint": "/prematch/winning-margin","tier": 1},
            {"id": "180s",              "endpoint": "/prematch/180s",          "tier": 1},
            {"id": "highest_checkout",  "endpoint": "/prematch/checkout",      "tier": 1},
            {"id": "nine_darter_prop",  "endpoint": "/props/nine-darter",      "tier": 2},
            {"id": "most_180s",         "endpoint": "/props/most-180s",        "tier": 2},
            {"id": "checkout_range",    "endpoint": "/props/checkout-range",   "tier": 2},
            {"id": "sgp",               "endpoint": "/sgp/price",              "tier": 2},
        ],
        "total_markets": 15,
        "status": "markets_available",
        "base_pricing_endpoint": "/api/v1/darts/prematch/price",
    }


# ---------------------------------------------------------------------------
# Ecosystem routing constants
# ---------------------------------------------------------------------------

# Maximum regime allowed per ecosystem.  Ecosystems with limited data cannot
# be promoted to higher regimes regardless of what the caller requests.
_ECOSYSTEM_MAX_REGIME: dict[str, int] = {
    "pdc_mens": 2,        # full R0/R1/R2 path
    "pdc_womens": 2,      # WomensTransferModel + R0/R1/R2 (if womens data available)
    "wdf_open": 0,        # R0 primarily — limited WDF match data
    "development": 0,     # R0 only — PDC Development / Challenge Tour
    "team_doubles": 0,    # routed to World Cup pricer — R0 fallback here
}

# Ecosystem-specific base margin adjustments (in addition to blending engine factors).
# Women's ecosystem also gets 1.20x from WomensTransferModel.get_ecosystem_margin_multiplier().
_ECOSYSTEM_BASE_MARGIN_OVERRIDE: dict[str, Optional[float]] = {
    "pdc_mens": None,      # use caller's base_margin
    "pdc_womens": None,    # use caller's base_margin; 1.20x multiplier applied separately
    "wdf_open": None,      # use caller's base_margin; blending engine applies 1.20x
    "development": None,   # use caller's base_margin; blending engine applies 1.20x
    "team_doubles": None,
}


class EcosystemPriceRequest(MatchPriceRequest):
    """
    Ecosystem-aware match pricing request.

    Extends MatchPriceRequest with explicit ecosystem routing.  The
    ecosystem is normally inferred from the competition format code, but
    this endpoint allows explicit override for testing and advanced use cases.
    """
    elo_p1: Optional[float] = Field(default=None, gt=0, lt=4000)
    elo_p2: Optional[float] = Field(default=None, gt=0, lt=4000)


# ---------------------------------------------------------------------------
# POST /prematch/ecosystem-price
# ---------------------------------------------------------------------------

@router.post(
    "/prematch/ecosystem-price",
    summary="Ecosystem-aware match winner pricing with full routing",
    response_model=dict,
    tags=["Pre-Match"],
)
async def price_ecosystem_match(request: EcosystemPriceRequest) -> dict[str, Any]:
    """
    Price a match with explicit ecosystem-aware routing.

    Routing logic
    -------------
    pdc_mens:
        Uses the standard hold/break model pipeline at the caller's regime.
        Full R0/R1/R2 path available.

    pdc_womens:
        Uses WomensTransferModel for win probability when ELO ratings are
        supplied.  Falls back to standard hold/break model if not fitted.
        Ecosystem margin widening of 1.20x is applied via the blending engine
        (``pdc_womens`` is in the low-liquidity ecosystem set).

    wdf_open:
        Enforces maximum regime R0.  Uses standard hold/break model.
        Blending engine applies 1.20x ecosystem widening.

    development:
        Enforces R0 only.  Standard hold/break model, no upgrade permitted.
        Blending engine applies 1.20x ecosystem widening.

    team_doubles:
        Returns a 422 directing the caller to the /worldcup/ endpoints.
    """
    hb, fmt = _compute_hold_break(request)
    ecosystem = fmt.ecosystem

    # Team doubles are not priced via this endpoint
    if ecosystem == "team_doubles":
        raise HTTPException(
            status_code=422,
            detail=(
                "team_doubles ecosystem matches must be priced via "
                "/api/v1/darts/worldcup/price"
            ),
        )

    # Enforce ecosystem regime ceiling
    max_regime = _ECOSYSTEM_MAX_REGIME.get(ecosystem, 2)
    effective_regime = min(request.regime, max_regime)

    if effective_regime != request.regime:
        logger.info(
            "ecosystem_regime_capped",
            ecosystem=ecosystem,
            requested_regime=request.regime,
            effective_regime=effective_regime,
        )

    # Women's ecosystem: optionally use WomensTransferModel
    womens_p1_win: Optional[float] = None
    if ecosystem == "pdc_womens" and request.elo_p1 and request.elo_p2:
        try:
            from models.womens_transfer import WomensTransferModel
            womens_model = WomensTransferModel()
            womens_p1_win = womens_model.predict_proba(
                features={"elo_p1": request.elo_p1, "elo_p2": request.elo_p2},
                player1_id=request.p1_id,
                player2_id=request.p2_id,
            )
        except Exception as exc:
            logger.warning(
                "womens_model_fallback",
                error=str(exc),
                fallback="hold_break_model",
            )

    # Compute match probabilities
    try:
        match_result = _match_engine.price_match(
            hold_break=hb,
            fmt=fmt,
            round_name=request.round_name,
            p1_starts_first=request.p1_starts_first,
        )
    except (DartsEngineError, DartsDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # For women's ecosystem with valid WomensTransferModel result, blend
    # the ELO-based prediction with the Markov chain result (50/50 blend).
    if womens_p1_win is not None:
        blended_p1 = 0.5 * womens_p1_win + 0.5 * match_result.p1_win
        blended_p2 = 1.0 - blended_p1
    else:
        blended_p1 = match_result.p1_win
        blended_p2 = match_result.p2_win

    # BUG-DARTS-PROB-NORM-001: normalize when draw > 0
    _draw2 = match_result.draw if match_result.draw > 0 else 0.0
    if _draw2 > 0:
        _scale2 = (1.0 - _draw2) / max(blended_p1 + blended_p2, 1e-9)
        true_probs: dict[str, float] = {
            "p1_win": blended_p1 * _scale2,
            "p2_win": blended_p2 * _scale2,
            "draw": _draw2,
        }
    else:
        true_probs: dict[str, float] = {
            "p1_win": blended_p1,
            "p2_win": blended_p2,
        }

    # Build a modified request with the effective regime for margin computation
    margin_req = request.model_copy(update={"regime": effective_regime})
    adjusted, margin = _apply_margin(margin_req, fmt, true_probs)

    decimal_odds = {
        k: round(1.0 / v, 4) if v > 1e-9 else None
        for k, v in adjusted.items()
    }

    logger.info(
        "ecosystem_price_computed",
        competition=request.competition_code,
        ecosystem=ecosystem,
        round=request.round_name,
        p1_id=request.p1_id,
        p2_id=request.p2_id,
        effective_regime=effective_regime,
        womens_model_used=(womens_p1_win is not None),
        p1_win_true=round(blended_p1, 4),
        margin=round(margin, 4),
    )

    return {
        "competition_code": request.competition_code,
        "round_name": request.round_name,
        "ecosystem": ecosystem,
        "effective_regime": effective_regime,
        "regime_capped": (effective_regime != request.regime),
        "p1_id": request.p1_id,
        "p2_id": request.p2_id,
        "womens_transfer_model_used": (womens_p1_win is not None),
        "true_probabilities": {k: round(v, 6) for k, v in true_probs.items()},
        "adjusted_probabilities": {k: round(v, 6) for k, v in adjusted.items()},
        "decimal_odds": decimal_odds,
        "applied_margin": round(margin, 5),
        "p1_hold": round(hb.p1_hold, 6),
        "p1_break": round(hb.p1_break, 6),
    }


# ---------------------------------------------------------------------------
# POST /prematch/race-to-x — Race to X legs market
# ---------------------------------------------------------------------------


class RaceToXRequest(BaseModel):
    """Race-to-X legs request."""
    competition_code: str = Field(..., description="Format code, e.g. 'PDC_WC'")
    round_name: str = Field(..., description="Round name")
    p1_id: str
    p2_id: str
    p1_three_da: float = Field(..., gt=0, lt=200)
    p2_three_da: float = Field(..., gt=0, lt=200)
    race_to: int = Field(..., ge=1, le=14, description="Legs required to win race")
    p1_starts_first: bool = True
    base_margin: float = Field(default=0.05, gt=0.0, le=0.15)
    double_start_required: Optional[bool] = None


@router.post(
    "/prematch/race-to-x",
    summary="Race to X legs market",
    response_description="Win probabilities and decimal odds for race-to-X legs",
    tags=["Pre-Match"],
)
async def race_to_x_price(request: RaceToXRequest) -> dict[str, Any]:
    """
    Price the 'Race to X legs' market.

    Given hold/break probabilities derived from the players' 3DA averages,
    compute the exact probability that player 1 reaches X legs before player 2.

    This is the canonical 'first to X legs' market:
    e.g., 'Race to 3 legs' in a Premier League match, or 'First to 5' in
    the first round of the World Championship.
    """
    from engines.match_layer.race_to_x_engine import race_to_x, apply_margin

    hb_req = MatchPriceRequest(
        competition_code=request.competition_code,
        round_name=request.round_name,
        p1_id=request.p1_id,
        p2_id=request.p2_id,
        p1_three_da=request.p1_three_da,
        p2_three_da=request.p2_three_da,
        p1_starts_first=request.p1_starts_first,
        regime=0,
        starter_confidence=1.0,
        source_confidence=1.0,
        model_agreement=1.0,
        market_liquidity="high",
        base_margin=request.base_margin,
        double_start_required=request.double_start_required,
    )
    hb, _ = _compute_hold_break(hb_req)

    try:
        result = race_to_x(
            target=request.race_to,
            p1_hold=hb.p1_hold,
            p2_hold=hb.p2_hold,
            p1_starts=request.p1_starts_first,
        )
    except (ValueError, Exception) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    true_probs = {"p1_win": round(result.p1_win, 6), "p2_win": round(result.p2_win, 6)}
    adjusted = apply_margin(result, request.base_margin)
    decimal_odds = {k: round(1.0 / v, 4) if v > 1e-9 else None for k, v in adjusted.items()}

    logger.info(
        "race_to_x_priced",
        competition=request.competition_code,
        round_name=request.round_name,
        race_to=request.race_to,
        p1_win_true=round(result.p1_win, 4),
    )

    return {
        "competition_code": request.competition_code,
        "round_name": request.round_name,
        "race_to": request.race_to,
        "p1_id": request.p1_id,
        "p2_id": request.p2_id,
        "p1_hold": round(hb.p1_hold, 6),
        "p1_break": round(hb.p1_break, 6),
        "true_probabilities": true_probs,
        "adjusted_probabilities": adjusted,
        "decimal_odds": decimal_odds,
        "applied_margin": round(request.base_margin, 5),
    }


# ---------------------------------------------------------------------------
# POST /prematch/smart-price — DB-backed price using stored player stats
# ---------------------------------------------------------------------------

class SmartPriceRequest(BaseModel):
    """Smart price request — fetches 3DA and ELO from DB automatically."""
    competition_code: str = Field(..., description="Format code, e.g. 'PDC_WC'")
    round_name: str = Field(..., description="Round name, e.g. 'Final'")
    p1_id: str = Field(..., description="Player 1 UUID (from /players/search)")
    p2_id: str = Field(..., description="Player 2 UUID")
    p1_starts_first: bool = True
    base_margin: float = Field(default=0.05, gt=0.0, le=0.15)
    fallback_3da: float = Field(default=50.0, gt=0, lt=200,
                                 description="3DA fallback if not in DB")
    # Pinnacle blend — optional.  When provided, the Markov+ML probability is
    # further blended with Pinnacle in logit space (30% model / 70% Pinnacle,
    # per darts BlendConfig) before margin is applied.
    fixture_id: Optional[str] = Field(
        None,
        description="Optic Odds fixture ID — triggers automatic Pinnacle fetch when OPTIC_ODDS_API_KEY is set",
    )
    pinnacle_home_odds: Optional[float] = Field(
        None, gt=1.0,
        description="Explicit Pinnacle decimal odds for player 1",
    )
    pinnacle_away_odds: Optional[float] = Field(
        None, gt=1.0,
        description="Explicit Pinnacle decimal odds for player 2",
    )


@router.post(
    "/prematch/smart-price",
    summary="Price match using player stats from database",
    response_model=dict,
    tags=["Pre-Match"],
)
async def smart_price(
    request: SmartPriceRequest,
    session: AsyncSession = Depends(get_session_dependency),
) -> dict[str, Any]:
    """
    Fetch player 3DA and ELO from the database, then compute prices.
    No need to supply p1_three_da / p2_three_da — they are loaded automatically.
    """
    from sqlalchemy import text as sql_text

    # Fetch both players from DB; fall back to request-supplied values if DB unavailable
    _db_error: Optional[str] = None
    p1_three_da: float = request.fallback_3da
    p2_three_da: float = request.fallback_3da
    p1_elo: float = 1500.0
    p2_elo: float = 1500.0
    src_conf: float = 0.85
    p1_rank: float = 200.0
    p2_rank: float = 200.0
    p1_country: str = ""
    p2_country: str = ""
    p1_name_db: Optional[str] = None
    p2_name_db: Optional[str] = None

    try:
        result = await session.execute(
            sql_text("""
                SELECT p.id,
                       COALESCE(p.dartsorakel_3da, :fallback) AS three_da,
                       COALESCE(e.rating_after, 1500.0) AS elo_rating,
                       p.source_confidence,
                       p.first_name, p.last_name, p.nickname,
                       COALESCE(p.pdc_ranking, 200) AS pdc_ranking,
                       p.country_code
                FROM darts_players p
                LEFT JOIN darts_elo_ratings e
                    ON e.player_id = p.id AND e.pool = 'pdc_mens'
                    AND e.match_date = (
                        SELECT MAX(e2.match_date) FROM darts_elo_ratings e2
                        WHERE e2.player_id = p.id AND e2.pool = 'pdc_mens'
                    )
                WHERE p.id IN (:p1_id, :p2_id)
            """),
            {"p1_id": request.p1_id, "p2_id": request.p2_id,
             "fallback": request.fallback_3da},
        )
        rows = {row[0]: row for row in result.fetchall()}

        if request.p1_id in rows:
            p1_row = rows[request.p1_id]
            p1_three_da = float(p1_row[1])
            p1_elo = float(p1_row[2])
            src_conf_p1 = float(p1_row[3] or 0.85)
            p1_rank = float(p1_row[7] or 200)
            p1_country = p1_row[8] or ""
            p1_nick = p1_row[6] or ""
            p1_name_db = p1_nick or f"{p1_row[4] or ''} {p1_row[5] or ''}".strip() or None
        else:
            src_conf_p1 = 0.85

        if request.p2_id in rows:
            p2_row = rows[request.p2_id]
            p2_three_da = float(p2_row[1])
            p2_elo = float(p2_row[2])
            src_conf_p2 = float(p2_row[3] or 0.85)
            p2_rank = float(p2_row[7] or 200)
            p2_country = p2_row[8] or ""
            p2_nick = p2_row[6] or ""
            p2_name_db = p2_nick or f"{p2_row[4] or ''} {p2_row[5] or ''}".strip() or None
        else:
            src_conf_p2 = 0.85

        src_conf = min(src_conf_p1, src_conf_p2)
    except Exception as exc:
        _db_error = str(exc)
        logger.warning("smart_price_db_unavailable", error=_db_error[:100])
        # Use fallback 3DA from request if provided, else use default
        p1_three_da = request.fallback_3da
        p2_three_da = request.fallback_3da

    # --- ML model blend: try R1 (file-based) then fall back to R0 (DB) ---
    ml_p1_win: Optional[float] = None
    ml_source = "unavailable"
    try:
        from models.r1_file_predictor import r1_file_predictor
        # Derive competition context for the 38-feature model
        _comp_code = request.competition_code
        _televised_formats = {
            "PDC_WC", "PDC_PL", "PDC_WM", "PDC_GS", "PDC_GP", "PDC_PCF", "PDC_UK",
        }
        try:
            _fmt_obj = get_format(_comp_code)
            _ecosystem = _fmt_obj.ecosystem
        except Exception:
            _ecosystem = "pdc_mens"
        r1_prob = r1_file_predictor.predict(
            p1_elo=p1_elo,
            p2_elo=p2_elo,
            p1_3da=p1_three_da,
            p2_3da=p2_three_da,
            format_code=_comp_code,
            ecosystem=_ecosystem,
            is_televised=_comp_code in _televised_formats,
        )
        if r1_prob is not None:
            ml_p1_win = r1_prob
            ml_source = "r1_file"
    except Exception:
        pass

    if ml_p1_win is None:
        try:
            from models.loader import model_store
            import numpy as np
            r0 = model_store.get("r0_logit")
            if r0 is not None:
                fmt = request.competition_code
                year = 2026
                feat = np.array([[
                    p1_elo - p2_elo,
                    p1_elo, p2_elo,
                    p1_three_da - p2_three_da,
                    p1_three_da, p2_three_da,
                    p2_rank - p1_rank,
                    min(p1_rank, 300) / 300.0,
                    min(p2_rank, 300) / 300.0,
                    1.0 if (p1_country and p1_country == p2_country) else 0.0,
                    (year - 2000) / 30.0,
                    1.0 if fmt.startswith("PDC_WC") else 0.0,
                    1.0 if fmt == "PDC_PL" else 0.0,
                    1.0 if not (fmt.startswith("PDC_WC") or fmt == "PDC_PL") else 0.0,
                ]], dtype=np.float32)
                scaler = r0["scaler"]
                model = r0["model"]
                feat_s = scaler.transform(feat)
                ml_p1_win = float(model.predict_proba(feat_s)[0, 1])
                ml_source = "r0_logit"
        except Exception:
            pass  # model unavailable — use Markov only

    # Build internal request
    inner = MatchPriceRequest(
        competition_code=request.competition_code,
        round_name=request.round_name,
        p1_id=request.p1_id,
        p2_id=request.p2_id,
        p1_three_da=p1_three_da,
        p2_three_da=p2_three_da,
        p1_starts_first=request.p1_starts_first,
        regime=0,  # R0 — no live visit data
        starter_confidence=0.85,
        source_confidence=src_conf,
        model_agreement=1.0,
        market_liquidity="high",
        base_margin=request.base_margin,
    )

    hb, fmt = _compute_hold_break(inner)
    try:
        match_result = _match_engine.price_match(
            hold_break=hb,
            fmt=fmt,
            round_name=request.round_name,
            p1_starts_first=request.p1_starts_first,
        )
    except (DartsEngineError, DartsDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    markov_p1_win = match_result.p1_win
    markov_p2_win = match_result.p2_win

    # Blend: 60% Markov (format-aware), 40% ML (R1 file or R0 fallback)
    if ml_p1_win is not None:
        blend_p1 = 0.60 * markov_p1_win + 0.40 * ml_p1_win
        blend_p2 = 1.0 - blend_p1
        final_p1, final_p2 = blend_p1, blend_p2
        pricing_model = f"markov_{ml_source}_blend"
    else:
        final_p1, final_p2 = markov_p1_win, markov_p2_win
        pricing_model = "markov_only"

    # --- Pinnacle logit blend (darts config: 30% model / 70% Pinnacle) ---
    # Applied on top of the Markov+ML blended probability.  When no Pinnacle
    # data is available the probability passes through unchanged.
    _smart_pin_home = request.pinnacle_home_odds
    _smart_pin_away = request.pinnacle_away_odds
    _smart_pinnacle_source = "none"

    if _smart_pin_home is None and _smart_pin_away is None and request.fixture_id:
        # Auto-fetch Pinnacle via PricingLayer when fixture_id provided.
        try:
            _smart_blend_pre = await _blend_layer.blend_pre_match(
                fixture_id=request.fixture_id,
                model_prob=final_p1,
                market_type="moneyline",
            )
            if _smart_blend_pre.source != "model_only":
                final_p1 = _smart_blend_pre.blended_prob
                final_p2 = 1.0 - final_p1
                _smart_pinnacle_source = "optic_auto"
                pricing_model = f"{pricing_model}_pinnacle"
        except Exception as _sp_err:
            logger.warning("smart_price_pinnacle_auto_fetch_failed", error=str(_sp_err))
    elif _smart_pin_home is not None and _smart_pin_away is not None:
        # Explicit Pinnacle odds provided — use blend_sync().
        _smart_blend_sync = _blend_layer.blend_sync(
            model_prob=final_p1,
            pinnacle_home_odds=float(_smart_pin_home),
            pinnacle_away_odds=float(_smart_pin_away),
            pinnacle_draw_odds=None,
        )
        if _smart_blend_sync.source != "model_only":
            final_p1 = _smart_blend_sync.blended_prob
            final_p2 = 1.0 - final_p1
            _smart_pinnacle_source = "explicit"
            pricing_model = f"{pricing_model}_pinnacle"

    # BUG-DARTS-PROB-NORM-001: normalize when draw > 0
    _draw3 = match_result.draw if match_result.draw > 0 else 0.0
    if _draw3 > 0:
        _scale3 = (1.0 - _draw3) / max(final_p1 + final_p2, 1e-9)
        true_probs: dict[str, float] = {"p1_win": final_p1 * _scale3, "p2_win": final_p2 * _scale3, "draw": _draw3}
    else:
        true_probs: dict[str, float] = {"p1_win": final_p1, "p2_win": final_p2}

    adjusted, margin = _apply_margin(inner, fmt, true_probs)
    decimal_odds = {k: round(1.0 / v, 4) if v > 1e-9 else None for k, v in adjusted.items()}

    p1_display = p1_name_db or request.p1_id
    p2_display = p2_name_db or request.p2_id

    return {
        "competition_code": request.competition_code,
        "round_name": request.round_name,
        "p1": {"id": request.p1_id, "name": p1_display,
               "three_da": round(p1_three_da, 2), "elo": round(p1_elo, 1),
               "pdc_ranking": int(p1_rank)},
        "p2": {"id": request.p2_id, "name": p2_display,
               "three_da": round(p2_three_da, 2), "elo": round(p2_elo, 1),
               "pdc_ranking": int(p2_rank)},
        "true_probabilities": {k: round(v, 6) for k, v in true_probs.items()},
        "adjusted_probabilities": {k: round(v, 6) for k, v in adjusted.items()},
        "decimal_odds": decimal_odds,
        "applied_margin": round(margin, 5),
        "p1_hold": round(hb.p1_hold, 6),
        "p1_break": round(hb.p1_break, 6),
        "pricing_model": pricing_model,
        "model_components": {
            "markov_p1_win": round(markov_p1_win, 6),
            "ml_p1_win": round(ml_p1_win, 6) if ml_p1_win is not None else None,
            "ml_source": ml_source,
            "pinnacle_blend_source": _smart_pinnacle_source,
        },
        "data_sources": {
            "three_da": "db" if not _db_error else "fallback",
            "elo": "db" if not _db_error else "default_1500",
        },
        "_db_available": _db_error is None,
    }


# ---------------------------------------------------------------------------
# GET /prematch/player-search — search players by name
# ---------------------------------------------------------------------------

@router.get(
    "/prematch/player-search",
    summary="Search players by name (returns IDs for smart-price)",
    response_model=dict,
    tags=["Pre-Match"],
)
async def search_players_for_pricing(
    q: str,
    limit: int = 20,
    session: AsyncSession = Depends(get_session_dependency),
) -> dict[str, Any]:
    """
    Search players by name. Returns player IDs suitable for use in
    /prematch/smart-price.
    """
    from sqlalchemy import text as sql_text

    result = await session.execute(
        sql_text("""
            SELECT p.id, p.first_name, p.last_name, p.nickname,
                   p.pdc_ranking, p.dartsorakel_3da, p.dartsorakel_rank,
                   p.country_code, p.pdc_id,
                   COALESCE(e.rating_after, 1500.0) AS elo_rating
            FROM darts_players p
            LEFT JOIN darts_elo_ratings e
                ON e.player_id = p.id AND e.pool = 'pdc_mens'
            WHERE p.first_name ILIKE :pattern
               OR p.last_name  ILIKE :pattern
               OR p.nickname   ILIKE :pattern
               OR (p.first_name || ' ' || p.last_name) ILIKE :pattern
            ORDER BY p.pdc_ranking ASC NULLS LAST
            LIMIT :limit
        """),
        {"pattern": f"%{q}%", "limit": min(limit, 50)},
    )
    rows = result.fetchall()
    return {
        "query": q,
        "count": len(rows),
        "players": [
            {
                "player_id": row[0],
                "name": f"{row[1] or ''} {row[2] or ''}".strip(),
                "nickname": row[3],
                "pdc_ranking": row[4],
                "dartsorakel_3da": row[5],
                "dartsorakel_rank": row[6],
                "country_code": row[7],
                "pdc_id": row[8],
                "elo_rating": round(float(row[9]), 1) if row[9] else None,
            }
            for row in rows
        ],
    }


# ---------------------------------------------------------------------------
# POST /prematch/winning-margin — P(win by exactly N legs) distribution
# ---------------------------------------------------------------------------


class WinningMarginRequest(BaseModel):
    """Winning margin distribution request."""
    competition_code: str = Field(..., description="Format code, e.g. 'PDC_WC'")
    round_name: str = Field(..., description="Round name")
    p1_id: str
    p2_id: str
    p1_three_da: float = Field(..., gt=0, lt=200)
    p2_three_da: float = Field(..., gt=0, lt=200)
    p1_starts_first: bool = True
    base_margin: float = Field(default=0.05, gt=0.0, le=0.15)
    double_start_required: Optional[bool] = None


@router.post(
    "/prematch/winning-margin",
    summary="Winning margin distribution",
    tags=["Pre-Match"],
)
async def winning_margin(request: WinningMarginRequest) -> dict[str, Any]:
    """
    Return the full winning-margin probability distribution.

    For each possible winning margin (win by 1, 2, 3 ... N legs), returns
    the probability that the winning player wins by exactly that margin.

    Covers both 'player 1 wins by X' and 'player 2 wins by X' outcomes.
    Useful for exact-score and margin-band markets.
    """
    from engines.match_layer.race_to_x_engine import race_to_x

    hb_req = MatchPriceRequest(
        competition_code=request.competition_code,
        round_name=request.round_name,
        p1_id=request.p1_id,
        p2_id=request.p2_id,
        p1_three_da=request.p1_three_da,
        p2_three_da=request.p2_three_da,
        p1_starts_first=request.p1_starts_first,
        regime=0,
        starter_confidence=1.0,
        source_confidence=1.0,
        model_agreement=1.0,
        market_liquidity="high",
        base_margin=request.base_margin,
        double_start_required=request.double_start_required,
    )
    hb, fmt = _compute_hold_break(hb_req)

    # Get exact score distribution from match combinatorics
    try:
        exact = _match_engine.exact_score_distribution(
            hold_break=hb,
            fmt=fmt,
            round_name=request.round_name,
            p1_starts_first=request.p1_starts_first,
        )
    except (AttributeError, NotImplementedError):
        # Fallback: use race_to_x for incremental margins
        exact = {}

    if not exact:
        # Build margin distribution from race-to-x differences
        # For a first-to-N legs format, compute P(score = W:L) for each W,L
        legs_to_win = fmt.legs_per_set * fmt.sets_to_win if hasattr(fmt, "sets_to_win") else 7
        margins: dict[str, float] = {}
        total = 0.0
        for lost in range(legs_to_win):
            # P(p1 wins W:lost)
            r1 = race_to_x(legs_to_win, hb.p1_hold, hb.p2_hold, request.p1_starts_first)
            margin_val = legs_to_win - lost
            key_p1 = f"p1_wins_{margin_val}"
            key_p2 = f"p2_wins_{margin_val}"
            # Approximate via symmetry
            margins[key_p1] = round(r1.p1_win / legs_to_win, 6)
            margins[key_p2] = round(r1.p2_win / legs_to_win, 6)
            total += margins[key_p1] + margins[key_p2]
        # Renormalise
        if total > 0:
            margins = {k: round(v / total, 6) for k, v in margins.items()}

        return {
            "competition_code": request.competition_code,
            "round_name": request.round_name,
            "p1_id": request.p1_id,
            "p2_id": request.p2_id,
            "margin_distribution": margins,
            "note": "Approximate distribution — exact score engine not available for this format.",
        }

    # Apply margin to each outcome
    margin_half = request.base_margin / (2 * len(exact)) if exact else 0.0
    adj = {k: round(v + margin_half, 6) for k, v in exact.items()}

    return {
        "competition_code": request.competition_code,
        "round_name": request.round_name,
        "p1_id": request.p1_id,
        "p2_id": request.p2_id,
        "margin_distribution": exact,
        "adjusted_probabilities": adj,
        "decimal_odds": {
            k: round(1.0 / v, 4) if v > 1e-9 else None for k, v in adj.items()
        },
        "applied_margin": round(request.base_margin, 5),
    }


# ---------------------------------------------------------------------------
# POST /prematch/first-leg — First leg winner market
# ---------------------------------------------------------------------------

@router.post(
    "/prematch/first-leg",
    summary="First leg winner market",
    tags=["Pre-Match"],
)
async def price_first_leg(request: FirstLegRequest) -> dict[str, Any]:
    """
    Price the first leg winner market.

    The first leg probability is the hold/break probability of the opening
    server: if P1 starts first, P(P1 wins leg 1) = P1's hold probability.

    Offered by Pinnacle, bet365, and Unibet for all major darts tournaments.
    """
    hb_req = MatchPriceRequest(
        competition_code=request.competition_code,
        round_name=request.round_name,
        p1_id=request.p1_id,
        p2_id=request.p2_id,
        p1_three_da=request.p1_three_da,
        p2_three_da=request.p2_three_da,
        p1_starts_first=request.p1_starts_first,
        regime=0,
        starter_confidence=1.0,
        source_confidence=1.0,
        model_agreement=1.0,
        market_liquidity="high",
        base_margin=request.base_margin,
        double_start_required=request.double_start_required,
    )
    hb, fmt = _compute_hold_break(hb_req)

    if request.p1_starts_first:
        p1_true = hb.p1_hold
        p2_true = 1.0 - hb.p1_hold
    else:
        p1_true = 1.0 - hb.p2_hold
        p2_true = hb.p2_hold

    adj = _shin_model.apply_shin_margin(
        true_probs={"p1_wins_leg1": p1_true, "p2_wins_leg1": p2_true},
        target_margin=request.base_margin,
    )

    return {
        "market": "first_leg_winner",
        "competition_code": request.competition_code,
        "round_name": request.round_name,
        "p1_id": request.p1_id,
        "p2_id": request.p2_id,
        "p1_starts_first": request.p1_starts_first,
        "true_probabilities": {
            "p1_wins_leg1": round(p1_true, 6),
            "p2_wins_leg1": round(p2_true, 6),
        },
        "adjusted_probabilities": {k: round(v, 6) for k, v in adj.items()},
        "decimal_odds": {
            "p1_wins_leg1": round(1.0 / adj["p1_wins_leg1"], 4) if adj.get("p1_wins_leg1", 0) > 1e-9 else None,
            "p2_wins_leg1": round(1.0 / adj["p2_wins_leg1"], 4) if adj.get("p2_wins_leg1", 0) > 1e-9 else None,
        },
        "applied_margin": round(request.base_margin, 5),
        "model_inputs": {
            "p1_hold": round(hb.p1_hold, 6),
            "p2_hold": round(hb.p2_hold, 6),
        },
    }


# ---------------------------------------------------------------------------
# POST /prematch/both-to-score — Both players to reach X legs market
# ---------------------------------------------------------------------------

@router.post(
    "/prematch/both-to-score",
    summary="Both players to score at least X legs",
    tags=["Pre-Match"],
)
async def price_both_to_score(request: BothToScoreRequest) -> dict[str, Any]:
    """
    Price 'both players to score X+ legs' market using exact DP.

    Covers markets like 'both to score 3+', 'both to score 5+'.
    Offered by Pinnacle, bet365, Unibet for PDC Premier League and WC formats.
    """
    from functools import lru_cache

    hb_req = MatchPriceRequest(
        competition_code=request.competition_code,
        round_name=request.round_name,
        p1_id=request.p1_id,
        p2_id=request.p2_id,
        p1_three_da=request.p1_three_da,
        p2_three_da=request.p2_three_da,
        p1_starts_first=request.p1_starts_first,
        regime=0,
        starter_confidence=1.0,
        source_confidence=1.0,
        model_agreement=1.0,
        market_liquidity="high",
        base_margin=request.base_margin,
        double_start_required=request.double_start_required,
    )
    hb, fmt = _compute_hold_break(hb_req)

    legs_to_win = getattr(fmt, "legs_to_win", None) or getattr(fmt, "legs_per_set", 7)
    n = request.min_legs_each

    if n >= legs_to_win:
        raise HTTPException(
            status_code=422,
            detail=f"min_legs_each={n} >= legs_to_win={legs_to_win}."
        )

    p1_hold = hb.p1_hold
    p1_break = 1.0 - hb.p2_hold

    @lru_cache(maxsize=None)
    def p_both(a: int, b: int, sv: int) -> float:
        if a >= legs_to_win:
            return 1.0 if b >= n else 0.0
        if b >= legs_to_win:
            return 1.0 if a >= n else 0.0
        p = p1_hold if sv == 1 else p1_break
        ns = 2 if sv == 1 else 1
        return p * p_both(a+1, b, ns) + (1-p) * p_both(a, b+1, ns)

    sv0 = 1 if request.p1_starts_first else 2
    p_yes = p_both(0, 0, sv0)
    p_no = 1.0 - p_yes

    adj = _shin_model.apply_shin_margin(
        true_probs={"yes": p_yes, "no": p_no},
        target_margin=request.base_margin,
    )

    return {
        "market": "both_to_score",
        "min_legs_each": n,
        "competition_code": request.competition_code,
        "round_name": request.round_name,
        "p1_id": request.p1_id,
        "p2_id": request.p2_id,
        "legs_to_win": legs_to_win,
        "true_probabilities": {
            "yes_both_score": round(p_yes, 6),
            "no": round(p_no, 6),
        },
        "adjusted_probabilities": {k: round(v, 6) for k, v in adj.items()},
        "decimal_odds": {
            "yes": round(1.0 / adj["yes"], 4) if adj.get("yes", 0) > 1e-9 else None,
            "no": round(1.0 / adj["no"], 4) if adj.get("no", 0) > 1e-9 else None,
        },
        "applied_margin": round(request.base_margin, 5),
    }


# ---------------------------------------------------------------------------
# POST /prematch/multi-totals — Multiple O/U leg lines (Pinnacle-style)
# ---------------------------------------------------------------------------

@router.post(
    "/prematch/multi-totals",
    summary="All total legs O/U lines in one call (Pinnacle-style)",
    tags=["Pre-Match"],
)
async def price_multi_totals(request: MultiLegLineRequest) -> dict[str, Any]:
    """
    Return over/under odds for all available leg total lines in a single call.

    Pinnacle lists 4-6 O/U total lines per match. This endpoint returns all
    lines with full Shin margin applied, suitable for direct display.

    Score distribution is computed via forward DP (exact, not Monte Carlo).
    """
    hb_req = MatchPriceRequest(
        competition_code=request.competition_code,
        round_name=request.round_name,
        p1_id=request.p1_id,
        p2_id=request.p2_id,
        p1_three_da=request.p1_three_da,
        p2_three_da=request.p2_three_da,
        p1_starts_first=request.p1_starts_first,
        regime=0,
        starter_confidence=1.0,
        source_confidence=1.0,
        model_agreement=1.0,
        market_liquidity="high",
        base_margin=request.base_margin,
        double_start_required=request.double_start_required,
    )
    hb, fmt = _compute_hold_break(hb_req)

    legs_to_win = getattr(fmt, "legs_to_win", None) or getattr(fmt, "legs_per_set", 7)
    min_total = legs_to_win
    max_total = 2 * legs_to_win - 1

    p1_hold = hb.p1_hold
    p1_break = 1.0 - hb.p2_hold

    # Forward DP: track probability mass at each state
    sv0 = 1 if request.p1_starts_first else 2
    states: dict[tuple[int, int, int], float] = {(0, 0, sv0): 1.0}
    score_dist: dict[int, float] = {}

    while states:
        new_states: dict[tuple[int, int, int], float] = {}
        for (a, b, sv), prob in states.items():
            p = p1_hold if sv == 1 else p1_break
            ns = 2 if sv == 1 else 1
            for leg_winner, p_leg in ((1, p), (2, 1 - p)):
                na = a + (1 if leg_winner == 1 else 0)
                nb = b + (1 if leg_winner == 2 else 0)
                if na >= legs_to_win or nb >= legs_to_win:
                    total = na + nb
                    score_dist[total] = score_dist.get(total, 0.0) + prob * p_leg
                else:
                    k = (na, nb, ns)
                    new_states[k] = new_states.get(k, 0.0) + prob * p_leg
        states = new_states

    if not score_dist:
        raise HTTPException(status_code=500, detail="Score distribution computation failed.")

    # Generate half-point lines
    lines = [t + 0.5 for t in range(min_total - 1, max_total)]
    result_lines = []
    for line in lines:
        p_over = sum(v for t, v in score_dist.items() if t > line)
        p_under = sum(v for t, v in score_dist.items() if t < line)
        total_p = p_over + p_under
        if total_p < 1e-9:
            continue
        p_over /= total_p
        p_under /= total_p
        adj = _shin_model.apply_shin_margin(
            true_probs={"over": p_over, "under": p_under},
            target_margin=request.base_margin,
        )
        result_lines.append({
            "line": line,
            "true_over": round(p_over, 6),
            "true_under": round(p_under, 6),
            "odds_over": round(1.0 / adj["over"], 4) if adj.get("over", 0) > 1e-9 else None,
            "odds_under": round(1.0 / adj["under"], 4) if adj.get("under", 0) > 1e-9 else None,
        })

    return {
        "market": "multi_totals",
        "competition_code": request.competition_code,
        "round_name": request.round_name,
        "p1_id": request.p1_id,
        "p2_id": request.p2_id,
        "legs_to_win": legs_to_win,
        "score_distribution": {str(k): round(v, 6) for k, v in sorted(score_dist.items())},
        "total_lines": result_lines,
        "applied_margin": round(request.base_margin, 5),
    }


# ---------------------------------------------------------------------------
# POST /prematch/set-betting — Correct set score (World Championship format)
# ---------------------------------------------------------------------------

class SetBettingRequest(BaseModel):
    """Set betting / set handicap request (PDC_WC and similar formats)."""
    competition_code: str = Field(default="PDC_WC")
    round_name: str = Field(default="Final")
    p1_id: str
    p2_id: str
    p1_three_da: float = Field(..., gt=0, lt=200)
    p2_three_da: float = Field(..., gt=0, lt=200)
    p1_starts_first: bool = True
    base_margin: float = Field(default=0.05, gt=0.0, le=0.15)
    double_start_required: Optional[bool] = None


@router.post(
    "/prematch/set-betting",
    summary="Correct set score market (PDC World Championship)",
    tags=["Pre-Match"],
)
async def price_set_betting(request: SetBettingRequest) -> dict[str, Any]:
    """
    Price the correct set score market for sets-and-legs formats.

    Used for PDC World Championship (e.g. 'P1 wins 7-5 sets').
    Returns a probability distribution over all possible set scores,
    plus the set winner market (P1/P2 wins the match by sets).

    Pinnacle and bet365 both offer this for the PDC World Championship.

    Works only with sets-format rounds (PDC_WC, WDF_WC).
    Raises 422 if called for a legs-only format.
    """
    hb_req = MatchPriceRequest(
        competition_code=request.competition_code,
        round_name=request.round_name,
        p1_id=request.p1_id,
        p2_id=request.p2_id,
        p1_three_da=request.p1_three_da,
        p2_three_da=request.p2_three_da,
        p1_starts_first=request.p1_starts_first,
        regime=0,
        starter_confidence=1.0,
        source_confidence=1.0,
        model_agreement=1.0,
        market_liquidity="high",
        base_margin=request.base_margin,
        double_start_required=request.double_start_required,
    )
    hb, fmt = _compute_hold_break(hb_req)

    round_fmt = fmt.get_round(request.round_name)
    if not round_fmt.is_sets_format:
        raise HTTPException(
            status_code=422,
            detail=f"Round '{request.round_name}' in '{request.competition_code}' is not a sets format. "
                   "Use /prematch/exact-score for legs-only formats."
        )

    sets_to_win = round_fmt.sets_to_win
    legs_per_set = round_fmt.legs_per_set
    legs_to_win_set = legs_per_set  # legs needed to win a set (first-to)

    p1_hold = hb.p1_hold
    p1_break = 1.0 - hb.p2_hold

    # Forward DP over set-level states
    # First compute P(P1 wins a set) and P(P2 wins a set) assuming leg alternation
    # A set is itself a race-to-legs_per_set within the set
    from engines.match_layer.race_to_x_engine import race_to_x

    # Set-level probabilities (assuming P1 starts first leg of each set)
    # In WC format, set starter alternates between sets based on match rules
    # Simplification: use overall hold/break to get set win prob
    set_result_p1 = race_to_x(
        target=legs_to_win_set,
        p1_hold=p1_hold,
        p2_hold=hb.p2_hold,
        p1_starts=request.p1_starts_first,
    )
    p1_wins_set = set_result_p1.p1_win
    p2_wins_set = 1.0 - p1_wins_set

    # Forward DP over match-level set scores
    sv0 = 1 if request.p1_starts_first else 2
    states: dict[tuple[int, int], float] = {(0, 0): 1.0}
    set_score_dist: dict[str, float] = {}

    while states:
        new_states: dict[tuple[int, int], float] = {}
        for (s1, s2), prob in states.items():
            for set_winner, p_set in ((1, p1_wins_set), (2, p2_wins_set)):
                ns1 = s1 + (1 if set_winner == 1 else 0)
                ns2 = s2 + (1 if set_winner == 2 else 0)
                if ns1 >= sets_to_win or ns2 >= sets_to_win:
                    score_key = f"p1_{ns1}_p2_{ns2}"
                    set_score_dist[score_key] = set_score_dist.get(score_key, 0.0) + prob * p_set
                else:
                    k = (ns1, ns2)
                    new_states[k] = new_states.get(k, 0.0) + prob * p_set
        states = new_states

    # Aggregate match winner
    p1_match_win = sum(v for k, v in set_score_dist.items() if k.startswith(f"p1_{sets_to_win}"))
    p2_match_win = 1.0 - p1_match_win

    # Apply margin to set score distribution
    total_outcomes = len(set_score_dist)
    margin_per = request.base_margin / total_outcomes if total_outcomes else request.base_margin

    adj_dist = {k: round(v + margin_per if v > 1e-9 else v, 6) for k, v in set_score_dist.items()}
    adj_match = _shin_model.apply_shin_margin(
        true_probs={"p1_wins": p1_match_win, "p2_wins": p2_match_win},
        target_margin=request.base_margin,
    )

    return {
        "market": "set_betting",
        "competition_code": request.competition_code,
        "round_name": request.round_name,
        "p1_id": request.p1_id,
        "p2_id": request.p2_id,
        "sets_to_win": sets_to_win,
        "legs_per_set": legs_per_set,
        "set_probabilities": {
            "p1_wins_set": round(p1_wins_set, 6),
            "p2_wins_set": round(p2_wins_set, 6),
        },
        "match_winner": {
            "true_probabilities": {
                "p1_wins": round(p1_match_win, 6),
                "p2_wins": round(p2_match_win, 6),
            },
            "decimal_odds": {
                "p1_wins": round(1.0 / adj_match["p1_wins"], 4) if adj_match.get("p1_wins", 0) > 1e-9 else None,
                "p2_wins": round(1.0 / adj_match["p2_wins"], 4) if adj_match.get("p2_wins", 0) > 1e-9 else None,
            },
        },
        "correct_set_score": {
            "true_probabilities": {k: round(v, 6) for k, v in sorted(set_score_dist.items())},
            "decimal_odds": {
                k: round(1.0 / v, 4) if v > 1e-9 else None
                for k, v in sorted(set_score_dist.items())
            },
        },
        "applied_margin": round(request.base_margin, 5),
    }
