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
POST /api/v1/darts/prematch/180s             — match 180 O/U
POST /api/v1/darts/prematch/checkout         — highest checkout O/U
POST /api/v1/darts/prematch/ecosystem-price  — ecosystem-aware match pricing
GET  /api/v1/darts/prematch/markets/{match_id} — all markets for a fixture
"""
from __future__ import annotations

from typing import Any, Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from competition.format_registry import get_format, DartsFormatError
from engines.errors import DartsDataError, DartsEngineError
from engines.leg_layer.hold_break_model import HoldBreakModel
from engines.match_layer.match_combinatorics import MatchCombinatorialEngine
from margin.blending_engine import DartsMarginEngine
from margin.shin_margin import ShinMarginModel
from props.prop_180 import Prop180Model

logger = structlog.get_logger(__name__)
router = APIRouter()

_hold_break_model = HoldBreakModel()
_match_engine = MatchCombinatorialEngine()
_margin_engine = DartsMarginEngine()
_shin_model = ShinMarginModel()
_prop_180_model = Prop180Model()


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

    true_probs: dict[str, float] = {
        "p1_win": match_result.p1_win,
        "p2_win": match_result.p2_win,
    }
    if match_result.draw > 0:
        true_probs["draw"] = match_result.draw

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
            "match_winner",
            "correct_score",
            "handicap",
            "total_legs",
            "180s",
            "highest_checkout",
        ],
        "status": "markets_available",
        "pricing_endpoint": f"/api/v1/darts/prematch/price",
        "note": (
            "POST to /prematch/price with competition_code, round_name, "
            "player IDs and 3DA values to receive computed prices."
        ),
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

    true_probs: dict[str, float] = {
        "p1_win": blended_p1,
        "p2_win": blended_p2,
    }
    if match_result.draw > 0:
        true_probs["draw"] = match_result.draw

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
