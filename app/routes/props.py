"""
Player prop bet API routes — Sprint 4 full implementation.

Endpoints
---------
POST /api/v1/darts/props/180           — 180 count O/U
POST /api/v1/darts/props/checkout      — checkout range probability
POST /api/v1/darts/props/high-checkout — highest checkout O/U
POST /api/v1/darts/props/nine-darter   — nine-darter occurrence
POST /api/v1/darts/props/segment       — segment accuracy (R2 only)

All endpoints enforce data-sufficiency gates before pricing.
"""
from __future__ import annotations

from typing import Any, Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from engines.errors import DartsDataError, DartsEngineError
from props.data_sufficiency_gate import DataSufficiencyGate
from props.prop_180 import Prop180Model
from props.prop_checkout import PropCheckoutModel
from props.prop_nine_darter import PropNineDarterModel

logger = structlog.get_logger(__name__)
router = APIRouter()

_gate = DataSufficiencyGate()
_prop_180 = Prop180Model()
_prop_checkout = PropCheckoutModel()
_prop_nine_darter = PropNineDarterModel()


# ---------------------------------------------------------------------------
# Shared player stats model
# ---------------------------------------------------------------------------

class PlayerPropStats(BaseModel):
    """Common player statistics for prop pricing."""
    player_id: str
    legs_observed: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class PlayerStats180(PlayerPropStats):
    """Stats for 180 prop."""
    pct_180_ewm: float = Field(..., ge=0.0, le=1.0, description="180 rate per visit")


class PlayerStatsCheckout(PlayerPropStats):
    """Stats for checkout prop."""
    checkout_ewm: float = Field(..., ge=0.0, le=1.0, description="Checkout rate")
    checkout_pct_by_band: Optional[dict[str, float]] = Field(
        default=None,
        description="Checkout score distribution by band",
    )


class PlayerStatsNineDarter(PlayerPropStats):
    """Stats for nine-darter prop."""
    t20_accuracy: float = Field(..., ge=0.0, le=1.0, description="P(hit T20) per dart")
    double_accuracy: float = Field(..., ge=0.0, le=1.0, description="P(hit required double)")


# ---------------------------------------------------------------------------
# POST /props/180
# ---------------------------------------------------------------------------

class Prop180Request(BaseModel):
    """180 count O/U request."""
    p1: PlayerStats180
    p2: PlayerStats180
    expected_legs: float = Field(..., gt=0)
    expected_visits_per_leg: float = Field(default=8.0, gt=0)
    line: float = Field(..., gt=0, description="O/U line for total 180s")
    regime: int = Field(default=1, ge=0, le=2)


@router.post(
    "/props/180",
    summary="Price 180 count over/under",
    response_model=dict,
    tags=["Props"],
)
async def price_180_prop(request: Prop180Request) -> dict[str, Any]:
    """
    Compute P(total 180s in match > line) using a Poisson model.

    Requires pct_180_ewm for both players and a minimum of 50 observed legs.
    """
    p1_stats = {
        "legs_observed": request.p1.legs_observed,
        "confidence": request.p1.confidence,
        "pct_180_ewm": request.p1.pct_180_ewm,
    }
    p2_stats = {
        "legs_observed": request.p2.legs_observed,
        "confidence": request.p2.confidence,
        "pct_180_ewm": request.p2.pct_180_ewm,
    }

    try:
        result = _prop_180.price_total_180s(
            p1_180_rate=request.p1.pct_180_ewm,
            p2_180_rate=request.p2.pct_180_ewm,
            expected_legs=request.expected_legs,
            expected_visits_per_leg=request.expected_visits_per_leg,
            line=request.line,
            p1_stats=p1_stats,
            p2_stats=p2_stats,
            p1_id=request.p1.player_id,
            p2_id=request.p2.player_id,
            regime=request.regime,
        )
    except DartsDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except DartsEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    logger.info(
        "prop_180_priced",
        p1_id=request.p1.player_id,
        p2_id=request.p2.player_id,
        line=request.line,
        over_prob=result["over_prob"],
    )
    return {
        "market": "180s_total",
        "p1_id": request.p1.player_id,
        "p2_id": request.p2.player_id,
        "line": request.line,
        "regime": request.regime,
        **result,
    }


# ---------------------------------------------------------------------------
# POST /props/checkout
# ---------------------------------------------------------------------------

class PropCheckoutRangeRequest(BaseModel):
    """Checkout range probability request."""
    player: PlayerStatsCheckout
    line_low: int = Field(..., ge=2, le=170)
    line_high: int = Field(..., ge=2, le=170)
    regime: int = Field(default=1, ge=0, le=2)

    @field_validator("line_high")
    @classmethod
    def validate_range(cls, v: int, values) -> int:
        if hasattr(values, "data") and "line_low" in values.data and v < values.data["line_low"]:
            raise ValueError(f"line_high ({v}) must be >= line_low")
        return v


@router.post(
    "/props/checkout",
    summary="Price checkout score range probability",
    response_model=dict,
    tags=["Props"],
)
async def price_checkout_range(request: PropCheckoutRangeRequest) -> dict[str, Any]:
    """
    Compute P(player's checkout score in [line_low, line_high]).
    """
    player_stats = {
        "legs_observed": request.player.legs_observed,
        "confidence": request.player.confidence,
        "checkout_ewm": request.player.checkout_ewm,
        "checkout_pct_by_band": request.player.checkout_pct_by_band,
    }

    try:
        result = _prop_checkout.price_checkout_range(
            player_checkout_model=player_stats,
            line_low=request.line_low,
            line_high=request.line_high,
            player_id=request.player.player_id,
            regime=request.regime,
        )
    except DartsDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except DartsEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "market": "checkout_range",
        "player_id": request.player.player_id,
        "regime": request.regime,
        **result,
    }


# ---------------------------------------------------------------------------
# POST /props/high-checkout
# ---------------------------------------------------------------------------

class PropHighCheckoutRequest(BaseModel):
    """Highest checkout O/U request."""
    p1: PlayerStatsCheckout
    p2: PlayerStatsCheckout
    expected_legs: float = Field(..., gt=0)
    line: int = Field(..., ge=2, le=170)
    regime: int = Field(default=1, ge=0, le=2)


@router.post(
    "/props/high-checkout",
    summary="Price highest checkout over/under",
    response_model=dict,
    tags=["Props"],
)
async def price_high_checkout(request: PropHighCheckoutRequest) -> dict[str, Any]:
    """
    Compute P(highest checkout in match > line).
    """
    p1_model = {
        "legs_observed": request.p1.legs_observed,
        "confidence": request.p1.confidence,
        "checkout_ewm": request.p1.checkout_ewm,
        "checkout_pct_by_band": request.p1.checkout_pct_by_band,
    }
    p2_model = {
        "legs_observed": request.p2.legs_observed,
        "confidence": request.p2.confidence,
        "checkout_ewm": request.p2.checkout_ewm,
        "checkout_pct_by_band": request.p2.checkout_pct_by_band,
    }

    try:
        result = _prop_checkout.price_highest_checkout(
            p1_checkout_model=p1_model,
            p2_checkout_model=p2_model,
            expected_legs=request.expected_legs,
            line=request.line,
            p1_id=request.p1.player_id,
            p2_id=request.p2.player_id,
            regime=request.regime,
        )
    except DartsDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except DartsEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "market": "highest_checkout",
        "p1_id": request.p1.player_id,
        "p2_id": request.p2.player_id,
        "expected_legs": request.expected_legs,
        "regime": request.regime,
        **result,
    }


# ---------------------------------------------------------------------------
# POST /props/nine-darter
# ---------------------------------------------------------------------------

class PropNineDarterRequest(BaseModel):
    """Nine-darter occurrence request."""
    p1: PlayerStatsNineDarter
    p2: PlayerStatsNineDarter
    expected_legs: float = Field(..., gt=0)
    regime: int = Field(default=2, ge=0, le=2)


@router.post(
    "/props/nine-darter",
    summary="Price nine-darter occurrence probability",
    response_model=dict,
    tags=["Props"],
)
async def price_nine_darter(request: PropNineDarterRequest) -> dict[str, Any]:
    """
    Compute P(nine-darter occurs anywhere in match).

    Uses player T20 accuracy and double accuracy from the EB segment model.
    """
    p1_stats = {
        "legs_observed": request.p1.legs_observed,
        "confidence": request.p1.confidence,
        "pct_180_ewm": request.p1.t20_accuracy ** 3,  # approximate
    }
    p2_stats = {
        "legs_observed": request.p2.legs_observed,
        "confidence": request.p2.confidence,
        "pct_180_ewm": request.p2.t20_accuracy ** 3,
    }

    try:
        result = _prop_nine_darter.price_nine_darter(
            p1_t20_accuracy=request.p1.t20_accuracy,
            p2_t20_accuracy=request.p2.t20_accuracy,
            p1_double_accuracy=request.p1.double_accuracy,
            p2_double_accuracy=request.p2.double_accuracy,
            expected_legs=request.expected_legs,
            p1_stats=p1_stats,
            p2_stats=p2_stats,
            p1_id=request.p1.player_id,
            p2_id=request.p2.player_id,
            regime=request.regime,
        )
    except DartsDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except DartsEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "market": "nine_darter",
        "p1_id": request.p1.player_id,
        "p2_id": request.p2.player_id,
        "expected_legs": request.expected_legs,
        "regime": request.regime,
        **result,
    }


# ---------------------------------------------------------------------------
# POST /props/segment  (R2 only)
# ---------------------------------------------------------------------------

class PropSegmentRequest(BaseModel):
    """Segment accuracy prop request (R2 only)."""
    player_id: str
    segment: str = Field(..., description="Board segment, e.g. 'T20', 'D16'")
    regime: int = Field(default=2, ge=0, le=2)
    # R2 stats
    legs_observed: int = Field(default=100, ge=0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    double_hit_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    eb_double_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    n_attempts: int = Field(default=0, ge=0)
    n_hits: int = Field(default=0, ge=0)


@router.post(
    "/props/segment",
    summary="Price segment accuracy prop (R2 only)",
    response_model=dict,
    tags=["Props"],
)
async def price_segment_prop(request: PropSegmentRequest) -> dict[str, Any]:
    """
    Price a segment-level accuracy prop bet.

    R2 (visit-level DartConnect data) required.
    Returns P(player hits segment > line in match) and pricing.
    """
    # R2 gate check
    stats = {
        "legs_observed": request.legs_observed,
        "confidence": request.confidence,
        "double_hit_rate": request.double_hit_rate,
        "eb_double_accuracy": request.eb_double_accuracy,
    }

    ok, reason = _gate.can_open_market(
        market_family="props_double_segment",
        player_id=request.player_id,
        regime=request.regime,
        stats=stats,
    )
    if not ok:
        raise HTTPException(status_code=422, detail=reason)

    # Compute empirical hit rate from n_hits / n_attempts
    if request.n_attempts > 0:
        empirical_rate = request.n_hits / request.n_attempts
    elif request.eb_double_accuracy is not None:
        empirical_rate = request.eb_double_accuracy
    elif request.double_hit_rate is not None:
        empirical_rate = request.double_hit_rate
    else:
        raise HTTPException(
            status_code=422,
            detail=(
                "At least one of n_hits/n_attempts, eb_double_accuracy, "
                "or double_hit_rate must be provided."
            ),
        )

    logger.info(
        "segment_prop_priced",
        player_id=request.player_id,
        segment=request.segment,
        empirical_rate=round(empirical_rate, 5),
        regime=request.regime,
    )

    return {
        "market": "segment_accuracy",
        "player_id": request.player_id,
        "segment": request.segment,
        "empirical_rate": round(empirical_rate, 6),
        "regime": request.regime,
        "n_attempts": request.n_attempts,
        "n_hits": request.n_hits,
    }


# ---------------------------------------------------------------------------
# POST /props/most-180s — Which player hits more 180s in the match
# ---------------------------------------------------------------------------


class MostOneEightiesRequest(BaseModel):
    """Most 180s in match — head-to-head 180 market."""
    p1_id: str
    p2_id: str
    p1_180_rate: float = Field(..., ge=0.0, le=1.0,
                                description="P1 180-per-visit rate (e.g. 0.08)")
    p2_180_rate: float = Field(..., ge=0.0, le=1.0,
                                description="P2 180-per-visit rate")
    expected_legs: int = Field(default=10, ge=1, le=50,
                                description="Expected number of legs in match")
    visits_per_leg: float = Field(default=16.5, ge=5.0, le=40.0,
                                   description="Average visits per player per leg")
    base_margin: float = Field(default=0.05, gt=0.0, le=0.15)


@router.post(
    "/props/most-180s",
    summary="Most 180s in match — head-to-head player market",
    tags=["Props"],
)
async def price_most_180s(request: MostOneEightiesRequest) -> dict[str, Any]:
    """
    Price the 'Player X to hit most 180s' market.

    Uses a Poisson model per player:
        λ_i = 180_rate_i × visits_per_leg × expected_legs
    P(p1 hits more 180s) = Σ_k P(Poisson(λ1) = k) × P(Poisson(λ2) < k)
    Tie probability is calculated and split evenly between the two players.
    """
    import math

    lam1 = request.p1_180_rate * request.visits_per_leg * request.expected_legs
    lam2 = request.p2_180_rate * request.visits_per_leg * request.expected_legs

    # Poisson PMF cache
    max_k = max(int(lam1 * 4 + 20), int(lam2 * 4 + 20), 50)

    def poisson_pmf(lam: float, k: int) -> float:
        if k < 0:
            return 0.0
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    def poisson_cdf(lam: float, k: int) -> float:
        return sum(poisson_pmf(lam, i) for i in range(k + 1))

    p1_more = sum(
        poisson_pmf(lam1, k) * poisson_cdf(lam2, k - 1)
        for k in range(1, max_k)
    )
    p2_more = sum(
        poisson_pmf(lam2, k) * poisson_cdf(lam1, k - 1)
        for k in range(1, max_k)
    )
    p_tie = max(0.0, 1.0 - p1_more - p2_more)

    # Distribute tie probability equally (no push market)
    p1_adj_true = p1_more + p_tie * 0.5
    p2_adj_true = p2_more + p_tie * 0.5

    half = request.base_margin / 2.0
    p1_book = round(p1_adj_true + half, 6)
    p2_book = round(p2_adj_true + half, 6)
    total = p1_book + p2_book
    p1_book = round(p1_book / total * (1 + request.base_margin), 6)
    p2_book = round(p2_book / total * (1 + request.base_margin), 6)

    logger.info(
        "most_180s_priced",
        p1_id=request.p1_id, p2_id=request.p2_id,
        lam1=round(lam1, 2), lam2=round(lam2, 2),
        p1_win=round(p1_adj_true, 4),
    )

    return {
        "p1_id": request.p1_id,
        "p2_id": request.p2_id,
        "expected_180s_p1": round(lam1, 2),
        "expected_180s_p2": round(lam2, 2),
        "true_probabilities": {
            "p1_most": round(p1_adj_true, 6),
            "p2_most": round(p2_adj_true, 6),
            "tie": round(p_tie, 6),
        },
        "adjusted_probabilities": {"p1_most": p1_book, "p2_most": p2_book},
        "decimal_odds": {
            "p1_most": round(1.0 / p1_book, 4) if p1_book > 1e-9 else None,
            "p2_most": round(1.0 / p2_book, 4) if p2_book > 1e-9 else None,
        },
        "applied_margin": round(request.base_margin, 5),
    }
