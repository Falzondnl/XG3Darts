"""
Same-Game Parlay (SGP) pricing API routes — Sprint 4 full implementation.

Endpoints
---------
POST /api/v1/darts/sgp/price    — price a SGP using Gaussian copula
POST /api/v1/darts/sgp/validate — validate SGP selections (no pricing)
GET  /api/v1/darts/sgp/markets  — list supported SGP market types

The SGP engine uses:
  1. DartsSGPCorrelationEstimator to build the correlation matrix
  2. DartsSGPBuilder (Gaussian copula MC) for joint probability
  3. DartsMarginEngine for overround allocation
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from engines.errors import DartsDataError, DartsEngineError
from margin.blending_engine import DartsMarginEngine
from sgp.copula_builder import DartsSGPBuilder, SGPSelection
from sgp.correlation_estimator import (
    MARKET_NAMES,
    DartsSGPCorrelationEstimator,
)

logger = structlog.get_logger(__name__)
router = APIRouter()

_sgp_builder = DartsSGPBuilder(n_mc=50_000)
_correlation_estimator = DartsSGPCorrelationEstimator()
_margin_engine = DartsMarginEngine()

# Market type → correlation matrix index
_MARKET_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(MARKET_NAMES)}

# Supported market types
SUPPORTED_SGP_MARKETS = frozenset(MARKET_NAMES)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class SGPLeg(BaseModel):
    """A single SGP selection leg."""
    market_type: str = Field(
        ...,
        description=(
            "Market type. One of: match_win, total_legs_over, handicap, "
            "180_over, checkout_over, exact_score, leg_winner_next"
        ),
    )
    outcome: str = Field(..., description="Human-readable outcome description")
    probability: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description="True (fair) probability of this selection",
    )

    @field_validator("market_type")
    @classmethod
    def validate_market_type(cls, v: str) -> str:
        if v not in SUPPORTED_SGP_MARKETS:
            raise ValueError(
                f"market_type {v!r} is not supported. "
                f"Supported: {sorted(SUPPORTED_SGP_MARKETS)}"
            )
        return v


class SGPPriceRequest(BaseModel):
    """SGP pricing request."""
    match_id: str = Field(..., description="Match identifier")
    competition_code: str = Field(
        default="PDC_WC",
        description="Competition format code for correlation lookup",
    )
    legs: list[SGPLeg] = Field(..., min_length=2, max_length=8)
    n_joint_outcomes: int = Field(
        default=0,
        ge=0,
        description="Number of historical joint outcomes for this competition",
    )
    regime: int = Field(default=1, ge=0, le=2)
    base_margin: float = Field(default=0.08, gt=0.0, le=0.15)
    ecosystem: str = Field(default="pdc_mens")
    market_liquidity: str = Field(default="medium")

    @field_validator("market_liquidity")
    @classmethod
    def validate_liquidity(cls, v: str) -> str:
        if v not in ("high", "medium", "low"):
            raise ValueError(f"market_liquidity must be high/medium/low, got {v!r}")
        return v


class SGPValidateRequest(BaseModel):
    """SGP validation request (checks selections without pricing)."""
    legs: list[SGPLeg]


# ---------------------------------------------------------------------------
# POST /sgp/price
# ---------------------------------------------------------------------------

@router.post(
    "/sgp/price",
    summary="Price a Same-Game Parlay",
    response_model=dict,
    tags=["SGP"],
)
async def price_sgp(request: SGPPriceRequest) -> dict[str, Any]:
    """
    Compute the joint probability for a Same-Game Parlay using a Gaussian
    copula with the darts-specific correlation matrix.

    The correlation matrix is estimated from historical joint outcomes for
    the competition and shrunk toward the global PDC prior when sample
    size is small.

    Steps:
    1. Estimate correlation matrix for the competition.
    2. Build SGPSelection objects with correlation indices.
    3. Run Gaussian copula Monte Carlo.
    4. Apply margin via DartsMarginEngine.
    5. Return true probability, bookmaker probability, decimal odds.
    """
    if len(request.legs) < 2:
        raise HTTPException(
            status_code=422,
            detail="SGP requires at least 2 legs.",
        )

    # Step 1: Estimate correlation matrix
    try:
        corr_matrix = _correlation_estimator.estimate_for_competition(
            competition_code=request.competition_code,
            n_samples=request.n_joint_outcomes,
        )
    except DartsEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Step 2: Build SGP selections with correlation indices
    selections: list[SGPSelection] = []
    for leg in request.legs:
        corr_idx = _MARKET_TO_INDEX.get(leg.market_type)
        selections.append(SGPSelection(
            market_type=leg.market_type,
            outcome=leg.outcome,
            probability=leg.probability,
            correlation_index=corr_idx,
        ))

    # Step 3: Gaussian copula MC pricing
    try:
        true_prob = _sgp_builder.price_parlay(
            selections=selections,
            correlation_matrix=corr_matrix,
        )
    except DartsEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Step 4: Apply margin
    try:
        margin = _margin_engine.compute_margin(
            base_margin=request.base_margin,
            regime=request.regime,
            starter_confidence=1.0,
            source_confidence=1.0,
            model_agreement=1.0,
            market_liquidity=request.market_liquidity,
            ecosystem=request.ecosystem,
        )
    except DartsEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if true_prob <= 0.0:
        raise HTTPException(
            status_code=422,
            detail="SGP joint probability is zero — selections may be contradictory.",
        )

    bookmaker_prob = min(1.0, true_prob * (1.0 + margin))
    decimal_odds = round(1.0 / bookmaker_prob, 2) if bookmaker_prob > 0 else None

    # Naive (independent) parlay for comparison
    naive_prob = 1.0
    for leg in request.legs:
        naive_prob *= leg.probability

    logger.info(
        "sgp_priced",
        match_id=request.match_id,
        competition_code=request.competition_code,
        n_legs=len(request.legs),
        true_prob=round(true_prob, 6),
        naive_prob=round(naive_prob, 6),
        margin=round(margin, 4),
        decimal_odds=decimal_odds,
    )

    return {
        "match_id": request.match_id,
        "competition_code": request.competition_code,
        "n_legs": len(request.legs),
        "legs": [
            {
                "market_type": leg.market_type,
                "outcome": leg.outcome,
                "probability": round(leg.probability, 6),
            }
            for leg in request.legs
        ],
        "true_probability": round(true_prob, 6),
        "naive_probability": round(naive_prob, 6),
        "correlation_adjustment": round(true_prob / naive_prob, 4) if naive_prob > 0 else None,
        "applied_margin": round(margin, 5),
        "bookmaker_probability": round(bookmaker_prob, 6),
        "decimal_odds": decimal_odds,
        "regime": request.regime,
        "n_joint_outcomes_used": request.n_joint_outcomes,
    }


# ---------------------------------------------------------------------------
# POST /sgp/validate
# ---------------------------------------------------------------------------

@router.post(
    "/sgp/validate",
    summary="Validate SGP selections",
    response_model=dict,
    tags=["SGP"],
)
async def validate_sgp(request: SGPValidateRequest) -> dict[str, Any]:
    """
    Validate SGP selections without pricing.

    Checks:
    - Each market_type is supported.
    - Probabilities are in (0, 1).
    - No duplicate market types (same market cannot be selected twice).

    Returns validation status and any issues found.
    """
    issues = []
    market_types_seen: set[str] = set()

    for i, leg in enumerate(request.legs):
        if leg.market_type in market_types_seen:
            issues.append(
                f"Leg {i}: duplicate market_type '{leg.market_type}'. "
                "A market can only appear once per SGP."
            )
        market_types_seen.add(leg.market_type)

        if not (0.0 < leg.probability < 1.0):
            issues.append(
                f"Leg {i}: probability {leg.probability} is not in (0, 1)."
            )

    valid = len(issues) == 0
    return {
        "valid": valid,
        "n_legs": len(request.legs),
        "issues": issues,
        "supported_markets": sorted(SUPPORTED_SGP_MARKETS),
    }


# ---------------------------------------------------------------------------
# GET /sgp/markets
# ---------------------------------------------------------------------------

@router.get(
    "/sgp/markets",
    summary="List supported SGP market types",
    response_model=dict,
    tags=["SGP"],
)
async def list_sgp_markets() -> dict[str, Any]:
    """
    Return the list of market types that can be included in a SGP,
    along with their correlation matrix index.
    """
    return {
        "supported_markets": [
            {
                "market_type": name,
                "correlation_index": idx,
                "description": _MARKET_DESCRIPTIONS.get(name, ""),
            }
            for name, idx in _MARKET_TO_INDEX.items()
        ],
        "max_legs": 8,
        "min_legs": 2,
        "pricing_endpoint": "/api/v1/darts/sgp/price",
    }


_MARKET_DESCRIPTIONS: dict[str, str] = {
    "match_win": "P(specified player wins the match)",
    "total_legs_over": "P(total legs in match > O/U line)",
    "handicap": "P(specified player covers the leg handicap)",
    "180_over": "P(total 180s in match > O/U line)",
    "checkout_over": "P(highest checkout in match > threshold)",
    "exact_score": "P(match ends with exact leg score X-Y)",
    "leg_winner_next": "P(specified player wins the next leg)",
}
