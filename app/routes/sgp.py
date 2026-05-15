"""
Same-Game Parlay (SGP) pricing API routes — Sprint 4 full implementation.

Endpoints
---------
POST /api/v1/darts/sgp/price          — price a SGP using Gaussian copula
POST /api/v1/darts/sgp/validate       — validate SGP selections (no pricing)
GET  /api/v1/darts/sgp/markets        — list supported SGP market types
GET  /api/v1/darts/sgp/combinations   — list popular SGP combination templates

The SGP engine uses:
  1. DartsSGPCorrelationEstimator to build the 15x15 correlation matrix
  2. DartsSGPBuilder (Gaussian copula MC) for joint probability
  3. DartsMarginEngine for overround allocation

Supported markets (15 total):
  match_win, total_legs_over, handicap, 180_over, checkout_over, exact_score,
  leg_winner_next, first_leg_winner, most_180s, highest_checkout,
  total_180s_band, player_checkout_over, sets_over, break_of_throw,
  nine_dart_finish
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
            "180_over, checkout_over, exact_score, leg_winner_next, "
            "first_leg_winner, most_180s, highest_checkout, total_180s_band, "
            "player_checkout_over, sets_over, break_of_throw, nine_dart_finish"
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
    # FIX-ODDS-FLOOR-001 (2026-05-14): clamp to 1.01 minimum before returning
    _raw_decimal = round(1.0 / bookmaker_prob, 2) if bookmaker_prob > 0 else None
    if _raw_decimal is not None and _raw_decimal < 1.01:
        logger.warning(
            "ODDS_FLOOR_TRIGGERED sport=darts context=sgp_price "
            "original_offered=%.4f clamped_to=1.01 — model confidence extreme",
            _raw_decimal,
        )
        _raw_decimal = 1.01
    decimal_odds = _raw_decimal

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
        "prediction_source": "model",
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
    # Original 7 markets
    "match_win":            "P(specified player wins the match)",
    "total_legs_over":      "P(total legs in match > O/U line)",
    "handicap":             "P(specified player covers the leg handicap)",
    "180_over":             "P(total 180s in match > O/U line)",
    "checkout_over":        "P(highest checkout in match > threshold)",
    "exact_score":          "P(match ends with exact leg score X-Y)",
    "leg_winner_next":      "P(specified player wins the next leg)",
    # Extended 8 markets (Sprint 4 expansion)
    "first_leg_winner":     "P(specified player wins the opening leg of the match)",
    "most_180s":            "P(specified player hits more 180s than their opponent)",
    "highest_checkout":     "P(specified player achieves the highest checkout of the match)",
    "total_180s_band":      "P(total 180s in match falls within a specific numeric band)",
    "player_checkout_over": "P(player-specific highest checkout exceeds threshold)",
    "sets_over":            "P(total sets in match > O/U line; sets-format matches only)",
    "break_of_throw":       "P(at least one break of throw occurs in the match)",
    "nine_dart_finish":     "P(a nine-dart finish is hit during the match)",
}

# Common SGP combination templates with correlation guidance.
# These are the most commercially popular parlays on darts.
_SGP_COMBINATION_TEMPLATES: list[dict] = [
    {
        "name": "First Leg Winner + Match Winner",
        "markets": ["first_leg_winner", "match_win"],
        "popularity": "very_high",
        "correlation_note": (
            "Moderate positive correlation (~0.35). Winning the first leg gives "
            "momentum and a break of throw opportunity, boosting match win probability. "
            "Do not treat as independent — the copula will correctly price the overlap."
        ),
        "example_odds_boost": "~0.80x versus naive multiplication",
    },
    {
        "name": "Handicap + Total Legs",
        "markets": ["handicap", "total_legs_over"],
        "popularity": "high",
        "correlation_note": (
            "Positive correlation (~0.32). A dominant handicap cover is more likely "
            "when the stronger player wins quickly (fewer legs), but a close match "
            "with many legs is also possible when P2 wins several. The copula "
            "captures this non-trivial dependency — naive multiplication will "
            "over-price this parlay."
        ),
        "example_odds_boost": "~0.85x versus naive multiplication",
    },
    {
        "name": "Match Winner + 180s Over",
        "markets": ["match_win", "180_over"],
        "popularity": "high",
        "correlation_note": (
            "Negative correlation (-0.40). The favourite winning decisively tends "
            "to produce fewer legs and thus fewer total 180s. An over on 180s implies "
            "a longer, more competitive match which slightly reduces the favourite's "
            "win probability. The copula discounts the naive joint price accordingly."
        ),
        "example_odds_boost": "~1.05x versus naive multiplication (slight boost due to negative corr)",
    },
    {
        "name": "First Leg Winner + Handicap",
        "markets": ["first_leg_winner", "handicap"],
        "popularity": "medium",
        "correlation_note": (
            "Moderate positive correlation (~0.20). Breaking throw in the first leg "
            "contributes to handicap coverage but does not guarantee it."
        ),
        "example_odds_boost": "~0.88x versus naive multiplication",
    },
    {
        "name": "180s Over + Total Legs Over",
        "markets": ["180_over", "total_legs_over"],
        "popularity": "medium",
        "correlation_note": (
            "Strong positive correlation (~0.38). More legs = more 180 opportunities. "
            "Both sides winning legs is a precondition for both markets — naive pricing "
            "significantly over-prices this parlay."
        ),
        "example_odds_boost": "~0.70x versus naive multiplication",
    },
    {
        "name": "Checkout Over + Match Winner",
        "markets": ["checkout_over", "match_win"],
        "popularity": "medium",
        "correlation_note": (
            "Positive correlation (~0.58). Higher-average players hit bigger checkouts. "
            "The match winner is likely the better player who also achieves the higher "
            "checkout — this parlay is often under-priced by naive multiplication."
        ),
        "example_odds_boost": "~0.80x versus naive multiplication",
    },
    {
        "name": "Break of Throw + Total Legs Over",
        "markets": ["break_of_throw", "total_legs_over"],
        "popularity": "medium",
        "correlation_note": (
            "Positive correlation (~0.42). Breaks of throw extend matches, making "
            "this one of the more correlated two-leg SGPs. The copula provides "
            "meaningful pricing correction versus the naive product."
        ),
        "example_odds_boost": "~0.75x versus naive multiplication",
    },
    {
        "name": "Nine Dart Finish + 180s Over",
        "markets": ["nine_dart_finish", "180_over"],
        "popularity": "low",
        "correlation_note": (
            "Weak positive correlation (~0.18). A nine-dart finish requires three "
            "180s in the same leg; a match with more 180s generally has more "
            "nine-dart opportunities. Correlation is real but small — the copula "
            "adjustment is modest (~5%)."
        ),
        "example_odds_boost": "~0.95x versus naive multiplication",
    },
]


# ---------------------------------------------------------------------------
# GET /sgp/combinations
# ---------------------------------------------------------------------------

@router.get(
    "/sgp/combinations",
    summary="List popular SGP combination templates with correlation guidance",
    response_model=dict,
    tags=["SGP"],
)
async def list_sgp_combinations() -> dict[str, Any]:
    """
    Return a catalogue of popular Same-Game Parlay combination templates.

    Each template includes:
    - The market pair / group
    - Popularity tier (very_high / high / medium / low)
    - A plain-language explanation of the correlation structure
    - An indicative pricing adjustment factor relative to naive multiplication

    These templates are for informational purposes only.  Actual pricing
    must be obtained via POST /sgp/price which runs the Gaussian copula
    Monte Carlo engine with the full 15x15 correlation matrix.

    Combination types specifically modelled:
    - first_leg_winner + match_win      (very common: moderate positive corr)
    - handicap + total_legs_over        (positive corr, naive over-prices)
    - match_win + 180_over              (negative corr, copula gives a slight boost)
    """
    return {
        "combinations": _SGP_COMBINATION_TEMPLATES,
        "n_combinations": len(_SGP_COMBINATION_TEMPLATES),
        "note": (
            "Correlation adjustments are directional guides only. "
            "Use POST /sgp/price for accurate joint probability pricing."
        ),
        "pricing_endpoint": "/api/v1/darts/sgp/price",
        "markets_endpoint": "/api/v1/darts/sgp/markets",
    }
