"""
Derivative markets for Darts — combinatorial probability engine.

All prices are computed from the supplied match-winner probabilities via
exact binomial combinatorics.  No hardcoded odds, no hardcoded probabilities.

Endpoints
---------
POST /api/v1/darts/derivatives/generate
    Generate the full derivative market set for a match.

GET  /api/v1/darts/derivatives/{match_id}
    Return cached (60 s TTL) derivative markets for a match.
    Requires a prior POST /generate call or an active pre-match pricing cycle.

GET  /api/v1/darts/derivatives/health
    Liveness check for this router.

Supported formats
-----------------
  bo3  — first to 2 legs (PDC Players Championship, Development Tour)
  bo5  — first to 3 legs (PDC European Tour, UK Open early rounds)
  bo7  — first to 4 legs (PDC events mid-rounds)
  bo9  — first to 5 legs (PDC events, World Championship early rounds)
  bo13 — first to 7 legs (PDC World Championship later rounds / final)

Market families
---------------
  1. Match Winner           — P1 / P2 with 4.5 % vig
  2. Handicap Legs          — ±0.5 through ±(N-1).5 lines
  3. Total Legs O/U         — multiple lines based on format
  4. Correct Score          — all possible {w-l, l-w} outcomes
  5. First Leg Winner       — P1 / P2 with 20 % regression toward 0.5
  6. Match to Final Leg     — P(decider required)
"""
from __future__ import annotations

import math
import time
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# In-process cache  {match_id: (unix_timestamp, markets_payload)}
# ---------------------------------------------------------------------------

_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL_SECONDS = 60.0


def _cache_get(match_id: str) -> Optional[Dict[str, Any]]:
    entry = _CACHE.get(match_id)
    if entry is None:
        return None
    ts, payload = entry
    if time.monotonic() - ts > _CACHE_TTL_SECONDS:
        del _CACHE[match_id]
        return None
    return payload


def _cache_set(match_id: str, payload: Dict[str, Any]) -> None:
    _CACHE[match_id] = (time.monotonic(), payload)


# ---------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------

#  format_key -> (legs_to_win, max_legs)
_FORMAT_MAP: Dict[str, Tuple[int, int]] = {
    "bo3": (2, 3),
    "bo5": (3, 5),
    "bo7": (4, 7),
    "bo9": (5, 9),
    "bo13": (7, 13),
}


def _resolve_format(fmt: str) -> Tuple[int, int]:
    """Return (legs_to_win, max_legs) or raise ValueError."""
    key = fmt.lower().strip()
    if key not in _FORMAT_MAP:
        raise ValueError(
            f"Unknown format '{fmt}'. Supported: {sorted(_FORMAT_MAP.keys())}"
        )
    return _FORMAT_MAP[key]


# ---------------------------------------------------------------------------
# Core combinatorics
# ---------------------------------------------------------------------------


def _binom(n: int, k: int) -> float:
    """Binomial coefficient C(n, k) as float."""
    if k < 0 or k > n:
        return 0.0
    return float(math.comb(n, k))


def _p_win_score(winner_legs: int, loser_legs: int, p_win_leg: float) -> float:
    """
    P(first-to-W player wins exactly W-L) in a race-to-W format.

    The winner wins their W-th leg on the final leg played, meaning
    the last leg is always won by the winner.  Out of the first
    (W + L - 1) legs the winner takes exactly (W - 1).

    P = C(W+L-1, W-1) * p^W * (1-p)^L
    """
    if loser_legs < 0:
        return 0.0
    p = float(p_win_leg)
    q = 1.0 - p
    coeff = _binom(winner_legs + loser_legs - 1, winner_legs - 1)
    return coeff * (p ** winner_legs) * (q ** loser_legs)


def _all_score_probs(legs_to_win: int, p1: float) -> Dict[Tuple[int, int], float]:
    """
    Return a mapping {(p1_legs, p2_legs): probability} for every
    possible final score in a first-to-N race.

    Keys cover both P1-wins (p1_legs == N) and P2-wins (p2_legs == N) outcomes.
    Probabilities sum to 1.0 (within float precision).
    """
    result: Dict[Tuple[int, int], float] = {}
    w = legs_to_win
    for loser in range(0, w):
        # P1 wins w - loser
        p1_wins = _p_win_score(w, loser, p1)
        result[(w, loser)] = p1_wins
        # P2 wins w - loser
        p2_wins = _p_win_score(w, loser, 1.0 - p1)
        result[(loser, w)] = p2_wins
    return result


def _prob_to_odds(prob: float, vig: float = 0.045) -> float:
    """
    Convert fair probability to decimal odds with a symmetric vig.

    vig = overround as a fraction (e.g. 0.045 for 4.5 %).
    The method scales the fair probability up by (1 + vig) then
    inverts, which is equivalent to the standard overround method.
    """
    if prob <= 0.0:
        return 1000.0
    biased_prob = prob * (1.0 + vig)
    biased_prob = min(biased_prob, 0.9999)
    return round(1.0 / biased_prob, 4)


def _market(
    name: str,
    selections: List[Dict[str, Any]],
    market_type: str = "two_way",
) -> Dict[str, Any]:
    return {
        "market_name": name,
        "market_type": market_type,
        "selections": selections,
    }


def _selection(label: str, price: float, probability: float) -> Dict[str, Any]:
    return {
        "label": label,
        "price": price,
        "fair_probability": round(probability, 6),
    }


# ---------------------------------------------------------------------------
# Market builders
# ---------------------------------------------------------------------------


def _build_match_winner(p1: float, p2: float, player1: str, player2: str) -> Dict[str, Any]:
    o1 = _prob_to_odds(p1)
    o2 = _prob_to_odds(p2)
    return _market(
        "Match Winner",
        [
            _selection(player1, o1, p1),
            _selection(player2, o2, p2),
        ],
    )


def _build_handicap_legs(
    legs_to_win: int,
    max_legs: int,
    p1: float,
    player1: str,
    player2: str,
    score_probs: Dict[Tuple[int, int], float],
) -> List[Dict[str, Any]]:
    """
    Build handicap-legs markets.

    For each half-integer handicap line h (e.g. -1.5, -0.5, +0.5 …):
      P(P1 covers -h) = P(P1 wins and total_p1_legs - total_p2_legs > h)
      P(P2 covers +h) = 1 - P(P1 covers -h)

    We enumerate from -(legs_to_win - 0.5) to +(legs_to_win - 0.5)
    in steps of 1.0 (i.e. every half-integer between 0.5 and W-0.5).
    """
    markets: List[Dict[str, Any]] = []

    # Half-integer lines: 0.5, 1.5, … (legs_to_win - 0.5)
    for offset_tenths in range(5, legs_to_win * 10, 10):  # 0.5, 1.5, ...
        h = offset_tenths / 10.0  # e.g. 0.5, 1.5

        # P(player1 wins by strictly more than h legs)
        p1_covers_minus_h = sum(
            prob
            for (s1, s2), prob in score_probs.items()
            if s1 > s2 and (s1 - s2) > h
        )
        # P(player2 wins by strictly more than h legs)
        p2_covers_minus_h = sum(
            prob
            for (s1, s2), prob in score_probs.items()
            if s2 > s1 and (s2 - s1) > h
        )

        # Market A: Player1 -h  vs  Player2 +h
        o_p1_minus = _prob_to_odds(p1_covers_minus_h)
        o_p2_plus = _prob_to_odds(1.0 - p1_covers_minus_h)
        label_minus = f"-{h:.1f}"
        label_plus = f"+{h:.1f}"

        markets.append(
            _market(
                f"Handicap Legs {player1} {label_minus}",
                [
                    _selection(f"{player1} {label_minus}", o_p1_minus, p1_covers_minus_h),
                    _selection(f"{player2} {label_plus}", o_p2_plus, 1.0 - p1_covers_minus_h),
                ],
            )
        )

        # Market B: Player2 -h  vs  Player1 +h
        o_p2_minus = _prob_to_odds(p2_covers_minus_h)
        o_p1_plus = _prob_to_odds(1.0 - p2_covers_minus_h)

        markets.append(
            _market(
                f"Handicap Legs {player2} {label_minus}",
                [
                    _selection(f"{player2} {label_minus}", o_p2_minus, p2_covers_minus_h),
                    _selection(f"{player1} {label_plus}", o_p1_plus, 1.0 - p2_covers_minus_h),
                ],
            )
        )

    return markets


def _build_total_legs(
    legs_to_win: int,
    max_legs: int,
    score_probs: Dict[Tuple[int, int], float],
) -> List[Dict[str, Any]]:
    """
    Build total-legs O/U markets.

    Lines are half-integers from (legs_to_win + 0.5) up to (max_legs - 0.5).
    E.g. Bo3: 2.5 only.  Bo5: 3.5, 4.5.  Bo7: 4.5, 5.5, 6.5.
    """
    markets: List[Dict[str, Any]] = []
    # min possible legs = legs_to_win (one side wins all)
    # max possible legs = max_legs (decider)

    for total in range(legs_to_win, max_legs):
        # Line = total + 0.5
        line = total + 0.5
        # P(total legs played > line) = P(total played >= total + 1)
        p_over = sum(
            prob
            for (s1, s2), prob in score_probs.items()
            if (s1 + s2) > line
        )
        p_under = 1.0 - p_over
        o_over = _prob_to_odds(p_over)
        o_under = _prob_to_odds(p_under)

        markets.append(
            _market(
                f"Total Legs O/U {line:.1f}",
                [
                    _selection(f"Over {line:.1f}", o_over, p_over),
                    _selection(f"Under {line:.1f}", o_under, p_under),
                ],
            )
        )

    return markets


def _build_correct_score(
    legs_to_win: int,
    score_probs: Dict[Tuple[int, int], float],
    player1: str,
    player2: str,
) -> Dict[str, Any]:
    """Build correct score market with all possible outcomes."""
    selections: List[Dict[str, Any]] = []

    # Sort for deterministic ordering: P1-wins descending by score, then P2-wins
    p1_outcomes = sorted(
        [(s1, s2, prob) for (s1, s2), prob in score_probs.items() if s1 > s2],
        key=lambda x: x[1],  # ascending loser legs
    )
    p2_outcomes = sorted(
        [(s1, s2, prob) for (s1, s2), prob in score_probs.items() if s2 > s1],
        key=lambda x: x[0],  # ascending loser legs
    )

    for s1, s2, prob in p1_outcomes:
        label = f"{player1} {s1}-{s2}"
        selections.append(_selection(label, _prob_to_odds(prob), prob))

    for s1, s2, prob in p2_outcomes:
        label = f"{player2} {s2}-{s1}"
        selections.append(_selection(label, _prob_to_odds(prob), prob))

    return _market(
        "Correct Score",
        selections,
        market_type="multi_way",
    )


def _build_first_leg(p1: float, player1: str, player2: str) -> Dict[str, Any]:
    """
    First Leg Winner.

    Regress the match probability 20 % toward 0.5 to reflect the
    higher per-leg variance compared to a full match.
    """
    p1_leg = 0.8 * p1 + 0.2 * 0.5
    p2_leg = 1.0 - p1_leg
    o1 = _prob_to_odds(p1_leg)
    o2 = _prob_to_odds(p2_leg)
    return _market(
        "First Leg Winner",
        [
            _selection(player1, o1, p1_leg),
            _selection(player2, o2, p2_leg),
        ],
    )


def _build_final_leg(
    legs_to_win: int,
    max_legs: int,
    score_probs: Dict[Tuple[int, int], float],
) -> Dict[str, Any]:
    """Match to go to final leg (decider)."""
    p_decider = sum(
        prob
        for (s1, s2), prob in score_probs.items()
        if (s1 + s2) == max_legs
    )
    p_no_decider = 1.0 - p_decider
    o_yes = _prob_to_odds(p_decider)
    o_no = _prob_to_odds(p_no_decider)
    return _market(
        f"Match Goes to Final Leg (Leg {max_legs})",
        [
            _selection("Yes", o_yes, p_decider),
            _selection("No", o_no, p_no_decider),
        ],
    )


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------


def generate_darts_derivatives(
    match_id: str,
    p_player1: float,
    p_player2: float,
    format_key: str,
    player1: str = "Player 1",
    player2: str = "Player 2",
) -> Dict[str, Any]:
    """
    Generate all derivative markets for a darts match.

    Parameters
    ----------
    match_id : str
    p_player1, p_player2 : float
        Match-winner probabilities (must sum to 1.0, after normalisation).
    format_key : str
        One of: bo3, bo5, bo7, bo9, bo13.
    player1, player2 : str
        Display names for selections.

    Returns
    -------
    dict  — full derivative payload
    """
    legs_to_win, max_legs = _resolve_format(format_key)

    # Normalise supplied probabilities
    total = p_player1 + p_player2
    if total <= 0.0:
        raise ValueError("p_player1 and p_player2 must be positive and non-zero.")
    p1 = p_player1 / total
    p2 = 1.0 - p1

    # Score probability map
    score_probs = _all_score_probs(legs_to_win, p1)

    markets: List[Dict[str, Any]] = []

    # 1. Match Winner
    markets.append(_build_match_winner(p1, p2, player1, player2))

    # 2. Handicap Legs
    markets.extend(_build_handicap_legs(legs_to_win, max_legs, p1, player1, player2, score_probs))

    # 3. Total Legs O/U
    markets.extend(_build_total_legs(legs_to_win, max_legs, score_probs))

    # 4. Correct Score
    markets.append(_build_correct_score(legs_to_win, score_probs, player1, player2))

    # 5. First Leg Winner
    markets.append(_build_first_leg(p1, player1, player2))

    # 6. Match to Final Leg
    markets.append(_build_final_leg(legs_to_win, max_legs, score_probs))

    payload: Dict[str, Any] = {
        "match_id": match_id,
        "format": format_key.lower(),
        "legs_to_win": legs_to_win,
        "max_legs": max_legs,
        "input_probabilities": {
            "p_player1": round(p1, 6),
            "p_player2": round(p2, 6),
        },
        "markets": markets,
        "total_markets": len(markets),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    return payload


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class DerivativeGenerateRequest(BaseModel):
    """Request body for POST /derivatives/generate."""

    match_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Optic Odds or internal match identifier.",
        examples=["optic_match_12345"],
    )
    p_player1: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description="Match-winner probability for Player 1 (before normalisation).",
        examples=[0.58],
    )
    p_player2: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description="Match-winner probability for Player 2 (before normalisation).",
        examples=[0.42],
    )
    format: str = Field(
        ...,
        description="Match format. One of: bo3, bo5, bo7, bo9, bo13.",
        examples=["bo5"],
    )
    player1: Optional[str] = Field(
        default="Player 1",
        max_length=128,
        description="Display name for Player 1.",
    )
    player2: Optional[str] = Field(
        default="Player 2",
        max_length=128,
        description="Display name for Player 2.",
    )

    @field_validator("p_player1", "p_player2")
    @classmethod
    def _check_probability(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("Probability must be strictly between 0 and 1.")
        return v

    @field_validator("format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        if v.lower().strip() not in _FORMAT_MAP:
            raise ValueError(
                f"Unknown format '{v}'. Supported: {sorted(_FORMAT_MAP.keys())}"
            )
        return v.lower().strip()


# ---------------------------------------------------------------------------
# Route: POST /derivatives/generate
# ---------------------------------------------------------------------------


@router.post(
    "/derivatives/generate",
    summary="Generate derivative markets for a darts match",
    tags=["Derivatives"],
    response_description="Full derivative market set.",
)
async def generate_derivatives(body: DerivativeGenerateRequest) -> Dict[str, Any]:
    """
    Generate all derivative markets for a darts match from match-winner
    probabilities and the declared format.

    The result is cached for 60 seconds by match_id so subsequent
    GET /derivatives/{match_id} calls are served from memory.
    """
    try:
        payload = generate_darts_derivatives(
            match_id=body.match_id,
            p_player1=body.p_player1,
            p_player2=body.p_player2,
            format_key=body.format,
            player1=body.player1 or "Player 1",
            player2=body.player2 or "Player 2",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(
            "darts_derivatives_generate_error",
            match_id=body.match_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Derivative generation failed: {exc}",
        ) from exc

    _cache_set(body.match_id, payload)

    logger.info(
        "darts_derivatives_generated",
        match_id=body.match_id,
        format=body.format,
        total_markets=payload["total_markets"],
    )

    return {
        "success": True,
        "data": payload,
        "meta": {
            "request_id": str(uuid.uuid4()),
            "cached": False,
        },
    }


# ---------------------------------------------------------------------------
# Route: GET /derivatives/{match_id}
# ---------------------------------------------------------------------------


@router.get(
    "/derivatives/{match_id}",
    summary="Retrieve cached derivative markets",
    tags=["Derivatives"],
    response_description="Cached derivative market set (60 s TTL).",
)
async def get_derivatives(match_id: str) -> Dict[str, Any]:
    """
    Return derivative markets previously generated for a match.

    Returns 404 if no generate call has been made within the last 60 seconds.
    """
    payload = _cache_get(match_id)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No derivative markets cached for match '{match_id}'. "
                "Call POST /derivatives/generate first."
            ),
        )
    return {
        "success": True,
        "data": payload,
        "meta": {
            "request_id": str(uuid.uuid4()),
            "cached": True,
        },
    }


# ---------------------------------------------------------------------------
# Route: GET /derivatives/health
# ---------------------------------------------------------------------------


@router.get(
    "/derivatives/health",
    summary="Health check for the derivatives router",
    tags=["Derivatives"],
    include_in_schema=True,
)
async def derivatives_health() -> Dict[str, Any]:
    """Liveness probe for the darts derivatives subsystem."""
    return {
        "status": "ok",
        "service": "xg3-darts-derivatives",
        "formats_supported": sorted(_FORMAT_MAP.keys()),
        "cache_size": len(_CACHE),
    }
