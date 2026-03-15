"""
Outright / tournament winner pricing API routes — Sprint 6 full implementation.

Endpoints
---------
POST /api/v1/darts/outrights/simulate            — run Monte Carlo simulation
GET  /api/v1/darts/outrights/upcoming            — list upcoming tournaments with top-8 winner odds
GET  /api/v1/darts/outrights/player/{player_name} — player's outright odds across all tournaments
GET  /api/v1/darts/outrights/{competition_id}    — retrieve cached / recomputed results
POST /api/v1/darts/outrights/update-field        — update field and resimulate

Note: /upcoming and /player/{name} are declared before /{competition_id} in the
router to prevent FastAPI matching "upcoming" or "player" as a competition_id path
parameter.
"""
from __future__ import annotations

from typing import Any, Optional

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from competition.format_registry import get_format, DartsFormatError
from engines.errors import DartsEngineError
from outrights.tournament_simulator import (
    DartsTournamentSimulator,
    TournamentField,
    OutrightSimResult,
)

logger = structlog.get_logger(__name__)
router = APIRouter()

_simulator = DartsTournamentSimulator()

# In-memory result cache keyed by competition_id.  In production this would
# be backed by Redis.
_result_cache: dict[str, OutrightSimResult] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PlayerEntry(BaseModel):
    """A single player entry in the tournament field."""
    player_id: str
    elo_rating: float = Field(..., gt=0, lt=4000)
    three_da: float = Field(..., gt=0, lt=200)


class SimulateRequest(BaseModel):
    """Request body for outright simulation."""
    competition_id: str = Field(..., min_length=1)
    format_code: str = Field(..., description="e.g. 'PDC_WC', 'PDC_PL'")
    players: list[PlayerEntry] = Field(..., min_length=2)
    n_simulations: int = Field(default=100_000, ge=10_000, le=1_000_000)
    use_antithetic: bool = Field(default=True)
    use_gpu: bool = Field(default=False)

    @field_validator("players")
    @classmethod
    def players_must_be_unique(cls, v: list[PlayerEntry]) -> list[PlayerEntry]:
        ids = [p.player_id for p in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Player IDs must be unique within the field.")
        return v


class UpdateFieldRequest(BaseModel):
    """Update an existing field (e.g. after player withdrawal) and resimulate."""
    competition_id: str
    format_code: str
    players: list[PlayerEntry] = Field(..., min_length=2)
    n_simulations: int = Field(default=100_000, ge=10_000, le=1_000_000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_field(req: SimulateRequest | UpdateFieldRequest) -> TournamentField:
    """Convert request data into a TournamentField."""
    player_ids = [p.player_id for p in req.players]
    elo_ratings = {p.player_id: p.elo_rating for p in req.players}
    three_da_stats = {p.player_id: p.three_da for p in req.players}

    # Build sequential bracket: 1 vs 2, 3 vs 4, etc.
    bracket: dict[str, list[str]] = {}
    n = len(player_ids)
    pair_idx = 1
    i = 0
    while i + 1 < n:
        bracket[f"R1_M{pair_idx}"] = [player_ids[i], player_ids[i + 1]]
        pair_idx += 1
        i += 2

    return TournamentField(
        competition_id=req.competition_id,
        format_code=req.format_code,
        players=player_ids,
        bracket=bracket,
        elo_ratings=elo_ratings,
        three_da_stats=three_da_stats,
    )


def _result_to_dict(result: OutrightSimResult) -> dict[str, Any]:
    """Serialise OutrightSimResult to a JSON-safe dict."""
    return {
        "competition_id": result.competition_id,
        "n_simulations": result.n_simulations,
        "player_win_probs": {
            pid: round(prob, 6)
            for pid, prob in sorted(
                result.player_win_probs.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        },
        "top4_probs": {
            pid: round(prob, 6)
            for pid, prob in sorted(
                result.top4_probs.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        },
        "confidence_intervals": {
            pid: {"lower": round(lo, 6), "upper": round(hi, 6)}
            for pid, (lo, hi) in result.confidence_intervals.items()
        },
    }


# ---------------------------------------------------------------------------
# POST /outrights/simulate
# ---------------------------------------------------------------------------

@router.post(
    "/outrights/simulate",
    summary="Run Monte Carlo tournament outright simulation",
    response_model=dict,
    tags=["Outrights"],
)
async def simulate_tournament(request: SimulateRequest) -> dict[str, Any]:
    """
    Run antithetic Monte Carlo simulation for tournament outright winner prices.

    Accepts a full field with ELO ratings and 3DA stats.  Returns empirical
    win probabilities, top-4 probabilities, and 95 % Wilson CIs for each
    player.

    The result is cached in memory and accessible via GET /outrights/{competition_id}.
    """
    # Validate format code
    try:
        get_format(request.format_code)
    except DartsFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    field = _build_field(request)

    try:
        result = _simulator.simulate(
            competition_id=request.competition_id,
            field=field,
            n_simulations=request.n_simulations,
            use_gpu=request.use_gpu,
            use_antithetic=request.use_antithetic,
        )
    except DartsEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Cache result
    _result_cache[request.competition_id] = result

    logger.info(
        "outrights_simulated",
        competition_id=request.competition_id,
        format_code=request.format_code,
        n_players=len(request.players),
        n_simulations=result.n_simulations,
    )

    return _result_to_dict(result)


# ---------------------------------------------------------------------------
# GET /outrights/upcoming
# ---------------------------------------------------------------------------

@router.get(
    "/outrights/upcoming",
    summary="List upcoming tournaments with winner odds (top 8 players)",
    response_model=dict,
    tags=["Outrights"],
)
async def get_upcoming_tournaments() -> dict[str, Any]:
    """
    Return all upcoming tournaments that have cached simulation results,
    showing the top 8 players by win probability for each tournament.

    A tournament is considered "upcoming" if it has a result in the
    in-memory cache (populated via POST /outrights/simulate).  In
    production this would be backed by Redis with TTL-based expiry and
    filtered against a competitions schedule table.

    Response shape per tournament::

        {
          "competition_id": "PDC_WC_2026",
          "n_simulations": 100000,
          "top_8_players": [
            {"player_id": "van_gerwen", "win_probability": 0.182, "decimal_odds": 5.49},
            ...
          ]
        }

    Returns an empty list when no simulations are cached.
    """
    tournaments: list[dict[str, Any]] = []

    for competition_id, result in _result_cache.items():
        # Sort by win probability descending, take top 8
        sorted_players = sorted(
            result.player_win_probs.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:8]

        top_8 = []
        for player_id, win_prob in sorted_players:
            decimal_odds = round(1.0 / win_prob, 2) if win_prob > 0 else None
            top_8.append({
                "player_id": player_id,
                "win_probability": round(win_prob, 6),
                "decimal_odds": decimal_odds,
            })

        tournaments.append({
            "competition_id": competition_id,
            "n_simulations": result.n_simulations,
            "top_8_players": top_8,
        })

    logger.info(
        "outrights_upcoming_listed",
        n_tournaments=len(tournaments),
    )

    return {
        "upcoming_tournaments": tournaments,
        "n_tournaments": len(tournaments),
        "note": (
            "Results are sourced from the in-memory simulation cache. "
            "POST to /outrights/simulate to add a tournament."
        ),
    }


# ---------------------------------------------------------------------------
# GET /outrights/player/{player_name}
# ---------------------------------------------------------------------------

@router.get(
    "/outrights/player/{player_name}",
    summary="Get a player's outright odds across all cached tournaments",
    response_model=dict,
    tags=["Outrights"],
)
async def get_player_outright_odds(player_name: str) -> dict[str, Any]:
    """
    Return a specific player's win probability and decimal odds across
    all tournaments currently in the simulation cache.

    ``player_name`` is matched case-insensitively against the player_id
    values stored in each cached simulation result.  In production the
    player_id is a UUID slug (e.g. ``van_gerwen_michael``); callers
    should use the slug form for reliable lookup.

    Parameters
    ----------
    player_name:
        Player ID slug to search for (case-insensitive).

    Raises
    ------
    HTTPException(404)
        If the player is not found in any cached tournament.
    HTTPException(404)
        If no tournaments are cached at all.
    """
    if not _result_cache:
        raise HTTPException(
            status_code=404,
            detail=(
                "No tournament simulation results are cached. "
                "POST to /outrights/simulate to run a simulation first."
            ),
        )

    player_name_lower = player_name.lower()
    appearances: list[dict[str, Any]] = []

    for competition_id, result in _result_cache.items():
        # Case-insensitive match on player_id keys
        matched_id: Optional[str] = None
        for pid in result.player_win_probs:
            if pid.lower() == player_name_lower:
                matched_id = pid
                break

        if matched_id is None:
            continue

        win_prob = result.player_win_probs[matched_id]
        top4_prob = result.top4_probs.get(matched_id, 0.0)
        ci = result.confidence_intervals.get(matched_id, (0.0, 0.0))
        decimal_odds = round(1.0 / win_prob, 2) if win_prob > 0 else None

        # Compute rank within this tournament
        sorted_probs = sorted(
            result.player_win_probs.values(),
            reverse=True,
        )
        rank = sorted_probs.index(win_prob) + 1

        appearances.append({
            "competition_id": competition_id,
            "win_probability": round(win_prob, 6),
            "top4_probability": round(top4_prob, 6),
            "decimal_odds": decimal_odds,
            "rank_in_field": rank,
            "field_size": len(result.player_win_probs),
            "confidence_interval": {
                "lower": round(ci[0], 6),
                "upper": round(ci[1], 6),
            },
            "n_simulations": result.n_simulations,
        })

    if not appearances:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Player '{player_name}' not found in any cached tournament. "
                "Check the player_id slug and ensure the tournament has been simulated."
            ),
        )

    # Sort by win probability descending (best tournament result first)
    appearances.sort(key=lambda x: x["win_probability"], reverse=True)

    logger.info(
        "outrights_player_looked_up",
        player_name=player_name,
        n_tournaments=len(appearances),
    )

    return {
        "player_id": player_name,
        "n_tournaments": len(appearances),
        "tournaments": appearances,
    }


# ---------------------------------------------------------------------------
# GET /outrights/catalog — must be BEFORE /{competition_id} to avoid shadowing
# ---------------------------------------------------------------------------

@router.get(
    "/outrights/catalog",
    summary="List all active outright markets",
    tags=["Outrights"],
)
async def list_outright_markets_early() -> dict[str, Any]:
    """Return available outright markets for all upcoming tournaments."""
    return {
        "markets": [
            {
                "competition_id": "pdc_pl_2026",
                "name": "PDC Premier League 2026",
                "status": "active",
                "format_code": "PDC_PL",
                "players": 8,
                "prize_fund": 1000000,
                "simulate_url": "/api/v1/darts/outrights/simulate",
                "quick_price_url": "/api/v1/darts/outrights/pdc_pl_2026/quick",
            },
            {
                "competition_id": "pdc_wc_2026_27",
                "name": "PDC World Championship 2026/27",
                "status": "upcoming",
                "format_code": "PDC_WC",
                "players": 96,
                "prize_fund": 3000000,
                "simulate_url": "/api/v1/darts/outrights/simulate",
                "quick_price_url": "/api/v1/darts/outrights/pdc_wc_2026_27/quick",
            },
            {
                "competition_id": "pdc_gs_2026",
                "name": "PDC Grand Slam of Darts 2026",
                "status": "upcoming",
                "format_code": "PDC_GS",
                "players": 32,
                "prize_fund": 650000,
                "simulate_url": "/api/v1/darts/outrights/simulate",
            },
            {
                "competition_id": "pdc_pcf_2026",
                "name": "PDC Players Championship Finals 2026",
                "status": "upcoming",
                "format_code": "PDC_PCF",
                "players": 64,
                "prize_fund": 750000,
                "simulate_url": "/api/v1/darts/outrights/simulate",
            },
        ]
    }


# ---------------------------------------------------------------------------
# GET /outrights/{competition_id}
# ---------------------------------------------------------------------------

@router.get(
    "/outrights/{competition_id}",
    summary="Retrieve outright simulation results",
    response_model=dict,
    tags=["Outrights"],
)
async def get_outright_prices(
    competition_id: str,
    n_simulations: int = Query(100_000, ge=10_000, le=1_000_000),
) -> dict[str, Any]:
    """
    Return cached outright simulation results for a competition.

    If no cached result exists, returns a 404.  To (re-)run the simulation,
    POST to /outrights/simulate.
    """
    if competition_id in _result_cache:
        logger.info(
            "outrights_cache_hit",
            competition_id=competition_id,
        )
        return _result_to_dict(_result_cache[competition_id])

    logger.info(
        "outrights_cache_miss",
        competition_id=competition_id,
    )
    raise HTTPException(
        status_code=404,
        detail=(
            f"No simulation results found for competition '{competition_id}'. "
            "POST to /outrights/simulate to run the simulation first."
        ),
    )


# ---------------------------------------------------------------------------
# POST /outrights/update-field
# ---------------------------------------------------------------------------

@router.post(
    "/outrights/update-field",
    summary="Update tournament field and resimulate",
    response_model=dict,
    tags=["Outrights"],
)
async def update_field(request: UpdateFieldRequest) -> dict[str, Any]:
    """
    Update a tournament field (e.g. after player withdrawal) and re-run simulation.

    Overwrites any cached result for the competition ID.
    """
    try:
        get_format(request.format_code)
    except DartsFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    field = _build_field(request)

    try:
        result = _simulator.simulate(
            competition_id=request.competition_id,
            field=field,
            n_simulations=request.n_simulations,
            use_gpu=False,
            use_antithetic=True,
        )
    except DartsEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    _result_cache[request.competition_id] = result

    logger.info(
        "outrights_field_updated",
        competition_id=request.competition_id,
        n_players=len(request.players),
        n_simulations=result.n_simulations,
    )

    return {
        "status": "updated",
        **_result_to_dict(result),
    }


# ---------------------------------------------------------------------------
# Pre-populated 2026 PDC Premier League field
# ---------------------------------------------------------------------------

# 2026 PDC Premier League participants with real ELO/3DA estimates
_PDC_PL_2026_FIELD = [
    {"player_id": "luke-littler",     "elo_rating": 2450.0, "three_da": 104.2},
    {"player_id": "michael-van-gerwen", "elo_rating": 2380.0, "three_da": 98.7},
    {"player_id": "luke-humphries",   "elo_rating": 2310.0, "three_da": 96.1},
    {"player_id": "michael-smith",    "elo_rating": 2180.0, "three_da": 92.4},
    {"player_id": "gerwyn-price",     "elo_rating": 2140.0, "three_da": 91.8},
    {"player_id": "peter-wright",     "elo_rating": 2090.0, "three_da": 88.5},
    {"player_id": "stephen-bunting",  "elo_rating": 2060.0, "three_da": 87.3},
    {"player_id": "chris-dobey",      "elo_rating": 2010.0, "three_da": 86.1},
]

# 2026 PDC World Championship participants (top 16 seeds)
_PDC_WC_2026_FIELD = [
    {"player_id": "luke-littler",     "elo_rating": 2450.0, "three_da": 104.2},
    {"player_id": "michael-van-gerwen", "elo_rating": 2380.0, "three_da": 98.7},
    {"player_id": "luke-humphries",   "elo_rating": 2310.0, "three_da": 96.1},
    {"player_id": "michael-smith",    "elo_rating": 2180.0, "three_da": 92.4},
    {"player_id": "gerwyn-price",     "elo_rating": 2140.0, "three_da": 91.8},
    {"player_id": "peter-wright",     "elo_rating": 2090.0, "three_da": 88.5},
    {"player_id": "stephen-bunting",  "elo_rating": 2060.0, "three_da": 87.3},
    {"player_id": "chris-dobey",      "elo_rating": 2010.0, "three_da": 86.1},
    {"player_id": "jonny-clayton",    "elo_rating": 1980.0, "three_da": 84.9},
    {"player_id": "dave-chisnall",    "elo_rating": 1950.0, "three_da": 83.5},
    {"player_id": "nathan-aspinall",  "elo_rating": 1940.0, "three_da": 83.1},
    {"player_id": "jose-de-sousa",    "elo_rating": 1910.0, "three_da": 82.4},
    {"player_id": "callan-rydz",      "elo_rating": 1870.0, "three_da": 80.7},
    {"player_id": "ross-smith",       "elo_rating": 1840.0, "three_da": 79.8},
    {"player_id": "gary-anderson",    "elo_rating": 1820.0, "three_da": 79.2},
    {"player_id": "damon-heta",       "elo_rating": 1800.0, "three_da": 78.5},
]


@router.get(
    "/outrights/catalog",
    summary="List all active outright markets",
    tags=["Outrights"],
)
async def list_outright_markets() -> dict[str, Any]:
    """
    Return available outright markets for all upcoming tournaments.

    Pre-populated with the 2026 PDC Premier League and World Championship fields.
    Simulate any market by POSTing to /outrights/simulate with the field data.
    """
    return {
        "markets": [
            {
                "competition_id": "pdc_pl_2026",
                "name": "PDC Premier League 2026",
                "status": "active",
                "format_code": "PDC_PL",
                "players": len(_PDC_PL_2026_FIELD),
                "prize_fund": 1000000,
                "simulate_url": "/api/v1/darts/outrights/simulate",
                "quick_price_url": "/api/v1/darts/outrights/pdc_pl_2026/quick",
            },
            {
                "competition_id": "pdc_wc_2026_27",
                "name": "PDC World Championship 2026/27",
                "status": "upcoming",
                "format_code": "PDC_WC",
                "players": 96,
                "prize_fund": 3000000,
                "simulate_url": "/api/v1/darts/outrights/simulate",
                "quick_price_url": "/api/v1/darts/outrights/pdc_wc_2026_27/quick",
            },
            {
                "competition_id": "pdc_gs_2026",
                "name": "PDC Grand Slam of Darts 2026",
                "status": "upcoming",
                "format_code": "PDC_GS",
                "players": 32,
                "prize_fund": 650000,
                "simulate_url": "/api/v1/darts/outrights/simulate",
            },
            {
                "competition_id": "pdc_pcf_2026",
                "name": "PDC Players Championship Finals 2026",
                "status": "upcoming",
                "format_code": "PDC_PCF",
                "players": 64,
                "prize_fund": 750000,
                "simulate_url": "/api/v1/darts/outrights/simulate",
            },
        ]
    }


@router.get(
    "/outrights/{competition_id}/quick",
    summary="Quick outright price using pre-populated field",
    tags=["Outrights"],
)
async def quick_outright_price(
    competition_id: str,
    n_simulations: int = Query(50_000, ge=10_000, le=500_000),
    base_margin: float = Query(0.06, gt=0.0, le=0.20),
) -> dict[str, Any]:
    """
    Run outright simulation using the pre-populated field for known competitions.

    Currently supports: pdc_pl_2026, pdc_wc_2026_27 (top 16 seeds).
    For other competitions POST to /outrights/simulate with a custom field.
    """
    from margin.blending_engine import DartsMarginEngine

    field_map = {
        "pdc_pl_2026":    ("PDC_PL", _PDC_PL_2026_FIELD),
        "pdc_wc_2026_27": ("PDC_WC", _PDC_WC_2026_FIELD),
    }
    if competition_id not in field_map:
        raise HTTPException(
            status_code=404,
            detail=f"No pre-populated field for '{competition_id}'. "
                   "Use POST /outrights/simulate with a custom field."
        )

    format_code, field_data = field_map[competition_id]

    # Check cache first
    cache_key = f"{competition_id}_quick_{n_simulations}"
    if cache_key in _result_cache:
        result = _result_cache[cache_key]
    else:
        from outrights.tournament_simulator import TournamentField
        player_ids = [p["player_id"] for p in field_data]
        elo_ratings = {p["player_id"]: p["elo_rating"] for p in field_data}
        three_da = {p["player_id"]: p["three_da"] for p in field_data}

        bracket: dict[str, list[str]] = {}
        for i in range(0, len(player_ids) - 1, 2):
            bracket[f"R1_M{i//2+1}"] = [player_ids[i], player_ids[i+1]]

        field = TournamentField(
            competition_id=competition_id,
            format_code=format_code,
            players=player_ids,
            bracket=bracket,
            elo_ratings=elo_ratings,
            three_da_stats=three_da,
        )

        try:
            get_format(format_code)
        except DartsFormatError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        try:
            result = _simulator.simulate(
                competition_id=competition_id,
                field=field,
                n_simulations=n_simulations,
                use_gpu=False,
                use_antithetic=True,
            )
        except DartsEngineError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        _result_cache[cache_key] = result

    # Apply Shin margin to win probs
    from margin.shin_margin import ShinMarginModel
    shin = ShinMarginModel()
    adj_probs = shin.apply_shin_margin(
        true_probs=result.player_win_probs,
        target_margin=base_margin,
    )

    sorted_players = sorted(
        result.player_win_probs.items(), key=lambda x: x[1], reverse=True
    )

    players_out = []
    for pid, true_p in sorted_players:
        adj_p = adj_probs.get(pid, true_p)
        players_out.append({
            "player_id": pid,
            "true_win_prob": round(true_p, 5),
            "win_odds": round(1.0 / adj_p, 2) if adj_p > 1e-9 else None,
            "top4_prob": round(result.top4_probs.get(pid, 0.0), 5),
            "top4_odds": round(1.0 / (result.top4_probs[pid] * (1 + base_margin)), 2)
                         if result.top4_probs.get(pid, 0) > 1e-9 else None,
        })

    return {
        "competition_id": competition_id,
        "format_code": format_code,
        "n_simulations": result.n_simulations,
        "base_margin": base_margin,
        "players": players_out,
    }
