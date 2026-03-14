"""
Outright / tournament winner pricing API routes — Sprint 6 full implementation.

Endpoints
---------
POST /api/v1/darts/outrights/simulate      — run Monte Carlo simulation
GET  /api/v1/darts/outrights/{competition_id} — retrieve cached / recomputed results
POST /api/v1/darts/outrights/update-field  — update field and resimulate
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
