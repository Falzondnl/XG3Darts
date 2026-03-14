"""
PDC World Cup of Darts API routes — Sprint 6 full implementation.

The World Cup format is a doubles competition: each national pair plays
2 singles matches and 1 doubles match per tie.  The team that wins 2 of
3 matches advances.

Endpoints
---------
POST /api/v1/darts/worldcup/price           — price a full knockout tie
POST /api/v1/darts/worldcup/group-stage     — price a group-stage tie

Pricing is entirely formula-driven (Markov chain leg model + match
combinatorics).  No hardcoded odds.
"""
from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from competition.format_registry import get_format, DartsFormatError
from engines.doubles.team_visit_model import DoublesTeam
from engines.doubles.world_cup_pricer import WorldCupMatchup, WorldCupPricer
from engines.errors import DartsEngineError, DartsDataError

logger = structlog.get_logger(__name__)
router = APIRouter()

_pricer = WorldCupPricer()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class WorldCupTeamInput(BaseModel):
    """One national team for a World Cup tie."""
    team_id: str = Field(..., description="Country code, e.g. 'ENG'")
    player_a_id: str
    player_b_id: str
    player_a_three_da: float = Field(..., gt=0, lt=200)
    player_b_three_da: float = Field(..., gt=0, lt=200)
    player_a_checkout_rate: float = Field(default=0.40, ge=0.0, le=1.0)
    player_b_checkout_rate: float = Field(default=0.40, ge=0.0, le=1.0)

    @field_validator("player_b_id")
    @classmethod
    def players_must_differ(cls, v: str, info) -> str:
        if info.data.get("player_a_id") == v:
            raise ValueError("player_a_id and player_b_id must be different.")
        return v


class WorldCupPriceRequest(BaseModel):
    """Request body for a full World Cup knockout tie."""
    team1: WorldCupTeamInput
    team2: WorldCupTeamInput
    stage: str = Field(
        default="Round 1",
        description=(
            "Stage name, e.g. 'Round 1', 'Quarter-Final', 'Semi-Final', 'Final'"
        ),
    )
    team1_starts_singles1: bool = True
    team1_starts_singles2: bool = False
    team1_starts_doubles: bool = True

    @field_validator("team2")
    @classmethod
    def teams_must_differ(cls, v: WorldCupTeamInput, info) -> WorldCupTeamInput:
        t1 = info.data.get("team1")
        if t1 and t1.team_id == v.team_id:
            raise ValueError("team1 and team2 must have different team_ids.")
        return v


class GroupStageRequest(BaseModel):
    """Request body for a World Cup group-stage tie (same format as knockout)."""
    team1: WorldCupTeamInput
    team2: WorldCupTeamInput
    team1_starts_singles1: bool = True
    team1_starts_singles2: bool = False
    team1_starts_doubles: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_doubles_team(t: WorldCupTeamInput, stage: bool) -> DoublesTeam:
    return DoublesTeam(
        team_id=t.team_id,
        player_a_id=t.player_a_id,
        player_b_id=t.player_b_id,
        player_a_three_da=t.player_a_three_da,
        player_b_three_da=t.player_b_three_da,
        player_a_checkout_rate=t.player_a_checkout_rate,
        player_b_checkout_rate=t.player_b_checkout_rate,
    )


def _stage_to_round_names(stage: str) -> tuple[str, str]:
    """Return (singles_round_name, doubles_round_name) for the given stage."""
    mapping = {
        "Round 1": ("Round 1 Singles", "Round 1 Doubles"),
        "Quarter-Final": ("Quarter-Final Singles", "Quarter-Final Doubles"),
        "Semi-Final": ("Semi-Final Singles", "Semi-Final Doubles"),
        "Final": ("Final Singles", "Final Doubles"),
    }
    if stage not in mapping:
        raise DartsFormatError(
            f"Unknown World Cup stage {stage!r}. "
            f"Valid stages: {list(mapping.keys())}"
        )
    return mapping[stage]


def _format_result(result, team1_id: str, team2_id: str) -> dict[str, Any]:
    """Serialise WorldCupPriceResult to JSON-safe dict."""
    base = {
        "team1_id": team1_id,
        "team2_id": team2_id,
        "team1_win_prob": round(result.team1_win, 6),
        "team2_win_prob": round(result.team2_win, 6),
        "team1_decimal_odds": round(1.0 / result.team1_win, 4) if result.team1_win > 1e-9 else None,
        "team2_decimal_odds": round(1.0 / result.team2_win, 4) if result.team2_win > 1e-9 else None,
    }
    if result.singles1_result:
        base["singles1"] = {
            "team1_player_win_prob": round(result.singles1_result.p1_win, 6),
            "team2_player_win_prob": round(result.singles1_result.p2_win, 6),
        }
    if result.singles2_result:
        base["singles2"] = {
            "team1_player_win_prob": round(result.singles2_result.p1_win, 6),
            "team2_player_win_prob": round(result.singles2_result.p2_win, 6),
        }
    if result.doubles_result:
        base["doubles"] = {
            "team1_win_prob": round(result.doubles_result.p1_win, 6),
            "team2_win_prob": round(result.doubles_result.p2_win, 6),
        }
    return base


# ---------------------------------------------------------------------------
# POST /worldcup/price
# ---------------------------------------------------------------------------

@router.post(
    "/worldcup/price",
    summary="Price a PDC World Cup knockout tie",
    response_model=dict,
    tags=["World Cup"],
)
async def price_worldcup_tie(request: WorldCupPriceRequest) -> dict[str, Any]:
    """
    Price a full PDC World Cup knockout tie (2 singles + 1 doubles).

    Returns P(team wins tie) and individual component match prices.
    Uses the doubles Markov engine — not the singles engine.
    """
    fmt_code = "PDC_WCUP"
    try:
        fmt = get_format(fmt_code)
        singles_round, doubles_round = _stage_to_round_names(request.stage)
        singles1_fmt = fmt.get_round(singles_round)
        singles2_fmt = fmt.get_round(singles_round)
        doubles_fmt = fmt.get_round(doubles_round)
    except DartsFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    stage_flag = request.stage in ("Semi-Final", "Final")
    team1 = _build_doubles_team(request.team1, stage=stage_flag)
    team2 = _build_doubles_team(request.team2, stage=stage_flag)
    matchup = WorldCupMatchup(team1=team1, team2=team2, stage=stage_flag)

    try:
        result = _pricer.price_full_tie(
            matchup=matchup,
            singles1_fmt=singles1_fmt,
            singles2_fmt=singles2_fmt,
            doubles_fmt=doubles_fmt,
            team1_starts_singles1=request.team1_starts_singles1,
            team1_starts_singles2=request.team1_starts_singles2,
            team1_starts_doubles=request.team1_starts_doubles,
        )
    except (DartsEngineError, DartsDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    logger.info(
        "worldcup_tie_priced",
        team1=request.team1.team_id,
        team2=request.team2.team_id,
        stage=request.stage,
        team1_win=round(result.team1_win, 4),
    )

    return {
        "format": fmt_code,
        "stage": request.stage,
        **_format_result(result, request.team1.team_id, request.team2.team_id),
    }


# ---------------------------------------------------------------------------
# POST /worldcup/group-stage
# ---------------------------------------------------------------------------

@router.post(
    "/worldcup/group-stage",
    summary="Price a PDC World Cup group-stage tie",
    response_model=dict,
    tags=["World Cup"],
)
async def price_worldcup_group_stage(request: GroupStageRequest) -> dict[str, Any]:
    """
    Price a PDC World Cup group-stage tie.

    The group stage uses the same 2 singles + 1 doubles format as knockout
    rounds, but using Round 1 leg formats.  No draw is possible in the
    World Cup (unlike Premier League group stage).
    """
    fmt_code = "PDC_WCUP"
    try:
        fmt = get_format(fmt_code)
        singles_fmt = fmt.get_round("Round 1 Singles")
        doubles_fmt = fmt.get_round("Round 1 Doubles")
    except DartsFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    team1 = _build_doubles_team(request.team1, stage=False)
    team2 = _build_doubles_team(request.team2, stage=False)
    matchup = WorldCupMatchup(team1=team1, team2=team2, stage=False)

    try:
        result = _pricer.price_full_tie(
            matchup=matchup,
            singles1_fmt=singles_fmt,
            singles2_fmt=singles_fmt,
            doubles_fmt=doubles_fmt,
            team1_starts_singles1=request.team1_starts_singles1,
            team1_starts_singles2=request.team1_starts_singles2,
            team1_starts_doubles=request.team1_starts_doubles,
        )
    except (DartsEngineError, DartsDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    logger.info(
        "worldcup_group_stage_priced",
        team1=request.team1.team_id,
        team2=request.team2.team_id,
        team1_win=round(result.team1_win, 4),
    )

    return {
        "format": fmt_code,
        "stage": "Group Stage",
        **_format_result(result, request.team1.team_id, request.team2.team_id),
    }
