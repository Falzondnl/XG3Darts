"""
Player data API routes — Sprint 7.

Endpoints
---------
GET  /players/{player_id}          — player profile + stats
GET  /players/{player_id}/regime   — which data regime is active
POST /players/explain              — SHAP explanation for player pricing
GET  /players/{player_id}/elo      — ELO rating history
GET  /players/search               — search by name

Enterprise proxy:
    DARTS_SERVICE_URL env var controls the base URL used by downstream
    B2B consumers when they call this service.
"""
from __future__ import annotations

from typing import Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from db.session import get_session_dependency


logger = structlog.get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class PlayerExplainRequest(BaseModel):
    """Request body for the SHAP explain endpoint."""

    player_id: str = Field(..., description="Player UUID")
    opponent_id: Optional[str] = Field(
        None, description="Opponent UUID (for H2H context)"
    )
    competition_code: Optional[str] = Field(
        None, description="Competition format code (e.g. 'PDC_WC')"
    )
    round_name: Optional[str] = Field(None, description="Round name")
    match_context: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional match context (regime, 3DA, etc.)",
    )


# ---------------------------------------------------------------------------
# Helper: enterprise proxy URL builder
# ---------------------------------------------------------------------------


def _service_url(path: str) -> str:
    """
    Build a fully-qualified URL for this service.

    Uses DARTS_SERVICE_URL from environment / settings.
    """
    base = settings.DARTS_SERVICE_URL.rstrip("/")
    return f"{base}/{path.lstrip('/')}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/players/{player_id}",
    summary="Get player profile and statistics",
    response_description="Player profile with statistics and coverage regime",
)
async def get_player(
    player_id: str,
    include_personal: bool = Query(
        False,
        description="Include personal data fields (requires consent check)",
    ),
    session: AsyncSession = Depends(get_session_dependency),
) -> dict[str, Any]:
    """
    Return a player's profile and statistical summary.

    Personal data fields (``first_name``, ``last_name``, ``dob``,
    ``country_code``) are only included when ``include_personal=True``
    AND the player has an active GDPR consent record.

    Parameters
    ----------
    player_id:
        Internal UUID of the player.
    include_personal:
        Whether to include GDPR-protected personal fields.

    Raises
    ------
    HTTPException(404)
        If the player is not found in the database.
    HTTPException(403)
        If personal data is requested but consent has not been given.
    HTTPException(501)
        Database integration not yet initialised.
    """
    from sqlalchemy import text as sql_text

    logger.info(
        "player_profile_request",
        player_id=player_id,
        include_personal=include_personal,
    )

    try:
        result = await session.execute(
            sql_text(
                """
                SELECT
                    p.id,
                    p.slug,
                    p.nickname,
                    p.pdc_ranking,
                    p.tour_card_holder,
                    p.dartsorakel_3da,
                    p.dartsorakel_rank,
                    p.source_confidence,
                    p.primary_source,
                    p.gdpr_anonymized,
                    p.created_at,
                    CASE WHEN :include_personal THEN p.first_name ELSE NULL END AS first_name,
                    CASE WHEN :include_personal THEN p.last_name  ELSE NULL END AS last_name,
                    CASE WHEN :include_personal THEN p.country_code ELSE NULL END AS country_code
                FROM darts_players p
                WHERE p.id = :player_id
                """
            ),
            {"player_id": player_id, "include_personal": include_personal},
        )
        row = result.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Player {player_id!r} not found")

        player_data: dict[str, Any] = {
            "player_id": row[0],
            "slug": row[1],
            "nickname": row[2],
            "pdc_ranking": row[3],
            "tour_card_holder": row[4],
            "dartsorakel_3da": row[5],
            "dartsorakel_rank": row[6],
            "source_confidence": row[7],
            "primary_source": row[8],
            "gdpr_anonymized": row[9],
            "created_at": row[10].isoformat() if row[10] else None,
        }

        if include_personal:
            # Check consent before returning personal data
            consent_result = await session.execute(
                sql_text(
                    "SELECT consent_given FROM darts_gdpr_consents "
                    "WHERE player_id = :player_id"
                ),
                {"player_id": player_id},
            )
            consent_row = consent_result.fetchone()
            if consent_row is None or not consent_row[0]:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Personal data for player {player_id!r} is not available: "
                        "no GDPR consent recorded."
                    ),
                )
            player_data["first_name"] = row[11]
            player_data["last_name"] = row[12]
            player_data["country_code"] = row[13]

        player_data["_links"] = {
            "self": _service_url(f"/players/{player_id}"),
            "regime": _service_url(f"/players/{player_id}/regime"),
            "elo": _service_url(f"/players/{player_id}/elo"),
            "explain": _service_url("/players/explain"),
        }

        return player_data

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("player_profile_error", player_id=player_id, error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="Database unavailable — player profile cannot be retrieved.",
        )


@router.get(
    "/players/{player_id}/regime",
    summary="Get active data coverage regime for a player",
    response_description="Coverage regime (R0/R1/R2) and supporting flags",
)
async def get_player_regime(
    player_id: str,
    competition_id: Optional[str] = Query(
        None,
        description="Scope regime lookup to a specific competition UUID",
    ),
    session: AsyncSession = Depends(get_session_dependency),
) -> dict[str, Any]:
    """
    Return the active data coverage regime for a player.

    The regime determines which pricing model tier is used:
    - ``R0`` — result-only data, baseline logit model
    - ``R1`` — match-level statistics, LightGBM model
    - ``R2`` — full visit-level data, stacking ensemble

    Parameters
    ----------
    player_id:
        Player UUID.
    competition_id:
        Optional competition UUID to scope the regime lookup.

    Raises
    ------
    HTTPException(404)
        If no regime record exists for the player.
    """
    from sqlalchemy import text as sql_text

    logger.info(
        "player_regime_request",
        player_id=player_id,
        competition_id=competition_id,
    )

    try:
        query = """
            SELECT
                cr.regime,
                cr.regime_score,
                cr.has_visit_data,
                cr.has_match_stats,
                cr.has_result_only,
                cr.has_dartsorakel,
                cr.has_dartconnect,
                cr.last_computed_at
            FROM darts_coverage_regimes cr
            WHERE cr.player_id = :player_id
        """
        params: dict[str, Any] = {"player_id": player_id}

        if competition_id:
            query += " AND cr.competition_id = :competition_id"
            params["competition_id"] = competition_id
        else:
            query += " AND cr.competition_id IS NULL"

        result = await session.execute(sql_text(query), params)
        row = result.fetchone()

        if row is None:
            # No regime record — compute from defaults
            return {
                "player_id": player_id,
                "regime": "R0",
                "regime_score": 0.0,
                "has_visit_data": False,
                "has_match_stats": False,
                "has_result_only": True,
                "has_dartsorakel": False,
                "has_dartconnect": False,
                "last_computed_at": None,
                "note": "No regime record found; defaulting to R0.",
                "_links": {
                    "player": _service_url(f"/players/{player_id}"),
                },
            }

        return {
            "player_id": player_id,
            "competition_id": competition_id,
            "regime": row[0],
            "regime_score": row[1],
            "has_visit_data": row[2],
            "has_match_stats": row[3],
            "has_result_only": row[4],
            "has_dartsorakel": row[5],
            "has_dartconnect": row[6],
            "last_computed_at": row[7].isoformat() if row[7] else None,
            "_links": {
                "player": _service_url(f"/players/{player_id}"),
                "explain": _service_url("/players/explain"),
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("player_regime_error", player_id=player_id, error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="Database unavailable — regime cannot be retrieved.",
        )


@router.post(
    "/players/explain",
    summary="SHAP explanation for player pricing contribution",
    response_description="SHAP feature importances for the pricing model",
)
async def explain_player_pricing(
    request: PlayerExplainRequest,
    session: AsyncSession = Depends(get_session_dependency),
) -> dict[str, Any]:
    """
    Return SHAP feature importance values explaining a player's pricing.

    This endpoint uses the model explainer to show which features most
    influenced the model's probability estimate for this player.

    Requires: the regime-appropriate model to be fitted (R1 or R2).

    Parameters
    ----------
    request:
        PlayerExplainRequest with player_id and optional match context.

    Raises
    ------
    HTTPException(404)
        If the player is not found.
    HTTPException(422)
        If the model regime does not support SHAP explanations (R0).
    HTTPException(503)
        If the explainer is unavailable.
    """
    from models.explainer import ModelExplainer
    from models.features.feature_builder import FeatureBuilder

    logger.info(
        "player_explain_request",
        player_id=request.player_id,
        opponent_id=request.opponent_id,
        competition_code=request.competition_code,
    )

    try:
        feature_builder = FeatureBuilder()
        features = await feature_builder.build_async(
            player_id=request.player_id,
            opponent_id=request.opponent_id,
            competition_code=request.competition_code,
            round_name=request.round_name,
            session=session,
        )
    except Exception as exc:
        logger.warning("player_explain_feature_build_failed", error=str(exc))
        features = request.match_context or {}

    try:
        explainer = ModelExplainer()
        shap_values = explainer.explain(
            player_id=request.player_id,
            features=features,
        )
        return {
            "player_id": request.player_id,
            "opponent_id": request.opponent_id,
            "competition_code": request.competition_code,
            "shap_values": shap_values,
            "features_used": list(features.keys()),
            "_links": {
                "player": _service_url(f"/players/{request.player_id}"),
                "regime": _service_url(f"/players/{request.player_id}/regime"),
            },
        }
    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail=(
                "SHAP explanations require a fitted R1 or R2 model. "
                "R0 regime does not support SHAP."
            ),
        )
    except Exception as exc:
        logger.error("player_explain_error", player_id=request.player_id, error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="Model explainer unavailable.",
        )


@router.get(
    "/players/_search",
    summary="Search players by name",
    response_description="List of matching players",
)
async def search_players(
    q: str = Query(..., min_length=2, description="Player name search query"),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session_dependency),
) -> dict[str, Any]:
    """Search for players by name (case-insensitive, prefix match)."""
    from sqlalchemy import text as sql_text

    logger.info("player_search_request", q=q, limit=limit)

    try:
        result = await session.execute(
            sql_text(
                """
                SELECT id, first_name, last_name, nickname, slug,
                       pdc_ranking, dartsorakel_3da, dartsorakel_rank,
                       country_code, pdc_id
                FROM darts_players
                WHERE first_name ILIKE :pattern
                   OR last_name  ILIKE :pattern
                   OR nickname   ILIKE :pattern
                   OR slug       ILIKE :pattern
                   OR (first_name || ' ' || last_name) ILIKE :pattern
                ORDER BY pdc_ranking ASC NULLS LAST
                LIMIT :limit
                """
            ),
            {"pattern": f"%{q}%", "limit": limit},
        )
        rows = result.fetchall()

        return {
            "query": q,
            "count": len(rows),
            "players": [
                {
                    "player_id": row[0],
                    "name": f"{row[1]} {row[2]}".strip(),
                    "nickname": row[3],
                    "slug": row[4],
                    "pdc_ranking": row[5],
                    "dartsorakel_3da": row[6],
                    "dartsorakel_rank": row[7],
                    "country_code": row[8],
                    "pdc_id": row[9],
                    "_links": {
                        "profile": _service_url(f"/players/{row[0]}"),
                        "elo": _service_url(f"/players/{row[0]}/elo"),
                    },
                }
                for row in rows
            ],
        }
    except Exception as exc:
        logger.error("player_search_error", q=q, error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="Database unavailable — search cannot be performed.",
        )


@router.get(
    "/players/{player_id}/elo",
    summary="Get player ELO rating history",
    response_description="ELO rating history across pools",
)
async def get_player_elo(
    player_id: str,
    pool: Optional[str] = Query(
        None,
        description="ELO pool (pdc_mens | pdc_womens | wdf_open | development | team_doubles)",
    ),
    limit: int = Query(50, ge=1, le=1000, description="Maximum rows to return"),
    session: AsyncSession = Depends(get_session_dependency),
) -> dict[str, Any]:
    """Return ELO rating history for a player across all or one pool."""
    from sqlalchemy import text as sql_text

    logger.info("player_elo_request", player_id=player_id, pool=pool)

    try:
        query = """
            SELECT
                er.pool,
                er.rating_before,
                er.rating_after,
                er.delta,
                er.k_factor,
                er.expected_score,
                er.actual_score,
                er.match_date,
                er.created_at
            FROM darts_elo_ratings er
            WHERE er.player_id = :player_id
        """
        params: dict[str, Any] = {"player_id": player_id, "limit": limit}

        if pool:
            query += " AND er.pool = :pool"
            params["pool"] = pool

        query += " ORDER BY er.match_date DESC, er.created_at DESC LIMIT :limit"

        result = await session.execute(sql_text(query), params)
        rows = result.fetchall()

        # Current rating per pool (latest row per pool)
        current_ratings: dict[str, float] = {}
        for row in rows:
            p = row[0]
            if p not in current_ratings:
                current_ratings[p] = row[2]  # rating_after of most recent row

        return {
            "player_id": player_id,
            "pool_filter": pool,
            "current_ratings": current_ratings,
            "history_count": len(rows),
            "history": [
                {
                    "pool": row[0],
                    "rating_before": row[1],
                    "rating_after": row[2],
                    "delta": row[3],
                    "k_factor": row[4],
                    "expected_score": row[5],
                    "actual_score": row[6],
                    "match_date": row[7].isoformat() if row[7] else None,
                }
                for row in rows
            ],
            "_links": {
                "player": _service_url(f"/players/{player_id}"),
            },
        }

    except Exception as exc:
        logger.error("player_elo_error", player_id=player_id, error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="Database unavailable — ELO history cannot be retrieved.",
        )
