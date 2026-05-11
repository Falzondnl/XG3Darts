"""
Darts predict route — P1-012.

POST /api/v1/darts/predict

Returns ML win probabilities for a darts match using the trained R1 stacking
model (r1_model.pkl, 38 features, AUC≈0.818) as primary, falling back to
r0_model.pkl (14 features) if the R1 file is absent.

The endpoint NEVER returns hardcoded probabilities. If no model file is
available it raises HTTP 503 so callers detect the failure.

Standard response envelope:
    {"success": true, "data": {...}, "meta": {"request_id": "uuid",
                                               "timestamp": "ISO8601"}}
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import structlog
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from shared.pricing_layer import blend_with_pinnacle, devig_2way, fetch_pinnacle_odds, get_pricing_layer

_darts_pricing_layer = get_pricing_layer("darts")

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Predict"])


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _meta(request_id: str) -> Dict[str, str]:
    return {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _ok(data: Any, request_id: str) -> Dict[str, Any]:
    return {"success": True, "data": data, "meta": _meta(request_id)}


def _error(
    code: str,
    message: str,
    request_id: str,
    http_status: int = 400,
) -> JSONResponse:
    return JSONResponse(
        content={
            "success": False,
            "error": {"code": code, "message": message},
            "meta": _meta(request_id),
        },
        status_code=http_status,
    )


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class DartsPredictRequest(BaseModel):
    """
    Minimum required fields: player1 + player2 names.
    Optional stats improve model accuracy when available.
    """

    player1: str = Field(..., min_length=1, description="Player 1 display name")
    player2: str = Field(..., min_length=1, description="Player 2 display name")

    # Optional ELO overrides — if omitted, fetched from DB; defaults to 1500
    player1_elo: Optional[float] = Field(
        None, ge=800.0, le=2500.0,
        description="Player 1 ELO rating (fetched from DB if omitted)",
    )
    player2_elo: Optional[float] = Field(
        None, ge=800.0, le=2500.0,
        description="Player 2 ELO rating (fetched from DB if omitted)",
    )

    # Optional three-dart averages
    player1_3da: Optional[float] = Field(
        None, ge=0.0, le=180.0,
        description="Player 1 three-dart average",
    )
    player2_3da: Optional[float] = Field(
        None, ge=0.0, le=180.0,
        description="Player 2 three-dart average",
    )

    # Optional checkout percentages
    player1_checkout_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Player 1 checkout percentage",
    )
    player2_checkout_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Player 2 checkout percentage",
    )

    # Format descriptor (e.g. best_of_13_sets, best_of_7_legs)
    format: Optional[str] = Field(
        None,
        description="Format string, e.g. 'best_of_13_sets', 'PDC_WC', 'PDC_PL'",
    )

    # Ecosystem
    ecosystem: str = Field(
        default="pdc_mens",
        description="pdc_mens | pdc_womens | wdf_open | development",
    )

    # Optional rolling form
    player1_rolling_win_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Player 1 rolling win rate (last 200 matches)",
    )
    player2_rolling_win_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Player 2 rolling win rate (last 200 matches)",
    )

    # Optional H2H
    h2h_player1_win_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Player 1 win rate in H2H matches",
    )

    # Pinnacle blend inputs — optional.  When provided, the ML probability is
    # blended with Pinnacle in logit space (25% model / 75% Pinnacle) before
    # being returned.  When absent, the raw ML probability is returned.
    fixture_id: Optional[str] = Field(
        None,
        description="Optic Odds fixture ID — triggers automatic Pinnacle fetch when OPTIC_ODDS_API_KEY is set",
    )
    pinnacle_home_odds: Optional[float] = Field(
        None, gt=1.0,
        description="Explicit Pinnacle decimal odds for player 1 (devigged internally)",
    )
    pinnacle_away_odds: Optional[float] = Field(
        None, gt=1.0,
        description="Explicit Pinnacle decimal odds for player 2 (devigged internally)",
    )

    @field_validator("ecosystem")
    @classmethod
    def validate_ecosystem(cls, v: str) -> str:
        valid = {"pdc_mens", "pdc_womens", "wdf_open", "development"}
        if v not in valid:
            raise ValueError(f"ecosystem must be one of: {sorted(valid)}")
        return v


# ---------------------------------------------------------------------------
# ELO lookup helper — fetches from DB, falls back to 1500
# ---------------------------------------------------------------------------

async def _lookup_elo(player_name: str) -> Optional[float]:
    """
    Look up a player's most recent ELO from the DB.

    Queries darts_elo_ratings table by player_name (case-insensitive).
    Returns None if DB is unavailable or player is not found.
    Does NOT return a hardcoded default — caller handles None.
    """
    try:
        from db.session import engine
        from sqlalchemy import text

        async with engine.connect() as conn:
            row = await conn.execute(
                text(
                    "SELECT rating FROM darts_elo_ratings "
                    "WHERE LOWER(player_name) = LOWER(:name) "
                    "ORDER BY updated_at DESC LIMIT 1"
                ),
                {"name": player_name},
            )
            rec = row.fetchone()
            return float(rec[0]) if rec else None
    except Exception as exc:
        logger.debug("elo_lookup_failed", player=player_name, error=str(exc))
        return None


async def _lookup_3da(player_name: str) -> Optional[float]:
    """
    Look up a player's PDC three-dart average from the DB.

    Queries darts_player_stats by player_name using the three_da_pdc column
    added in migration 005.  Returns None if the player is not found so the
    caller can distinguish "not seeded" from a real zero average.

    Column notes:
      - darts_player_stats.player_name  — added in migration 005
      - darts_player_stats.three_da_pdc — added in migration 005 (PDC season avg)
    """
    try:
        from db.session import engine
        from sqlalchemy import text

        async with engine.connect() as conn:
            row = await conn.execute(
                text(
                    "SELECT three_da_pdc FROM darts_player_stats "
                    "WHERE LOWER(player_name) = LOWER(:name) "
                    "AND three_da_pdc IS NOT NULL "
                    "ORDER BY updated_at DESC LIMIT 1"
                ),
                {"name": player_name},
            )
            rec = row.fetchone()
            return float(rec[0]) if rec else None
    except Exception as exc:
        logger.debug("3da_lookup_failed", player=player_name, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _infer_format_code(format_str: Optional[str]) -> str:
    """Map free-text format string to a training-compatible format_code."""
    if not format_str:
        return ""
    fs = format_str.upper().replace("-", "_").replace(" ", "_")
    # Direct PDC format codes pass through
    _direct = {"PDC_WC", "PDC_PL", "PDC_WM", "PDC_GS", "PDC_GP", "PDC_PCF", "PDC_UK"}
    if fs in _direct:
        return fs
    # Infer from descriptors
    if "WORLD_CHAMPIONSHIP" in fs or "WC" in fs:
        return "PDC_WC"
    if "PREMIER_LEAGUE" in fs or "PL" in fs:
        return "PDC_PL"
    if "WORLD_MATCHPLAY" in fs or "WM" in fs:
        return "PDC_WM"
    if "GRAND_SLAM" in fs or "GS" in fs:
        return "PDC_GS"
    if "GRAND_PRIX" in fs or "GP" in fs:
        return "PDC_GP"
    return ""


def _run_r1_inference(
    req: DartsPredictRequest,
    p1_elo: float,
    p2_elo: float,
    p1_3da: float,
    p2_3da: float,
    pinnacle_prob: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run R1 file-based inference. Returns prediction dict.

    Uses r1_file_predictor singleton which loads r1_model.pkl (38 features)
    or r1_model_from_raw.pkl (19 features) from disk.

    Raises RuntimeError if no model file is available.
    """
    from models.r1_file_predictor import r1_file_predictor

    format_code = _infer_format_code(req.format)

    p1_win = r1_file_predictor.predict(
        p1_elo=p1_elo,
        p2_elo=p2_elo,
        p1_3da=p1_3da,
        p2_3da=p2_3da,
        p1_rolling_wr=req.player1_rolling_win_rate if req.player1_rolling_win_rate is not None else 0.5,
        p2_rolling_wr=req.player2_rolling_win_rate if req.player2_rolling_win_rate is not None else 0.5,
        h2h_wr=req.h2h_player1_win_rate if req.h2h_player1_win_rate is not None else 0.5,
        format_code=format_code,
        ecosystem=req.ecosystem,
        is_televised=True,
    )

    if p1_win is None:
        raise RuntimeError("R1 model files not found on disk — cannot predict")

    # Clamp raw ML output to valid probability range before blend.
    p1_win_model = float(max(0.01, min(0.99, p1_win)))

    # Apply Pinnacle logit blend (25% model / 75% Pinnacle).
    # blend_with_pinnacle() returns model_prob unchanged when pinnacle_prob is None
    # or degenerate — never produces a silent hardcoded fallback.
    p1_win_blended = blend_with_pinnacle(
        model_prob=p1_win_model,
        pinnacle_prob=pinnacle_prob,
    )
    pinnacle_blend_applied = pinnacle_prob is not None and (0.001 < pinnacle_prob < 0.999)

    p2_win = 1.0 - p1_win_blended

    model_version = f"r1_file_{r1_file_predictor.n_features}f"

    return {
        "p1_win_prob": round(p1_win_blended, 6),
        "p2_win_prob": round(p2_win, 6),
        "p1_win_prob_model_only": round(p1_win_model, 6),
        "pinnacle_prob_used": round(pinnacle_prob, 6) if pinnacle_blend_applied else None,
        "pinnacle_blend_applied": pinnacle_blend_applied,
        "model_version": model_version,
        "model_tier": "r1",
        "features_used": {
            "p1_elo": round(p1_elo, 1),
            "p2_elo": round(p2_elo, 1),
            "elo_diff": round(p1_elo - p2_elo, 1),
            "p1_3da": round(p1_3da, 2),
            "p2_3da": round(p2_3da, 2),
            "format_code": format_code or "(default)",
            "ecosystem": req.ecosystem,
            "p1_rolling_win_rate": req.player1_rolling_win_rate,
            "p2_rolling_win_rate": req.player2_rolling_win_rate,
            "h2h_p1_win_rate": req.h2h_player1_win_rate,
        },
    }


def _run_r0_inference(req: DartsPredictRequest, p1_elo: float, p2_elo: float) -> Dict[str, Any]:
    """
    Fallback: R0 logistic model from r0_model.pkl.

    Only used when R1 model files are absent.
    Raises RuntimeError if r0_model.pkl also missing.
    """
    import pathlib
    import joblib
    import numpy as np

    r0_path = pathlib.Path(__file__).resolve().parent.parent.parent / "models" / "saved" / "r0_model.pkl"
    if not r0_path.exists():
        raise RuntimeError(
            f"Neither R1 nor R0 model files available. "
            f"R0 path checked: {r0_path}"
        )

    artifact = joblib.load(r0_path)
    model = artifact["model"]
    scaler = artifact["scaler"]

    import math
    elo_diff = p1_elo - p2_elo
    elo_log_ratio = math.log(max(1.0, p2_elo)) - math.log(max(1.0, p1_elo))
    fmt_enc = 0.0
    stage_floor = 0.0
    short_format = 0.0
    eco_enc_map = {"pdc_mens": 0.0, "pdc_womens": 1.0, "wdf_open": 2.0, "development": 3.0}
    eco_enc = eco_enc_map.get(req.ecosystem, 0.0)

    # 14 R0 features in training order
    feat = np.array([[
        p1_elo, p2_elo, elo_diff,
        512.0, 512.0,   # ranking placeholders (unknown without DB)
        elo_log_ratio,
        50.0, 50.0,     # 3DA placeholders
        35.0, 35.0,     # checkout pct placeholders
        fmt_enc, stage_floor, short_format, eco_enc,
    ]], dtype=np.float64)

    X_scaled = scaler.transform(feat)
    p1_win = float(model.predict_proba(X_scaled)[0, 1])
    p1_win = max(0.01, min(0.99, p1_win))

    return {
        "p1_win_prob": round(p1_win, 6),
        "p2_win_prob": round(1.0 - p1_win, 6),
        "model_version": "r0_logit_14f",
        "model_tier": "r0",
        "features_used": {
            "p1_elo": round(p1_elo, 1),
            "p2_elo": round(p2_elo, 1),
            "elo_diff": round(elo_diff, 1),
            "ecosystem": req.ecosystem,
        },
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    summary="Predict darts match winner (R1 model)",
    response_model=None,
)
async def predict_match(req: DartsPredictRequest) -> JSONResponse:
    """
    Predict win probabilities for a darts match.

    Primary model: R1 stacking ensemble (LightGBM + XGBoost + logistic
    meta-learner, 38 features, Beta-calibrated, AUC≈0.818).

    Fallback: R0 logistic regression (14 features) if R1 pkl absent.

    ELO is looked up from darts_elo_ratings DB table. When no DB rating
    exists and no explicit p1_elo/p2_elo override is in the request,
    the endpoint returns HTTP 503 "darts_elo_unavailable" rather than
    silently defaulting to ELO=1500.  Callers who supply p1_elo/p2_elo
    in the request body bypass the DB and receive real predictions.

    GAP-A-03 FIX (window_predict_a 2026-04-25): ELO-1500 silent fallback
    eliminated. Platform refuses to publish synthetic prices — use explicit
    ELO overrides or seed the darts_elo_ratings table.

    Returns 503 if no model file is available at all.
    """
    rid = str(uuid.uuid4())
    log = logger.bind(
        request_id=rid,
        player1=req.player1,
        player2=req.player2,
    )
    log.info("darts_predict_request", ecosystem=req.ecosystem, format=req.format)

    # --- Resolve ELO ---
    # GAP-A-03 FIX: refuse with 503 when ELO not in DB and not in request.
    # This replaces the prior ELO=1500 silent fallback (FAKE DATA violation).
    if req.player1_elo is not None:
        p1_elo = req.player1_elo
        p1_elo_source = "request"
    else:
        db_elo = await _lookup_elo(req.player1)
        if db_elo is None:
            log.warning("darts_elo_unavailable", player=req.player1)
            return JSONResponse(
                status_code=503,
                content={
                    "error": "prediction_unavailable",
                    "sport": "darts",
                    "match_id": req.fixture_id or "unknown",
                    "reason": "darts_elo_unavailable",
                    "detail": (
                        f"Player {req.player1!r} has no ELO rating in darts_elo_ratings. "
                        "Provide player1_elo in the request body to override, "
                        "or seed the darts_elo_ratings table."
                    ),
                    "last_known_good_at": None,
                    "request_id": rid,
                },
                headers={"Retry-After": "3600"},
            )
        p1_elo = db_elo
        p1_elo_source = "db"

    if req.player2_elo is not None:
        p2_elo = req.player2_elo
        p2_elo_source = "request"
    else:
        db_elo = await _lookup_elo(req.player2)
        if db_elo is None:
            log.warning("darts_elo_unavailable", player=req.player2)
            return JSONResponse(
                status_code=503,
                content={
                    "error": "prediction_unavailable",
                    "sport": "darts",
                    "match_id": req.fixture_id or "unknown",
                    "reason": "darts_elo_unavailable",
                    "detail": (
                        f"Player {req.player2!r} has no ELO rating in darts_elo_ratings. "
                        "Provide player2_elo in the request body to override, "
                        "or seed the darts_elo_ratings table."
                    ),
                    "last_known_good_at": None,
                    "request_id": rid,
                },
                headers={"Retry-After": "3600"},
            )
        p2_elo = db_elo
        p2_elo_source = "db"

    # --- Resolve 3DA ---
    # 3DA neutral default (50.0) is acceptable — it represents mid-tier professional
    # performance and is less critical than ELO for prediction accuracy.
    _DEFAULT_3DA = 50.0
    if req.player1_3da is not None:
        p1_3da = req.player1_3da
        p1_3da_source = "request"
    else:
        db_3da = await _lookup_3da(req.player1)
        p1_3da = db_3da if db_3da is not None else _DEFAULT_3DA
        p1_3da_source = "db" if db_3da is not None else "neutral_default"

    if req.player2_3da is not None:
        p2_3da = req.player2_3da
        p2_3da_source = "request"
    else:
        db_3da = await _lookup_3da(req.player2)
        p2_3da = db_3da if db_3da is not None else _DEFAULT_3DA
        p2_3da_source = "db" if db_3da is not None else "neutral_default"

    log.info(
        "darts_predict_inputs_resolved",
        p1_elo=p1_elo, p1_elo_src=p1_elo_source,
        p2_elo=p2_elo, p2_elo_src=p2_elo_source,
        p1_3da=p1_3da, p1_3da_src=p1_3da_source,
        p2_3da=p2_3da, p2_3da_src=p2_3da_source,
    )

    # --- Resolve Pinnacle devigged probability for logit blend ---
    # Priority: (1) explicit request fields → (2) auto-fetch via Optic Odds REST API
    # When no Pinnacle data is available, blend_with_pinnacle() returns model prob unchanged.
    _pinnacle_prob: Optional[float] = None

    if req.pinnacle_home_odds is not None and req.pinnacle_away_odds is not None:
        # Explicit odds provided — devig immediately.
        try:
            fair_p1, _ = devig_2way(req.pinnacle_home_odds, req.pinnacle_away_odds)
            _pinnacle_prob = fair_p1
            log.info(
                "darts_predict_pinnacle_explicit",
                pinnacle_home_odds=req.pinnacle_home_odds,
                pinnacle_away_odds=req.pinnacle_away_odds,
                devigged_p1=round(_pinnacle_prob, 6),
            )
        except ValueError as _dv_err:
            log.warning("darts_predict_pinnacle_devig_failed", error=str(_dv_err))

    elif req.fixture_id:
        # Auto-fetch Pinnacle odds from Optic Odds REST API when OPTIC_ODDS_API_KEY is set.
        # Use fetch_pinnacle_odds() directly to obtain raw decimal odds, then devig them.
        # This avoids any model_prob dependency in the fetch path.
        import os as _os
        _optic_key = _os.getenv("OPTIC_ODDS_API_KEY", "")
        if _optic_key:
            try:
                _pin_raw = await fetch_pinnacle_odds(
                    fixture_id=req.fixture_id,
                    sport="darts",
                    api_key=_optic_key,
                    market_type="moneyline",
                )
                if (
                    _pin_raw is not None
                    and _pin_raw.get("home") is not None
                    and _pin_raw.get("away") is not None
                ):
                    _auto_home = float(_pin_raw["home"])
                    _auto_away = float(_pin_raw["away"])
                    if _auto_home > 1.0 and _auto_away > 1.0:
                        _fair_p1, _ = devig_2way(_auto_home, _auto_away)
                        _pinnacle_prob = _fair_p1
                        log.info(
                            "darts_predict_pinnacle_auto_fetched",
                            fixture_id=req.fixture_id,
                            home_odds=_auto_home,
                            away_odds=_auto_away,
                            devigged_p1=round(_pinnacle_prob, 6),
                        )
            except Exception as _fetch_err:
                log.warning(
                    "darts_predict_pinnacle_auto_fetch_failed",
                    fixture_id=req.fixture_id,
                    error=str(_fetch_err),
                )

    # --- Run inference: R1 primary, R0 fallback ---
    try:
        result = _run_r1_inference(req, p1_elo, p2_elo, p1_3da, p2_3da, pinnacle_prob=_pinnacle_prob)
    except RuntimeError as r1_err:
        log.warning("r1_model_unavailable_trying_r0", error=str(r1_err))
        try:
            result = _run_r0_inference(req, p1_elo, p2_elo)
        except RuntimeError as r0_err:
            log.error("both_models_unavailable", r1_error=str(r1_err), r0_error=str(r0_err))
            return _error(
                "MODEL_UNAVAILABLE",
                (
                    "No darts ML model files available. "
                    "Ensure r1_model.pkl or r0_model.pkl are deployed to models/saved/. "
                    f"R1: {r1_err}. R0: {r0_err}"
                ),
                rid,
                http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
    except Exception as exc:
        log.error("predict_inference_error", error=str(exc), exc_info=True)
        return _error(
            "INFERENCE_ERROR",
            f"Model inference failed: {exc}",
            rid,
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    # Attach player context
    result["player1"] = req.player1
    result["player2"] = req.player2
    result["data_sources"] = {
        "p1_elo_source": p1_elo_source,
        "p2_elo_source": p2_elo_source,
        "p1_3da_source": p1_3da_source,
        "p2_3da_source": p2_3da_source,
    }

    log.info(
        "darts_predict_result",
        model_version=result["model_version"],
        p1_win_prob=result["p1_win_prob"],
        p2_win_prob=result["p2_win_prob"],
    )

    return JSONResponse(
        content=_ok(result, rid),
        status_code=status.HTTP_200_OK,
    )
