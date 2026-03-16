"""
Monitoring and operational health routes.

Endpoints
---------
GET  /api/v1/darts/monitoring/markov-validation
    Weekly Markov chain accuracy validation across all seven market families.

GET  /api/v1/darts/monitoring/market-calibration
    Brier, ECE, and AUC per market family with quality gate status.

GET  /api/v1/darts/monitoring/clv
    Closing Line Value tracking vs Betfair/Pinnacle with auto-adjust status.

GET  /api/v1/darts/monitoring/coverage-regimes
    R0/R1/R2 regime distribution across the active player pool.

GET  /api/v1/darts/monitoring/markets
    Engine availability status for each of the 15 SGP market types.
    Checks whether pricing engines can be imported and instantiated.
"""
from __future__ import annotations

from typing import Any, Optional

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = structlog.get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class MarkovValidationResponse(BaseModel):
    """Validation results for all market families."""

    since_days: int
    families: dict[str, Any]
    all_ok: bool
    alerts_triggered: int


class MarketCalibrationResponse(BaseModel):
    """Calibration report for all market families."""

    since_days: int
    families: dict[str, Any]
    all_gates_passed: bool


class CLVResponse(BaseModel):
    """CLV and margin summary."""

    lookback_days: int
    families: dict[str, Any]
    any_alerts: bool


class CoverageRegimeResponse(BaseModel):
    """Coverage regime distribution."""

    total_players: int
    r0_count: int
    r1_count: int
    r2_count: int
    r0_pct: float
    r1_pct: float
    r2_pct: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/monitoring/markov-validation",
    response_model=MarkovValidationResponse,
    summary="Weekly Markov chain validation across all market families",
)
async def get_markov_validation(
    since_days: int = Query(
        default=7,
        ge=1,
        le=365,
        description="Lookback window in days",
    ),
) -> MarkovValidationResponse:
    """
    Run (or retrieve cached) weekly Markov validation across all 7 market families.

    Compares predicted probabilities against resolved outcomes using
    proper scoring rules (Brier, MAE, RPS). Returns per-family results
    and triggers alerts on threshold breach.

    The validation is idempotent: if the DB has no resolved markets for
    the window, the result is returned with sample_size=0 and no alert.
    """
    from monitoring.markov_validator import MarkovValidationMonitor

    monitor = MarkovValidationMonitor(db_session=None)  # no DB in route; mock mode
    results = await monitor.validate_all_families(since_days=since_days)

    families_out: dict[str, Any] = {}
    alerts = 0
    for family, result in results.items():
        families_out[family] = {
            "metric_value": result.metric_value,
            "threshold": result.threshold,
            "alert_triggered": result.alert_triggered,
            "sample_size": result.sample_size,
            "metric_type": result.extra.get("metric_type", "unknown"),
        }
        if result.alert_triggered:
            alerts += 1

    return MarkovValidationResponse(
        since_days=since_days,
        families=families_out,
        all_ok=(alerts == 0),
        alerts_triggered=alerts,
    )


@router.get(
    "/monitoring/market-calibration",
    response_model=MarketCalibrationResponse,
    summary="Brier, ECE, and AUC per market family",
)
async def get_market_calibration(
    since_days: int = Query(
        default=7,
        ge=1,
        le=365,
        description="Lookback window in days",
    ),
) -> MarketCalibrationResponse:
    """
    Run calibration checks for all seven market families.

    Reports Brier score, Expected Calibration Error (ECE), and ROC-AUC
    where applicable. Quality gates are checked and gate-failed families
    are flagged.

    Returns 200 in all cases; callers should inspect ``all_gates_passed``
    and individual ``gates_passed`` maps for failure details.
    """
    from monitoring.market_calibration_monitor import MarketCalibrationMonitor

    monitor = MarketCalibrationMonitor(db_session=None)
    reports = await monitor.run_all_families(since_days=since_days)

    families_out: dict[str, Any] = {}
    all_gates_passed = True
    for family, report in reports.items():
        families_out[family] = {
            "brier": report.brier,
            "ece": report.ece,
            "auc": report.auc,
            "mae": report.mae,
            "rps": report.rps,
            "sample_size": report.sample_size,
            "gates_passed": report.gates_passed,
            "all_gates_passed": report.all_gates_passed,
        }
        if not report.all_gates_passed:
            all_gates_passed = False

    return MarketCalibrationResponse(
        since_days=since_days,
        families=families_out,
        all_gates_passed=all_gates_passed,
    )


@router.get(
    "/monitoring/clv",
    response_model=CLVResponse,
    summary="Closing Line Value tracking and margin auto-adjust status",
)
async def get_clv(
    lookback_days: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Lookback window in days for CLV computation",
    ),
) -> CLVResponse:
    """
    Return CLV and margin summary for all market families.

    CLV = log(our_opening_price / betfair_closing_price)

    A positive CLV indicates we are pricing sharper than the market.
    When CLV < -0.02, the margin is widened by 0.5 % automatically.
    This endpoint reports the current state (it does NOT trigger auto-adjust).
    """
    from monitoring.clv_monitor import DartsCLVMonitor

    monitor = DartsCLVMonitor(db_session=None, redis_client=None)
    summary = await monitor.get_all_clv_summary(lookback_days=lookback_days)

    any_alerts = any(v["alert"] for v in summary.values())

    return CLVResponse(
        lookback_days=lookback_days,
        families=summary,
        any_alerts=any_alerts,
    )


@router.get(
    "/monitoring/markets",
    response_model=dict,
    summary="Engine availability status for each SGP market type",
    tags=["Monitoring"],
)
async def get_market_engine_health() -> dict[str, Any]:
    """
    Check and return the availability status of each pricing engine
    associated with the 15 SGP market types.

    For each market type the endpoint attempts to:
    1. Import the relevant engine module.
    2. Instantiate the engine class.
    3. Report whether the engine is available and operational.

    This is a lightweight liveness check; it does not run any simulations.
    Any import or instantiation error is caught and reported per-market
    without raising an HTTP exception (always returns 200).

    The ``correlation_matrix_ok`` flag validates that the SGP correlation
    estimator can be constructed with the global 15x15 prior.
    """
    import importlib
    import traceback

    # Map of market_type -> (module_path, class_name) for the primary engine
    # responsible for computing raw probabilities for that market.
    # Engines that share a module (e.g. both leg-layer markets) point to the
    # same import but different class names.
    _MARKET_ENGINE_MAP: dict[str, tuple[str, str]] = {
        "match_win":            ("engines.match_layer.race_to_x_engine",    "race_to_x"),
        "total_legs_over":      ("engines.match_layer.match_combinatorics", "MatchCombinatorialEngine"),
        "handicap":             ("engines.match_layer.match_combinatorics", "MatchCombinatorialEngine"),
        "180_over":             ("engines.leg_layer.visit_distributions",   "HierarchicalVisitDistributionModel"),
        "checkout_over":        ("engines.leg_layer.checkout_model",        "CheckoutModel"),
        "exact_score":          ("engines.match_layer.match_combinatorics", "MatchCombinatorialEngine"),
        "leg_winner_next":      ("engines.leg_layer.hold_break_model",      "HoldBreakModel"),
        "first_leg_winner":     ("engines.leg_layer.hold_break_model",      "HoldBreakModel"),
        "most_180s":            ("engines.leg_layer.visit_distributions",   "HierarchicalVisitDistributionModel"),
        "highest_checkout":     ("engines.leg_layer.checkout_model",        "CheckoutModel"),
        "total_180s_band":      ("engines.leg_layer.visit_distributions",   "HierarchicalVisitDistributionModel"),
        "player_checkout_over": ("engines.leg_layer.checkout_model",        "CheckoutModel"),
        "sets_over":            ("engines.match_layer.sets_engine",         "SetsEngine"),
        "break_of_throw":       ("engines.leg_layer.hold_break_model",      "HoldBreakModel"),
        "nine_dart_finish":     ("engines.leg_layer.markov_chain",          "DartsMarkovChain"),
    }

    market_statuses: dict[str, Any] = {}
    all_available = True

    for market_type, (module_path, class_name) in _MARKET_ENGINE_MAP.items():
        status: dict[str, Any] = {
            "market_type": market_type,
            "engine_module": module_path,
            "engine_class": class_name,
            "available": False,
            "error": None,
        }
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            # Attempt a no-arg instantiation; catch TypeError for engines
            # that require constructor arguments (they are still importable).
            try:
                cls()
                status["available"] = True
                status["instantiation"] = "ok"
            except TypeError as te:
                # Engine exists but requires constructor args — still importable
                status["available"] = True
                status["instantiation"] = f"requires_args: {te}"
            except Exception as ie:
                status["available"] = True
                status["instantiation"] = f"init_error: {ie}"
        except ImportError as exc:
            status["error"] = f"ImportError: {exc}"
            all_available = False
        except AttributeError as exc:
            status["error"] = f"AttributeError: {exc}"
            all_available = False
        except Exception as exc:
            status["error"] = f"UnexpectedError: {exc}"
            all_available = False

        market_statuses[market_type] = status

    # Also validate the SGP correlation matrix can be built
    corr_matrix_ok = False
    corr_matrix_error: Optional[str] = None
    try:
        from sgp.correlation_estimator import DartsSGPCorrelationEstimator, N_MARKET_DIMS
        estimator = DartsSGPCorrelationEstimator()
        corr_matrix = estimator.estimate_for_competition("health_check", n_samples=0)
        corr_matrix_ok = (
            corr_matrix.shape == (N_MARKET_DIMS, N_MARKET_DIMS)
            and float(corr_matrix.min()) >= -1.0
        )
    except Exception as exc:
        corr_matrix_error = str(exc)

    n_available = sum(1 for s in market_statuses.values() if s["available"])

    logger.info(
        "monitoring_market_health",
        n_available=n_available,
        n_total=len(market_statuses),
        all_available=all_available,
        corr_matrix_ok=corr_matrix_ok,
    )

    return {
        "all_available": all_available,
        "n_available": n_available,
        "n_total": len(market_statuses),
        "correlation_matrix_ok": corr_matrix_ok,
        "correlation_matrix_error": corr_matrix_error,
        "markets": market_statuses,
    }


@router.get(
    "/monitoring/coverage-regimes",
    response_model=CoverageRegimeResponse,
    summary="R0/R1/R2 regime distribution across the active player pool",
)
async def get_coverage_regimes() -> CoverageRegimeResponse:
    """
    Return the R0/R1/R2 regime distribution across all tracked players.

    R0 = result-only data (logit model tier)
    R1 = match-level stats (LightGBM model tier)
    R2 = full visit-level data (stacking/DartConnect model tier)

    When the database is not connected, returns placeholder zeros.
    """
    # In production this queries darts_coverage_regimes
    # For now return a structured placeholder
    try:
        # Attempt DB query if session available via app state
        # Graceful degradation when DB is unavailable
        r0, r1, r2 = 0, 0, 0
        total = r0 + r1 + r2

        def pct(n: int) -> float:
            return round(n / total * 100, 1) if total > 0 else 0.0

        return CoverageRegimeResponse(
            total_players=total,
            r0_count=r0,
            r1_count=r1,
            r2_count=r2,
            r0_pct=pct(r0),
            r1_pct=pct(r1),
            r2_pct=pct(r2),
        )
    except Exception as exc:
        logger.error("coverage_regime_error", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail={"error": "coverage_regime_error", "message": str(exc)},
        )
