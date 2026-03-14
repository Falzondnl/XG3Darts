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
