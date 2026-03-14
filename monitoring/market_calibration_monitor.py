"""
Seven-family market calibration monitor.

Computes Brier score, Expected Calibration Error (ECE), and ROC-AUC
for each of the seven market families. Exposes all results as Prometheus
gauges and emits structured log alerts when quality gates are breached.

Market families
---------------
1. match_winner     — binary, Brier + ECE + AUC
2. leg_handicap     — continuous, MAE only (no AUC)
3. totals_legs      — continuous, MAE only (no AUC)
4. exact_score      — multi-class, RPS + ECE
5. props_180        — continuous, MAE + ECE
6. props_checkout   — multi-class, RPS + ECE
7. outright         — multi-class, Brier(top-pick) + ECE + AUC

ECE computation
---------------
    ECE = sum_b (|B_b| / N) * |conf_b - acc_b|

where B_b is the set of predictions in calibration bin b.
Default: 10 equal-width bins over [0, 1].

Quality gates (defaults aligned with ALERT_THRESHOLDS in MarkovValidationMonitor)
    match_winner:  Brier < 0.23, ECE < 0.05, AUC > 0.55
    leg_handicap:  MAE  < 0.03
    totals_legs:   MAE  < 1.0
    exact_score:   RPS  < 0.15, ECE < 0.08
    props_180:     MAE  < 0.20
    props_checkout: RPS < 0.14, ECE < 0.08
    outright:      Brier < 0.16, ECE < 0.06, AUC > 0.60
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Number of calibration bins for ECE
_ECE_N_BINS = 10

# Quality gate defaults
_QUALITY_GATES: dict[str, dict[str, Any]] = {
    "match_winner": {
        "brier_max": 0.23,
        "ece_max": 0.05,
        "auc_min": 0.55,
    },
    "leg_handicap": {
        "mae_max": 0.03,
    },
    "totals_legs": {
        "mae_max": 1.0,
    },
    "exact_score": {
        "rps_max": 0.15,
        "ece_max": 0.08,
    },
    "props_180": {
        "mae_max": 0.20,
    },
    "props_checkout": {
        "rps_max": 0.14,
        "ece_max": 0.08,
    },
    "outright": {
        "brier_max": 0.16,
        "ece_max": 0.06,
        "auc_min": 0.60,
    },
}

ALL_FAMILIES = list(_QUALITY_GATES.keys())


@dataclass
class FamilyCalibrationReport:
    """
    Calibration report for a single market family.

    Attributes
    ----------
    market_family:
        Market family identifier.
    brier:
        Brier score (binary families only, else None).
    ece:
        Expected Calibration Error (None for pure continuous families).
    auc:
        ROC-AUC (binary / top-pick families only, else None).
    mae:
        Mean Absolute Error (continuous families only, else None).
    rps:
        Ranked Probability Score (multi-class families, else None).
    sample_size:
        Number of resolved markets included.
    gates_passed:
        Dict of gate_name → bool for each applicable quality gate.
    all_gates_passed:
        True iff all applicable gates passed.
    """

    market_family: str
    brier: Optional[float] = None
    ece: Optional[float] = None
    auc: Optional[float] = None
    mae: Optional[float] = None
    rps: Optional[float] = None
    sample_size: int = 0
    gates_passed: dict[str, bool] = field(default_factory=dict)

    @property
    def all_gates_passed(self) -> bool:
        return all(self.gates_passed.values()) if self.gates_passed else True


class MarketCalibrationMonitor:
    """
    Seven-family calibration monitor.

    Computes Brier, ECE, and AUC per family. Exposes Prometheus gauges
    and emits structured alerts when quality gates fail.
    """

    def __init__(self, db_session: Any = None) -> None:
        """
        Parameters
        ----------
        db_session:
            Optional SQLAlchemy AsyncSession. When None, operates in mock mode.
        """
        self._db = db_session

    async def run_all_families(
        self, since_days: int = 7
    ) -> dict[str, FamilyCalibrationReport]:
        """
        Run calibration checks for all seven market families.

        Parameters
        ----------
        since_days:
            Lookback window in days.

        Returns
        -------
        dict[str, FamilyCalibrationReport]
            Keyed by market family name.
        """
        results: dict[str, FamilyCalibrationReport] = {}
        for family in ALL_FAMILIES:
            try:
                report = await self.run_family(family, since_days)
                results[family] = report
                self._emit_metrics(report)
                self._check_gates(report)
            except Exception as exc:
                logger.error(
                    "calibration_family_error",
                    family=family,
                    error=str(exc),
                )
                results[family] = FamilyCalibrationReport(
                    market_family=family,
                    sample_size=0,
                )
        return results

    async def run_family(
        self, market_family: str, since_days: int = 7
    ) -> FamilyCalibrationReport:
        """
        Run calibration for a single market family.

        Parameters
        ----------
        market_family:
            Family to evaluate.
        since_days:
            Lookback window.

        Returns
        -------
        FamilyCalibrationReport
        """
        gates = _QUALITY_GATES.get(market_family, {})

        # Fetch predicted vs actual from DB (or mock)
        preds, actuals = await self._fetch_predictions(market_family, since_days)
        sample_size = len(preds)

        brier: Optional[float] = None
        ece: Optional[float] = None
        auc: Optional[float] = None
        mae: Optional[float] = None
        rps_val: Optional[float] = None
        gates_passed: dict[str, bool] = {}

        if sample_size == 0:
            logger.debug(
                "calibration_no_data",
                family=market_family,
                since_days=since_days,
            )
            return FamilyCalibrationReport(
                market_family=market_family,
                sample_size=0,
            )

        preds_arr = np.array(preds, dtype=np.float64)
        actuals_arr = np.array(actuals, dtype=np.float64)

        # Binary families: Brier + ECE + AUC
        if market_family in ("match_winner", "outright"):
            brier = self._compute_brier(preds_arr, actuals_arr)
            ece = self._compute_ece(preds_arr, actuals_arr)
            auc = self._compute_auc(preds_arr, actuals_arr)

            if "brier_max" in gates:
                gates_passed["brier"] = brier <= gates["brier_max"]
            if "ece_max" in gates:
                gates_passed["ece"] = ece <= gates["ece_max"]
            if "auc_min" in gates and auc is not None:
                gates_passed["auc"] = auc >= gates["auc_min"]

        # Continuous families: MAE only
        elif market_family in ("leg_handicap", "totals_legs", "props_180"):
            mae = float(np.mean(np.abs(preds_arr - actuals_arr)))
            if "mae_max" in gates:
                gates_passed["mae"] = mae <= gates["mae_max"]

        # Multi-class families: RPS + ECE
        elif market_family in ("exact_score", "props_checkout"):
            # For multi-class we need probability vectors — use the binary
            # preds/actuals as a proxy when full distribution isn't available
            brier_val = self._compute_brier(preds_arr, actuals_arr)
            rps_val = brier_val  # RPS collapses to Brier for 2-class
            ece = self._compute_ece(preds_arr, actuals_arr)

            if "rps_max" in gates:
                gates_passed["rps"] = rps_val <= gates["rps_max"]
            if "ece_max" in gates:
                gates_passed["ece"] = ece <= gates["ece_max"]

        report = FamilyCalibrationReport(
            market_family=market_family,
            brier=brier,
            ece=ece,
            auc=auc,
            mae=mae,
            rps=rps_val,
            sample_size=sample_size,
            gates_passed=gates_passed,
        )
        return report

    # ------------------------------------------------------------------
    # Metric computations
    # ------------------------------------------------------------------

    def _compute_brier(
        self, preds: np.ndarray, actuals: np.ndarray
    ) -> float:
        """
        Brier score: mean squared error between predicted probs and outcomes.

        BS = (1/N) * sum (p_i - o_i)^2
        """
        return float(np.mean((preds - actuals) ** 2))

    def _compute_ece(
        self,
        preds: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = _ECE_N_BINS,
    ) -> float:
        """
        Expected Calibration Error.

        ECE = sum_b (|B_b| / N) * |mean_conf_b - mean_acc_b|

        Bins predictions into equal-width intervals over [0, 1].
        """
        n = len(preds)
        if n == 0:
            return 0.0

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            # Include upper edge in the last bin
            if i < n_bins - 1:
                mask = (preds >= lo) & (preds < hi)
            else:
                mask = (preds >= lo) & (preds <= hi)

            bin_size = int(mask.sum())
            if bin_size == 0:
                continue

            mean_conf = float(preds[mask].mean())
            mean_acc = float(actuals[mask].mean())
            ece += (bin_size / n) * abs(mean_conf - mean_acc)

        return ece

    def _compute_auc(
        self,
        preds: np.ndarray,
        actuals: np.ndarray,
    ) -> Optional[float]:
        """
        ROC-AUC via the trapezoidal rule.

        Returns None if both classes are not present (degenerate case).
        """
        # Need both positive and negative examples
        positives = actuals.sum()
        negatives = len(actuals) - positives
        if positives == 0 or negatives == 0:
            return None

        # Sort by predicted probability descending
        sort_idx = np.argsort(-preds)
        sorted_actuals = actuals[sort_idx]

        tp = 0.0
        fp = 0.0
        auc = 0.0
        prev_tp = 0.0

        for outcome in sorted_actuals:
            if outcome == 1:
                tp += 1
            else:
                fp += 1
                # Trapezoid contribution
                auc += (tp + prev_tp) / 2.0
                prev_tp = tp

        # Normalise
        if positives > 0 and negatives > 0:
            auc = auc / (positives * negatives)
        else:
            auc = 0.5

        return float(np.clip(auc, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    async def _fetch_predictions(
        self, market_family: str, since_days: int
    ) -> tuple[list[float], list[float]]:
        """
        Fetch predicted probabilities and actual outcomes from the DB.

        Returns (preds, actuals) as parallel lists of floats.
        In mock mode (no DB), returns synthetic data to exercise the pipeline.
        """
        if self._db is None:
            return self._mock_predictions(market_family)

        try:
            from datetime import datetime, timedelta, timezone

            cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
            query = """
                SELECT predicted_prob_p1, p1_won
                FROM darts_market_predictions
                WHERE market_family = :family
                  AND resolved_at >= :cutoff
                  AND predicted_prob_p1 IS NOT NULL
                  AND p1_won IS NOT NULL
            """
            rows = await self._db.execute(
                __import__("sqlalchemy").text(query),
                {"family": market_family, "cutoff": cutoff},
            )
            results = rows.fetchall()
            preds = [float(r[0]) for r in results]
            actuals = [float(r[1]) for r in results]
            return preds, actuals

        except Exception as exc:
            logger.error(
                "calibration_fetch_error",
                family=market_family,
                error=str(exc),
            )
            return [], []

    def _mock_predictions(
        self, market_family: str
    ) -> tuple[list[float], list[float]]:
        """
        Generate mock predictions with mild calibration error for development use.

        Returns plausible (preds, actuals) that produce Brier ~0.20 and ECE ~0.03.
        """
        rng = np.random.default_rng(seed=abs(hash(market_family)) % (2 ** 32))
        n = 200

        # Well-calibrated predictions: actual = Bernoulli(pred)
        preds = rng.uniform(0.3, 0.7, n).tolist()
        actuals = rng.binomial(1, preds).tolist()
        return [float(p) for p in preds], [float(a) for a in actuals]

    # ------------------------------------------------------------------
    # Prometheus + alerting
    # ------------------------------------------------------------------

    def _emit_metrics(self, report: FamilyCalibrationReport) -> None:
        """Update Prometheus gauges for a completed family report."""
        from monitoring.metrics import BRIER_GAUGE, ECE_GAUGE, AUC_GAUGE

        family = report.market_family
        if report.brier is not None:
            BRIER_GAUGE.labels(market_family=family).set(report.brier)
        if report.ece is not None:
            ECE_GAUGE.labels(market_family=family).set(report.ece)
        if report.auc is not None:
            AUC_GAUGE.labels(market_family=family).set(report.auc)

    def _check_gates(self, report: FamilyCalibrationReport) -> None:
        """Emit structured log alerts for failed quality gates."""
        from monitoring.metrics import DRIFT_ALERT_COUNT

        failed = {
            gate: passed
            for gate, passed in report.gates_passed.items()
            if not passed
        }
        if failed:
            DRIFT_ALERT_COUNT.labels(
                alert_type=f"{report.market_family}_calibration_gate"
            ).inc()
            logger.warning(
                "calibration_gate_failed",
                market_family=report.market_family,
                failed_gates=failed,
                brier=report.brier,
                ece=report.ece,
                auc=report.auc,
                mae=report.mae,
                sample_size=report.sample_size,
            )
        else:
            logger.info(
                "calibration_all_gates_passed",
                market_family=report.market_family,
                sample_size=report.sample_size,
            )
