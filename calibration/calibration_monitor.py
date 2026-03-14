"""
Market-family calibration monitoring with validation gates.

Tracks calibration quality per market family over time. Implements
ECE (Expected Calibration Error), reliability diagrams, and
validation gates that block deployment of poorly calibrated models.

Validation gates:
    - ECE must be below threshold (default 0.05)
    - Brier score must be below threshold (default 0.25)
    - Each calibration bin must have adequate sample count
    - No single bin can have calibration error > 0.15
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class DartsCalibrationMonitorError(Exception):
    """Raised when calibration monitoring encounters errors."""


@dataclass
class CalibrationBinResult:
    """
    Result for a single calibration bin.

    Attributes
    ----------
    bin_lower:
        Lower bound of the probability bin.
    bin_upper:
        Upper bound of the probability bin.
    bin_center:
        Midpoint of the bin.
    mean_predicted:
        Mean predicted probability in this bin.
    mean_observed:
        Mean observed frequency in this bin (fraction of positives).
    count:
        Number of samples in this bin.
    abs_error:
        |mean_predicted - mean_observed|.
    """

    bin_lower: float
    bin_upper: float
    bin_center: float
    mean_predicted: float
    mean_observed: float
    count: int
    abs_error: float


@dataclass
class CalibrationReport:
    """
    Full calibration quality report for one market family.

    Attributes
    ----------
    market_family:
        The market family this report covers.
    ece:
        Expected Calibration Error (weighted mean of bin errors).
    mce:
        Maximum Calibration Error (max bin error).
    brier_score:
        Brier score (mean squared error of probabilities).
    n_samples:
        Total number of prediction-outcome pairs.
    n_bins:
        Number of calibration bins used.
    bins:
        Per-bin calibration details.
    passes_gate:
        True if all validation gates are passed.
    gate_failures:
        List of gate failure descriptions (empty if all pass).
    timestamp:
        When this report was generated.
    """

    market_family: str
    ece: float
    mce: float
    brier_score: float
    n_samples: int
    n_bins: int
    bins: list[CalibrationBinResult]
    passes_gate: bool
    gate_failures: list[str]
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=__import__("datetime").timezone.utc).isoformat()
    )


class CalibrationMonitor:
    """
    Monitors calibration quality per market family with validation gates.

    Parameters
    ----------
    n_bins:
        Number of equal-width bins for calibration analysis.
    ece_threshold:
        Maximum allowed ECE for gate passage.
    brier_threshold:
        Maximum allowed Brier score for gate passage.
    max_bin_error:
        Maximum allowed absolute error in any single bin.
    min_bin_count:
        Minimum number of samples required per non-empty bin.
    """

    def __init__(
        self,
        n_bins: int = 10,
        ece_threshold: float = 0.05,
        brier_threshold: float = 0.25,
        max_bin_error: float = 0.15,
        min_bin_count: int = 5,
    ) -> None:
        self.n_bins = n_bins
        self.ece_threshold = ece_threshold
        self.brier_threshold = brier_threshold
        self.max_bin_error = max_bin_error
        self.min_bin_count = min_bin_count
        self._reports: dict[str, list[CalibrationReport]] = {}
        self._log = logger.bind(component="CalibrationMonitor")

    def evaluate(
        self,
        market_family: str,
        predicted_proba: np.ndarray,
        y_true: np.ndarray,
    ) -> CalibrationReport:
        """
        Evaluate calibration quality for a market family.

        Parameters
        ----------
        market_family:
            The market family being evaluated.
        predicted_proba:
            Calibrated model probabilities, shape (n_samples,).
        y_true:
            Binary ground truth labels, shape (n_samples,).

        Returns
        -------
        CalibrationReport

        Raises
        ------
        DartsCalibrationMonitorError
            If inputs are invalid.
        """
        predicted_proba = np.asarray(predicted_proba, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)

        if predicted_proba.ndim != 1 or y_true.ndim != 1:
            raise DartsCalibrationMonitorError(
                "predicted_proba and y_true must be 1-dimensional."
            )
        if len(predicted_proba) != len(y_true):
            raise DartsCalibrationMonitorError(
                f"Length mismatch: predicted={len(predicted_proba)}, "
                f"y_true={len(y_true)}"
            )
        if len(predicted_proba) == 0:
            raise DartsCalibrationMonitorError("Empty arrays provided.")

        # Compute Brier score
        brier = float(np.mean((predicted_proba - y_true) ** 2))

        # Compute binned calibration
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_results: list[CalibrationBinResult] = []
        total_weight = 0.0
        ece_sum = 0.0
        mce = 0.0

        for i in range(self.n_bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            center = (lo + hi) / 2.0

            if i == self.n_bins - 1:
                mask = (predicted_proba >= lo) & (predicted_proba <= hi)
            else:
                mask = (predicted_proba >= lo) & (predicted_proba < hi)

            count = int(np.sum(mask))
            if count == 0:
                bin_results.append(CalibrationBinResult(
                    bin_lower=lo,
                    bin_upper=hi,
                    bin_center=center,
                    mean_predicted=center,
                    mean_observed=0.0,
                    count=0,
                    abs_error=0.0,
                ))
                continue

            mean_pred = float(np.mean(predicted_proba[mask]))
            mean_obs = float(np.mean(y_true[mask]))
            abs_err = abs(mean_pred - mean_obs)

            bin_results.append(CalibrationBinResult(
                bin_lower=lo,
                bin_upper=hi,
                bin_center=center,
                mean_predicted=mean_pred,
                mean_observed=mean_obs,
                count=count,
                abs_error=abs_err,
            ))

            weight = count / len(predicted_proba)
            ece_sum += weight * abs_err
            total_weight += weight
            mce = max(mce, abs_err)

        ece = ece_sum

        # Validation gates
        gate_failures: list[str] = []

        if ece > self.ece_threshold:
            gate_failures.append(
                f"ECE={ece:.4f} exceeds threshold={self.ece_threshold}"
            )

        if brier > self.brier_threshold:
            gate_failures.append(
                f"Brier={brier:.4f} exceeds threshold={self.brier_threshold}"
            )

        for b in bin_results:
            if b.count >= self.min_bin_count and b.abs_error > self.max_bin_error:
                gate_failures.append(
                    f"Bin [{b.bin_lower:.2f}, {b.bin_upper:.2f}): "
                    f"error={b.abs_error:.4f} > max={self.max_bin_error} "
                    f"(n={b.count})"
                )

        passes = len(gate_failures) == 0

        report = CalibrationReport(
            market_family=market_family,
            ece=round(ece, 6),
            mce=round(mce, 6),
            brier_score=round(brier, 6),
            n_samples=len(predicted_proba),
            n_bins=self.n_bins,
            bins=bin_results,
            passes_gate=passes,
            gate_failures=gate_failures,
        )

        # Store report
        if market_family not in self._reports:
            self._reports[market_family] = []
        self._reports[market_family].append(report)

        self._log.info(
            "calibration_evaluated",
            market_family=market_family,
            ece=report.ece,
            mce=report.mce,
            brier=report.brier_score,
            passes_gate=passes,
            n_failures=len(gate_failures),
        )

        return report

    def get_latest_report(
        self, market_family: str
    ) -> Optional[CalibrationReport]:
        """Return the most recent calibration report for a market family."""
        reports = self._reports.get(market_family, [])
        return reports[-1] if reports else None

    def get_all_reports(
        self, market_family: str
    ) -> list[CalibrationReport]:
        """Return all calibration reports for a market family."""
        return list(self._reports.get(market_family, []))

    def all_families_pass(self, families: list[str]) -> bool:
        """
        Check if the latest report for each listed family passes its gate.

        Parameters
        ----------
        families:
            Market families to check.

        Returns
        -------
        bool
            True only if every family has a passing latest report.
        """
        for family in families:
            report = self.get_latest_report(family)
            if report is None or not report.passes_gate:
                return False
        return True
