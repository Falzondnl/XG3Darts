"""
Brier drift detection and Population Stability Index (PSI).

Detects two classes of model degradation:
  1. Brier score drift — the model's predictive accuracy is deteriorating.
  2. Feature distribution drift — the input distribution has shifted
     (PSI > 0.20 is the industry-standard alert threshold).

Brier score monitoring
----------------------
The rolling Brier score is compared against BRIER_DRIFT_THRESHOLD = 0.28.
If it exceeds the threshold, an alert is emitted and the Prometheus
drift counter is incremented.

Population Stability Index (PSI)
---------------------------------
PSI quantifies how much a feature distribution has changed between a
reference period and a current period:

    PSI = sum_i (actual_i - expected_i) * log(actual_i / expected_i)

where the sum is over n_bins histogram buckets.

PSI interpretation:
    < 0.10  : no significant shift
    0.10–0.20 : moderate shift, monitor
    > 0.20  : significant shift, alert and retrain
"""
from __future__ import annotations

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Brier score threshold above which drift is declared
_BRIER_DRIFT_THRESHOLD = 0.28

# PSI thresholds
_PSI_MONITOR_THRESHOLD = 0.10
_PSI_ALERT_THRESHOLD = 0.20

# Small constant to avoid log(0)
_PSI_EPSILON = 1e-6


class MarketDriftDetector:
    """
    Monitors Brier score drift and feature distribution drift (PSI).

    Integrates with Prometheus metrics to expose drift signals to dashboards.
    """

    BRIER_DRIFT_THRESHOLD: float = _BRIER_DRIFT_THRESHOLD

    def __init__(self) -> None:
        # History of recent Brier scores per market family
        self._brier_history: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Brier drift
    # ------------------------------------------------------------------

    def detect_brier_drift(self, recent_brier: float) -> bool:
        """
        Return True if Brier drift is detected (recent_brier > threshold).

        Parameters
        ----------
        recent_brier:
            The most recently computed Brier score.

        Returns
        -------
        bool
            True if drift is detected and an alert should be raised.
        """
        is_drift = recent_brier > self.BRIER_DRIFT_THRESHOLD

        if is_drift:
            from monitoring.metrics import DRIFT_ALERT_COUNT
            DRIFT_ALERT_COUNT.labels(alert_type="brier_drift").inc()
            logger.warning(
                "brier_drift_detected",
                recent_brier=round(recent_brier, 6),
                threshold=self.BRIER_DRIFT_THRESHOLD,
            )
        else:
            logger.debug(
                "brier_ok",
                recent_brier=round(recent_brier, 6),
                threshold=self.BRIER_DRIFT_THRESHOLD,
            )

        return is_drift

    def detect_brier_drift_for_family(
        self, market_family: str, recent_brier: float
    ) -> bool:
        """
        Detect Brier drift for a specific market family and record history.

        Parameters
        ----------
        market_family:
            Market family identifier.
        recent_brier:
            Most recent Brier score.

        Returns
        -------
        bool
            True if drift detected.
        """
        if market_family not in self._brier_history:
            self._brier_history[market_family] = []
        self._brier_history[market_family].append(recent_brier)
        # Keep rolling window of 52 weeks (weekly monitoring)
        if len(self._brier_history[market_family]) > 52:
            self._brier_history[market_family].pop(0)

        is_drift = self.detect_brier_drift(recent_brier)

        if is_drift:
            from monitoring.metrics import BRIER_GAUGE
            BRIER_GAUGE.labels(market_family=market_family).set(recent_brier)

        return is_drift

    # ------------------------------------------------------------------
    # Population Stability Index
    # ------------------------------------------------------------------

    def compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute the Population Stability Index between two distributions.

        PSI = sum_i (actual_i - expected_i) * log(actual_i / expected_i)

        Parameters
        ----------
        expected:
            Reference distribution. 1-D array of observed feature values
            from the reference/training period.
        actual:
            Current distribution. 1-D array of observed feature values
            from the current scoring period.
        n_bins:
            Number of histogram bins. Default: 10.

        Returns
        -------
        float
            PSI value. 0.0 = identical distributions. >0.20 = significant shift.

        Notes
        -----
        - Both arrays must be non-empty.
        - Bin edges are derived from the expected distribution.
        - Bin proportions floored at ``_PSI_EPSILON`` to prevent log(0).
        """
        if len(expected) == 0 or len(actual) == 0:
            logger.warning(
                "psi_empty_distribution",
                expected_len=len(expected),
                actual_len=len(actual),
            )
            return 0.0

        expected_arr = np.asarray(expected, dtype=np.float64).ravel()
        actual_arr = np.asarray(actual, dtype=np.float64).ravel()

        # Define bin edges from the expected distribution
        min_val = float(min(expected_arr.min(), actual_arr.min()))
        max_val = float(max(expected_arr.max(), actual_arr.max()))

        if min_val == max_val:
            # Degenerate: all values identical → no shift
            return 0.0

        bin_edges = np.linspace(min_val, max_val, n_bins + 1)

        # Compute histograms (normalised to proportions)
        expected_counts, _ = np.histogram(expected_arr, bins=bin_edges)
        actual_counts, _ = np.histogram(actual_arr, bins=bin_edges)

        expected_props = expected_counts / max(expected_counts.sum(), 1)
        actual_props = actual_counts / max(actual_counts.sum(), 1)

        # Floor to epsilon to avoid log(0)
        expected_props = np.where(
            expected_props > 0, expected_props, _PSI_EPSILON
        )
        actual_props = np.where(
            actual_props > 0, actual_props, _PSI_EPSILON
        )

        psi = float(
            np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
        )

        logger.debug(
            "psi_computed",
            psi=round(psi, 6),
            n_bins=n_bins,
            expected_n=len(expected_arr),
            actual_n=len(actual_arr),
        )

        return max(0.0, psi)

    def detect_psi_drift(
        self,
        feature_name: str,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
    ) -> tuple[float, bool]:
        """
        Compute PSI and emit alerts if above threshold.

        Parameters
        ----------
        feature_name:
            Human-readable feature identifier (for metrics labels).
        expected:
            Reference feature distribution.
        actual:
            Current feature distribution.
        n_bins:
            Histogram bin count.

        Returns
        -------
        (psi, alert_triggered)
            psi — computed PSI value.
            alert_triggered — True when PSI > 0.20.
        """
        from monitoring.metrics import PSI_GAUGE, DRIFT_ALERT_COUNT

        psi = self.compute_psi(expected, actual, n_bins=n_bins)
        PSI_GAUGE.labels(feature=feature_name).set(psi)

        alert = psi > _PSI_ALERT_THRESHOLD
        if alert:
            DRIFT_ALERT_COUNT.labels(alert_type="psi_drift").inc()
            logger.warning(
                "psi_drift_alert",
                feature=feature_name,
                psi=round(psi, 6),
                threshold=_PSI_ALERT_THRESHOLD,
            )
        elif psi > _PSI_MONITOR_THRESHOLD:
            logger.info(
                "psi_monitor_warning",
                feature=feature_name,
                psi=round(psi, 6),
                threshold=_PSI_MONITOR_THRESHOLD,
            )
        else:
            logger.debug(
                "psi_ok",
                feature=feature_name,
                psi=round(psi, 6),
            )

        return psi, alert

    def rolling_brier_trend(self, market_family: str) -> Optional[float]:
        """
        Compute the linear trend (slope) of Brier score over history.

        A positive slope indicates degradation.

        Returns
        -------
        float or None
            Slope per period. None when fewer than 3 historical values exist.
        """
        history = self._brier_history.get(market_family, [])
        if len(history) < 3:
            return None

        x = np.arange(len(history), dtype=np.float64)
        y = np.array(history, dtype=np.float64)
        # Linear regression via numpy
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)


# Module-level type annotation for Optional (imported at top of file)
from typing import Optional  # noqa: E402 (after class def to avoid circular use in docstring)
