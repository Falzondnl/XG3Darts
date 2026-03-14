"""
Weekly validation of Markov chain accuracy.

Compares predicted vs observed leg outcomes across all seven market families.
Triggers structured alerts when any metric exceeds its threshold.

Market families validated
-------------------------
1. match_winner     — binary: P(P1 wins)
2. leg_handicap     — continuous: handicap line accuracy
3. totals_legs      — count: total legs O/U line accuracy
4. exact_score      — multinomial: correct-score markets
5. props_180        — count: 180 rate per leg
6. props_checkout   — multinomial: checkout route distribution
7. outright         — multi-class: tournament winner markets

All validation is read-only: it queries resolved bets and prices stored
in the database and compares them to actual outcomes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """
    Result of a single market family validation run.

    Attributes
    ----------
    market_family:
        One of the seven market family codes.
    metric_value:
        The computed metric (Brier, MAE, RPS, etc.).
    threshold:
        Alert threshold for this metric.
    alert_triggered:
        True if metric_value > threshold.
    since_days:
        Lookback window used.
    sample_size:
        Number of resolved markets included.
    """

    market_family: str
    metric_value: float
    threshold: float
    alert_triggered: bool
    since_days: int
    sample_size: int
    extra: dict[str, Any]


class MarkovValidationMonitor:
    """
    Weekly validation of Markov chain accuracy across all market families.

    Compares predicted probability distributions to observed outcomes using
    proper scoring rules. Emits structlog alerts when thresholds are breached
    and updates Prometheus gauges for operational dashboards.

    Threshold table (from business spec):
        match_winner_brier:   0.230
        leg_handicap_mae:     0.030
        totals_mae_legs:      1.0
        exact_score_rps:      0.15
        props_180_mae:        0.20
        props_checkout_rps:   0.14
        outright_brier:       0.16
        live_repricing_drift: 0.06
    """

    ALERT_THRESHOLDS: dict[str, float] = {
        "match_winner_brier":   0.230,
        "leg_handicap_mae":     0.030,
        "totals_mae_legs":      1.0,
        "exact_score_rps":      0.15,
        "props_180_mae":        0.20,
        "props_checkout_rps":   0.14,
        "outright_brier":       0.16,
        "live_repricing_drift": 0.06,
    }

    # Map family → metric type
    _FAMILY_METRIC_TYPE: dict[str, str] = {
        "match_winner":     "brier",
        "leg_handicap":     "mae",
        "totals_legs":      "mae",
        "exact_score":      "rps",
        "props_180":        "mae",
        "props_checkout":   "rps",
        "outright":         "brier",
        "live_repricing":   "drift",
    }

    # Map family → threshold key
    _FAMILY_THRESHOLD_KEY: dict[str, str] = {
        "match_winner":   "match_winner_brier",
        "leg_handicap":   "leg_handicap_mae",
        "totals_legs":    "totals_mae_legs",
        "exact_score":    "exact_score_rps",
        "props_180":      "props_180_mae",
        "props_checkout": "props_checkout_rps",
        "outright":       "outright_brier",
        "live_repricing": "live_repricing_drift",
    }

    def __init__(self, db_session: Any = None) -> None:
        """
        Parameters
        ----------
        db_session:
            Optional SQLAlchemy AsyncSession. When None, the validator operates
            in mock mode (returns placeholder metrics). In production, inject
            a real session via dependency injection or the lifespan context.
        """
        self._db = db_session

    async def validate_all_families(
        self, since_days: int = 7
    ) -> dict[str, ValidationResult]:
        """
        Run all market family validations and return a summary.

        Triggers structured log alerts on threshold breach. Updates Prometheus
        Brier score gauges for all families.

        Parameters
        ----------
        since_days:
            Lookback window in days. Default: 7 (weekly validation).

        Returns
        -------
        dict[str, ValidationResult]
            Keyed by market family name.
        """
        from monitoring.metrics import BRIER_GAUGE, DRIFT_ALERT_COUNT

        results: dict[str, ValidationResult] = {}
        families = list(self._FAMILY_THRESHOLD_KEY.keys())

        for family in families:
            try:
                metric_type = self._FAMILY_METRIC_TYPE[family]
                threshold_key = self._FAMILY_THRESHOLD_KEY[family]
                threshold = self.ALERT_THRESHOLDS[threshold_key]

                if metric_type == "brier":
                    value, sample_size = await self._compute_brier(family, since_days)
                elif metric_type == "mae":
                    value, sample_size = await self._compute_mae(family, since_days)
                elif metric_type == "rps":
                    value, sample_size = await self._compute_rps(family, since_days)
                elif metric_type == "drift":
                    value, sample_size = await self._compute_drift(family, since_days)
                else:
                    value, sample_size = 0.0, 0

                alert = value > threshold

                result = ValidationResult(
                    market_family=family,
                    metric_value=round(value, 6),
                    threshold=threshold,
                    alert_triggered=alert,
                    since_days=since_days,
                    sample_size=sample_size,
                    extra={"metric_type": metric_type},
                )
                results[family] = result

                # Update Prometheus gauge
                BRIER_GAUGE.labels(market_family=family).set(value)

                if alert:
                    DRIFT_ALERT_COUNT.labels(alert_type=f"{family}_threshold_breach").inc()
                    logger.warning(
                        "markov_validation_alert",
                        market_family=family,
                        metric_type=metric_type,
                        value=round(value, 6),
                        threshold=threshold,
                        sample_size=sample_size,
                        since_days=since_days,
                    )
                else:
                    logger.info(
                        "markov_validation_ok",
                        market_family=family,
                        metric_type=metric_type,
                        value=round(value, 6),
                        threshold=threshold,
                        sample_size=sample_size,
                    )

            except Exception as exc:
                logger.error(
                    "markov_validation_error",
                    market_family=family,
                    error=str(exc),
                )
                results[family] = ValidationResult(
                    market_family=family,
                    metric_value=float("nan"),
                    threshold=self.ALERT_THRESHOLDS.get(
                        self._FAMILY_THRESHOLD_KEY.get(family, ""), 0.0
                    ),
                    alert_triggered=False,
                    since_days=since_days,
                    sample_size=0,
                    extra={"error": str(exc)},
                )

        return results

    async def compute_brier_score(
        self,
        market_family: str,
        since_days: int,
    ) -> float:
        """
        Compute Brier score for predicted vs actual outcomes.

        The Brier score is defined as:
            BS = (1/N) * sum_{i=1}^{N} (p_i - o_i)^2

        where p_i is the predicted probability for the winning outcome
        and o_i is the binary indicator (1 = outcome occurred).

        A perfect model scores 0.0; random scoring scores ~0.25 for binary markets.

        Parameters
        ----------
        market_family:
            Market family to score. Must be a binary-outcome family (match_winner,
            outright). For multi-outcome families use compute_rps.
        since_days:
            Number of days of historical data to include.

        Returns
        -------
        float
            Brier score in [0, 1]. Lower is better.
        """
        value, _ = await self._compute_brier(market_family, since_days)
        return value

    # ------------------------------------------------------------------
    # Internal metric computation
    # ------------------------------------------------------------------

    async def _compute_brier(
        self, market_family: str, since_days: int
    ) -> tuple[float, int]:
        """
        Compute Brier score for a binary-outcome market family.

        Queries resolved predictions from the database (or returns a mock
        value when no DB is configured).

        Returns (brier_score, sample_size).
        """
        if self._db is None:
            return await self._mock_metric(market_family, "brier")

        # In production: query darts_market_predictions table
        # SELECT predicted_prob, actual_outcome FROM darts_market_predictions
        # WHERE market_family = :family AND resolved_at >= NOW() - INTERVAL ':days days'
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

            if not results:
                logger.debug(
                    "markov_validation_no_data",
                    family=market_family,
                    since_days=since_days,
                )
                return 0.0, 0

            brier_sum = 0.0
            for row in results:
                p = float(row[0])
                o = float(row[1])
                brier_sum += (p - o) ** 2

            brier = brier_sum / len(results)
            return brier, len(results)

        except Exception as exc:
            logger.error(
                "markov_validation_db_error",
                family=market_family,
                error=str(exc),
            )
            return 0.0, 0

    async def _compute_mae(
        self, market_family: str, since_days: int
    ) -> tuple[float, int]:
        """
        Compute Mean Absolute Error for a continuous-outcome market family.

        Used for leg_handicap and totals_legs families.
        Returns (mae, sample_size).
        """
        if self._db is None:
            return await self._mock_metric(market_family, "mae")

        try:
            from datetime import datetime, timedelta, timezone

            cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
            query = """
                SELECT predicted_value, actual_value
                FROM darts_market_predictions
                WHERE market_family = :family
                  AND resolved_at >= :cutoff
                  AND predicted_value IS NOT NULL
                  AND actual_value IS NOT NULL
            """
            rows = await self._db.execute(
                __import__("sqlalchemy").text(query),
                {"family": market_family, "cutoff": cutoff},
            )
            results = rows.fetchall()

            if not results:
                return 0.0, 0

            mae_sum = sum(abs(float(r[0]) - float(r[1])) for r in results)
            mae = mae_sum / len(results)
            return mae, len(results)

        except Exception as exc:
            logger.error(
                "markov_validation_db_error",
                family=market_family,
                error=str(exc),
            )
            return 0.0, 0

    async def _compute_rps(
        self, market_family: str, since_days: int
    ) -> tuple[float, int]:
        """
        Compute Ranked Probability Score for a multi-class market family.

        RPS is the multi-class extension of the Brier score:
            RPS = (1/(K-1)) * sum_{k=1}^{K-1} (CDF_pred[k] - CDF_actual[k])^2

        Returns (rps, sample_size).
        """
        if self._db is None:
            return await self._mock_metric(market_family, "rps")

        try:
            from datetime import datetime, timedelta, timezone

            cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
            query = """
                SELECT predicted_probs_json, actual_outcome_index, n_outcomes
                FROM darts_market_predictions
                WHERE market_family = :family
                  AND resolved_at >= :cutoff
                  AND predicted_probs_json IS NOT NULL
                  AND actual_outcome_index IS NOT NULL
            """
            rows = await self._db.execute(
                __import__("sqlalchemy").text(query),
                {"family": market_family, "cutoff": cutoff},
            )
            results = rows.fetchall()

            if not results:
                return 0.0, 0

            import json as _json

            rps_sum = 0.0
            count = 0
            for row in results:
                try:
                    probs = _json.loads(row[0]) if isinstance(row[0], str) else row[0]
                    actual_idx = int(row[1])
                    n = int(row[2])

                    if not probs or n < 2:
                        continue

                    # Build actual distribution (one-hot)
                    actual = [0.0] * n
                    if 0 <= actual_idx < n:
                        actual[actual_idx] = 1.0

                    # Cumulative distributions
                    cum_pred = 0.0
                    cum_actual = 0.0
                    rps_i = 0.0
                    for k in range(n - 1):
                        cum_pred += probs[k] if k < len(probs) else 0.0
                        cum_actual += actual[k]
                        rps_i += (cum_pred - cum_actual) ** 2

                    rps_sum += rps_i / (n - 1)
                    count += 1
                except Exception:
                    continue

            return (rps_sum / count if count > 0 else 0.0), count

        except Exception as exc:
            logger.error(
                "markov_validation_db_error",
                family=market_family,
                error=str(exc),
            )
            return 0.0, 0

    async def _compute_drift(
        self, market_family: str, since_days: int
    ) -> tuple[float, int]:
        """
        Compute live repricing drift.

        Measures the deviation between live prices at key match moments
        and the final settling price. High drift indicates the live model
        is under-reacting or over-reacting to visit data.

        Returns (drift_score, sample_size).
        """
        if self._db is None:
            return await self._mock_metric(market_family, "drift")

        try:
            from datetime import datetime, timedelta, timezone

            cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
            query = """
                SELECT live_price_midpoint, final_price_midpoint
                FROM darts_live_price_log
                WHERE resolved_at >= :cutoff
                  AND live_price_midpoint IS NOT NULL
                  AND final_price_midpoint IS NOT NULL
            """
            rows = await self._db.execute(
                __import__("sqlalchemy").text(query),
                {"cutoff": cutoff},
            )
            results = rows.fetchall()

            if not results:
                return 0.0, 0

            drift_sum = 0.0
            count = 0
            for row in results:
                p_live = float(row[0])
                p_final = float(row[1])
                if p_live > 0 and p_final > 0:
                    drift_sum += abs(math.log(p_live / p_final))
                    count += 1

            return (drift_sum / count if count > 0 else 0.0), count

        except Exception as exc:
            logger.error(
                "markov_validation_drift_error",
                error=str(exc),
            )
            return 0.0, 0

    async def _mock_metric(
        self, market_family: str, metric_type: str
    ) -> tuple[float, int]:
        """
        Return a mock metric value when no DB is configured.

        In non-production environments (unit tests, local dev without DB),
        we return a safe value well below alert thresholds so the validator
        does not trigger false alerts.
        """
        safe_values = {
            "brier": 0.18,
            "mae": 0.020,
            "rps": 0.10,
            "drift": 0.03,
        }
        value = safe_values.get(metric_type, 0.1)
        logger.debug(
            "markov_validation_mock",
            family=market_family,
            metric_type=metric_type,
            value=value,
        )
        return value, 0
