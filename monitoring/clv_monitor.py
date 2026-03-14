"""
CLV (Closing Line Value) monitor.

Tracks our opening price vs Betfair/Pinnacle closing price to measure
long-run model alpha. A positive CLV indicates we are pricing sharper
than the closing market.

Auto-adjustment
---------------
When CLV < -0.02 for a market family, the margin is widened by 0.5 %
to protect against negative expected value. This is applied
incrementally and capped at MAX_MARGIN to prevent over-pricing.

CLV formula
-----------
    CLV = log(our_opening_price / betfair_closing_price)

Positive CLV = we opened sharper than the market settled.
Negative CLV = the market drifted against our opening price.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

# Auto-adjust parameters
_CLV_TRIGGER_THRESHOLD = -0.02       # widen margin when CLV is below this
_MARGIN_WIDEN_STEP = 0.005           # 0.5 % per trigger
_MAX_MARGIN = 0.15                   # cap margin at 15 %
_DEFAULT_MARGIN = 0.04               # 4 % default margin (if not set in Redis)

# Redis key template for current margin per market family
_MARGIN_REDIS_KEY = "xg3:margin:{family}"

# Lookback for CLV computation when none specified
_DEFAULT_LOOKBACK_DAYS = 30


class DartsCLVMonitor:
    """
    CLV monitor that compares our opening prices to Betfair/Pinnacle closing.

    Integrates with Redis for margin storage and the market prediction log
    in PostgreSQL for historical CLV computation.

    Parameters
    ----------
    db_session:
        AsyncSession for querying stored opening and closing prices.
    redis_client:
        redis.asyncio client for reading/writing current margins.
    """

    def __init__(
        self,
        db_session: Any = None,
        redis_client: Any = None,
    ) -> None:
        self._db = db_session
        self._redis = redis_client
        # In-memory margin cache (fallback when Redis unavailable)
        self._margin_cache: dict[str, float] = {}

    async def compute_clv(
        self,
        market_family: str,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    ) -> float:
        """
        Compute the rolling CLV for a market family.

        CLV = mean[ log(our_opening_price / betfair_closing_price) ]

        Parameters
        ----------
        market_family:
            One of the seven market family codes (match_winner, leg_handicap, etc.).
        lookback_days:
            How many days of settled markets to include.

        Returns
        -------
        float
            Rolling CLV. Positive = model is sharper than market.
            Returns 0.0 when no historical data is available.
        """
        from monitoring.metrics import CLV_GAUGE

        clv = await self._compute_clv_from_db(market_family, lookback_days)
        CLV_GAUGE.labels(market_family=market_family).set(clv)

        logger.info(
            "clv_computed",
            market_family=market_family,
            clv=round(clv, 6),
            lookback_days=lookback_days,
        )
        return clv

    async def auto_adjust_margin(
        self,
        market_family: str,
        clv: float,
    ) -> float:
        """
        Auto-adjust margin when CLV goes below the trigger threshold.

        If CLV < -0.02, widen the market margin by 0.5 % (capped at 15 %).
        If CLV >= -0.02, leave the margin unchanged.

        Parameters
        ----------
        market_family:
            Market family to adjust.
        clv:
            Current CLV for this family.

        Returns
        -------
        float
            New (or unchanged) margin value.
        """
        from monitoring.metrics import MARGIN_GAUGE

        if clv < _CLV_TRIGGER_THRESHOLD:
            current = await self._get_current_margin(market_family)
            new_margin = min(_MAX_MARGIN, current + _MARGIN_WIDEN_STEP)
            await self._set_margin(market_family, new_margin)

            MARGIN_GAUGE.labels(market_family=market_family).set(new_margin)
            logger.warning(
                "clv_margin_widened",
                market_family=market_family,
                clv=round(clv, 6),
                old_margin=round(current, 4),
                new_margin=round(new_margin, 4),
            )
            return new_margin

        current = await self._get_current_margin(market_family)
        MARGIN_GAUGE.labels(market_family=market_family).set(current)
        return current

    async def get_all_clv_summary(
        self, lookback_days: int = _DEFAULT_LOOKBACK_DAYS
    ) -> dict[str, dict[str, Any]]:
        """
        Return a CLV + margin summary for all market families.

        Returns
        -------
        dict[str, dict]
            Keyed by market family. Each value contains:
            ``clv``, ``margin``, ``alert`` (bool), ``lookback_days``.
        """
        families = [
            "match_winner",
            "leg_handicap",
            "totals_legs",
            "exact_score",
            "props_180",
            "props_checkout",
            "outright",
        ]
        summary: dict[str, dict[str, Any]] = {}
        for family in families:
            clv = await self.compute_clv(family, lookback_days)
            margin = await self._get_current_margin(family)
            summary[family] = {
                "clv": round(clv, 6),
                "margin": round(margin, 4),
                "alert": clv < _CLV_TRIGGER_THRESHOLD,
                "lookback_days": lookback_days,
            }
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _compute_clv_from_db(
        self, market_family: str, lookback_days: int
    ) -> float:
        """Query the DB for opening/closing price pairs and compute CLV."""
        if self._db is None:
            # Mock: return a neutral CLV (slightly positive) in dev mode
            return 0.005

        try:
            from datetime import datetime, timedelta, timezone

            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            # Query our price log vs closing price log
            query = """
                SELECT
                    mpl.predicted_prob_p1 AS our_open_price,
                    cpl.closing_price_p1  AS closing_price
                FROM darts_market_predictions mpl
                JOIN darts_closing_prices cpl
                  ON mpl.match_id = cpl.match_id
                  AND mpl.market_family = cpl.market_family
                WHERE mpl.market_family = :family
                  AND mpl.created_at >= :cutoff
                  AND mpl.predicted_prob_p1 > 0
                  AND cpl.closing_price_p1 > 0
            """
            rows = await self._db.execute(
                __import__("sqlalchemy").text(query),
                {"family": market_family, "cutoff": cutoff},
            )
            results = rows.fetchall()

            if not results:
                logger.debug(
                    "clv_no_data",
                    family=market_family,
                    since_days=lookback_days,
                )
                return 0.0

            clv_sum = 0.0
            count = 0
            for row in results:
                our_price = float(row[0])
                closing = float(row[1])
                if our_price > 0 and closing > 0:
                    clv_sum += math.log(our_price / closing)
                    count += 1

            return clv_sum / count if count > 0 else 0.0

        except Exception as exc:
            logger.error(
                "clv_db_error",
                family=market_family,
                error=str(exc),
            )
            return 0.0

    async def _get_current_margin(self, market_family: str) -> float:
        """Read current margin from Redis, falling back to the in-memory cache."""
        # Try Redis first
        if self._redis is not None:
            key = _MARGIN_REDIS_KEY.format(family=market_family)
            try:
                val = await self._redis.get(key)
                if val is not None:
                    return float(val)
            except Exception as exc:
                logger.warning(
                    "clv_redis_margin_read_error",
                    family=market_family,
                    error=str(exc),
                )

        # In-memory fallback
        return self._margin_cache.get(market_family, _DEFAULT_MARGIN)

    async def _set_margin(self, market_family: str, margin: float) -> None:
        """Persist updated margin to Redis and the in-memory cache."""
        self._margin_cache[market_family] = margin

        if self._redis is not None:
            key = _MARGIN_REDIS_KEY.format(family=market_family)
            try:
                # Persist indefinitely (margin should survive restarts)
                await self._redis.set(key, str(margin))
                logger.debug(
                    "clv_margin_saved",
                    family=market_family,
                    margin=round(margin, 4),
                )
            except Exception as exc:
                logger.warning(
                    "clv_redis_margin_write_error",
                    family=market_family,
                    error=str(exc),
                )
