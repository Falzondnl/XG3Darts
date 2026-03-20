"""Shared stale-price enforcement guard for XG3 sport microservices.

Prevents live pricing endpoints from serving stale prices as if current.
Modeled after the proven Soccer/Table Tennis feed_latency_guard pattern.

Usage in a sport microservice::

    from shared.stale_price_guard import StalePriceGuard, StalePriceStatus

    guard = StalePriceGuard(sport="basketball")

    # On every live update from feed:
    guard.record_update(match_id)

    # Before returning live prices:
    status = guard.check(match_id)
    if status.should_suspend:
        return JSONResponse(status_code=503, content={
            "error": "live_suspended",
            "reason": status.reason,
            "stale_seconds": status.stale_seconds,
        })
    # Otherwise apply margin_factor and stake_factor:
    adjusted_margin = base_margin * status.margin_factor
    adjusted_max_stake = max_stake * status.stake_factor

Thresholds (configurable per sport):
    < 5s:   NORMAL   — full pricing, no adjustment
    5-15s:  CAUTION  — margin +50%, full stakes
    15-30s: WARNING  — margin +100%, stakes halved
    > 30s:  SUSPEND  — no pricing, 503 returned

These match the Soccer/TT proven production thresholds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FreshnessLevel(str, Enum):
    """Feed freshness classification."""

    NORMAL = "normal"
    CAUTION = "caution"
    WARNING = "warning"
    SUSPENDED = "suspended"


@dataclass(frozen=True)
class StalePriceStatus:
    """Result of a staleness check for a specific match or feed."""

    level: FreshnessLevel
    stale_seconds: float
    margin_factor: float
    stake_factor: float
    should_suspend: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "stale_seconds": round(self.stale_seconds, 1),
            "margin_factor": self.margin_factor,
            "stake_factor": self.stake_factor,
            "should_suspend": self.should_suspend,
            "reason": self.reason,
        }


@dataclass
class StalePriceGuard:
    """Enforces stale-price safety for a sport's live pricing.

    Parameters
    ----------
    sport:
        Sport name for logging context.
    threshold_caution_s:
        Seconds before CAUTION level (margin widening).
    threshold_warning_s:
        Seconds before WARNING level (margin + stake reduction).
    threshold_suspend_s:
        Seconds before SUSPEND level (block all live pricing).
    """

    sport: str
    threshold_caution_s: float = 5.0
    threshold_warning_s: float = 15.0
    threshold_suspend_s: float = 30.0

    # Per-match last-update timestamps (monotonic clock)
    _last_update: dict[str, float] = field(default_factory=dict)
    # Global last update (any match)
    _last_any_update: float = field(default_factory=time.monotonic)

    def record_update(self, match_id: str) -> None:
        """Record that fresh data was received for a match."""
        now = time.monotonic()
        self._last_update[match_id] = now
        self._last_any_update = now

    def remove_match(self, match_id: str) -> None:
        """Remove tracking for a completed/settled match."""
        self._last_update.pop(match_id, None)

    def check(self, match_id: str) -> StalePriceStatus:
        """Check freshness for a specific match.

        Returns a StalePriceStatus with the appropriate enforcement level.
        If the match has never been updated, returns SUSPENDED.
        """
        last = self._last_update.get(match_id)
        if last is None:
            return StalePriceStatus(
                level=FreshnessLevel.SUSPENDED,
                stale_seconds=999.0,
                margin_factor=1.0,
                stake_factor=0.0,
                should_suspend=True,
                reason=f"No live data ever received for match {match_id}",
            )

        age = time.monotonic() - last
        return self._classify(age, match_id)

    def check_global(self) -> StalePriceStatus:
        """Check freshness across all matches (global feed health)."""
        age = time.monotonic() - self._last_any_update
        return self._classify(age, "global")

    def _classify(self, age: float, context: str) -> StalePriceStatus:
        """Classify staleness into enforcement level."""
        if age >= self.threshold_suspend_s:
            logger.warning(
                "stale_price_suspended: sport=%s context=%s stale_seconds=%.1f",
                self.sport,
                context,
                age,
            )
            return StalePriceStatus(
                level=FreshnessLevel.SUSPENDED,
                stale_seconds=age,
                margin_factor=1.0,
                stake_factor=0.0,
                should_suspend=True,
                reason=f"Feed stale for {age:.0f}s (>{self.threshold_suspend_s}s) — suspended",
            )

        if age >= self.threshold_warning_s:
            return StalePriceStatus(
                level=FreshnessLevel.WARNING,
                stale_seconds=age,
                margin_factor=2.0,
                stake_factor=0.5,
                should_suspend=False,
                reason=f"Feed stale {age:.0f}s — margins doubled, stakes halved",
            )

        if age >= self.threshold_caution_s:
            return StalePriceStatus(
                level=FreshnessLevel.CAUTION,
                stale_seconds=age,
                margin_factor=1.5,
                stake_factor=1.0,
                should_suspend=False,
                reason=f"Feed stale {age:.0f}s — margins +50%",
            )

        return StalePriceStatus(
            level=FreshnessLevel.NORMAL,
            stale_seconds=age,
            margin_factor=1.0,
            stake_factor=1.0,
            should_suspend=False,
            reason="Live data fresh",
        )

    def get_all_statuses(self) -> dict[str, StalePriceStatus]:
        """Return freshness status for all tracked matches."""
        return {mid: self.check(mid) for mid in self._last_update}

    def active_match_count(self) -> int:
        """Return number of matches being tracked."""
        return len(self._last_update)
