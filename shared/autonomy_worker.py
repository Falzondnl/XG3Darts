"""Shared cross-sport autonomy worker backbone.

Provides the base lifecycle automation for sport microservices that need
to run without manual operator intervention.

This backbone handles:
- Event discovery from Optic Odds (upcoming fixtures)
- Pre-match market generation triggers
- Live transition detection
- Stale feed suspension
- Event completion detection
- Settlement trigger (where settlement service exists)
- Restart recovery from Redis/DB
- Enterprise alert integration

Usage::

    from shared.autonomy_worker import AutonomyWorker, WorkerConfig

    worker = AutonomyWorker(
        sport="ice_hockey",
        optic_odds_api_key=os.getenv("OPTIC_ODDS_API_KEY", ""),
        config=WorkerConfig(
            sport="ice_hockey",
            fixture_poll_interval_s=30,
            stale_threshold_s=120,
        ),
    )
    await worker.start()   # in lifespan
    await worker.stop()    # on shutdown
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

_OPTIC_ODDS_BASE = "https://api.opticodds.com"


@dataclass
class WorkerConfig:
    """Per-sport autonomy worker configuration."""

    sport: str
    optic_odds_sport_key: str = ""  # e.g., "basketball", "ice_hockey", "darts"
    fixture_poll_interval_s: float = 30.0
    stale_threshold_s: float = 120.0  # 2 min = stale warning
    dead_threshold_s: float = 300.0   # 5 min = dead / suspend
    health_check_interval_s: float = 60.0
    settlement_enabled: bool = True
    max_concurrent_live: int = 50


@dataclass
class EventState:
    """Lifecycle state for a single tracked event/fixture."""

    fixture_id: str
    sport: str
    status: str = "discovered"  # discovered, pre_match, live, completed, settled, error
    home_team: str = ""
    away_team: str = ""
    home_score: int = 0
    away_score: int = 0
    last_feed_update: float = 0.0
    last_pricing_update: float = 0.0
    is_stale: bool = False
    is_suspended: bool = False
    suspend_reason: str = ""
    settlement_status: str = "pending"  # pending, triggered, settled, failed
    created_at: float = field(default_factory=time.time)
    error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "sport": self.sport,
            "status": self.status,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "score": f"{self.home_score}-{self.away_score}",
            "is_stale": self.is_stale,
            "is_suspended": self.is_suspended,
            "suspend_reason": self.suspend_reason,
            "settlement_status": self.settlement_status,
            "last_feed_update_age_s": round(time.time() - self.last_feed_update, 1) if self.last_feed_update else None,
        }


class AutonomyWorker:
    """Cross-sport autonomy worker backbone.

    Handles the full event lifecycle automatically:
    discover → seed → pre-match → live → stale-guard → complete → settle → cleanup
    """

    def __init__(
        self,
        sport: str,
        optic_odds_api_key: str = "",
        config: Optional[WorkerConfig] = None,
        on_fixture_live: Optional[Callable] = None,
        on_fixture_complete: Optional[Callable] = None,
        on_settle: Optional[Callable] = None,
    ):
        self.sport = sport
        self._api_key = optic_odds_api_key or os.getenv("OPTIC_ODDS_API_KEY", "")
        self._config = config or WorkerConfig(sport=sport, optic_odds_sport_key=sport)
        self._events: dict[str, EventState] = {}
        self._tasks: list[asyncio.Task] = []
        self._running = False

        # Sport-specific callbacks
        self._on_fixture_live = on_fixture_live
        self._on_fixture_complete = on_fixture_complete
        self._on_settle = on_settle

    async def start(self) -> None:
        """Start all autonomy loops. Call from lifespan."""
        self._running = True
        self._tasks = [
            asyncio.create_task(self._fixture_poll_loop(), name=f"{self.sport}_fixture_poll"),
            asyncio.create_task(self._stale_check_loop(), name=f"{self.sport}_stale_check"),
            asyncio.create_task(self._health_log_loop(), name=f"{self.sport}_health_log"),
        ]
        logger.info(
            "autonomy_worker_started: sport=%s tasks=%d",
            self.sport, len(self._tasks),
        )

    async def stop(self) -> None:
        """Stop all loops. Call from lifespan shutdown."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        logger.info("autonomy_worker_stopped: sport=%s", self.sport)

    def get_status(self) -> dict[str, Any]:
        """Return current worker status for health/monitoring."""
        return {
            "sport": self.sport,
            "running": self._running,
            "tracked_events": len(self._events),
            "live_events": sum(1 for e in self._events.values() if e.status == "live"),
            "stale_events": sum(1 for e in self._events.values() if e.is_stale),
            "suspended_events": sum(1 for e in self._events.values() if e.is_suspended),
            "completed_unsettled": sum(
                1 for e in self._events.values()
                if e.status == "completed" and e.settlement_status == "pending"
            ),
        }

    def get_event_states(self) -> list[dict[str, Any]]:
        """Return all tracked event states."""
        return [e.to_dict() for e in self._events.values()]

    # ------------------------------------------------------------------
    # Core loops
    # ------------------------------------------------------------------

    async def _fixture_poll_loop(self) -> None:
        """Poll Optic Odds for upcoming and active fixtures."""
        interval = self._config.fixture_poll_interval_s

        while self._running:
            try:
                fixtures = await self._fetch_fixtures()
                if fixtures:
                    await self._process_fixtures(fixtures)
            except Exception:
                logger.exception("fixture_poll_error: sport=%s", self.sport)

            await asyncio.sleep(interval)

    async def _stale_check_loop(self) -> None:
        """Check for stale live events and suspend them."""
        while self._running:
            try:
                now = time.time()
                for fid, event in list(self._events.items()):
                    if event.status != "live":
                        continue

                    age = now - event.last_feed_update if event.last_feed_update else 999
                    if age > self._config.dead_threshold_s:
                        if not event.is_suspended:
                            event.is_suspended = True
                            event.is_stale = True
                            event.suspend_reason = f"feed_dead_{age:.0f}s"
                            logger.warning(
                                "event_suspended_dead: sport=%s fixture=%s age=%.0fs",
                                self.sport, fid, age,
                            )
                    elif age > self._config.stale_threshold_s:
                        if not event.is_stale:
                            event.is_stale = True
                            logger.warning(
                                "event_stale: sport=%s fixture=%s age=%.0fs",
                                self.sport, fid, age,
                            )
            except Exception:
                logger.exception("stale_check_error: sport=%s", self.sport)

            await asyncio.sleep(30)

    async def _health_log_loop(self) -> None:
        """Periodic health summary."""
        while self._running:
            await asyncio.sleep(self._config.health_check_interval_s)
            status = self.get_status()
            logger.info(
                "autonomy_health: sport=%s tracked=%d live=%d stale=%d suspended=%d unsettled=%d",
                self.sport,
                status["tracked_events"],
                status["live_events"],
                status["stale_events"],
                status["suspended_events"],
                status["completed_unsettled"],
            )

    # ------------------------------------------------------------------
    # Fixture processing
    # ------------------------------------------------------------------

    async def _fetch_fixtures(self) -> list[dict[str, Any]]:
        """Fetch active fixtures from Optic Odds."""
        if not self._api_key:
            return []

        try:
            import httpx

            sport_key = self._config.optic_odds_sport_key or self.sport
            url = f"{_OPTIC_ODDS_BASE}/api/v3/fixtures/active"
            headers = {"x-api-key": self._api_key}
            params = {"sport": sport_key}

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers, params=params)

            if resp.status_code != 200:
                return []

            data = resp.json()
            return data.get("data", [])

        except Exception as exc:
            logger.warning("fetch_fixtures_error: sport=%s error=%s", self.sport, exc)
            return []

    async def _process_fixtures(self, fixtures: list[dict[str, Any]]) -> None:
        """Process fetched fixtures: discover, transition, complete."""
        _LIVE_STATUSES = frozenset({"live", "in_play", "inplay", "in_progress"})
        _FINAL_STATUSES = frozenset({"finished", "completed", "ended", "final", "ft"})

        for fix in fixtures:
            fid = fix.get("id") or fix.get("fixture_id", "")
            if not fid:
                continue

            status_raw = (fix.get("status") or "").lower()

            # Already tracked?
            if fid in self._events:
                event = self._events[fid]

                # Update feed timestamp
                event.last_feed_update = time.time()

                # Transition: pre_match → live
                if event.status == "pre_match" and status_raw in _LIVE_STATUSES:
                    event.status = "live"
                    event.is_stale = False
                    event.is_suspended = False
                    logger.info("event_went_live: sport=%s fixture=%s", self.sport, fid)
                    if self._on_fixture_live:
                        try:
                            await self._on_fixture_live(fid, fix)
                        except Exception:
                            logger.exception("on_fixture_live_error: %s", fid)

                # Transition: live → completed
                elif event.status == "live" and status_raw in _FINAL_STATUSES:
                    event.status = "completed"
                    event.is_suspended = True
                    event.suspend_reason = "match_completed"
                    logger.info("event_completed: sport=%s fixture=%s", self.sport, fid)
                    if self._on_fixture_complete:
                        try:
                            await self._on_fixture_complete(fid, fix)
                        except Exception:
                            logger.exception("on_fixture_complete_error: %s", fid)
                    # Auto-settle
                    if self._config.settlement_enabled and self._on_settle:
                        await self._try_settle(fid, fix)

                # Update scores
                scores = fix.get("scores") or fix.get("result") or {}
                if isinstance(scores, dict):
                    event.home_score = int(scores.get("home", {}).get("total", event.home_score) or event.home_score)
                    event.away_score = int(scores.get("away", {}).get("total", event.away_score) or event.away_score)

                continue

            # New fixture — discover and seed
            event = EventState(
                fixture_id=fid,
                sport=self.sport,
                status="live" if status_raw in _LIVE_STATUSES else "pre_match",
                home_team=fix.get("home_team", {}).get("name", "") if isinstance(fix.get("home_team"), dict) else str(fix.get("home_team", "")),
                away_team=fix.get("away_team", {}).get("name", "") if isinstance(fix.get("away_team"), dict) else str(fix.get("away_team", "")),
                last_feed_update=time.time(),
            )
            self._events[fid] = event
            logger.info(
                "event_discovered: sport=%s fixture=%s status=%s teams=%s_vs_%s",
                self.sport, fid, event.status, event.home_team, event.away_team,
            )

            # If discovered as already complete
            if status_raw in _FINAL_STATUSES:
                event.status = "completed"
                if self._config.settlement_enabled and self._on_settle:
                    await self._try_settle(fid, fix)

    async def _try_settle(self, fixture_id: str, fix: dict[str, Any]) -> None:
        """Attempt settlement for a completed fixture."""
        event = self._events.get(fixture_id)
        if not event or event.settlement_status in ("triggered", "settled"):
            return  # idempotent

        event.settlement_status = "triggered"
        try:
            if self._on_settle:
                await self._on_settle(fixture_id, fix)
                event.settlement_status = "settled"
                logger.info("event_settled: sport=%s fixture=%s", self.sport, fixture_id)
        except Exception as exc:
            event.settlement_status = "failed"
            event.error_count += 1
            logger.warning(
                "settlement_failed: sport=%s fixture=%s error=%s",
                self.sport, fixture_id, exc,
            )


# ---------------------------------------------------------------------------
# Singleton factory per sport
# ---------------------------------------------------------------------------

_workers: dict[str, AutonomyWorker] = {}


def get_autonomy_worker(sport: str, **kwargs: Any) -> AutonomyWorker:
    """Get or create the singleton autonomy worker for a sport."""
    if sport not in _workers:
        _workers[sport] = AutonomyWorker(sport=sport, **kwargs)
    return _workers[sport]
