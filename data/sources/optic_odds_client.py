"""
Optic Odds live feed client.

Provides two transport modes:
  1. RabbitMQ consumer — primary, low-latency, push-based.
  2. HTTP REST polling  — fallback when RabbitMQ is unavailable.

Usage
-----
    feed = OpticOddsLiveFeed()
    await feed.connect()

    # Subscribe to a match
    await feed.subscribe_match("match-123", my_callback)

    # Or poll
    state = await feed.get_live_state("match-123")

The ``LiveMatchData`` dataclass is the normalised representation
returned by both transports so the live engine is transport-agnostic.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

import aiohttp
import structlog

from app.config import settings
from engines.errors import DartsDataError

logger = structlog.get_logger(__name__)

# HTTP base URL for Optic Odds REST API
OPTIC_ODDS_HTTP_BASE = "https://api.opticodds.com/api/v3"

# RabbitMQ AMQP URL (constructed from settings)
_AMQP_PATTERN = "amqp://{user}:{password}@{host}:5672/"


@dataclass
class LiveMatchData:
    """
    Normalised live match state as returned by Optic Odds.

    This dataclass is the interface contract between the Optic Odds
    feed client and the live engine. All fields are optional because
    the feed may only have partial data.
    """

    match_id: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    legs_home: Optional[int] = None
    legs_away: Optional[int] = None
    sets_home: Optional[int] = None
    sets_away: Optional[int] = None
    current_server: Optional[str] = None    # player ID or "home"/"away"
    period: Optional[str] = None            # e.g. "leg_3", "set_2_leg_1"
    is_live: bool = False
    feed_timestamp: float = field(default_factory=time.time)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def feed_age_ms(self) -> float:
        """Milliseconds since this data was received from the feed."""
        return (time.time() - self.feed_timestamp) * 1000.0


CallbackType = Callable[[LiveMatchData], Coroutine[Any, Any, None]]


class OpticOddsLiveFeed:
    """
    Consumes darts live data from Optic Odds.

    Transport hierarchy:
    1. RabbitMQ (primary) — real-time push from Optic Odds broker.
    2. HTTP polling (fallback) — activated when RabbitMQ is unavailable or
       when the caller explicitly requests a snapshot.

    Thread safety: designed for use within a single asyncio event loop.
    Do not share instances across loops.
    """

    RABBITMQ_USER: str = settings.OPTIC_ODDS_RABBITMQ_USER
    RABBITMQ_PASS: str = settings.OPTIC_ODDS_RABBITMQ_PASS
    RABBITMQ_HOST: str = settings.OPTIC_ODDS_RABBITMQ_HOST

    # Polling interval when HTTP fallback is active
    HTTP_POLL_INTERVAL_SECONDS: float = 2.0

    # Number of consecutive HTTP errors before we log a warning
    HTTP_ERROR_WARN_THRESHOLD: int = 3

    def __init__(self) -> None:
        self._http_session: Optional[aiohttp.ClientSession] = None
        # match_id → list of subscriber callbacks
        self._subscriptions: dict[str, list[CallbackType]] = {}
        # match_id → latest cached state
        self._state_cache: dict[str, LiveMatchData] = {}
        # Whether the RabbitMQ connection is active
        self._rmq_connected: bool = False
        # Background polling tasks keyed on match_id
        self._poll_tasks: dict[str, asyncio.Task] = {}  # type: ignore[type-arg]
        self._http_error_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Initialise the HTTP session and attempt RabbitMQ connection.

        Gracefully degrades to HTTP-only if RabbitMQ credentials are absent
        or the broker is unreachable.
        """
        self._http_session = aiohttp.ClientSession(
            headers={
                "x-api-key": settings.OPTIC_ODDS_API_KEY,
            },
            timeout=aiohttp.ClientTimeout(total=10),
        )
        logger.info("optic_odds_http_session_created")

        if self.RABBITMQ_HOST and self.RABBITMQ_USER and self.RABBITMQ_PASS:
            await self._try_connect_rabbitmq()
        else:
            logger.warning(
                "optic_odds_rabbitmq_credentials_missing",
                hint="Set OPTIC_ODDS_RABBITMQ_* env vars to enable RabbitMQ transport.",
            )

    async def close(self) -> None:
        """Gracefully close all connections and cancel background tasks."""
        for task in self._poll_tasks.values():
            task.cancel()
        if self._poll_tasks:
            await asyncio.gather(*self._poll_tasks.values(), return_exceptions=True)
        self._poll_tasks.clear()

        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        logger.info("optic_odds_feed_closed")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_live_state(self, match_id: str) -> Optional[LiveMatchData]:
        """
        Get the current live match state.

        Returns the cached RabbitMQ state if available, otherwise falls back
        to a single HTTP poll. Returns None if the match is not found.

        Parameters
        ----------
        match_id:
            Optic Odds match identifier.

        Returns
        -------
        LiveMatchData or None
        """
        # Prefer cache (populated by RabbitMQ subscriber)
        if match_id in self._state_cache:
            cached = self._state_cache[match_id]
            logger.debug(
                "optic_odds_cache_hit",
                match_id=match_id,
                feed_age_ms=round(cached.feed_age_ms, 1),
            )
            return cached

        # HTTP fallback
        return await self._http_poll_match(match_id)

    async def subscribe_match(self, match_id: str, callback: CallbackType) -> None:
        """
        Subscribe to live updates for a match.

        When RabbitMQ is active, updates arrive via the broker. When only HTTP
        is available, a background polling task is started.

        Parameters
        ----------
        match_id:
            Optic Odds match identifier.
        callback:
            Async callable invoked with ``LiveMatchData`` on each update.
        """
        if match_id not in self._subscriptions:
            self._subscriptions[match_id] = []
        self._subscriptions[match_id].append(callback)

        if self._rmq_connected:
            logger.info("optic_odds_rmq_subscription", match_id=match_id)
            # In a real implementation, bind a RabbitMQ queue here
            # For now we fall through to HTTP polling as the implementation
            # of the actual AMQP binding depends on the Optic Odds queue topology
            await self._ensure_poll_task(match_id)
        else:
            logger.info(
                "optic_odds_http_poll_subscription",
                match_id=match_id,
                poll_interval_s=self.HTTP_POLL_INTERVAL_SECONDS,
            )
            await self._ensure_poll_task(match_id)

    async def unsubscribe_match(self, match_id: str) -> None:
        """Remove all subscriptions and stop polling for a match."""
        self._subscriptions.pop(match_id, None)
        task = self._poll_tasks.pop(match_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.info("optic_odds_unsubscribed", match_id=match_id)

    async def get_live_odds(self, sport: str = "darts") -> list[dict[str, Any]]:
        """
        Fetch current live odds via the Optic Odds HTTP API.

        Parameters
        ----------
        sport:
            Sport filter. Defaults to "darts".

        Returns
        -------
        list[dict]
            Raw odds objects as returned by the API. Empty list on error.
        """
        if not self._http_session:
            raise DartsDataError("OpticOddsLiveFeed not connected. Call connect() first.")

        url = f"{OPTIC_ODDS_HTTP_BASE}/odds"
        params: dict[str, Any] = {
            "sport": sport,
            "is_live": True,
        }

        try:
            async with self._http_session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    odds_list: list[dict[str, Any]] = data.get("data", [])
                    logger.info(
                        "optic_odds_live_odds_fetched",
                        sport=sport,
                        count=len(odds_list),
                    )
                    return odds_list
                else:
                    body = await resp.text()
                    logger.warning(
                        "optic_odds_http_error",
                        status=resp.status,
                        url=url,
                        body=body[:200],
                    )
                    return []
        except aiohttp.ClientError as exc:
            logger.error("optic_odds_http_client_error", error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _try_connect_rabbitmq(self) -> None:
        """
        Attempt to establish a RabbitMQ connection.

        Uses aio_pika if available; gracefully degrades to HTTP-only if not.
        The actual queue/exchange topology follows Optic Odds documentation:
        - Exchange: ``darts.live`` (topic)
        - Routing key: ``match.<match_id>``
        """
        try:
            import aio_pika  # type: ignore[import]

            amqp_url = _AMQP_PATTERN.format(
                user=self.RABBITMQ_USER,
                password=self.RABBITMQ_PASS,
                host=self.RABBITMQ_HOST,
            )
            connection = await aio_pika.connect_robust(
                amqp_url,
                login=self.RABBITMQ_USER,
                password=self.RABBITMQ_PASS,
                virtualhost="/",
                timeout=5,
            )
            self._rmq_connection = connection
            self._rmq_connected = True
            logger.info(
                "optic_odds_rabbitmq_connected",
                host=self.RABBITMQ_HOST,
            )
        except ImportError:
            logger.warning(
                "aio_pika_not_installed",
                hint="Install aio_pika to enable RabbitMQ transport.",
            )
            self._rmq_connected = False
        except Exception as exc:
            logger.warning(
                "optic_odds_rabbitmq_connect_failed",
                error=str(exc),
                hint="Falling back to HTTP polling.",
            )
            self._rmq_connected = False

    async def _ensure_poll_task(self, match_id: str) -> None:
        """Start an HTTP polling background task for a match if not already running."""
        if match_id in self._poll_tasks and not self._poll_tasks[match_id].done():
            return
        task = asyncio.create_task(
            self._poll_loop(match_id),
            name=f"optic_odds_poll_{match_id}",
        )
        self._poll_tasks[match_id] = task

    async def _poll_loop(self, match_id: str) -> None:
        """Background coroutine: poll HTTP every HTTP_POLL_INTERVAL_SECONDS."""
        logger.debug("optic_odds_poll_loop_started", match_id=match_id)
        while True:
            try:
                state = await self._http_poll_match(match_id)
                if state is not None:
                    self._state_cache[match_id] = state
                    callbacks = self._subscriptions.get(match_id, [])
                    for cb in callbacks:
                        try:
                            await cb(state)
                        except Exception as exc:
                            logger.error(
                                "optic_odds_callback_error",
                                match_id=match_id,
                                error=str(exc),
                            )
                    self._http_error_count = 0
            except asyncio.CancelledError:
                logger.debug("optic_odds_poll_loop_cancelled", match_id=match_id)
                raise
            except Exception as exc:
                self._http_error_count += 1
                if self._http_error_count >= self.HTTP_ERROR_WARN_THRESHOLD:
                    logger.warning(
                        "optic_odds_poll_repeated_errors",
                        match_id=match_id,
                        error=str(exc),
                        consecutive_errors=self._http_error_count,
                    )
                else:
                    logger.debug(
                        "optic_odds_poll_error",
                        match_id=match_id,
                        error=str(exc),
                    )

            await asyncio.sleep(self.HTTP_POLL_INTERVAL_SECONDS)

    async def _http_poll_match(self, match_id: str) -> Optional[LiveMatchData]:
        """
        Single HTTP GET to fetch current live state for a match.

        Returns None if the match is not found or the request fails.
        """
        if not self._http_session:
            logger.error("optic_odds_no_http_session", match_id=match_id)
            return None

        url = f"{OPTIC_ODDS_HTTP_BASE}/fixtures/{match_id}"
        try:
            async with self._http_session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_match_response(match_id, data)
                elif resp.status == 404:
                    logger.debug("optic_odds_match_not_found", match_id=match_id)
                    return None
                else:
                    body = await resp.text()
                    logger.warning(
                        "optic_odds_unexpected_status",
                        match_id=match_id,
                        status=resp.status,
                        body=body[:200],
                    )
                    return None
        except aiohttp.ClientError as exc:
            logger.error(
                "optic_odds_http_request_failed",
                match_id=match_id,
                error=str(exc),
            )
            return None

    def _parse_match_response(
        self, match_id: str, data: dict[str, Any]
    ) -> LiveMatchData:
        """
        Parse raw Optic Odds API response into a normalised LiveMatchData.

        The Optic Odds response schema is:
          data.score.home / data.score.away
          data.status (live/finished/not_started)
          data.period
          data.participants[0].id / data.participants[1].id
        """
        score = data.get("score") or {}
        participants = data.get("participants") or []
        server_id: Optional[str] = None
        if participants:
            # The first participant in a live darts match is typically "home"
            server_id = participants[0].get("id") if participants else None

        home_score_raw = score.get("home")
        away_score_raw = score.get("away")

        def _safe_int(v: Any) -> Optional[int]:
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        return LiveMatchData(
            match_id=match_id,
            home_score=_safe_int(home_score_raw),
            away_score=_safe_int(away_score_raw),
            legs_home=_safe_int(score.get("legs_home")),
            legs_away=_safe_int(score.get("legs_away")),
            sets_home=_safe_int(score.get("sets_home")),
            sets_away=_safe_int(score.get("sets_away")),
            current_server=server_id,
            period=data.get("period"),
            is_live=data.get("status") == "live",
            feed_timestamp=time.time(),
            raw=data,
        )
