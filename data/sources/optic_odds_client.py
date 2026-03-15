"""
Optic Odds live feed client — complete rebuild.

Provides two transport modes:
  1. RabbitMQ / Copilot queue  — primary, low-latency, push-based.
     Flow: POST /copilot/queue/start → get queue_name →
           consume from amqp://user:pass@host:5672/api (virtualhost=api)
  2. HTTP REST polling + SSE stream — fallback when RabbitMQ unavailable.

Auth: X-Api-Key header (Optic Odds API v3).

Usage
-----
    feed = OpticOddsLiveFeed()
    await feed.connect()

    # Discover active darts fixtures
    fixtures = await feed.get_active_fixtures()

    # Subscribe to live updates
    await feed.subscribe_match("optic-match-id", my_callback)

    # Snapshot poll
    state = await feed.get_live_state("optic-match-id")

    await feed.close()
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Coroutine, Optional

import aiohttp
import structlog

from app.config import settings
from engines.errors import DartsDataError

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPTIC_ODDS_BASE = "https://api.opticodds.com/api/v3"

# Copilot RabbitMQ virtualhost is "api" (not "/")
_AMQP_VHOST = "api"
_AMQP_PORT = 5672

# SSE reconnect delay on error
_SSE_RECONNECT_DELAY_S = 5.0

# HTTP polling interval when both RabbitMQ and SSE are unavailable
_HTTP_POLL_INTERVAL_S = 2.0

# Max consecutive HTTP errors before escalating to WARNING
_HTTP_ERROR_WARN_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class OpticOddsFixture:
    """
    Normalised fixture record from Optic Odds /fixtures/active.

    The ``optic_id`` is used as the key for all subscription/polling calls.
    """

    optic_id: str
    sport: str
    home_team: str
    away_team: str
    home_id: str
    away_id: str
    start_time: str           # ISO-8601
    league: str
    league_id: str
    status: str               # not_started | live | finished
    is_live: bool
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveMatchData:
    """
    Normalised live match state returned by both RabbitMQ and HTTP transports.

    This is the interface contract between the Optic Odds feed and the
    DartsLiveEngine — transport-agnostic by design.
    """

    match_id: str                           # Optic Odds fixture ID
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    legs_home: Optional[int] = None
    legs_away: Optional[int] = None
    sets_home: Optional[int] = None
    sets_away: Optional[int] = None
    current_server: Optional[str] = None   # home_id or away_id
    period: Optional[str] = None           # e.g. "leg_3", "set_2_leg_1"
    is_live: bool = False
    home_id: Optional[str] = None
    away_id: Optional[str] = None
    home_name: Optional[str] = None
    away_name: Optional[str] = None
    feed_timestamp: float = field(default_factory=time.time)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def feed_age_ms(self) -> float:
        """Milliseconds since this data was received from the feed."""
        return (time.time() - self.feed_timestamp) * 1000.0


CallbackType = Callable[[LiveMatchData], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class OpticOddsLiveFeed:
    """
    Consumes darts live data from Optic Odds API v3.

    Transport hierarchy:
      1. RabbitMQ via Copilot queue (primary) — sub-second latency.
      2. SSE stream (secondary) — streamed HTTP events for odds changes.
      3. HTTP REST polling (fallback) — 2s interval snapshot polling.

    All transports emit ``LiveMatchData`` objects through the same callback
    interface, making the live engine transport-agnostic.

    Thread safety: designed for a single asyncio event loop.
    """

    def __init__(self) -> None:
        self._api_key: str = settings.OPTIC_ODDS_API_KEY
        self._rmq_user: str = settings.OPTIC_ODDS_RABBITMQ_USER
        self._rmq_pass: str = settings.OPTIC_ODDS_RABBITMQ_PASS
        self._rmq_host: str = settings.OPTIC_ODDS_RABBITMQ_HOST

        self._http_session: Optional[aiohttp.ClientSession] = None

        # match_id → subscriber callbacks
        self._subscriptions: dict[str, list[CallbackType]] = {}
        # match_id → latest cached state
        self._state_cache: dict[str, LiveMatchData] = {}

        self._rmq_connected: bool = False
        self._rmq_queue_name: Optional[str] = None

        # Background tasks
        self._poll_tasks: dict[str, asyncio.Task[None]] = {}
        self._sse_task: Optional[asyncio.Task[None]] = None
        self._rmq_consume_task: Optional[asyncio.Task[None]] = None

        self._http_error_counts: dict[str, int] = {}
        self._log = logger.bind(component="OpticOddsLiveFeed")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Open the HTTP session and attempt RabbitMQ Copilot queue setup.

        Degrades gracefully to SSE → HTTP polling if RabbitMQ unavailable.
        """
        self._http_session = aiohttp.ClientSession(
            headers={
                "X-Api-Key": self._api_key,
                "Accept": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=15),
        )
        self._log.info("optic_odds_http_session_created")

        if self._rmq_host and self._rmq_user and self._rmq_pass:
            await self._setup_copilot_queue()
        else:
            self._log.warning(
                "optic_odds_rmq_credentials_missing",
                hint="Set OPTIC_ODDS_RABBITMQ_* env vars for RabbitMQ transport.",
            )

    async def close(self) -> None:
        """Cancel all background tasks and close connections."""
        tasks_to_cancel: list[asyncio.Task[None]] = []

        if self._sse_task:
            tasks_to_cancel.append(self._sse_task)
        if self._rmq_consume_task:
            tasks_to_cancel.append(self._rmq_consume_task)
        tasks_to_cancel.extend(self._poll_tasks.values())

        for task in tasks_to_cancel:
            task.cancel()
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        self._poll_tasks.clear()
        self._sse_task = None
        self._rmq_consume_task = None

        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

        self._log.info("optic_odds_feed_closed")

    # ------------------------------------------------------------------
    # Public API — fixture discovery
    # ------------------------------------------------------------------

    async def get_active_fixtures(self, sport: str = "darts") -> list[OpticOddsFixture]:
        """
        Fetch currently active/upcoming darts fixtures from Optic Odds.

        Calls ``GET /fixtures/active?sport=darts``.

        Parameters
        ----------
        sport:
            Optic Odds sport slug. Default ``"darts"``.

        Returns
        -------
        list[OpticOddsFixture]
            Parsed fixture list. Raises ``DartsDataError`` if the API call fails.
        """
        self._ensure_connected()
        url = f"{OPTIC_ODDS_BASE}/fixtures/active"
        params: dict[str, str] = {"sport": sport}

        try:
            async with self._http_session.get(url, params=params) as resp:  # type: ignore[union-attr]
                if resp.status != 200:
                    body = await resp.text()
                    raise DartsDataError(
                        f"Optic Odds /fixtures/active returned HTTP {resp.status}: {body[:300]}"
                    )
                payload = await resp.json()
        except aiohttp.ClientError as exc:
            raise DartsDataError(f"Optic Odds HTTP error fetching fixtures: {exc}") from exc

        raw_fixtures: list[dict[str, Any]] = payload.get("data", [])
        self._log.info(
            "optic_odds_fixtures_fetched",
            sport=sport,
            count=len(raw_fixtures),
        )
        return [self._parse_fixture(f) for f in raw_fixtures]

    async def get_fixture(self, fixture_id: str) -> Optional[LiveMatchData]:
        """
        Fetch a single fixture's current state by its Optic Odds ID.

        Calls ``GET /fixtures/{fixture_id}``.

        Returns None if the fixture is not found.
        """
        self._ensure_connected()
        url = f"{OPTIC_ODDS_BASE}/fixtures/{fixture_id}"

        try:
            async with self._http_session.get(url) as resp:  # type: ignore[union-attr]
                if resp.status == 404:
                    self._log.debug("optic_odds_fixture_not_found", fixture_id=fixture_id)
                    return None
                if resp.status != 200:
                    body = await resp.text()
                    self._log.warning(
                        "optic_odds_fixture_error",
                        fixture_id=fixture_id,
                        status=resp.status,
                        body=body[:200],
                    )
                    return None
                data = await resp.json()
        except aiohttp.ClientError as exc:
            self._log.error(
                "optic_odds_http_error",
                fixture_id=fixture_id,
                error=str(exc),
            )
            return None

        fixture_data: dict[str, Any] = data.get("data", data)
        return self._parse_live_state(fixture_id, fixture_data)

    async def get_live_odds(self, fixture_id: str) -> list[dict[str, Any]]:
        """
        Fetch current odds for a fixture.

        Calls ``GET /odds?fixture_id={fixture_id}``.

        Returns raw odds list (normalised by the caller).
        """
        self._ensure_connected()
        url = f"{OPTIC_ODDS_BASE}/odds"
        params = {"fixture_id": fixture_id, "sport": "darts"}

        try:
            async with self._http_session.get(url, params=params) as resp:  # type: ignore[union-attr]
                if resp.status != 200:
                    return []
                payload = await resp.json()
                return payload.get("data", [])
        except aiohttp.ClientError as exc:
            self._log.error("optic_odds_odds_error", fixture_id=fixture_id, error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Public API — subscriptions / live state
    # ------------------------------------------------------------------

    async def get_live_state(self, match_id: str) -> Optional[LiveMatchData]:
        """
        Return the current live state for a match.

        Prefers the RabbitMQ/SSE cache; falls back to HTTP GET.

        Parameters
        ----------
        match_id:
            Optic Odds fixture identifier.
        """
        if match_id in self._state_cache:
            cached = self._state_cache[match_id]
            self._log.debug(
                "optic_odds_cache_hit",
                match_id=match_id,
                feed_age_ms=round(cached.feed_age_ms, 1),
            )
            return cached

        return await self.get_fixture(match_id)

    async def subscribe_match(self, match_id: str, callback: CallbackType) -> None:
        """
        Subscribe to live updates for a match.

        If RabbitMQ is active, updates arrive via the broker.
        Otherwise a background HTTP polling task is started.

        Parameters
        ----------
        match_id:
            Optic Odds fixture identifier.
        callback:
            Async callable invoked with ``LiveMatchData`` on each update.
        """
        self._subscriptions.setdefault(match_id, []).append(callback)

        if not self._rmq_connected:
            self._log.info(
                "optic_odds_http_poll_fallback",
                match_id=match_id,
                poll_interval_s=_HTTP_POLL_INTERVAL_S,
            )
            await self._ensure_poll_task(match_id)
        else:
            self._log.info("optic_odds_rmq_subscribed", match_id=match_id)

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
        self._log.info("optic_odds_unsubscribed", match_id=match_id)

    # ------------------------------------------------------------------
    # RabbitMQ — Copilot queue flow
    # ------------------------------------------------------------------

    async def _setup_copilot_queue(self) -> None:
        """
        Register a Copilot queue with Optic Odds and start consuming.

        Flow:
          1. POST /copilot/queue/start  →  {queue_name: "xg3-darts-<uuid>"}
          2. Connect aio_pika to amqp://user:pass@host:5672/api
          3. Consume from the dynamic queue name
        """
        self._ensure_connected()
        url = f"{OPTIC_ODDS_BASE}/copilot/queue/start"
        payload = {"sport": "darts"}

        try:
            async with self._http_session.post(url, json=payload) as resp:  # type: ignore[union-attr]
                if resp.status not in (200, 201):
                    body = await resp.text()
                    self._log.warning(
                        "optic_odds_copilot_queue_failed",
                        status=resp.status,
                        body=body[:300],
                        hint="Falling back to HTTP polling / SSE.",
                    )
                    return
                data = await resp.json()
        except aiohttp.ClientError as exc:
            self._log.warning(
                "optic_odds_copilot_http_error",
                error=str(exc),
                hint="Falling back to HTTP polling / SSE.",
            )
            return

        queue_name: Optional[str] = (
            data.get("queue_name")
            or data.get("data", {}).get("queue_name")
        )
        if not queue_name:
            self._log.warning(
                "optic_odds_no_queue_name",
                response=data,
                hint="Falling back to HTTP polling / SSE.",
            )
            return

        self._rmq_queue_name = queue_name
        self._log.info("optic_odds_copilot_queue_registered", queue_name=queue_name)

        # Start RabbitMQ consumer in background
        self._rmq_consume_task = asyncio.create_task(
            self._rmq_consumer_loop(queue_name),
            name="optic_odds_rmq_consumer",
        )

    async def _rmq_consumer_loop(self, queue_name: str) -> None:
        """
        Persistent RabbitMQ consumer.  Reconnects automatically on failure.

        Virtualhost: ``api``
        Auth:        OPTIC_ODDS_RABBITMQ_USER / OPTIC_ODDS_RABBITMQ_PASS
        """
        try:
            import aio_pika  # type: ignore[import]
        except ImportError:
            self._log.warning(
                "aio_pika_not_installed",
                hint="pip install aio-pika to enable RabbitMQ transport.",
            )
            return

        amqp_url = (
            f"amqp://{self._rmq_user}:{self._rmq_pass}"
            f"@{self._rmq_host}:{_AMQP_PORT}/{_AMQP_VHOST}"
        )

        while True:
            try:
                connection = await aio_pika.connect_robust(
                    amqp_url,
                    timeout=10,
                )
                self._rmq_connected = True
                self._log.info(
                    "optic_odds_rmq_connected",
                    host=self._rmq_host,
                    vhost=_AMQP_VHOST,
                    queue=queue_name,
                )

                async with connection:
                    channel = await connection.channel()
                    await channel.set_qos(prefetch_count=10)

                    queue = await channel.get_queue(queue_name, ensure=False)
                    async with queue.iterator() as q_iter:
                        async for message in q_iter:
                            async with message.process():
                                await self._handle_rmq_message(message.body)

            except asyncio.CancelledError:
                self._log.info("optic_odds_rmq_consumer_cancelled")
                self._rmq_connected = False
                return
            except Exception as exc:
                self._rmq_connected = False
                self._log.warning(
                    "optic_odds_rmq_error",
                    error=str(exc),
                    retry_in_s=_SSE_RECONNECT_DELAY_S,
                )
                await asyncio.sleep(_SSE_RECONNECT_DELAY_S)

    async def _handle_rmq_message(self, body: bytes) -> None:
        """Parse and dispatch a raw RabbitMQ message body."""
        try:
            payload = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._log.warning("optic_odds_rmq_parse_error", error=str(exc))
            return

        # Optic Odds Copilot messages: {"fixture_id": "...", "data": {...}}
        fixture_id: Optional[str] = (
            payload.get("fixture_id")
            or payload.get("id")
            or (payload.get("data") or {}).get("fixture_id")
        )
        if not fixture_id:
            self._log.debug("optic_odds_rmq_no_fixture_id", payload_keys=list(payload.keys()))
            return

        inner: dict[str, Any] = payload.get("data") or payload
        state = self._parse_live_state(fixture_id, inner)
        if state:
            self._state_cache[fixture_id] = state
            await self._dispatch_callbacks(fixture_id, state)

    # ------------------------------------------------------------------
    # SSE stream
    # ------------------------------------------------------------------

    async def start_sse_stream(self, sport: str = "darts") -> None:
        """
        Start an SSE consumer for all live darts odds changes.

        Calls ``GET /stream/odds/{sport}``.
        Runs as a background task; reconnects automatically.
        """
        if self._sse_task and not self._sse_task.done():
            return
        self._sse_task = asyncio.create_task(
            self._sse_consume_loop(sport),
            name=f"optic_odds_sse_{sport}",
        )
        self._log.info("optic_odds_sse_started", sport=sport)

    async def _sse_consume_loop(self, sport: str) -> None:
        """Persistent SSE consumer with reconnect."""
        url = f"{OPTIC_ODDS_BASE}/stream/odds/{sport}"
        self._ensure_connected()

        while True:
            try:
                async with self._http_session.get(  # type: ignore[union-attr]
                    url,
                    headers={"Accept": "text/event-stream"},
                    timeout=aiohttp.ClientTimeout(total=None, connect=15),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        self._log.warning(
                            "optic_odds_sse_bad_status",
                            status=resp.status,
                            body=body[:200],
                        )
                        await asyncio.sleep(_SSE_RECONNECT_DELAY_S)
                        continue

                    self._log.info("optic_odds_sse_connected", sport=sport)
                    async for event_data in self._parse_sse_stream(resp):
                        await self._handle_sse_event(event_data)

            except asyncio.CancelledError:
                self._log.info("optic_odds_sse_cancelled")
                return
            except Exception as exc:
                self._log.warning(
                    "optic_odds_sse_error",
                    error=str(exc),
                    retry_in_s=_SSE_RECONNECT_DELAY_S,
                )
                await asyncio.sleep(_SSE_RECONNECT_DELAY_S)

    async def _parse_sse_stream(
        self, resp: aiohttp.ClientResponse
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed JSON objects from an SSE stream response."""
        buffer = ""
        async for line_bytes in resp.content:
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
            if line.startswith("data:"):
                raw_json = line[5:].strip()
                if raw_json == "[DONE]" or not raw_json:
                    continue
                try:
                    yield json.loads(raw_json)
                except json.JSONDecodeError:
                    self._log.debug("optic_odds_sse_non_json_line", line=line[:100])
            elif not line:
                # blank line = end of event; buffer already dispatched
                buffer = ""

    async def _handle_sse_event(self, payload: dict[str, Any]) -> None:
        """Parse and dispatch an SSE event payload."""
        fixture_id: Optional[str] = (
            payload.get("fixture_id")
            or payload.get("id")
            or (payload.get("fixture") or {}).get("id")
        )
        if not fixture_id:
            return

        state = self._parse_live_state(fixture_id, payload)
        if state:
            self._state_cache[fixture_id] = state
            await self._dispatch_callbacks(fixture_id, state)

    # ------------------------------------------------------------------
    # HTTP polling fallback
    # ------------------------------------------------------------------

    async def _ensure_poll_task(self, match_id: str) -> None:
        """Start an HTTP polling background task if not already running."""
        existing = self._poll_tasks.get(match_id)
        if existing and not existing.done():
            return
        task = asyncio.create_task(
            self._poll_loop(match_id),
            name=f"optic_odds_poll_{match_id}",
        )
        self._poll_tasks[match_id] = task

    async def _poll_loop(self, match_id: str) -> None:
        """Background HTTP poll every _HTTP_POLL_INTERVAL_S seconds."""
        self._log.debug("optic_odds_poll_started", match_id=match_id)
        while True:
            try:
                state = await self.get_fixture(match_id)
                if state is not None:
                    self._state_cache[match_id] = state
                    await self._dispatch_callbacks(match_id, state)
                    self._http_error_counts[match_id] = 0
            except asyncio.CancelledError:
                self._log.debug("optic_odds_poll_cancelled", match_id=match_id)
                raise
            except Exception as exc:
                count = self._http_error_counts.get(match_id, 0) + 1
                self._http_error_counts[match_id] = count
                if count >= _HTTP_ERROR_WARN_THRESHOLD:
                    self._log.warning(
                        "optic_odds_poll_repeated_errors",
                        match_id=match_id,
                        error=str(exc),
                        consecutive=count,
                    )
                else:
                    self._log.debug(
                        "optic_odds_poll_error",
                        match_id=match_id,
                        error=str(exc),
                    )

            await asyncio.sleep(_HTTP_POLL_INTERVAL_S)

    # ------------------------------------------------------------------
    # Callback dispatch
    # ------------------------------------------------------------------

    async def _dispatch_callbacks(self, match_id: str, state: LiveMatchData) -> None:
        """Invoke all subscriber callbacks for a match update."""
        for callback in self._subscriptions.get(match_id, []):
            try:
                await callback(state)
            except Exception as exc:
                self._log.error(
                    "optic_odds_callback_error",
                    match_id=match_id,
                    error=str(exc),
                )

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_fixture(raw: dict[str, Any]) -> OpticOddsFixture:
        """Parse a raw fixture dict into an ``OpticOddsFixture``."""
        participants: list[dict[str, Any]] = raw.get("participants") or []
        home: dict[str, Any] = participants[0] if len(participants) > 0 else {}
        away: dict[str, Any] = participants[1] if len(participants) > 1 else {}

        league: dict[str, Any] = raw.get("league") or raw.get("tournament") or {}

        return OpticOddsFixture(
            optic_id=str(raw.get("id", "")),
            sport=str(raw.get("sport", "darts")),
            home_team=str(home.get("name", home.get("id", ""))),
            away_team=str(away.get("name", away.get("id", ""))),
            home_id=str(home.get("id", "")),
            away_id=str(away.get("id", "")),
            start_time=str(raw.get("start_date", raw.get("start_time", ""))),
            league=str(league.get("name", "")),
            league_id=str(league.get("id", "")),
            status=str(raw.get("status", "not_started")),
            is_live=raw.get("status") == "live",
            raw=raw,
        )

    @staticmethod
    def _parse_live_state(
        match_id: str, data: dict[str, Any]
    ) -> Optional[LiveMatchData]:
        """
        Parse a raw Optic Odds fixture/event payload into ``LiveMatchData``.

        Handles both the flat fixture response and nested Copilot/SSE payloads.
        """
        if not data:
            return None

        score: dict[str, Any] = data.get("score") or {}
        participants: list[dict[str, Any]] = data.get("participants") or []

        home_p = participants[0] if len(participants) > 0 else {}
        away_p = participants[1] if len(participants) > 1 else {}

        def _int(v: Any) -> Optional[int]:
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        # Score may be in various locations depending on event type
        home_score = _int(score.get("home") or data.get("home_score"))
        away_score = _int(score.get("away") or data.get("away_score"))
        legs_home = _int(score.get("legs_home") or score.get("home_legs"))
        legs_away = _int(score.get("legs_away") or score.get("away_legs"))
        sets_home = _int(score.get("sets_home") or score.get("home_sets"))
        sets_away = _int(score.get("sets_away") or score.get("away_sets"))

        return LiveMatchData(
            match_id=match_id,
            home_score=home_score,
            away_score=away_score,
            legs_home=legs_home,
            legs_away=legs_away,
            sets_home=sets_home,
            sets_away=sets_away,
            current_server=str(home_p.get("id", "")) or None,
            period=data.get("period") or data.get("current_period"),
            is_live=data.get("status") == "live",
            home_id=str(home_p.get("id", "")) or None,
            away_id=str(away_p.get("id", "")) or None,
            home_name=str(home_p.get("name", "")) or None,
            away_name=str(away_p.get("name", "")) or None,
            feed_timestamp=time.time(),
            raw=data,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """Raise if connect() has not been called."""
        if self._http_session is None or self._http_session.closed:
            raise DartsDataError(
                "OpticOddsLiveFeed not connected. Call await feed.connect() first."
            )
