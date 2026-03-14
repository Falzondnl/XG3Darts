"""
DartConnect API client.

R2 premium mode — only activates when DARTCONNECT_API_KEY is configured.
Provides visit-level data, per-dart segment accuracies, and checkout routes.

This module replaces the earlier stub that only exposed a module-level
async function.  It now uses a class-based design consistent with the
rest of the data-source layer.

Error contract
--------------
- ``DartsDataError`` is raised at construction time if the API key is absent.
- ``DartsDataError`` is also raised for non-retryable API errors (4xx).
- Transient network errors are logged and returned as empty / None results
  after exhausting retries so the live engine can fall back gracefully.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp
import structlog

from app.config import settings
from engines.errors import DartsDataError

logger = structlog.get_logger(__name__)

# DartConnect base URL (from settings, override-able)
_DEFAULT_BASE_URL = "https://api.dartconnect.com"

# Maximum number of retry attempts on transient failures (429 / 5xx)
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 0.5   # seconds; doubles each attempt


@dataclass
class VisitRecord:
    """
    Normalised per-visit record from DartConnect.

    Attributes
    ----------
    visit_number:
        1-based visit index within the leg.
    player_id:
        DartConnect player identifier (may differ from canonical XG3 ID).
    score:
        Value scored this visit (0 on bust, 180 for max).
    remaining_before:
        Score remaining before this visit.
    remaining_after:
        Score remaining after this visit (same as remaining_before on bust).
    dart1 / dart2 / dart3:
        Individual dart segment strings (e.g. "T20", "D16", "25", "M" for miss).
        None when not available.
    checkout_attempted:
        True if the player was in checkout territory this visit.
    checkout_hit:
        True if the player successfully checked out.
    is_bust:
        True if the visit ended in a bust.
    """

    visit_number: int
    player_id: str
    score: int
    remaining_before: int
    remaining_after: int
    dart1: Optional[str] = None
    dart2: Optional[str] = None
    dart3: Optional[str] = None
    checkout_attempted: bool = False
    checkout_hit: bool = False
    is_bust: bool = False

    @property
    def is_180(self) -> bool:
        """True when the visit scored the maximum 180."""
        return self.score == 180


class DartConnectClient:
    """
    DartConnect premium API client (R2 mode).

    Raises
    ------
    DartsDataError
        At construction time if DARTCONNECT_API_KEY is not configured.
    """

    def __init__(self) -> None:
        self.api_key: str = settings.DARTCONNECT_API_KEY
        if not self.api_key:
            raise DartsDataError(
                "DARTCONNECT_API_KEY not configured — R2 mode unavailable. "
                "Set DARTCONNECT_API_KEY in the environment or .env file."
            )
        self.base_url: str = settings.DARTCONNECT_BASE_URL or _DEFAULT_BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return (or lazily create) the shared aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_match_visit_data(
        self,
        match_id: str,
        tournament_id: str,
    ) -> list[VisitRecord]:
        """
        Fetch per-visit data for a match from DartConnect.

        Parameters
        ----------
        match_id:
            DartConnect match identifier (dartconnect_match_id in darts_matches).
        tournament_id:
            DartConnect tournament identifier.

        Returns
        -------
        list[VisitRecord]
            Ordered list of all visits, both players, across all legs.
            Empty list if the match has no visit data or an error occurs.

        Raises
        ------
        DartsDataError
            On 404 (match not found) or 403 (unauthorised).
        """
        url = f"{self.base_url}/v1/tournaments/{tournament_id}/matches/{match_id}/visits"
        raw = await self._request_with_retry("GET", url)
        if raw is None:
            return []

        visits_raw: list[dict[str, Any]] = raw.get("visits") or raw.get("data") or []
        records: list[VisitRecord] = []
        for v in visits_raw:
            try:
                records.append(self._parse_visit(v))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "dartconnect_visit_parse_error",
                    match_id=match_id,
                    error=str(exc),
                    raw_visit=v,
                )
        logger.info(
            "dartconnect_visits_fetched",
            match_id=match_id,
            tournament_id=tournament_id,
            visit_count=len(records),
        )
        return records

    async def get_player_segment_accuracies(
        self,
        player_id: str,
        since_date: str,
    ) -> dict[str, float]:
        """
        Retrieve per-segment hit rates from DartConnect historical data.

        Parameters
        ----------
        player_id:
            DartConnect player identifier.
        since_date:
            ISO 8601 date string (``YYYY-MM-DD``). Only data from this date
            forward is included.

        Returns
        -------
        dict[str, float]
            Maps segment name (e.g. ``"T20"``, ``"D16"``, ``"25"``) to hit
            rate in [0, 1].  Missing segments are not included.
        """
        url = (
            f"{self.base_url}/v1/players/{player_id}/segment-accuracies"
            f"?since={since_date}"
        )
        raw = await self._request_with_retry("GET", url)
        if raw is None:
            return {}

        accuracies_raw: list[dict[str, Any]] = (
            raw.get("segment_accuracies") or raw.get("data") or []
        )
        result: dict[str, float] = {}
        for entry in accuracies_raw:
            segment = entry.get("segment")
            rate = entry.get("hit_rate")
            if segment and rate is not None:
                try:
                    result[segment] = float(rate)
                except (TypeError, ValueError):
                    pass

        logger.info(
            "dartconnect_segment_accuracies_fetched",
            player_id=player_id,
            since_date=since_date,
            segment_count=len(result),
        )
        return result

    async def get_live_match_state(self, match_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch the current live state for an in-progress match.

        Used by the live engine when DartConnect is the primary feed source.
        Returns None if the match is not currently live or not found.

        Parameters
        ----------
        match_id:
            DartConnect match identifier.

        Returns
        -------
        dict or None
            Raw response payload. Structure:
              {
                "match_id": "...",
                "status": "live",
                "current_leg": 3,
                "legs_home": 1,
                "legs_away": 1,
                "score_home": 212,
                "score_away": 187,
                "current_thrower": "home" | "away",
                "last_visit": { "score": 60, "dart1": "T20", ... },
                "feed_timestamp": "2025-01-01T12:34:56Z"
              }
        """
        url = f"{self.base_url}/v1/matches/{match_id}/live"
        raw = await self._request_with_retry("GET", url)
        if raw is None:
            return None

        if raw.get("status") not in ("live", "in_progress"):
            logger.debug(
                "dartconnect_match_not_live",
                match_id=match_id,
                status=raw.get("status"),
            )
            return None

        logger.debug(
            "dartconnect_live_state_fetched",
            match_id=match_id,
            current_leg=raw.get("current_leg"),
        )
        return raw

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        json_body: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Make an HTTP request with exponential backoff on transient failures.

        Returns the parsed JSON body on success, None on exhausted retries.
        Raises DartsDataError on definitive 4xx errors.
        """
        session = await self._get_session()
        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES):
            try:
                async with session.request(
                    method, url, json=json_body
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()

                    if resp.status == 404:
                        logger.debug(
                            "dartconnect_not_found",
                            url=url,
                            status=404,
                        )
                        return None

                    if resp.status == 403:
                        body = await resp.text()
                        raise DartsDataError(
                            f"DartConnect unauthorised (403) for {url!r}: {body[:200]}"
                        )

                    if resp.status == 401:
                        raise DartsDataError(
                            f"DartConnect API key invalid (401). "
                            "Check DARTCONNECT_API_KEY."
                        )

                    if resp.status == 429:
                        # Rate limited — back off
                        body = await resp.text()
                        logger.warning(
                            "dartconnect_rate_limited",
                            url=url,
                            attempt=attempt + 1,
                        )
                        backoff = _RETRY_BACKOFF_BASE * (2 ** attempt)
                        await asyncio.sleep(backoff)
                        continue

                    if resp.status >= 500:
                        body = await resp.text()
                        logger.warning(
                            "dartconnect_server_error",
                            url=url,
                            status=resp.status,
                            attempt=attempt + 1,
                        )
                        backoff = _RETRY_BACKOFF_BASE * (2 ** attempt)
                        await asyncio.sleep(backoff)
                        continue

                    # Unexpected status
                    body = await resp.text()
                    logger.error(
                        "dartconnect_unexpected_status",
                        url=url,
                        status=resp.status,
                        body=body[:200],
                    )
                    return None

            except DartsDataError:
                raise
            except aiohttp.ClientError as exc:
                last_exc = exc
                logger.warning(
                    "dartconnect_request_error",
                    url=url,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))

        logger.error(
            "dartconnect_request_failed_all_retries",
            url=url,
            retries=_MAX_RETRIES,
            last_error=str(last_exc),
        )
        return None

    def _parse_visit(self, raw: dict[str, Any]) -> VisitRecord:
        """
        Parse a raw DartConnect visit dict into a VisitRecord.

        DartConnect visit structure:
          {
            "visit_number": 1,
            "player_id": "dc-123",
            "score": 60,
            "remaining_before": 501,
            "remaining_after": 441,
            "dart1": "T20", "dart2": "M", "dart3": "M",
            "checkout_attempted": false,
            "checkout_hit": false,
            "is_bust": false
          }
        """
        remaining_before = int(raw.get("remaining_before", 501))
        remaining_after = int(raw.get("remaining_after", remaining_before))
        score = int(raw.get("score", remaining_before - remaining_after))
        is_bust = bool(raw.get("is_bust", False))

        return VisitRecord(
            visit_number=int(raw.get("visit_number", 0)),
            player_id=str(raw.get("player_id", "")),
            score=score,
            remaining_before=remaining_before,
            remaining_after=remaining_after,
            dart1=raw.get("dart1"),
            dart2=raw.get("dart2"),
            dart3=raw.get("dart3"),
            checkout_attempted=bool(raw.get("checkout_attempted", False)),
            checkout_hit=bool(raw.get("checkout_hit", False)),
            is_bust=is_bust,
        )
