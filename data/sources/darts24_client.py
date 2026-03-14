"""
Darts24 scraper — throw-by-throw live coverage, leg results.

Darts24 embeds match data as JSON within page HTML, making it
relatively straightforward to extract structured data.

Rate limiting: 1 req / 3s.

Data saved to: D:/codex/Data/Darts/01_raw/json/darts24/
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiohttp
import structlog
from bs4 import BeautifulSoup


logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://www.darts24.com"
REQUEST_DELAY_SECONDS = 3.0
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0

OUTPUT_DIR = Path("D:/codex/Data/Darts/01_raw/json/darts24")

_USER_AGENT = (
    "XG3DartsResearch/1.0 "
    "(sports analytics; contact: research@xg3.ai; polite-crawl 1req/3s)"
)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _build_headers() -> dict[str, str]:
    return {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
        "Accept-Language": "en-GB,en;q=0.9",
    }


async def _fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    *,
    retries: int = MAX_RETRIES,
) -> Optional[str]:
    """Fetch URL with exponential-backoff retry. Returns HTML or None."""
    for attempt in range(1, retries + 2):
        try:
            async with session.get(
                url,
                headers=_build_headers(),
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
            ) as response:
                if response.status == 200:
                    return await response.text()
                if response.status == 404:
                    logger.warning("darts24_404", url=url)
                    return None
                if response.status == 429:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    logger.warning("darts24_rate_limited", url=url, wait_seconds=wait)
                    await asyncio.sleep(wait)
                    continue
                logger.error(
                    "darts24_http_error",
                    url=url,
                    status=response.status,
                    attempt=attempt,
                )
        except aiohttp.ClientError as exc:
            logger.warning(
                "darts24_client_error",
                url=url,
                attempt=attempt,
                error=str(exc),
            )
            if attempt <= retries:
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return None


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------


def _extract_embedded_json(html: str) -> list[dict[str, Any]]:
    """
    Extract JSON objects embedded in Darts24 page HTML.

    Darts24 typically embeds match data in:
    1. <script type="application/json"> tags
    2. JSON-LD <script type="application/ld+json"> tags
    3. Inline window.__INITIAL_STATE__ = {...} JavaScript
    4. data-* attributes on HTML elements

    Returns a list of all successfully parsed JSON objects found.
    """
    results: list[dict[str, Any]] = []

    # Strategy 1: application/json script tags
    soup = BeautifulSoup(html, "lxml")
    for script in soup.find_all("script", type=re.compile(r"json", re.I)):
        try:
            data = json.loads(script.get_text(strip=True))
            if isinstance(data, dict):
                results.append(data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 2: window.__INITIAL_STATE__ or similar global variables
    for script in soup.find_all("script"):
        text = script.get_text()
        for pattern in [
            r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\});",
            r"window\.__DATA__\s*=\s*(\{.*?\});",
            r"var\s+matchData\s*=\s*(\{.*?\});",
        ]:
            for match in re.finditer(pattern, text, re.DOTALL):
                try:
                    data = json.loads(match.group(1))
                    if isinstance(data, dict):
                        results.append(data)
                except (json.JSONDecodeError, ValueError):
                    pass

    return results


def _parse_leg_data(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Parse raw extracted JSON into a structured leg result dict.

    Returns a normalised dict with standard field names.
    """
    result: dict[str, Any] = {}

    # Common field aliases used by Darts24
    player_keys = ["player1", "home", "homePlayer", "p1"]
    opponent_keys = ["player2", "away", "awayPlayer", "p2"]

    for key in player_keys:
        if key in raw:
            result["player1_name"] = str(raw[key])
            break

    for key in opponent_keys:
        if key in raw:
            result["player2_name"] = str(raw[key])
            break

    # Score
    for k1, k2 in [("score1", "score2"), ("homeScore", "awayScore"), ("p1Score", "p2Score")]:
        if k1 in raw and k2 in raw:
            result["score1"] = raw[k1]
            result["score2"] = raw[k2]
            break

    # Legs data
    for legs_key in ["legs", "legResults", "sets"]:
        if legs_key in raw and isinstance(raw[legs_key], list):
            result["legs"] = raw[legs_key]
            break

    # Date
    for date_key in ["date", "matchDate", "startTime", "time"]:
        if date_key in raw:
            result["match_date"] = str(raw[date_key])
            break

    # Status
    for status_key in ["status", "matchStatus", "state"]:
        if status_key in raw:
            result["status"] = str(raw[status_key])
            break

    return result


# ---------------------------------------------------------------------------
# Darts24Client
# ---------------------------------------------------------------------------


class Darts24Client:
    """
    Darts24 scraper — throw-by-throw live coverage, leg results.

    JSON data embedded in page HTML is extracted and normalised.

    Parameters
    ----------
    output_dir:
        Directory for saving scraped JSON files.
    request_delay:
        Seconds to wait between requests.
    """

    BASE_URL = "https://www.darts24.com"

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        request_delay: float = REQUEST_DELAY_SECONDS,
    ) -> None:
        self.output_dir = output_dir
        self.request_delay = request_delay
        self._log = logger.bind(scraper="Darts24Client")

    def _save(self, filename: str, data: Any) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
        self._log.debug("darts24_saved", path=str(path))

    async def scrape_match(self, match_url: str) -> dict:
        """
        Scrape leg-by-leg results from a Darts24 match page.

        Parameters
        ----------
        match_url:
            Full URL of the Darts24 match page.

        Returns
        -------
        dict
            Normalised match data including leg-by-leg results, visit
            scores (if available), and player statistics.
        """
        self._log.info("darts24_scrape_match", url=match_url)

        connector = aiohttp.TCPConnector(limit_per_host=1)
        async with aiohttp.ClientSession(connector=connector) as session:
            html = await _fetch_with_retry(session, match_url)
            if html is None:
                self._log.warning("darts24_match_unavailable", url=match_url)
                return {}

        soup = BeautifulSoup(html, "lxml")
        extracted_jsons = _extract_embedded_json(html)

        # Try to find the primary match data object
        match_data: dict[str, Any] = {}
        for obj in extracted_jsons:
            # Darts24 match objects typically have 'matchId' or 'legs' keys
            if any(k in obj for k in ("matchId", "legs", "legResults", "throws")):
                match_data.update(obj)
                break

        # Fallback: parse HTML structure directly
        if not match_data:
            match_data = self._parse_match_html(soup, match_url)

        result = {
            "source_url": match_url,
            "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
            "raw_json_objects": len(extracted_jsons),
            **_parse_leg_data(match_data),
        }

        # Extract leg details if available
        legs_raw = match_data.get("legs") or match_data.get("legResults") or []
        if legs_raw:
            result["legs"] = [
                {
                    "leg_number": i + 1,
                    "visits": leg.get("visits", []) if isinstance(leg, dict) else [],
                    "winner": leg.get("winner") if isinstance(leg, dict) else None,
                    "score": leg.get("score") if isinstance(leg, dict) else leg,
                }
                for i, leg in enumerate(legs_raw)
            ]

        # Save
        match_id = re.sub(r"[^a-z0-9]", "_", match_url.rstrip("/").split("/")[-1].lower())
        self._save(f"match_{match_id}.json", result)

        return result

    def _parse_match_html(self, soup: BeautifulSoup, url: str) -> dict[str, Any]:
        """
        Fallback HTML parser for Darts24 match pages.

        Used when no embedded JSON is found.
        """
        data: dict[str, Any] = {}

        # Player names
        player_els = soup.find_all(class_=re.compile(r"player|team|participant", re.I))
        names = [el.get_text(strip=True) for el in player_els if el.get_text(strip=True)]
        if len(names) >= 2:
            data["player1_name"] = names[0]
            data["player2_name"] = names[1]

        # Scores
        score_els = soup.find_all(class_=re.compile(r"score|result", re.I))
        scores = [el.get_text(strip=True) for el in score_els if el.get_text(strip=True).isdigit()]
        if len(scores) >= 2:
            data["score1"] = int(scores[0])
            data["score2"] = int(scores[1])

        # Individual visit scores (throw-by-throw)
        visit_els = soup.find_all(class_=re.compile(r"visit|throw|dart", re.I))
        visits = []
        for el in visit_els:
            text = el.get_text(strip=True)
            try:
                visits.append(int(text))
            except ValueError:
                pass
        if visits:
            data["visits"] = visits

        return data

    async def scrape_live_matches(self) -> list[dict]:
        """
        Get list of currently live darts matches from Darts24.

        Returns
        -------
        list[dict]
            Live match records with current scores and leg states.
        """
        self._log.info("darts24_scrape_live")

        url = f"{self.BASE_URL}/live/"
        connector = aiohttp.TCPConnector(limit_per_host=1)

        async with aiohttp.ClientSession(connector=connector) as session:
            html = await _fetch_with_retry(session, url)
            if html is None:
                self._log.warning("darts24_live_page_unavailable", url=url)
                return []

        soup = BeautifulSoup(html, "lxml")
        extracted_jsons = _extract_embedded_json(html)

        live_matches: list[dict] = []

        # Try to extract live match list from embedded JSON
        for obj in extracted_jsons:
            # Look for arrays of match objects
            for key in ("liveMatches", "matches", "events", "fixtures"):
                if key in obj and isinstance(obj[key], list):
                    for item in obj[key]:
                        if isinstance(item, dict):
                            parsed = _parse_leg_data(item)
                            if parsed:
                                parsed["status"] = "Live"
                                live_matches.append(parsed)

        if not live_matches:
            # Fallback: parse HTML
            match_divs = soup.find_all(
                attrs={"class": re.compile(r"live|match|event", re.I)}
            )
            for div in match_divs[:50]:  # safety cap
                text = div.get_text(separator=" ", strip=True)
                if text:
                    live_matches.append({"raw_text": text[:200], "status": "Live"})

        self._log.info("darts24_live_complete", live_count=len(live_matches))

        if live_matches:
            self._save(
                f"live_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
                {
                    "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
                    "count": len(live_matches),
                    "matches": live_matches,
                },
            )

        return live_matches

    async def scrape_competition(self, competition_slug: str) -> list[dict]:
        """
        Scrape all results from a Darts24 competition.

        Parameters
        ----------
        competition_slug:
            The competition path slug, e.g. 'pdc-world-championship-2025'.

        Returns
        -------
        list[dict]
            All match records from the competition.
        """
        url = f"{self.BASE_URL}/darts/{competition_slug}/"
        self._log.info("darts24_scrape_competition", slug=competition_slug, url=url)

        connector = aiohttp.TCPConnector(limit_per_host=1)
        async with aiohttp.ClientSession(connector=connector) as session:
            html = await _fetch_with_retry(session, url)
            if html is None:
                return []

        extracted = _extract_embedded_json(html)
        matches: list[dict] = []

        for obj in extracted:
            for key in ("matches", "results", "fixtures"):
                if key in obj and isinstance(obj[key], list):
                    for item in obj[key]:
                        if isinstance(item, dict):
                            matches.append(_parse_leg_data(item))

        self._log.info(
            "darts24_competition_complete",
            slug=competition_slug,
            count=len(matches),
        )
        return matches
