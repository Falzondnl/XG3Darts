"""
DartsDatabase scraper.

Fetches event and player pages from https://www.dartsdatabase.co.uk

Target pages:
- Event pages: /event/{event_id} — match results for tournaments
- Player pages: /player/{player_id} — career stats, head-to-head

Data extracted:
- Match results (player names, scores, round, date)
- Career statistics per player (win%, legs won, averages when present)
- Head-to-head records between players

Rate limiting: 1 request per 3 seconds.
Output: D:/codex/Data/Darts/02_processed/json/dartsdatabase/
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiohttp
import structlog
from bs4 import BeautifulSoup


logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://www.dartsdatabase.co.uk"
EVENT_URL = "{base}/event/{event_id}"
PLAYER_URL = "{base}/player/{player_id}"

REQUEST_DELAY_SECONDS = 3.0
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 6.0

OUTPUT_DIR = Path("D:/codex/Data/Darts/02_processed/json/dartsdatabase")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _build_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "XG3DartsResearch/1.0 "
            "(sports analytics; contact: research@xg3.ai; "
            "polite-crawl 1req/3s)"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-GB,en;q=0.9",
    }


async def _fetch(
    session: aiohttp.ClientSession,
    url: str,
    *,
    retries: int = MAX_RETRIES,
) -> Optional[str]:
    """Fetch a URL with retry/backoff.  Returns HTML or None."""
    for attempt in range(1, retries + 2):
        try:
            async with session.get(
                url,
                headers=_build_headers(),
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
                allow_redirects=True,
            ) as resp:
                if resp.status == 200:
                    return await resp.text()
                if resp.status in (404, 410):
                    logger.warning("dartsdatabase_not_found", url=url, status=resp.status)
                    return None
                if resp.status == 429 or resp.status >= 500:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    logger.warning(
                        "dartsdatabase_retry",
                        url=url,
                        status=resp.status,
                        wait=wait,
                    )
                    await asyncio.sleep(wait)
        except aiohttp.ClientError as exc:
            if attempt <= retries:
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
            else:
                logger.error("dartsdatabase_client_error", url=url, error=str(exc))
    return None


# ---------------------------------------------------------------------------
# HTML parsers
# ---------------------------------------------------------------------------

def parse_event_page(
    html: str,
    event_id: str,
) -> dict[str, Any]:
    """
    Parse a DartsDatabase event page.

    Parameters
    ----------
    html:
        Raw HTML content.
    event_id:
        Event identifier.

    Returns
    -------
    dict[str, Any]
        Parsed event data including match results.
    """
    soup = BeautifulSoup(html, "lxml")
    data: dict[str, Any] = {
        "event_id": event_id,
        "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_url": EVENT_URL.format(base=BASE_URL, event_id=event_id),
        "matches": [],
    }

    # Event name typically in <h1> or <title>
    h1 = soup.find("h1")
    if h1:
        data["event_name"] = h1.get_text(strip=True)

    # Look for results tables
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows[1:]:  # skip header row
            cells = row.find_all(["td", "th"])
            if len(cells) >= 3:
                match_record = _parse_match_row(cells)
                if match_record:
                    data["matches"].append(match_record)

    logger.debug(
        "dartsdatabase_event_parsed",
        event_id=event_id,
        match_count=len(data["matches"]),
    )
    return data


def _parse_match_row(cells: list) -> Optional[dict[str, Any]]:
    """Parse a table row into a match record dict."""
    texts = [c.get_text(strip=True) for c in cells]
    if len(texts) < 3:
        return None

    # Try to find round, player names, scores
    # Common layout: Round | Player1 | Score | Player2 | ...
    record: dict[str, Any] = {}

    # Heuristic: look for score patterns like "6-4" or "3-2"
    for i, text in enumerate(texts):
        score_match = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", text.strip())
        if score_match:
            record["score_raw"] = text.strip()
            record["p1_score"] = int(score_match.group(1))
            record["p2_score"] = int(score_match.group(2))
            if i > 0:
                record["player1_name"] = texts[i - 1]
            if i < len(texts) - 1:
                record["player2_name"] = texts[i + 1]
            if i > 1:
                record["round_name"] = texts[i - 2]
            break

    # Also try to detect player links
    player_links = []
    for cell in cells:
        links = cell.find_all("a", href=lambda h: h and "/player/" in h)
        for link in links:
            href = link.get("href", "")
            pid_match = re.search(r"/player/([^/]+)", href)
            if pid_match:
                player_links.append({
                    "name": link.get_text(strip=True),
                    "id": pid_match.group(1),
                })
    if player_links:
        record["player_links"] = player_links

    return record if record else None


def parse_player_page(
    html: str,
    player_id: str,
) -> dict[str, Any]:
    """
    Parse a DartsDatabase player page.

    Parameters
    ----------
    html:
        Raw HTML content.
    player_id:
        Player identifier.

    Returns
    -------
    dict[str, Any]
        Player career stats and match history.
    """
    soup = BeautifulSoup(html, "lxml")
    data: dict[str, Any] = {
        "player_id": player_id,
        "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_url": PLAYER_URL.format(base=BASE_URL, player_id=player_id),
        "career_stats": {},
        "recent_matches": [],
    }

    # Player name from <h1>
    h1 = soup.find("h1")
    if h1:
        data["player_name"] = h1.get_text(strip=True)

    # Stats from dl/dt/dd pairs or tables
    dl_elements = soup.find_all("dl")
    for dl in dl_elements:
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            key = dt.get_text(strip=True).lower().replace(" ", "_")
            val = dd.get_text(strip=True)
            data["career_stats"][key] = val

    # Stats tables
    tables = soup.find_all("table")
    for table in tables:
        caption = table.find("caption")
        caption_text = caption.get_text(strip=True).lower() if caption else ""

        if "recent" in caption_text or "matches" in caption_text:
            rows = table.find_all("tr")[1:]
            for row in rows:
                cells = row.find_all(["td", "th"])
                match_rec = _parse_match_row(cells)
                if match_rec:
                    data["recent_matches"].append(match_rec)
        else:
            # Generic stat table
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) == 2:
                    key = cells[0].get_text(strip=True).lower().replace(" ", "_")
                    val = cells[1].get_text(strip=True)
                    data["career_stats"][key] = val

    return data


# ---------------------------------------------------------------------------
# Scraper class
# ---------------------------------------------------------------------------

class DartsDatabaseScraper:
    """
    Scrape event and player pages from DartsDatabase.

    Parameters
    ----------
    output_dir:
        Root output directory.
    request_delay:
        Seconds between requests.
    resume:
        Skip already-saved files.
    """

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        request_delay: float = REQUEST_DELAY_SECONDS,
        resume: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.request_delay = request_delay
        self.resume = resume
        self._log = logger.bind(scraper="DartsDatabaseScraper")

    def _event_path(self, event_id: str) -> Path:
        return self.output_dir / "events" / f"{event_id}.json"

    def _player_path(self, player_id: str) -> Path:
        return self.output_dir / "players" / f"{player_id}.json"

    def _save(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    async def scrape_event(
        self,
        session: aiohttp.ClientSession,
        event_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Scrape a single event page.

        Parameters
        ----------
        session:
            Active aiohttp session.
        event_id:
            DartsDatabase event ID.

        Returns
        -------
        dict | None
        """
        path = self._event_path(event_id)
        if self.resume and path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)

        url = EVENT_URL.format(base=BASE_URL, event_id=event_id)
        html = await _fetch(session, url)
        if html is None:
            return None
        data = parse_event_page(html, event_id)
        self._save(path, data)
        return data

    async def scrape_player(
        self,
        session: aiohttp.ClientSession,
        player_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Scrape a single player page.

        Parameters
        ----------
        session:
            Active aiohttp session.
        player_id:
            DartsDatabase player ID.

        Returns
        -------
        dict | None
        """
        path = self._player_path(player_id)
        if self.resume and path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)

        url = PLAYER_URL.format(base=BASE_URL, player_id=player_id)
        html = await _fetch(session, url)
        if html is None:
            return None
        data = parse_player_page(html, player_id)
        self._save(path, data)
        return data

    async def run_events(
        self,
        event_ids: list[str],
    ) -> list[dict[str, Any]]:
        """
        Scrape a list of event IDs sequentially.

        Parameters
        ----------
        event_ids:
            List of event ID strings.

        Returns
        -------
        list[dict]
            Successfully scraped events.
        """
        results: list[dict[str, Any]] = []
        connector = aiohttp.TCPConnector(limit_per_host=1)
        async with aiohttp.ClientSession(connector=connector) as session:
            for i, eid in enumerate(event_ids):
                data = await self.scrape_event(session, eid)
                if data:
                    results.append(data)
                if i < len(event_ids) - 1:
                    await asyncio.sleep(self.request_delay)
        self._log.info(
            "events_scrape_complete",
            total=len(event_ids),
            scraped=len(results),
        )
        return results

    async def run_players(
        self,
        player_ids: list[str],
    ) -> list[dict[str, Any]]:
        """
        Scrape a list of player IDs sequentially.

        Parameters
        ----------
        player_ids:
            List of player ID strings.

        Returns
        -------
        list[dict]
            Successfully scraped players.
        """
        results: list[dict[str, Any]] = []
        connector = aiohttp.TCPConnector(limit_per_host=1)
        async with aiohttp.ClientSession(connector=connector) as session:
            for i, pid in enumerate(player_ids):
                data = await self.scrape_player(session, pid)
                if data:
                    results.append(data)
                if i < len(player_ids) - 1:
                    await asyncio.sleep(self.request_delay)
        self._log.info(
            "players_scrape_complete",
            total=len(player_ids),
            scraped=len(results),
        )
        return results
