"""
DartsOrakel scraper.

Fetches per-player detailed statistics from:
    https://dartsorakel.com/player/details/{player_key}/

Data captured per player:
- 3-dart average (3DA)
- First-9 average
- Checkout percentage
- Hold rate / Break rate
- Form (last N matches)
- Head-to-head win rates

Rate limiting: 1 request per 2 seconds (polite crawl).
Output: Saves to D:/codex/Data/Darts/02_processed/json/dartsorakel/ with
        one file per player: ``{player_key}.json``

The module also reads the existing ``stats_player.json`` seed file to
determine which player keys to fetch.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
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

BASE_URL = "https://dartsorakel.com"
PLAYER_DETAIL_URL = "{base}/player/details/{player_key}/"
REQUEST_DELAY_SECONDS = 2.0   # 1 req / 2s
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0

SEED_FILE = Path("D:/codex/Data/Darts/01_raw/json/dartsorakel/stats_player.json")
OUTPUT_DIR = Path("D:/codex/Data/Darts/02_processed/json/dartsorakel/player_details")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _build_headers() -> dict[str, str]:
    """Return request headers that identify us as a legitimate crawler."""
    return {
        "User-Agent": (
            "XG3DartsResearch/1.0 "
            "(sports analytics; contact: research@xg3.ai; "
            "polite-crawl 1req/2s)"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-GB,en;q=0.9",
    }


async def _fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    *,
    retries: int = MAX_RETRIES,
) -> Optional[str]:
    """
    Fetch a URL with exponential-backoff retry logic.

    Parameters
    ----------
    session:
        Active aiohttp session.
    url:
        Target URL.
    retries:
        Number of retry attempts.

    Returns
    -------
    str | None
        HTML content, or None if all retries failed.
    """
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
                    logger.warning("dartsorakel_404", url=url)
                    return None
                if response.status == 429:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    logger.warning(
                        "dartsorakel_rate_limited",
                        url=url,
                        wait_seconds=wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error(
                    "dartsorakel_http_error",
                    url=url,
                    status=response.status,
                    attempt=attempt,
                )
        except aiohttp.ClientError as exc:
            logger.warning(
                "dartsorakel_client_error",
                url=url,
                attempt=attempt,
                error=str(exc),
            )
            if attempt <= retries:
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
    return None


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def parse_player_detail_page(
    html: str,
    player_key: int,
    player_name: str,
) -> dict[str, Any]:
    """
    Parse a DartsOrakel player detail page into a structured dict.

    Parameters
    ----------
    html:
        Raw HTML content of the page.
    player_key:
        DartsOrakel player key.
    player_name:
        Player name (from seed data).

    Returns
    -------
    dict[str, Any]
        Parsed player statistics.
    """
    soup = BeautifulSoup(html, "lxml")
    data: dict[str, Any] = {
        "player_key": player_key,
        "player_name": player_name,
        "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_url": PLAYER_DETAIL_URL.format(base=BASE_URL, player_key=player_key),
    }

    # Extract stat cards / metric panels
    # DartsOrakel uses Bootstrap card components with stat values
    stat_cards = soup.find_all("div", class_=lambda c: c and "card" in c.split())
    for card in stat_cards:
        title_el = card.find(["h5", "h6", "strong", "b"])
        value_el = card.find(class_=lambda c: c and any(
            kw in c for kw in ("stat", "value", "number", "big")
        ))
        if title_el and value_el:
            title = title_el.get_text(strip=True).lower()
            value_text = value_el.get_text(strip=True)
            # Map known stat titles to canonical keys
            if "3-dart" in title or "average" in title or "3da" in title:
                data["three_dart_average"] = _parse_float(value_text)
            elif "first" in title and "9" in title:
                data["first_nine_average"] = _parse_float(value_text)
            elif "checkout" in title:
                data["checkout_percentage"] = _parse_float(value_text)
            elif "hold" in title:
                data["hold_rate"] = _parse_float(value_text)
            elif "break" in title:
                data["break_rate"] = _parse_float(value_text)

    # Extract stats from table rows (alternative layout)
    stat_rows = soup.find_all("tr")
    for row in stat_rows:
        cells = row.find_all(["td", "th"])
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True).lower()
            val_text = cells[1].get_text(strip=True)
            if "3-dart" in label or "average" in label:
                data.setdefault("three_dart_average", _parse_float(val_text))
            elif "first 9" in label or "first nine" in label:
                data.setdefault("first_nine_average", _parse_float(val_text))
            elif "checkout" in label:
                data.setdefault("checkout_percentage", _parse_float(val_text))
            elif "180" in label:
                data.setdefault("count_180", _parse_int(val_text))

    # Extract form (recent match results)
    form_section = soup.find(id=lambda i: i and "form" in i.lower())
    if form_section:
        result_badges = form_section.find_all(
            class_=lambda c: c and any(kw in c for kw in ("win", "loss", "draw", "W", "L"))
        )
        form_string = "".join(
            "W" if "win" in (b.get("class", []) or []) else
            "L" if "loss" in (b.get("class", []) or []) else "D"
            for b in result_badges[:10]
        )
        if form_string:
            data["form_last_10"] = form_string

    return data


def _parse_float(text: str) -> Optional[float]:
    """Extract a float from a text string, or return None."""
    cleaned = text.replace("%", "").replace(",", ".").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_int(text: str) -> Optional[int]:
    """Extract an int from a text string, or return None."""
    cleaned = re.sub(r"[^\d]", "", text)
    try:
        return int(cleaned)
    except ValueError:
        return None


def _load_seed_players() -> list[dict[str, Any]]:
    """
    Load the DartsOrakel player list from the seed JSON file.

    Returns
    -------
    list[dict]
        List of player dicts with ``player_key`` and ``player_name``.

    Raises
    ------
    FileNotFoundError
        If the seed file does not exist.
    """
    if not SEED_FILE.exists():
        raise FileNotFoundError(f"Seed file not found: {SEED_FILE}")

    with SEED_FILE.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    players = raw.get("data", [])
    logger.info("dartsorakel_seed_loaded", count=len(players))
    return players


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

class DartsOrakelScraper:
    """
    Scrape per-player detail pages from DartsOrakel.

    Parameters
    ----------
    output_dir:
        Directory for saving scraped JSON files.
    request_delay:
        Seconds to wait between requests.
    max_players:
        Maximum number of players to scrape (None = all).
    resume:
        If True, skip player keys that already have an output file.
    """

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        request_delay: float = REQUEST_DELAY_SECONDS,
        max_players: Optional[int] = None,
        resume: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.request_delay = request_delay
        self.max_players = max_players
        self.resume = resume
        self._log = logger.bind(scraper="DartsOrakelScraper")

    def _output_path(self, player_key: int) -> Path:
        return self.output_dir / f"{player_key}.json"

    def _is_already_scraped(self, player_key: int) -> bool:
        return self.resume and self._output_path(player_key).exists()

    def _save(self, player_key: int, data: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_path(player_key)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    async def scrape_player(
        self,
        session: aiohttp.ClientSession,
        player_key: int,
        player_name: str,
    ) -> Optional[dict[str, Any]]:
        """
        Scrape a single player's detail page.

        Parameters
        ----------
        session:
            Active aiohttp session.
        player_key:
            DartsOrakel player key.
        player_name:
            Player name from seed data.

        Returns
        -------
        dict | None
            Parsed player data, or None if the page was unavailable.
        """
        if self._is_already_scraped(player_key):
            self._log.debug(
                "skip_already_scraped",
                player_key=player_key,
                player_name=player_name,
            )
            with self._output_path(player_key).open("r", encoding="utf-8") as fh:
                return json.load(fh)

        url = PLAYER_DETAIL_URL.format(base=BASE_URL, player_key=player_key)
        self._log.info(
            "scraping_player",
            player_key=player_key,
            player_name=player_name,
            url=url,
        )

        html = await _fetch_with_retry(session, url)
        if html is None:
            self._log.warning(
                "scrape_failed",
                player_key=player_key,
                player_name=player_name,
            )
            return None

        data = parse_player_detail_page(html, player_key, player_name)
        self._save(player_key, data)
        return data

    async def run(self) -> list[dict[str, Any]]:
        """
        Run the full scrape pipeline for all players from the seed file.

        Returns
        -------
        list[dict]
            All successfully scraped player records.
        """
        players = _load_seed_players()
        if self.max_players:
            players = players[: self.max_players]

        results: list[dict[str, Any]] = []
        failed_count = 0

        connector = aiohttp.TCPConnector(limit_per_host=1)
        async with aiohttp.ClientSession(connector=connector) as session:
            for i, player in enumerate(players):
                player_key = player["player_key"]
                player_name = player.get("player_name", f"Player_{player_key}")

                data = await self.scrape_player(session, player_key, player_name)
                if data:
                    results.append(data)
                else:
                    failed_count += 1

                # Rate-limit: wait between requests
                if i < len(players) - 1:
                    await asyncio.sleep(self.request_delay)

        self._log.info(
            "scrape_run_complete",
            total=len(players),
            scraped=len(results),
            failed=failed_count,
        )
        return results


# ---------------------------------------------------------------------------
# Missing import
# ---------------------------------------------------------------------------

import re  # noqa: E402  (needed by _parse_int, placed after BeautifulSoup import)


async def scrape_all_players(
    max_players: Optional[int] = None,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """
    Entry point for running the DartsOrakel scraper.

    Parameters
    ----------
    max_players:
        Limit for testing / partial runs.
    resume:
        Skip already-scraped players.

    Returns
    -------
    list[dict]
        Successfully scraped player records.
    """
    scraper = DartsOrakelScraper(
        max_players=max_players,
        resume=resume,
    )
    return await scraper.run()
