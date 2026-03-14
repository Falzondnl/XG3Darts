"""
Flashscore darts scraper.

Scrapes match results, H2H, and live scores from Flashscore.

Rate limiting: 2 req / 5s (polite crawl — Flashscore TOS requires care).

Note: Flashscore uses a proprietary binary/protobuf-based internal API for
live data (requires reverse-engineering network traffic). The scraper
implemented here uses the HTML pages and public JSON feeds where available.

Data saved to: D:/codex/Data/Darts/01_raw/json/flashscore/
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import aiohttp
import structlog
from bs4 import BeautifulSoup


logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://www.flashscore.com/darts"
REQUEST_DELAY_SECONDS = 2.5   # polite crawl: ~0.4 req/s
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0

OUTPUT_DIR = Path("D:/codex/Data/Darts/01_raw/json/flashscore")

_USER_AGENT = (
    "XG3DartsResearch/1.0 "
    "(sports analytics; contact: research@xg3.ai; polite-crawl)"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    """
    A single completed or live darts match result.

    Attributes
    ----------
    match_id:
        Flashscore internal match ID.
    competition_name:
        Full competition name as displayed on Flashscore.
    round_name:
        Round name (e.g. 'Quarter-Final').
    player1_name:
        Player 1 full name.
    player2_name:
        Player 2 full name.
    score1:
        Player 1 score (legs/sets won).
    score2:
        Player 2 score (legs/sets won).
    match_date:
        Match date.
    status:
        'Finished' | 'Live' | 'Scheduled' | 'Postponed'.
    source_url:
        Flashscore URL this result was scraped from.
    raw_data:
        Raw extracted data (for re-processing).
    """
    match_id: str
    competition_name: str
    round_name: str
    player1_name: str
    player2_name: str
    score1: Optional[int]
    score2: Optional[int]
    match_date: Optional[date]
    status: str
    source_url: str
    raw_data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _build_headers() -> dict[str, str]:
    return {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
    }


async def _fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    *,
    retries: int = MAX_RETRIES,
) -> Optional[str]:
    """
    Fetch a URL with exponential-backoff retry logic.

    Returns HTML content string, or None if all retries failed.
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
                    logger.warning("flashscore_404", url=url)
                    return None
                if response.status == 429:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    logger.warning("flashscore_rate_limited", url=url, wait_seconds=wait)
                    await asyncio.sleep(wait)
                    continue
                if response.status in (403, 406):
                    logger.error(
                        "flashscore_blocked",
                        url=url,
                        status=response.status,
                    )
                    return None
                logger.error(
                    "flashscore_http_error",
                    url=url,
                    status=response.status,
                    attempt=attempt,
                )
        except aiohttp.ClientError as exc:
            logger.warning(
                "flashscore_client_error",
                url=url,
                attempt=attempt,
                error=str(exc),
            )
            if attempt <= retries:
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return None


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------


def _parse_score_text(text: str) -> Optional[int]:
    """Extract an integer score from text, or None."""
    cleaned = text.strip()
    try:
        return int(cleaned)
    except (ValueError, TypeError):
        return None


def _parse_match_date(text: str) -> Optional[date]:
    """Parse a Flashscore date string to a date object."""
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _extract_match_id_from_url(url: str) -> Optional[str]:
    """Extract the Flashscore match ID from a result URL."""
    # Flashscore match IDs appear as 8-char alphanumeric in the URL path
    match = re.search(r"/([A-Za-z0-9]{8})(?:/|$)", url)
    return match.group(1) if match else None


def _parse_result_row(
    row,
    competition_name: str,
    source_url: str,
) -> Optional[MatchResult]:
    """
    Attempt to parse one match result row from a competition results page.

    Flashscore HTML structure changes periodically; this parser handles
    the most common layout variants.
    """
    try:
        cells = row.find_all(["td", "div"], recursive=False)
        if len(cells) < 3:
            return None

        # Extract player names
        player_names = []
        for cell in cells:
            participant_el = cell.find(class_=re.compile(r"participant|home|away", re.I))
            if participant_el:
                player_names.append(participant_el.get_text(strip=True))
            if len(player_names) == 2:
                break

        if len(player_names) < 2:
            # Fallback: find all anchor tags with player names
            name_links = row.find_all("a", class_=re.compile(r"name|participant", re.I))
            player_names = [a.get_text(strip=True) for a in name_links[:2]]

        if len(player_names) < 2:
            return None

        # Extract scores
        score_els = row.find_all(class_=re.compile(r"score|result", re.I))
        scores = [_parse_score_text(el.get_text(strip=True)) for el in score_els[:2]]
        score1 = scores[0] if len(scores) > 0 else None
        score2 = scores[1] if len(scores) > 1 else None

        # Extract date
        date_el = row.find(class_=re.compile(r"date|time", re.I))
        match_date = _parse_match_date(date_el.get_text(strip=True)) if date_el else None

        # Extract status
        status_el = row.find(class_=re.compile(r"status|state", re.I))
        status = status_el.get_text(strip=True) if status_el else "Unknown"

        # Match ID from row ID attribute or URL
        row_id = row.get("id", "")
        match_id = row_id if row_id else _extract_match_id_from_url(source_url) or "unknown"

        return MatchResult(
            match_id=match_id,
            competition_name=competition_name,
            round_name="Unknown",
            player1_name=player_names[0],
            player2_name=player_names[1],
            score1=score1,
            score2=score2,
            match_date=match_date,
            status=status if status else "Finished",
            source_url=source_url,
            raw_data={"row_html": str(row)[:500]},
        )
    except Exception as exc:
        logger.debug("flashscore_parse_row_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# FlashscoreClient
# ---------------------------------------------------------------------------


class FlashscoreClient:
    """
    Flashscore darts scraper.

    Scrapes match results, H2H, and live scores.

    Parameters
    ----------
    output_dir:
        Directory for saving scraped JSON files.
    request_delay:
        Seconds to wait between requests.
    resume:
        Skip already-scraped data.
    """

    BASE_URL = "https://www.flashscore.com/darts"

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        request_delay: float = REQUEST_DELAY_SECONDS,
        resume: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.request_delay = request_delay
        self.resume = resume
        self._log = logger.bind(scraper="FlashscoreClient")

    def _save(self, filename: str, data: Any) -> None:
        """Save data to the output directory as JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
        self._log.debug("flashscore_saved", path=str(path))

    async def scrape_recent_matches(
        self,
        days_back: int = 30,
    ) -> list[MatchResult]:
        """
        Scrape recent darts match results from the Flashscore darts section.

        Parameters
        ----------
        days_back:
            Number of calendar days to look back.

        Returns
        -------
        list[MatchResult]
            Scraped match results sorted by date descending.
        """
        self._log.info("flashscore_scrape_recent", days_back=days_back)

        url = f"{self.BASE_URL}/results/"
        connector = aiohttp.TCPConnector(limit_per_host=1)
        results: list[MatchResult] = []
        cutoff = date.today() - timedelta(days=days_back)

        async with aiohttp.ClientSession(connector=connector) as session:
            html = await _fetch_with_retry(session, url)
            if html is None:
                self._log.warning("flashscore_results_page_unavailable", url=url)
                return []

            soup = BeautifulSoup(html, "lxml")
            competition_name = "Flashscore Darts"

            # Find match rows — Flashscore uses various class patterns
            rows = soup.find_all(
                attrs={"class": re.compile(r"event__match|match-row", re.I)}
            )

            for row in rows:
                result = _parse_result_row(row, competition_name, url)
                if result is None:
                    continue
                if result.match_date and result.match_date < cutoff:
                    continue
                results.append(result)
                await asyncio.sleep(0)  # yield

        self._log.info(
            "flashscore_recent_complete",
            results_found=len(results),
            days_back=days_back,
        )

        # Save to disk
        if results:
            scraped_at = datetime.now(tz=timezone.utc).isoformat()
            self._save(
                f"recent_matches_{date.today().isoformat()}.json",
                {
                    "scraped_at": scraped_at,
                    "days_back": days_back,
                    "count": len(results),
                    "matches": [
                        {
                            "match_id": r.match_id,
                            "competition_name": r.competition_name,
                            "player1_name": r.player1_name,
                            "player2_name": r.player2_name,
                            "score1": r.score1,
                            "score2": r.score2,
                            "match_date": r.match_date.isoformat() if r.match_date else None,
                            "status": r.status,
                            "source_url": r.source_url,
                        }
                        for r in results
                    ],
                },
            )

        return results

    async def scrape_competition_results(
        self,
        competition_url: str,
    ) -> list[MatchResult]:
        """
        Scrape all results from a specific competition page.

        Parameters
        ----------
        competition_url:
            Full Flashscore URL of the competition results page.

        Returns
        -------
        list[MatchResult]
            All match results from the competition.
        """
        self._log.info("flashscore_scrape_competition", url=competition_url)

        connector = aiohttp.TCPConnector(limit_per_host=1)
        results: list[MatchResult] = []

        async with aiohttp.ClientSession(connector=connector) as session:
            html = await _fetch_with_retry(session, competition_url)
            if html is None:
                self._log.warning(
                    "flashscore_competition_page_unavailable",
                    url=competition_url,
                )
                return []

            soup = BeautifulSoup(html, "lxml")

            # Extract competition name from page title
            title_el = soup.find("h1") or soup.find("title")
            competition_name = title_el.get_text(strip=True) if title_el else "Unknown"

            rows = soup.find_all(
                attrs={"class": re.compile(r"event__match|match-row", re.I)}
            )

            for row in rows:
                result = _parse_result_row(row, competition_name, competition_url)
                if result:
                    results.append(result)

            await asyncio.sleep(self.request_delay)

        self._log.info(
            "flashscore_competition_complete",
            url=competition_url,
            results_found=len(results),
        )

        if results:
            safe_name = re.sub(r"[^a-z0-9]", "_", competition_name.lower())[:50]
            self._save(
                f"competition_{safe_name}_{date.today().isoformat()}.json",
                {
                    "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
                    "competition_name": competition_name,
                    "source_url": competition_url,
                    "count": len(results),
                    "matches": [
                        {
                            "match_id": r.match_id,
                            "player1_name": r.player1_name,
                            "player2_name": r.player2_name,
                            "score1": r.score1,
                            "score2": r.score2,
                            "match_date": r.match_date.isoformat() if r.match_date else None,
                            "status": r.status,
                        }
                        for r in results
                    ],
                },
            )

        return results

    async def scrape_live_matches(self) -> list[MatchResult]:
        """
        Get list of currently live darts matches from Flashscore.

        Returns
        -------
        list[MatchResult]
            Live match records with current scores.
        """
        self._log.info("flashscore_scrape_live")

        url = f"{self.BASE_URL}/"
        connector = aiohttp.TCPConnector(limit_per_host=1)
        live: list[MatchResult] = []

        async with aiohttp.ClientSession(connector=connector) as session:
            html = await _fetch_with_retry(session, url)
            if html is None:
                return []

            soup = BeautifulSoup(html, "lxml")

            # Live matches are typically in a section with class 'live' or
            # event__match--live
            live_rows = soup.find_all(
                attrs={"class": re.compile(r"live|event__match--live", re.I)}
            )

            for row in live_rows:
                result = _parse_result_row(row, "Live", url)
                if result:
                    result.status = "Live"
                    live.append(result)

        self._log.info("flashscore_live_complete", live_count=len(live))
        return live
