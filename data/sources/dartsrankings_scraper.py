"""
DartsRankings.com scraper — historical order of merit snapshots.

Scrapes annual (and where available, monthly) PDC Order of Merit ranking
snapshots from DartsRankings.com.

Rate limiting: 1 req / 3s (polite crawl).

Data saved to: D:/codex/Data/Darts/01_raw/json/dartsrankings/
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

BASE_URL = "https://www.dartsrankings.com"
REQUEST_DELAY_SECONDS = 3.0
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0

OUTPUT_DIR = Path("D:/codex/Data/Darts/01_raw/json/dartsrankings")

_USER_AGENT = (
    "XG3DartsResearch/1.0 "
    "(sports analytics; contact: research@xg3.ai; polite-crawl 1req/3s)"
)

# Earliest year of useful PDC ranking data
_MIN_YEAR = 2003


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RankingEntry:
    """
    A single player's entry in a ranking snapshot.

    Attributes
    ----------
    rank:
        Ranking position (1 = top).
    player_name:
        Full player name as listed.
    prize_money:
        Prize money figure (integer, in GBP or applicable currency).
    nationality:
        Nationality code or country name.
    raw_data:
        Raw row data from the page.
    """
    rank: int
    player_name: str
    prize_money: Optional[int]
    nationality: Optional[str]
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankingSnapshot:
    """
    A complete ranking snapshot at a point in time.

    Attributes
    ----------
    year:
        The year of this snapshot.
    month:
        The month (1-12), or None for an annual snapshot.
    snapshot_date:
        The exact date of the snapshot (approximated if month-only).
    ranking_type:
        e.g. 'pdc_order_of_merit' | 'pdc_tour_card' | 'wdf'
    entries:
        List of ranked player entries.
    source_url:
        The URL this snapshot was scraped from.
    scraped_at:
        ISO timestamp when this snapshot was scraped.
    """
    year: int
    month: Optional[int]
    snapshot_date: date
    ranking_type: str
    entries: list[RankingEntry]
    source_url: str
    scraped_at: str


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
    """Fetch URL with retry logic. Returns HTML or None."""
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
                    logger.warning("dartsrankings_404", url=url)
                    return None
                if response.status == 429:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    logger.warning(
                        "dartsrankings_rate_limited", url=url, wait_seconds=wait
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error(
                    "dartsrankings_http_error",
                    url=url,
                    status=response.status,
                    attempt=attempt,
                )
        except aiohttp.ClientError as exc:
            logger.warning(
                "dartsrankings_client_error",
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


def _parse_prize_money(text: str) -> Optional[int]:
    """Parse a prize money text (e.g. '£1,234,567') to an integer."""
    cleaned = re.sub(r"[£$€,\s]", "", text.strip())
    try:
        return int(cleaned)
    except (ValueError, TypeError):
        return None


def _parse_ranking_table(
    html: str,
    source_url: str,
    year: int,
    month: Optional[int],
) -> list[RankingEntry]:
    """
    Parse a DartsRankings ranking table page into RankingEntry records.

    The site uses a standard HTML table with columns:
    Rank | Player | Prize Money | Nationality (order may vary).
    """
    soup = BeautifulSoup(html, "lxml")
    entries: list[RankingEntry] = []

    # Find the main ranking table
    tables = soup.find_all("table")
    if not tables:
        logger.warning("dartsrankings_no_table_found", url=source_url)
        return entries

    # Use the table with the most rows (likely the rankings table)
    main_table = max(tables, key=lambda t: len(t.find_all("tr")))
    rows = main_table.find_all("tr")

    # Detect column order from header row
    header_row = rows[0] if rows else None
    col_order: dict[str, int] = {}
    if header_row:
        headers = [th.get_text(strip=True).lower() for th in header_row.find_all(["th", "td"])]
        for i, h in enumerate(headers):
            if "rank" in h or "pos" in h or "#" in h:
                col_order["rank"] = i
            elif "player" in h or "name" in h or "darter" in h:
                col_order["player"] = i
            elif "prize" in h or "money" in h or "earning" in h:
                col_order["prize"] = i
            elif "country" in h or "national" in h or "flag" in h:
                col_order["nationality"] = i

    # Default column order if not detected
    rank_col = col_order.get("rank", 0)
    player_col = col_order.get("player", 1)
    prize_col = col_order.get("prize", 2)
    nationality_col = col_order.get("nationality", 3)

    for row in rows[1:]:  # Skip header
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue

        try:
            rank_text = cells[rank_col].get_text(strip=True) if rank_col < len(cells) else ""
            rank = int(re.sub(r"[^0-9]", "", rank_text) or "0")
            if rank <= 0:
                continue

            player_name = (
                cells[player_col].get_text(strip=True)
                if player_col < len(cells) else ""
            )
            if not player_name:
                continue

            prize = None
            if prize_col < len(cells):
                prize = _parse_prize_money(cells[prize_col].get_text(strip=True))

            nationality = None
            if nationality_col < len(cells):
                nationality = cells[nationality_col].get_text(strip=True) or None

            entries.append(RankingEntry(
                rank=rank,
                player_name=player_name,
                prize_money=prize,
                nationality=nationality,
                raw_data={"row_text": " | ".join(c.get_text(strip=True) for c in cells)},
            ))
        except (ValueError, IndexError):
            continue

    logger.debug(
        "dartsrankings_parsed_table",
        url=source_url,
        year=year,
        month=month,
        entries=len(entries),
    )
    return entries


# ---------------------------------------------------------------------------
# DartsRankingsScraper
# ---------------------------------------------------------------------------


class DartsRankingsScraper:
    """
    DartsRankings.com scraper — historical order of merit snapshots.

    Parameters
    ----------
    output_dir:
        Directory for saving scraped JSON files.
    request_delay:
        Seconds between requests (rate limiting).
    resume:
        Skip years already saved to disk.
    """

    BASE_URL = "https://www.dartsrankings.com"

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        request_delay: float = REQUEST_DELAY_SECONDS,
        resume: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.request_delay = request_delay
        self.resume = resume
        self._log = logger.bind(scraper="DartsRankingsScraper")

    def _output_path(self, year: int, month: Optional[int] = None) -> Path:
        if month:
            return self.output_dir / f"rankings_{year}_{month:02d}.json"
        return self.output_dir / f"rankings_{year}.json"

    def _is_already_scraped(self, year: int, month: Optional[int] = None) -> bool:
        return self.resume and self._output_path(year, month).exists()

    def _save_snapshot(self, snapshot: RankingSnapshot) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_path(snapshot.year, snapshot.month)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "year": snapshot.year,
                    "month": snapshot.month,
                    "snapshot_date": snapshot.snapshot_date.isoformat(),
                    "ranking_type": snapshot.ranking_type,
                    "source_url": snapshot.source_url,
                    "scraped_at": snapshot.scraped_at,
                    "entry_count": len(snapshot.entries),
                    "entries": [
                        {
                            "rank": e.rank,
                            "player_name": e.player_name,
                            "prize_money": e.prize_money,
                            "nationality": e.nationality,
                        }
                        for e in snapshot.entries
                    ],
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )

    def _build_ranking_url(self, year: int, month: Optional[int] = None) -> str:
        """
        Build the DartsRankings URL for a given year (and optional month).

        URL patterns used by DartsRankings.com:
        - Annual: /rankings/{year}/
        - Monthly: /rankings/{year}/{month:02d}/
        """
        if month:
            return f"{self.BASE_URL}/rankings/{year}/{month:02d}/"
        return f"{self.BASE_URL}/rankings/{year}/"

    async def scrape_year(
        self,
        session: aiohttp.ClientSession,
        year: int,
    ) -> Optional[RankingSnapshot]:
        """
        Scrape the annual ranking snapshot for a given year.

        Parameters
        ----------
        session:
            Active aiohttp session.
        year:
            The year to scrape.

        Returns
        -------
        RankingSnapshot | None
            Snapshot, or None if the page was unavailable.
        """
        if self._is_already_scraped(year):
            self._log.debug("dartsrankings_skip_year", year=year)
            path = self._output_path(year)
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Reconstruct snapshot from saved data
            entries = [
                RankingEntry(
                    rank=e["rank"],
                    player_name=e["player_name"],
                    prize_money=e.get("prize_money"),
                    nationality=e.get("nationality"),
                )
                for e in data.get("entries", [])
            ]
            return RankingSnapshot(
                year=data["year"],
                month=data.get("month"),
                snapshot_date=date.fromisoformat(data["snapshot_date"]),
                ranking_type=data.get("ranking_type", "pdc_order_of_merit"),
                entries=entries,
                source_url=data["source_url"],
                scraped_at=data["scraped_at"],
            )

        url = self._build_ranking_url(year)
        self._log.info("dartsrankings_scrape_year", year=year, url=url)

        html = await _fetch_with_retry(session, url)
        if html is None:
            self._log.warning("dartsrankings_year_unavailable", year=year, url=url)
            return None

        entries = _parse_ranking_table(html, url, year, None)
        if not entries:
            self._log.warning("dartsrankings_no_entries", year=year, url=url)
            return None

        snapshot = RankingSnapshot(
            year=year,
            month=None,
            snapshot_date=date(year, 12, 31),  # Use year-end date for annual snapshot
            ranking_type="pdc_order_of_merit",
            entries=entries,
            source_url=url,
            scraped_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        self._save_snapshot(snapshot)
        return snapshot

    async def scrape_historical_rankings(
        self,
        start_year: int = 2010,
        end_year: int = 2026,
    ) -> list[RankingSnapshot]:
        """
        Scrape annual ranking snapshots for a range of years.

        Parameters
        ----------
        start_year:
            First year to scrape (minimum 2003).
        end_year:
            Last year to scrape (inclusive).

        Returns
        -------
        list[RankingSnapshot]
            Successfully scraped ranking snapshots, one per year.
        """
        start_year = max(start_year, _MIN_YEAR)
        end_year = min(end_year, date.today().year)

        self._log.info(
            "dartsrankings_historical_start",
            start_year=start_year,
            end_year=end_year,
        )

        snapshots: list[RankingSnapshot] = []
        connector = aiohttp.TCPConnector(limit_per_host=1)

        async with aiohttp.ClientSession(connector=connector) as session:
            years = list(range(start_year, end_year + 1))
            for i, year in enumerate(years):
                snapshot = await self.scrape_year(session, year)
                if snapshot:
                    snapshots.append(snapshot)
                else:
                    self._log.warning("dartsrankings_year_failed", year=year)

                # Rate limit between requests
                if i < len(years) - 1 and not self._is_already_scraped(year):
                    await asyncio.sleep(self.request_delay)

        self._log.info(
            "dartsrankings_historical_complete",
            total_years=end_year - start_year + 1,
            scraped=len(snapshots),
        )
        return snapshots

    async def scrape_current_rankings(self) -> Optional[RankingSnapshot]:
        """
        Scrape the most recent ranking snapshot.

        Returns
        -------
        RankingSnapshot | None
        """
        current_year = date.today().year
        url = f"{self.BASE_URL}/rankings/"
        self._log.info("dartsrankings_current", url=url)

        connector = aiohttp.TCPConnector(limit_per_host=1)
        async with aiohttp.ClientSession(connector=connector) as session:
            html = await _fetch_with_retry(session, url)
            if html is None:
                return None

        entries = _parse_ranking_table(html, url, current_year, None)
        if not entries:
            return None

        snapshot = RankingSnapshot(
            year=current_year,
            month=date.today().month,
            snapshot_date=date.today(),
            ranking_type="pdc_order_of_merit",
            entries=entries,
            source_url=url,
            scraped_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        self._save_snapshot(snapshot)
        return snapshot
