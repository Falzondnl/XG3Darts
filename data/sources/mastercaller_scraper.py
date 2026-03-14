"""
Mastercaller scraper.

Processes the 836 discovered links from:
    D:/codex/Data/Darts/02_processed/csv/mastercaller/discovered_links.csv

Mastercaller (https://mastercaller.com) provides match data, 180 counts,
checkout sequences, and scored visit data for major PDC events.

Data extracted:
- Match results with player names and scores
- Per-leg 180 counts
- Checkout details (score, darts used)
- Visit-level scores where available

Rate limiting: 1 request per 2 seconds.
Output: D:/codex/Data/Darts/02_processed/json/mastercaller/
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import aiohttp
import structlog
from bs4 import BeautifulSoup


logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LINKS_CSV = Path(
    "D:/codex/Data/Darts/02_processed/csv/mastercaller/discovered_links.csv"
)
OUTPUT_DIR = Path("D:/codex/Data/Darts/02_processed/json/mastercaller")

REQUEST_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _build_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "XG3DartsResearch/1.0 "
            "(sports analytics; contact: research@xg3.ai; "
            "polite-crawl 1req/2s)"
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
    """Fetch a URL with retry/backoff."""
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
                    logger.warning("mastercaller_not_found", url=url, status=resp.status)
                    return None
                if resp.status == 429 or resp.status >= 500:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    await asyncio.sleep(wait)
        except aiohttp.ClientError as exc:
            if attempt <= retries:
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
            else:
                logger.error("mastercaller_error", url=url, error=str(exc))
    return None


# ---------------------------------------------------------------------------
# URL classification
# ---------------------------------------------------------------------------

def classify_url(url: str) -> str:
    """
    Classify a Mastercaller URL by its content type.

    Returns one of: ``"match"``, ``"event"``, ``"player"``, ``"other"``

    Parameters
    ----------
    url:
        Full URL string.
    """
    path = urlparse(url).path.lower()
    if re.search(r"/match(es)?/", path) or re.search(r"/result(s)?/", path):
        return "match"
    if re.search(r"/event(s)?/", path) or re.search(r"/tournament(s)?/", path):
        return "event"
    if re.search(r"/player(s)?/", path) or re.search(r"/profile/", path):
        return "player"
    return "other"


def url_to_filename(url: str) -> str:
    """Convert a URL to a stable filename using its MD5 hash."""
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()
    path_part = urlparse(url).path.strip("/").replace("/", "_")
    # Truncate path_part to 80 chars
    path_part = path_part[:80]
    return f"{path_part}_{digest[:8]}.json"


# ---------------------------------------------------------------------------
# HTML parsers
# ---------------------------------------------------------------------------

def parse_match_page(html: str, url: str) -> dict[str, Any]:
    """
    Parse a Mastercaller match page.

    Extracts:
    - Player names and scores
    - Per-leg details: 180 counts, checkout
    - Visit scores where structured data is present

    Parameters
    ----------
    html:
        Page HTML.
    url:
        Source URL.

    Returns
    -------
    dict[str, Any]
    """
    soup = BeautifulSoup(html, "lxml")
    data: dict[str, Any] = {
        "url": url,
        "page_type": "match",
        "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
        "players": [],
        "legs": [],
        "visits": [],
    }

    # Title / match headline
    title = soup.find("title")
    if title:
        data["page_title"] = title.get_text(strip=True)

    h1 = soup.find("h1")
    if h1:
        data["match_headline"] = h1.get_text(strip=True)

    # Score extraction
    score_patterns = soup.find_all(
        string=lambda t: t and re.match(r"^\d+\s*[-–]\s*\d+$", t.strip())
    )
    for sp in score_patterns[:1]:  # take first score match
        score_match = re.match(r"(\d+)\s*[-–]\s*(\d+)", sp.strip())
        if score_match:
            data["p1_score"] = int(score_match.group(1))
            data["p2_score"] = int(score_match.group(2))

    # 180 counts
    text_content = soup.get_text()
    one80_patterns = re.findall(r"180[s]?\s*[:\-=]\s*(\d+)", text_content, re.IGNORECASE)
    if one80_patterns:
        data["count_180_mentions"] = [int(x) for x in one80_patterns]

    # Checkout information
    checkout_patterns = re.findall(
        r"checkout[s]?\s*[:\-=]?\s*(\d+)%?",
        text_content,
        re.IGNORECASE,
    )
    if checkout_patterns:
        data["checkout_mentions"] = checkout_patterns

    # Player names from links or strong elements
    player_els = soup.find_all(class_=lambda c: c and "player" in c.lower())
    for el in player_els[:2]:
        name = el.get_text(strip=True)
        if name and name not in data["players"]:
            data["players"].append(name)

    # Visit table if present
    tables = soup.find_all("table")
    for table in tables:
        thead = table.find("thead")
        if not thead:
            continue
        headers = [th.get_text(strip=True).lower() for th in thead.find_all("th")]
        if "score" in headers or "visit" in headers or "throw" in headers:
            rows = table.find_all("tr")[1:]
            for row in rows:
                cells = row.find_all(["td", "th"])
                if cells:
                    visit = {
                        headers[i]: cells[i].get_text(strip=True)
                        for i in range(min(len(headers), len(cells)))
                    }
                    data["visits"].append(visit)

    return data


def parse_event_page(html: str, url: str) -> dict[str, Any]:
    """
    Parse a Mastercaller event/tournament page.

    Parameters
    ----------
    html:
        Page HTML.
    url:
        Source URL.

    Returns
    -------
    dict[str, Any]
    """
    soup = BeautifulSoup(html, "lxml")
    data: dict[str, Any] = {
        "url": url,
        "page_type": "event",
        "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
        "matches": [],
    }

    h1 = soup.find("h1")
    if h1:
        data["event_name"] = h1.get_text(strip=True)

    # Extract match links from the page for crawl depth-2
    match_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/match" in href.lower() or "/result" in href.lower():
            full_url = href if href.startswith("http") else f"https://mastercaller.com{href}"
            match_links.append(full_url)
    data["match_links"] = list(set(match_links))

    # Result rows
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")[1:]
        for row in rows:
            cells = row.find_all(["td", "th"])
            texts = [c.get_text(strip=True) for c in cells]
            if texts:
                data["matches"].append({"raw_cells": texts})

    return data


def parse_generic_page(html: str, url: str) -> dict[str, Any]:
    """
    Parse an unclassified Mastercaller page — extract all text for indexing.

    Parameters
    ----------
    html:
        Page HTML.
    url:
        Source URL.

    Returns
    -------
    dict[str, Any]
    """
    soup = BeautifulSoup(html, "lxml")
    return {
        "url": url,
        "page_type": "other",
        "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
        "title": soup.title.get_text(strip=True) if soup.title else None,
        "text_preview": soup.get_text(strip=True)[:500],
    }


# ---------------------------------------------------------------------------
# Scraper class
# ---------------------------------------------------------------------------

class MastercallerScraper:
    """
    Scrape Mastercaller pages from the discovered links CSV.

    Parameters
    ----------
    links_csv:
        Path to the discovered_links.csv file.
    output_dir:
        Root output directory.
    request_delay:
        Seconds between requests.
    resume:
        Skip already-saved pages.
    max_pages:
        Maximum pages to scrape (None = all 836).
    """

    def __init__(
        self,
        links_csv: Path = LINKS_CSV,
        output_dir: Path = OUTPUT_DIR,
        request_delay: float = REQUEST_DELAY_SECONDS,
        resume: bool = True,
        max_pages: Optional[int] = None,
    ) -> None:
        self.links_csv = links_csv
        self.output_dir = output_dir
        self.request_delay = request_delay
        self.resume = resume
        self.max_pages = max_pages
        self._log = logger.bind(scraper="MastercallerScraper")

    def _load_links(self) -> list[str]:
        """
        Load URLs from the discovered links CSV.

        Returns
        -------
        list[str]
            List of URL strings.

        Raises
        ------
        FileNotFoundError
            If the links CSV does not exist.
        """
        if not self.links_csv.exists():
            raise FileNotFoundError(f"Links CSV not found: {self.links_csv}")

        urls: list[str] = []
        with self.links_csv.open("r", encoding="utf-8") as fh:
            # Skip header
            lines = fh.readlines()
            header = lines[0].strip().lower() if lines else ""
            start = 1 if "url" in header else 0
            for line in lines[start:]:
                url = line.strip().strip('"')
                if url and url.startswith("http"):
                    urls.append(url)

        self._log.info("links_loaded", count=len(urls))
        return urls

    def _output_path(self, url: str) -> Path:
        page_type = classify_url(url)
        filename = url_to_filename(url)
        return self.output_dir / page_type / filename

    def _save(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    def _parse(self, html: str, url: str) -> dict[str, Any]:
        """Route parsing based on classified URL type."""
        page_type = classify_url(url)
        if page_type == "match":
            return parse_match_page(html, url)
        if page_type == "event":
            return parse_event_page(html, url)
        return parse_generic_page(html, url)

    async def scrape_url(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> Optional[dict[str, Any]]:
        """
        Scrape and parse a single URL.

        Parameters
        ----------
        session:
            Active aiohttp session.
        url:
            Target URL.

        Returns
        -------
        dict | None
        """
        path = self._output_path(url)
        if self.resume and path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)

        html = await _fetch(session, url)
        if html is None:
            return None

        data = self._parse(html, url)
        self._save(path, data)
        return data

    async def run(self) -> dict[str, list[dict[str, Any]]]:
        """
        Scrape all 836 discovered Mastercaller links.

        Returns
        -------
        dict[str, list[dict]]
            Results grouped by page type: ``{"match": [...], "event": [...], ...}``
        """
        urls = self._load_links()
        if self.max_pages:
            urls = urls[: self.max_pages]

        results: dict[str, list[dict[str, Any]]] = {
            "match": [],
            "event": [],
            "player": [],
            "other": [],
        }
        failed = 0

        connector = aiohttp.TCPConnector(limit_per_host=1)
        async with aiohttp.ClientSession(connector=connector) as session:
            for i, url in enumerate(urls):
                data = await self.scrape_url(session, url)
                if data:
                    page_type = data.get("page_type", "other")
                    results.setdefault(page_type, []).append(data)
                else:
                    failed += 1

                if i < len(urls) - 1:
                    await asyncio.sleep(self.request_delay)

        self._log.info(
            "mastercaller_run_complete",
            total=len(urls),
            match=len(results.get("match", [])),
            event=len(results.get("event", [])),
            other=len(results.get("other", [])),
            failed=failed,
        )
        return results


async def scrape_all(
    max_pages: Optional[int] = None,
    resume: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """
    Entry point for running the Mastercaller scraper.

    Parameters
    ----------
    max_pages:
        Limit for testing / partial runs.
    resume:
        Skip already-scraped pages.

    Returns
    -------
    dict[str, list[dict]]
        Scraped data grouped by page type.
    """
    scraper = MastercallerScraper(max_pages=max_pages, resume=resume)
    return await scraper.run()
