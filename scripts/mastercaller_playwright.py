"""
Mastercaller Playwright scraper — JS-rendered match data.

Mastercaller (https://mastercaller.com) is a JavaScript SPA that renders
match scorecards, leg-by-leg breakdowns, 180 counts, and checkout routes
client-side. Standard HTTP requests only get the shell HTML with no data.

This script uses Playwright (headless Chromium) to:
  1. Load each match/event URL
  2. Wait for the JS app to render
  3. Extract the rendered DOM for:
     - Match result (player names, set/leg scores)
     - Per-leg breakdown (180s per player, checkout score, darts used)
     - Visit-level data where available (per-visit scores in leg tables)
  4. Save to D:/codex/Data/Darts/02_processed/json/mastercaller_playwright/

Requirements:
    pip install playwright
    playwright install chromium

Usage:
    # Scrape all event tournament result pages (27 collected URLs)
    python scripts/mastercaller_playwright.py --input-csv data/sources/mastercaller_tournament_urls.csv

    # Scrape a specific URL for testing
    python scripts/mastercaller_playwright.py --url https://mastercaller.com/tournaments/...

    # Resume from checkpoint (skips already-scraped pages)
    python scripts/mastercaller_playwright.py --resume

Rate limiting: 2-second polite delay between requests.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("D:/codex/Data/Darts/02_processed/json/mastercaller_playwright")
DEFAULT_CSV = Path(
    "D:/codex/Data/Darts/02_processed/csv/mastercaller/discovered_links.csv"
)

REQUEST_DELAY_S = 2.0
PAGE_LOAD_TIMEOUT_MS = 30_000    # 30s
CONTENT_WAIT_TIMEOUT_MS = 10_000  # 10s after navigation

# CSS selectors / text patterns that indicate match content has loaded
_CONTENT_LOADED_SELECTORS = [
    "table",
    ".match-stats",
    ".leg-breakdown",
    ".score-table",
    ".tournament-draw",
    "[class*='player']",
    "[class*='match']",
    "[class*='score']",
]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _url_to_filename(url: str) -> str:
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    path = urlparse(url).path.strip("/").replace("/", "_")[:80]
    return f"{path}_{digest}.json"


def _classify_url(url: str) -> str:
    path = urlparse(url).path.lower()
    if re.search(r"/(match(es)?|result(s)?)/", path):
        return "match"
    # matchcenter/YYYY/M/D pages contain match-day data (scores, averages)
    if re.match(r"/matchcenter/\d{4}/\d{1,2}/\d{1,2}$", path):
        return "event"
    if re.search(r"/(event(s)?|tournament(s)?)/", path):
        return "event"
    if re.search(r"/(player(s)?|profile)/", path):
        return "player"
    return "other"


async def _extract_match_data(page: Any, url: str) -> dict[str, Any]:
    """
    Extract match scorecard data from a rendered Mastercaller match page.

    Targets:
    - Player names (from headings, table headers, or player divs)
    - Match score (sets/legs)
    - Per-leg breakdown table (P1 180s, P2 180s, P1 checkout, P2 checkout, darts)
    - Visit-level data (per-visit scores where rendered in tables)
    - Linked match URLs for depth-2 crawl
    """
    data: dict[str, Any] = {
        "url": url,
        "page_type": "match",
        "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
        "players": [],
        "match_score": {},
        "legs": [],
        "visits": [],
        "linked_match_urls": [],
    }

    # Page title
    title = await page.title()
    data["page_title"] = title

    # --- Players ---
    # Try common player name selectors
    for sel in ["h1", "h2", ".player-name", "[class*='player']", "strong"]:
        elements = await page.query_selector_all(sel)
        for el in elements:
            text = (await el.text_content() or "").strip()
            if text and 3 <= len(text) <= 60 and text not in data["players"]:
                # Filter out obvious non-player text
                if not any(
                    kw in text.lower()
                    for kw in ("match", "leg", "set", "score", "180", "checkout", "tournament")
                ):
                    data["players"].append(text)
        if len(data["players"]) >= 2:
            break

    # --- Match score (look for "X - Y" pattern) ---
    body_text = await page.evaluate("() => document.body.innerText")
    score_patterns = re.findall(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b", body_text)
    if score_patterns:
        # First match is likely the headline result
        p1_score, p2_score = score_patterns[0]
        data["match_score"] = {"p1": int(p1_score), "p2": int(p2_score)}

    # --- Per-leg breakdown tables ---
    tables = await page.query_selector_all("table")
    for table in tables:
        # Get column headers
        headers_els = await table.query_selector_all("thead th, thead td")
        headers = []
        for h in headers_els:
            text = (await h.text_content() or "").strip().lower()
            headers.append(text)

        if not headers:
            # Try first row as header
            first_row = await table.query_selector("tr")
            if first_row:
                cells = await first_row.query_selector_all("th, td")
                for c in cells:
                    text = (await c.text_content() or "").strip().lower()
                    headers.append(text)

        # Determine if this is a leg breakdown or visit table
        is_leg_table = any(
            kw in " ".join(headers)
            for kw in ("leg", "180", "checkout", "darts", "avg", "average")
        )
        is_visit_table = any(
            kw in " ".join(headers)
            for kw in ("visit", "score", "throw", "dart")
        )

        rows_els = await table.query_selector_all("tbody tr, tr")
        for row_el in rows_els:
            cells_els = await row_el.query_selector_all("td, th")
            cells = []
            for c in cells_els:
                text = (await c.text_content() or "").strip()
                cells.append(text)

            if not cells or all(c == "" for c in cells):
                continue

            if is_leg_table and cells:
                row_dict: dict[str, str] = {}
                for i, h in enumerate(headers):
                    if i < len(cells):
                        row_dict[h or f"col_{i}"] = cells[i]
                if row_dict:
                    data["legs"].append(row_dict)
            elif is_visit_table and cells:
                row_dict = {}
                for i, h in enumerate(headers):
                    if i < len(cells):
                        row_dict[h or f"col_{i}"] = cells[i]
                if row_dict:
                    data["visits"].append(row_dict)

    # --- 180 counts from text ---
    one80_patterns = re.findall(
        r"180[s]?\s*[:\-=]?\s*(\d+)", body_text, re.IGNORECASE
    )
    if one80_patterns:
        data["180_counts_raw"] = [int(x) for x in one80_patterns]

    # --- Checkout patterns ---
    checkout_patterns = re.findall(
        r"checkout[s]?\s*[:\-=]?\s*(\d+)%?", body_text, re.IGNORECASE
    )
    if checkout_patterns:
        data["checkout_raw"] = checkout_patterns[:10]

    # --- Average patterns ---
    avg_patterns = re.findall(
        r"(?:avg|average)[:\s]*(\d{2,3}\.?\d*)", body_text, re.IGNORECASE
    )
    if avg_patterns:
        data["averages_raw"] = avg_patterns[:6]

    # --- Linked match URLs ---
    links = await page.query_selector_all("a[href]")
    for link in links:
        href = await link.get_attribute("href")
        if not href:
            continue
        if any(kw in href.lower() for kw in ("/match", "/result", "/tournament", "/event")):
            full_url = href if href.startswith("http") else urljoin(url, href)
            if "mastercaller.com" in full_url and full_url not in data["linked_match_urls"]:
                data["linked_match_urls"].append(full_url)

    return data


async def _extract_event_data(page: Any, url: str) -> dict[str, Any]:
    """Extract tournament bracket / draw data from an event page."""
    data: dict[str, Any] = {
        "url": url,
        "page_type": "event",
        "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
        "event_name": "",
        "match_links": [],
        "rounds": [],
    }

    title = await page.title()
    data["event_name"] = title

    h1_el = await page.query_selector("h1")
    if h1_el:
        data["event_name"] = (await h1_el.text_content() or "").strip()

    # Collect all match links
    links = await page.query_selector_all("a[href]")
    for link in links:
        href = await link.get_attribute("href")
        if not href:
            continue
        if any(kw in href.lower() for kw in ("/match", "/result", "/player")):
            full_url = href if href.startswith("http") else urljoin(url, href)
            if "mastercaller.com" in full_url:
                data["match_links"].append(full_url)

    data["match_links"] = list(set(data["match_links"]))

    # Extract draw/bracket tables
    tables = await page.query_selector_all("table")
    for table in tables:
        rows_els = await table.query_selector_all("tr")
        for row_el in rows_els:
            cells_els = await row_el.query_selector_all("td, th")
            cells = []
            for c in cells_els:
                cells.append((await c.text_content() or "").strip())
            if cells and any(c for c in cells):
                data["rounds"].append({"raw_cells": cells})

    return data


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class MastercallerPlaywrightScraper:
    """
    Headless browser scraper for Mastercaller JS-rendered pages.

    Parameters
    ----------
    output_dir:
        Root directory for saving scraped JSON files.
    request_delay:
        Seconds to wait between page loads.
    resume:
        Skip URLs that already have a saved JSON file.
    headless:
        Run browser in headless mode (default True).
    max_pages:
        Limit number of pages to scrape (None = all).
    """

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        request_delay: float = REQUEST_DELAY_S,
        resume: bool = True,
        headless: bool = True,
        max_pages: Optional[int] = None,
    ) -> None:
        self.output_dir = output_dir
        self.request_delay = request_delay
        self.resume = resume
        self.headless = headless
        self.max_pages = max_pages
        self._log = logger.bind(scraper="MastercallerPlaywrightScraper")

    def _output_path(self, url: str) -> Path:
        page_type = _classify_url(url)
        filename = _url_to_filename(url)
        return self.output_dir / page_type / filename

    def _save(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    async def scrape_url(
        self,
        page: Any,
        url: str,
    ) -> Optional[dict[str, Any]]:
        """
        Navigate to a URL and extract structured match/event data.

        Waits for JS rendering before extraction.
        """
        out_path = self._output_path(url)
        if self.resume and out_path.exists():
            self._log.debug("skipping_cached", url=url)
            with out_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)

        try:
            await page.goto(url, timeout=PAGE_LOAD_TIMEOUT_MS, wait_until="networkidle")
        except Exception as exc:
            self._log.warning("page_load_failed", url=url, error=str(exc))
            # Try domcontentloaded as fallback
            try:
                await page.goto(url, timeout=PAGE_LOAD_TIMEOUT_MS, wait_until="domcontentloaded")
            except Exception as exc2:
                self._log.error("page_load_failed_fallback", url=url, error=str(exc2))
                return None

        # Wait for content to render
        for selector in _CONTENT_LOADED_SELECTORS:
            try:
                await page.wait_for_selector(selector, timeout=CONTENT_WAIT_TIMEOUT_MS)
                break
            except Exception:
                pass

        # Additional wait for JS rendering
        await asyncio.sleep(1.0)

        page_type = _classify_url(url)
        try:
            if page_type == "match":
                data = await _extract_match_data(page, url)
            elif page_type == "event":
                data = await _extract_event_data(page, url)
            else:
                # Generic: extract all text content
                title = await page.title()
                body_text = await page.evaluate("() => document.body.innerText")
                data = {
                    "url": url,
                    "page_type": "other",
                    "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
                    "title": title,
                    "text_preview": body_text[:1000],
                }
        except Exception as exc:
            self._log.error("extraction_error", url=url, error=str(exc))
            return None

        self._save(out_path, data)
        return data

    async def run(self, urls: list[str]) -> dict[str, list[dict[str, Any]]]:
        """
        Scrape all provided URLs using a persistent browser context.

        Returns results grouped by page type.
        """
        from playwright.async_api import async_playwright

        results: dict[str, list[dict[str, Any]]] = {
            "match": [],
            "event": [],
            "player": [],
            "other": [],
        }
        failed = 0
        total = len(urls)

        self._log.info("scraper_start", total=total)

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=self.headless,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ],
            )
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
                locale="en-GB",
            )
            page = await context.new_page()

            for i, url in enumerate(urls):
                if self.max_pages and i >= self.max_pages:
                    break

                self._log.info(
                    "scraping",
                    index=i + 1,
                    total=total,
                    url=url,
                )

                data = await self.scrape_url(page, url)
                if data:
                    page_type = data.get("page_type", "other")
                    results.setdefault(page_type, []).append(data)
                else:
                    failed += 1

                if i < total - 1:
                    await asyncio.sleep(self.request_delay)

            await context.close()
            await browser.close()

        self._log.info(
            "scraper_complete",
            total=total,
            match_pages=len(results.get("match", [])),
            event_pages=len(results.get("event", [])),
            failed=failed,
        )
        return results


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_urls_from_csv(csv_path: Path) -> list[str]:
    """Load URLs from a CSV file (first column or 'url' column)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    urls: list[str] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header:
            url_col = 0
            for i, col in enumerate(header):
                if "url" in col.lower():
                    url_col = i
                    break
        for row in reader:
            if row and len(row) > url_col:
                url = row[url_col].strip().strip('"')
                if url.startswith("http"):
                    urls.append(url)

    return urls


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Mastercaller Playwright scraper — JS-rendered match data"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV file with URLs to scrape",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Scrape a single URL (for testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip already-scraped pages",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Re-scrape all pages",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to scrape",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY_S,
        help="Delay between requests in seconds",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        default=False,
        help="Show browser window (for debugging)",
    )
    args = parser.parse_args()

    scraper = MastercallerPlaywrightScraper(
        output_dir=args.output_dir,
        request_delay=args.delay,
        resume=args.resume,
        headless=not args.no_headless,
        max_pages=args.max_pages,
    )

    if args.url:
        urls = [args.url]
    else:
        try:
            urls = load_urls_from_csv(args.input_csv)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"Scraping {len(urls)} URLs -> {args.output_dir}")
    results = await scraper.run(urls)

    match_count = len(results.get("match", []))
    event_count = len(results.get("event", []))
    other_count = len(results.get("other", []))
    print(
        f"Done - match={match_count} event={event_count} other={other_count}"
    )

    # Summary of what was extracted
    total_legs = sum(len(m.get("legs", [])) for m in results.get("match", []))
    total_visits = sum(len(m.get("visits", [])) for m in results.get("match", []))
    print(f"Extracted: {total_legs} leg rows, {total_visits} visit rows")


if __name__ == "__main__":
    asyncio.run(_main())
