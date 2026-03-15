"""
Runner script for DartsDatabase event scraper.

Reads event URLs from:
  D:/codex/Data/Darts/02_processed/csv/dartsdatabase/event_links.csv

The URLs have the format:
  https://www.dartsdatabase.co.uk/display-event.php?eid=24213&tna=...&eda=...

Each event page has one table with rows like:
  | Player1(avg) | N V N | Player2(avg) |
where the score uses non-breaking spaces: "N\xa0V\xa0N"

Data extracted per event:
  - event_id (eid param)
  - event_name (from page title)
  - event_date (eda param)
  - matches: list of {round, player1_name, player1_avg, player1_id,
                       score_p1, score_p2, player2_name, player2_avg, player2_id}

Rate limit: 1 req / 3s.
Output: D:/codex/Data/Darts/01_raw/json/dartsdatabase/events/
Checkpoint: D:/codex/Data/Darts/01_raw/json/dartsdatabase/_checkpoint.json
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from bs4 import BeautifulSoup

# Force UTF-8 stdout on Windows to handle non-ASCII characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVENT_LINKS_CSV = Path("D:/codex/Data/Darts/02_processed/csv/dartsdatabase/event_links.csv")
OUTPUT_DIR = Path("D:/codex/Data/Darts/01_raw/json/dartsdatabase/events")
CHECKPOINT_FILE = Path("D:/codex/Data/Darts/01_raw/json/dartsdatabase/_checkpoint.json")

REQUEST_DELAY = 3.0   # seconds between requests
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "XG3DartsResearch/1.0 "
        "(sports analytics; contact: research@xg3.ai; "
        "polite-crawl 1req/3s)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-GB,en;q=0.9",
}


def _sanitize_url(url: str) -> str:
    """
    URL-encode spaces/control characters in a URL.

    Python 3.13 is strict about control characters in URLs.
    urlparse misbehaves on URLs with literal spaces, so we avoid it:
    split on '?' to separate base from query, then re-encode the query.
    """
    if " " not in url and "\t" not in url:
        return url
    if "?" not in url:
        return urllib.parse.quote(url, safe="%/:@#!~*'(),;")
    base, query = url.split("?", 1)
    encoded_query = urllib.parse.urlencode(
        urllib.parse.parse_qsl(query, keep_blank_values=True, separator="&"),
    )
    return f"{base}?{encoded_query}"


def fetch_url(url: str) -> Optional[str]:
    """Fetch URL with retry/backoff.  Returns HTML or None."""
    url = _sanitize_url(url)
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                if resp.status == 200:
                    return resp.read().decode("utf-8", errors="replace")
                if resp.status in (404, 410):
                    print(f"  {resp.status}: {url}")
                    return None
                if resp.status == 429 or resp.status >= 500:
                    wait = 10 * attempt
                    print(f"  {resp.status} error, sleeping {wait}s ...")
                    time.sleep(wait)
                    continue
                print(f"  HTTP {resp.status}: {url} (attempt {attempt})")
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                print(f"  {e.code}: {url}")
                return None
            print(f"  HTTPError {e.code}: {url} (attempt {attempt})")
            if attempt <= MAX_RETRIES:
                time.sleep(6 * attempt)
        except Exception as e:
            print(f"  Error: {e} (attempt {attempt})")
            if attempt <= MAX_RETRIES:
                time.sleep(6 * attempt)
    return None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_player_cell(text: str) -> tuple[str, Optional[float]]:
    """
    Parse 'PlayerName(avg)' into (name, avg).

    Examples:
        'Wayne Warren(93.59)' -> ('Wayne Warren', 93.59)
        'Jim Williams(94.53)' -> ('Jim Williams', 94.53)
        'Some Player'         -> ('Some Player', None)
    """
    m = re.match(r"^(.+?)\s*\(([0-9.]+)\)\s*$", text.strip())
    if m:
        return m.group(1).strip(), float(m.group(2))
    return text.strip(), None


def _parse_score(text: str) -> tuple[Optional[int], Optional[int]]:
    """
    Parse score text like '3\xa0V\xa01' or '3 V 1' or '3-1'.

    Returns (p1_score, p2_score) or (None, None).
    """
    # Normalize: replace non-breaking spaces and em-dashes
    clean = text.replace("\xa0", " ").replace("–", "-").strip()
    # Try "N V N" format
    m = re.match(r"^(\d+)\s+V\s+(\d+)$", clean, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Try "N-N" format
    m = re.match(r"^(\d+)\s*[-]\s*(\d+)$", clean)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _extract_player_id(cell) -> Optional[str]:
    """Extract player ID from player-profile-live.php?pid=NNN link."""
    for a in cell.find_all("a", href=True):
        href = a.get("href", "")
        m = re.search(r"pid=(\d+)", href)
        if m:
            return m.group(1)
    return None


def parse_event_page(html: str, event_id: str, event_url: str) -> dict[str, Any]:
    """
    Parse a DartsDatabase event page.

    The page has a single table structured as:
      - Section header rows: one cell spanning multiple columns (e.g. "Last 40", "Quarter Final")
      - Match rows: 3 cells: Player1(avg) | Score | Player2(avg)

    Returns structured dict with all matches and round information.
    """
    soup = BeautifulSoup(html, "lxml")

    # Extract event metadata from title
    title_text = soup.title.get_text(strip=True) if soup.title else ""

    data: dict[str, Any] = {
        "event_id": event_id,
        "event_url": event_url,
        "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
        "event_title": title_text,
        "matches": [],
    }

    tables = soup.find_all("table")
    if not tables:
        data["parse_error"] = "no_tables_found"
        return data

    current_round = "Unknown"
    match_count = 0

    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue

            # Section header: single cell with colspan spanning multiple columns
            # OR single-cell row that is a round name like "Last 40", "Quarter Final", etc.
            if len(cells) == 1:
                cell_text = cells[0].get_text(strip=True)
                if cell_text:
                    current_round = cell_text
                continue

            # Empty or header row
            texts = [c.get_text(strip=True) for c in cells]
            if all(not t for t in texts):
                continue

            # Match row: expect 3 cells (player1, score, player2)
            if len(cells) >= 3:
                p1_text = texts[0]
                score_text = texts[1]
                p2_text = texts[2]

                p1_score, p2_score = _parse_score(score_text)
                if p1_score is None:
                    # Not a valid match row
                    continue

                p1_name, p1_avg = _parse_player_cell(p1_text)
                p2_name, p2_avg = _parse_player_cell(p2_text)
                p1_id = _extract_player_id(cells[0])
                p2_id = _extract_player_id(cells[2])

                match = {
                    "round": current_round,
                    "player1_name": p1_name,
                    "player1_avg": p1_avg,
                    "player1_id": p1_id,
                    "score_p1": p1_score,
                    "score_p2": p2_score,
                    "player2_name": p2_name,
                    "player2_avg": p2_avg,
                    "player2_id": p2_id,
                }
                data["matches"].append(match)
                match_count += 1

    data["match_count"] = match_count
    return data


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def extract_event_id(url: str) -> str:
    """Extract eid parameter from event URL."""
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    eids = params.get("eid", [])
    return eids[0] if eids else re.sub(r"[^a-zA-Z0-9_-]", "_", url[-30:])


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        try:
            with CHECKPOINT_FILE.open("r") as f:
                return set(json.load(f).get("done", []))
        except Exception:
            return set()
    return set()


def save_checkpoint(done: set[str]) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_FILE.open("w") as f:
        json.dump({
            "done": sorted(done),
            "updated": datetime.now(tz=timezone.utc).isoformat(),
        }, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load event URLs
    if not EVENT_LINKS_CSV.exists():
        raise FileNotFoundError(f"Event links CSV not found: {EVENT_LINKS_CSV}")

    with EVENT_LINKS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        event_urls = [row["event_url"] for row in reader if row.get("event_url")]

    print(f"Total event URLs: {len(event_urls)}")
    print(f"Output directory: {OUTPUT_DIR}")

    done = load_checkpoint()
    print(f"Already scraped: {len(done)} events (from checkpoint)")
    print()

    scraped_count = 0
    failed = []
    total_matches = 0

    for i, url in enumerate(event_urls):
        event_id = extract_event_id(url)
        out_path = OUTPUT_DIR / f"{event_id}.json"

        if event_id in done or out_path.exists():
            print(f"  [{i+1}/{len(event_urls)}] SKIP eid={event_id}")
            if event_id not in done:
                done.add(event_id)
            continue

        print(f"  [{i+1}/{len(event_urls)}] Scraping eid={event_id} ...")

        html = fetch_url(url)
        if html is None:
            print(f"    -> FAILED")
            failed.append({"event_id": event_id, "url": url})
        else:
            data = parse_event_page(html, event_id, url)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            done.add(event_id)
            scraped_count += 1
            mc = data.get("match_count", 0)
            total_matches += mc
            print(f"    -> OK  matches={mc}  title={data.get('event_title','?')[:60]}")

        # Checkpoint every 10 events
        if i % 10 == 0:
            save_checkpoint(done)

        # Rate limit (skip on last iteration)
        if i < len(event_urls) - 1:
            time.sleep(REQUEST_DELAY)

    save_checkpoint(done)

    if failed:
        failed_path = Path("D:/codex/Data/Darts/01_raw/json/dartsdatabase/_failed.json")
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        with failed_path.open("w") as f:
            json.dump(failed, f, indent=2)
        print(f"\nFailed events logged to: {failed_path}")

    summary = {
        "total_event_urls": len(event_urls),
        "newly_scraped": scraped_count,
        "total_done": len(done),
        "failed_count": len(failed),
        "total_matches_scraped": total_matches,
        "failed": failed,
        "output_dir": str(OUTPUT_DIR),
    }
    return summary


if __name__ == "__main__":
    result = run()
    print("\n=== SUMMARY ===")
    print(json.dumps(result, indent=2))
