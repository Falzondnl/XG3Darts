"""
Runner script for DartsOrakel per-player detail scraper.

What this actually extracts from the static HTML:
  - Annual win/loss/legs data (from data-new-array on #annual_w_l_chart)
  - FDI rating history (from data-new-array on #fdi_history_chart)
  - Tournament best results (from HTML tables: major results + other victories)
  - Player bio (country, DOB)

The "single-player-stat" values (Highest average, earnings, etc.) in the
HTML are all rendered as 0 because they are populated via JS after page load.
The actual per-match averages come from the /player/matches/ page (separate
endpoint).

Rate limit: 1 req / 2s.
Output: D:/codex/Data/Darts/01_raw/json/dartsorakel/player_details/
Checkpoint: D:/codex/Data/Darts/01_raw/json/dartsorakel/player_details/_checkpoint.json
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from bs4 import BeautifulSoup

# Force UTF-8 stdout on Windows to handle non-ASCII player names
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SEED_FILE = Path("D:/codex/Data/Darts/01_raw/json/dartsorakel/stats_player.json")
OUTPUT_DIR = Path("D:/codex/Data/Darts/01_raw/json/dartsorakel/player_details")
CHECKPOINT_FILE = OUTPUT_DIR / "_checkpoint.json"

BASE_URL = "https://dartsorakel.com"
DETAIL_URL = "{base}/player/details/{player_key}/"
REQUEST_DELAY = 2.0   # seconds between requests
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "XG3DartsResearch/1.0 "
        "(sports analytics; contact: research@xg3.ai; "
        "polite-crawl 1req/2s)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-GB,en;q=0.9",
}


def fetch_url(url: str) -> Optional[str]:
    """Fetch URL with retry/backoff.  Returns HTML or None."""
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                if resp.status == 200:
                    return resp.read().decode("utf-8", errors="replace")
                if resp.status == 404:
                    print(f"  404: {url}")
                    return None
                if resp.status == 429:
                    wait = 10 * attempt
                    print(f"  429 rate-limited, sleeping {wait}s ...")
                    time.sleep(wait)
                    continue
                print(f"  HTTP {resp.status}: {url} (attempt {attempt})")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  404: {url}")
                return None
            print(f"  HTTPError {e.code}: {url} (attempt {attempt})")
            if attempt <= MAX_RETRIES:
                time.sleep(5 * attempt)
        except Exception as e:
            print(f"  Error: {e} (attempt {attempt})")
            if attempt <= MAX_RETRIES:
                time.sleep(5 * attempt)
    return None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_player_detail_page(
    html: str,
    player_key: int,
    player_name: str,
) -> dict[str, Any]:
    """
    Parse DartsOrakel player detail page extracting all static HTML data.

    Data extracted:
    - annual_wl_data: list of {year_active, wins, losses, legs_won, legs_lost, matches}
    - fdi_history: dict of {date: {fdi, fdi_tv, fdi_rank}}
    - major_results: list of {tournament, best_result, year}
    - other_victories: list of {tournament, best_result, year}
    - bio: country, date_of_birth fields
    """
    soup = BeautifulSoup(html, "lxml")
    data: dict[str, Any] = {
        "player_key": player_key,
        "player_name": player_name,
        "scraped_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_url": DETAIL_URL.format(base=BASE_URL, player_key=player_key),
    }

    # --- Annual win/loss chart (data-new-array) ---
    awl_elem = soup.find(id="annual_w_l_chart")
    if awl_elem:
        raw = awl_elem.get("data-new-array", "")
        if raw:
            try:
                data["annual_wl_data"] = json.loads(raw)
            except json.JSONDecodeError:
                pass

    # --- FDI history chart (data-new-array) ---
    # Multiple elements may have data-new-array; FDI one has date keys
    for elem in soup.find_all(attrs={"data-new-array": True}):
        if elem.get("id") == "annual_w_l_chart":
            continue
        raw = elem.get("data-new-array", "")
        if raw and raw.startswith("{") and "fdi" in raw:
            try:
                data["fdi_history"] = json.loads(raw)
            except json.JSONDecodeError:
                pass

    # --- Player bio from stats-and-info section ---
    stats_info = soup.find("div", class_="stats-and-info")
    if stats_info:
        texts = [t.strip() for t in stats_info.stripped_strings if t.strip()]
        # First meaningful texts are usually: Name, Country, DOB(Age)
        if len(texts) >= 2:
            data["bio_country"] = texts[1] if len(texts) > 1 else None
        # Look for date pattern DD/MM/YYYY
        for t in texts:
            m = re.match(r"(\d{2}/\d{2}/\d{4})", t)
            if m:
                data["bio_dob"] = m.group(1)

    # --- Tournament results tables ---
    # Find all h3 headings then pair with the following table
    tables_data: dict[str, list] = {}
    for h3 in soup.find_all("h3"):
        heading = h3.get_text(strip=True)
        # Find the next table sibling
        sibling = h3
        for _ in range(10):
            sibling = sibling.find_next_sibling()
            if sibling is None:
                break
            if sibling.name == "table":
                rows = sibling.find_all("tr")
                records = []
                for row in rows[1:]:  # skip header
                    cells = row.find_all(["td", "th"])
                    row_texts = [c.get_text(strip=True) for c in cells]
                    if len(row_texts) >= 2 and any(row_texts):
                        record = {
                            "tournament": row_texts[0],
                            "best_result": row_texts[1] if len(row_texts) > 1 else "",
                            "year": row_texts[2] if len(row_texts) > 2 else "",
                        }
                        records.append(record)
                if records:
                    tables_data[heading] = records
                break
            if sibling.name == "h3":
                break

    if tables_data:
        data["tournament_records"] = tables_data

    # --- player-records-table (another name for the major results table) ---
    prt = soup.find(id="player-records-table")
    if prt and "tournament_records" not in data:
        rows = prt.find_all("tr")
        records = []
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            row_texts = [c.get_text(strip=True) for c in cells]
            if len(row_texts) >= 2:
                records.append({
                    "tournament": row_texts[0],
                    "best_result": row_texts[1] if len(row_texts) > 1 else "",
                    "year": row_texts[2] if len(row_texts) > 2 else "",
                })
        if records:
            data["major_records_table"] = records

    return data


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> set[int]:
    """Return set of player_keys already scraped."""
    if CHECKPOINT_FILE.exists():
        try:
            with CHECKPOINT_FILE.open("r") as f:
                return set(json.load(f).get("done", []))
        except Exception:
            return set()
    return set()


def save_checkpoint(done: set[int]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_FILE.open("w") as f:
        json.dump({"done": sorted(done), "updated": datetime.now(tz=timezone.utc).isoformat()}, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(max_players: int = 200) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load seed players
    if not SEED_FILE.exists():
        raise FileNotFoundError(f"Seed file not found: {SEED_FILE}")
    with SEED_FILE.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    all_players = raw.get("data", [])

    # Sort by rank (ascending) to get top players first
    all_players_sorted = sorted(all_players, key=lambda p: p.get("rank", 9999))
    players_to_scrape = all_players_sorted[:max_players]

    print(f"Total players in seed: {len(all_players)}")
    print(f"Scraping top {max_players} players by ranking")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    done = load_checkpoint()
    print(f"Already scraped: {len(done)} players (from checkpoint)")

    scraped_count = 0
    failed = []

    for i, player in enumerate(players_to_scrape):
        player_key = player["player_key"]
        player_name = player.get("player_name", f"Player_{player_key}")
        rank = player.get("rank", "?")
        out_path = OUTPUT_DIR / f"{player_key}.json"

        # Resume: skip if already done (checkpoint OR file exists)
        if player_key in done or out_path.exists():
            print(f"  [{i+1}/{max_players}] SKIP rank={rank} {player_name} (key={player_key})")
            if player_key not in done:
                done.add(player_key)
            continue

        url = DETAIL_URL.format(base=BASE_URL, player_key=player_key)
        print(f"  [{i+1}/{max_players}] Scraping rank={rank} {player_name} (key={player_key}) ...")

        html = fetch_url(url)
        if html is None:
            print(f"    -> FAILED (no response)")
            failed.append({"player_key": player_key, "player_name": player_name, "rank": rank})
        else:
            data = parse_player_detail_page(html, player_key, player_name)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            done.add(player_key)
            scraped_count += 1
            has_wl = "annual_wl_data" in data
            has_fdi = "fdi_history" in data
            has_tr = "tournament_records" in data
            print(f"    -> OK (wl_data={has_wl}, fdi={has_fdi}, tournament_records={has_tr})")

        # Save checkpoint every 10 players
        if i % 10 == 0:
            save_checkpoint(done)

        # Rate limit (skip on last iteration)
        if i < len(players_to_scrape) - 1:
            time.sleep(REQUEST_DELAY)

    # Final checkpoint
    save_checkpoint(done)

    # Save failures log
    if failed:
        failed_path = OUTPUT_DIR / "_failed.json"
        with failed_path.open("w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
        print(f"\nFailed players logged to: {failed_path}")

    summary = {
        "total_in_seed": len(all_players),
        "requested": max_players,
        "newly_scraped": scraped_count,
        "total_done": len(done),
        "failed_count": len(failed),
        "failed": failed,
        "output_dir": str(OUTPUT_DIR),
    }
    return summary


if __name__ == "__main__":
    max_p = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    result = run(max_players=max_p)
    print("\n=== SUMMARY ===")
    print(json.dumps(result, indent=2))
