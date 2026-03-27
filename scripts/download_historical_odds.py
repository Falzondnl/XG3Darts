"""
OddsPortal Historical Odds Downloader - Darts
==============================================
Downloads historical closing odds for darts tournaments from OddsPortal.
This is the CLV (Closing Line Value) infrastructure for the XG3 Darts model.

Coverage:
  - PDC World Championship (2015-2026)
  - PDC Premier League (2015-2026)
  - PDC World Matchplay (2015-2026)
  - PDC World Grand Prix (2015-2026)
  - PDC UK Open (2015-2026)
  - PDC Players Championship Finals (2015-2026)
  - PDC European Championship (2015-2026)
  - PDC Grand Slam of Darts (2015-2026)
  - BDO World Championship (historic, 2015-2020)

OddsPortal data provides:
  - Opening and closing moneyline/1X2 odds (darts is 2-way - no draw)
  - Handicap legs lines (equivalent to spread)
  - Over/Under legs totals closing lines

Notes on darts season format:
  - The PDC World Championship spans December-January.
    OddsPortal stores it under the January calendar year on most slugs.
    e.g. the 2024/2025 Worlds appears as season 2024 or 2025.
    Both single-URL and cross-year formats are tried (see download_league_season_page).
  - All other PDC events are within a single calendar year.
    season_format="calendar" is used for those.

CLV formula:
  CLV = (1/closing_prob - 1) - (bet_price - 1)
  Positive CLV = bet was taken at better-than-closing price.

Run: python download_historical_odds.py
     python download_historical_odds.py --league pdc_worlds --season 2023
     python download_historical_odds.py --dry-run --limit 2
Output: E:/DF/XG3V10/darts/XG3Darts/data/raw/odds/
"""

import argparse
import logging
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Logging - file + stdout (mirrors IH pattern)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("oddsportal_darts_download.log"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP session - browser-like headers to avoid bot detection
# ---------------------------------------------------------------------------
BASE_DIR = Path("E:/DF/XG3V10/darts/XG3Darts/data/raw/odds")
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.oddsportal.com/",
    }
)

OP_BASE = "https://www.oddsportal.com"

# ---------------------------------------------------------------------------
# LEAGUE CONFIGURATION
# OddsPortal URL slugs for PDC darts events.
# All PDC events appear under /darts/world/ on OddsPortal because the PDC
# runs its major events as international competitions.
# ---------------------------------------------------------------------------
LEAGUES: dict[str, dict] = {
    # ------------------------------------------------------------------
    # PDC World Championship
    # Takes place December-January. OddsPortal typically groups the full
    # tournament under the January year (e.g. Jan 2024 finish -> season 2024).
    # We attempt both "calendar" (single year) and cross-year forms.
    # ------------------------------------------------------------------
    "pdc_worlds": {
        "url": "/darts/world/pdc-world-championship",
        "output_dir": BASE_DIR / "pdc/worlds",
        "seasons": list(range(2015, 2026)),
        "season_format": "calendar",   # 2024 -> "2024" (January finish year)
        "description": "PDC World Darts Championship",
    },
    # ------------------------------------------------------------------
    # PDC Premier League Darts
    # Runs February-May, single calendar year.
    # ------------------------------------------------------------------
    "pdc_premier_league": {
        "url": "/darts/world/premier-league",
        "output_dir": BASE_DIR / "pdc/premier_league",
        "seasons": list(range(2015, 2026)),
        "season_format": "calendar",
        "description": "PDC Premier League Darts",
    },
    # ------------------------------------------------------------------
    # PDC World Matchplay
    # July, single calendar year (Blackpool event).
    # ------------------------------------------------------------------
    "pdc_world_matchplay": {
        "url": "/darts/world/pdc-world-matchplay",
        "output_dir": BASE_DIR / "pdc/world_matchplay",
        "seasons": list(range(2015, 2026)),
        "season_format": "calendar",
        "description": "PDC World Matchplay",
    },
    # ------------------------------------------------------------------
    # PDC World Grand Prix
    # October, single calendar year (double-start format).
    # ------------------------------------------------------------------
    "pdc_grand_prix": {
        "url": "/darts/world/pdc-world-grand-prix",
        "output_dir": BASE_DIR / "pdc/grand_prix",
        "seasons": list(range(2015, 2026)),
        "season_format": "calendar",
        "description": "PDC World Grand Prix",
    },
    # ------------------------------------------------------------------
    # PDC UK Open
    # February/March, single calendar year.
    # ------------------------------------------------------------------
    "pdc_uk_open": {
        "url": "/darts/world/pdc-uk-open",
        "output_dir": BASE_DIR / "pdc/uk_open",
        "seasons": list(range(2015, 2026)),
        "season_format": "calendar",
        "description": "PDC UK Open",
    },
    # ------------------------------------------------------------------
    # PDC Players Championship Finals
    # November, single calendar year.
    # ------------------------------------------------------------------
    "pdc_players_championship_finals": {
        "url": "/darts/world/pdc-players-championship-finals",
        "output_dir": BASE_DIR / "pdc/players_championship_finals",
        "seasons": list(range(2015, 2026)),
        "season_format": "calendar",
        "description": "PDC Players Championship Finals",
    },
    # ------------------------------------------------------------------
    # BDO / WDF World Championship (historic - BDO wound down 2020)
    # OddsPortal may carry limited coverage for earlier years.
    # ------------------------------------------------------------------
    "bdo_worlds": {
        "url": "/darts/world/bdo-world-championship",
        "output_dir": BASE_DIR / "bdo/worlds",
        "seasons": list(range(2015, 2021)),
        "season_format": "calendar",
        "description": "BDO World Darts Championship (historic, until 2020)",
    },
    # ------------------------------------------------------------------
    # PDC European Championship
    # October, single calendar year.
    # ------------------------------------------------------------------
    "pdc_european_championship": {
        "url": "/darts/world/pdc-european-championship",
        "output_dir": BASE_DIR / "pdc/european_championship",
        "seasons": list(range(2015, 2026)),
        "season_format": "calendar",
        "description": "PDC European Championship",
    },
    # ------------------------------------------------------------------
    # PDC Grand Slam of Darts
    # November, single calendar year (Wolverhampton).
    # ------------------------------------------------------------------
    "pdc_grand_slam": {
        "url": "/darts/world/pdc-grand-slam-of-darts",
        "output_dir": BASE_DIR / "pdc/grand_slam",
        "seasons": list(range(2015, 2026)),
        "season_format": "calendar",
        "description": "PDC Grand Slam of Darts",
    },
}


# ---------------------------------------------------------------------------
# Core HTTP helpers - exact mirror of IH pattern (07_download_oddsportal.py)
# ---------------------------------------------------------------------------

def _safe_get(url: str, retries: int = 3, delay: float = 2.0) -> str | None:
    """HTTP GET with retry, 429/403/404 handling.

    Mirrors 07_download_oddsportal.py exactly so future changes to the
    IH scraper can be copy-pasted here without adaptation.
    """
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 404:
                log.debug(f"404: {url}")
                return None
            if resp.status_code == 429:
                log.warning("Rate limited by OddsPortal - sleeping 60s")
                time.sleep(60)
                continue
            if resp.status_code == 403:
                log.warning(f"403 Forbidden - OddsPortal may require login: {url}")
                return None
            resp.raise_for_status()
            time.sleep(delay)
            return resp.text
        except Exception as exc:
            log.warning(f"Attempt {attempt + 1}/{retries}: {exc}")
            time.sleep(5 * (attempt + 1))
    return None


def _season_str(season: int, fmt: str) -> str:
    """Convert integer season start year to OddsPortal season string.

    fmt="calendar"  -> "2023"         (single calendar year - all PDC majors)
    fmt="cross"     -> "2023-2024"    (cross-calendar - domestic leagues)
    """
    if fmt == "calendar":
        return str(season)
    return f"{season}-{season + 1}"


def download_league_season_page(
    league: str,
    config: dict,
    season: int,
    dry_run: bool = False,
) -> bool:
    """Download a single OddsPortal tournament-season results page.

    Page contains match-level links for accessing individual match odds.
    Format: /darts/world/{tournament}/results/#/season:{season_str}/

    For the PDC Worlds (December-January) OddsPortal sometimes uses the
    cross-year format (YYYY-YYYY) even when the slug implies single-year.
    Both forms are tried before giving up.

    Returns True if page was saved (or already cached), False on failure.
    """
    fmt = config.get("season_format", "calendar")
    season_str = _season_str(season, fmt)

    out_dir = config["output_dir"] / "html"
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{league}_{season}_results.html"
    out_path = out_dir / filename

    url = f"{OP_BASE}{config['url']}/results/#/season:{season_str}/"

    if dry_run:
        log.info(f"[DRY-RUN] Would fetch: {url}")
        log.info(f"           Would save: {out_path}")
        return True

    if out_path.exists():
        log.info(f"SKIP (exists): {filename}")
        return True

    content = _safe_get(url)
    if content:
        out_path.write_text(content, encoding="utf-8")
        log.info(f"SAVED: {filename}")
        return True

    # For the Worlds, OddsPortal sometimes uses cross-year format even when
    # the slug implies single-year. Try alternate format if primary fails.
    if fmt == "calendar":
        alt_season_str = f"{season - 1}-{season}"
        url_alt = f"{OP_BASE}{config['url']}/results/#/season:{alt_season_str}/"
        content = _safe_get(url_alt)
        if content:
            out_path.write_text(content, encoding="utf-8")
            log.info(f"SAVED (alt cross-year): {filename}")
            return True

    # Final fallback - no season parameter (current season)
    url_fallback = f"{OP_BASE}{config['url']}/results/"
    content = _safe_get(url_fallback)
    if content:
        out_path.write_text(content, encoding="utf-8")
        log.info(f"SAVED (alt no-season): {filename}")
        return True

    log.warning(f"No OP data for {league} {season}")
    return False


def download_league_season_pages(
    league: str,
    config: dict,
    season: int,
    dry_run: bool = False,
    page_limit: int = 10,
    pages_total_counter: list[int] | None = None,
    limit_total: int | None = None,
) -> int:
    """Download paginated results for a tournament season.

    OddsPortal paginates at ~50 matches/page. Darts tournaments are
    smaller (typically 32-128 players, ~60-190 matches) so 4 pages is
    usually the max, but we follow the IH ceiling of 10 for safety.

    pages_total_counter and limit_total let the caller enforce a global
    page cap across all seasons and leagues (used by --limit N).  The cap
    is evaluated before each individual page fetch so --limit 2 truly stops
    after exactly 2 pages, not after a full season.

    Returns number of pages downloaded (or that would be downloaded in
    dry-run mode).
    """
    fmt = config.get("season_format", "calendar")
    season_str = _season_str(season, fmt)

    out_dir = config["output_dir"] / "html"
    out_dir.mkdir(parents=True, exist_ok=True)

    pages_downloaded = 0
    max_pages = min(page_limit, 10)  # never exceed 10 - OddsPortal hard limit

    for page in range(1, max_pages + 1):
        # Check global page cap before every individual page
        if limit_total is not None and pages_total_counter is not None:
            if pages_total_counter[0] >= limit_total:
                log.info(f"[LIMIT] Reached --limit {limit_total} pages, stopping.")
                break

        filename = f"{league}_{season}_p{page}.html"
        out_path = out_dir / filename
        url = f"{OP_BASE}{config['url']}/results/#/season:{season_str}/page:{page}/"

        if dry_run:
            log.info(f"[DRY-RUN] Would fetch: {url}")
            log.info(f"           Would save: {out_path}")
            pages_downloaded += 1
            if pages_total_counter is not None:
                pages_total_counter[0] += 1
            continue

        if out_path.exists():
            pages_downloaded += 1
            if pages_total_counter is not None:
                pages_total_counter[0] += 1
            continue

        content = _safe_get(url)

        if not content:
            break

        # Stop if page has no match content
        if "No data available" in content or len(content) < 5000:
            log.info(f"  No more pages for {league} {season} after page {page - 1}")
            break

        out_path.write_text(content, encoding="utf-8")
        log.info(f"SAVED: {filename}")
        pages_downloaded += 1
        if pages_total_counter is not None:
            pages_total_counter[0] += 1

    return pages_downloaded


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def download_league(
    league_key: str,
    config: dict,
    seasons: list[int] | None,
    dry_run: bool = False,
    page_limit: int = 10,
    pages_total_counter: list[int] | None = None,
    limit_total: int | None = None,
) -> int:
    """Download all seasons for a single tournament.  Returns total pages fetched."""
    target_seasons = seasons if seasons is not None else config["seasons"]
    log.info(
        f"\n--- {config['description']} ({league_key}) ---"
        f" seasons: {target_seasons[0]}-{target_seasons[-1]}"
    )
    config["output_dir"].mkdir(parents=True, exist_ok=True)

    league_pages = 0
    for season in target_seasons:
        # Check global cap between seasons too (fast-exit if already hit)
        if limit_total is not None and pages_total_counter is not None:
            if pages_total_counter[0] >= limit_total:
                log.info(f"[LIMIT] Reached --limit {limit_total}, stopping league loop.")
                break

        pages = download_league_season_pages(
            league_key,
            config,
            season,
            dry_run=dry_run,
            page_limit=page_limit,
            pages_total_counter=pages_total_counter,
            limit_total=limit_total,
        )
        log.info(f"  {league_key} {season}: {pages} pages")
        league_pages += pages

    return league_pages


def download_all(
    league_filter: str = "all",
    season_filter: str = "all",
    dry_run: bool = False,
    limit: int | None = None,
) -> None:
    """Download all historical darts odds data."""
    log.info("=" * 60)
    log.info("OddsPortal Historical Odds Downloader - XG3 Darts")
    log.info("=" * 60)
    log.info("")
    log.info("PURPOSE : Historical closing odds for CLV infrastructure.")
    log.info("PRIMARY  : Optic Odds (already integrated in platform).")
    log.info("THIS     : Supplementary historical data (pre-Optic-Odds era).")
    if dry_run:
        log.info("[DRY-RUN MODE] URLs will be printed but not fetched.")
    log.info("")

    # Determine which tournaments to process
    if league_filter == "all":
        leagues_to_run = LEAGUES
    else:
        if league_filter not in LEAGUES:
            log.error(
                f"Unknown league '{league_filter}'. "
                f"Valid options: {', '.join(LEAGUES.keys())}"
            )
            return
        leagues_to_run = {league_filter: LEAGUES[league_filter]}

    # Determine season filter
    season_list: list[int] | None = None
    if season_filter != "all":
        try:
            season_list = [int(season_filter)]
        except ValueError:
            log.error(
                f"Invalid --season value: '{season_filter}' - must be an integer year."
            )
            return

    # Shared mutable counter for cross-league global page cap (--limit N)
    pages_counter: list[int] = [0]

    total_pages = 0
    for league_key, config in leagues_to_run.items():
        if limit is not None and pages_counter[0] >= limit:
            log.info(f"[LIMIT] Reached --limit {limit}, stopping all downloads.")
            break

        pages = download_league(
            league_key,
            config,
            seasons=season_list,
            dry_run=dry_run,
            pages_total_counter=pages_counter,
            limit_total=limit,
        )
        total_pages += pages

    log.info("")
    log.info("=" * 60)
    log.info(
        f"OddsPortal Darts download COMPLETE - {total_pages} page(s) processed"
    )
    log.info("=" * 60)
    log.info("")
    log.info("NEXT STEPS FOR CLV PIPELINE:")
    log.info("1. Parse saved HTML pages to extract match IDs and match-level odds")
    log.info("2. Cross-reference with canonical results (dartsorakel / live.dartsorakel)")
    log.info("3. Compute CLV = (1/closing_prob - 1) - (bet_odds - 1)")
    log.info("4. Optic Odds is PRIMARY source for ongoing CLV tracking")
    log.info("5. OddsPortal fills historical gap for pre-2022 data")
    log.info("6. Note: Darts matching uses player name - use name-normaliser from enrich_dartsorakel.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    league_help_lines = "\n  ".join(
        f"{k:<35} {v['description']}" for k, v in LEAGUES.items()
    )
    parser = argparse.ArgumentParser(
        description="Download historical darts odds from OddsPortal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Download everything
  python download_historical_odds.py

  # Download only PDC Worlds
  python download_historical_odds.py --league pdc_worlds

  # Download only one season of Premier League
  python download_historical_odds.py --league pdc_premier_league --season 2023

  # Dry-run: print URLs without fetching (first 2 pages total)
  python download_historical_odds.py --dry-run --limit 2

  # List all available league keys
  python download_historical_odds.py --list-leagues

Available league keys:
  {league_help_lines}
""",
    )
    parser.add_argument(
        "--league",
        default="all",
        help='Tournament key to download, or "all" (default: all). '
             "Use --list-leagues to see all keys.",
    )
    parser.add_argument(
        "--season",
        default="all",
        help=(
            'Season calendar year to download (e.g. 2023), or "all" (default: all).'
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs that would be fetched without making any HTTP requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N total pages have been processed (useful for testing).",
    )
    parser.add_argument(
        "--list-leagues",
        action="store_true",
        help="Print all available tournament keys and exit.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.list_leagues:
        print("\nAvailable darts league/tournament keys:\n")
        for key, cfg in LEAGUES.items():
            seasons = cfg["seasons"]
            print(
                f"  {key:<35} {cfg['description']:<55} "
                f"seasons: {seasons[0]}-{seasons[-1]}"
            )
        print()
        return

    download_all(
        league_filter=args.league,
        season_filter=args.season,
        dry_run=args.dry_run,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
