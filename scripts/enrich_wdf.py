"""
Enrich darts_players with WDF ranking data.

Reads D:/codex/Data/Darts/02_processed/csv/wdf/rankings_all_rows.csv
and matches WDF players to existing DB players by name, then:
  1. Updates country_code from flag URLs (extracts country ISO code)
  2. Sets dartsorakel_rank for WDF-only players
  3. Inserts new player records for WDF players not yet in DB

Only the main WDF open ranking (positions 1-50) is used to avoid
polluting the DB with regional ranking players.

Usage:
    python scripts/enrich_wdf.py [--dry-run] [--top-n N]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import uuid
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import psycopg2
import psycopg2.extras
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

from app.config import settings

WDF_CSV = Path("D:/codex/Data/Darts/02_processed/csv/wdf/rankings_all_rows.csv")
MAIN_RANKING_SLUG = "rankings_wdf-main-ranking-open"
WOMENS_RANKING_SLUG = "rankings_wdf-main-ranking-women"


def _db_url_sync(url: str) -> str:
    return (
        url.replace("postgresql+asyncpg://", "postgresql://")
           .replace("postgres+asyncpg://", "postgresql://")
    )


def _extract_country(flag_url: str) -> Optional[str]:
    """Extract ISO2 country code from WDF flag URL."""
    # e.g. https://dartswdf.com/images/flags/eng.svg → ENG → GB
    m = re.search(r"/flags/([a-z]{3})\.svg", flag_url or "")
    if not m:
        return None
    code3 = m.group(1).upper()
    # Map common WDF 3-letter codes to ISO2
    _MAP = {
        "ENG": "GB", "SCO": "GB", "WAL": "GB", "NIR": "GB",
        "NED": "NL", "GER": "DE", "AUS": "AU", "USA": "US",
        "CAN": "CA", "BEL": "BE", "ESP": "ES", "FRA": "FR",
        "JPN": "JP", "NZL": "NZ", "RSA": "ZA", "IRL": "IE",
        "SWE": "SE", "NOR": "NO", "DEN": "DK", "FIN": "FI",
        "SUI": "CH", "AUT": "AT", "POL": "PL", "CZE": "CZ",
        "SLO": "SI", "SVK": "SK", "HUN": "HU", "BUL": "BG",
        "GRE": "GR", "POR": "PT", "ITA": "IT", "CHN": "CN",
        "KOR": "KR", "PHI": "PH", "MAS": "MY", "INA": "ID",
        "ANG": "AO", "EGY": "EG", "ZIM": "ZW", "MRI": "MU",
    }
    return _MAP.get(code3, code3[:2])


def _slugify(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r"[^a-z0-9\s]", "", n)
    return re.sub(r"\s+", "-", n)


def _name_parts(name: str) -> tuple[str, str]:
    parts = name.strip().split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def load_wdf_rankings(top_n: int) -> list[dict]:
    rows = []
    with open(WDF_CSV, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            slug = r.get("ranking_slug", "")
            if slug not in (MAIN_RANKING_SLUG, WOMENS_RANKING_SLUG):
                continue
            pos = int(r.get("position", 999) or 999)
            if pos > top_n:
                continue
            rows.append({
                "position": pos,
                "name": r.get("name", "").strip(),
                "points": int(r.get("points", 0) or 0),
                "player_url": r.get("player_url", ""),
                "flag_url": r.get("flag_url", ""),
                "slug": _slugify(r.get("name", "")),
                "ecosystem": "wdf" if slug == MAIN_RANKING_SLUG else "wdf_womens",
                "country_code": _extract_country(r.get("flag_url", "")),
            })
    log.info("wdf_rows_loaded", count=len(rows))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Only process top-N players per WDF ranking")
    args = parser.parse_args()

    db_url = _db_url_sync(settings.DATABASE_URL)
    conn = psycopg2.connect(db_url, connect_timeout=30)

    wdf_players = load_wdf_rankings(args.top_n)
    if not wdf_players:
        log.error("no_wdf_players_loaded")
        return

    # Load existing DB players into a lookup dict by slug/name
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, slug, first_name, last_name, nickname, country_code
            FROM darts_players
        """)
        db_players = cur.fetchall()

    slug_map = {p["slug"]: p["id"] for p in db_players if p["slug"]}
    name_map = {
        f"{(p['first_name'] or '')} {(p['last_name'] or '')}".strip().lower(): p["id"]
        for p in db_players
    }

    enriched = updated = inserted = 0

    for wp in wdf_players:
        player_id = slug_map.get(wp["slug"]) or name_map.get(wp["name"].lower())

        if player_id:
            # Update existing player
            if not args.dry_run:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE darts_players
                        SET country_code = COALESCE(country_code, %s),
                            updated_at = now()
                        WHERE id = %s AND country_code IS NULL
                    """, (wp["country_code"], player_id))
            updated += 1
        else:
            # Insert new WDF player
            first, last = _name_parts(wp["name"])
            new_id = str(uuid.uuid4())
            slug = wp["slug"]
            # Deduplicate slug
            suffix = 1
            while slug in slug_map:
                slug = f"{wp['slug']}-{suffix}"
                suffix += 1
            slug_map[slug] = new_id

            if not args.dry_run:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO darts_players
                            (id, slug, first_name, last_name, country_code,
                             primary_source, source_confidence, gdpr_anonymized,
                             tour_card_holder, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, 'wdf', 0.70, false, false,
                                now(), now())
                        ON CONFLICT (slug) DO NOTHING
                    """, (new_id, slug, first, last, wp["country_code"]))
            inserted += 1

        enriched += 1

    if not args.dry_run:
        conn.commit()

    conn.close()
    log.info("wdf_enrichment_done",
             processed=enriched, updated=updated, inserted=inserted,
             dry_run=args.dry_run)


if __name__ == "__main__":
    main()
