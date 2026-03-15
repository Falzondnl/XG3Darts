"""
Enrich darts_players with DartsOrakel 3DA stats and dartsorakel_key.

Matches players by name similarity (slug matching + fuzzy fallback).
Updates dartsorakel_3da, dartsorakel_rank, dartsorakel_key fields.

Usage:
    python scripts/enrich_dartsorakel.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
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

DATA_ROOT = Path(settings.DATA_ROOT)
STATS_CSV = DATA_ROOT / "02_processed" / "csv" / "dartsorakel" / "stats_player.csv"


def _db_url_sync(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgres+asyncpg://", "postgresql://"
    )


def _slugify(name: str) -> str:
    """Convert 'Luke van Gerwen' → 'luke-van-gerwen'."""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^\w\s-]", "", name.lower())
    return re.sub(r"[\s_]+", "-", name.strip())


def _normalise(name: str) -> str:
    """Lowercase, strip accents, remove punctuation."""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db-url", default=None,
                        help="Override DATABASE_URL for local runs against public proxy")
    args = parser.parse_args()

    db_url = _db_url_sync(args.db_url or settings.DATABASE_URL)
    conn = psycopg2.connect(db_url, connect_timeout=30)

    # Load all players from DB
    log.info("loading_players_from_db")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, first_name, last_name, pdc_id FROM darts_players")
        db_players = cur.fetchall()
    log.info("db_players_loaded", count=len(db_players))

    # Build lookup: normalised_name → player_id
    norm_to_id: dict[str, str] = {}
    slug_to_id: dict[str, str] = {}
    for p in db_players:
        full = f"{p['first_name']} {p['last_name']}".strip()
        norm_to_id[_normalise(full)] = p["id"]
        slug_to_id[_slugify(full)] = p["id"]

    # Load DartsOrakel stats
    log.info("loading_dartsorakel_stats")
    with STATS_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log.info("dartsorakel_stats_loaded", count=len(rows))

    updates = []
    matched = 0
    unmatched = 0

    for row in rows:
        key = row.get("player_key", "").strip()
        name = row.get("player_name", "").strip()
        rank_raw = row.get("rank", "").strip()
        stat_raw = row.get("stat", "").strip()

        if not name:
            continue

        try:
            do_key = int(key) if key else None
        except ValueError:
            do_key = None

        try:
            stat = float(stat_raw) if stat_raw else None
        except ValueError:
            stat = None

        try:
            rank = int(rank_raw) if rank_raw else None
        except ValueError:
            rank = None

        # Try slug match first, then normalised name
        player_id = slug_to_id.get(_slugify(name)) or norm_to_id.get(_normalise(name))

        # Also try extracting slug from URL
        if not player_id:
            url = row.get("player_profile_url", "") or ""
            m = re.search(r"/player/details/\d+/([^/]+)", url)
            if m:
                player_id = slug_to_id.get(m.group(1))

        if player_id:
            updates.append((do_key, stat, rank, player_id))
            matched += 1
        else:
            unmatched += 1

    log.info("matching_done", matched=matched, unmatched=unmatched)

    if dry_run := args.dry_run:
        log.info("dry_run_skip", would_update=len(updates))
        conn.close()
        return

    if updates:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                UPDATE darts_players SET
                  dartsorakel_key  = data.key::BIGINT,
                  dartsorakel_3da  = data.stat::FLOAT,
                  dartsorakel_rank = data.rank::INT,
                  source_confidence = GREATEST(source_confidence, 0.90),
                  updated_at = NOW()
                FROM (VALUES %s) AS data(key, stat, rank, id)
                WHERE darts_players.id = data.id
                """,
                [(u[0], u[1], u[2], u[3]) for u in updates],
                page_size=500,
            )
        conn.commit()
        log.info("updated", count=len(updates))

    conn.close()
    log.info("done")


if __name__ == "__main__":
    main()
