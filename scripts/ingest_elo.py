"""
Load pre-computed ELO ratings into darts_elo_ratings table.

Reads elo_ratings.json (4,811 players rated from 219,748 matches)
and upserts into darts_elo_ratings as a snapshot record.

Usage:
    python scripts/ingest_elo.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

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
ELO_JSON = DATA_ROOT / "02_processed" / "elo_ratings.json"


def _db_url_sync(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgres+asyncpg://", "postgresql://"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db-url", default=None,
                        help="Override DATABASE_URL (e.g. public proxy for local runs)")
    args = parser.parse_args()

    log.info("loading_elo_json")
    with ELO_JSON.open("r") as f:
        elo_data = json.load(f)

    # Handle {"ratings": {...}} wrapper or direct dict
    if isinstance(elo_data, dict) and "ratings" in elo_data and isinstance(elo_data["ratings"], dict):
        ratings_dict = elo_data["ratings"]
    else:
        ratings_dict = elo_data

    log.info("elo_loaded", players=len(ratings_dict))

    db_url = _db_url_sync(args.db_url or settings.DATABASE_URL)
    conn = psycopg2.connect(db_url, connect_timeout=30)

    # Load pdc_id → player_id map
    log.info("loading_player_map")
    with conn.cursor() as cur:
        cur.execute("SELECT id, pdc_id FROM darts_players WHERE pdc_id IS NOT NULL")
        pdc_to_uuid: dict[str, str] = {str(r[1]): r[0] for r in cur.fetchall()}
    log.info("player_map_loaded", count=len(pdc_to_uuid))

    snapshot_date = date.today()
    now = datetime.now(timezone.utc)
    records = []
    unmatched = 0

    for pdc_id_str, rating_info in ratings_dict.items():
        player_id = pdc_to_uuid.get(str(pdc_id_str))
        if not player_id:
            unmatched += 1
            continue

        # elo_ratings.json stores: {pdc_id: {"rating": float, "games": int}}
        # or just a float rating
        if isinstance(rating_info, dict):
            rating = float(rating_info.get("rating", 1500.0))
            games = int(rating_info.get("games", 0))
        else:
            rating = float(rating_info)
            games = 0

        records.append((
            str(uuid.uuid4()),
            player_id,
            "pdc_mens",   # pool
            None,         # match_id (snapshot, not per-match)
            1500.0,       # rating_before (initial)
            rating,       # rating_after (current)
            rating - 1500.0,  # delta
            20.0,         # k_factor (standard)
            0.5,          # expected_score
            0.5,          # actual_score
            games,        # games_played_at_time
            snapshot_date,
            now,
        ))

    log.info("records_prepared", matched=len(records), unmatched=unmatched)

    if args.dry_run:
        log.info("dry_run_skip")
        conn.close()
        return

    if records:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO darts_elo_ratings
                  (id, player_id, pool, match_id,
                   rating_before, rating_after, delta,
                   k_factor, expected_score, actual_score,
                   games_played_at_time, match_date, created_at)
                VALUES %s
                ON CONFLICT DO NOTHING
                """,
                records,
                page_size=500,
            )
        conn.commit()
        log.info("elo_inserted", count=len(records))

    conn.close()
    log.info("done")


if __name__ == "__main__":
    main()
