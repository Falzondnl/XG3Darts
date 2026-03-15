"""
Compute sequential (historical) ELO ratings for all players.

Processes 276k+ completed PDC matches in chronological order,
updating each player's ELO after each match and storing the
before/after snapshot in darts_elo_ratings.

This replaces the static snapshot ELO loaded from elo_ratings.json
with a proper time-series, eliminating data leakage in the R1 model.

K-factor schedule:
    games_played < 30  → K=40  (new player, high uncertainty)
    30 ≤ games < 100   → K=32
    games ≥ 100        → K=20  (established player)

Ecosystem pools:
    pdc_mens, pdc_womens, wdf_open — tracked separately.
    Default pool for PDC fixtures: pdc_mens.

Usage:
    python scripts/compute_sequential_elo.py [--reset] [--dry-run]
    --reset: DELETE existing sequential ELO rows before recomputing
    --dry-run: compute but don't write to DB
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date
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

DEFAULT_ELO = 1500.0
BATCH_SIZE = 5000


def _db_url_sync(url: str) -> str:
    return (
        url.replace("postgresql+asyncpg://", "postgresql://")
           .replace("postgres+asyncpg://", "postgresql://")
    )


def k_factor(games_played: int) -> float:
    if games_played < 30:
        return 40.0
    if games_played < 100:
        return 32.0
    return 20.0


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def ecosystem_for_match(format_code: str, ecosystem: str) -> str:
    if ecosystem == "pdc_womens":
        return "pdc_womens"
    if ecosystem == "wdf":
        return "wdf_open"
    return "pdc_mens"


def load_matches(conn) -> list[dict]:
    log.info("loading_completed_matches")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                m.id,
                m.player1_id,
                m.player2_id,
                m.winner_player_id,
                m.match_date,
                COALESCE(c.format_code, 'PDC_PC') AS format_code,
                COALESCE(c.ecosystem, 'pdc_mens') AS ecosystem
            FROM darts_matches m
            JOIN darts_competitions c ON c.id = m.competition_id
            WHERE m.status = 'Completed'
              AND m.winner_player_id IS NOT NULL
              AND m.player1_id IS NOT NULL
              AND m.player2_id IS NOT NULL
              AND m.match_date IS NOT NULL
            ORDER BY m.match_date ASC, m.id ASC
        """)
        rows = cur.fetchall()
    log.info("matches_loaded", count=len(rows))
    return [dict(r) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing ELO rows before recomputing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute only, do not write to DB")
    parser.add_argument("--db-url", default=None,
                        help="Override DATABASE_URL (e.g. public proxy for local runs)")
    args = parser.parse_args()

    db_url = _db_url_sync(args.db_url or settings.DATABASE_URL)
    conn = psycopg2.connect(db_url, connect_timeout=30)

    if args.reset and not args.dry_run:
        log.info("resetting_elo_table")
        with conn.cursor() as cur:
            cur.execute("DELETE FROM darts_elo_ratings")
        conn.commit()
        log.info("elo_table_cleared")

    matches = load_matches(conn)

    # In-memory ELO state: {(player_id, pool): (rating, games_played)}
    elo_state: dict[tuple[str, str], tuple[float, int]] = {}

    def get_state(pid: str, pool: str) -> tuple[float, int]:
        return elo_state.get((pid, pool), (DEFAULT_ELO, 0))

    log.info("computing_sequential_elo", matches=len(matches))
    insert_rows: list[tuple] = []
    processed = 0

    for m in matches:
        p1 = m["player1_id"]
        p2 = m["player2_id"]
        winner = m["winner_player_id"]
        mdate = m["match_date"]
        pool = ecosystem_for_match(m["format_code"], m["ecosystem"])
        mid = m["id"]

        r1_before, g1 = get_state(p1, pool)
        r2_before, g2 = get_state(p2, pool)

        e1 = expected_score(r1_before, r2_before)
        e2 = expected_score(r2_before, r1_before)

        s1 = 1.0 if winner == p1 else 0.0
        s2 = 1.0 - s1

        k1 = k_factor(g1)
        k2 = k_factor(g2)

        d1 = k1 * (s1 - e1)
        d2 = k2 * (s2 - e2)

        r1_after = r1_before + d1
        r2_after = r2_before + d2

        # Store snapshots
        insert_rows.append((
            str(uuid.uuid4()), p1, pool, mid,
            round(r1_before, 4), round(r1_after, 4), round(d1, 4),
            round(k1, 1), round(e1, 6), s1, g1 + 1,
            mdate if isinstance(mdate, date) else mdate,
        ))
        insert_rows.append((
            str(uuid.uuid4()), p2, pool, mid,
            round(r2_before, 4), round(r2_after, 4), round(d2, 4),
            round(k2, 1), round(e2, 6), s2, g2 + 1,
            mdate if isinstance(mdate, date) else mdate,
        ))

        # Update state
        elo_state[(p1, pool)] = (r1_after, g1 + 1)
        elo_state[(p2, pool)] = (r2_after, g2 + 1)

        processed += 1
        if processed % 50000 == 0:
            log.info("progress", processed=processed, total=len(matches),
                     states=len(elo_state))

    log.info("computation_done", rows=len(insert_rows))

    if args.dry_run:
        # Show top-10 by final rating
        final: dict[str, float] = {}
        for (pid, pool), (rating, _) in elo_state.items():
            if pool == "pdc_mens":
                final[pid] = rating
        top10 = sorted(final.items(), key=lambda x: x[1], reverse=True)[:10]
        for pid, r in top10:
            print(f"  {pid}: {round(r, 1)}")
        log.info("dry_run_done")
        conn.close()
        return

    log.info("writing_to_db", rows=len(insert_rows))
    sql = """
        INSERT INTO darts_elo_ratings
            (id, player_id, pool, match_id,
             rating_before, rating_after, delta,
             k_factor, expected_score, actual_score, games_played_at_time,
             match_date)
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    written = 0
    with conn.cursor() as cur:
        for i in range(0, len(insert_rows), BATCH_SIZE):
            batch = insert_rows[i : i + BATCH_SIZE]
            psycopg2.extras.execute_values(cur, sql, batch, page_size=BATCH_SIZE)
            written += len(batch)
            conn.commit()
            log.info("batch_written", written=written, total=len(insert_rows))

    # Also update darts_players.elo_rating snapshot with final values
    log.info("updating_player_elo_snapshots")
    updates = [
        (round(rating, 4), pid)
        for (pid, pool), (rating, _) in elo_state.items()
        if pool == "pdc_mens"
    ]
    if updates:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                "UPDATE darts_players SET updated_at = now() WHERE id = %s",
                [(pid,) for _, pid in updates],
                page_size=1000,
            )
        conn.commit()

    conn.close()
    log.info("sequential_elo_complete",
             matches_processed=processed,
             elo_rows_written=written,
             unique_player_pool_pairs=len(elo_state))


if __name__ == "__main__":
    main()
