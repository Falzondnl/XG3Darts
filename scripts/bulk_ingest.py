"""
Bulk ingestion script using psycopg2 for maximum throughput.

Loads PDC CSV data into Railway PostgreSQL using batch upserts (psycopg2).

Usage:
    python scripts/bulk_ingest.py [--dry-run] [--step {players,competitions,fixtures,all}]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
PDC_DIR = DATA_ROOT / "02_processed" / "csv" / "pdc"
FIXTURES_CSV  = PDC_DIR / "fixtures.csv"
PARTICIPANTS_CSV = PDC_DIR / "participants.csv"
TOURNAMENTS_CSV  = PDC_DIR / "tournaments.csv"

TOURNAMENT_TYPE_FORMAT_MAP: dict[int, str] = {
    1: "PDC_PC", 2: "PDC_WC", 3: "PDC_PL", 4: "PDC_GP", 5: "PDC_UK",
    6: "PDC_ET", 7: "PDC_WM", 8: "PDC_GS", 9: "PDC_PCF", 10: "PDC_PC",
    11: "PDC_MASTERS", 12: "PDC_WS", 13: "PDC_WCUP", 14: "PDC_WOM_SERIES",
    15: "PDC_DEVTOUR", 16: "PDC_CHALLENGE", 17: "PDC_WYC",
    77: "PDC_WM", 78: "PDC_MASTERS", 81: "WDF_OPEN", 92: "WDF_OPEN",
}


def _db_url_sync(url: str) -> str:
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    url = url.replace("postgres+asyncpg://", "postgresql://")
    return url


def _safe_int(v: Any) -> Optional[int]:
    if v is None or str(v).strip() in ("", "None", "nan"):
        return None
    try:
        return int(float(str(v)))
    except (ValueError, TypeError):
        return None


def _safe_float(v: Any) -> Optional[float]:
    if v is None or str(v).strip() in ("", "None", "nan"):
        return None
    try:
        return float(str(v))
    except (ValueError, TypeError):
        return None


def _safe_date(v: Any) -> Optional[date]:
    if not v or str(v).strip() in ("", "None", "nan"):
        return None
    s = str(v).strip()[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _format_code(type_id: Optional[int], name: str) -> str:
    if type_id and type_id in TOURNAMENT_TYPE_FORMAT_MAP:
        return TOURNAMENT_TYPE_FORMAT_MAP[type_id]
    name_lower = name.lower()
    if "world championship" in name_lower: return "PDC_WC"
    if "premier league" in name_lower: return "PDC_PL"
    if "world matchplay" in name_lower: return "PDC_WM"
    if "grand prix" in name_lower: return "PDC_GP"
    if "uk open" in name_lower: return "PDC_UK"
    if "grand slam" in name_lower: return "PDC_GS"
    if "players championship" in name_lower: return "PDC_PC"
    if "european" in name_lower: return "PDC_ET"
    if "world cup" in name_lower: return "PDC_WCUP"
    if "masters" in name_lower: return "PDC_MASTERS"
    if "women" in name_lower: return "PDC_WOM_SERIES"
    return "PDC_PC"


def _round_name(stage_raw: Any) -> str:
    if not stage_raw or str(stage_raw).strip() in ("", "None", "nan"):
        return "Unknown"
    s = str(stage_raw)
    m = re.search(r"'name'\s*:\s*'([^']+)'", s)
    return m.group(1) if m else "Unknown"


def _split_name(full: str) -> tuple[str, str]:
    """Split 'First Last' → ('First', 'Last'). Handles single names."""
    parts = full.strip().split(None, 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0] if parts else "Unknown", ""


def ingest_players(conn, dry_run: bool) -> dict[str, str]:
    """Load participants.csv → darts_players. Returns pdc_id → uuid map."""
    log.info("players_start")
    with PARTICIPANTS_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log.info("players_read", count=len(rows))

    now = datetime.now(timezone.utc)
    records = []
    pdc_id_to_uuid: dict[str, str] = {}

    for row in rows:
        pdc_id_raw = row.get("id", "").strip()
        if not pdc_id_raw:
            continue
        pdc_id = _safe_int(pdc_id_raw)
        if pdc_id is None:
            continue
        player_uuid = str(uuid.uuid4())
        pdc_id_to_uuid[str(pdc_id)] = player_uuid

        # CSV has separate first_name/last_name (or fall back to name field)
        first_name = (row.get("first_name", "") or "").strip()
        last_name = (row.get("last_name", "") or "").strip()
        if not first_name and not last_name:
            full = (row.get("name", "") or f"Player {pdc_id}").strip()
            first_name, last_name = _split_name(full)
        first_name = (first_name or "Unknown")[:120]
        last_name = (last_name or "")[:120]

        nickname = (row.get("nickname", "") or "").strip() or None
        country_code = (row.get("country_code", "") or row.get("nationality", "") or "").strip()[:10] or None
        dob_raw = row.get("dob") or row.get("date_of_birth")
        pdc_ranking = _safe_int(row.get("ranking") or row.get("pdc_ranking"))
        prize_money = _safe_int(row.get("prize_money"))
        tour_card_holder_raw = str(row.get("tour_card_holder", "") or "").strip().lower()
        tour_card_holder = tour_card_holder_raw in ("true", "1", "yes")

        records.append((
            player_uuid,
            first_name, last_name,
            nickname,
            None,   # slug (set later via entity resolution)
            pdc_id,
            None,   # dartsorakel_key
            None,   # dartsdatabase_id
            None,   # dartconnect_id
            None,   # wdf_id
            0.85,   # source_confidence
            "pdc",  # primary_source
            _safe_date(dob_raw),
            country_code,
            False,  # gdpr_anonymized
            None,   # gdpr_anonymized_at
            pdc_ranking,
            prize_money,
            tour_card_holder,
            None,   # tour_card_years (jsonb)
            None,   # dartsorakel_3da
            None,   # dartsorakel_rank
            now,    # created_at
            now,    # updated_at
        ))

    log.info("players_prepared", count=len(records))
    if dry_run:
        log.info("players_dry_run_skip")
        return pdc_id_to_uuid

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO darts_players
              (id, first_name, last_name, nickname, slug,
               pdc_id, dartsorakel_key, dartsdatabase_id, dartconnect_id, wdf_id,
               source_confidence, primary_source,
               dob, country_code, gdpr_anonymized, gdpr_anonymized_at,
               pdc_ranking, prize_money, tour_card_holder, tour_card_years,
               dartsorakel_3da, dartsorakel_rank,
               created_at, updated_at)
            VALUES %s
            ON CONFLICT (pdc_id) DO UPDATE
              SET first_name = EXCLUDED.first_name,
                  last_name  = EXCLUDED.last_name,
                  country_code = EXCLUDED.country_code,
                  updated_at = EXCLUDED.updated_at
            """,
            records,
            page_size=500,
        )
    conn.commit()

    # Rebuild UUID map from DB (handles conflict resolution)
    with conn.cursor() as cur:
        cur.execute("SELECT id, pdc_id FROM darts_players WHERE pdc_id IS NOT NULL")
        for row_db in cur.fetchall():
            pdc_id_to_uuid[str(row_db[1])] = row_db[0]

    log.info("players_done", count=len(records))
    return pdc_id_to_uuid


def ingest_competitions(conn, dry_run: bool) -> dict[str, str]:
    """Load tournaments.csv → darts_competitions. Returns pdc_tournament_id → uuid."""
    log.info("competitions_start")
    with TOURNAMENTS_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log.info("competitions_read", count=len(rows))

    now = datetime.now(timezone.utc)
    records = []
    tid_to_uuid: dict[str, str] = {}

    for row in rows:
        tid_raw = row.get("id", "").strip()
        if not tid_raw:
            continue
        tid = _safe_int(tid_raw)
        if tid is None:
            continue
        comp_uuid = str(uuid.uuid4())
        tid_to_uuid[str(tid)] = comp_uuid

        name = row.get("name", "").strip() or f"Tournament {tid}"
        season_year = _safe_int(row.get("season_id") or row.get("season") or row.get("year") or row.get("season_year")) or 2000
        type_id = _safe_int(row.get("tournament_type_id") or row.get("type_id"))
        format_code = _format_code(type_id, name)
        start_date = _safe_date(row.get("start_date"))
        end_date = _safe_date(row.get("end_date"))
        venue = (row.get("venue", "") or "").strip()[:200] or None
        city = (row.get("city", "") or "").strip()[:100] or None
        is_ranked_raw = str(row.get("is_ranked", "True") or "True").strip().lower()
        is_ranked = is_ranked_raw not in ("false", "0", "no")
        is_televised_raw = str(row.get("is_televised", "False") or "False").strip().lower()
        is_televised = is_televised_raw in ("true", "1", "yes")
        dartconnect_id = (row.get("dart_connect_id", "") or "").strip() or None
        sport_radar_tid = (row.get("sport_radar_tournament_id", "") or "").strip() or None

        records.append((
            comp_uuid, tid,
            sport_radar_tid,
            dartconnect_id,
            name[:300], season_year, format_code,
            None,  # format_era_code
            "PDC",  # organiser
            "pdc_mens",  # ecosystem
            venue, city,
            start_date, end_date,
            None,  # field_size
            is_ranked, is_televised,
            None,   # winner_player_id
            None,   # metadata_json
            now,    # created_at
        ))

    log.info("competitions_prepared", count=len(records))
    if dry_run:
        log.info("competitions_dry_run_skip")
        return tid_to_uuid

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO darts_competitions
              (id, pdc_tournament_id, sport_radar_tournament_id, dartconnect_id,
               name, season_year, format_code, format_era_code,
               organiser, ecosystem,
               venue, city, start_date, end_date,
               field_size, is_ranked, is_televised,
               winner_player_id, metadata_json,
               created_at)
            VALUES %s
            ON CONFLICT (pdc_tournament_id) DO UPDATE
              SET name = EXCLUDED.name,
                  format_code = EXCLUDED.format_code,
                  season_year = EXCLUDED.season_year
            """,
            records,
            page_size=500,
        )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT id, pdc_tournament_id FROM darts_competitions WHERE pdc_tournament_id IS NOT NULL")
        for row_db in cur.fetchall():
            tid_to_uuid[str(row_db[1])] = row_db[0]

    log.info("competitions_done", count=len(records))
    return tid_to_uuid


def ingest_fixtures(conn, pdc_id_to_uuid: dict, tid_to_uuid: dict, dry_run: bool) -> int:
    """Load fixtures.csv → darts_matches in batches of 5000."""
    log.info("fixtures_start")
    with FIXTURES_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log.info("fixtures_read", count=len(rows))

    now = datetime.now(timezone.utc)
    BATCH = 5000
    inserted = 0
    skipped = 0

    for batch_start in range(0, len(rows), BATCH):
        batch = rows[batch_start:batch_start + BATCH]
        records = []

        for row in batch:
            fid_raw = row.get("id", "").strip()
            if not fid_raw:
                skipped += 1
                continue
            pdc_fixture_id = _safe_int(fid_raw)

            p1_pdc_id = str(_safe_int(row.get("participant1_id")) or "")
            p2_pdc_id = str(_safe_int(row.get("participant2_id")) or "")
            player1_id = pdc_id_to_uuid.get(p1_pdc_id)
            player2_id = pdc_id_to_uuid.get(p2_pdc_id)

            if not player1_id or not player2_id:
                skipped += 1
                continue

            tid_raw = str(_safe_int(row.get("tournament_id")) or "")
            competition_id = tid_to_uuid.get(tid_raw)
            if not competition_id:
                skipped += 1
                continue

            match_uuid = str(uuid.uuid4())
            match_date = _safe_date(row.get("start_date") or row.get("date") or row.get("match_date"))
            match_time = (row.get("start_time", "") or "").strip()[:8] or None
            round_name = _round_name(row.get("stage") or row.get("round"))[:100]
            status_raw = (row.get("status", "") or "Completed").strip()
            # Normalise: PDC uses "Result" for completed matches
            status_map = {"Result": "Completed", "Fixture": "Fixture", "Live": "Live",
                          "Cancelled": "Cancelled", "Postponed": "Postponed"}
            status = status_map.get(status_raw, status_raw)[:30]
            sport_radar_id = (row.get("sport_radar_id", "") or "").strip() or None

            p1_score = _safe_int(row.get("participant1_score"))
            p2_score = _safe_int(row.get("participant2_score"))

            # Use winner_participant_id from CSV if available
            winner_pdc_id = str(_safe_int(row.get("winner_participant_id")) or "")
            if winner_pdc_id and winner_pdc_id in pdc_id_to_uuid:
                winner_id = pdc_id_to_uuid[winner_pdc_id]
            elif p1_score is not None and p2_score is not None:
                if p1_score > p2_score:
                    winner_id = player1_id
                elif p2_score > p1_score:
                    winner_id = player2_id
                else:
                    winner_id = None
            else:
                winner_id = None

            records.append((
                match_uuid,
                pdc_fixture_id,
                sport_radar_id,
                None,   # dartconnect_match_id
                competition_id,
                player1_id, player2_id,
                round_name, match_date, match_time,
                status,
                p1_score, p2_score,
                None,   # result_type
                winner_id,
                None,   # starter_player_id
                0.5,    # starter_confidence (unknown)
                0.0,    # visit_data_coverage
                "R0",   # coverage_regime
                None,   # raw_source_data
                "pdc",  # source_name
                now, now,
            ))

        if dry_run:
            inserted += len(records)
            if batch_start % 50000 == 0:
                log.info("fixtures_progress_dry", processed=batch_start + len(batch), total=len(rows))
            continue

        if records:
            # Deduplicate by pdc_fixture_id within batch to avoid intra-batch conflicts
            seen_fids: set = set()
            deduped = []
            for rec in records:
                fid = rec[1]  # pdc_fixture_id is index 1
                if fid is None or fid not in seen_fids:
                    deduped.append(rec)
                    if fid is not None:
                        seen_fids.add(fid)
            records = deduped

            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO darts_matches
                      (id, pdc_fixture_id, sport_radar_id, dartconnect_match_id,
                       competition_id, player1_id, player2_id,
                       round_name, match_date, match_time, status,
                       player1_score, player2_score, result_type, winner_player_id,
                       starter_player_id, starter_confidence,
                       visit_data_coverage, coverage_regime,
                       raw_source_data, source_name,
                       created_at, updated_at)
                    VALUES %s
                    ON CONFLICT (pdc_fixture_id) WHERE pdc_fixture_id IS NOT NULL
                    DO UPDATE SET
                      player1_score = EXCLUDED.player1_score,
                      player2_score = EXCLUDED.player2_score,
                      winner_player_id = EXCLUDED.winner_player_id,
                      status = EXCLUDED.status,
                      updated_at = EXCLUDED.updated_at
                    """,
                    records,
                    page_size=500,
                )
            conn.commit()

        inserted += len(records)
        if batch_start % 25000 < BATCH:
            log.info(
                "fixtures_progress",
                inserted=inserted,
                skipped=skipped,
                pct=f"{100*(batch_start+len(batch))//len(rows)}%",
            )

    log.info("fixtures_done", inserted=inserted, skipped=skipped, total=len(rows))
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--step", choices=["players", "competitions", "fixtures", "all"], default="all")
    args = parser.parse_args()

    db_url = _db_url_sync(settings.DATABASE_URL)
    log.info("connecting", host=db_url.split("@")[-1][:40])

    conn = psycopg2.connect(db_url, connect_timeout=30)
    conn.autocommit = False

    try:
        pdc_id_to_uuid: dict[str, str] = {}
        tid_to_uuid: dict[str, str] = {}

        if args.step in ("players", "all"):
            pdc_id_to_uuid = ingest_players(conn, args.dry_run)

        if args.step in ("competitions", "all"):
            tid_to_uuid = ingest_competitions(conn, args.dry_run)

        if args.step in ("fixtures", "all"):
            if not pdc_id_to_uuid:
                log.info("loading_player_map_from_db")
                with conn.cursor() as cur:
                    cur.execute("SELECT id, pdc_id FROM darts_players WHERE pdc_id IS NOT NULL")
                    for r in cur.fetchall():
                        pdc_id_to_uuid[str(r[1])] = r[0]
                log.info("player_map_loaded", count=len(pdc_id_to_uuid))
            if not tid_to_uuid:
                log.info("loading_competition_map_from_db")
                with conn.cursor() as cur:
                    cur.execute("SELECT id, pdc_tournament_id FROM darts_competitions WHERE pdc_tournament_id IS NOT NULL")
                    for r in cur.fetchall():
                        tid_to_uuid[str(r[1])] = r[0]
                log.info("competition_map_loaded", count=len(tid_to_uuid))
            ingest_fixtures(conn, pdc_id_to_uuid, tid_to_uuid, args.dry_run)

        log.info("ingest_complete")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
