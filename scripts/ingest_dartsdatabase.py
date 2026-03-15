"""
Ingest DartsDatabase event match data into darts_matches and darts_player_stats.

Reads scraped JSON files from:
    D:/codex/Data/Darts/01_raw/json/dartsdatabase/events/

For each scraped event:
  1. Upsert competition row in darts_competitions
  2. Name-match players to existing darts_players rows
  3. Register new DartsDatabase-only players
  4. Upsert each match into darts_matches (source_name='dartsdatabase')
  5. Upsert per-match averages into darts_player_stats

Run with:
    python scripts/ingest_dartsdatabase.py [--db-url <url>] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional
from unicodedata import normalize

import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVENT_DIR = Path("D:/codex/Data/Darts/01_raw/json/dartsdatabase/events")
DEFAULT_DB_URL = "postgresql://postgres:ccclcQbnJbRiNzAcirrYPOsLnmYlcICT@metro.proxy.rlwy.net:26269/railway"


# ---------------------------------------------------------------------------
# Name normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip accents, remove punctuation for fuzzy matching."""
    text = normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9 ]", "", text.lower())
    return " ".join(text.split())


def _split_name(full_name: str) -> tuple[str, str]:
    """Split 'Firstname Lastname' — last token is last_name, rest is first_name."""
    parts = full_name.strip().split()
    if len(parts) == 1:
        return parts[0], ""
    return " ".join(parts[:-1]), parts[-1]


# ---------------------------------------------------------------------------
# Competition title → code mapping
# ---------------------------------------------------------------------------

TITLE_TO_CODE: list[tuple[str, str]] = [
    ("bdo world youth", "WDF_WC"),
    ("bdo world", "WDF_WC"),
    ("pdc world youth", "PDC_WYC"),
    ("premier league", "PDC_PL"),
    ("world masters", "PDC_WM"),
    ("pdpa players championship", "PDC_PC"),
    ("pdc players championship", "PDC_PC"),
    ("challenge tour", "PDC_CHALLENGE"),
    ("development tour", "PDC_DEVTOUR"),
    ("women", "PDC_WOM_SERIES"),
    ("womens", "PDC_WOM_SERIES"),
    ("masters", "PDC_MASTERS"),
    ("grand slam", "PDC_GS"),
    ("uk open", "PDC_UK"),
    ("world cup", "PDC_WCUP"),
    ("european", "PDC_ET"),
    ("dutch open", "PDC_ET"),
    ("scotland", "PDC_ET"),
    ("scottish", "PDC_ET"),
    ("poland", "PDC_ET"),
    ("bahrain", "PDC_MASTERS"),
    ("saudi", "PDC_MASTERS"),
]


def _event_code(title: str) -> str:
    lower = title.lower()
    for fragment, code in TITLE_TO_CODE:
        if fragment in lower:
            return code
    return "PDC_PC"  # default bucket


def _parse_date_from_title(title: str) -> Optional[date]:
    """Extract first date from DartsDatabase event title."""
    # Patterns like '04/01/2020' or '15/01/2026'
    m = re.search(r"(\d{2})/(\d{2})/(\d{4})", title)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    # Patterns like 'Week N DD/MM/YYYY'
    m = re.search(r"(\d{1,2})/(\d{2})/(\d{4})", title)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def run(db_url: str, dry_run: bool = False) -> None:
    event_files = sorted(EVENT_DIR.glob("*.json"))
    if not event_files:
        print(f"No event files found in {EVENT_DIR}")
        sys.exit(1)

    print(f"Event files: {len(event_files)}")

    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # ------------------------------------------------------------------
    # Build player lookup maps
    # ------------------------------------------------------------------
    print("Loading player lookup tables from DB...")
    cur.execute(
        """
        SELECT id, first_name, last_name, nickname, dartsdatabase_id
        FROM darts_players
        """
    )
    players_all = cur.fetchall()

    # Map: normalised full_name → player_id
    name_to_id: dict[str, str] = {}
    # Map: dartsdatabase_id → player_id
    dd_id_to_player: dict[str, str] = {}

    for row in players_all:
        full = f"{row['first_name']} {row['last_name']}".strip()
        norm = _normalise(full)
        name_to_id[norm] = row["id"]
        if row["nickname"]:
            name_to_id[_normalise(row["nickname"])] = row["id"]
        if row["dartsdatabase_id"]:
            dd_id_to_player[row["dartsdatabase_id"]] = row["id"]

    print(f"  Players loaded: {len(players_all)} ({len(dd_id_to_player)} with DartsDB IDs)")

    # ------------------------------------------------------------------
    # Process each event file
    # ------------------------------------------------------------------
    stats = {
        "events": 0,
        "competitions_upserted": 0,
        "matches_inserted": 0,
        "matches_skipped_dup": 0,
        "players_linked": 0,
        "players_created": 0,
        "player_stats_upserted": 0,
        "matches_unlinked": 0,
    }

    for ef in event_files:
        data = json.loads(ef.read_text(encoding="utf-8"))
        matches = data.get("matches", [])
        if not matches:
            continue

        stats["events"] += 1
        event_title = data.get("event_title", "Unknown")
        event_url = data.get("event_url", "")
        event_id_str = data.get("event_id", ef.stem)

        comp_code = _event_code(event_title)
        comp_id = f"dartsdatabase_{event_id_str}"
        event_date = _parse_date_from_title(event_title)
        season_year = event_date.year if event_date else 2026
        ecosystem = (
            "WDF" if comp_code.startswith("WDF") or "BDO" in event_title.upper()
            else "PDC"
        )

        if not dry_run:
            cur.execute(
                """
                INSERT INTO darts_competitions
                  (id, name, season_year, format_code, organiser, ecosystem,
                   start_date, is_ranked, is_televised, metadata_json, created_at)
                VALUES (%s, %s, %s, %s, 'dartsdatabase', %s, %s,
                        false, false, %s::jsonb, NOW())
                ON CONFLICT (id) DO UPDATE
                  SET name = EXCLUDED.name,
                      season_year = EXCLUDED.season_year
                """,
                (
                    comp_id,
                    event_title.split("  ")[0].strip(),
                    season_year,
                    comp_code,
                    ecosystem,
                    event_date,
                    json.dumps({"source": "dartsdatabase", "event_url": event_url}),
                ),
            )
        stats["competitions_upserted"] += 1

        for match in matches:
            p1_name: str = match.get("player1_name", "")
            p2_name: str = match.get("player2_name", "")
            p1_score: Optional[int] = match.get("score_p1")
            p2_score: Optional[int] = match.get("score_p2")
            p1_avg: Optional[float] = match.get("player1_avg")
            p2_avg: Optional[float] = match.get("player2_avg")
            p1_dd_id: Optional[str] = match.get("player1_id")
            p2_dd_id: Optional[str] = match.get("player2_id")
            round_name: str = match.get("round", "Unknown")

            if not p1_name or not p2_name:
                continue
            if p1_score is None or p2_score is None:
                continue

            p1_id = _resolve_player(
                cur, p1_name, p1_dd_id, name_to_id, dd_id_to_player, stats, dry_run
            )
            p2_id = _resolve_player(
                cur, p2_name, p2_dd_id, name_to_id, dd_id_to_player, stats, dry_run
            )

            if p1_id is None or p2_id is None:
                stats["matches_unlinked"] += 1
                continue

            winner_id = (
                p1_id if p1_score > p2_score
                else p2_id if p2_score > p1_score
                else None
            )

            match_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"dartsdatabase:{event_id_str}:{round_name}:{p1_dd_id}:{p2_dd_id}",
            ))

            if not dry_run:
                cur.execute(
                    """
                    INSERT INTO darts_matches
                      (id, competition_id, player1_id, player2_id,
                       round_name, match_date, status,
                       player1_score, player2_score,
                       result_type, winner_player_id,
                       coverage_regime, source_name, raw_source_data,
                       created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, 'result', %s, %s,
                            'legs', %s, 'R0', 'dartsdatabase', %s,
                            NOW(), NOW())
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        match_id, comp_id, p1_id, p2_id,
                        round_name, event_date,
                        p1_score, p2_score, winner_id,
                        json.dumps({
                            "player1_avg": p1_avg,
                            "player2_avg": p2_avg,
                            "event_id": event_id_str,
                            "event_title": event_title,
                            "dartsdatabase_p1_id": p1_dd_id,
                            "dartsdatabase_p2_id": p2_dd_id,
                        }),
                    ),
                )
                if cur.rowcount == 0:
                    stats["matches_skipped_dup"] += 1
                else:
                    stats["matches_inserted"] += 1
            else:
                stats["matches_inserted"] += 1

            if p1_avg is not None:
                if not dry_run:
                    _upsert_player_stat(cur, p1_id, comp_id, season_year, p1_avg, event_date)
                stats["player_stats_upserted"] += 1
            if p2_avg is not None:
                if not dry_run:
                    _upsert_player_stat(cur, p2_id, comp_id, season_year, p2_avg, event_date)
                stats["player_stats_upserted"] += 1

        if not dry_run:
            conn.commit()

    cur.close()
    conn.close()

    print("\n=== INGEST SUMMARY ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def _resolve_player(
    cur,
    full_name: str,
    dd_id: Optional[str],
    name_to_id: dict[str, str],
    dd_id_to_player: dict[str, str],
    stats: dict,
    dry_run: bool,
) -> Optional[str]:
    """Resolve player to internal UUID. Creates a stub if not found."""
    # Try DartsDatabase ID first (most reliable)
    if dd_id and dd_id in dd_id_to_player:
        return dd_id_to_player[dd_id]

    # Try name normalisation
    norm = _normalise(full_name)
    if norm in name_to_id:
        player_id = name_to_id[norm]
        # Back-populate dartsdatabase_id if we have it
        if dd_id:
            if not dry_run:
                cur.execute(
                    "UPDATE darts_players SET dartsdatabase_id = %s WHERE id = %s AND dartsdatabase_id IS NULL",
                    (dd_id, player_id),
                )
                if cur.rowcount > 0:
                    dd_id_to_player[dd_id] = player_id
        stats["players_linked"] += 1
        return player_id

    # Try last-name only for common single-token matches
    parts = full_name.strip().split()
    if len(parts) >= 2:
        # Try reversed order (some sources put last name first)
        reversed_norm = _normalise(f"{parts[-1]} {' '.join(parts[:-1])}")
        if reversed_norm in name_to_id:
            player_id = name_to_id[reversed_norm]
            stats["players_linked"] += 1
            return player_id

    # Create stub player
    first_name, last_name = _split_name(full_name)
    slug = re.sub(r"[^a-z0-9-]", "-", _normalise(full_name).replace(" ", "-"))
    # Make slug unique with dd_id suffix
    if dd_id:
        slug = f"{slug}-dd{dd_id}"

    player_id = str(uuid.uuid4())

    if not dry_run:
        try:
            cur.execute(
                """
                INSERT INTO darts_players
                  (id, first_name, last_name, slug, dartsdatabase_id,
                   source_confidence, primary_source, gdpr_anonymized)
                VALUES (%s, %s, %s, %s, %s, 0.6, 'dartsdatabase', false)
                ON CONFLICT (slug) DO NOTHING
                """,
                (player_id, first_name, last_name, slug, dd_id),
            )
            if cur.rowcount > 0:
                name_to_id[_normalise(full_name)] = player_id
                if dd_id:
                    dd_id_to_player[dd_id] = player_id
                stats["players_created"] += 1
            else:
                # Slug collision — try to retrieve existing
                cur.execute(
                    "SELECT id FROM darts_players WHERE slug = %s", (slug,)
                )
                row = cur.fetchone()
                if row:
                    player_id = row[0]
                    name_to_id[_normalise(full_name)] = player_id
                    return player_id
                return None
        except Exception as e:
            print(f"  WARNING: failed to create player '{full_name}': {e}")
            return None

    stats["players_created"] += 1
    return player_id


def _upsert_player_stat(
    cur,
    player_id: str,
    comp_id: str,
    season_year: int,
    avg: float,
    stat_date: Optional[date],
) -> None:
    """Upsert a player stat row with the 3-dart average from a match."""
    stat_id = str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"dartsdatabase_stat:{player_id}:{comp_id}",
    ))
    cur.execute(
        """
        INSERT INTO darts_player_stats
          (id, player_id, competition_id, source, stat_season_year,
           three_dart_average, stat_date, created_at, updated_at)
        VALUES (%s, %s, %s, 'dartsdatabase', %s, %s, %s, NOW(), NOW())
        ON CONFLICT (id) DO UPDATE
          SET three_dart_average = EXCLUDED.three_dart_average,
              updated_at = NOW()
        """,
        (stat_id, player_id, comp_id, season_year, avg, stat_date),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest DartsDatabase event data into DB")
    parser.add_argument("--db-url", default=DEFAULT_DB_URL, help="PostgreSQL connection URL")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report without writing")
    args = parser.parse_args()

    print(f"DartsDatabase ingest — {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Event dir: {EVENT_DIR}")
    print()

    run(args.db_url, dry_run=args.dry_run)
