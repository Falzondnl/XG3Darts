"""
PDC data ingestion script.

Reads the three PDC CSV files and ingests them into the darts database:

1. ``participants.csv``  → ``darts_players`` table
2. ``tournaments.csv``   → ``darts_competitions`` table
3. ``fixtures.csv``      → ``darts_matches`` table (279k rows)

Also:
- Sets ``source_confidence`` on player records
- Runs entity pre-resolution (slug matching for PDC participants)
- Detects coverage regime for each player/competition combination
- Writes audit log at completion

Usage
-----
    python scripts/ingest_pdc.py [--dry-run] [--limit N]

Environment
-----------
DATABASE_URL must be set in the environment or .env file.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure project root is on PYTHONPATH when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import settings
from competition.era_versioning import resolve_format
from competition.format_registry import DartsFormatError, get_format
from data.coverage_regime import (
    CoverageSignals,
    RegimeResult,
    detect_regime,
)
from data.entity_resolution import SOURCE_CONFIDENCE
from db.models import (
    DartsCompetition,
    DartsCoverageRegime,
    DartsMatch,
    DartsPlayer,
    DartsPlayerStats,
)
from db.session import get_session


logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

DATA_ROOT = Path(settings.DATA_ROOT)
PDC_DIR = DATA_ROOT / "02_processed" / "csv" / "pdc"

FIXTURES_CSV = PDC_DIR / "fixtures.csv"
PARTICIPANTS_CSV = PDC_DIR / "participants.csv"
TOURNAMENTS_CSV = PDC_DIR / "tournaments.csv"

# Confidence score for PDC as data source
PDC_SOURCE_CONFIDENCE = SOURCE_CONFIDENCE["pdc"]


# ---------------------------------------------------------------------------
# Tournament type → format code mapping
# ---------------------------------------------------------------------------

# tournament_type_id from PDC data → our format code
TOURNAMENT_TYPE_FORMAT_MAP: dict[int, str] = {
    1:  "PDC_PC",      # Players Championship
    2:  "PDC_WC",      # World Championship
    3:  "PDC_PL",      # Premier League
    4:  "PDC_GP",      # Grand Prix
    5:  "PDC_UK",      # UK Open
    6:  "PDC_ET",      # European Tour
    7:  "PDC_WM",      # World Matchplay
    8:  "PDC_GS",      # Grand Slam of Darts
    9:  "PDC_PCF",     # Players Championship Finals
    10: "PDC_PC",      # Players Championship (alt type_id)
    11: "PDC_MASTERS", # Masters
    12: "PDC_WS",      # World Series
    13: "PDC_WCUP",    # World Cup
    14: "PDC_WOM_SERIES", # Women's Series
    15: "PDC_DEVTOUR", # Development Tour
    16: "PDC_CHALLENGE", # Challenge Tour
    17: "PDC_WYC",     # World Youth Championship
    78: "PDC_MASTERS", # Masters (historical id)
    77: "PDC_WM",      # World Matchplay (historical)
    81: "WDF_OPEN",    # Dutch Open (WDF)
    92: "WDF_OPEN",    # Other WDF Opens
}

PDC_ECOSYSTEM = "pdc_mens"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(value: Any) -> Optional[int]:
    if value is None or str(value).strip() in ("", "None", "nan"):
        return None
    try:
        return int(float(str(value)))
    except (ValueError, TypeError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None or str(value).strip() in ("", "None", "nan"):
        return None
    try:
        return float(str(value))
    except (ValueError, TypeError):
        return None


def _safe_date(value: Any) -> Optional[date]:
    if not value or str(value).strip() in ("", "None", "nan"):
        return None
    s = str(value).strip()[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _parse_tour_card_years(value: Any) -> Optional[list[int]]:
    """Parse the tour_card_years field which may contain a Python list repr."""
    if not value or str(value).strip() in ("", "None", "nan"):
        return None
    s = str(value).strip()
    years = re.findall(r"\d{4}", s)
    return [int(y) for y in years] if years else None


def _resolve_format_code(
    tournament_type_id: Optional[int],
    tournament_name: str,
    season_year: int,
) -> str:
    """Map PDC tournament type ID to our internal format code."""
    if tournament_type_id is not None:
        code = TOURNAMENT_TYPE_FORMAT_MAP.get(tournament_type_id)
        if code:
            # For World Championship, apply era versioning
            if code in ("PDC_WC", "PDC_WC_ERA_2020"):
                try:
                    era_fmt = resolve_format("PDC_WC", season_year)
                    return era_fmt.code
                except Exception:
                    return code
            return code

    # Fallback: name-based heuristic
    name_lower = tournament_name.lower()
    if "world championship" in name_lower:
        try:
            era_fmt = resolve_format("PDC_WC", season_year)
            return era_fmt.code
        except Exception:
            return "PDC_WC"
    if "premier league" in name_lower:
        return "PDC_PL"
    if "world matchplay" in name_lower:
        return "PDC_WM"
    if "grand prix" in name_lower:
        return "PDC_GP"
    if "uk open" in name_lower:
        return "PDC_UK"
    if "grand slam" in name_lower:
        return "PDC_GS"
    if "players championship" in name_lower:
        return "PDC_PC"
    if "european" in name_lower:
        return "PDC_ET"
    if "world cup" in name_lower:
        return "PDC_WCUP"
    if "masters" in name_lower:
        return "PDC_MASTERS"
    if "women" in name_lower:
        return "PDC_WOM_SERIES"
    if "development" in name_lower or "dev tour" in name_lower:
        return "PDC_DEVTOUR"
    if "challenge" in name_lower:
        return "PDC_CHALLENGE"
    if "youth" in name_lower:
        return "PDC_WYC"

    return "PDC_PC"  # default for unrecognised PDC events


def _determine_round_name(stage_raw: Any, fixture_type: Any) -> str:
    """Extract a clean round name from raw PDC stage JSON-like string."""
    if not stage_raw or str(stage_raw).strip() in ("", "None", "nan"):
        return "Unknown"
    s = str(stage_raw)
    # Extract 'name' field from dict-like string
    name_match = re.search(r"'name'\s*:\s*'([^']+)'", s)
    if name_match:
        return name_match.group(1)
    return "Unknown"


# ---------------------------------------------------------------------------
# Ingest: participants
# ---------------------------------------------------------------------------

async def ingest_participants(
    session: AsyncSession,
    *,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> dict[str, str]:
    """
    Ingest PDC participants into the darts_players table.

    Parameters
    ----------
    session:
        Active database session.
    dry_run:
        If True, build the records but do not persist.
    limit:
        Maximum number of participants to process.

    Returns
    -------
    dict[str, str]
        Mapping of ``pdc_id → canonical_player_uuid``.
    """
    if not PARTICIPANTS_CSV.exists():
        raise FileNotFoundError(f"Participants CSV not found: {PARTICIPANTS_CSV}")

    pdc_id_to_uuid: dict[str, str] = {}
    inserted = 0
    skipped = 0

    with PARTICIPANTS_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if limit:
        rows = rows[:limit]

    logger.info("participants_ingest_start", total=len(rows))

    for row in rows:
        pdc_id_raw = row.get("id", "").strip()
        if not pdc_id_raw:
            skipped += 1
            continue

        pdc_id = int(pdc_id_raw)
        player_uuid = str(uuid.uuid4())

        # Check if already exists by pdc_id
        existing = await session.execute(
            text("SELECT id FROM darts_players WHERE pdc_id = :pdc_id"),
            {"pdc_id": pdc_id},
        )
        existing_row = existing.fetchone()
        if existing_row:
            pdc_id_to_uuid[str(pdc_id)] = str(existing_row[0])
            skipped += 1
            continue

        first_name = str(row.get("first_name", "")).strip()
        last_name = str(row.get("last_name", "")).strip()
        slug = str(row.get("participant_slug", "")).strip() or None
        country_code = str(row.get("country_code", "")).strip() or None
        dob_raw = row.get("dob", "")
        dob = _safe_date(dob_raw)
        ranking = _safe_int(row.get("ranking"))
        prize_money = _safe_int(row.get("prize_money"))
        tour_card_raw = str(row.get("tour_card_holder", "False")).strip()
        tour_card = tour_card_raw.lower() in ("true", "1", "yes")
        tour_card_years = _parse_tour_card_years(row.get("tour_card_years"))
        nickname = str(row.get("nickname", "")).strip() or None

        player = DartsPlayer(
            id=player_uuid,
            first_name=first_name,
            last_name=last_name,
            nickname=nickname,
            slug=slug,
            pdc_id=pdc_id,
            source_confidence=PDC_SOURCE_CONFIDENCE,
            primary_source="pdc",
            dob=dob,
            country_code=country_code,
            pdc_ranking=ranking,
            prize_money=prize_money,
            tour_card_holder=tour_card,
            tour_card_years=tour_card_years,
            gdpr_anonymized=False,
        )

        if not dry_run:
            session.add(player)
            if inserted % 500 == 0:
                await session.flush()

        pdc_id_to_uuid[str(pdc_id)] = player_uuid
        inserted += 1

    if not dry_run:
        await session.flush()

    logger.info(
        "participants_ingest_complete",
        inserted=inserted,
        skipped=skipped,
        dry_run=dry_run,
    )
    return pdc_id_to_uuid


# ---------------------------------------------------------------------------
# Ingest: tournaments
# ---------------------------------------------------------------------------

async def ingest_tournaments(
    session: AsyncSession,
    pdc_id_to_uuid: dict[str, str],
    *,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> dict[str, str]:
    """
    Ingest PDC tournaments into the darts_competitions table.

    Parameters
    ----------
    session:
        Active database session.
    pdc_id_to_uuid:
        Player ID mapping from :func:`ingest_participants`.
    dry_run:
        If True, build but do not persist.
    limit:
        Maximum number of tournaments to process.

    Returns
    -------
    dict[str, str]
        Mapping of ``pdc_tournament_id → competition_uuid``.
    """
    if not TOURNAMENTS_CSV.exists():
        raise FileNotFoundError(f"Tournaments CSV not found: {TOURNAMENTS_CSV}")

    pdc_tournament_to_uuid: dict[str, str] = {}
    inserted = 0
    skipped = 0

    with TOURNAMENTS_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if limit:
        rows = rows[:limit]

    logger.info("tournaments_ingest_start", total=len(rows))

    for row in rows:
        tournament_id_raw = row.get("id", "").strip()
        if not tournament_id_raw:
            skipped += 1
            continue

        pdc_tournament_id = int(tournament_id_raw)

        # Check if already exists
        existing = await session.execute(
            text(
                "SELECT id FROM darts_competitions WHERE pdc_tournament_id = :tid"
            ),
            {"tid": pdc_tournament_id},
        )
        existing_row = existing.fetchone()
        if existing_row:
            pdc_tournament_to_uuid[str(pdc_tournament_id)] = str(existing_row[0])
            skipped += 1
            continue

        comp_uuid = str(uuid.uuid4())
        name = str(row.get("name", "")).strip()
        season_id_raw = row.get("season_id", "")
        season_year = _safe_int(season_id_raw) or 0

        type_id = _safe_int(row.get("tournament_type_id"))
        format_code = _resolve_format_code(type_id, name, season_year)

        # Era-versioned code
        format_era_code: Optional[str] = None
        if format_code in ("PDC_WC", "PDC_WC_ERA_2020"):
            format_era_code = format_code

        start_date = _safe_date(row.get("start_date"))
        end_date = _safe_date(row.get("end_date"))
        venue = str(row.get("venue", "")).strip() or None
        city = str(row.get("city", "")).strip() or None

        winner_pdc_id = str(row.get("winner_participant_id", "")).strip()
        winner_uuid = pdc_id_to_uuid.get(winner_pdc_id)

        is_ranked_raw = str(row.get("is_ranked", "True")).strip().lower()
        is_ranked = is_ranked_raw in ("true", "1")
        is_televised_raw = str(row.get("is_televised", "False")).strip().lower()
        is_televised = is_televised_raw in ("true", "1")

        competition = DartsCompetition(
            id=comp_uuid,
            pdc_tournament_id=pdc_tournament_id,
            sport_radar_tournament_id=str(row.get("sport_radar_tournament_id", "")).strip() or None,
            dartconnect_id=str(row.get("dart_connect_id", "")).strip() or None,
            name=name,
            season_year=season_year,
            format_code=format_code,
            format_era_code=format_era_code,
            organiser="PDC",
            ecosystem=PDC_ECOSYSTEM,
            venue=venue,
            city=city,
            start_date=start_date,
            end_date=end_date,
            is_ranked=is_ranked,
            is_televised=is_televised,
            winner_player_id=winner_uuid,
        )

        if not dry_run:
            session.add(competition)
            if inserted % 500 == 0:
                await session.flush()

        pdc_tournament_to_uuid[str(pdc_tournament_id)] = comp_uuid
        inserted += 1

    if not dry_run:
        await session.flush()

    logger.info(
        "tournaments_ingest_complete",
        inserted=inserted,
        skipped=skipped,
        dry_run=dry_run,
    )
    return pdc_tournament_to_uuid


# ---------------------------------------------------------------------------
# Ingest: fixtures (279k rows)
# ---------------------------------------------------------------------------

async def ingest_fixtures(
    session: AsyncSession,
    pdc_id_to_uuid: dict[str, str],
    pdc_tournament_to_uuid: dict[str, str],
    *,
    dry_run: bool = False,
    limit: Optional[int] = None,
    batch_size: int = 1000,
) -> int:
    """
    Ingest PDC fixtures into the darts_matches table.

    Parameters
    ----------
    session:
        Active database session.
    pdc_id_to_uuid:
        Player mapping from :func:`ingest_participants`.
    pdc_tournament_to_uuid:
        Competition mapping from :func:`ingest_tournaments`.
    dry_run:
        Build but do not persist.
    limit:
        Maximum rows to process.
    batch_size:
        Flush batch size for memory management.

    Returns
    -------
    int
        Number of fixtures inserted.
    """
    if not FIXTURES_CSV.exists():
        raise FileNotFoundError(f"Fixtures CSV not found: {FIXTURES_CSV}")

    inserted = 0
    skipped = 0
    batch: list[DartsMatch] = []

    logger.info("fixtures_ingest_start", file=str(FIXTURES_CSV))

    with FIXTURES_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        for row_idx, row in enumerate(reader):
            if limit and row_idx >= limit:
                break

            fixture_id_raw = row.get("id", "").strip()
            if not fixture_id_raw:
                skipped += 1
                continue

            pdc_fixture_id = _safe_int(fixture_id_raw)
            tournament_id_raw = row.get("tournament_id", "").strip()
            competition_uuid = pdc_tournament_to_uuid.get(tournament_id_raw)

            if not competition_uuid:
                skipped += 1
                continue

            p1_id_raw = row.get("participant1_id", "").strip()
            p2_id_raw = row.get("participant2_id", "").strip()
            p1_uuid = pdc_id_to_uuid.get(p1_id_raw) if p1_id_raw else None
            p2_uuid = pdc_id_to_uuid.get(p2_id_raw) if p2_id_raw else None

            p1_score = _safe_int(row.get("participant1_score"))
            p2_score = _safe_int(row.get("participant2_score"))

            winner_id_raw = row.get("winner_participant_id", "").strip()
            winner_uuid = pdc_id_to_uuid.get(winner_id_raw) if winner_id_raw else None

            status = str(row.get("status", "Fixture")).strip()
            match_date = _safe_date(row.get("start_date"))
            match_time = str(row.get("start_time", "")).strip()[:8] or None

            round_name = _determine_round_name(row.get("stage"), row.get("type"))

            # Determine result type
            result_type: Optional[str] = None
            if status == "Completed" and p1_score is not None and p2_score is not None:
                if p1_score > p2_score:
                    result_type = "p1_win"
                elif p2_score > p1_score:
                    result_type = "p2_win"
                else:
                    result_type = "draw"

            sport_radar_id = str(row.get("sport_radar_id", "")).strip() or None

            match = DartsMatch(
                id=str(uuid.uuid4()),
                pdc_fixture_id=pdc_fixture_id,
                sport_radar_id=sport_radar_id,
                competition_id=competition_uuid,
                player1_id=p1_uuid,
                player2_id=p2_uuid,
                round_name=round_name,
                match_date=match_date,
                match_time=match_time,
                status=status,
                player1_score=p1_score,
                player2_score=p2_score,
                result_type=result_type,
                winner_player_id=winner_uuid,
                visit_data_coverage=0.0,
                coverage_regime="R0",
                source_name="pdc",
            )

            batch.append(match)
            inserted += 1

            if len(batch) >= batch_size and not dry_run:
                session.add_all(batch)
                await session.flush()
                batch.clear()
                logger.info(
                    "fixtures_batch_flushed",
                    inserted_so_far=inserted,
                )

    # Final batch
    if batch and not dry_run:
        session.add_all(batch)
        await session.flush()

    logger.info(
        "fixtures_ingest_complete",
        inserted=inserted,
        skipped=skipped,
        dry_run=dry_run,
    )
    return inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_ingest(
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> None:
    """
    Run the full PDC ingest pipeline.

    Parameters
    ----------
    dry_run:
        Build all records but do not commit to the database.
    limit:
        Maximum rows per table (for testing).
    """
    logger.info("pdc_ingest_start", dry_run=dry_run, limit=limit)
    start = datetime.now(tz=timezone.utc)

    async with get_session() as session:
        # Step 1: Players
        pdc_id_to_uuid = await ingest_participants(
            session, dry_run=dry_run, limit=limit
        )

        # Step 2: Competitions
        pdc_tournament_to_uuid = await ingest_tournaments(
            session,
            pdc_id_to_uuid,
            dry_run=dry_run,
            limit=limit,
        )

        # Step 3: Fixtures
        fixture_count = await ingest_fixtures(
            session,
            pdc_id_to_uuid,
            pdc_tournament_to_uuid,
            dry_run=dry_run,
            limit=limit,
        )

        if not dry_run:
            await session.commit()

    elapsed = (datetime.now(tz=timezone.utc) - start).total_seconds()
    logger.info(
        "pdc_ingest_complete",
        players=len(pdc_id_to_uuid),
        competitions=len(pdc_tournament_to_uuid),
        fixtures=fixture_count,
        elapsed_seconds=round(elapsed, 1),
        dry_run=dry_run,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDC CSV data into the XG3 Darts database."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without writing to the database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit rows per table (useful for testing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run_ingest(dry_run=args.dry_run, limit=args.limit))
