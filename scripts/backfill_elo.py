"""
ELO rating backfill script.

Reads all completed matches from the database (ordered chronologically)
and runs the full ELO pipeline to compute historical ratings.

Also supports CSV fallback: reads from
  D:/codex/Data/Darts/02_processed/csv/pdc/fixtures.csv (279k rows)

The CSV file has columns:
  fixture_id, date, competition, round_name, player1_name, player2_name,
  player1_score, player2_score, result_type, ecosystem

The script processes matches chronologically across all ELO pools.

Usage
-----
    python scripts/backfill_elo.py [--pool POOL] [--dry-run] [--from-year YEAR]
    python scripts/backfill_elo.py --csv [--pool POOL] [--dry-run]

Options
-------
    --pool        Filter to a specific ELO pool
    --dry-run     Compute ratings but do not write to database
    --from-year   Start year (default: 2003)
    --csv         Read from CSV fixture file instead of database
    --batch-size  Database write batch size (default: 5000)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import structlog
from sqlalchemy import select, text


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from db.models import DartsEloRating, DartsMatch
from db.session import get_session
from elo.elo_pipeline import (
    EloMatchInput,
    EloPipeline,
    VALID_POOLS,
)


logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CSV_FIXTURE_PATH = Path("D:/codex/Data/Darts/02_processed/csv/pdc/fixtures.csv")

ECOSYSTEM_TO_POOL: dict[str, str] = {
    "pdc_mens":      "pdc_mens",
    "pdc_womens":    "pdc_womens",
    "wdf_open":      "wdf_open",
    "development":   "development",
    "team_doubles":  "team_doubles",
}

# Default ecosystem to pool mapping for CSV records without explicit ecosystem
COMPETITION_NAME_TO_ECOSYSTEM: dict[str, str] = {
    "world championship": "pdc_mens",
    "premier league": "pdc_mens",
    "world matchplay": "pdc_mens",
    "grand prix": "pdc_mens",
    "grand slam": "pdc_mens",
    "uk open": "pdc_mens",
    "masters": "pdc_mens",
    "players championship": "pdc_mens",
    "european tour": "pdc_mens",
    "world cup": "team_doubles",
    "women": "pdc_womens",
    "development": "development",
    "challenge": "development",
    "wdf": "wdf_open",
}


def _infer_ecosystem(competition_name: str) -> str:
    """Infer ecosystem from competition name for CSV records."""
    name_lower = competition_name.lower()
    for keyword, ecosystem in COMPETITION_NAME_TO_ECOSYSTEM.items():
        if keyword in name_lower:
            return ecosystem
    return "pdc_mens"


# ---------------------------------------------------------------------------
# Progress bar (tqdm or fallback)
# ---------------------------------------------------------------------------


def _make_progress(total: int, desc: str = ""):
    """Create a progress iterator, falling back to plain output if tqdm unavailable."""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="matches")
    except ImportError:
        class SimpleProgress:
            def __init__(self, total: int, desc: str) -> None:
                self.total = total
                self.n = 0
                self.desc = desc
                logger.info("backfill_progress_start", desc=desc, total=total)

            def update(self, n: int = 1) -> None:
                self.n += n
                if self.n % 5000 == 0 or self.n == self.total:
                    pct = 100.0 * self.n / max(self.total, 1)
                    logger.info(
                        "backfill_progress",
                        processed=self.n,
                        total=self.total,
                        pct=round(pct, 1),
                    )

            def close(self) -> None:
                logger.info("backfill_progress_complete", total=self.n)

        return SimpleProgress(total, desc)


# ---------------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------------


async def load_completed_matches(
    session,
    pool_filter: Optional[str] = None,
    from_year: Optional[int] = None,
) -> list[dict]:
    """
    Load all completed matches from the database, ordered chronologically.

    Parameters
    ----------
    session:
        Active async database session.
    pool_filter:
        Optional ELO pool filter.
    from_year:
        Only process matches from this year onwards.

    Returns
    -------
    list[dict]
        Match records with player IDs, format, result, date, and ecosystem.
    """
    query = """
        SELECT
            m.id,
            m.player1_id,
            m.player2_id,
            m.result_type,
            m.round_name,
            m.match_date,
            c.format_code,
            c.ecosystem,
            p1.pdc_ranking AS p1_rank,
            p2.pdc_ranking AS p2_rank
        FROM darts_matches m
        JOIN darts_competitions c ON c.id = m.competition_id
        LEFT JOIN darts_players p1 ON p1.id = m.player1_id
        LEFT JOIN darts_players p2 ON p2.id = m.player2_id
        WHERE m.status = 'Completed'
          AND m.result_type IS NOT NULL
          AND m.player1_id IS NOT NULL
          AND m.player2_id IS NOT NULL
          AND m.match_date IS NOT NULL
    """

    params: dict = {}
    if from_year:
        query += " AND EXTRACT(YEAR FROM m.match_date) >= :from_year"
        params["from_year"] = from_year
    if pool_filter:
        query += " AND c.ecosystem = :ecosystem"
        params["ecosystem"] = pool_filter

    query += " ORDER BY m.match_date ASC, m.id ASC"

    result = await session.execute(text(query), params)
    rows = result.fetchall()

    logger.info("completed_matches_loaded", count=len(rows))

    return [
        {
            "match_id": str(row[0]),
            "player1_id": str(row[1]),
            "player2_id": str(row[2]),
            "result_type": row[3],
            "round_name": row[4],
            "match_date": row[5],
            "format_code": row[6],
            "ecosystem": row[7],
            "p1_rank": row[8],
            "p2_rank": row[9],
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_matches_from_csv(
    csv_path: Path = CSV_FIXTURE_PATH,
    pool_filter: Optional[str] = None,
    from_year: Optional[int] = None,
) -> list[dict]:
    """
    Load completed matches from the CSV fixture file.

    The CSV is expected to have columns (order may vary):
    fixture_id, date, competition, round_name, player1_name, player2_name,
    player1_score, player2_score, result_type, ecosystem

    Parameters
    ----------
    csv_path:
        Path to the fixtures CSV file.
    pool_filter:
        Optional ELO pool filter.
    from_year:
        Only include matches from this year onwards.

    Returns
    -------
    list[dict]
        Sorted chronologically by date.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV fixture file not found: {csv_path}")

    logger.info("csv_load_start", path=str(csv_path))

    matches: list[dict] = []

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        # Handle BOM
        content = fh.read()
        if content.startswith("\ufeff"):
            content = content[1:]

    import io
    reader = csv.DictReader(io.StringIO(content))

    # Normalise column names (lowercase, strip whitespace)
    for raw_row in reader:
        row = {k.strip().lower(): v.strip() for k, v in raw_row.items()}

        # Map column aliases
        fixture_id = row.get("fixture_id") or row.get("id") or row.get("match_id", "")
        date_str = (
            row.get("date")
            or row.get("match_date")
            or row.get("event_date")
            or ""
        )
        competition = row.get("competition") or row.get("tournament") or ""
        round_name = row.get("round_name") or row.get("round") or "Unknown"
        p1_name = (
            row.get("player1_name")
            or row.get("player1")
            or row.get("home_player")
            or ""
        )
        p2_name = (
            row.get("player2_name")
            or row.get("player2")
            or row.get("away_player")
            or ""
        )
        result_type = (
            row.get("result_type")
            or row.get("result")
            or row.get("outcome")
            or ""
        )
        ecosystem = row.get("ecosystem") or _infer_ecosystem(competition)
        format_code = row.get("format_code") or "PDC_PC"

        # Parse date
        match_date = None
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%m/%d/%Y"):
            try:
                match_date = datetime.strptime(date_str, fmt).date()
                break
            except (ValueError, TypeError):
                pass

        if match_date is None:
            continue

        if from_year and match_date.year < from_year:
            continue

        # Filter by pool
        pool = ECOSYSTEM_TO_POOL.get(ecosystem, "pdc_mens")
        if pool_filter and pool != pool_filter:
            continue

        # Normalise result_type
        if result_type.lower() in ("p1_win", "1", "home", "player1"):
            result_type = "p1_win"
        elif result_type.lower() in ("p2_win", "2", "away", "player2"):
            result_type = "p2_win"
        elif result_type.lower() in ("draw", "d", "tie"):
            result_type = "draw"
        else:
            continue  # skip invalid result types

        # Use player names as IDs for CSV data (entity resolution done separately)
        p1_id = row.get("player1_id") or f"name:{p1_name}"
        p2_id = row.get("player2_id") or f"name:{p2_name}"

        matches.append({
            "match_id": fixture_id,
            "player1_id": p1_id,
            "player2_id": p2_id,
            "result_type": result_type,
            "round_name": round_name,
            "match_date": match_date,
            "format_code": format_code,
            "ecosystem": ecosystem,
            "p1_rank": None,
            "p2_rank": None,
        })

    # Sort chronologically
    matches.sort(key=lambda m: (m["match_date"] or date.min, m["match_id"]))

    logger.info(
        "csv_load_complete",
        path=str(csv_path),
        total_rows=len(matches),
        pool_filter=pool_filter,
        from_year=from_year,
    )
    return matches


# ---------------------------------------------------------------------------
# Core backfill pipeline
# ---------------------------------------------------------------------------


def _build_match_inputs(
    matches: list[dict],
    pool_filter: Optional[str],
) -> list[EloMatchInput]:
    """Convert raw match dicts to EloMatchInput objects."""
    inputs: list[EloMatchInput] = []
    skip_count = 0

    for m in matches:
        ecosystem = m.get("ecosystem", "pdc_mens")
        pool = ECOSYSTEM_TO_POOL.get(ecosystem, "pdc_mens")

        if pool_filter and pool != pool_filter:
            skip_count += 1
            continue

        try:
            match_input = EloMatchInput(
                player1_id=m["player1_id"],
                player2_id=m["player2_id"],
                result_type=m["result_type"],
                pool=pool,
                format_code=m.get("format_code", "PDC_PC"),
                round_name=m.get("round_name") or "Unknown",
                match_date=m["match_date"],
                player1_rank=m.get("p1_rank"),
                player2_rank=m.get("p2_rank"),
                weight=1.0,
            )
            inputs.append(match_input)
        except Exception as exc:
            logger.warning(
                "match_input_skip",
                match_id=m.get("match_id", "?"),
                error=str(exc),
            )
            skip_count += 1

    if skip_count:
        logger.info("match_inputs_skipped", count=skip_count)

    logger.info("match_inputs_prepared", count=len(inputs))
    return inputs


async def _write_elo_results(
    elo_results,
    match_inputs: list[EloMatchInput],
    matches: list[dict],
    batch_size: int = 5000,
) -> None:
    """Write ELO results to the database in batches."""
    match_id_by_idx = {i: matches[i].get("match_id") for i in range(len(matches))}
    progress = _make_progress(len(elo_results), desc="Writing ELO ratings")

    try:
        async with get_session() as session:
            rating_rows: list[DartsEloRating] = []

            for i, result in enumerate(elo_results):
                match_id = match_id_by_idx.get(i)
                pool_name = match_inputs[i].pool if i < len(match_inputs) else "pdc_mens"
                match_date = match_inputs[i].match_date if i < len(match_inputs) else None

                for (player_id, old_r, new_r, delta, k, expected, actual) in [
                    (
                        result.player1_id,
                        result.old_rating_p1,
                        result.new_rating_p1,
                        result.delta_p1,
                        result.k_p1,
                        result.expected_p1,
                        0.0 if result.result_type == "p2_win" else (
                            0.5 if result.result_type == "draw" else 1.0
                        ),
                    ),
                    (
                        result.player2_id,
                        result.old_rating_p2,
                        result.new_rating_p2,
                        result.delta_p2,
                        result.k_p2,
                        result.expected_p2,
                        0.0 if result.result_type == "p1_win" else (
                            0.5 if result.result_type == "draw" else 1.0
                        ),
                    ),
                ]:
                    rating_rows.append(
                        DartsEloRating(
                            player_id=player_id,
                            pool=pool_name,
                            match_id=match_id,
                            rating_before=old_r,
                            rating_after=new_r,
                            delta=delta,
                            k_factor=k,
                            expected_score=expected,
                            actual_score=actual,
                            games_played_at_time=0,
                            match_date=match_date or date.today(),
                        )
                    )

                if len(rating_rows) >= batch_size:
                    session.add_all(rating_rows)
                    await session.flush()
                    rating_rows.clear()

                progress.update(1)

            if rating_rows:
                session.add_all(rating_rows)
                await session.flush()

            await session.commit()

    finally:
        progress.close()


async def run_backfill(
    pool_filter: Optional[str] = None,
    dry_run: bool = False,
    from_year: Optional[int] = None,
    batch_size: int = 5000,
    use_csv: bool = False,
) -> None:
    """
    Run the ELO backfill pipeline.

    Parameters
    ----------
    pool_filter:
        Only process this ELO pool (None = all pools).
    dry_run:
        Compute ratings but do not write to the database.
    from_year:
        Start year for backfill (None = all historical data).
    batch_size:
        DB write batch size.
    use_csv:
        If True, read from CSV fixture file instead of database.
    """
    logger.info(
        "elo_backfill_start",
        pool_filter=pool_filter,
        dry_run=dry_run,
        from_year=from_year,
        use_csv=use_csv,
    )
    start = datetime.now(tz=timezone.utc)

    # Load matches
    if use_csv:
        matches = load_matches_from_csv(
            pool_filter=pool_filter,
            from_year=from_year,
        )
    else:
        async with get_session() as session:
            matches = await load_completed_matches(
                session, pool=pool_filter, from_year=from_year
            )

    if not matches:
        logger.warning("no_completed_matches_found")
        return

    logger.info("matches_to_process", total=len(matches))

    # Build EloMatchInput objects
    pipeline = EloPipeline()
    match_inputs = _build_match_inputs(matches, pool_filter)

    if not match_inputs:
        logger.warning("no_valid_match_inputs")
        return

    # Process with progress bar
    progress = _make_progress(len(match_inputs), desc="Computing ELO ratings")
    elo_results = []
    try:
        # Process in chunks to show progress
        chunk_size = 1000
        for chunk_start in range(0, len(match_inputs), chunk_size):
            chunk = match_inputs[chunk_start: chunk_start + chunk_size]
            chunk_results = pipeline.process_batch(chunk)
            elo_results.extend(chunk_results)
            progress.update(len(chunk))
    finally:
        progress.close()

    logger.info("elo_computation_complete", results=len(elo_results))

    if not dry_run:
        await _write_elo_results(elo_results, match_inputs, matches, batch_size)
    else:
        logger.info("dry_run_skipping_writes", results_computed=len(elo_results))

    elapsed = (datetime.now(tz=timezone.utc) - start).total_seconds()
    stats = pipeline.pool_stats()
    logger.info(
        "elo_backfill_complete",
        matches_processed=len(elo_results),
        pool_stats=stats,
        elapsed_seconds=round(elapsed, 1),
        dry_run=dry_run,
        source="csv" if use_csv else "database",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill ELO ratings from historical match data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pool",
        choices=sorted(VALID_POOLS),
        default=None,
        help="ELO pool to process (default: all pools)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute ratings but do not write to database",
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=None,
        help="Start year for backfill (default: all history from 2003)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Database write batch size (default: 5000)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        dest="use_csv",
        help=f"Read from CSV fixture file: {CSV_FIXTURE_PATH}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        run_backfill(
            pool_filter=args.pool,
            dry_run=args.dry_run,
            from_year=args.from_year,
            batch_size=args.batch_size,
            use_csv=args.use_csv,
        )
    )
