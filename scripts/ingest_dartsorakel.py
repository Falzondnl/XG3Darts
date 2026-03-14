"""
DartsOrakel data ingestion script.

Reads the DartsOrakel stats_player.json seed and per-player detail JSON files
(scraped by darts_orakel_scraper.py) and ingests into:
- darts_players (dartsorakel_key, dartsorakel_3da, dartsorakel_rank)
- darts_player_stats (three_dart_average, source="dartsorakel")

Usage
-----
    python scripts/ingest_dartsorakel.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
from sqlalchemy import text

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import settings
from data.entity_resolution import EntityResolver
from db.models import DartsPlayer, DartsPlayerStats
from db.session import get_session


logger = structlog.get_logger(__name__)

SEED_FILE = Path(settings.DATA_ROOT) / "01_raw" / "json" / "dartsorakel" / "stats_player.json"
DETAIL_DIR = Path(settings.DATA_ROOT) / "02_processed" / "json" / "dartsorakel" / "player_details"


async def ingest_dartsorakel_seed(
    session,
    *,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> dict[str, str]:
    """
    Ingest DartsOrakel seed data into darts_players.

    Returns
    -------
    dict[str, str]
        Mapping of ``dartsorakel_key → canonical_player_uuid``.
    """
    if not SEED_FILE.exists():
        raise FileNotFoundError(f"Seed file not found: {SEED_FILE}")

    with SEED_FILE.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    players = raw.get("data", [])
    if limit:
        players = players[:limit]

    resolver = EntityResolver()
    key_to_uuid: dict[str, str] = {}
    inserted = updated = skipped = 0

    logger.info("dartsorakel_seed_ingest_start", total=len(players))

    for entry in players:
        player_key = entry.get("player_key")
        if player_key is None:
            skipped += 1
            continue

        player_record = resolver.resolve_from_dartsorakel(entry)
        player_name = entry.get("player_name", "")
        three_da_str = entry.get("stat")
        three_da = float(three_da_str) if three_da_str else None
        rank = entry.get("rank")

        # Try to match existing player by slug or name
        slug = player_record.computed_slug
        existing = await session.execute(
            text(
                "SELECT id FROM darts_players WHERE slug = :slug "
                "OR (lower(first_name || ' ' || last_name) = lower(:name))"
            ),
            {"slug": slug, "name": player_name},
        )
        existing_row = existing.fetchone()

        if existing_row:
            # Update dartsorakel fields on existing player
            player_uuid = str(existing_row[0])
            if not dry_run:
                await session.execute(
                    text(
                        "UPDATE darts_players SET "
                        "dartsorakel_key = :dk, "
                        "dartsorakel_3da = :da, "
                        "dartsorakel_rank = :rk "
                        "WHERE id = :pid"
                    ),
                    {
                        "dk": player_key,
                        "da": three_da,
                        "rk": rank,
                        "pid": player_uuid,
                    },
                )
            updated += 1
        else:
            # Create new player (unmatched — from DartsOrakel only)
            player_uuid = str(uuid.uuid4())
            new_player = DartsPlayer(
                id=player_uuid,
                first_name=player_record.first_name,
                last_name=player_record.last_name,
                slug=slug,
                dartsorakel_key=player_key,
                dartsorakel_3da=three_da,
                dartsorakel_rank=rank,
                country_code=entry.get("country"),
                source_confidence=0.85,
                primary_source="dartsorakel",
                gdpr_anonymized=False,
            )
            if not dry_run:
                session.add(new_player)
            inserted += 1

        key_to_uuid[str(player_key)] = player_uuid

        # Create player stats record
        if three_da is not None and not dry_run:
            stats = DartsPlayerStats(
                id=str(uuid.uuid4()),
                player_id=player_uuid,
                source="dartsorakel",
                three_dart_average=three_da,
                dartsorakel_sumfield1=entry.get("sumField1"),
                dartsorakel_sumfield2=entry.get("sumField2"),
            )
            session.add(stats)

        if (inserted + updated) % 500 == 0:
            if not dry_run:
                await session.flush()

    if not dry_run:
        await session.flush()

    logger.info(
        "dartsorakel_seed_ingest_complete",
        inserted=inserted,
        updated=updated,
        skipped=skipped,
        dry_run=dry_run,
    )
    return key_to_uuid


async def run_ingest(
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> None:
    """Run the full DartsOrakel ingestion pipeline."""
    logger.info("dartsorakel_ingest_start", dry_run=dry_run, limit=limit)
    async with get_session() as session:
        key_to_uuid = await ingest_dartsorakel_seed(
            session, dry_run=dry_run, limit=limit
        )
        if not dry_run:
            await session.commit()
    logger.info(
        "dartsorakel_ingest_complete",
        players_processed=len(key_to_uuid),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest DartsOrakel stats into the XG3 Darts database."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run_ingest(dry_run=args.dry_run, limit=args.limit))
