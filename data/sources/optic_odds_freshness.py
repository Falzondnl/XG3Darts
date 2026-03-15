"""
Optic Odds Data Freshness Service.

Polls ``GET /fixtures/active?sport=darts`` on a configurable interval,
compares against existing DB rows, and upserts new upcoming/live fixtures
into ``darts_matches`` (with ``external_source_id`` = Optic Odds fixture ID).

Run as an autonomous background service:

    from data.sources.optic_odds_freshness import OpticOddsDataFreshnessService

    svc = OpticOddsDataFreshnessService(db_url=settings.DATABASE_URL)
    await svc.start()
    # runs until cancelled

Or as a one-shot batch:

    count = await svc.sync_once()
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg
import structlog

from app.config import settings
from data.sources.optic_odds_client import OpticOddsFixture, OpticOddsLiveFeed

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How often to poll for new fixtures (seconds)
_POLL_INTERVAL_S: float = 60.0

# Minimum time in the future for a fixture to be inserted (seconds)
# Avoids inserting fixtures that already started before we polled
_MIN_FUTURE_HORIZON_S: float = -300.0  # allow up to 5 min past start


# ---------------------------------------------------------------------------
# Name matching helpers
# ---------------------------------------------------------------------------

def _normalise_name(name: str) -> str:
    """Lower, strip accents (ASCII only), collapse whitespace."""
    name = name.lower().strip()
    # Remove punctuation except hyphens and apostrophes
    name = re.sub(r"[^\w\s\-']", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _name_fingerprint(name: str) -> str:
    """MD5 fingerprint of the normalised name — used for fast lookup."""
    return hashlib.md5(_normalise_name(name).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class OpticOddsDataFreshnessService:
    """
    Background service that keeps darts match data fresh via Optic Odds.

    On each poll cycle:
      1. Fetch ``GET /fixtures/active?sport=darts`` from Optic Odds.
      2. For each fixture, attempt to match against ``darts_players`` by name.
      3. Upsert into ``darts_matches`` using ``external_source_id`` = optic fixture ID
         as the idempotency key.
      4. Log counts of inserted / updated / skipped.

    Parameters
    ----------
    db_url:
        asyncpg-compatible Postgres URL (``postgresql://...``).
    poll_interval_s:
        Seconds between polls. Default 60.
    feed:
        Optional pre-existing ``OpticOddsLiveFeed`` to reuse.  A new one is
        created if not supplied.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        poll_interval_s: float = _POLL_INTERVAL_S,
        feed: Optional[OpticOddsLiveFeed] = None,
    ) -> None:
        self._db_url = db_url or settings.DATABASE_URL
        # asyncpg wants postgresql:// not postgresql+asyncpg://
        self._db_url = self._db_url.replace("postgresql+asyncpg://", "postgresql://")
        self._poll_interval_s = poll_interval_s
        self._feed = feed or OpticOddsLiveFeed()
        self._own_feed = feed is None  # close it on shutdown only if we own it
        self._pool: Optional[asyncpg.Pool] = None
        self._log = logger.bind(service="OpticOddsDataFreshnessService")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Start the poll loop. Runs indefinitely until cancelled.

        Creates a DB connection pool and initialises the HTTP feed.
        """
        self._pool = await asyncpg.create_pool(
            self._db_url,
            min_size=1,
            max_size=3,
            command_timeout=30,
        )
        if self._own_feed:
            await self._feed.connect()

        self._log.info(
            "optic_odds_freshness_started",
            poll_interval_s=self._poll_interval_s,
        )
        try:
            while True:
                try:
                    result = await self.sync_once()
                    self._log.info(
                        "optic_odds_freshness_cycle_done",
                        inserted=result["inserted"],
                        updated=result["updated"],
                        skipped=result["skipped"],
                        errors=result["errors"],
                    )
                except Exception as exc:
                    self._log.error(
                        "optic_odds_freshness_cycle_error",
                        error=str(exc),
                    )
                await asyncio.sleep(self._poll_interval_s)
        except asyncio.CancelledError:
            self._log.info("optic_odds_freshness_stopping")
            raise
        finally:
            await self._close()

    async def _close(self) -> None:
        if self._pool:
            await self._pool.close()
        if self._own_feed:
            await self._feed.close()

    # ------------------------------------------------------------------
    # Core sync
    # ------------------------------------------------------------------

    async def sync_once(self) -> dict[str, int]:
        """
        Run a single sync cycle.

        Returns
        -------
        dict with keys: inserted, updated, skipped, errors
        """
        if self._pool is None:
            raise RuntimeError("Service not started — call await svc.start() first.")

        fixtures = await self._feed.get_active_fixtures(sport="darts")
        self._log.info("optic_odds_fixtures_received", count=len(fixtures))

        inserted = 0
        updated = 0
        skipped = 0
        errors = 0

        async with self._pool.acquire() as conn:
            for fixture in fixtures:
                try:
                    action = await self._upsert_fixture(conn, fixture)
                    if action == "inserted":
                        inserted += 1
                    elif action == "updated":
                        updated += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    errors += 1
                    self._log.error(
                        "optic_odds_fixture_upsert_error",
                        fixture_id=fixture.optic_id,
                        home=fixture.home_team,
                        away=fixture.away_team,
                        error=str(exc),
                    )

        return {"inserted": inserted, "updated": updated, "skipped": skipped, "errors": errors}

    # ------------------------------------------------------------------
    # Fixture upsert logic
    # ------------------------------------------------------------------

    async def _upsert_fixture(
        self,
        conn: asyncpg.Connection,
        fixture: OpticOddsFixture,
    ) -> str:
        """
        Upsert a single Optic Odds fixture into ``darts_matches``.

        Returns
        -------
        "inserted" | "updated" | "skipped"
        """
        # Check if we already have this fixture via external_source_id
        existing = await conn.fetchrow(
            """
            SELECT id, status FROM darts_matches
            WHERE external_source_id = $1
            LIMIT 1
            """,
            f"optic:{fixture.optic_id}",
        )

        # Map Optic Odds status to our internal status
        internal_status = _map_status(fixture.status)

        if existing:
            # Update status if it changed (e.g. not_started → live → result)
            if existing["status"] != internal_status:
                await conn.execute(
                    """
                    UPDATE darts_matches
                    SET status = $1, updated_at = NOW()
                    WHERE id = $2
                    """,
                    internal_status,
                    existing["id"],
                )
                self._log.debug(
                    "optic_odds_match_status_updated",
                    match_id=existing["id"],
                    old_status=existing["status"],
                    new_status=internal_status,
                )
                return "updated"
            return "skipped"

        # New fixture — resolve player IDs
        p1_id = await self._resolve_player_id(conn, fixture.home_team, fixture.home_id)
        p2_id = await self._resolve_player_id(conn, fixture.away_team, fixture.away_id)

        if p1_id is None or p2_id is None:
            self._log.debug(
                "optic_odds_players_not_resolved",
                home=fixture.home_team,
                away=fixture.away_team,
                home_id=p1_id,
                away_id=p2_id,
            )
            # Still insert the fixture with NULL player IDs so we track it
            # and can re-resolve later when player data is enriched.

        # Resolve competition
        competition_id = await self._resolve_competition(conn, fixture)

        # Parse start time
        start_dt = _parse_iso(fixture.start_time)

        # Skip fixtures that started too long ago
        if start_dt is not None:
            age_s = (datetime.now(tz=timezone.utc) - start_dt).total_seconds()
            if age_s > abs(_MIN_FUTURE_HORIZON_S) and internal_status == "Upcoming":
                self._log.debug(
                    "optic_odds_fixture_too_old",
                    fixture_id=fixture.optic_id,
                    start_time=fixture.start_time,
                    age_s=round(age_s),
                )
                return "skipped"

        match_id = str(uuid.uuid4())
        await conn.execute(
            """
            INSERT INTO darts_matches (
                id, competition_id, player1_id, player2_id,
                match_date, status, format_code,
                external_source_id, raw_source_data,
                created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7,
                $8, $9,
                NOW(), NOW()
            )
            ON CONFLICT (external_source_id) DO NOTHING
            """,
            match_id,
            competition_id,
            p1_id,
            p2_id,
            start_dt,
            internal_status,
            _infer_format_code(fixture),
            f"optic:{fixture.optic_id}",
            _build_raw_json(fixture),
        )
        self._log.info(
            "optic_odds_match_inserted",
            match_id=match_id,
            fixture_id=fixture.optic_id,
            home=fixture.home_team,
            away=fixture.away_team,
            start=fixture.start_time,
        )
        return "inserted"

    # ------------------------------------------------------------------
    # Player resolution
    # ------------------------------------------------------------------

    async def _resolve_player_id(
        self,
        conn: asyncpg.Connection,
        name: str,
        optic_participant_id: str,
    ) -> Optional[str]:
        """
        Resolve an Optic Odds participant to an internal ``darts_players.id``.

        Resolution order:
          1. ``optic_participant_id`` stored in ``raw_source_data`` JSONB
          2. Exact normalised name match on ``first_name || ' ' || last_name``
          3. Surname-only match (last resort, only when unique)
        """
        if not name:
            return None

        # Step 1: look up by stored Optic participant ID
        if optic_participant_id:
            row = await conn.fetchrow(
                """
                SELECT id FROM darts_players
                WHERE raw_source_data->>'optic_participant_id' = $1
                LIMIT 1
                """,
                optic_participant_id,
            )
            if row:
                return str(row["id"])

        norm = _normalise_name(name)

        # Step 2: full normalised name match
        row = await conn.fetchrow(
            """
            SELECT id FROM darts_players
            WHERE lower(trim(first_name || ' ' || last_name)) = $1
               OR lower(trim(nickname)) = $1
            LIMIT 1
            """,
            norm,
        )
        if row:
            return str(row["id"])

        # Step 3: last name only (unique match)
        parts = norm.split()
        if len(parts) >= 2:
            last = parts[-1]
            rows = await conn.fetch(
                """
                SELECT id FROM darts_players
                WHERE lower(trim(last_name)) = $1
                LIMIT 2
                """,
                last,
            )
            if len(rows) == 1:
                return str(rows[0]["id"])

        return None

    # ------------------------------------------------------------------
    # Competition resolution
    # ------------------------------------------------------------------

    async def _resolve_competition(
        self,
        conn: asyncpg.Connection,
        fixture: OpticOddsFixture,
    ) -> Optional[str]:
        """
        Match the Optic Odds league to an existing ``darts_competitions.id``.

        Returns None if no match found — the match is still inserted without
        a competition link.
        """
        if not fixture.league:
            return None

        row = await conn.fetchrow(
            """
            SELECT id FROM darts_competitions
            WHERE lower(name) = lower($1)
               OR lower(format_code) = lower($1)
            LIMIT 1
            """,
            fixture.league,
        )
        if row:
            return str(row["id"])

        # Try partial match on league name
        row = await conn.fetchrow(
            """
            SELECT id FROM darts_competitions
            WHERE lower(name) LIKE lower($1)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            f"%{fixture.league[:30]}%",
        )
        return str(row["id"]) if row else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_status(optic_status: str) -> str:
    """Map Optic Odds fixture status to internal darts_matches status."""
    mapping = {
        "not_started": "Upcoming",
        "live": "InProgress",
        "finished": "Completed",
        "cancelled": "Cancelled",
        "postponed": "Postponed",
        "suspended": "Suspended",
    }
    return mapping.get(optic_status.lower(), "Upcoming")


def _infer_format_code(fixture: OpticOddsFixture) -> str:
    """Infer a competition format code from the fixture league name."""
    league_lower = fixture.league.lower()
    if "world championship" in league_lower or "world champ" in league_lower:
        return "PDC_WC"
    if "premier league" in league_lower:
        return "PDC_PL"
    if "world matchplay" in league_lower:
        return "PDC_WM"
    if "grand prix" in league_lower:
        return "PDC_GP"
    if "uk open" in league_lower:
        return "PDC_UK"
    if "grand slam" in league_lower:
        return "PDC_GS"
    if "players championship" in league_lower:
        return "PDC_PC"
    if "masters" in league_lower:
        return "PDC_MASTERS"
    if "wdf" in league_lower:
        return "WDF_OPEN"
    return "PDC_GP"  # safest generic PDC format


def _parse_iso(s: str) -> Optional[datetime]:
    """Parse ISO-8601 string → timezone-aware datetime. Returns None on failure."""
    if not s:
        return None
    try:
        # Handle both Z suffix and +00:00
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _build_raw_json(fixture: OpticOddsFixture) -> dict[str, Any]:
    """Build the raw_source_data JSONB payload for a new match row."""
    return {
        "source": "optic_odds",
        "optic_fixture_id": fixture.optic_id,
        "optic_participant_home": fixture.home_id,
        "optic_participant_away": fixture.away_id,
        "league": fixture.league,
        "league_id": fixture.league_id,
        "scraped_status": fixture.status,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Optic Odds data freshness service")
    parser.add_argument("--db-url", default=None, help="PostgreSQL URL")
    parser.add_argument(
        "--once", action="store_true", help="Run one cycle and exit"
    )
    parser.add_argument(
        "--interval", type=float, default=_POLL_INTERVAL_S,
        help="Poll interval in seconds (default 60)",
    )
    args = parser.parse_args()

    svc = OpticOddsDataFreshnessService(
        db_url=args.db_url,
        poll_interval_s=args.interval,
    )

    if args.once:
        # One-shot mode — create pool + feed manually
        import asyncpg as _asyncpg
        db_url = (args.db_url or settings.DATABASE_URL).replace(
            "postgresql+asyncpg://", "postgresql://"
        )
        svc._pool = await _asyncpg.create_pool(db_url, min_size=1, max_size=2)
        await svc._feed.connect()
        result = await svc.sync_once()
        print(
            f"Sync complete — inserted={result['inserted']} "
            f"updated={result['updated']} skipped={result['skipped']} "
            f"errors={result['errors']}"
        )
        await svc._close()
    else:
        await svc.start()


if __name__ == "__main__":
    asyncio.run(_main())
