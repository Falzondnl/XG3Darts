"""
Alembic version stamp fix.

Runs at startup BEFORE `alembic upgrade head`.

Problem: migration 004 was originally created with revision ID
"004_liability_and_trader_overrides" and then corrected to "004".
If the Railway DB was stamped with the old ID, alembic cannot locate
the revision and fails at startup.

This script connects to the DB, inspects alembic_version, and corrects
any stale revision IDs so that `alembic upgrade head` can proceed cleanly.

Stale → correct mappings:
    "004_liability_and_trader_overrides" → "004"
    "003_ml_model_artifacts"             → "003"
"""
from __future__ import annotations

import asyncio
import os
import sys


_STALE_MAP: dict[str, str] = {
    "004_liability_and_trader_overrides": "004",
    "003_ml_model_artifacts": "003",
}


def _normalise_url(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


async def fix_stale_revisions() -> None:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        print("[stamp-fix] DATABASE_URL not set — skipping.", flush=True)
        return

    url = _normalise_url(url)

    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text
    except ImportError as exc:
        print(f"[stamp-fix] SQLAlchemy not available: {exc} — skipping.", flush=True)
        return

    engine = create_async_engine(
        url,
        echo=False,
        connect_args={"timeout": 10, "statement_cache_size": 0},
    )
    try:
        async with engine.begin() as conn:
            # Check whether alembic_version table exists
            result = await conn.execute(text(
                "SELECT to_regclass('public.alembic_version')"
            ))
            row = result.scalar()
            if row is None:
                print("[stamp-fix] alembic_version table does not exist yet — nothing to fix.", flush=True)
                return

            result = await conn.execute(text("SELECT version_num FROM alembic_version"))
            rows = result.fetchall()
            current_versions = {r[0] for r in rows}
            print(f"[stamp-fix] Current alembic_version: {current_versions}", flush=True)

            fixed = False
            for stale, correct in _STALE_MAP.items():
                if stale in current_versions:
                    await conn.execute(
                        text("UPDATE alembic_version SET version_num = :correct WHERE version_num = :stale"),
                        {"correct": correct, "stale": stale},
                    )
                    print(f"[stamp-fix] Fixed stale revision: {stale!r} → {correct!r}", flush=True)
                    fixed = True

            if not fixed:
                print("[stamp-fix] No stale revisions found — alembic_version is clean.", flush=True)
    except Exception as exc:
        # Non-fatal: log and continue — alembic will report its own error
        print(f"[stamp-fix] WARNING: could not fix alembic_version: {exc}", flush=True)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(fix_stale_revisions())
    sys.exit(0)
