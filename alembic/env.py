"""Alembic environment — async SQLAlchemy with asyncpg."""
from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from db.models import Base

# Alembic Config object
config = context.config

# Logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_url() -> str:
    url = os.environ.get("DATABASE_URL", config.get_main_option("sqlalchemy.url", ""))
    # Normalise to asyncpg scheme
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def run_migrations_offline() -> None:
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Isolated version table — prevents collision with gateway's alembic_version
        # on the shared Supabase database (DARTS-TABLES-MIGRATION-NEEDED fix 2026-04-30)
        version_table="alembic_version_darts",
    )
    with context.begin_transaction():
        context.run_migrations()


SPORT_VERSION_TABLE = "alembic_version_darts"


def _bootstrap_version_table_if_needed(connection) -> None:
    """If the per-sport version_table is empty but our tables already exist
    (post-cutover state), stamp the table with script-directory HEAD revision
    to avoid DuplicateTableError on first deploy. Idempotent."""
    from sqlalchemy import text
    from alembic.script import ScriptDirectory

    connection.execute(
        text(
            f"CREATE TABLE IF NOT EXISTS {SPORT_VERSION_TABLE} "
            f"(version_num VARCHAR(32) NOT NULL, "
            f"CONSTRAINT {SPORT_VERSION_TABLE}_pkc PRIMARY KEY (version_num))"
        )
    )

    row = connection.execute(text(f"SELECT version_num FROM {SPORT_VERSION_TABLE} LIMIT 1")).first()
    if row is not None:
        return

    sample_tables: list = []
    md_list = target_metadata if isinstance(target_metadata, list) else [target_metadata]
    for md in md_list:
        if md is not None:
            sample_tables.extend(md.tables.keys())
    if not sample_tables:
        return

    existing = connection.execute(
        text(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name = ANY(:names) LIMIT 1"
        ),
        {"names": sample_tables},
    ).first()
    if existing is None:
        return

    script = ScriptDirectory.from_config(context.config)
    head = script.get_current_head()
    if head:
        connection.execute(
            text(f"INSERT INTO {SPORT_VERSION_TABLE} (version_num) VALUES (:v)"),
            {"v": head},
        )


def do_run_migrations(connection):
    _bootstrap_version_table_if_needed(connection)
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Isolated version table — prevents collision with gateway's alembic_version
        # on the shared Supabase database (DARTS-TABLES-MIGRATION-NEEDED fix 2026-04-30)
        version_table="alembic_version_darts",
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    url = _get_url()
    # pgbouncer transaction-mode compatibility (Supabase pooler)
    connectable = create_async_engine(
        url,
        echo=False,
        connect_args={"statement_cache_size": 0},
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
