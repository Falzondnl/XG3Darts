"""
Async SQLAlchemy session factory.

Uses ``asyncpg`` as the driver.  The connection URL is read from the
``DATABASE_URL`` environment variable (see ``.env.example``).

Usage
-----
    from db.session import get_session

    async def my_handler():
        async with get_session() as session:
            result = await session.execute(...)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings


def _build_engine() -> AsyncEngine:
    """Build the async SQLAlchemy engine from application settings.

    SQLite (aiosqlite) does not support pool_size, max_overflow, pool_timeout,
    pool_recycle, or pool_pre_ping — those kwargs are PostgreSQL/asyncpg only.
    We detect SQLite by URL prefix and strip the incompatible kwargs.
    """
    url = str(settings.DATABASE_URL)
    is_sqlite = url.startswith("sqlite")

    kwargs: dict = {
        "echo": settings.DB_ECHO,
    }

    if not is_sqlite:
        # PostgreSQL / asyncpg — full pool config
        kwargs.update({
            "pool_size": settings.DB_POOL_SIZE,
            "max_overflow": settings.DB_MAX_OVERFLOW,
            "pool_pre_ping": True,
            "pool_recycle": 1800,
            "pool_timeout": 20,
            "connect_args": {
                "timeout": 10,           # asyncpg TCP connect timeout (seconds)
                "command_timeout": 30,   # per-query timeout
                # pgbouncer transaction-mode compatibility (Supabase pooler)
                "statement_cache_size": 0,
            },
        })

    return create_async_engine(url, **kwargs)


# Module-level engine and session factory — initialised once at import time.
engine: AsyncEngine = _build_engine()

_async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that yields a database session.

    Automatically commits on success and rolls back on exception.

    Yields
    ------
    AsyncSession
        A live database session.

    Raises
    ------
    Any SQLAlchemy exception that occurs during the session.
    """
    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency version of :func:`get_session`.

    Use with ``Depends(get_session_dependency)`` in route handlers.
    """
    async with get_session() as session:
        yield session
