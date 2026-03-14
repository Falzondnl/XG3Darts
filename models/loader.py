"""
ML model loader — fetches trained artifacts from PostgreSQL at startup.

Models are stored as BYTEA in darts_ml_model_artifacts. On first access,
the artifact is pulled from DB and cached in memory for the lifetime of
the process.

Usage:
    from models.loader import model_store
    r0 = model_store.get("r0_logit")   # returns loaded joblib dict or None
"""
from __future__ import annotations

import io
import threading
from typing import Any, Optional

import joblib
import structlog

logger = structlog.get_logger(__name__)

_NOT_FOUND = object()  # sentinel distinct from None


class ModelStore:
    """Thread-safe lazy loader backed by PostgreSQL BYTEA storage."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, model_name: str, version: str = "1") -> Optional[dict]:
        """
        Return the loaded model dict for *model_name*, or None if not found.

        First call queries the DB; subsequent calls use the in-process cache.
        """
        cache_key = f"{model_name}:{version}"
        with self._lock:
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                return None if cached is _NOT_FOUND else cached

        result = self._load_from_db(model_name, version)
        with self._lock:
            self._cache[cache_key] = result if result is not None else _NOT_FOUND
        return result

    def _load_from_db(self, model_name: str, version: str) -> Optional[dict]:
        """Pull artifact BYTEA from the database and deserialize via joblib."""
        try:
            import psycopg2
            from app.config import settings

            db_url = (
                settings.DATABASE_URL
                .replace("postgresql+asyncpg://", "postgresql://")
                .replace("postgres+asyncpg://", "postgresql://")
            )
            conn = psycopg2.connect(db_url, connect_timeout=10)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT artifact FROM darts_ml_model_artifacts "
                        "WHERE model_name = %s AND version = %s",
                        (model_name, version),
                    )
                    row = cur.fetchone()
            finally:
                conn.close()

            if row is None:
                logger.warning("model_not_in_db", model=model_name, version=version)
                return None

            artifact_bytes = bytes(row[0])
            model_dict = joblib.load(io.BytesIO(artifact_bytes))
            logger.info(
                "model_loaded_from_db",
                model=model_name,
                version=version,
                size_kb=round(len(artifact_bytes) / 1024, 1),
            )
            return model_dict

        except Exception as exc:
            logger.error("model_load_failed", model=model_name, error=str(exc))
            return None

    def preload(self, *model_names: str) -> None:
        """Pre-warm the cache for the given model names (version=1)."""
        for name in model_names:
            self.get(name)

    def invalidate(self, model_name: str, version: str = "1") -> None:
        """Remove a cached model (forces re-fetch on next access)."""
        with self._lock:
            self._cache.pop(f"{model_name}:{version}", None)


# Module-level singleton
model_store = ModelStore()
