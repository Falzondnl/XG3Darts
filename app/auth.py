"""
API key authentication for XG3 Darts.

B2B clients authenticate using an X-Api-Key header.  Keys are stored as
HMAC-SHA256 hashes in the API_KEYS environment variable (comma-separated
list of plaintext keys — hashed at startup and never stored in plain text).

The API_KEY_SALT environment variable must be set in production.

Endpoints excluded from auth:
  /health, /ready, /docs, /redoc, /openapi.json

Usage in routes (optional per-route enforcement):
    from app.auth import require_api_key
    @router.get("/sensitive", dependencies=[Depends(require_api_key)])
    async def sensitive_endpoint(): ...

OR applied globally as middleware in main.py (current implementation).
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Optional

import structlog
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths that bypass authentication entirely
# ---------------------------------------------------------------------------

_PUBLIC_PREFIXES: tuple[str, ...] = (
    "/health",
    "/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
)

# ---------------------------------------------------------------------------
# Key registry — built once at module import from environment
# ---------------------------------------------------------------------------


def _load_valid_key_hashes() -> frozenset[str]:
    """
    Load API keys from the API_KEYS environment variable.

    API_KEYS must be a comma-separated list of plaintext API keys.
    They are hashed with HMAC-SHA256 (keyed by API_KEY_SALT) so that
    the plaintext keys are never retained in memory beyond this call.

    Returns an empty frozenset if the environment variable is not set,
    which disables key enforcement (development mode).
    """
    salt = os.environ.get("API_KEY_SALT", "")
    raw = os.environ.get("API_KEYS", "")
    if not raw:
        return frozenset()
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    hashed = frozenset(
        hmac.new(salt.encode(), key.encode(), hashlib.sha256).hexdigest()
        for key in keys
    )
    logger.info("api_key_registry_loaded", count=len(hashed))
    return hashed


_VALID_KEY_HASHES: frozenset[str] = _load_valid_key_hashes()


def _hash_key(key: str) -> str:
    """Hash an API key with the salt for constant-time comparison."""
    salt = os.environ.get("API_KEY_SALT", "")
    return hmac.new(salt.encode(), key.encode(), hashlib.sha256).hexdigest()


def _is_valid_key(key: str) -> bool:
    """
    Return True if key is in the registry.

    Uses hmac.compare_digest for timing-safe comparison.
    If no keys are registered (development mode), all keys are accepted.
    """
    if not _VALID_KEY_HASHES:
        # Fail-closed in production: refuse to accept if no keys configured.
        import os

        env = os.getenv("ENVIRONMENT", "development")
        if env == "production":
            logger.error("auth_fail_closed: API_KEYS empty in production — rejecting")
            return False
        return True  # no keys configured → open (dev mode only)
    candidate_hash = _hash_key(key)
    return any(
        hmac.compare_digest(candidate_hash, valid)
        for valid in _VALID_KEY_HASHES
    )


# ---------------------------------------------------------------------------
# FastAPI middleware
# ---------------------------------------------------------------------------


async def api_key_middleware(request: Request, call_next):
    """
    ASGI middleware that enforces X-Api-Key authentication on all non-public routes.

    Public routes (health, docs) bypass auth entirely.
    If API_KEYS is empty, all requests are allowed (development mode).
    On failure: returns 401 with a structured JSON error body.
    """
    path = request.url.path

    # Bypass for health checks and docs
    if any(path.startswith(prefix) for prefix in _PUBLIC_PREFIXES):
        return await call_next(request)

    # If no keys configured: allow in dev, block in production
    if not _VALID_KEY_HASHES:
        import os

        if os.getenv("ENVIRONMENT", "development") == "production":
            logger.critical(
                "auth_fail_closed_middleware",
                path=path,
                message="API_KEYS empty in production — blocking all non-public requests",
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "auth_not_configured",
                    "message": "Service API keys not configured. Contact operator.",
                },
            )
        return await call_next(request)

    api_key: Optional[str] = request.headers.get("X-Api-Key")
    if not api_key:
        logger.warning("auth_missing_key", path=path, method=request.method)
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "unauthorized",
                "message": "X-Api-Key header is required.",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not _is_valid_key(api_key):
        logger.warning("auth_invalid_key", path=path, method=request.method)
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "unauthorized",
                "message": "Invalid API key.",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return await call_next(request)
