"""
XG3 Darts FastAPI application entry point.

Registers all API routers, configures structured logging via structlog,
sets up middleware, and exposes health/readiness endpoints.

API prefix: /api/v1/darts

# ---------------------------------------------------------------------------
# DESIGN NOTE — Autonomy Worker Architecture
# ---------------------------------------------------------------------------
#
# Two worker classes exist in this repo:
#
#   app/workers/live_ops_worker.py  (DartsLiveOpsWorker)
#       The Darts-specific live ops worker. Fully implements W1-W9:
#       feed connection, auto-seed, auto-settle, completion detection,
#       live state enumeration, restart recovery, in-process freshness
#       worker, and singleton live engine. Started in lifespan below.
#       This is the ACTIVE worker for Darts.
#
#   shared/autonomy_worker.py  (AutonomyWorker)
#       A shared cross-sport backbone with the same fixture-poll /
#       stale-check / health-log loop structure, used by other sport
#       microservices that lack a dedicated worker.
#
# WHY AutonomyWorker IS NOT STARTED IN THIS LIFESPAN:
#   DartsLiveOpsWorker already covers every responsibility the shared
#   AutonomyWorker provides. Running both would create a duplicate
#   fixture-poll loop calling Optic Odds /api/v3/fixtures/active every
#   30 s, doubling API load and risking duplicate settlement triggers on
#   completed events. The shared worker is retained for cross-sport reuse
#   only; Darts does not need it.
#
# IF YOU ARE ADDING A NEW SPORT MS without a dedicated worker, start the
# shared worker like this in its lifespan:
#
#   from shared.autonomy_worker import get_autonomy_worker
#   worker = get_autonomy_worker("your_sport", optic_odds_api_key=settings.OPTIC_ODDS_API_KEY)
#   await worker.start()
# ---------------------------------------------------------------------------
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

# ---------------------------------------------------------------------------
# Sentry error monitoring (set SENTRY_DSN env var to activate in production)
# ---------------------------------------------------------------------------
_SENTRY_DSN = os.getenv("SENTRY_DSN", "")
if _SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            integrations=[
                StarletteIntegration(transaction_style="endpoint"),
                FastApiIntegration(transaction_style="endpoint"),
            ],
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.1")),
            environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
            release=f"xg3-darts@{os.getenv('RAILWAY_GIT_COMMIT_SHA', 'local')[:8]}",
            send_default_pii=False,
        )
        import logging as _sentry_logging
        _sentry_logging.getLogger("xg3_darts").info("[Sentry] Initialized — DSN configured")
    except Exception as _sentry_err:
        import logging as _sentry_logging
        _sentry_logging.getLogger("xg3_darts").warning("[Sentry] Init failed: %s", _sentry_err)
else:
    import logging as _sentry_logging
    _sentry_logging.getLogger("xg3_darts").warning(
        "[Sentry] SENTRY_DSN not set — unhandled exceptions will NOT be captured. "
        "Set SENTRY_DSN env var in Railway for production error monitoring."
    )

import structlog
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.auth import api_key_middleware
from app.config import settings
from app.routes import (
    derivatives,
    events,
    feeds,
    form,
    h2h,
    liability,
    live,
    monitoring,
    outrights,
    players,
    prematch,
    props,
    settlement,
    sgp,
    trader,
    trading_controls,
    worldcup,
)
from app.routes.monitoring import router as monitoring_router
from app.routes.predict import router as predict_router


# ---------------------------------------------------------------------------
# Structured logging setup
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    """Configure structlog with appropriate processors for the environment."""
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.is_production:
        # JSON output for log aggregation (Datadog, CloudWatch, etc.)
        structlog.configure(
            processors=shared_processors
            + [
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                structlog.stdlib.NAME_TO_LEVEL.get(settings.LOG_LEVEL, 20)
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Human-readable output for development
        structlog.configure(
            processors=shared_processors
            + [
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                structlog.stdlib.NAME_TO_LEVEL.get(settings.LOG_LEVEL, 20)
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )


_configure_logging()
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle manager."""
    logger.info(
        "startup",
        service=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        environment=settings.ENVIRONMENT,
    )

    # ---------------------------------------------------------------------------
    # Live Operations Worker
    # Wires W1-W9: feed connection, auto-seed, auto-settle, completion detection,
    # live state enumeration, restart recovery, in-process freshness worker,
    # and singleton live engine.
    # ---------------------------------------------------------------------------
    from app.workers.live_ops_worker import get_live_ops_worker

    worker = get_live_ops_worker()
    try:
        await worker.start()
    except Exception as exc:
        # Worker failure is non-fatal: app continues serving pre-match traffic
        logger.warning(
            "live_ops_worker_start_failed",
            error=str(exc),
            hint="Pre-match endpoints remain available; live pricing degraded",
        )

    yield

    # Shutdown: stop the worker gracefully
    try:
        await worker.stop()
    except Exception as exc:
        logger.warning("live_ops_worker_stop_error", error=str(exc))

    logger.info("shutdown", service=settings.SERVICE_NAME)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        Configured application instance.
    """
    app = FastAPI(
        title="XG3 Darts Pricing Engine",
        description=(
            "Tier-1 sports betting microservice for PDC/WDF darts markets. "
            "Provides pre-match, live, outright, SGP, and prop markets."
        ),
        version=settings.SERVICE_VERSION,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["*"],
    )

    # API key authentication (enforced when API_KEYS env var is set)
    app.middleware("http")(api_key_middleware)

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            path=str(request.url.path),
            method=request.method,
        )
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "http_request",
            status_code=response.status_code,
            elapsed_ms=round(elapsed_ms, 2),
        )
        return response

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled_exception",
            path=str(request.url.path),
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred.",
                "service": settings.SERVICE_NAME,
            },
        )

    # Routers
    api_prefix = "/api/v1/darts"
    app.include_router(prematch.router, prefix=api_prefix, tags=["Pre-Match"])
    app.include_router(derivatives.router, prefix=api_prefix, tags=["Derivatives"])
    app.include_router(trading_controls.router, prefix=api_prefix, tags=["Trading Controls"])
    app.include_router(events.router, prefix=api_prefix, tags=["Events"])
    app.include_router(live.router, prefix=api_prefix, tags=["Live"])
    app.include_router(outrights.router, prefix=api_prefix, tags=["Outrights"])
    app.include_router(sgp.router, prefix=api_prefix, tags=["SGP"])
    app.include_router(props.router, prefix=api_prefix, tags=["Props"])
    app.include_router(worldcup.router, prefix=api_prefix, tags=["World Cup"])
    app.include_router(players.router, prefix=api_prefix, tags=["Players"])
    app.include_router(monitoring_router, prefix=api_prefix, tags=["Monitoring"])
    app.include_router(liability.router, prefix=api_prefix, tags=["Liability"])
    app.include_router(trader.router, prefix=api_prefix, tags=["Trader"])
    app.include_router(feeds.router, prefix=api_prefix, tags=["Feeds"])
    app.include_router(settlement.router, prefix=api_prefix, tags=["Settlement"])
    app.include_router(h2h.router, prefix=api_prefix, tags=["H2H"])
    app.include_router(form.router, prefix=api_prefix, tags=["Form"])
    app.include_router(predict_router, prefix=api_prefix, tags=["Predict"])

    # Prometheus metrics endpoint (scraped by external Prometheus server)
    try:
        from prometheus_client import make_asgi_app as _prom_asgi

        _metrics_app = _prom_asgi()
        app.mount("/metrics", _metrics_app)
        logger.info("prometheus_metrics_endpoint_mounted", path="/metrics")
    except ImportError:
        logger.warning("prometheus_client not installed — /metrics not available")

    # Health / readiness (no versioned prefix — checked by load balancers)
    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": settings.SERVICE_NAME,
            "version": settings.SERVICE_VERSION,
        }

    @app.get("/health/live", include_in_schema=False)
    async def health_live() -> dict[str, Any]:
        """Liveness probe — always returns 200 while the process is running."""
        return {
            "status": "alive",
            "service": settings.SERVICE_NAME,
            "version": settings.SERVICE_VERSION,
        }

    @app.get("/ready", include_in_schema=False)
    @app.get("/health/ready", include_in_schema=False)
    async def readiness() -> dict[str, Any]:
        """
        Readiness probe — checks DB and Redis connectivity.
        Returns 200 when all dependencies are healthy.
        Available at both /ready (legacy) and /health/ready (standard).
        """
        checks: dict[str, str] = {}

        # Database check
        try:
            from db.session import engine
            async with engine.connect() as conn:
                await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            checks["database"] = "ok"
        except Exception as exc:
            logger.warning("readiness_db_failed", error=str(exc))
            checks["database"] = "unavailable"

        # Redis check
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
            await client.ping()
            await client.aclose()
            checks["redis"] = "ok"
        except Exception as exc:
            logger.warning("readiness_redis_failed", error=str(exc))
            checks["redis"] = "unavailable"

        all_ok = all(v == "ok" for v in checks.values())
        return JSONResponse(
            status_code=200 if all_ok else 503,
            content={
                "status": "ready" if all_ok else "degraded",
                "checks": checks,
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=False,  # handled by our middleware
    )
