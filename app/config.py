"""
Application configuration.

All settings are loaded from environment variables via Pydantic Settings.
Copy ``.env.example`` to ``.env`` and fill in the required values before
running the service.

No secrets or defaults that look like real credentials are hardcoded here.
"""
from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    XG3 Darts service settings.

    All values are read from environment variables.  The ``.env`` file
    (not committed to version control) is loaded automatically.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Service identity
    # ------------------------------------------------------------------ #
    SERVICE_NAME: str = "xg3-darts"
    SERVICE_VERSION: str = "0.1.0"
    ENVIRONMENT: str = Field(default="development", description="production | staging | development")
    LOG_LEVEL: str = Field(default="INFO", description="DEBUG | INFO | WARNING | ERROR")

    # ------------------------------------------------------------------ #
    # API keys — Optic Odds
    # ------------------------------------------------------------------ #
    OPTIC_ODDS_API_KEY: str = Field(
        default="",
        description="Optic Odds HTTP API key",
    )
    OPTIC_ODDS_HTTP_PASSWORD: str = Field(
        default="",
        description="Optic Odds HTTP password",
    )
    OPTIC_ODDS_RABBITMQ_USER: str = Field(
        default="",
        description="Optic Odds RabbitMQ username",
    )
    OPTIC_ODDS_RABBITMQ_PASS: str = Field(
        default="",
        description="Optic Odds RabbitMQ password",
    )
    OPTIC_ODDS_RABBITMQ_HOST: str = Field(
        default="",
        description="Optic Odds RabbitMQ host",
    )

    # ------------------------------------------------------------------ #
    # DartConnect
    # ------------------------------------------------------------------ #
    DARTCONNECT_API_KEY: str = Field(
        default="",
        description="DartConnect API key",
    )
    DARTCONNECT_BASE_URL: str = Field(
        default="https://api.dartconnect.com",
        description="DartConnect API base URL",
    )

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost:5432/xg3darts",
        description="Async PostgreSQL connection URL",
    )
    DB_ECHO: bool = Field(
        default=False,
        description="Enable SQLAlchemy query echo (dev only)",
    )
    DB_POOL_SIZE: int = Field(default=10, ge=1, le=100)
    DB_MAX_OVERFLOW: int = Field(default=20, ge=0, le=100)

    # ------------------------------------------------------------------ #
    # Redis
    # ------------------------------------------------------------------ #
    REDIS_URL: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL",
    )
    REDIS_CACHE_TTL_SECONDS: int = Field(
        default=300,
        ge=1,
        description="Default cache TTL in seconds",
    )

    # ------------------------------------------------------------------ #
    # Service URLs
    # ------------------------------------------------------------------ #
    DARTS_SERVICE_URL: str = Field(
        default="http://localhost:8000/api/v1/darts",
        description="This service's own base URL (used in internal links)",
    )

    # ------------------------------------------------------------------ #
    # GDPR
    # ------------------------------------------------------------------ #
    GDPR_PSEUDONYM_SECRET: str = Field(
        default="",
        description="HMAC secret for GDPR pseudonymization (must be set in prod)",
    )

    # ------------------------------------------------------------------ #
    # Data paths
    # ------------------------------------------------------------------ #
    DATA_ROOT: str = Field(
        default="D:/codex/Data/Darts",
        description="Root directory for all darts data files",
    )

    # ------------------------------------------------------------------ #
    # Feature flags
    # ------------------------------------------------------------------ #
    ENABLE_LIVE_PRICING: bool = Field(
        default=False,
        description="Enable live in-play pricing engine",
    )
    ENABLE_GPU_SIMULATION: bool = Field(
        default=False,
        description="Enable GPU-accelerated tournament simulation",
    )

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid = {"production", "staging", "development", "test"}
        if v.lower() not in valid:
            raise ValueError(f"ENVIRONMENT must be one of {sorted(valid)}, got {v!r}")
        return v.lower()

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(valid)}, got {v!r}")
        return v.upper()

    @property
    def is_production(self) -> bool:
        """True when running in production environment."""
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        """True when running in development environment."""
        return self.ENVIRONMENT == "development"


# Module-level singleton — import this from other modules
settings = Settings()
