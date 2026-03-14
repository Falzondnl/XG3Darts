"""
SQLAlchemy ORM models for the XG3 Darts microservice.

All models use the ``async`` SQLAlchemy 2.x mapped-column style.
GDPR-sensitive fields are stored separately and controlled by
:mod:`compliance.gdpr_anonymizer`.

Table Overview
--------------
- ``darts_players``         – Canonical player records
- ``darts_competitions``    – Tournament / competition records
- ``darts_matches``         – Individual match records
- ``darts_legs``            – Individual leg records within a match
- ``darts_visits``          – Per-visit (3-dart throw) records
- ``darts_player_stats``    – Aggregated per-player statistics per competition
- ``darts_coverage_regimes``– R0/R1/R2 data regime per player/competition
- ``darts_elo_ratings``     – ELO rating history per player/pool
- ``darts_gdpr_consents``   – GDPR consent records (separate table)
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Shared declarative base for all Darts ORM models."""


# ---------------------------------------------------------------------------
# darts_players
# ---------------------------------------------------------------------------

class DartsPlayer(Base):
    """
    Canonical player record, one row per real-world person.

    Source IDs from external systems are stored as nullable foreign keys
    so entity resolution can progressively link records.  GDPR-sensitive
    fields (``dob``, ``nationality``) are stored in this table but must
    only be returned to authorised consumers; the ``gdpr_anonymized``
    flag indicates whether the record has been scrubbed.
    """

    __tablename__ = "darts_players"
    __table_args__ = (
        Index("ix_darts_players_pdc_id", "pdc_id"),
        Index("ix_darts_players_dartsorakel_key", "dartsorakel_key"),
        Index("ix_darts_players_slug", "slug"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Display / search fields
    first_name: Mapped[str] = mapped_column(String(120), nullable=False)
    last_name: Mapped[str] = mapped_column(String(120), nullable=False)
    nickname: Mapped[Optional[str]] = mapped_column(String(120))
    slug: Mapped[Optional[str]] = mapped_column(String(200), unique=True)

    # Source system IDs (used by entity_resolution)
    pdc_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    dartsorakel_key: Mapped[Optional[int]] = mapped_column(BigInteger)
    dartsdatabase_id: Mapped[Optional[str]] = mapped_column(String(50))
    dartconnect_id: Mapped[Optional[str]] = mapped_column(String(50))
    wdf_id: Mapped[Optional[str]] = mapped_column(String(50))

    # Source confidence (set by entity_resolution)
    source_confidence: Mapped[Optional[float]] = mapped_column(Float)
    primary_source: Mapped[Optional[str]] = mapped_column(String(30))

    # GDPR-sensitive fields
    dob: Mapped[Optional[date]] = mapped_column(Date)
    country_code: Mapped[Optional[str]] = mapped_column(String(10))
    gdpr_anonymized: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    gdpr_anonymized_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Ranking / tour data
    pdc_ranking: Mapped[Optional[int]] = mapped_column(Integer)
    prize_money: Mapped[Optional[int]] = mapped_column(BigInteger)
    tour_card_holder: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    tour_card_years: Mapped[Optional[list]] = mapped_column(JSONB)

    # DartsOrakel 3DA stat (3-dart average, the primary stat)
    dartsorakel_3da: Mapped[Optional[float]] = mapped_column(Float)
    dartsorakel_rank: Mapped[Optional[int]] = mapped_column(Integer)

    # Audit
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    elo_ratings: Mapped[list["DartsEloRating"]] = relationship(
        "DartsEloRating", back_populates="player", lazy="noload"
    )
    stats: Mapped[list["DartsPlayerStats"]] = relationship(
        "DartsPlayerStats", back_populates="player", lazy="noload"
    )
    coverage_regimes: Mapped[list["DartsCoverageRegime"]] = relationship(
        "DartsCoverageRegime", back_populates="player", lazy="noload"
    )


# ---------------------------------------------------------------------------
# darts_competitions
# ---------------------------------------------------------------------------

class DartsCompetition(Base):
    """
    Tournament / competition record.

    The ``format_code`` references the :mod:`competition.format_registry`
    and ``format_era_code`` holds the era-specific variant where applicable
    (e.g. ``PDC_WC_ERA_2020`` vs ``PDC_WC``).
    """

    __tablename__ = "darts_competitions"
    __table_args__ = (
        Index("ix_darts_competitions_pdc_tournament_id", "pdc_tournament_id"),
        Index("ix_darts_competitions_season_year", "season_year"),
        Index("ix_darts_competitions_format_code", "format_code"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    pdc_tournament_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    sport_radar_tournament_id: Mapped[Optional[str]] = mapped_column(String(50))
    dartconnect_id: Mapped[Optional[str]] = mapped_column(String(50))

    name: Mapped[str] = mapped_column(String(300), nullable=False)
    season_year: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    format_code: Mapped[str] = mapped_column(String(50), nullable=False)
    format_era_code: Mapped[Optional[str]] = mapped_column(String(50))
    organiser: Mapped[str] = mapped_column(String(30), nullable=False)
    ecosystem: Mapped[str] = mapped_column(String(30), nullable=False)

    venue: Mapped[Optional[str]] = mapped_column(String(200))
    city: Mapped[Optional[str]] = mapped_column(String(100))

    start_date: Mapped[Optional[date]] = mapped_column(Date)
    end_date: Mapped[Optional[date]] = mapped_column(Date)

    field_size: Mapped[Optional[int]] = mapped_column(SmallInteger)
    is_ranked: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_televised: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    winner_player_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_players.id")
    )

    # Extra metadata (prize pool, surface, sponsor, etc.)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    matches: Mapped[list["DartsMatch"]] = relationship(
        "DartsMatch", back_populates="competition", lazy="noload"
    )


# ---------------------------------------------------------------------------
# darts_matches
# ---------------------------------------------------------------------------

class DartsMatch(Base):
    """
    Individual match record.

    ``starter_confidence`` records the probability assigned by the starter
    inference engine that player 1 threw first.  ``visit_data_coverage``
    is a fraction [0, 1] indicating how many visits were captured.
    """

    __tablename__ = "darts_matches"
    __table_args__ = (
        Index("ix_darts_matches_competition_id", "competition_id"),
        Index("ix_darts_matches_player1_id", "player1_id"),
        Index("ix_darts_matches_player2_id", "player2_id"),
        Index("ix_darts_matches_match_date", "match_date"),
        Index("ix_darts_matches_pdc_fixture_id", "pdc_fixture_id"),
        Index("ix_darts_matches_status", "status"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    pdc_fixture_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    sport_radar_id: Mapped[Optional[str]] = mapped_column(String(50))
    dartconnect_match_id: Mapped[Optional[str]] = mapped_column(String(50))

    competition_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_competitions.id"), nullable=False
    )
    player1_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_players.id")
    )
    player2_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_players.id")
    )

    round_name: Mapped[str] = mapped_column(String(100), nullable=False)
    match_date: Mapped[Optional[date]] = mapped_column(Date)
    match_time: Mapped[Optional[str]] = mapped_column(String(8))

    # Status: "Fixture" | "Live" | "Completed" | "Cancelled" | "Postponed"
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="Fixture")

    # Scores
    player1_score: Mapped[Optional[int]] = mapped_column(SmallInteger)
    player2_score: Mapped[Optional[int]] = mapped_column(SmallInteger)
    result_type: Mapped[Optional[str]] = mapped_column(String(10))
    winner_player_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_players.id")
    )

    # Data quality / coverage
    starter_player_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_players.id")
    )
    starter_confidence: Mapped[Optional[float]] = mapped_column(Float)
    visit_data_coverage: Mapped[Optional[float]] = mapped_column(Float)
    coverage_regime: Mapped[Optional[str]] = mapped_column(String(5))

    # Raw source data snapshot (enables re-ingestion without re-scraping)
    raw_source_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    source_name: Mapped[Optional[str]] = mapped_column(String(30))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    competition: Mapped["DartsCompetition"] = relationship(
        "DartsCompetition", back_populates="matches", lazy="noload"
    )
    legs: Mapped[list["DartsLeg"]] = relationship(
        "DartsLeg", back_populates="match", lazy="noload"
    )


# ---------------------------------------------------------------------------
# darts_legs
# ---------------------------------------------------------------------------

class DartsLeg(Base):
    """Individual leg within a match."""

    __tablename__ = "darts_legs"
    __table_args__ = (
        Index("ix_darts_legs_match_id", "match_id"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    match_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_matches.id"), nullable=False
    )

    leg_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    set_number: Mapped[Optional[int]] = mapped_column(SmallInteger)

    # Who threw first this leg
    starter_player_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_players.id")
    )
    starter_is_player1: Mapped[Optional[bool]] = mapped_column(Boolean)

    winner_player_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_players.id")
    )

    # Leg statistics
    total_visits: Mapped[Optional[int]] = mapped_column(SmallInteger)
    winning_checkout: Mapped[Optional[int]] = mapped_column(SmallInteger)
    winning_checkout_darts: Mapped[Optional[int]] = mapped_column(SmallInteger)
    had_180: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    nine_darter: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # DartConnect raw data
    dartconnect_leg_id: Mapped[Optional[str]] = mapped_column(String(50))

    match: Mapped["DartsMatch"] = relationship(
        "DartsMatch", back_populates="legs", lazy="noload"
    )
    visits: Mapped[list["DartsVisit"]] = relationship(
        "DartsVisit", back_populates="leg", lazy="noload"
    )


# ---------------------------------------------------------------------------
# darts_visits
# ---------------------------------------------------------------------------

class DartsVisit(Base):
    """
    A single 3-dart visit within a leg.

    ``score`` is the value scored this visit (0–180).
    ``remaining_before`` and ``remaining_after`` track the checkout count.
    Checkout information is captured in ``checkout_attempted``,
    ``checkout_hit``, and ``checkout_dart_count``.
    """

    __tablename__ = "darts_visits"
    __table_args__ = (
        Index("ix_darts_visits_leg_id", "leg_id"),
        Index("ix_darts_visits_player_id", "player_id"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    leg_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_legs.id"), nullable=False
    )
    player_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_players.id")
    )

    visit_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    score: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    remaining_before: Mapped[Optional[int]] = mapped_column(SmallInteger)
    remaining_after: Mapped[Optional[int]] = mapped_column(SmallInteger)

    dart1: Mapped[Optional[str]] = mapped_column(String(5))
    dart2: Mapped[Optional[str]] = mapped_column(String(5))
    dart3: Mapped[Optional[str]] = mapped_column(String(5))

    checkout_attempted: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    checkout_hit: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    checkout_dart_count: Mapped[Optional[int]] = mapped_column(SmallInteger)

    leg: Mapped["DartsLeg"] = relationship(
        "DartsLeg", back_populates="visits", lazy="noload"
    )


# ---------------------------------------------------------------------------
# darts_player_stats
# ---------------------------------------------------------------------------

class DartsPlayerStats(Base):
    """
    Aggregated per-player statistics, scoped to a competition season.

    Populated by the ingest pipeline and updated incrementally as new
    match data arrives.  All rate stats are stored as floats in [0, 1]
    unless otherwise noted.
    """

    __tablename__ = "darts_player_stats"
    __table_args__ = (
        UniqueConstraint(
            "player_id", "competition_id", "source",
            name="uq_player_stats_player_comp_source",
        ),
        Index("ix_darts_player_stats_player_id", "player_id"),
        Index("ix_darts_player_stats_competition_id", "competition_id"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    player_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_players.id"), nullable=False
    )
    competition_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_competitions.id")
    )
    source: Mapped[str] = mapped_column(String(30), nullable=False)
    stat_season_year: Mapped[Optional[int]] = mapped_column(SmallInteger)

    # Core stats
    three_dart_average: Mapped[Optional[float]] = mapped_column(Float)
    first_nine_average: Mapped[Optional[float]] = mapped_column(Float)
    checkout_percentage: Mapped[Optional[float]] = mapped_column(Float)
    hold_rate: Mapped[Optional[float]] = mapped_column(Float)
    break_rate: Mapped[Optional[float]] = mapped_column(Float)

    # 180 and high-finish stats
    count_180: Mapped[Optional[int]] = mapped_column(Integer)
    count_180_per_leg: Mapped[Optional[float]] = mapped_column(Float)
    high_finish_count: Mapped[Optional[int]] = mapped_column(Integer)

    # Match record
    matches_played: Mapped[Optional[int]] = mapped_column(Integer)
    matches_won: Mapped[Optional[int]] = mapped_column(Integer)
    legs_played: Mapped[Optional[int]] = mapped_column(Integer)
    legs_won: Mapped[Optional[int]] = mapped_column(Integer)

    # Visit-level counts (from DartConnect / Mastercaller)
    visits_counted: Mapped[Optional[int]] = mapped_column(Integer)

    # DartsOrakel fields
    dartsorakel_sumfield1: Mapped[Optional[int]] = mapped_column(BigInteger)
    dartsorakel_sumfield2: Mapped[Optional[int]] = mapped_column(BigInteger)

    stat_date: Mapped[Optional[date]] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    player: Mapped["DartsPlayer"] = relationship(
        "DartsPlayer", back_populates="stats", lazy="noload"
    )


# ---------------------------------------------------------------------------
# darts_coverage_regimes
# ---------------------------------------------------------------------------

class DartsCoverageRegime(Base):
    """
    Per-player, per-competition data coverage regime.

    ``regime`` is one of:
    - ``"R0"`` – only result data available (win/loss, no visit data)
    - ``"R1"`` – match-level stats available (3DA, checkout%) but no visit detail
    - ``"R2"`` – full visit-level data available (DartConnect or Mastercaller)

    The regime determines which model tier (logit / LightGBM / stacking) is
    used for pricing.
    """

    __tablename__ = "darts_coverage_regimes"
    __table_args__ = (
        UniqueConstraint(
            "player_id", "competition_id",
            name="uq_coverage_regime_player_comp",
        ),
        Index("ix_darts_coverage_regimes_player_id", "player_id"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    player_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_players.id"), nullable=False
    )
    competition_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_competitions.id")
    )

    regime: Mapped[str] = mapped_column(String(5), nullable=False)
    regime_score: Mapped[Optional[float]] = mapped_column(Float)

    # Flags indicating which data sources are available
    has_visit_data: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_match_stats: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_result_only: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    has_dartsorakel: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_dartconnect: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    last_computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    player: Mapped["DartsPlayer"] = relationship(
        "DartsPlayer", back_populates="coverage_regimes", lazy="noload"
    )


# ---------------------------------------------------------------------------
# darts_elo_ratings
# ---------------------------------------------------------------------------

class DartsEloRating(Base):
    """
    Point-in-time ELO rating snapshot for a player in a pool.

    A new row is inserted after every match that changes the rating.
    The most recent row per (player_id, pool) is the current rating.
    """

    __tablename__ = "darts_elo_ratings"
    __table_args__ = (
        Index("ix_darts_elo_ratings_player_id", "player_id"),
        Index("ix_darts_elo_ratings_pool", "pool"),
        Index("ix_darts_elo_ratings_match_date", "match_date"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    player_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_players.id"), nullable=False
    )
    pool: Mapped[str] = mapped_column(String(30), nullable=False)
    match_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_matches.id")
    )

    rating_before: Mapped[float] = mapped_column(Float, nullable=False)
    rating_after: Mapped[float] = mapped_column(Float, nullable=False)
    delta: Mapped[float] = mapped_column(Float, nullable=False)
    k_factor: Mapped[float] = mapped_column(Float, nullable=False)
    expected_score: Mapped[float] = mapped_column(Float, nullable=False)
    actual_score: Mapped[float] = mapped_column(Float, nullable=False)
    games_played_at_time: Mapped[int] = mapped_column(Integer, nullable=False)

    match_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    player: Mapped["DartsPlayer"] = relationship(
        "DartsPlayer", back_populates="elo_ratings", lazy="noload"
    )


# ---------------------------------------------------------------------------
# darts_gdpr_consents
# ---------------------------------------------------------------------------

class DartsGdprConsent(Base):
    """
    GDPR consent record for a player.

    Tracks whether a player has consented to personal data processing.
    Separate from the player record so consent status can be audited
    independently.
    """

    __tablename__ = "darts_gdpr_consents"
    __table_args__ = (
        Index("ix_darts_gdpr_consents_player_id", "player_id"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    player_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("darts_players.id"), nullable=False, unique=True
    )

    consent_given: Mapped[bool] = mapped_column(Boolean, nullable=False)
    consent_given_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    consent_withdrawn_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    consent_version: Mapped[str] = mapped_column(String(20), nullable=False)
    consent_method: Mapped[Optional[str]] = mapped_column(String(50))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
