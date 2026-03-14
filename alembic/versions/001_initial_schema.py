"""Initial schema — full XG3 Darts database.

Revision ID: 001_initial_schema
Revises: (none — initial migration)
Create Date: 2026-03-14

Creates all core tables for the XG3 Darts microservice:
  - darts_players           (GDPR, source confidence)
  - darts_competitions      (format versioning)
  - darts_matches           (starter confidence, visit_data_coverage)
  - darts_legs
  - darts_visits
  - darts_player_stats
  - darts_coverage_regimes
  - darts_elo_ratings
  - darts_gdpr_consents

Design decisions:
  - All primary keys are UUIDs stored as VARCHAR(36) for portability.
  - GDPR-sensitive fields are in the main player table but controlled
    by the ``gdpr_anonymized`` flag and separate consent records.
  - visit_data_coverage is a float [0, 1] indicating how many visits
    have full DartConnect/Mastercaller data.
  - starter_confidence is the probability that player1 threw first,
    used to widen margin when uncertain.
  - format_code is a string reference to the format_registry — the
    database stores the code only; format details are resolved in code.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

revision: str = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


# ---------------------------------------------------------------------------
# Upgrade
# ---------------------------------------------------------------------------


def upgrade() -> None:
    """Create all tables for the initial schema."""

    # ------------------------------------------------------------------
    # darts_players
    # ------------------------------------------------------------------
    op.create_table(
        "darts_players",
        sa.Column("id", sa.String(36), primary_key=True),
        # Display / search
        sa.Column("first_name", sa.String(120), nullable=False),
        sa.Column("last_name", sa.String(120), nullable=False),
        sa.Column("nickname", sa.String(120), nullable=True),
        sa.Column("slug", sa.String(200), nullable=True, unique=True),
        # Source system IDs
        sa.Column("pdc_id", sa.BigInteger, nullable=True),
        sa.Column("dartsorakel_key", sa.BigInteger, nullable=True),
        sa.Column("dartsdatabase_id", sa.String(50), nullable=True),
        sa.Column("dartconnect_id", sa.String(50), nullable=True),
        sa.Column("wdf_id", sa.String(50), nullable=True),
        # Source confidence
        sa.Column("source_confidence", sa.Float, nullable=True),
        sa.Column("primary_source", sa.String(30), nullable=True),
        # GDPR-sensitive fields
        sa.Column("dob", sa.Date, nullable=True),
        sa.Column("country_code", sa.String(10), nullable=True),
        sa.Column("gdpr_anonymized", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "gdpr_anonymized_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        # Ranking / tour data
        sa.Column("pdc_ranking", sa.Integer, nullable=True),
        sa.Column("prize_money", sa.BigInteger, nullable=True),
        sa.Column("tour_card_holder", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "tour_card_years",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        # DartsOrakel stats
        sa.Column("dartsorakel_3da", sa.Float, nullable=True),
        sa.Column("dartsorakel_rank", sa.Integer, nullable=True),
        # Audit
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("ix_darts_players_pdc_id", "darts_players", ["pdc_id"])
    op.create_index(
        "ix_darts_players_dartsorakel_key", "darts_players", ["dartsorakel_key"]
    )
    op.create_index("ix_darts_players_slug", "darts_players", ["slug"])

    # ------------------------------------------------------------------
    # darts_competitions
    # ------------------------------------------------------------------
    op.create_table(
        "darts_competitions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("pdc_tournament_id", sa.BigInteger, nullable=True),
        sa.Column("sport_radar_tournament_id", sa.String(50), nullable=True),
        sa.Column("dartconnect_id", sa.String(50), nullable=True),
        sa.Column("name", sa.String(300), nullable=False),
        sa.Column("season_year", sa.SmallInteger, nullable=False),
        # format_code references competition.format_registry — no FK (registry in code)
        sa.Column("format_code", sa.String(50), nullable=False),
        sa.Column("format_era_code", sa.String(50), nullable=True),
        sa.Column("organiser", sa.String(30), nullable=False),
        sa.Column("ecosystem", sa.String(30), nullable=False),
        sa.Column("venue", sa.String(200), nullable=True),
        sa.Column("city", sa.String(100), nullable=True),
        sa.Column("start_date", sa.Date, nullable=True),
        sa.Column("end_date", sa.Date, nullable=True),
        sa.Column("field_size", sa.SmallInteger, nullable=True),
        sa.Column("is_ranked", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("is_televised", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "winner_player_id",
            sa.String(36),
            sa.ForeignKey("darts_players.id"),
            nullable=True,
        ),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "ix_darts_competitions_pdc_tournament_id",
        "darts_competitions",
        ["pdc_tournament_id"],
    )
    op.create_index(
        "ix_darts_competitions_season_year",
        "darts_competitions",
        ["season_year"],
    )
    op.create_index(
        "ix_darts_competitions_format_code",
        "darts_competitions",
        ["format_code"],
    )

    # ------------------------------------------------------------------
    # darts_matches
    # ------------------------------------------------------------------
    op.create_table(
        "darts_matches",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("pdc_fixture_id", sa.BigInteger, nullable=True),
        sa.Column("sport_radar_id", sa.String(50), nullable=True),
        sa.Column("dartconnect_match_id", sa.String(50), nullable=True),
        sa.Column(
            "competition_id",
            sa.String(36),
            sa.ForeignKey("darts_competitions.id"),
            nullable=False,
        ),
        sa.Column(
            "player1_id",
            sa.String(36),
            sa.ForeignKey("darts_players.id"),
            nullable=True,
        ),
        sa.Column(
            "player2_id",
            sa.String(36),
            sa.ForeignKey("darts_players.id"),
            nullable=True,
        ),
        sa.Column("round_name", sa.String(100), nullable=False),
        sa.Column("match_date", sa.Date, nullable=True),
        sa.Column("match_time", sa.String(8), nullable=True),
        # Status: Fixture | Live | Completed | Cancelled | Postponed
        sa.Column("status", sa.String(30), nullable=False, server_default="'Fixture'"),
        # Scores
        sa.Column("player1_score", sa.SmallInteger, nullable=True),
        sa.Column("player2_score", sa.SmallInteger, nullable=True),
        sa.Column("result_type", sa.String(10), nullable=True),
        sa.Column(
            "winner_player_id",
            sa.String(36),
            sa.ForeignKey("darts_players.id"),
            nullable=True,
        ),
        # Data quality / coverage
        sa.Column(
            "starter_player_id",
            sa.String(36),
            sa.ForeignKey("darts_players.id"),
            nullable=True,
        ),
        # starter_confidence: P(player1 threw first) [0, 1]
        # Required per G5 QA gate — always set, never NULL after computation
        sa.Column("starter_confidence", sa.Float, nullable=True),
        # visit_data_coverage: fraction of visits with full DartConnect data
        sa.Column("visit_data_coverage", sa.Float, nullable=True),
        # coverage_regime: 'R0' | 'R1' | 'R2'
        sa.Column("coverage_regime", sa.String(5), nullable=True),
        # Raw source data snapshot
        sa.Column(
            "raw_source_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("source_name", sa.String(30), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "ix_darts_matches_competition_id", "darts_matches", ["competition_id"]
    )
    op.create_index("ix_darts_matches_player1_id", "darts_matches", ["player1_id"])
    op.create_index("ix_darts_matches_player2_id", "darts_matches", ["player2_id"])
    op.create_index("ix_darts_matches_match_date", "darts_matches", ["match_date"])
    op.create_index(
        "ix_darts_matches_pdc_fixture_id", "darts_matches", ["pdc_fixture_id"]
    )
    op.create_index("ix_darts_matches_status", "darts_matches", ["status"])

    # ------------------------------------------------------------------
    # darts_legs
    # ------------------------------------------------------------------
    op.create_table(
        "darts_legs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "match_id",
            sa.String(36),
            sa.ForeignKey("darts_matches.id"),
            nullable=False,
        ),
        sa.Column("leg_number", sa.SmallInteger, nullable=False),
        sa.Column("set_number", sa.SmallInteger, nullable=True),
        sa.Column(
            "starter_player_id",
            sa.String(36),
            sa.ForeignKey("darts_players.id"),
            nullable=True,
        ),
        sa.Column("starter_is_player1", sa.Boolean, nullable=True),
        sa.Column(
            "winner_player_id",
            sa.String(36),
            sa.ForeignKey("darts_players.id"),
            nullable=True,
        ),
        sa.Column("total_visits", sa.SmallInteger, nullable=True),
        sa.Column("winning_checkout", sa.SmallInteger, nullable=True),
        sa.Column("winning_checkout_darts", sa.SmallInteger, nullable=True),
        sa.Column("had_180", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("nine_darter", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("dartconnect_leg_id", sa.String(50), nullable=True),
    )
    op.create_index("ix_darts_legs_match_id", "darts_legs", ["match_id"])

    # ------------------------------------------------------------------
    # darts_visits
    # ------------------------------------------------------------------
    op.create_table(
        "darts_visits",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "leg_id",
            sa.String(36),
            sa.ForeignKey("darts_legs.id"),
            nullable=False,
        ),
        sa.Column(
            "player_id",
            sa.String(36),
            sa.ForeignKey("darts_players.id"),
            nullable=True,
        ),
        sa.Column("visit_number", sa.SmallInteger, nullable=False),
        sa.Column("score", sa.SmallInteger, nullable=False),
        sa.Column("remaining_before", sa.SmallInteger, nullable=True),
        sa.Column("remaining_after", sa.SmallInteger, nullable=True),
        sa.Column("dart1", sa.String(5), nullable=True),
        sa.Column("dart2", sa.String(5), nullable=True),
        sa.Column("dart3", sa.String(5), nullable=True),
        sa.Column("checkout_attempted", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("checkout_hit", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("checkout_dart_count", sa.SmallInteger, nullable=True),
    )
    op.create_index("ix_darts_visits_leg_id", "darts_visits", ["leg_id"])
    op.create_index("ix_darts_visits_player_id", "darts_visits", ["player_id"])

    # ------------------------------------------------------------------
    # darts_player_stats
    # ------------------------------------------------------------------
    op.create_table(
        "darts_player_stats",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "player_id",
            sa.String(36),
            sa.ForeignKey("darts_players.id"),
            nullable=False,
        ),
        sa.Column(
            "competition_id",
            sa.String(36),
            sa.ForeignKey("darts_competitions.id"),
            nullable=True,
        ),
        sa.Column("source", sa.String(30), nullable=False),
        sa.Column("stat_season_year", sa.SmallInteger, nullable=True),
        # Core stats
        sa.Column("three_dart_average", sa.Float, nullable=True),
        sa.Column("first_nine_average", sa.Float, nullable=True),
        sa.Column("checkout_percentage", sa.Float, nullable=True),
        sa.Column("hold_rate", sa.Float, nullable=True),
        sa.Column("break_rate", sa.Float, nullable=True),
        # 180 / high finish
        sa.Column("count_180", sa.Integer, nullable=True),
        sa.Column("count_180_per_leg", sa.Float, nullable=True),
        sa.Column("high_finish_count", sa.Integer, nullable=True),
        # Match record
        sa.Column("matches_played", sa.Integer, nullable=True),
        sa.Column("matches_won", sa.Integer, nullable=True),
        sa.Column("legs_played", sa.Integer, nullable=True),
        sa.Column("legs_won", sa.Integer, nullable=True),
        sa.Column("visits_counted", sa.Integer, nullable=True),
        # DartsOrakel
        sa.Column("dartsorakel_sumfield1", sa.BigInteger, nullable=True),
        sa.Column("dartsorakel_sumfield2", sa.BigInteger, nullable=True),
        sa.Column("stat_date", sa.Date, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.UniqueConstraint(
            "player_id", "competition_id", "source",
            name="uq_player_stats_player_comp_source",
        ),
    )
    op.create_index(
        "ix_darts_player_stats_player_id", "darts_player_stats", ["player_id"]
    )
    op.create_index(
        "ix_darts_player_stats_competition_id", "darts_player_stats", ["competition_id"]
    )

    # ------------------------------------------------------------------
    # darts_coverage_regimes
    # ------------------------------------------------------------------
    op.create_table(
        "darts_coverage_regimes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "player_id",
            sa.String(36),
            sa.ForeignKey("darts_players.id"),
            nullable=False,
        ),
        sa.Column(
            "competition_id",
            sa.String(36),
            sa.ForeignKey("darts_competitions.id"),
            nullable=True,
        ),
        # regime: 'R0' | 'R1' | 'R2' — loaded from this table per G8
        sa.Column("regime", sa.String(5), nullable=False),
        sa.Column("regime_score", sa.Float, nullable=True),
        # Data availability flags
        sa.Column("has_visit_data", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("has_match_stats", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("has_result_only", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("has_dartsorakel", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("has_dartconnect", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "last_computed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.UniqueConstraint(
            "player_id", "competition_id",
            name="uq_coverage_regime_player_comp",
        ),
    )
    op.create_index(
        "ix_darts_coverage_regimes_player_id", "darts_coverage_regimes", ["player_id"]
    )

    # ------------------------------------------------------------------
    # darts_elo_ratings
    # ------------------------------------------------------------------
    op.create_table(
        "darts_elo_ratings",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "player_id",
            sa.String(36),
            sa.ForeignKey("darts_players.id"),
            nullable=False,
        ),
        sa.Column("pool", sa.String(30), nullable=False),
        sa.Column(
            "match_id",
            sa.String(36),
            sa.ForeignKey("darts_matches.id"),
            nullable=True,
        ),
        sa.Column("rating_before", sa.Float, nullable=False),
        sa.Column("rating_after", sa.Float, nullable=False),
        sa.Column("delta", sa.Float, nullable=False),
        sa.Column("k_factor", sa.Float, nullable=False),
        sa.Column("expected_score", sa.Float, nullable=False),
        sa.Column("actual_score", sa.Float, nullable=False),
        sa.Column("games_played_at_time", sa.Integer, nullable=False),
        sa.Column("match_date", sa.Date, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "ix_darts_elo_ratings_player_id", "darts_elo_ratings", ["player_id"]
    )
    op.create_index("ix_darts_elo_ratings_pool", "darts_elo_ratings", ["pool"])
    op.create_index(
        "ix_darts_elo_ratings_match_date", "darts_elo_ratings", ["match_date"]
    )

    # ------------------------------------------------------------------
    # darts_gdpr_consents
    # ------------------------------------------------------------------
    op.create_table(
        "darts_gdpr_consents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "player_id",
            sa.String(36),
            sa.ForeignKey("darts_players.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("consent_given", sa.Boolean, nullable=False),
        sa.Column("consent_given_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consent_withdrawn_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consent_version", sa.String(20), nullable=False),
        sa.Column("consent_method", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "ix_darts_gdpr_consents_player_id", "darts_gdpr_consents", ["player_id"]
    )


# ---------------------------------------------------------------------------
# Downgrade
# ---------------------------------------------------------------------------


def downgrade() -> None:
    """Drop all tables in reverse dependency order."""
    op.drop_table("darts_gdpr_consents")
    op.drop_table("darts_elo_ratings")
    op.drop_table("darts_coverage_regimes")
    op.drop_table("darts_player_stats")
    op.drop_table("darts_visits")
    op.drop_table("darts_legs")
    op.drop_table("darts_matches")
    op.drop_table("darts_competitions")
    op.drop_table("darts_players")
