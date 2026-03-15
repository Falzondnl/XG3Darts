"""Liability management + trader override tables.

Revision ID: 004_liability_and_trader_overrides
Revises: 003_ml_model_artifacts
Create Date: 2026-03-15

Creates:
  - darts_liability_limits        — per-market exposure ceiling configuration
  - darts_bet_exposure            — individual bet records for live exposure tracking
  - darts_trader_overrides        — manual trader actions (suspend/widen/pull)
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:

    # ------------------------------------------------------------------
    # darts_liability_limits
    # Per-market-type exposure ceiling configuration.
    # ------------------------------------------------------------------
    op.create_table(
        "darts_liability_limits",
        sa.Column("id", sa.String(36), primary_key=True),
        # Scope — either global (competition_id IS NULL) or per-competition
        sa.Column("competition_id", sa.String(36), nullable=True),
        sa.Column("match_id", sa.String(36), nullable=True),
        # Market identification
        sa.Column(
            "market_family",
            sa.String(50),
            nullable=False,
            comment="match_winner | leg_handicap | totals | exact_score | props_180 | props_checkout | outright",
        ),
        # Limit values (all in the operator's base currency, e.g. EUR cents)
        sa.Column(
            "max_exposure_per_outcome",
            sa.BigInteger,
            nullable=False,
            comment="Maximum net liability on any single outcome (currency minor units)",
        ),
        sa.Column(
            "max_exposure_per_match",
            sa.BigInteger,
            nullable=False,
            comment="Maximum net liability summed across all outcomes for one match",
        ),
        sa.Column(
            "max_single_stake",
            sa.BigInteger,
            nullable=False,
            comment="Maximum single bet stake accepted",
        ),
        sa.Column(
            "auto_suspend_threshold",
            sa.Float,
            nullable=False,
            server_default="0.9",
            comment="Fraction of max_exposure_per_outcome at which market auto-suspends [0,1]",
        ),
        # Audit
        sa.Column("created_by", sa.String(100), nullable=True),
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
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
    )
    op.create_index(
        "ix_liability_limits_competition_market",
        "darts_liability_limits",
        ["competition_id", "market_family"],
    )
    op.create_index(
        "ix_liability_limits_match_market",
        "darts_liability_limits",
        ["match_id", "market_family"],
    )

    # ------------------------------------------------------------------
    # darts_bet_exposure
    # Individual accepted bets for real-time exposure tracking.
    # ------------------------------------------------------------------
    op.create_table(
        "darts_bet_exposure",
        sa.Column("id", sa.String(36), primary_key=True),
        # External bet reference from the B2B operator
        sa.Column("external_bet_id", sa.String(100), nullable=False, unique=True),
        sa.Column("match_id", sa.String(36), nullable=False),
        sa.Column("competition_id", sa.String(36), nullable=True),
        sa.Column(
            "market_family",
            sa.String(50),
            nullable=False,
        ),
        sa.Column(
            "market_key",
            sa.String(200),
            nullable=False,
            comment="e.g. 'match_winner:p1' | 'totals:over:8.5' | 'exact_score:6-4'",
        ),
        # Financials (currency minor units, e.g. EUR cents)
        sa.Column("stake", sa.BigInteger, nullable=False),
        sa.Column("odds_decimal", sa.Float, nullable=False),
        sa.Column(
            "potential_payout",
            sa.BigInteger,
            nullable=False,
            comment="stake * odds_decimal rounded to minor units",
        ),
        sa.Column(
            "potential_loss",
            sa.BigInteger,
            nullable=False,
            comment="potential_payout - stake (operator net liability if bet wins)",
        ),
        # Bet lifecycle
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="open",
            comment="open | settled_win | settled_lose | voided | cancelled",
        ),
        sa.Column(
            "placed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        # Match state snapshot when bet was placed (for audit)
        sa.Column(
            "state_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_bet_exposure_match_market",
        "darts_bet_exposure",
        ["match_id", "market_family", "status"],
    )
    op.create_index(
        "ix_bet_exposure_external_id",
        "darts_bet_exposure",
        ["external_bet_id"],
    )

    # ------------------------------------------------------------------
    # darts_trader_overrides
    # Manual trader actions: suspend markets, widen margins, pull matches.
    # ------------------------------------------------------------------
    op.create_table(
        "darts_trader_overrides",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("match_id", sa.String(36), nullable=True, comment="NULL = applies to all matches in competition"),
        sa.Column("competition_id", sa.String(36), nullable=True),
        sa.Column(
            "scope",
            sa.String(20),
            nullable=False,
            comment="match | competition | global",
        ),
        sa.Column(
            "action",
            sa.String(30),
            nullable=False,
            comment="SUSPEND | RESUME | WIDEN_MARGIN | PULL | RESTRICT_STAKES",
        ),
        sa.Column(
            "market_family",
            sa.String(50),
            nullable=True,
            comment="NULL = all markets for this match",
        ),
        # Action parameters (varies by action type)
        sa.Column(
            "params",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'"),
            comment=(
                "WIDEN_MARGIN: {extra_margin_pct: 5.0} | "
                "RESTRICT_STAKES: {max_stake: 500} | "
                "SUSPEND: {reason: 'injury'}"
            ),
        ),
        # Reason (displayed in audit log)
        sa.Column("reason", sa.String(500), nullable=True),
        sa.Column(
            "reason_code",
            sa.String(50),
            nullable=True,
            comment="INJURY | WITHDRAWAL | NEWS | SUSPICIOUS_BETTING | TECHNICAL | OTHER",
        ),
        # Validity window
        sa.Column(
            "valid_from",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "valid_until",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="NULL = active until manually revoked",
        ),
        # Status
        sa.Column(
            "is_active",
            sa.Boolean,
            nullable=False,
            server_default="true",
        ),
        # Audit
        sa.Column("created_by", sa.String(100), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("revoked_by", sa.String(100), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_trader_overrides_match_active",
        "darts_trader_overrides",
        ["match_id", "is_active"],
    )
    op.create_index(
        "ix_trader_overrides_competition_active",
        "darts_trader_overrides",
        ["competition_id", "is_active"],
    )


def downgrade() -> None:
    op.drop_table("darts_trader_overrides")
    op.drop_table("darts_bet_exposure")
    op.drop_table("darts_liability_limits")
