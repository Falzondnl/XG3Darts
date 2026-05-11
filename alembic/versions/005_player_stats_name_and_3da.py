"""Add player_name + three_da_pdc to darts_player_stats for name-keyed 3DA lookup.

Revision ID: 005_player_stats_name_and_3da
Revises: 004
Create Date: 2026-05-11

Problem: _lookup_3da() in predict.py queries darts_player_stats by player_name
but that column does not exist.  The table has no rows and no player_name column.
This means every predict call returns neutral_default 3DA = 50.0 regardless of
whether player stats are available.

Fix:
  1. Add player_name VARCHAR(200) with a unique index to darts_player_stats
     (mirrors the design of darts_elo_ratings which also uses player_name directly).
  2. Add three_da_pdc FLOAT — the PDC three-dart average to be queried by
     _lookup_3da().  Mirrors the feature builder's three_da_pdc key name exactly.
  3. Add checkout_pct_pdc FLOAT — PDC checkout percentage (also consumed by
     the feature builder as checkout_pct_pdc).

No data migrations in this file — data is seeded by scripts/seed_pdc_player_stats.py.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision: str = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add player_name column — nullable first so existing (empty) rows are fine.
    op.add_column(
        "darts_player_stats",
        sa.Column("player_name", sa.String(200), nullable=True),
    )
    op.add_column(
        "darts_player_stats",
        sa.Column("three_da_pdc", sa.Float(), nullable=True),
    )
    op.add_column(
        "darts_player_stats",
        sa.Column("checkout_pct_pdc", sa.Float(), nullable=True),
    )

    # Unique index on player_name so the lookup query hits index and prevents
    # duplicates from concurrent seed runs.
    op.create_index(
        "ix_darts_player_stats_player_name",
        "darts_player_stats",
        ["player_name"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_darts_player_stats_player_name", table_name="darts_player_stats")
    op.drop_column("darts_player_stats", "checkout_pct_pdc")
    op.drop_column("darts_player_stats", "three_da_pdc")
    op.drop_column("darts_player_stats", "player_name")
