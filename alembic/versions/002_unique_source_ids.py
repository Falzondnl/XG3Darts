"""Add unique constraints on pdc_id and pdc_tournament_id.

Revision ID: 002_unique_source_ids
Revises: 001_initial_schema
Create Date: 2026-03-14

Allows ON CONFLICT upserts in bulk_ingest.py and prevents duplicate
source-system records from accumulating through re-ingestion.
"""
from __future__ import annotations

from alembic import op

revision: str = "002_unique_source_ids"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Players — unique pdc_id
    op.create_unique_constraint(
        "uq_darts_players_pdc_id",
        "darts_players",
        ["pdc_id"],
    )
    # Competitions — unique pdc_tournament_id
    op.create_unique_constraint(
        "uq_darts_competitions_pdc_tournament_id",
        "darts_competitions",
        ["pdc_tournament_id"],
    )
    # Matches — unique pdc_fixture_id (nullable, so exclude NULLs via partial index)
    op.execute(
        """
        CREATE UNIQUE INDEX uq_darts_matches_pdc_fixture_id
        ON darts_matches (pdc_fixture_id)
        WHERE pdc_fixture_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_darts_matches_pdc_fixture_id")
    op.drop_constraint("uq_darts_competitions_pdc_tournament_id", "darts_competitions")
    op.drop_constraint("uq_darts_players_pdc_id", "darts_players")
