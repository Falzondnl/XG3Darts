"""003 — ML model artifact storage table

Revision ID: 003
Revises: 002
Create Date: 2026-03-14
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "darts_ml_model_artifacts",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("version", sa.String(32), nullable=False, server_default="1"),
        sa.Column("artifact", sa.LargeBinary(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("feature_count", sa.Integer(), nullable=True),
        sa.Column("brier_test", sa.Float(), nullable=True),
        sa.Column("auc_test", sa.Float(), nullable=True),
        sa.Column("train_size", sa.BigInteger(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_unique_constraint(
        "uq_darts_ml_model_artifacts_name_version",
        "darts_ml_model_artifacts",
        ["model_name", "version"],
    )
    op.create_index(
        "ix_darts_ml_model_artifacts_model_name",
        "darts_ml_model_artifacts",
        ["model_name"],
    )


def downgrade() -> None:
    op.drop_table("darts_ml_model_artifacts")
