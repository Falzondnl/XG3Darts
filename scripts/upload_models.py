"""
Upload trained ML model artifacts to PostgreSQL for persistent storage.

This script reads .pkl files from models/saved/ and upserts them into
the darts_ml_model_artifacts table. Run this locally after training;
the API will download models from DB at startup.

Usage:
    python scripts/upload_models.py [--model r0|r1|all]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import psycopg2
import psycopg2.extras
import joblib
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

from app.config import settings


def _db_url_sync(url: str) -> str:
    return (
        url.replace("postgresql+asyncpg://", "postgresql://")
           .replace("postgres+asyncpg://", "postgresql://")
    )


def upload_model(conn, model_name: str, pkl_path: Path) -> None:
    if not pkl_path.exists():
        log.warning("model_file_missing", path=str(pkl_path))
        return

    log.info("reading_model", name=model_name, path=str(pkl_path))
    artifact_bytes = pkl_path.read_bytes()
    meta = joblib.load(pkl_path)

    # Extract metadata (exclude large objects)
    metadata = {
        k: v for k, v in meta.items()
        if k not in ("model", "scaler", "lgbm", "meta", "artifact")
        and not hasattr(v, "predict")
    }

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO darts_ml_model_artifacts
                (model_name, version, artifact, size_bytes,
                 feature_count, brier_test, auc_test, train_size, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_name, version)
            DO UPDATE SET
                artifact     = EXCLUDED.artifact,
                size_bytes   = EXCLUDED.size_bytes,
                feature_count = EXCLUDED.feature_count,
                brier_test   = EXCLUDED.brier_test,
                auc_test     = EXCLUDED.auc_test,
                train_size   = EXCLUDED.train_size,
                metadata     = EXCLUDED.metadata,
                created_at   = now()
            """,
            (
                model_name,
                "1",
                psycopg2.Binary(artifact_bytes),
                len(artifact_bytes),
                meta.get("feature_count"),
                meta.get("brier_test"),
                meta.get("auc_test"),
                meta.get("train_size"),
                psycopg2.extras.Json(metadata),
            ),
        )
    conn.commit()
    log.info(
        "model_uploaded",
        name=model_name,
        size_kb=round(len(artifact_bytes) / 1024, 1),
        brier=meta.get("brier_test"),
        auc=meta.get("auc_test"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["r0", "r1", "all"], default="all")
    args = parser.parse_args()

    save_dir = _ROOT / "models" / "saved"
    db_url = _db_url_sync(settings.DATABASE_URL)
    conn = psycopg2.connect(db_url, connect_timeout=30)

    try:
        if args.model in ("r0", "all"):
            upload_model(conn, "r0_logit", save_dir / "r0_model.pkl")
        if args.model in ("r1", "all"):
            # r1_model_from_raw.pkl: sequential ELO, no leakage (AUC=0.8097)
            # replaces old leaky r1_model.pkl (AUC=0.9746 was due to static ELO snapshot)
            r1_raw = save_dir / "r1_model_from_raw.pkl"
            r1_old = save_dir / "r1_model.pkl"
            upload_model(conn, "r1_lgbm", r1_raw if r1_raw.exists() else r1_old)
    finally:
        conn.close()

    log.info("done")


if __name__ == "__main__":
    main()
