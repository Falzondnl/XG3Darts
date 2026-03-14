"""
Train the R0 logistic regression model from Railway PostgreSQL data.

Pulls 276k completed PDC matches + player ELO ratings + DartsOrakel 3DA stats,
builds the 14-feature matrix, runs temporal-split training + calibration,
and saves the model to models/saved/r0_model.pkl.

Usage:
    python scripts/train_r0.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
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
from models.trainer import DartsModelTrainer


def _db_url_sync(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgres+asyncpg://", "postgresql://"
    )


def load_matches(conn) -> pd.DataFrame:
    log.info("loading_matches")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                m.id AS match_id,
                m.player1_id,
                m.player2_id,
                m.winner_player_id,
                m.match_date,
                m.player1_score,
                m.player2_score,
                m.coverage_regime AS regime,
                m.source_name,
                c.format_code,
                c.ecosystem,
                c.season_year
            FROM darts_matches m
            JOIN darts_competitions c ON c.id = m.competition_id
            WHERE m.status = 'Completed'
              AND m.winner_player_id IS NOT NULL
              AND m.player1_id IS NOT NULL
              AND m.player2_id IS NOT NULL
              AND m.match_date IS NOT NULL
            ORDER BY m.match_date ASC
        """)
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    log.info("matches_loaded", count=len(df))
    return df


def load_player_stats(conn) -> pd.DataFrame:
    """Load per-player stats: elo_rating, 3da, ranking."""
    log.info("loading_player_stats")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                p.id AS player_id,
                p.dartsorakel_3da AS avg_3da,
                p.dartsorakel_rank AS orakel_rank,
                p.pdc_ranking,
                p.source_confidence,
                p.country_code,
                e.rating_after AS elo_rating,
                e.games_played_at_time AS games_played
            FROM darts_players p
            LEFT JOIN darts_elo_ratings e
                ON e.player_id = p.id AND e.pool = 'pdc_mens'
        """)
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    log.info("player_stats_loaded", count=len(df), with_elo=df["elo_rating"].notna().sum())
    return df


def build_r0_features(matches_df: pd.DataFrame, stats_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build the 14 R0 features for each match.

    Features (R0):
    1.  elo_diff          = P1 ELO - P2 ELO
    2.  elo_p1            = P1 ELO rating
    3.  elo_p2            = P2 ELO rating
    4.  avg3da_diff       = P1 3DA - P2 3DA
    5.  avg3da_p1         = P1 3DA
    6.  avg3da_p2         = P2 3DA
    7.  rank_diff         = P2 rank - P1 rank (lower rank number = higher rank)
    8.  rank_p1           = P1 PDC ranking (normalized)
    9.  rank_p2           = P2 PDC ranking (normalized)
    10. same_country      = (P1 country == P2 country) ? 1 : 0
    11. season_year_norm  = (season_year - 2000) / 30
    12. format_pdcwc      = 1 if format_code starts with PDC_WC
    13. format_pdcpl      = 1 if PDC_PL
    14. format_other      = 1 otherwise (baseline)
    """
    stats = stats_df.set_index("player_id")
    DEFAULT_ELO = 1500.0

    def get_stat(pid, col, default):
        if pid in stats.index:
            v = stats.at[pid, col]
            if pd.isna(v):
                return default
            try:
                return float(v)
            except (ValueError, TypeError):
                return v  # return as-is for non-numeric (e.g. country_code)
        return default

    log.info("building_features", matches=len(matches_df))
    features = []
    labels = []

    for _, row in matches_df.iterrows():
        p1 = row["player1_id"]
        p2 = row["player2_id"]
        winner = row["winner_player_id"]

        elo1 = get_stat(p1, "elo_rating", DEFAULT_ELO)
        elo2 = get_stat(p2, "elo_rating", DEFAULT_ELO)
        da1  = get_stat(p1, "avg_3da", 50.0)
        da2  = get_stat(p2, "avg_3da", 50.0)
        r1   = get_stat(p1, "pdc_ranking", 200.0) or 200.0
        r2   = get_stat(p2, "pdc_ranking", 200.0) or 200.0
        c1   = get_stat(p1, "country_code", "")
        c2   = get_stat(p2, "country_code", "")

        fmt = str(row.get("format_code", "") or "")
        year = int(row.get("season_year", 2015) or 2015)

        feat = [
            elo1 - elo2,           # 1
            elo1,                   # 2
            elo2,                   # 3
            da1 - da2,             # 4
            da1,                    # 5
            da2,                    # 6
            r2 - r1,               # 7
            min(r1, 300) / 300.0,  # 8
            min(r2, 300) / 300.0,  # 9
            1.0 if (c1 and c1 == c2) else 0.0,  # 10
            (year - 2000) / 30.0,  # 11
            1.0 if fmt.startswith("PDC_WC") else 0.0,  # 12
            1.0 if fmt == "PDC_PL" else 0.0,            # 13
            1.0 if not (fmt.startswith("PDC_WC") or fmt == "PDC_PL") else 0.0,  # 14
        ]
        features.append(feat)
        labels.append(1.0 if winner == p1 else 0.0)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    log.info("features_built", shape=X.shape, p1_win_rate=f"{y.mean():.3f}")
    return pd.DataFrame(X, columns=[
        "elo_diff", "elo_p1", "elo_p2",
        "avg3da_diff", "avg3da_p1", "avg3da_p2",
        "rank_diff", "rank_p1_norm", "rank_p2_norm",
        "same_country", "season_year_norm",
        "fmt_wc", "fmt_pl", "fmt_other"
    ]), y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_url = _db_url_sync(settings.DATABASE_URL)
    conn = psycopg2.connect(db_url, connect_timeout=30)

    matches_df = load_matches(conn)
    stats_df   = load_player_stats(conn)
    conn.close()

    X, y = build_r0_features(matches_df, stats_df)

    if args.dry_run:
        log.info("dry_run_done", features=X.shape, labels=y.shape)
        return

    # Temporal train/val/test split (70/15/15)
    n = len(X)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end].values, y[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end].values, y[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:].values, y[val_end:]

    log.info("split_done", train=len(X_train), val=len(X_val), test=len(X_test))

    # Import and train R0 model
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    import joblib

    save_dir = _ROOT / "models" / "saved"
    save_dir.mkdir(exist_ok=True)

    log.info("training_r0_logistic")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    base_clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    base_clf.fit(X_train_s, y_train)

    # Platt scaling calibration on validation set
    calibrated = CalibratedClassifierCV(base_clf, method="sigmoid", cv="prefit")
    calibrated.fit(X_val_s, y_val)

    # Evaluate on test set
    proba_test = calibrated.predict_proba(X_test_s)[:, 1]
    brier = brier_score_loss(y_test, proba_test)
    logloss = log_loss(y_test, proba_test)
    auc = roc_auc_score(y_test, proba_test)

    log.info("r0_eval",
        brier=round(brier, 4),
        log_loss=round(logloss, 4),
        auc=round(auc, 4),
        test_size=len(X_test),
    )

    # Save
    joblib.dump({
        "scaler": scaler,
        "model": calibrated,
        "feature_names": X.columns.tolist(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_size": len(X_train),
        "brier_test": brier,
        "auc_test": auc,
    }, save_dir / "r0_model.pkl")

    log.info("r0_saved", path=str(save_dir / "r0_model.pkl"))


if __name__ == "__main__":
    main()
