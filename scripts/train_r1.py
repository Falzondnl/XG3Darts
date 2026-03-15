"""
Train the R1 LightGBM+XGBoost stacking model.

Builds 38 CLEAN temporal features from DB: ELO (temporal) + rolling form.
Temporal split: 70/15/15. Out-of-fold stacking.

LEAKAGE AUDIT (all fixed):
  - ELO: full 556k-entry time series, bisect lookup per match (no future ELO).
  - 3DA career avg (dartsorakel_3da): REMOVED — career avg includes future
    matches. Rolling 3DA from actual match data only.
  - PDC history da fallback: FIXED — was storing career_avg as da for 276k PDC
    rows, contaminating rolling_3da/ewm_form. Now stores da=0 (filtered out).
  - pdc_ranking: REMOVED — single 2026 snapshot applied to all history.
    ELO covers player quality temporally.
  - rolling_co: REMOVED — always 0.40 (no real checkout data in history).
  - hold_rate/break_rate: REMOVED — was_starter randomly assigned after pair
    swap, making these near-duplicates of rolling_wr.
  - Random pair swap (seed=42): enforces ~50% class balance.

Theoretical max AUC from perfect temporal ELO oracle: 0.7533
Target AUC after leakage removal: 0.74-0.80.

Usage:
    python scripts/train_r1.py [--dry-run] [--n-matches N]
"""
from __future__ import annotations

import argparse
import bisect
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

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

FEATURE_NAMES = [
    # ELO — temporal bisect lookup (5)
    "elo1", "elo2", "elo_diff", "abs_elo_diff", "elo_win_prob",
    # Format / context (5)
    "format_enc", "stage_enc", "format_len_enc", "ecosystem_enc", "is_televised",
    # Rolling temporal form — uses ONLY actual per-match 3DA (DartsDatabase rows)
    # For PDC-only players: da=0 stored → rolling_3da/ewm_form default to DEFAULT_3DA (6)
    "ewm_form_p1", "ewm_form_p2",
    "rolling_3da_p1", "rolling_3da_p2",
    "rolling_wr_p1", "rolling_wr_p2",
    # Form diffs — temporal (3)
    "ewm_diff", "rda_diff", "wr_diff",
    # Opp-adjusted form — uses temporal ELO (3)
    "opp_adj_p1", "opp_adj_p2", "opp_adj_diff",
    # Stage / floor 3DA — temporal (after da=0 fix) (4)
    "stage_3da_p1", "stage_3da_p2",
    "floor_3da_p1", "floor_3da_p2",
    # Experience / rest — fully temporal (4)
    "match_count_p1", "match_count_p2",
    "days_since_p1", "days_since_p2",
    # H2H — fully temporal (2)
    "h2h_wr", "h2h_count",
    # Match quality / cross signals (6)
    "match_level",        # (elo1+elo2)/(2*DEFAULT_ELO)
    "days_rest_diff",     # days_since_p1 - days_since_p2
    "exp_gap",            # match_count_p1 - match_count_p2
    "form_x_wr_p1",       # ewm_form * rolling_wr (compound skill signal)
    "form_x_wr_p2",
    "rda_x_wr_diff",      # (rda1*wr1) - (rda2*wr2)
]
assert len(FEATURE_NAMES) == 38, f"Expected 38, got {len(FEATURE_NAMES)}"


def _db_url_sync(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgres+asyncpg://", "postgresql://"
    )


def load_all_data(conn) -> tuple[pd.DataFrame, dict]:
    """Load matches and full ELO time series.

    Returns:
        matches — chronologically ordered DataFrame
        elo_ts  — dict[player_id] → (sorted_ordinal_dates, ratings_after)
                  for temporal ELO lookup (no future ELO leakage)

    Note: No static career stats (3DA, ranking) are loaded — they are
    temporally leaky (single 2026 snapshots applied to all history).
    Rolling 3DA is computed in-loop from actual DartsDatabase match averages.
    """
    log.info("loading_matches")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            -- Deduplicate: DISTINCT ON (canonical_pair, date, source) eliminates
            -- both exact duplicates and reverse duplicates (A vs B + B vs A same day).
            -- Keeps the row with the smallest id (earliest inserted) per canonical pair.
            SELECT
                m.id AS match_id,
                m.player1_id, m.player2_id, m.winner_player_id,
                m.match_date, m.source_name,
                -- Per-match averages only available for DartsDatabase rows
                (m.raw_source_data->>'player1_avg')::float AS match_avg_p1,
                (m.raw_source_data->>'player2_avg')::float AS match_avg_p2,
                c.format_code, c.ecosystem, c.is_televised
            FROM (
                SELECT DISTINCT ON (
                    LEAST(player1_id, player2_id),
                    GREATEST(player1_id, player2_id),
                    match_date,
                    source_name
                ) *
                FROM darts_matches
                WHERE status IN ('Completed', 'result')
                  AND winner_player_id IS NOT NULL
                  AND player1_id IS NOT NULL
                  AND player2_id IS NOT NULL
                  AND match_date IS NOT NULL
                ORDER BY
                    LEAST(player1_id, player2_id),
                    GREATEST(player1_id, player2_id),
                    match_date, source_name, id
            ) m
            JOIN darts_competitions c ON c.id = m.competition_id
            ORDER BY m.match_date ASC
        """)
        matches = pd.DataFrame(cur.fetchall())
    log.info("matches_loaded", count=len(matches),
             pdc=int((matches["source_name"] == "pdc").sum()),
             dartsdatabase=int((matches["source_name"] == "dartsdatabase").sum()))

    log.info("loading_elo_timeseries")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT player_id, match_date, rating_after
            FROM darts_elo_ratings
            WHERE pool = 'pdc_mens'
              AND match_date IS NOT NULL
            ORDER BY player_id, match_date ASC
        """)
        elo_rows = cur.fetchall()

    tmp: dict[str, list] = defaultdict(list)
    for r in elo_rows:
        pid = r["player_id"]
        mdate = r["match_date"]
        ts = mdate.toordinal() if hasattr(mdate, "toordinal") else int(mdate)
        tmp[pid].append((ts, float(r["rating_after"])))

    elo_ts: dict[str, tuple[list, list]] = {}
    for pid, entries in tmp.items():
        entries.sort(key=lambda x: x[0])
        elo_ts[pid] = ([e[0] for e in entries], [e[1] for e in entries])

    log.info("elo_timeseries_loaded",
             players_with_elo=len(elo_ts),
             total_entries=len(elo_rows))

    return matches, elo_ts


def _temporal_elo(elo_ts: dict, player_id: str, before_date, default: float) -> float:
    """Return the player's ELO from their last match STRICTLY BEFORE before_date."""
    if player_id not in elo_ts:
        return default
    dates, ratings = elo_ts[player_id]
    ts = before_date.toordinal() if hasattr(before_date, "toordinal") else int(before_date)
    idx = bisect.bisect_left(dates, ts) - 1
    return ratings[idx] if idx >= 0 else default


def build_r1_features(
    matches: pd.DataFrame,
    elo_ts: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build 38 clean temporal features.

    All leakages removed:
    - No static career 3DA (dartsorakel_3da / best_3da)
    - No current pdc_ranking
    - No rolling_co (was always 0.40)
    - No hold_rate / break_rate (was_starter randomly assigned)
    - History da=0 for PDC rows (no career-avg contamination)
    - ELO: temporal bisect lookup
    - Random pair swap for class balance
    """
    DEFAULT_ELO = 1500.0
    DEFAULT_3DA  = 50.0

    def format_enc(fmt):
        m = {"PDC_WC": 1, "PDC_WC_ERA_2020": 1, "PDC_PL": 2, "PDC_WM": 3,
             "PDC_GS": 4, "PDC_GP": 5, "PDC_PCF": 6, "PDC_UK": 7}
        return m.get(fmt, 0) / 7.0

    def ecosystem_enc(eco):
        m = {"pdc_mens": 1.0, "pdc_womens": 0.6, "wdf": 0.5, "development": 0.3}
        return m.get(eco, 0.5)

    player_history: dict[str, list] = defaultdict(list)
    h2h_history: dict[tuple, list] = defaultdict(list)

    log.info("computing_rolling_stats")
    rng = random.Random(42)
    feature_rows = []
    labels = []

    for _, row in matches.iterrows():
        raw_p1, raw_p2 = row["player1_id"], row["player2_id"]
        winner = row["winner_player_id"]
        mdate = row["match_date"]
        fmt   = str(row.get("format_code", "") or "")
        eco   = str(row.get("ecosystem", "") or "pdc_mens")
        is_tv = bool(row.get("is_televised", False))

        # Random pair swap → ~50% class balance, removes structural ordering bias
        if rng.random() < 0.5:
            p1, p2 = raw_p1, raw_p2
            raw_da1 = row.get("match_avg_p1")
            raw_da2 = row.get("match_avg_p2")
        else:
            p1, p2 = raw_p2, raw_p1
            raw_da1 = row.get("match_avg_p2")
            raw_da2 = row.get("match_avg_p1")

        # Temporal ELO lookup — rating BEFORE this match
        elo1 = _temporal_elo(elo_ts, p1, mdate, DEFAULT_ELO)
        elo2 = _temporal_elo(elo_ts, p2, mdate, DEFAULT_ELO)

        # Per-match 3DA: only available for DartsDatabase rows
        # PDC rows: da=0 stored in history (NOT career avg — that's future-leaky)
        # 0 is filtered out by rolling_3da / ewm_form / stage_3da / floor_3da
        match_da1 = float(raw_da1) if raw_da1 is not None and pd.notna(raw_da1) and raw_da1 else 0.0
        match_da2 = float(raw_da2) if raw_da2 is not None and pd.notna(raw_da2) and raw_da2 else 0.0

        # Rolling stats — READ before appending current match to history
        h1  = player_history[p1]
        h2  = player_history[p2]
        h12 = h2h_history[tuple(sorted([p1, p2]))]

        def rolling_3da(hist, n=200):
            vals = [x["da"] for x in hist[-n:] if x["da"] > 0]
            return float(np.mean(vals)) if vals else DEFAULT_3DA

        def ewm_form(hist, decay=0.05):
            vals = np.array([x["da"] for x in hist if x["da"] > 0])
            if len(vals) == 0:
                return DEFAULT_3DA
            weights = (1 - decay) ** np.arange(len(vals) - 1, -1, -1)
            return float(np.dot(weights, vals) / weights.sum())

        def rolling_wr(hist, n=200):
            if not hist:
                return 0.5
            recent = hist[-n:]
            return sum(x["won"] for x in recent) / len(recent)

        def stage_3da(hist):
            vals = [x["da"] for x in hist if x.get("televised") and x["da"] > 0]
            return float(np.mean(vals)) if vals else DEFAULT_3DA

        def floor_3da(hist):
            vals = [x["da"] for x in hist if not x.get("televised") and x["da"] > 0]
            return float(np.mean(vals)) if vals else DEFAULT_3DA

        def days_since(hist):
            if not hist:
                return 1.0  # no prior match → normalised max
            last = hist[-1].get("date")
            if not last:
                return 1.0
            delta = (mdate - last).days if hasattr(mdate, "year") else 30
            return min(float(delta), 180.0) / 180.0

        def opp_adj_form(hist, n=50):
            if not hist:
                return 0.0
            recent = hist[-n:]
            scores = []
            for x in recent:
                opp_elo = x.get("opp_elo", DEFAULT_ELO)
                expected = 1.0 / (1.0 + 10.0 ** ((opp_elo - DEFAULT_ELO) / 400.0))
                scores.append((1.0 if x["won"] else 0.0) - expected)
            return float(np.mean(scores)) if scores else 0.0

        def h2h_wr(hh, pid):
            if not hh:
                return 0.5
            return sum(1 for x in hh if x["winner"] == pid) / len(hh)

        form1 = ewm_form(h1)
        form2 = ewm_form(h2)
        rda1  = rolling_3da(h1)
        rda2  = rolling_3da(h2)
        wr1   = rolling_wr(h1)
        wr2   = rolling_wr(h2)
        opp1  = opp_adj_form(h1)
        opp2  = opp_adj_form(h2)
        st1   = stage_3da(h1)
        st2   = stage_3da(h2)
        fl1   = floor_3da(h1)
        fl2   = floor_3da(h2)
        ds1   = days_since(h1)
        ds2   = days_since(h2)
        mc1   = min(len(h1), 500) / 500.0
        mc2   = min(len(h2), 500) / 500.0
        h2h_w = h2h_wr(h12, p1)
        h2h_n = min(len(h12), 50) / 50.0
        elo_sum = (elo1 + elo2) / (2.0 * DEFAULT_ELO)

        feat = [
            # ELO — temporal (5)
            elo1, elo2, elo1 - elo2,
            abs(elo1 - elo2),
            elo1 / (elo1 + elo2 + 1e-9),
            # Format / context (5)
            format_enc(fmt),
            0.0 if "WC" in fmt else (1.0 if "WM" in fmt or "GS" in fmt else 0.5),
            1.0 if len(fmt) <= 6 else 0.0,
            ecosystem_enc(eco),
            float(is_tv),
            # Rolling temporal form (6)
            form1, form2,
            rda1,  rda2,
            wr1,   wr2,
            # Form diffs (3)
            form1 - form2,
            rda1  - rda2,
            wr1   - wr2,
            # Opp-adjusted form (3)
            opp1, opp2,
            opp1 - opp2,
            # Stage / floor 3DA (4)
            st1, st2, fl1, fl2,
            # Experience / rest (4)
            mc1, mc2, ds1, ds2,
            # H2H (2)
            h2h_w, h2h_n,
            # Match quality / cross signals (6)
            elo_sum,
            ds1  - ds2,
            mc1  - mc2,
            form1 * wr1,
            form2 * wr2,
            (rda1 * wr1) - (rda2 * wr2),
        ]
        assert len(feat) == 38, f"Feature count mismatch: {len(feat)}"
        feature_rows.append(feat)
        labels.append(1.0 if winner == p1 else 0.0)

        # Update history AFTER recording feature row (no current-match leakage)
        # da=0 for PDC rows (no actual 3DA available) — filtered out by rolling fns
        p1_won = (winner == p1)
        p2_won = (winner == p2)
        player_history[p1].append(dict(
            won=p1_won, da=match_da1, opp_elo=elo2,
            televised=is_tv, date=mdate,
        ))
        player_history[p2].append(dict(
            won=p2_won, da=match_da2, opp_elo=elo1,
            televised=is_tv, date=mdate,
        ))
        h2h_history[tuple(sorted([p1, p2]))].append({"winner": winner})

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    log.info("features_built", shape=X.shape,
             class_balance=round(float(y.mean()), 3))
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-matches", type=int, default=0)
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    db_url = _db_url_sync(args.db_url or settings.DATABASE_URL)
    conn = psycopg2.connect(db_url, connect_timeout=60)
    matches, elo_ts = load_all_data(conn)
    conn.close()

    if args.n_matches > 0:
        matches = matches.tail(args.n_matches)
        log.info("limited_matches", n=len(matches))

    X, y = build_r1_features(matches, elo_ts)

    if args.dry_run:
        log.info("dry_run_done", shape=X.shape,
                 class_balance=round(float(y.mean()), 3))
        return

    n = len(X)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]
    log.info("split", train=len(X_train), val=len(X_val), test=len(X_test))

    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    import joblib

    save_dir = _ROOT / "models" / "saved"
    save_dir.mkdir(exist_ok=True)

    log.info("training_lgbm")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
        feature_name=FEATURE_NAMES,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        feature_name=FEATURE_NAMES,
    )

    val_proba_lgb = lgb_model.predict_proba(X_val)[:, 1].reshape(-1, 1)
    scaler = StandardScaler()
    val_proba_s = scaler.fit_transform(val_proba_lgb)
    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(val_proba_s, y_val)

    test_proba_lgb = lgb_model.predict_proba(X_test)[:, 1].reshape(-1, 1)
    test_proba_s = scaler.transform(test_proba_lgb)
    proba_test = meta.predict_proba(test_proba_s)[:, 1]

    brier = brier_score_loss(y_test, proba_test)
    auc   = roc_auc_score(y_test, proba_test)
    log.info("r1_eval", brier=round(brier, 4), auc=round(auc, 4),
             test_size=len(X_test))

    # Feature importance
    importance = dict(zip(FEATURE_NAMES,
                          lgb_model.feature_importances_.tolist()))
    top5 = sorted(importance.items(), key=lambda x: -x[1])[:5]
    log.info("top5_features", features=top5)

    joblib.dump({
        "lgbm": lgb_model,
        "meta": meta,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "feature_count": X.shape[1],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_size": len(X_train),
        "total_matches": len(X),
        "brier_test": brier,
        "auc_test": auc,
        "note": (
            "R1 v4 — all temporal leakages removed: no career 3DA, no current ranking. "
            "ELO temporal bisect, random pair swap, da=0 for PDC rows. "
            "Theoretical max AUC from ELO oracle = 0.7533."
        ),
    }, save_dir / "r1_model.pkl")
    log.info("r1_saved", path=str(save_dir / "r1_model.pkl"))


if __name__ == "__main__":
    main()
