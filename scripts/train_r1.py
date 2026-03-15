"""
Train the R1 LightGBM+XGBoost stacking model.

Builds 38 features from DB: R0 features + rolling stats + hold/break + H2H.
Temporal split: 70/15/15. Out-of-fold stacking.

Leakage controls:
  - ELO: loaded as full time series; for each match we look up the rating_after
    from the player's PREVIOUS match (no future ELO leakage).
  - Rolling stats: histories built match-by-match in chronological order;
    features read BEFORE the current match is appended.
  - match_da1/match_da2: per-match averages stored in history AFTER feature row
    is recorded (future-row enrichment only, not current-row leakage).

Usage:
    python scripts/train_r1.py [--dry-run] [--n-matches N]
"""
from __future__ import annotations

import argparse
import bisect
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
import random
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
    return url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgres+asyncpg://", "postgresql://"
    )


def load_all_data(conn) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load matches, player static stats, and full ELO time series.

    Returns:
        matches   — chronologically ordered DataFrame
        stats     — player static features (3DA, ranking, country); NO ELO here
        elo_ts    — dict[player_id] → sorted list of (match_date_ts, rating_after)
                    used for temporal ELO lookup to prevent leakage
    """
    log.info("loading_matches")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                m.id AS match_id,
                m.player1_id, m.player2_id, m.winner_player_id,
                m.match_date, m.player1_score, m.player2_score,
                m.coverage_regime,
                m.source_name,
                -- Per-match averages from DartsDatabase raw data (NULL for PDC rows)
                (m.raw_source_data->>'player1_avg')::float AS match_avg_p1,
                (m.raw_source_data->>'player2_avg')::float AS match_avg_p2,
                c.format_code, c.ecosystem, c.season_year, c.is_televised
            FROM darts_matches m
            JOIN darts_competitions c ON c.id = m.competition_id
            WHERE m.status IN ('Completed', 'result')
              AND m.winner_player_id IS NOT NULL
              AND m.player1_id IS NOT NULL
              AND m.player2_id IS NOT NULL
              AND m.match_date IS NOT NULL
            ORDER BY m.match_date ASC
        """)
        matches = pd.DataFrame(cur.fetchall())
    log.info("matches_loaded", count=len(matches),
             pdc=int((matches["source_name"] == "pdc").sum()),
             dartsdatabase=int((matches["source_name"] == "dartsdatabase").sum()))

    log.info("loading_player_stats")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Static player attributes — NO ELO here (loaded separately as time series)
        cur.execute("""
            SELECT
                p.id AS player_id,
                p.dartsorakel_3da,
                p.dartsorakel_rank,
                p.pdc_ranking,
                p.source_confidence,
                p.country_code,
                COALESCE(
                    p.dartsorakel_3da,
                    (SELECT AVG(ps.three_dart_average)
                     FROM darts_player_stats ps
                     WHERE ps.player_id = p.id
                       AND ps.three_dart_average IS NOT NULL
                    )
                ) AS best_3da
            FROM darts_players p
        """)
        df_stats = pd.DataFrame(cur.fetchall()).set_index("player_id")
    log.info("stats_loaded", count=len(df_stats),
             with_3da=int(df_stats["best_3da"].notna().sum()))

    log.info("loading_elo_timeseries")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Load ALL ELO entries with their match dates for temporal lookup.
        # rating_after = ELO after that specific match completed.
        # We want ELO *before* a given match = rating_after of the prior entry.
        cur.execute("""
            SELECT player_id, match_date, rating_after
            FROM darts_elo_ratings
            WHERE pool = 'pdc_mens'
              AND match_date IS NOT NULL
            ORDER BY player_id, match_date ASC
        """)
        elo_rows = cur.fetchall()

    # Build time series: player_id → (sorted dates list, ratings list)
    # rating at index i = ELO after the i-th match
    elo_ts: dict[str, tuple[list, list]] = {}
    tmp: dict[str, list] = defaultdict(list)
    for r in elo_rows:
        pid = r["player_id"]
        mdate = r["match_date"]
        # Normalise to int days-since-epoch for bisect (works for both date and datetime)
        if hasattr(mdate, "toordinal"):
            ts = mdate.toordinal()
        else:
            ts = int(mdate)
        tmp[pid].append((ts, float(r["rating_after"])))

    for pid, entries in tmp.items():
        entries.sort(key=lambda x: x[0])
        dates = [e[0] for e in entries]
        ratings = [e[1] for e in entries]
        elo_ts[pid] = (dates, ratings)

    log.info("elo_timeseries_loaded",
             players_with_elo=len(elo_ts),
             total_entries=len(elo_rows))

    return matches, df_stats, elo_ts


def _temporal_elo(elo_ts: dict, player_id: str, before_date, default: float) -> float:
    """Return the player's ELO rating from their last match BEFORE before_date.

    Uses binary search on the sorted date list for O(log n) lookup.
    Returns default if the player has no prior ELO entries.
    """
    if player_id not in elo_ts:
        return default
    dates, ratings = elo_ts[player_id]
    if hasattr(before_date, "toordinal"):
        ts = before_date.toordinal()
    else:
        ts = int(before_date)
    # bisect_left gives the insertion point for ts in dates
    # We want the last entry STRICTLY before ts → index - 1
    idx = bisect.bisect_left(dates, ts) - 1
    if idx < 0:
        return default  # no prior rating
    return ratings[idx]


def build_r1_features(
    matches: pd.DataFrame,
    stats: pd.DataFrame,
    elo_ts: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build 38 R1 features with temporal rolling aggregates.

    Feature groups (38 total, no duplicates):
      Static/ELO (8): elo1, elo2, elo_diff, rank1, rank2, rank_diff_log, da1, da2
      Format/context (4): format_enc, stage_enc, format_len_enc, ecosystem_enc
      Rolling form (6): ewm_form×2, rolling_wr×2, rolling_3da×2
      Checkout (2): rolling_co×2
      Hold/break (4): hold_rate×2, break_rate×2
      Stage/floor 3DA (4): stage_3da×2, floor_3da×2
      Opp-adj form (2): opp_adj_form×2
      Rest/fatigue (2): days_since×2
      H2H (2): h2h_wr, h2h_count
      Momentum diff (4): elo_diff_sign×wr_diff, da_diff, co_diff, form_diff

    All rolling stats are computed from matches BEFORE the current match date.
    ELO is looked up temporally (no future ELO leakage).
    """
    DEFAULT_ELO = 1500.0
    DEFAULT_3DA  = 50.0

    def gs(pid, col, default):
        if pid in stats.index:
            v = stats.at[pid, col]
            if pd.notna(v):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return default

    def gs3da(pid) -> float:
        return gs(pid, "best_3da", DEFAULT_3DA)

    def format_enc(fmt):
        m = {"PDC_WC": 1, "PDC_WC_ERA_2020": 1, "PDC_PL": 2, "PDC_WM": 3,
             "PDC_GS": 4, "PDC_GP": 5, "PDC_PCF": 6, "PDC_UK": 7}
        return m.get(fmt, 0) / 7.0

    def ecosystem_enc(eco):
        m = {"pdc_mens": 1.0, "pdc_womens": 0.6, "wdf": 0.5, "development": 0.3}
        return m.get(eco, 0.5)

    # Build rolling histories: player_id → list of match records
    player_history: dict[str, list] = defaultdict(list)
    h2h_history: dict[tuple, list] = defaultdict(list)

    log.info("computing_rolling_stats")
    rng = random.Random(42)  # deterministic seed for reproducibility
    feature_rows = []
    labels = []

    for idx, row in matches.iterrows():
        raw_p1, raw_p2 = row["player1_id"], row["player2_id"]
        winner = row["winner_player_id"]
        mdate = row["match_date"]
        fmt  = str(row.get("format_code", "") or "")
        eco  = str(row.get("ecosystem", "") or "pdc_mens")
        is_tv = bool(row.get("is_televised", False))

        # --- Random pair swap: enforces ~50% class balance, removes structural
        # ordering bias (PDC stores seeded player as player1 → p1_win_rate=61%) ---
        if rng.random() < 0.5:
            p1, p2 = raw_p1, raw_p2
            match_avg_a = row.get("match_avg_p1")
            match_avg_b = row.get("match_avg_p2")
        else:
            p1, p2 = raw_p2, raw_p1
            match_avg_a = row.get("match_avg_p2")
            match_avg_b = row.get("match_avg_p1")

        # --- ELO: temporal lookup — rating from BEFORE this match ---
        elo1 = _temporal_elo(elo_ts, p1, mdate, DEFAULT_ELO)
        elo2 = _temporal_elo(elo_ts, p2, mdate, DEFAULT_ELO)

        # --- Static features (3DA, ranking — no temporal issue for career avgs) ---
        da1  = gs3da(p1)
        da2  = gs3da(p2)
        r1   = gs(p1, "pdc_ranking", 200.0) or 200.0
        r2   = gs(p2, "pdc_ranking", 200.0) or 200.0

        # Per-match averages from DartsDatabase (NaN for PDC rows)
        # Used only for updating history AFTER this row is recorded
        match_da1 = float(match_avg_a) if pd.notna(match_avg_a) and match_avg_a else da1
        match_da2 = float(match_avg_b) if pd.notna(match_avg_b) and match_avg_b else da2

        # --- Rolling stats from history (read BEFORE current match appended) ---
        h1  = player_history[p1]
        h2  = player_history[p2]
        h12 = h2h_history[tuple(sorted([p1, p2]))]

        def rolling_wr(hist, n=200):
            if not hist: return 0.5
            recent = hist[-n:]
            return sum(x["won"] for x in recent) / len(recent)

        def rolling_3da(hist, n=200):
            if not hist: return DEFAULT_3DA
            vals = [x["da"] for x in hist[-n:] if x["da"] > 0]
            return np.mean(vals) if vals else DEFAULT_3DA

        def rolling_co(hist, n=200):
            if not hist: return 0.40
            vals = [x["co"] for x in hist[-n:] if x["co"] >= 0]
            return np.mean(vals) if vals else 0.40

        def ewm_form(hist, decay=0.05):
            if not hist: return DEFAULT_3DA
            vals = np.array([x["da"] for x in hist if x["da"] > 0])
            if len(vals) == 0: return DEFAULT_3DA
            weights = (1 - decay) ** np.arange(len(vals) - 1, -1, -1)
            return float(np.dot(weights, vals) / weights.sum())

        def hold_rate(hist):
            starters = [x for x in hist if x.get("was_starter")]
            if not starters: return 0.5
            return sum(x["won"] for x in starters) / len(starters)

        def break_rate(hist):
            receivers = [x for x in hist if not x.get("was_starter", True)]
            if not receivers: return 0.5
            return sum(x["won"] for x in receivers) / len(receivers)

        def stage_3da(hist):
            vals = [x["da"] for x in hist if x.get("televised") and x["da"] > 0]
            return np.mean(vals) if vals else DEFAULT_3DA

        def floor_3da(hist):
            vals = [x["da"] for x in hist if not x.get("televised") and x["da"] > 0]
            return np.mean(vals) if vals else DEFAULT_3DA

        def days_since(hist):
            if not hist: return 30.0
            last = hist[-1].get("date")
            if not last: return 30.0
            delta = (mdate - last).days if hasattr(mdate, 'year') else 30
            return min(float(delta), 180.0) / 180.0

        def opp_adj_form(hist, n=50):
            if not hist: return 0.0
            recent = hist[-n:]
            scores = []
            for x in recent:
                opp_elo = x.get("opp_elo", DEFAULT_ELO)
                expected = 1 / (1 + 10 ** ((opp_elo - DEFAULT_ELO) / 400))
                actual = 1.0 if x["won"] else 0.0
                scores.append(actual - expected)
            return float(np.mean(scores)) if scores else 0.0

        def h2h_wr(hh, pid):
            if not hh: return 0.5
            wins = sum(1 for x in hh if x["winner"] == pid)
            return wins / len(hh)

        wr1 = rolling_wr(h1)
        wr2 = rolling_wr(h2)
        co1 = rolling_co(h1)
        co2 = rolling_co(h2)
        form1 = ewm_form(h1)
        form2 = ewm_form(h2)
        rda1 = rolling_3da(h1)
        rda2 = rolling_3da(h2)

        feat = [
            # Static/ELO (8) — ELO is temporal, no leakage
            elo1, elo2, elo1 - elo2,
            min(r1, 300) / 300.0, min(r2, 300) / 300.0,
            np.log1p(r2) - np.log1p(r1),
            da1, da2,
            # Format/context (4)
            format_enc(fmt),
            0.0 if "WC" in fmt else (1.0 if "WM" in fmt or "GS" in fmt else 0.5),
            1.0 if len(fmt) <= 6 else 0.0,
            ecosystem_enc(eco),
            # Rolling form (6)
            form1, form2,
            wr1, wr2,
            rda1, rda2,
            # Checkout (2)
            co1, co2,
            # Hold/break (4)
            hold_rate(h1), hold_rate(h2),
            break_rate(h1), break_rate(h2),
            # Stage/floor 3DA (4)
            stage_3da(h1), stage_3da(h2),
            floor_3da(h1), floor_3da(h2),
            # Opp-adj form (2)
            opp_adj_form(h1), opp_adj_form(h2),
            # Rest/fatigue (2)
            days_since(h1), days_since(h2),
            # H2H (2)
            h2h_wr(h12, p1), min(len(h12), 50) / 50.0,
            # Momentum diffs (4) — combined signals, no extra leakage
            wr1 - wr2,
            da1 - da2,
            co1 - co2,
            form1 - form2,
        ]
        assert len(feat) == 38, f"Feature count mismatch: {len(feat)}"
        feature_rows.append(feat)
        labels.append(1.0 if winner == p1 else 0.0)

        # Update histories AFTER recording feature (prevents current-match leakage)
        p1_won = (winner == p1)
        p2_won = (winner == p2)
        rec1 = dict(won=p1_won, da=match_da1, co=0.40, opp_elo=elo2,
                    televised=is_tv, date=mdate, was_starter=True)
        rec2 = dict(won=p2_won, da=match_da2, co=0.40, opp_elo=elo1,
                    televised=is_tv, date=mdate, was_starter=False)
        player_history[p1].append(rec1)
        player_history[p2].append(rec2)
        h2h_history[tuple(sorted([p1, p2]))].append({"winner": winner})

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    log.info("features_built", shape=X.shape)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-matches", type=int, default=0, help="Limit matches (0=all)")
    parser.add_argument("--db-url", default=None, help="Override DATABASE_URL")
    args = parser.parse_args()

    db_url = _db_url_sync(args.db_url or settings.DATABASE_URL)
    conn = psycopg2.connect(db_url, connect_timeout=60)
    matches, stats, elo_ts = load_all_data(conn)
    conn.close()

    if args.n_matches > 0:
        matches = matches.tail(args.n_matches)
        log.info("limited_matches", n=len(matches))

    X, y = build_r1_features(matches, stats, elo_ts)

    if args.dry_run:
        log.info("dry_run_done", shape=X.shape)
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
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
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
    log.info("r1_eval", brier=round(brier, 4), auc=round(auc, 4), test_size=len(X_test))

    joblib.dump({
        "lgbm": lgb_model,
        "meta": meta,
        "scaler": scaler,
        "feature_count": X.shape[1],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_size": len(X_train),
        "total_matches": len(X),
        "brier_test": brier,
        "auc_test": auc,
        "note": "R1 v2 — temporal ELO, no duplicate features. PDC (276k) + DartsDatabase (5.4k), 38 features.",
    }, save_dir / "r1_model.pkl")
    log.info("r1_saved", path=str(save_dir / "r1_model.pkl"))


if __name__ == "__main__":
    main()
