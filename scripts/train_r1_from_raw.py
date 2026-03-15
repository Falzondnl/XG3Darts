"""
Train R1 model from raw DartsDatabase JSON files with sequential ELO.

Fixes the data leakage in the original train_r1.py where static ELO
ratings from the DB snapshot were used as features instead of the
ELO ratings at the time of each match.

Pipeline:
1. Load all DartsDatabase match events (56 events, ~5500 matches)
2. Parse event dates from titles/URLs
3. Sort matches chronologically
4. Compute ELO sequentially in-loop (pre-match ELO as feature, update after)
5. Build R1-like feature vector using only data available before each match
6. Temporal split 70/15/15 → train LightGBM + meta-learner
7. Save model to models/saved/r1_model_from_raw.pkl

Usage:
    cd E:/DF/XG3V10/darts/XG3Darts
    python scripts/train_r1_from_raw.py [--dry-run] [--n-matches N]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
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


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVENTS_DIR = Path("D:/codex/Data/Darts/01_raw/json/dartsdatabase/events")
ORAKEL_SEED = Path("D:/codex/Data/Darts/01_raw/json/dartsorakel/stats_player.json")
SAVE_DIR = _ROOT / "models" / "saved"


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_event_date(event: dict) -> date | None:
    """Extract the best available date for an event (end date preferred)."""
    url = event.get("event_url", "")
    title = event.get("event_title", "")

    # Try URL: eda=DD/MM/YYYY  (end date)
    m = re.search(r"eda=(\d{2}/\d{2}/\d{4})", url)
    if m:
        d, mo, y = m.group(1).split("/")
        return date(int(y), int(mo), int(d))

    # Try title: DD/MM/YYYY - DD/MM/YYYY  (start date)
    m2 = re.search(r"(\d{2}/\d{2}/\d{4})", title)
    if m2:
        d, mo, y = m2.group(1).split("/")
        return date(int(y), int(mo), int(d))

    return None


# ---------------------------------------------------------------------------
# Load and sort all matches
# ---------------------------------------------------------------------------

def load_matches(rng_seed: int = 42) -> list[dict]:
    """
    Load all matches from DartsDatabase event files.

    NOTE: DartsDatabase always stores the winner as player1.
    We randomly swap ~50% of pairs so the model sees balanced labels.
    The 'result_type' field is set AFTER the swap.

    Returns a flat list of match dicts, each augmented with:
    - event_date: date object
    - event_id: str
    - event_title: str
    - result_type: "p1_win" | "p2_win"  (winner = original winner, may be p2 after swap)
    """
    rng = np.random.default_rng(rng_seed)
    all_matches = []

    for f in EVENTS_DIR.glob("*.json"):
        event = json.loads(f.read_text(encoding="utf-8"))
        event_date = _parse_event_date(event)
        event_id = event.get("event_id", f.stem)
        event_title = event.get("event_title", "")

        for m in event.get("matches", []):
            s1 = m.get("score_p1")
            s2 = m.get("score_p2")
            if s1 is None or s2 is None:
                continue
            # DartsDatabase always has s1 > s2 (winner is player1)
            # Skip any degenerate rows
            if s1 <= s2:
                continue

            # Original assignment: p1 = winner, p2 = loser
            winner_id = str(m.get("player1_id", ""))
            loser_id = str(m.get("player2_id", ""))
            winner_name = m.get("player1_name", "")
            loser_name = m.get("player2_name", "")
            winner_avg = float(m.get("player1_avg", 0) or 0)
            loser_avg = float(m.get("player2_avg", 0) or 0)

            if not winner_id or not loser_id:
                continue

            # Random swap with 50% probability for class balance
            if rng.random() < 0.5:
                # Swap: p1 becomes loser, p2 becomes winner → result = p2_win
                all_matches.append({
                    "event_id": event_id,
                    "event_title": event_title,
                    "event_date": event_date,
                    "player1_id": loser_id,
                    "player2_id": winner_id,
                    "player1_name": loser_name,
                    "player2_name": winner_name,
                    "player1_avg": loser_avg,
                    "player2_avg": winner_avg,
                    "round": m.get("round", ""),
                    "result_type": "p2_win",
                })
            else:
                # No swap: p1 = winner → result = p1_win
                all_matches.append({
                    "event_id": event_id,
                    "event_title": event_title,
                    "event_date": event_date,
                    "player1_id": winner_id,
                    "player2_id": loser_id,
                    "player1_name": winner_name,
                    "player2_name": loser_name,
                    "player1_avg": winner_avg,
                    "player2_avg": loser_avg,
                    "round": m.get("round", ""),
                    "result_type": "p1_win",
                })

    # Sort chronologically; None dates go last
    all_matches.sort(key=lambda x: (
        x["event_date"] or date(2099, 1, 1),
        x["event_id"],
    ))

    p1_wins = sum(1 for m in all_matches if m["result_type"] == "p1_win")
    log.info("matches_loaded", count=len(all_matches),
             p1_win_pct=round(p1_wins / len(all_matches), 3))
    return all_matches


# ---------------------------------------------------------------------------
# Sequential ELO
# ---------------------------------------------------------------------------

DEFAULT_ELO = 1500.0
ELO_K = 32.0  # uniform K for simplicity with this dataset


def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (r_b - r_a) / 400.0))


def elo_update(r_a: float, r_b: float, s_a: float, k: float = ELO_K) -> tuple[float, float]:
    """Return (new_r_a, new_r_b)."""
    e_a = elo_expected(r_a, r_b)
    e_b = 1.0 - e_a
    s_b = 1.0 - s_a
    return r_a + k * (s_a - e_a), r_b + k * (s_b - e_b)


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

DEFAULT_3DA = 60.0
DEFAULT_CO = 0.40


def build_features(matches: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build feature matrix with sequential ELO (no leakage).

    For each match i, ELO ratings are those computed from matches 0..i-1.
    Rolling stats (3DA, win-rate, form, H2H) also use only past data.

    Returns (X, y, match_ids).
    """
    # Sequential ELO state
    elo: dict[str, float] = {}
    elo_games: dict[str, int] = {}

    # Rolling histories: player_id → list of past match records
    player_hist: dict[str, list] = defaultdict(list)
    h2h_hist: dict[tuple, list] = defaultdict(list)

    feature_rows = []
    labels = []
    ids = []

    def get_elo(pid: str) -> float:
        return elo.get(pid, DEFAULT_ELO)

    def rolling_wr(hist: list, n: int = 100) -> float:
        if not hist:
            return 0.5
        recent = hist[-n:]
        return sum(x["won"] for x in recent) / len(recent)

    def rolling_3da(hist: list, n: int = 100) -> float:
        if not hist:
            return DEFAULT_3DA
        vals = [x["da"] for x in hist[-n:] if x["da"] > 0]
        return float(np.mean(vals)) if vals else DEFAULT_3DA

    def ewm_form(hist: list, decay: float = 0.05) -> float:
        if not hist:
            return DEFAULT_3DA
        vals = np.array([x["da"] for x in hist if x["da"] > 0])
        if len(vals) == 0:
            return DEFAULT_3DA
        weights = (1 - decay) ** np.arange(len(vals) - 1, -1, -1)
        return float(np.dot(weights, vals) / weights.sum())

    def opp_adj_form(hist: list, n: int = 50) -> float:
        if not hist:
            return 0.0
        recent = hist[-n:]
        scores = []
        for x in recent:
            opp_e = x.get("opp_elo", DEFAULT_ELO)
            expected = 1.0 / (1.0 + 10 ** ((opp_e - DEFAULT_ELO) / 400))
            actual = 1.0 if x["won"] else 0.0
            scores.append(actual - expected)
        return float(np.mean(scores))

    def h2h_wr(hh: list, pid: str) -> float:
        if not hh:
            return 0.5
        return sum(1 for x in hh if x["winner"] == pid) / len(hh)

    def days_since_last(hist: list, current_date: date | None) -> float:
        if not hist or not current_date:
            return 0.15  # normalised default (~27 days)
        last = hist[-1].get("date")
        if not last:
            return 0.15
        delta = (current_date - last).days
        return min(float(delta), 180.0) / 180.0

    for match in matches:
        p1 = match["player1_id"]
        p2 = match["player2_id"]
        mdate = match["event_date"]
        result = match["result_type"]

        # --- Pre-match ELO (sequential, no leakage) ---
        e1 = get_elo(p1)
        e2 = get_elo(p2)
        e1_games = elo_games.get(p1, 0)
        e2_games = elo_games.get(p2, 0)

        # --- Rolling histories (pre-match) ---
        h1 = player_hist[p1]
        h2 = player_hist[p2]
        h12 = h2h_hist[tuple(sorted([p1, p2]))]

        # Build feature vector (39 features)
        feat = [
            # Sequential ELO (3)
            e1,
            e2,
            e1 - e2,
            # ELO experience (2)
            min(e1_games, 200) / 200.0,
            min(e2_games, 200) / 200.0,
            # ELO expected score (1)
            elo_expected(e1, e2),
            # Rolling 3DA (4)
            rolling_3da(h1),
            rolling_3da(h2),
            ewm_form(h1),
            ewm_form(h2),
            # Win rate (2)
            rolling_wr(h1),
            rolling_wr(h2),
            # Opp-adj form (2)
            opp_adj_form(h1),
            opp_adj_form(h2),
            # Days since last match (2)
            days_since_last(h1, mdate),
            days_since_last(h2, mdate),
            # H2H (2)
            h2h_wr(h12, p1),
            min(len(h12), 20) / 20.0,
            # Match avg difference (1) — current match's 3DA
            # (These are known at result time; use rolling as proxy)
            rolling_3da(h1, n=20) - rolling_3da(h2, n=20),
        ]

        feature_rows.append(feat)
        label = 1.0 if result == "p1_win" else 0.0
        labels.append(label)
        ids.append(f"{match['event_id']}_{p1}_{p2}")

        # --- Update ELO sequentially (AFTER recording feature) ---
        s1 = 1.0 if result == "p1_win" else 0.0
        new_e1, new_e2 = elo_update(e1, e2, s1)
        elo[p1] = new_e1
        elo[p2] = new_e2
        elo_games[p1] = e1_games + 1
        elo_games[p2] = e2_games + 1

        # Update rolling histories
        p1_won = result == "p1_win"
        da1 = match["player1_avg"]
        da2 = match["player2_avg"]
        player_hist[p1].append({
            "won": p1_won, "da": da1, "opp_elo": e2, "date": mdate,
        })
        player_hist[p2].append({
            "won": not p1_won, "da": da2, "opp_elo": e1, "date": mdate,
        })
        h2h_hist[tuple(sorted([p1, p2]))].append({"winner": p1 if p1_won else p2})

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    log.info("features_built", shape=X.shape, positive_rate=float(y.mean().round(4)))
    return X, y, ids


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(X: np.ndarray, y: np.ndarray, dry_run: bool = False) -> dict:
    import joblib
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    log.info("split", train=len(X_train), val=len(X_val), test=len(X_test))

    if dry_run:
        log.info("dry_run_complete", features=X.shape[1])
        return {}

    # LightGBM
    log.info("training_lgbm")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=24,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.2,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )

    # Meta-learner: Platt calibration via logistic regression
    val_proba = lgb_model.predict_proba(X_val)[:, 1].reshape(-1, 1)
    scaler = StandardScaler()
    val_proba_s = scaler.fit_transform(val_proba)
    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(val_proba_s, y_val)

    # Test evaluation
    test_proba_lgb = lgb_model.predict_proba(X_test)[:, 1].reshape(-1, 1)
    test_proba_s = scaler.transform(test_proba_lgb)
    proba_test = meta.predict_proba(test_proba_s)[:, 1]

    brier = brier_score_loss(y_test, proba_test)
    auc = roc_auc_score(y_test, proba_test)
    log.info("r1_eval", brier=round(brier, 4), auc=round(auc, 4), test_size=len(X_test))

    SAVE_DIR.mkdir(exist_ok=True)
    artifact = {
        "lgbm": lgb_model,
        "meta": meta,
        "scaler": scaler,
        "feature_count": X.shape[1],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_size": len(X_train),
        "brier_test": round(brier, 4),
        "auc_test": round(auc, 4),
        "note": "Sequential ELO; trained from raw DartsDatabase JSON",
    }
    out_path = SAVE_DIR / "r1_model_from_raw.pkl"
    joblib.dump(artifact, out_path)
    log.info("r1_saved", path=str(out_path))
    return {"brier": brier, "auc": auc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Build features only, skip training")
    parser.add_argument("--n-matches", type=int, default=0, help="Limit matches (0=all)")
    args = parser.parse_args()

    matches = load_matches()

    if args.n_matches > 0:
        matches = matches[: args.n_matches]
        log.info("limited_matches", n=len(matches))

    X, y, ids = build_features(matches)

    if args.dry_run:
        # Show ELO distribution check
        from collections import Counter
        log.info("dry_run_feature_stats",
                 X_shape=X.shape,
                 y_mean=float(y.mean().round(4)),
                 elo_diff_range=[float(X[:, 2].min().round(1)), float(X[:, 2].max().round(1))],
                 elo_diff_std=float(X[:, 2].std().round(2)))
        return

    train(X, y, dry_run=False)


if __name__ == "__main__":
    main()
