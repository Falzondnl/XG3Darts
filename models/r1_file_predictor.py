"""
File-based R1 match-winner predictor.

Loads r1_model_from_raw.pkl directly from disk (no DB required).
Uses simplified 19-feature vector from available match context:
  ELO (3), ELO games (2), ELO expected (1), 3DA (4), win-rate (2),
  opp-adj form (2), rest days (2), H2H (2), 3DA diff (1).

When player history is unavailable (no DartsDatabase ID lookup),
sensible defaults are used for rolling stats.
ELO values are from the production darts_players table (or fallback 1500).

Usage:
    from models.r1_file_predictor import R1FilePredictor
    pred = R1FilePredictor()
    p1_win = pred.predict(p1_elo=1650, p2_elo=1520, p1_3da=96.2, p2_3da=91.5)
"""
from __future__ import annotations

import math
import pathlib
import threading
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_ELO = 1500.0
_DEFAULT_3DA = 60.0
_DEFAULT_WR = 0.5
_DEFAULT_OPP_ADJ = 0.0
_DEFAULT_REST_NORM = 0.15   # ~27 days normalised to [0,1] over 180 days
_DEFAULT_H2H_WR = 0.5
_DEFAULT_H2H_COUNT = 0.0


def _elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (r_b - r_a) / 400.0))


def _build_feature_vector(
    p1_elo: float,
    p2_elo: float,
    p1_3da: float,
    p2_3da: float,
    p1_elo_games: int = 0,
    p2_elo_games: int = 0,
    p1_win_rate: float = _DEFAULT_WR,
    p2_win_rate: float = _DEFAULT_WR,
    p1_opp_adj: float = _DEFAULT_OPP_ADJ,
    p2_opp_adj: float = _DEFAULT_OPP_ADJ,
    p1_rest_norm: float = _DEFAULT_REST_NORM,
    p2_rest_norm: float = _DEFAULT_REST_NORM,
    h2h_p1_wr: float = _DEFAULT_H2H_WR,
    h2h_count_norm: float = _DEFAULT_H2H_COUNT,
) -> list[float]:
    """
    Build 19-element feature vector matching train_r1_from_raw.py.

    Feature order (must match training):
      0  e1                    (p1 ELO)
      1  e2                    (p2 ELO)
      2  e1 - e2               (ELO diff)
      3  min(g1,200)/200       (p1 ELO experience, 0-1)
      4  min(g2,200)/200       (p2 ELO experience, 0-1)
      5  elo_expected(e1,e2)   (p1 expected win prob from ELO)
      6  rolling_3da_p1        (p1 mean 3DA from last 100 matches)
      7  rolling_3da_p2
      8  ewm_form_p1           (EWM 3DA, decay=0.05)
      9  ewm_form_p2
     10  rolling_wr_p1         (p1 win rate last 100)
     11  rolling_wr_p2
     12  opp_adj_form_p1       (form adj for opp quality)
     13  opp_adj_form_p2
     14  days_since_last_p1    (normalised 0-1 over 180d)
     15  days_since_last_p2
     16  h2h_p1_win_rate
     17  h2h_count_norm        (min(count,20)/20)
     18  3da_diff              (rolling_3da_p1(n=20) - rolling_3da_p2(n=20))
    """
    e1 = p1_elo
    e2 = p2_elo
    return [
        e1,
        e2,
        e1 - e2,
        min(p1_elo_games, 200) / 200.0,
        min(p2_elo_games, 200) / 200.0,
        _elo_expected(e1, e2),
        p1_3da,
        p2_3da,
        p1_3da,   # ewm_form ≈ 3DA without per-match history
        p2_3da,
        p1_win_rate,
        p2_win_rate,
        p1_opp_adj,
        p2_opp_adj,
        p1_rest_norm,
        p2_rest_norm,
        h2h_p1_wr,
        h2h_count_norm,
        p1_3da - p2_3da,
    ]


class R1FilePredictor:
    """
    R1 match-winner predictor loaded from a pkl file on disk.

    Thread-safe singleton via module-level instance.

    Loads lazily on first predict() call.
    Returns None if the pkl file is missing (caller should fall back to R0).
    """

    _MODEL_PATH = (
        pathlib.Path(__file__).resolve().parent / "saved" / "r1_model_from_raw.pkl"
    )

    def __init__(self) -> None:
        self._artifact: Optional[dict] = None
        self._load_attempted = False
        self._lock = threading.Lock()
        self._log = logger.bind(component="R1FilePredictor")

    def _ensure_loaded(self) -> bool:
        """Lazy-load the pkl. Returns True if model is available."""
        with self._lock:
            if self._load_attempted:
                return self._artifact is not None
            self._load_attempted = True
            try:
                import joblib
                self._artifact = joblib.load(self._MODEL_PATH)
                self._log.info(
                    "r1_file_model_loaded",
                    path=str(self._MODEL_PATH),
                    auc=self._artifact.get("auc_test"),
                    brier=self._artifact.get("brier_test"),
                    feature_count=self._artifact.get("feature_count"),
                )
                return True
            except Exception as exc:
                self._log.warning("r1_file_model_unavailable", error=str(exc))
                return False

    def predict(
        self,
        p1_elo: float = _DEFAULT_ELO,
        p2_elo: float = _DEFAULT_ELO,
        p1_3da: float = _DEFAULT_3DA,
        p2_3da: float = _DEFAULT_3DA,
        *,
        p1_elo_games: int = 0,
        p2_elo_games: int = 0,
        p1_win_rate: float = _DEFAULT_WR,
        p2_win_rate: float = _DEFAULT_WR,
        p1_opp_adj: float = _DEFAULT_OPP_ADJ,
        p2_opp_adj: float = _DEFAULT_OPP_ADJ,
        p1_rest_norm: float = _DEFAULT_REST_NORM,
        p2_rest_norm: float = _DEFAULT_REST_NORM,
        h2h_p1_wr: float = _DEFAULT_H2H_WR,
        h2h_count_norm: float = _DEFAULT_H2H_COUNT,
    ) -> Optional[float]:
        """
        Predict P(player1 wins).

        Returns None if the model file is unavailable.

        Parameters
        ----------
        p1_elo, p2_elo:
            ELO ratings (from darts_players or default 1500).
        p1_3da, p2_3da:
            Three-dart average (from darts_orakel_stats or request fallback).
        All other kwargs:
            Optional per-player history stats. When omitted, sensible
            defaults matching the training distribution are used.

        Returns
        -------
        float in [0, 1] — P(p1 wins), or None if unavailable.
        """
        if not self._ensure_loaded():
            return None

        import numpy as np

        feat = _build_feature_vector(
            p1_elo, p2_elo, p1_3da, p2_3da,
            p1_elo_games=p1_elo_games,
            p2_elo_games=p2_elo_games,
            p1_win_rate=p1_win_rate,
            p2_win_rate=p2_win_rate,
            p1_opp_adj=p1_opp_adj,
            p2_opp_adj=p2_opp_adj,
            p1_rest_norm=p1_rest_norm,
            p2_rest_norm=p2_rest_norm,
            h2h_p1_wr=h2h_p1_wr,
            h2h_count_norm=h2h_count_norm,
        )

        X = np.array([feat], dtype=np.float32)

        art = self._artifact
        lgbm = art["lgbm"]
        scaler = art["scaler"]
        meta = art["meta"]

        lgbm_proba = lgbm.predict_proba(X)[:, 1].reshape(-1, 1)
        lgbm_scaled = scaler.transform(lgbm_proba)
        p1_win = float(meta.predict_proba(lgbm_scaled)[0, 1])
        return p1_win


# Module-level singleton
r1_file_predictor = R1FilePredictor()
