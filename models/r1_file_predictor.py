"""
File-based R1 match-winner predictor.

Loads r1_model.pkl (38 features, AUC=0.8182) from disk.
Falls back to r1_model_from_raw.pkl (19 features, AUC=0.8097) if absent.

38-feature vector matches train_r1.py:
  ELO (5), Format/context (5), Rolling form (6), Form diffs (3),
  Opp-adj form (3), Stage/floor 3DA (4), Experience/rest (4),
  H2H (2), Match quality / cross signals (6).

When per-match history is unavailable (typical pre-match API call),
statistically-neutral defaults are used — the same DEFAULT_3DA=50.0
and DEFAULT_ELO=1500.0 as training.

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
_DEFAULT_3DA = 50.0       # matches training DEFAULT_3DA (no career-avg leakage)
_DEFAULT_WR = 0.5
_DEFAULT_OPP_ADJ = 0.0
_DEFAULT_REST_NORM = 0.167   # ~30 days / 180
_DEFAULT_H2H_WR = 0.5
_DEFAULT_H2H_COUNT = 0.0


def _elo_expected(r_a: float, r_b: float) -> float:
    return r_a / (r_a + r_b + 1e-9)


# ---------------------------------------------------------------------------
# Format / context encoders — mirror train_r1.py exactly
# ---------------------------------------------------------------------------

def _format_enc(fmt: str) -> float:
    m = {
        "PDC_WC": 1, "PDC_WC_ERA_2020": 1,
        "PDC_PL": 2, "PDC_WM": 3, "PDC_GS": 4,
        "PDC_GP": 5, "PDC_PCF": 6, "PDC_UK": 7,
    }
    return m.get(fmt, 0) / 7.0


def _stage_enc(fmt: str) -> float:
    if "WC" in fmt:
        return 0.0
    if "WM" in fmt or "GS" in fmt:
        return 1.0
    return 0.5


def _format_len_enc(fmt: str) -> float:
    return 1.0 if len(fmt) <= 6 else 0.0


def _ecosystem_enc(eco: str) -> float:
    m = {"pdc_mens": 1.0, "pdc_womens": 0.6, "wdf": 0.5, "development": 0.3}
    return m.get(eco, 0.5)


# ---------------------------------------------------------------------------
# 38-feature vector builder — mirrors train_r1.py feat list exactly
# ---------------------------------------------------------------------------

def _build_feature_vector_38(
    p1_elo: float,
    p2_elo: float,
    p1_3da: float,
    p2_3da: float,
    *,
    # Rolling form (optional — defaults to training neutrals)
    p1_ewm_form: float = _DEFAULT_3DA,
    p2_ewm_form: float = _DEFAULT_3DA,
    p1_rolling_wr: float = _DEFAULT_WR,
    p2_rolling_wr: float = _DEFAULT_WR,
    p1_opp_adj: float = _DEFAULT_OPP_ADJ,
    p2_opp_adj: float = _DEFAULT_OPP_ADJ,
    p1_stage_3da: float = _DEFAULT_3DA,
    p2_stage_3da: float = _DEFAULT_3DA,
    p1_floor_3da: float = _DEFAULT_3DA,
    p2_floor_3da: float = _DEFAULT_3DA,
    p1_match_count_norm: float = 0.3,   # ~150 matches (50th pct) / 500
    p2_match_count_norm: float = 0.3,
    p1_days_since_norm: float = _DEFAULT_REST_NORM,
    p2_days_since_norm: float = _DEFAULT_REST_NORM,
    h2h_wr: float = _DEFAULT_H2H_WR,
    h2h_count_norm: float = _DEFAULT_H2H_COUNT,
    # Competition context (optional)
    format_code: str = "",
    ecosystem: str = "pdc_mens",
    is_televised: bool = True,
) -> list[float]:
    """
    Build 38-element feature vector matching train_r1.py exactly.

    Feature order (must match FEATURE_NAMES in train_r1.py):
      [0-4]   ELO: elo1, elo2, elo_diff, abs_elo_diff, elo_win_prob
      [5-9]   Format: format_enc, stage_enc, format_len_enc, ecosystem_enc, is_televised
      [10-15] Rolling: ewm_form_p1/p2, rolling_3da_p1/p2, rolling_wr_p1/p2
      [16-18] Form diffs: ewm_diff, rda_diff, wr_diff
      [19-21] Opp-adj: opp_adj_p1/p2, opp_adj_diff
      [22-25] Stage/floor 3DA: stage_3da_p1/p2, floor_3da_p1/p2
      [26-29] Experience/rest: match_count_p1/p2, days_since_p1/p2
      [30-31] H2H: h2h_wr, h2h_count
      [32-37] Cross signals: match_level, days_rest_diff, exp_gap,
                              form_x_wr_p1, form_x_wr_p2, rda_x_wr_diff
    """
    e1, e2 = p1_elo, p2_elo
    elo_win_prob = _elo_expected(e1, e2)
    elo_sum = (e1 + e2) / (2.0 * _DEFAULT_ELO)

    return [
        # ELO (5)
        e1,
        e2,
        e1 - e2,
        abs(e1 - e2),
        elo_win_prob,
        # Format / context (5)
        _format_enc(format_code),
        _stage_enc(format_code),
        _format_len_enc(format_code),
        _ecosystem_enc(ecosystem),
        float(is_televised),
        # Rolling form (6)
        p1_ewm_form,
        p2_ewm_form,
        p1_3da,
        p2_3da,
        p1_rolling_wr,
        p2_rolling_wr,
        # Form diffs (3)
        p1_ewm_form - p2_ewm_form,
        p1_3da - p2_3da,
        p1_rolling_wr - p2_rolling_wr,
        # Opp-adjusted form (3)
        p1_opp_adj,
        p2_opp_adj,
        p1_opp_adj - p2_opp_adj,
        # Stage / floor 3DA (4)
        p1_stage_3da,
        p2_stage_3da,
        p1_floor_3da,
        p2_floor_3da,
        # Experience / rest (4)
        p1_match_count_norm,
        p2_match_count_norm,
        p1_days_since_norm,
        p2_days_since_norm,
        # H2H (2)
        h2h_wr,
        h2h_count_norm,
        # Match quality / cross signals (6)
        elo_sum,
        p1_days_since_norm - p2_days_since_norm,
        p1_match_count_norm - p2_match_count_norm,
        p1_ewm_form * p1_rolling_wr,
        p2_ewm_form * p2_rolling_wr,
        (p1_3da * p1_rolling_wr) - (p2_3da * p2_rolling_wr),
    ]


# ---------------------------------------------------------------------------
# 19-feature vector builder — legacy (r1_model_from_raw.pkl fallback)
# ---------------------------------------------------------------------------

def _build_feature_vector_19(
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
    e1, e2 = p1_elo, p2_elo
    elo_exp = 1.0 / (1.0 + math.pow(10.0, (e2 - e1) / 400.0))
    return [
        e1, e2, e1 - e2,
        min(p1_elo_games, 200) / 200.0,
        min(p2_elo_games, 200) / 200.0,
        elo_exp,
        p1_3da, p2_3da,
        p1_3da,   # ewm_form ≈ 3DA
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


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class R1FilePredictor:
    """
    R1 match-winner predictor loaded from a pkl file on disk.

    Tries r1_model.pkl (38 features, AUC≈0.818) first.
    Falls back to r1_model_from_raw.pkl (19 features, AUC≈0.810) if absent.

    Thread-safe singleton via module-level instance.
    Returns None if no model file is available (caller falls back to R0).
    """

    _SAVED_DIR = pathlib.Path(__file__).resolve().parent / "saved"
    _FULL_MODEL_PATH = _SAVED_DIR / "r1_model.pkl"
    _RAW_MODEL_PATH  = _SAVED_DIR / "r1_model_from_raw.pkl"

    def __init__(self) -> None:
        self._artifact: Optional[dict] = None
        self._n_features: int = 0
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
                # Prefer the full 38-feature model
                if self._FULL_MODEL_PATH.exists():
                    self._artifact = joblib.load(self._FULL_MODEL_PATH)
                    self._n_features = self._artifact.get("feature_count", 38)
                    self._log.info(
                        "r1_full_model_loaded",
                        path=str(self._FULL_MODEL_PATH),
                        auc=self._artifact.get("auc_test"),
                        brier=self._artifact.get("brier_test"),
                        n_features=self._n_features,
                    )
                elif self._RAW_MODEL_PATH.exists():
                    self._artifact = joblib.load(self._RAW_MODEL_PATH)
                    self._n_features = self._artifact.get("feature_count", 19)
                    self._log.info(
                        "r1_raw_model_loaded_fallback",
                        path=str(self._RAW_MODEL_PATH),
                        auc=self._artifact.get("auc_test"),
                        n_features=self._n_features,
                    )
                else:
                    self._log.warning("r1_model_not_found",
                                      full=str(self._FULL_MODEL_PATH),
                                      raw=str(self._RAW_MODEL_PATH))
                    return False
                return True
            except Exception as exc:
                self._log.warning("r1_file_model_unavailable", error=str(exc))
                return False

    @property
    def n_features(self) -> int:
        """Number of features the loaded model expects (0 if not loaded)."""
        self._ensure_loaded()
        return self._n_features

    def predict(
        self,
        p1_elo: float = _DEFAULT_ELO,
        p2_elo: float = _DEFAULT_ELO,
        p1_3da: float = _DEFAULT_3DA,
        p2_3da: float = _DEFAULT_3DA,
        *,
        # 38-feature extras (used when full model is loaded)
        p1_ewm_form: Optional[float] = None,
        p2_ewm_form: Optional[float] = None,
        p1_rolling_wr: float = _DEFAULT_WR,
        p2_rolling_wr: float = _DEFAULT_WR,
        p1_opp_adj: float = _DEFAULT_OPP_ADJ,
        p2_opp_adj: float = _DEFAULT_OPP_ADJ,
        p1_stage_3da: Optional[float] = None,
        p2_stage_3da: Optional[float] = None,
        p1_floor_3da: Optional[float] = None,
        p2_floor_3da: Optional[float] = None,
        p1_match_count_norm: float = 0.3,
        p2_match_count_norm: float = 0.3,
        p1_days_since_norm: float = _DEFAULT_REST_NORM,
        p2_days_since_norm: float = _DEFAULT_REST_NORM,
        h2h_wr: float = _DEFAULT_H2H_WR,
        h2h_count_norm: float = _DEFAULT_H2H_COUNT,
        format_code: str = "",
        ecosystem: str = "pdc_mens",
        is_televised: bool = True,
        # 19-feature legacy extras (used when raw model is loaded)
        p1_elo_games: int = 0,
        p2_elo_games: int = 0,
        h2h_p1_wr: float = _DEFAULT_H2H_WR,
    ) -> Optional[float]:
        """
        Predict P(player1 wins).

        Returns None if no model file is available.

        Core parameters (available from DB/DartsOrakel):
            p1_elo, p2_elo: ELO ratings
            p1_3da, p2_3da: Three-dart averages

        Extended parameters improve accuracy for the 38-feature model.
        All are optional — sensible defaults matching training are used.
        """
        if not self._ensure_loaded():
            return None

        import numpy as np

        art = self._artifact
        lgbm = art["lgbm"]
        scaler = art["scaler"]
        meta = art["meta"]

        if self._n_features == 38:
            feat = _build_feature_vector_38(
                p1_elo, p2_elo, p1_3da, p2_3da,
                p1_ewm_form=p1_ewm_form if p1_ewm_form is not None else p1_3da,
                p2_ewm_form=p2_ewm_form if p2_ewm_form is not None else p2_3da,
                p1_rolling_wr=p1_rolling_wr,
                p2_rolling_wr=p2_rolling_wr,
                p1_opp_adj=p1_opp_adj,
                p2_opp_adj=p2_opp_adj,
                p1_stage_3da=p1_stage_3da if p1_stage_3da is not None else p1_3da,
                p2_stage_3da=p2_stage_3da if p2_stage_3da is not None else p2_3da,
                p1_floor_3da=p1_floor_3da if p1_floor_3da is not None else p1_3da,
                p2_floor_3da=p2_floor_3da if p2_floor_3da is not None else p2_3da,
                p1_match_count_norm=p1_match_count_norm,
                p2_match_count_norm=p2_match_count_norm,
                p1_days_since_norm=p1_days_since_norm,
                p2_days_since_norm=p2_days_since_norm,
                h2h_wr=h2h_wr,
                h2h_count_norm=h2h_count_norm,
                format_code=format_code,
                ecosystem=ecosystem,
                is_televised=is_televised,
            )
        else:
            feat = _build_feature_vector_19(
                p1_elo, p2_elo, p1_3da, p2_3da,
                p1_elo_games=p1_elo_games,
                p2_elo_games=p2_elo_games,
                p1_win_rate=p1_rolling_wr,
                p2_win_rate=p2_rolling_wr,
                p1_opp_adj=p1_opp_adj,
                p2_opp_adj=p2_opp_adj,
                p1_rest_norm=p1_days_since_norm,
                p2_rest_norm=p2_days_since_norm,
                h2h_p1_wr=h2h_p1_wr,
                h2h_count_norm=h2h_count_norm,
            )

        if self._n_features == 38:
            import pandas as pd
            feature_names = art.get("feature_names")
            if feature_names:
                X = pd.DataFrame([feat], columns=feature_names)
            else:
                X = np.array([feat], dtype=np.float32)
        else:
            X = np.array([feat], dtype=np.float32)

        lgbm_proba = lgbm.predict_proba(X)[:, 1].reshape(-1, 1)
        lgbm_scaled = scaler.transform(lgbm_proba)
        p1_win = float(meta.predict_proba(lgbm_scaled)[0, 1])
        return p1_win


    def warmup(self) -> None:
        """Run a dummy prediction at startup to eliminate first-request 2s cold-start.

        LightGBM + sklearn initialize their thread pools on the first predict_proba
        call. Calling this once during the lifespan means the real first customer
        prediction pays no warmup cost.
        """
        try:
            result = self.predict(
                p1_elo=_DEFAULT_ELO,
                p2_elo=_DEFAULT_ELO,
                p1_3da=_DEFAULT_3DA,
                p2_3da=_DEFAULT_3DA,
            )
            if result is not None:
                self._log.info("r1_model_warmed_up", dummy_prob=round(result, 4))
            else:
                self._log.warning("r1_warmup_skipped_no_model")
        except Exception as exc:
            self._log.warning("r1_warmup_error", error=str(exc))


# Module-level singleton
r1_file_predictor = R1FilePredictor()
