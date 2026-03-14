"""
R2 model: Full stacking ensemble.

CatBoost + LightGBM + XGBoost + Empirical Bayes skills (+ optional LSTM).
Meta-learner: Logistic regression on out-of-fold predictions.
Post-processing: Beta calibration.
68 features including visit distributions, route choice params,
EB skill estimates, LSTM momentum state, and segment accuracies.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from calibration.beta_calibrator import BetaCalibrator
from models import DartsMLError
from models.r1_lgbm import R1_FEATURES

logger = structlog.get_logger(__name__)


# R2 features: R1 (38) + visit distributions, route choice, EB skills,
# LSTM state, segment accuracies (30 more = 68)
R2_FEATURES: list[str] = list(R1_FEATURES) + [
    # --- Conditional visit distributions by band (10) ---
    "visit_mean_open_p1",       # mean visit score in open band
    "visit_mean_open_p2",
    "visit_mean_setup_p1",      # mean visit score in setup band
    "visit_mean_setup_p2",
    "visit_std_open_p1",        # scoring variability
    "visit_std_open_p2",
    "visit_bust_rate_pressure_p1",  # bust rate in pressure band
    "visit_bust_rate_pressure_p2",
    "visit_180_rate_p1",        # 180 frequency per visit
    "visit_180_rate_p2",
    # --- Route choice parameters (4) ---
    "preferred_double_pct_p1",  # % of checkouts on preferred double
    "preferred_double_pct_p2",
    "route_aggression_p1",      # route choice aggressiveness index
    "route_aggression_p2",
    # --- EB skill estimates (6) ---
    "eb_t20_accuracy_p1",       # P(hitting T20) from EB posterior
    "eb_t20_accuracy_p2",
    "eb_d_accuracy_p1",         # P(hitting double) from EB posterior
    "eb_d_accuracy_p2",
    "eb_bull_accuracy_p1",      # P(hitting bull) from EB posterior
    "eb_bull_accuracy_p2",
    # --- LSTM momentum state (4) ---
    "lstm_momentum_p1",         # LSTM hidden state summary
    "lstm_momentum_p2",
    "lstm_volatility_p1",       # LSTM variance of momentum
    "lstm_volatility_p2",
    # --- Segment accuracies (6) ---
    "segment_t20_hit_rate_p1",  # raw T20 hit rate
    "segment_t20_hit_rate_p2",
    "segment_t19_hit_rate_p1",  # raw T19 hit rate
    "segment_t19_hit_rate_p2",
    "segment_d_hit_rate_p1",    # raw double hit rate (all doubles)
    "segment_d_hit_rate_p2",
]

R2_FEATURE_COUNT: int = len(R2_FEATURES)


def _create_catboost() -> object:
    """Create a CatBoost classifier with tuned hyperparameters."""
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        thread_count=-1,
    )


def _create_lgbm_r2() -> object:
    """Create LightGBM for R2 ensemble (deeper than R1)."""
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=15,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )


def _create_xgb_r2() -> object:
    """Create XGBoost for R2 ensemble (deeper than R1)."""
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    )


class R2StackingEnsemble:
    """
    R2 full stacking ensemble for darts match outcome prediction.

    Base learners:
        - CatBoost
        - LightGBM
        - XGBoost
        - Empirical Bayes skill model (as probability output)

    Meta-learner:
        - Logistic regression on OOF base predictions

    Post-processing:
        - Beta calibration

    Parameters
    ----------
    n_folds:
        Number of CV folds for OOF prediction generation.
    use_lstm:
        Whether to include LSTM features (requires PyTorch model).
    use_eb:
        Whether to include the Empirical Bayes base learner.
    """

    BASE_LEARNER_NAMES: list[str] = ["catboost", "lightgbm", "xgboost", "eb_skills"]

    def __init__(
        self,
        n_folds: int = 5,
        use_lstm: bool = False,
        use_eb: bool = True,
    ) -> None:
        self.base_learners: dict[str, object] = {
            "catboost": _create_catboost(),
            "lightgbm": _create_lgbm_r2(),
            "xgboost": _create_xgb_r2(),
        }
        self.use_eb = use_eb
        self.use_lstm = use_lstm
        self.eb_model: Optional[object] = None
        self.meta_learner = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        self.calibrator = BetaCalibrator(market_family="r2_stacking")
        self.n_folds = n_folds
        self.fitted_: bool = False
        self._feature_names: list[str] = list(R2_FEATURES)
        self._log = logger.bind(component="R2StackingEnsemble")

    def validate_features(self, X: pd.DataFrame) -> None:
        """
        Validate that all R2 features are present.

        Raises
        ------
        DartsMLError
            If any required feature is missing.
        """
        missing = set(R2_FEATURES) - set(X.columns)
        if missing:
            raise DartsMLError(
                f"Missing R2 features: {sorted(missing)}. "
                f"Required: {R2_FEATURE_COUNT} features."
            )

        r2_data = X[R2_FEATURES]
        nan_cols = r2_data.columns[r2_data.isnull().any()].tolist()
        if nan_cols:
            raise DartsMLError(
                f"R2 features contain NaN values: {nan_cols}"
            )

    def set_eb_model(self, eb_model: object) -> None:
        """
        Attach an Empirical Bayes skill model as a base learner.

        Parameters
        ----------
        eb_model:
            An EmpiricalBayesSkillModel instance with a predict_proba method.
        """
        if not hasattr(eb_model, "predict_proba"):
            raise DartsMLError(
                "EB model must have a predict_proba method."
            )
        self.eb_model = eb_model
        self._log.info("eb_model_attached")

    def _n_base_learners(self) -> int:
        """Number of active base learners."""
        n = len(self.base_learners)
        if self.use_eb and self.eb_model is not None:
            n += 1
        return n

    def _create_base_learner(self, name: str) -> object:
        """Create a fresh base learner by name."""
        creators = {
            "catboost": _create_catboost,
            "lightgbm": _create_lgbm_r2,
            "xgboost": _create_xgb_r2,
        }
        if name not in creators:
            raise DartsMLError(f"Unknown base learner: {name}")
        return creators[name]()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> None:
        """
        Fit the R2 stacking ensemble.

        Steps:
            1. Generate OOF predictions from tree-based base learners.
            2. If EB model attached, generate EB predictions as additional column.
            3. Fit meta-learner on OOF predictions.
            4. Refit base learners on full training data.
            5. Fit Beta calibrator on validation set predictions.

        Parameters
        ----------
        X_train:
            Training features (n_train, 68).
        y_train:
            Training labels.
        X_val:
            Validation features.
        y_val:
            Validation labels.
        """
        self.validate_features(X_train)
        self.validate_features(X_val)

        y_train = np.asarray(y_train, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)

        X_tr = X_train[R2_FEATURES].values.astype(np.float64)
        X_v = X_val[R2_FEATURES].values.astype(np.float64)

        n_samples = len(X_tr)
        n_base = self._n_base_learners()

        # Step 1: Generate OOF predictions from tree learners
        oof_preds = np.zeros((n_samples, n_base), dtype=np.float64)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_tr)):
            X_fold_train = X_tr[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_tr[val_idx]

            for i, (name, _) in enumerate(self.base_learners.items()):
                fold_learner = self._create_base_learner(name)
                fold_learner.fit(X_fold_train, y_fold_train)  # type: ignore[union-attr]
                fold_proba = fold_learner.predict_proba(X_fold_val)  # type: ignore[union-attr]
                oof_preds[val_idx, i] = fold_proba[:, 1]

            self._log.debug(
                "r2_fold_complete",
                fold=fold_idx + 1,
                n_folds=self.n_folds,
            )

        # Step 2: EB predictions as additional base learner
        if self.use_eb and self.eb_model is not None:
            eb_idx = len(self.base_learners)
            eb_proba = self.eb_model.predict_proba(X_tr)  # type: ignore[union-attr]
            if eb_proba.ndim == 2:
                oof_preds[:, eb_idx] = eb_proba[:, 1]
            else:
                oof_preds[:, eb_idx] = eb_proba

        # Step 3: Fit meta-learner
        self.meta_learner.fit(oof_preds, y_train)

        # Step 4: Refit base learners on full data
        for name, learner in self.base_learners.items():
            learner.fit(X_tr, y_train)  # type: ignore[union-attr]

        # Step 5: Fit calibrator
        val_meta_input = self._base_predict(X_v)
        val_raw_proba = self.meta_learner.predict_proba(val_meta_input)[:, 1]

        if len(np.unique(y_val)) >= 2:
            self.calibrator.fit(val_raw_proba, y_val)

        self.fitted_ = True

        self._log.info(
            "r2_model_fitted",
            n_train=len(X_train),
            n_val=len(X_val),
            n_features=R2_FEATURE_COUNT,
            n_base_learners=n_base,
            n_folds=self.n_folds,
        )

    def _base_predict(self, X: np.ndarray) -> np.ndarray:
        """Generate base learner predictions for meta-learner input."""
        n_base = self._n_base_learners()
        preds = np.zeros((len(X), n_base), dtype=np.float64)

        for i, (name, learner) in enumerate(self.base_learners.items()):
            proba = learner.predict_proba(X)  # type: ignore[union-attr]
            preds[:, i] = proba[:, 1]

        if self.use_eb and self.eb_model is not None:
            eb_idx = len(self.base_learners)
            eb_proba = self.eb_model.predict_proba(X)  # type: ignore[union-attr]
            if eb_proba.ndim == 2:
                preds[:, eb_idx] = eb_proba[:, 1]
            else:
                preds[:, eb_idx] = eb_proba

        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated P(P1 wins).

        Parameters
        ----------
        X:
            Feature dataframe with R2 features.

        Returns
        -------
        np.ndarray
            Calibrated P(P1 wins), shape (n_samples,).
        """
        if not self.fitted_:
            raise DartsMLError(
                "R2 model has not been fitted. Call .fit() first."
            )
        self.validate_features(X)

        X_r2 = X[R2_FEATURES].values.astype(np.float64)
        base_preds = self._base_predict(X_r2)
        raw_proba = self.meta_learner.predict_proba(base_preds)[:, 1]

        if self.calibrator.fitted_:
            return self.calibrator.predict_proba(raw_proba)
        return raw_proba

    def predict_proba_full(self, X: pd.DataFrame) -> np.ndarray:
        """Return [P(P2 wins), P(P1 wins)] array."""
        p1 = self.predict_proba(X)
        return np.column_stack([1.0 - p1, p1])

    def save(self, path: str | pathlib.Path) -> None:
        """Serialize the ensemble to disk."""
        if not self.fitted_:
            raise DartsMLError("Cannot save unfitted R2 model.")
        state = {
            "base_learners": self.base_learners,
            "meta_learner": self.meta_learner,
            "calibrator": self.calibrator,
            "feature_names": self._feature_names,
            "n_folds": self.n_folds,
            "use_eb": self.use_eb,
            "use_lstm": self.use_lstm,
        }
        joblib.dump(state, path)
        self._log.info("r2_model_saved", path=str(path))

    def load(self, path: str | pathlib.Path) -> None:
        """Load a previously saved ensemble."""
        path = pathlib.Path(path)
        if not path.exists():
            raise DartsMLError(f"Model file not found: {path}")
        state = joblib.load(path)
        self.base_learners = state["base_learners"]
        self.meta_learner = state["meta_learner"]
        self.calibrator = state["calibrator"]
        self._feature_names = state["feature_names"]
        self.n_folds = state["n_folds"]
        self.use_eb = state["use_eb"]
        self.use_lstm = state["use_lstm"]
        self.fitted_ = True
        self._log.info("r2_model_loaded", path=str(path))
