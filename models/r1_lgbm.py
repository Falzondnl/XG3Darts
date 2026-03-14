"""
R1 model: LightGBM + XGBoost stacking with logistic regression meta-learner.

38 features. Beta calibration applied post-prediction.
No data leakage -- temporal split only.

Stacking architecture:
    Base learners:
        - LightGBM (gradient boosting on decision trees)
        - XGBoost (gradient boosting with regularisation)
    Meta-learner:
        - Logistic regression on out-of-fold base learner predictions
    Post-processing:
        - Beta calibration for probability calibration
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

logger = structlog.get_logger(__name__)


# R1 features: R0 (14) + rolling stats, hold/break, stage/floor, H2H (24 more = 38)
R1_FEATURES: list[str] = [
    # --- R0 features (14) ---
    "elo_p1", "elo_p2", "elo_diff",
    "ranking_p1", "ranking_p2", "ranking_log_ratio",
    "three_da_p1_pdc", "three_da_p2_pdc",
    "checkout_pct_p1_pdc", "checkout_pct_p2_pdc",
    "format_type_encoded",
    "stage_floor",
    "short_format",
    "ecosystem_encoded",
    # --- Rolling stats (8) ---
    "ewm_form_p1",              # EWM 3DA with decay=0.05
    "ewm_form_p2",
    "rolling_win_rate_p1",      # win rate over last 200 matches
    "rolling_win_rate_p2",
    "rolling_3da_p1",           # rolling mean 3DA
    "rolling_3da_p2",
    "rolling_checkout_pct_p1",  # rolling checkout %
    "rolling_checkout_pct_p2",
    # --- Hold/break rates (4) ---
    "hold_rate_p1",             # P(win leg when starting)
    "hold_rate_p2",
    "break_rate_p1",            # P(win leg when receiving)
    "break_rate_p2",
    # --- Stage/floor splits (4) ---
    "stage_3da_p1",             # 3DA in stage (televised) matches
    "stage_3da_p2",
    "floor_3da_p1",             # 3DA in floor matches
    "floor_3da_p2",
    # --- Throw-first/second (2) ---
    "throw_first_win_rate_p1",
    "throw_first_win_rate_p2",
    # --- Opponent-adjusted form (2) ---
    "opp_adj_form_p1",          # form adjusted for opponent quality
    "opp_adj_form_p2",
    # --- Rest/travel (2) ---
    "days_since_last_match_p1",
    "days_since_last_match_p2",
    # --- H2H (2) ---
    "h2h_p1_win_rate",          # P1 win rate in H2H encounters
    "h2h_total_matches",        # total H2H matches played
]

R1_FEATURE_COUNT: int = len(R1_FEATURES)


def _create_lgbm() -> object:
    """Create a LightGBM classifier with tuned hyperparameters."""
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )


def _create_xgb() -> object:
    """Create an XGBoost classifier with tuned hyperparameters."""
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    )


class R1StackingModel:
    """
    R1 stacking model: LightGBM + XGBoost base learners with
    logistic regression meta-learner and Beta calibration.

    Parameters
    ----------
    n_folds:
        Number of folds for generating OOF predictions.
    calibrate:
        Whether to apply Beta calibration post-prediction.
    """

    def __init__(
        self,
        n_folds: int = 5,
        calibrate: bool = True,
    ) -> None:
        self.base_learners: dict[str, object] = {
            "lgbm": _create_lgbm(),
            "xgb": _create_xgb(),
        }
        self.meta_learner = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        self.calibrator = BetaCalibrator(market_family="r1_stacking")
        self.n_folds = n_folds
        self._calibrate = calibrate
        self.fitted_: bool = False
        self._feature_names: list[str] = list(R1_FEATURES)
        self._log = logger.bind(component="R1StackingModel")

    def validate_features(self, X: pd.DataFrame) -> None:
        """
        Validate that all R1 features are present.

        Raises
        ------
        DartsMLError
            If any required feature is missing or contains NaN/inf.
        """
        missing = set(R1_FEATURES) - set(X.columns)
        if missing:
            raise DartsMLError(
                f"Missing R1 features: {sorted(missing)}. "
                f"Required: {len(R1_FEATURES)} features."
            )

        r1_data = X[R1_FEATURES]
        nan_cols = r1_data.columns[r1_data.isnull().any()].tolist()
        if nan_cols:
            raise DartsMLError(
                f"R1 features contain NaN values: {nan_cols}"
            )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> None:
        """
        Fit the stacking model.

        1. Generate out-of-fold predictions from base learners on X_train.
        2. Fit meta-learner on OOF predictions.
        3. Refit base learners on full X_train.
        4. Fit calibrator on X_val predictions.

        Parameters
        ----------
        X_train:
            Training features.
        y_train:
            Training labels.
        X_val:
            Validation features (for calibration).
        y_val:
            Validation labels.

        Raises
        ------
        DartsMLError
            If features are invalid or fitting fails.
        """
        self.validate_features(X_train)
        self.validate_features(X_val)

        y_train = np.asarray(y_train, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)

        X_tr = X_train[R1_FEATURES].values.astype(np.float64)
        X_v = X_val[R1_FEATURES].values.astype(np.float64)

        n_samples = len(X_tr)
        n_base = len(self.base_learners)

        # Step 1: Generate OOF predictions
        oof_preds = np.zeros((n_samples, n_base), dtype=np.float64)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_tr)):
            X_fold_train = X_tr[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_tr[val_idx]

            for i, (name, learner) in enumerate(self.base_learners.items()):
                # Create fresh learner for fold
                if name == "lgbm":
                    fold_learner = _create_lgbm()
                else:
                    fold_learner = _create_xgb()

                fold_learner.fit(X_fold_train, y_fold_train)  # type: ignore[union-attr]
                fold_proba = fold_learner.predict_proba(X_fold_val)  # type: ignore[union-attr]
                oof_preds[val_idx, i] = fold_proba[:, 1]

            self._log.debug(
                "r1_fold_complete",
                fold=fold_idx + 1,
                n_folds=self.n_folds,
            )

        # Step 2: Fit meta-learner on OOF predictions
        self.meta_learner.fit(oof_preds, y_train)

        # Step 3: Refit base learners on full training data
        for name, learner in self.base_learners.items():
            learner.fit(X_tr, y_train)  # type: ignore[union-attr]

        # Step 4: Fit calibrator on validation predictions
        val_meta_input = self._base_predict(X_v)
        val_raw_proba = self.meta_learner.predict_proba(val_meta_input)[:, 1]

        if self._calibrate and len(np.unique(y_val)) >= 2:
            self.calibrator.fit(val_raw_proba, y_val)

        self.fitted_ = True

        self._log.info(
            "r1_model_fitted",
            n_train=len(X_train),
            n_val=len(X_val),
            n_features=R1_FEATURE_COUNT,
            n_folds=self.n_folds,
            train_class_balance=float(np.mean(y_train)),
            val_class_balance=float(np.mean(y_val)),
        )

    def _base_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate base learner predictions for the meta-learner.

        Parameters
        ----------
        X:
            Feature array, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Base learner predictions, shape (n_samples, n_base_learners).
        """
        preds = np.zeros((len(X), len(self.base_learners)), dtype=np.float64)
        for i, (name, learner) in enumerate(self.base_learners.items()):
            proba = learner.predict_proba(X)  # type: ignore[union-attr]
            preds[:, i] = proba[:, 1]
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated probabilities.

        Parameters
        ----------
        X:
            Feature dataframe with R1 features.

        Returns
        -------
        np.ndarray
            Calibrated P(P1 wins) for each sample, shape (n_samples,).

        Raises
        ------
        DartsMLError
            If the model is not fitted or features are invalid.
        """
        if not self.fitted_:
            raise DartsMLError(
                "R1 model has not been fitted. Call .fit() first."
            )
        self.validate_features(X)

        X_r1 = X[R1_FEATURES].values.astype(np.float64)
        base_preds = self._base_predict(X_r1)
        raw_proba = self.meta_learner.predict_proba(base_preds)[:, 1]

        if self._calibrate and self.calibrator.fitted_:
            return self.calibrator.predict_proba(raw_proba)
        return raw_proba

    def predict_proba_full(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return full [P(P2 wins), P(P1 wins)] array.

        Parameters
        ----------
        X:
            Feature dataframe.

        Returns
        -------
        np.ndarray
            Shape (n_samples, 2).
        """
        p1 = self.predict_proba(X)
        return np.column_stack([1.0 - p1, p1])

    def save(self, path: str | pathlib.Path) -> None:
        """Serialize the model to disk."""
        if not self.fitted_:
            raise DartsMLError("Cannot save unfitted R1 model.")
        state = {
            "base_learners": self.base_learners,
            "meta_learner": self.meta_learner,
            "calibrator": self.calibrator,
            "feature_names": self._feature_names,
            "n_folds": self.n_folds,
        }
        joblib.dump(state, path)
        self._log.info("r1_model_saved", path=str(path))

    def load(self, path: str | pathlib.Path) -> None:
        """Load a previously saved model."""
        path = pathlib.Path(path)
        if not path.exists():
            raise DartsMLError(f"Model file not found: {path}")
        state = joblib.load(path)
        self.base_learners = state["base_learners"]
        self.meta_learner = state["meta_learner"]
        self.calibrator = state["calibrator"]
        self._feature_names = state["feature_names"]
        self.n_folds = state["n_folds"]
        self.fitted_ = True
        self._log.info("r1_model_loaded", path=str(path))
