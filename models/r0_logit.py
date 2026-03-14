"""
R0 model: Logistic regression on 14 features.

No data leakage -- features computed strictly pre-event.
Random pair swap for ~50% class balance.

Feature set (14):
    elo_p1, elo_p2, elo_diff,
    ranking_p1, ranking_p2, ranking_log_ratio,
    three_da_p1_pdc, three_da_p2_pdc,
    checkout_pct_p1_pdc, checkout_pct_p2_pdc,
    format_type_encoded,
    stage_floor,
    short_format,
    ecosystem_encoded,
"""
from __future__ import annotations

import pathlib
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from models import DartsMLError

logger = structlog.get_logger(__name__)


R0_FEATURES: list[str] = [
    "elo_p1",
    "elo_p2",
    "elo_diff",
    "ranking_p1",
    "ranking_p2",
    "ranking_log_ratio",
    "three_da_p1_pdc",
    "three_da_p2_pdc",
    "checkout_pct_p1_pdc",
    "checkout_pct_p2_pdc",
    "format_type_encoded",
    "stage_floor",
    "short_format",
    "ecosystem_encoded",
]

R0_FEATURE_COUNT: int = len(R0_FEATURES)


class R0LogisticModel:
    """
    R0 model: Logistic regression on 14 official metadata features.

    This is the baseline model for matches where only result data and
    official rankings/ratings are available. No per-match or per-visit
    statistics are required.

    The model uses L2-regularized logistic regression with standardized
    features for numerical stability.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
            penalty="l2",
        )
        self.fitted_: bool = False
        self._feature_names: list[str] = list(R0_FEATURES)
        self._log = logger.bind(component="R0LogisticModel")

    def validate_features(self, X: pd.DataFrame) -> None:
        """
        Validate that all R0 features are present in the input.

        Parameters
        ----------
        X:
            Feature dataframe.

        Raises
        ------
        DartsMLError
            If any required R0 feature is missing.
        """
        missing = set(R0_FEATURES) - set(X.columns)
        if missing:
            raise DartsMLError(
                f"Missing R0 features: {sorted(missing)}. "
                f"Required features: {R0_FEATURES}"
            )

        # Check for NaN/inf in required features
        r0_cols = X[R0_FEATURES]
        nan_cols = r0_cols.columns[r0_cols.isnull().any()].tolist()
        if nan_cols:
            raise DartsMLError(
                f"R0 features contain NaN values in columns: {nan_cols}"
            )

        inf_mask = np.isinf(r0_cols.values)
        if inf_mask.any():
            inf_cols = [
                R0_FEATURES[i]
                for i in range(len(R0_FEATURES))
                if inf_mask[:, i].any()
            ]
            raise DartsMLError(
                f"R0 features contain infinite values in columns: {inf_cols}"
            )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Fit the R0 logistic regression model.

        Parameters
        ----------
        X:
            Feature dataframe with R0 features, shape (n_samples, n_features).
        y:
            Binary labels (0 or 1), shape (n_samples,).

        Raises
        ------
        DartsMLError
            If features are invalid or fitting fails.
        """
        self.validate_features(X)
        y = np.asarray(y, dtype=np.float64)

        if len(X) != len(y):
            raise DartsMLError(
                f"X and y length mismatch: {len(X)} vs {len(y)}"
            )
        if len(np.unique(y)) < 2:
            raise DartsMLError(
                "y must contain both classes (0 and 1) for training."
            )

        X_r0 = X[R0_FEATURES].values.astype(np.float64)
        X_scaled = self._scaler.fit_transform(X_r0)
        self._model.fit(X_scaled, y)
        self.fitted_ = True

        self._log.info(
            "r0_model_fitted",
            n_samples=len(X),
            n_features=R0_FEATURE_COUNT,
            class_balance=float(np.mean(y)),
            coef_norm=float(np.linalg.norm(self._model.coef_)),
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for P1 winning.

        Parameters
        ----------
        X:
            Feature dataframe with R0 features.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 2) with [P(P2 wins), P(P1 wins)].

        Raises
        ------
        DartsMLError
            If the model is not fitted or features are invalid.
        """
        if not self.fitted_:
            raise DartsMLError(
                "R0 model has not been fitted. Call .fit() first."
            )
        self.validate_features(X)

        X_r0 = X[R0_FEATURES].values.astype(np.float64)
        X_scaled = self._scaler.transform(X_r0)
        return self._model.predict_proba(X_scaled)

    def predict_p1_win_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability of player 1 winning, shape (n_samples,).

        Parameters
        ----------
        X:
            Feature dataframe.

        Returns
        -------
        np.ndarray
            P(P1 wins) for each sample.
        """
        proba = self.predict_proba(X)
        return proba[:, 1]

    def save(self, path: str | pathlib.Path) -> None:
        """
        Serialize the model to disk via joblib.

        Parameters
        ----------
        path:
            File path for the saved model.

        Raises
        ------
        DartsMLError
            If the model is not fitted.
        """
        if not self.fitted_:
            raise DartsMLError("Cannot save unfitted R0 model.")
        state = {
            "scaler": self._scaler,
            "model": self._model,
            "feature_names": self._feature_names,
        }
        joblib.dump(state, path)
        self._log.info("r0_model_saved", path=str(path))

    def load(self, path: str | pathlib.Path) -> None:
        """
        Load a previously saved model from disk.

        Parameters
        ----------
        path:
            File path to the saved model.

        Raises
        ------
        DartsMLError
            If loading fails.
        """
        path = pathlib.Path(path)
        if not path.exists():
            raise DartsMLError(f"Model file not found: {path}")
        state = joblib.load(path)
        self._scaler = state["scaler"]
        self._model = state["model"]
        self._feature_names = state["feature_names"]
        self.fitted_ = True
        self._log.info("r0_model_loaded", path=str(path))

    @property
    def coefficients(self) -> Optional[np.ndarray]:
        """Return model coefficients (None if not fitted)."""
        if not self.fitted_:
            return None
        return self._model.coef_[0]

    @property
    def intercept(self) -> Optional[float]:
        """Return model intercept (None if not fitted)."""
        if not self.fitted_:
            return None
        return float(self._model.intercept_[0])

    def feature_importances(self) -> dict[str, float]:
        """
        Return feature importances based on absolute coefficient values.

        Returns
        -------
        dict[str, float]
            Feature name -> absolute coefficient value (scaled).

        Raises
        ------
        DartsMLError
            If the model is not fitted.
        """
        if not self.fitted_:
            raise DartsMLError("Model not fitted.")
        coefs = np.abs(self._model.coef_[0])
        total = coefs.sum()
        if total == 0:
            return {f: 0.0 for f in R0_FEATURES}
        normed = coefs / total
        return dict(zip(R0_FEATURES, normed.tolist()))
