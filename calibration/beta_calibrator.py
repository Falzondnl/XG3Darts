"""
Beta distribution calibrator for probability outputs.

Fits Beta(a, b) to model outputs via maximum likelihood estimation.
Maps raw model probabilities through a learned Beta CDF to produce
well-calibrated probabilities.

Reference:
    Kull, M., Silva Filho, T., Flach, P. (2017).
    "Beta calibration: a well-founded and easily implemented improvement
     on logistic calibration for binary classifiers."
    AISTATS 2017.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import structlog
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import beta as beta_dist

logger = structlog.get_logger(__name__)


class BetaCalibrationError(Exception):
    """Raised when beta calibration encounters invalid inputs or fit failure."""


class BetaCalibrator:
    """
    Beta distribution calibrator for probability outputs.

    Fits the transformation:
        calibrated_p = Beta_CDF(raw_p; a, b)

    where a and b are fitted via MLE on (raw_proba, y_true) pairs.

    Parameters
    ----------
    market_family:
        Optional label for the market family this calibrator serves.
        Used for logging and audit purposes.
    """

    def __init__(self, market_family: Optional[str] = None) -> None:
        self.market_family: Optional[str] = market_family
        self.a_: float = 1.0  # prior: uniform
        self.b_: float = 1.0  # prior: uniform
        self.fitted_: bool = False
        self._n_train_: int = 0
        self._log = logger.bind(
            component="BetaCalibrator",
            market_family=market_family,
        )

    def fit(self, proba: np.ndarray, y_true: np.ndarray) -> None:
        """
        Fit Beta calibration parameters via maximum likelihood.

        Parameters
        ----------
        proba:
            Raw model probabilities, shape (n_samples,). Values must be in (0, 1).
        y_true:
            Binary ground truth labels, shape (n_samples,). Values in {0, 1}.

        Raises
        ------
        BetaCalibrationError
            If inputs are invalid or optimization fails to converge.
        """
        proba = np.asarray(proba, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)

        if proba.ndim != 1 or y_true.ndim != 1:
            raise BetaCalibrationError("proba and y_true must be 1-dimensional.")
        if len(proba) != len(y_true):
            raise BetaCalibrationError(
                f"Length mismatch: proba={len(proba)}, y_true={len(y_true)}"
            )
        if len(proba) < 2:
            raise BetaCalibrationError(
                "Need at least 2 samples for calibration fitting."
            )

        unique_labels = set(np.unique(y_true))
        if not unique_labels.issubset({0.0, 1.0}):
            raise BetaCalibrationError(
                f"y_true must contain only 0 and 1, got unique values: {unique_labels}"
            )
        if len(unique_labels) < 2:
            raise BetaCalibrationError(
                "y_true must contain both 0 and 1 labels for calibration."
            )

        # Clip probabilities to avoid log(0)
        eps = 1e-7
        proba_clipped = np.clip(proba, eps, 1.0 - eps)

        # MLE: maximize sum of y*log(Beta_CDF(p;a,b)) + (1-y)*log(1-Beta_CDF(p;a,b))
        def neg_log_likelihood(params: np.ndarray) -> float:
            a, b = params
            if a <= 0 or b <= 0:
                return 1e12
            cal = beta_dist.cdf(proba_clipped, a, b)
            cal = np.clip(cal, eps, 1.0 - eps)
            ll = np.sum(
                y_true * np.log(cal) + (1.0 - y_true) * np.log(1.0 - cal)
            )
            return -ll

        # Initialize with method of moments from positive/negative class means
        pos_mean = np.mean(proba_clipped[y_true == 1.0])
        neg_mean = np.mean(proba_clipped[y_true == 0.0])
        a_init = max(0.5, pos_mean * 2.0)
        b_init = max(0.5, (1.0 - neg_mean) * 2.0)

        result = minimize(
            neg_log_likelihood,
            x0=np.array([a_init, b_init]),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
        )

        if not result.success and result.fun > 1e10:
            raise BetaCalibrationError(
                f"Beta calibration optimization failed: {result.message}"
            )

        self.a_ = float(max(result.x[0], 1e-4))
        self.b_ = float(max(result.x[1], 1e-4))
        self.fitted_ = True
        self._n_train_ = len(proba)

        self._log.info(
            "beta_calibrator_fitted",
            a=round(self.a_, 6),
            b=round(self.b_, 6),
            n_samples=self._n_train_,
            neg_ll=round(float(result.fun), 4),
        )

    def predict_proba(self, raw_proba: np.ndarray) -> np.ndarray:
        """
        Apply beta calibration to raw probabilities.

        Parameters
        ----------
        raw_proba:
            Raw model probabilities, shape (n_samples,) or (n_samples, 2).

        Returns
        -------
        np.ndarray
            Calibrated probabilities in [0, 1], same shape as input.

        Raises
        ------
        BetaCalibrationError
            If the calibrator has not been fitted.
        """
        if not self.fitted_:
            raise BetaCalibrationError(
                "BetaCalibrator has not been fitted. Call .fit() first."
            )

        raw_proba = np.asarray(raw_proba, dtype=np.float64)
        input_2d = False
        if raw_proba.ndim == 2 and raw_proba.shape[1] == 2:
            input_2d = True
            p = raw_proba[:, 1]
        elif raw_proba.ndim == 1:
            p = raw_proba
        else:
            raise BetaCalibrationError(
                f"raw_proba must be 1D or 2D with shape (n, 2), got shape {raw_proba.shape}"
            )

        eps = 1e-7
        p_clipped = np.clip(p, eps, 1.0 - eps)
        calibrated = beta_dist.cdf(p_clipped, self.a_, self.b_)
        calibrated = np.clip(calibrated, eps, 1.0 - eps)

        if input_2d:
            out = np.column_stack([1.0 - calibrated, calibrated])
            return out

        return calibrated

    def get_params(self) -> dict[str, float]:
        """Return fitted parameters."""
        return {
            "a": self.a_,
            "b": self.b_,
            "fitted": float(self.fitted_),
            "n_train": float(self._n_train_),
        }

    def __repr__(self) -> str:
        status = "fitted" if self.fitted_ else "unfitted"
        return (
            f"BetaCalibrator(market_family={self.market_family!r}, "
            f"a={self.a_:.4f}, b={self.b_:.4f}, {status})"
        )
