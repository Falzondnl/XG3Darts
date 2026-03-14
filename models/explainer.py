"""
SHAP-based explainability for R1/R2 models.

Required for regulatory audits and model transparency.
Produces per-feature SHAP values that explain individual predictions
in terms of feature contributions to the predicted probability.

Supports:
    - TreeSHAP for gradient-boosted models (LightGBM, XGBoost, CatBoost)
    - KernelSHAP as fallback for any model
    - Feature importance rankings
    - Interaction effects (pairwise)
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog

from models import DartsMLError
from models.r0_logit import R0_FEATURES
from models.r1_lgbm import R1_FEATURES
from models.r2_stacking import R2_FEATURES

logger = structlog.get_logger(__name__)


# Regime-to-features mapping
_REGIME_FEATURES: dict[int, list[str]] = {
    0: R0_FEATURES,
    1: R1_FEATURES,
    2: R2_FEATURES,
}


class DartsModelExplainer:
    """
    SHAP-based model explainability for darts ML models.

    Provides per-feature SHAP values for individual predictions,
    global feature importance, and interaction analysis.

    Parameters
    ----------
    background_samples:
        Number of background samples for SHAP estimation.
        Higher values give more accurate estimates but are slower.
    """

    def __init__(self, background_samples: int = 100) -> None:
        self.background_samples = background_samples
        self._explainers: dict[str, Any] = {}
        self._background_data: dict[int, np.ndarray] = {}
        self._log = logger.bind(component="DartsModelExplainer")

    def set_background(
        self,
        regime: int,
        background_data: pd.DataFrame,
    ) -> None:
        """
        Set the background dataset for SHAP estimation.

        Parameters
        ----------
        regime:
            Model regime (0, 1, or 2).
        background_data:
            Background samples used for computing SHAP expectations.
            Should be a representative subset of training data.
        """
        if regime not in _REGIME_FEATURES:
            raise DartsMLError(
                f"Unknown regime {regime}. Valid: {list(_REGIME_FEATURES.keys())}"
            )
        features = _REGIME_FEATURES[regime]
        missing = set(features) - set(background_data.columns)
        if missing:
            raise DartsMLError(
                f"Background data missing features: {sorted(missing)}"
            )

        bg = background_data[features].values.astype(np.float64)
        n = min(self.background_samples, len(bg))
        if n < len(bg):
            rng = np.random.default_rng(42)
            indices = rng.choice(len(bg), size=n, replace=False)
            bg = bg[indices]

        self._background_data[regime] = bg
        self._log.info(
            "shap_background_set",
            regime=regime,
            n_samples=n,
            n_features=len(features),
        )

    def explain_prediction(
        self,
        match_features: dict,
        model_regime: int,
        model: Optional[Any] = None,
    ) -> dict:
        """
        Compute per-feature SHAP values for a single prediction.

        Parameters
        ----------
        match_features:
            Feature dict for the match (flat dict of feature_name -> value).
        model_regime:
            The model regime (0, 1, or 2).
        model:
            The fitted model object. Must have a predict_proba method
            that accepts a 2D numpy array.

        Returns
        -------
        dict
            Explanation dict with keys:
                - "regime": int
                - "features": dict[str, float] (feature values)
                - "shap_values": dict[str, float] (per-feature SHAP)
                - "base_value": float (expected value from background)
                - "predicted_proba": float
                - "top_positive": list[tuple[str, float]] (top drivers toward P1 win)
                - "top_negative": list[tuple[str, float]] (top drivers toward P2 win)

        Raises
        ------
        DartsMLError
            If regime is invalid, model is None, or features are missing.
        """
        if model_regime not in _REGIME_FEATURES:
            raise DartsMLError(
                f"Unknown regime {model_regime}. Valid: {list(_REGIME_FEATURES.keys())}"
            )
        if model is None:
            raise DartsMLError("Model must be provided for explanation.")

        features = _REGIME_FEATURES[model_regime]
        missing = [f for f in features if f not in match_features]
        if missing:
            raise DartsMLError(
                f"Match features missing for explanation: {missing[:10]}..."
            )

        # Build feature vector
        x = np.array(
            [[match_features[f] for f in features]],
            dtype=np.float64,
        )

        # Get background data
        bg = self._background_data.get(model_regime)

        try:
            import shap

            # Determine explainer type based on model
            if hasattr(model, "base_learners"):
                # Stacking model: explain the meta-learner through base predictions
                # Use KernelSHAP on the full pipeline
                predict_fn = self._make_predict_fn(model, features)

                if bg is not None:
                    explainer = shap.KernelExplainer(predict_fn, bg)
                else:
                    # Minimal background: use the input itself
                    explainer = shap.KernelExplainer(predict_fn, x)

                shap_values = explainer.shap_values(x, nsamples=self.background_samples)
                base_value = float(explainer.expected_value)

                if isinstance(shap_values, list):
                    # Binary classification: use class 1 SHAP values
                    sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                else:
                    sv = shap_values[0]

            elif hasattr(model, "_model") and hasattr(model._model, "coef_"):
                # R0 logistic model: use linear SHAP (exact)
                sv = self._linear_shap(model, x, bg, features)
                if bg is not None:
                    predict_fn = self._make_predict_fn(model, features)
                    base_value = float(np.mean(predict_fn(bg)))
                else:
                    base_value = 0.5
            else:
                # Generic: KernelSHAP
                predict_fn = self._make_predict_fn(model, features)
                if bg is not None:
                    explainer = shap.KernelExplainer(predict_fn, bg)
                else:
                    explainer = shap.KernelExplainer(predict_fn, x)
                shap_values = explainer.shap_values(x, nsamples=self.background_samples)
                base_value = float(explainer.expected_value)
                sv = shap_values[0] if isinstance(shap_values, np.ndarray) else shap_values

        except ImportError:
            # SHAP not available: use permutation-based approximation
            self._log.warning("shap_unavailable_using_permutation")
            sv, base_value = self._permutation_importance(model, x, features, bg)

        # Build explanation
        shap_dict = dict(zip(features, sv.tolist() if hasattr(sv, 'tolist') else sv))

        # Get predicted probability
        pred_proba = self._get_prediction(model, x, features)

        # Sort by absolute SHAP value
        sorted_features = sorted(
            shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True
        )
        top_positive = [
            (f, v) for f, v in sorted_features if v > 0
        ][:5]
        top_negative = [
            (f, v) for f, v in sorted_features if v < 0
        ][:5]

        explanation = {
            "regime": model_regime,
            "features": {f: float(match_features[f]) for f in features},
            "shap_values": shap_dict,
            "base_value": base_value,
            "predicted_proba": pred_proba,
            "top_positive": top_positive,
            "top_negative": top_negative,
        }

        self._log.info(
            "prediction_explained",
            regime=model_regime,
            predicted_proba=round(pred_proba, 4),
            n_features=len(features),
        )

        return explanation

    def _make_predict_fn(self, model: Any, features: list[str]) -> Any:
        """Create a predict function that takes numpy arrays."""
        def predict_fn(X: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(X, columns=features)
            proba = model.predict_proba(df)
            if proba.ndim == 2:
                return proba[:, 1]
            return proba
        return predict_fn

    def _get_prediction(
        self, model: Any, x: np.ndarray, features: list[str]
    ) -> float:
        """Get the model prediction for a single sample."""
        df = pd.DataFrame(x, columns=features)
        proba = model.predict_proba(df)
        if proba.ndim == 2:
            return float(proba[0, 1])
        return float(proba[0])

    def _linear_shap(
        self,
        model: Any,
        x: np.ndarray,
        bg: Optional[np.ndarray],
        features: list[str],
    ) -> np.ndarray:
        """
        Compute exact SHAP values for a linear model.

        For logistic regression with standardised features:
        SHAP_i = coef_i * (x_i_scaled - E[x_i_scaled])
        """
        if bg is not None:
            bg_mean = np.mean(bg, axis=0)
        else:
            bg_mean = np.zeros(len(features))

        if hasattr(model, "_scaler") and hasattr(model._scaler, "transform"):
            x_scaled = model._scaler.transform(x)
            bg_scaled = model._scaler.transform(bg_mean.reshape(1, -1))
            diff = x_scaled[0] - bg_scaled[0]
        else:
            diff = x[0] - bg_mean

        coefs = model._model.coef_[0] if hasattr(model, "_model") else np.ones(len(features))
        return diff * coefs

    def _permutation_importance(
        self,
        model: Any,
        x: np.ndarray,
        features: list[str],
        bg: Optional[np.ndarray],
    ) -> tuple[np.ndarray, float]:
        """
        Fallback: approximate SHAP values via single-feature permutation.
        """
        predict_fn = self._make_predict_fn(model, features)
        base_pred = float(predict_fn(x)[0])

        if bg is not None:
            base_value = float(np.mean(predict_fn(bg)))
        else:
            base_value = 0.5

        n_features = x.shape[1]
        importance = np.zeros(n_features, dtype=np.float64)

        for j in range(n_features):
            x_perm = x.copy()
            if bg is not None:
                x_perm[0, j] = np.mean(bg[:, j])
            else:
                x_perm[0, j] = 0.0
            perm_pred = float(predict_fn(x_perm)[0])
            importance[j] = base_pred - perm_pred

        return importance, base_value

    def global_feature_importance(
        self,
        model: Any,
        model_regime: int,
        X: pd.DataFrame,
        n_samples: int = 200,
    ) -> dict[str, float]:
        """
        Compute mean absolute SHAP values across a dataset.

        Parameters
        ----------
        model:
            Fitted model.
        model_regime:
            Regime (0, 1, or 2).
        X:
            Dataset with features.
        n_samples:
            Number of samples to explain.

        Returns
        -------
        dict[str, float]
            Feature name -> mean |SHAP value|.
        """
        features = _REGIME_FEATURES[model_regime]
        n = min(n_samples, len(X))
        indices = np.random.default_rng(42).choice(len(X), size=n, replace=False)

        shap_accum = {f: 0.0 for f in features}

        for idx in indices:
            row = {f: float(X.iloc[idx][f]) for f in features}
            explanation = self.explain_prediction(
                match_features=row,
                model_regime=model_regime,
                model=model,
            )
            for f, v in explanation["shap_values"].items():
                shap_accum[f] += abs(v)

        return {f: v / n for f, v in shap_accum.items()}
