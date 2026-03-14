"""
Women's Series transfer learning model.

Architecture
------------
Uses the men's PDC R0 logistic model as a base (weight initialisation via
partial pooling).  Fine-tuned on women's data with a hierarchical prior:

    player -> women's ecosystem -> men's ecosystem

Since women's match data is sparse the model uses partial pooling to share
statistical strength with the men's ecosystem.  The women's-specific
intercept shift is learned from available women's match results.

Ecosystem widening
------------------
The ``get_ecosystem_margin_multiplier()`` method returns 1.20, which is
applied by the margin engine on top of the regime widening for all
pdc_womens / wdf_womens competitions.  This reflects the thinner markets
and lower data volumes in the women's ecosystem.

Usage
-----
>>> model = WomensTransferModel()
>>> model.fit(womens_features_df, womens_labels)
>>> p_win = model.predict_proba(features_dict, "player_a", "player_b")
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import structlog

from models import DartsMLError

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Ecosystem constant
# ---------------------------------------------------------------------------

_WOMENS_MARGIN_MULTIPLIER: float = 1.20

# Partial pooling shrinkage weight: 0 = full shrinkage to men's prior,
# 1 = full trust in women's data.
_DEFAULT_SHRINKAGE: float = 0.40


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class WomensTransferModel:
    """
    Transfer learning model for women's darts pricing.

    The model maintains:
    - A base logistic regression weight vector (shared from men's R0 model).
    - An intercept adjustment learned from women's match data.
    - A women's-specific ELO differential coefficient fine-tuned from the
      women's data distribution.

    When no women's data is provided the model degrades gracefully to the
    men's ELO-only formula with the ecosystem widening factor applied.

    Partial pooling
    ---------------
    The women's coefficient vector is a linear interpolation between the
    men's prior and the maximum-likelihood estimate from women's data:

        coef_womens = shrinkage * coef_women_mle + (1 - shrinkage) * coef_men

    The shrinkage weight increases as more women's match data becomes
    available (empirical Bayes style).
    """

    def __init__(self, shrinkage: float = _DEFAULT_SHRINKAGE) -> None:
        """
        Parameters
        ----------
        shrinkage:
            Partial pooling weight (0–1).  Higher = more trust in women's data.
        """
        if not 0.0 <= shrinkage <= 1.0:
            raise DartsMLError(
                f"shrinkage must be in [0, 1], got {shrinkage}"
            )
        self._shrinkage = shrinkage
        self._fitted = False

        # Men's prior weights (ELO differential coefficient)
        # These are defaults matching the calibrated R0 model behaviour;
        # in production they would be loaded from the saved R0 model artefact.
        self._men_elo_coef: float = 0.0050   # logit shift per ELO point
        self._men_intercept: float = 0.0      # balanced prior

        # Women's fine-tuned parameters (set by fit())
        self._women_elo_coef: float = self._men_elo_coef
        self._women_intercept: float = self._men_intercept
        self._n_womens_matches: int = 0

        self._log = logger.bind(component="WomensTransferModel")

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def fit(
        self,
        features: Any,   # pd.DataFrame with ELO columns
        labels: Any,     # array-like binary labels
    ) -> None:
        """
        Fine-tune on women's match data.

        Parameters
        ----------
        features:
            DataFrame with at least ``elo_diff`` column
            (elo_p1 - elo_p2, from the player's perspective).
        labels:
            Binary labels: 1 if player 1 won, 0 otherwise.

        Raises
        ------
        DartsMLError
            If training data is insufficient or malformed.
        """
        try:
            import pandas as pd
            X = pd.DataFrame(features)
            y = np.asarray(labels, dtype=np.float64)
        except Exception as exc:
            raise DartsMLError(f"Failed to prepare women's training data: {exc}") from exc

        if len(X) < 5:
            self._log.warning(
                "womens_data_too_sparse",
                n_matches=len(X),
                fallback="men_prior",
            )
            self._fitted = True
            self._n_womens_matches = len(X)
            return

        if "elo_diff" not in X.columns:
            raise DartsMLError(
                "women's feature set must include 'elo_diff' (elo_p1 - elo_p2)."
            )

        elo_diffs = X["elo_diff"].values.astype(np.float64)

        # Fit intercept + ELO coef via logistic regression (gradient descent)
        coef_mle, intercept_mle = self._fit_logistic_gd(elo_diffs, y)

        # Partial pooling: blend with men's prior
        effective_shrinkage = min(
            1.0,
            self._shrinkage * math.sqrt(len(X) / 100.0)
        )

        self._women_elo_coef = (
            effective_shrinkage * coef_mle
            + (1.0 - effective_shrinkage) * self._men_elo_coef
        )
        self._women_intercept = (
            effective_shrinkage * intercept_mle
            + (1.0 - effective_shrinkage) * self._men_intercept
        )
        self._n_womens_matches = len(X)
        self._fitted = True

        self._log.info(
            "womens_model_fitted",
            n_matches=len(X),
            effective_shrinkage=round(effective_shrinkage, 3),
            women_elo_coef=round(self._women_elo_coef, 5),
            women_intercept=round(self._women_intercept, 5),
        )

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict_proba(
        self,
        features: dict,
        player1_id: str,
        player2_id: str,
    ) -> float:
        """
        P(player1 wins) for a women's match.

        Parameters
        ----------
        features:
            Dict with at least:
              - ``elo_p1`` (float): Player 1 ELO rating.
              - ``elo_p2`` (float): Player 2 ELO rating.
        player1_id:
            Player 1 identifier (for logging only).
        player2_id:
            Player 2 identifier (for logging only).

        Returns
        -------
        float
            P(player1 wins) in (0, 1).

        Raises
        ------
        DartsMLError
            If required features are missing.
        """
        elo_p1 = features.get("elo_p1")
        elo_p2 = features.get("elo_p2")

        if elo_p1 is None or elo_p2 is None:
            raise DartsMLError(
                f"Women's model requires 'elo_p1' and 'elo_p2' features. "
                f"Got keys: {list(features.keys())}"
            )

        elo_diff = float(elo_p1) - float(elo_p2)

        # Logit = intercept + coef * elo_diff
        logit = self._women_intercept + self._women_elo_coef * elo_diff
        p_win = 1.0 / (1.0 + math.exp(-logit))

        # Clip to avoid degenerate probabilities
        p_win = max(0.01, min(0.99, p_win))

        self._log.debug(
            "womens_predict",
            player1_id=player1_id,
            player2_id=player2_id,
            elo_diff=round(elo_diff, 1),
            p_win=round(p_win, 4),
            fitted=self._fitted,
        )

        return p_win

    # -----------------------------------------------------------------------
    # Ecosystem widening
    # -----------------------------------------------------------------------

    def get_ecosystem_margin_multiplier(self) -> float:
        """
        Return the ecosystem widening multiplier for women's markets.

        This is applied multiplicatively on top of the regime-specific
        base margin in the DartsMarginEngine.  The value of 1.20 reflects
        the 20% margin widening documented in Sprint 6 requirements.

        Returns
        -------
        float
            1.20
        """
        return _WOMENS_MARGIN_MULTIPLIER

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted on women's data."""
        return self._fitted

    @property
    def n_womens_matches(self) -> int:
        """Number of women's matches used for fine-tuning."""
        return self._n_womens_matches

    @property
    def shrinkage(self) -> float:
        """Partial pooling shrinkage weight."""
        return self._shrinkage

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _fit_logistic_gd(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        n_iter: int = 1000,
        l2: float = 0.1,
    ) -> tuple[float, float]:
        """
        Fit a 1-feature logistic regression via mini-batch gradient descent.

        Parameters
        ----------
        x:
            1-D feature array (elo_diff values).
        y:
            Binary labels.
        lr:
            Learning rate.
        n_iter:
            Number of gradient descent iterations.
        l2:
            L2 regularisation strength.

        Returns
        -------
        (coef, intercept)
        """
        n = len(x)
        coef = 0.0
        intercept = 0.0

        for _ in range(n_iter):
            logits = intercept + coef * x
            # Numerically stable sigmoid
            p = np.where(
                logits >= 0,
                1.0 / (1.0 + np.exp(-logits)),
                np.exp(logits) / (1.0 + np.exp(logits)),
            )
            residuals = p - y
            grad_coef = float(np.dot(residuals, x)) / n + l2 * coef
            grad_intercept = float(np.sum(residuals)) / n

            coef -= lr * grad_coef
            intercept -= lr * grad_intercept

        return coef, intercept
