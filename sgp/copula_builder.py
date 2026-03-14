"""
Gaussian copula SGP builder.

Prices a Same-Game Parlay (SGP) from marginal probabilities and a
correlation matrix using a Gaussian copula with Monte Carlo integration.

Algorithm
---------
1. Convert each marginal probability to a standard-normal quantile (inverse CDF).
2. Simulate joint standard-normal vectors using Cholesky decomposition of
   the correlation matrix (so that the marginal distributions are N(0,1)
   and the pairwise correlations match the correlation matrix).
3. Each simulation step: the parlay wins if the standard-normal outcome for
   each selection falls on the "correct" side of its quantile threshold.
4. The SGP probability is estimated as the fraction of simulations in which
   ALL selections win simultaneously.

Properties
----------
- Captures linear (and implicitly some non-linear) dependencies.
- Consistent with the specified marginal probabilities by construction.
- Accurate to O(1/sqrt(N_MC)) where N_MC is the number of MC samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from engines.errors import DartsEngineError
from sgp.psd_enforcer import higham_psd

logger = structlog.get_logger(__name__)

_DEFAULT_N_MC = 50_000
_DEFAULT_SEED = 42


@dataclass
class SGPSelection:
    """
    A single SGP leg (selection).

    Attributes
    ----------
    market_type:
        Market identifier (e.g. ``"match_win"``, ``"total_legs_over"``,
        ``"180_over"``).  Used for logging and correlation lookup.
    outcome:
        Human-readable outcome description.
    probability:
        True probability of this selection winning, in (0, 1).
    correlation_index:
        Index into the correlation matrix row/column for this market type.
        If None, the selection is treated as independent.
    """
    market_type: str
    outcome: str
    probability: float
    correlation_index: int | None = None


class DartsSGPBuilder:
    """
    Build SGP (Same-Game Parlay) prices using a Gaussian copula.

    The builder is stateless; all required data is passed per call.
    """

    def __init__(
        self,
        n_mc: int = _DEFAULT_N_MC,
        random_seed: int | None = _DEFAULT_SEED,
    ) -> None:
        """
        Parameters
        ----------
        n_mc:
            Number of Monte Carlo samples.  Higher = more accurate but slower.
            50,000 gives ~±0.1% accuracy on typical SGP prices.
        random_seed:
            Seed for reproducibility.  Set to None for production randomness.
        """
        if n_mc < 1000:
            raise DartsEngineError(
                f"n_mc must be >= 1000 for meaningful accuracy, got {n_mc}"
            )
        self._n_mc = n_mc
        self._rng = np.random.default_rng(random_seed)

    def price_parlay(
        self,
        selections: list[SGPSelection],
        correlation_matrix: np.ndarray,
    ) -> float:
        """
        Compute the SGP probability using a Gaussian copula.

        Parameters
        ----------
        selections:
            List of SGP selections.  Each must have a valid probability in (0, 1)
            and a correlation_index pointing to its row/column in
            correlation_matrix.
        correlation_matrix:
            Square PSD correlation matrix.  Its size must be >= the maximum
            correlation_index + 1 in selections.

        Returns
        -------
        float
            Joint probability of all selections winning simultaneously.
            In [0, 1].

        Raises
        ------
        DartsEngineError
            If inputs are invalid, the correlation matrix is not PSD, or
            any selection has an out-of-range probability.
        """
        if not selections:
            raise DartsEngineError("selections must not be empty")

        # Validate selections
        for i, sel in enumerate(selections):
            if not (0.0 < sel.probability < 1.0):
                raise DartsEngineError(
                    f"Selection {i} ('{sel.market_type}') has probability "
                    f"{sel.probability:.6f} which is not in (0, 1)."
                )

        # Validate correlation matrix
        n_markets = correlation_matrix.shape[0]
        if correlation_matrix.ndim != 2 or correlation_matrix.shape[1] != n_markets:
            raise DartsEngineError(
                f"correlation_matrix must be square, got {correlation_matrix.shape}"
            )

        # Check all correlation_index values are in range
        for i, sel in enumerate(selections):
            if sel.correlation_index is not None:
                if not (0 <= sel.correlation_index < n_markets):
                    raise DartsEngineError(
                        f"Selection {i} has correlation_index={sel.correlation_index} "
                        f"which is out of range [0, {n_markets - 1}]."
                    )

        # Identify selections with and without correlation indices
        corr_selections = [s for s in selections if s.correlation_index is not None]
        indep_selections = [s for s in selections if s.correlation_index is None]

        # Independent selection joint probability (exact)
        p_indep = 1.0
        for sel in indep_selections:
            p_indep *= sel.probability

        if not corr_selections:
            # All independent — no copula needed
            logger.debug(
                "sgp_all_independent",
                n_selections=len(selections),
                p_joint=round(p_indep, 6),
            )
            return p_indep

        # Build sub-matrix for correlated selections
        indices = [s.correlation_index for s in corr_selections]
        sub_corr = correlation_matrix[np.ix_(indices, indices)]

        # Ensure PSD (project if needed)
        min_eig = float(np.min(np.linalg.eigvalsh(sub_corr)))
        if min_eig < -1e-8:
            logger.warning(
                "sgp_correlation_not_psd",
                min_eigenvalue=round(min_eig, 8),
                projecting=True,
            )
            sub_corr = higham_psd(sub_corr, eigenvalue_floor=1e-8)

        # Compute joint probability via Gaussian copula MC
        p_corr = self._gaussian_copula_mc(
            probabilities=[s.probability for s in corr_selections],
            correlation=sub_corr,
        )

        # Combine with independent selections
        p_joint = p_corr * p_indep

        logger.debug(
            "sgp_priced",
            n_selections=len(selections),
            n_correlated=len(corr_selections),
            n_independent=len(indep_selections),
            p_corr=round(p_corr, 6),
            p_indep=round(p_indep, 6),
            p_joint=round(p_joint, 6),
            n_mc=self._n_mc,
        )

        return max(0.0, min(1.0, p_joint))

    def price_parlay_with_margin(
        self,
        selections: list[SGPSelection],
        correlation_matrix: np.ndarray,
        margin: float = 0.05,
    ) -> dict[str, Any]:
        """
        Price a SGP and apply a bookmaker margin.

        Returns the true probability, bookmaker probability, and decimal odds.
        """
        true_prob = self.price_parlay(selections, correlation_matrix)

        if true_prob <= 0.0:
            return {
                "true_prob": 0.0,
                "bookmaker_prob": None,
                "decimal_odds": None,
                "margin": margin,
            }

        bookmaker_prob = true_prob * (1.0 + margin)
        bookmaker_prob = min(1.0, bookmaker_prob)
        decimal_odds = 1.0 / bookmaker_prob if bookmaker_prob > 0 else None

        return {
            "true_prob": round(true_prob, 6),
            "bookmaker_prob": round(bookmaker_prob, 6),
            "decimal_odds": round(decimal_odds, 4) if decimal_odds else None,
            "margin": margin,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _gaussian_copula_mc(
        self,
        probabilities: list[float],
        correlation: np.ndarray,
    ) -> float:
        """
        Estimate joint probability via Gaussian copula Monte Carlo.

        Steps:
          1. Convert marginal probabilities to normal quantiles (thresholds).
          2. Generate correlated standard-normal samples via Cholesky.
          3. For each simulation: parlay wins iff all Z_i < threshold_i
             (i.e. the normal CDF at threshold = marginal probability).

        Parameters
        ----------
        probabilities:
            List of k marginal probabilities, one per correlated selection.
        correlation:
            k x k PSD correlation matrix.

        Returns
        -------
        float
            Estimated joint probability.
        """
        from scipy.stats import norm  # type: ignore

        k = len(probabilities)
        assert correlation.shape == (k, k)

        # Step 1: marginal quantile thresholds
        # P(Z_i < threshold_i) = p_i  →  threshold_i = Phi^{-1}(p_i)
        thresholds = np.array([norm.ppf(p) for p in probabilities], dtype=np.float64)

        # Step 2: Cholesky decomposition of correlation matrix
        try:
            L = np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError:
            # Fallback: add a small diagonal jitter
            jitter = 1e-6
            L = np.linalg.cholesky(correlation + jitter * np.eye(k))

        # Step 3: simulate correlated standard normals
        # Z = L @ eps  where eps ~ N(0, I_k)
        eps = self._rng.standard_normal((self._n_mc, k))  # (N, k)
        Z = eps @ L.T  # (N, k) — each row is one simulation

        # Step 4: count successes
        # Parlay wins if all Z_i < threshold_i for all i
        # Shape: (N, k); True where Z_i < threshold_i
        wins = np.all(Z < thresholds[np.newaxis, :], axis=1)  # (N,)
        p_joint = float(np.mean(wins))

        logger.debug(
            "gaussian_copula_mc",
            k=k,
            n_mc=self._n_mc,
            p_joint=round(p_joint, 6),
            thresholds=thresholds.round(4).tolist(),
        )

        return p_joint
