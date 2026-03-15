"""
SGP correlation estimator for darts markets.

Uses Spearman rank correlations from historical joint outcomes,
with hierarchical shrinkage toward a global prior when sample size
is insufficient, followed by Higham PSD projection.

Market dimensions (columns/rows 0-14):
  0:  match_win            P(P1 wins)
  1:  total_legs_over      P(total legs > line)
  2:  handicap             P(P1 covers handicap)
  3:  180_over             P(total 180s > line)
  4:  checkout_over        P(highest checkout > line)
  5:  exact_score          P(specific exact score)
  6:  leg_winner_next      P(P1 wins next leg)
  7:  first_leg_winner     P(P1 wins the first leg)
  8:  most_180s            P(P1 hits more 180s than P2)
  9:  highest_checkout     P(P1 achieves highest checkout)
  10: total_180s_band      P(total 180s in a specific band)
  11: player_checkout_over P(player-specific checkout > threshold)
  12: sets_over            P(total sets > line, sets-format matches)
  13: break_of_throw       P(at least one break of throw in match)
  14: nine_dart_finish     P(nine-dart finish in match)

The global prior is derived from domain expertise and
calibration against historical PDC tournament data.
It captures well-known structural relationships:
  - Match winner strongly correlated with handicap cover
  - Total legs correlated with both-player quality (positively)
  - 180 count correlated with total legs (more legs = more 180s)
  - First leg winner moderately correlated with match winner
  - Most 180s moderately correlated with match winner (stronger player)
  - Break of throw positively correlated with total legs (longer matches)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from engines.errors import DartsEngineError
from sgp.psd_enforcer import higham_psd

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Global correlation prior
# ---------------------------------------------------------------------------
# Dimensions (15x15):
#   0:  match_win            1:  total_legs_over     2:  handicap
#   3:  180_over             4:  checkout_over        5:  exact_score
#   6:  leg_winner_next      7:  first_leg_winner     8:  most_180s
#   9:  highest_checkout     10: total_180s_band      11: player_checkout_over
#   12: sets_over            13: break_of_throw       14: nine_dart_finish
#
# Correlation values reasoned from darts domain expertise:
#   - first_leg_winner (7): moderately correlated with match_win (+0.35),
#     weakly with handicap (+0.20), very weakly with leg_winner_next (+0.15).
#     Negatively correlated with total_legs_over (-0.12) since winning the
#     first leg gives a head-start that can shorten matches.
#   - most_180s (8): correlated with match_win (+0.30, stronger players hit
#     more maximums), with 180_over (+0.65, collinear: more 180s in match →
#     likely one player leads), with total_legs_over (+0.35).
#   - highest_checkout (9): strongly correlated with checkout_over (+0.72),
#     moderately with match_win (+0.40), weakly with most_180s (+0.10).
#   - total_180s_band (10): correlated with 180_over (+0.55) and
#     total_legs_over (+0.40), weakly negative with exact_score (-0.10).
#   - player_checkout_over (11): strongly correlated with checkout_over (+0.60),
#     moderately with highest_checkout (+0.55), weakly with match_win (+0.25).
#   - sets_over (12): correlated with total_legs_over (+0.58), weakly with
#     180_over (+0.25). Negative with exact_score (-0.30) since sets formats
#     increase variance.
#   - break_of_throw (13): positively correlated with total_legs_over (+0.42),
#     180_over (+0.20), negatively with match_win (-0.15, dominant winners
#     rarely get broken).
#   - nine_dart_finish (14): weakly correlated with 180_over (+0.18),
#     checkout_over (+0.22), player_checkout_over (+0.20). Near-independent
#     of most markets — rare event.

DARTS_GLOBAL_CORRELATION_PRIOR: np.ndarray = np.array([
    # mw      totl    h_cap   180     co      ex_sc   lw      flw     m180    hco     t180b   pco     sets    bot     9df
    [ 1.00, -0.28,  0.52, -0.40,  0.58, -0.45,  0.12,  0.35,  0.30,  0.40, -0.18,  0.25, -0.20, -0.15,  0.05],  # 0  match_win
    [-0.28,  1.00,  0.32,  0.38, -0.10, -0.08,  0.62, -0.12,  0.35, -0.08,  0.40, -0.08,  0.58,  0.42,  0.08],  # 1  total_legs_over
    [ 0.52,  0.32,  1.00,  0.08,  0.35, -0.22,  0.18,  0.20,  0.15,  0.28, -0.05,  0.22,  0.15, -0.10,  0.04],  # 2  handicap
    [-0.40,  0.38,  0.08,  1.00, -0.15,  0.05,  0.48,  0.05,  0.65, -0.10,  0.55,  0.08,  0.25,  0.20,  0.18],  # 3  180_over
    [ 0.58, -0.10,  0.35, -0.15,  1.00, -0.38,  0.08,  0.18,  0.08,  0.72, -0.12,  0.60, -0.08, -0.08,  0.22],  # 4  checkout_over
    [-0.45, -0.08, -0.22,  0.05, -0.38,  1.00, -0.12, -0.20, -0.15, -0.28, -0.10, -0.22, -0.30, -0.05, -0.05],  # 5  exact_score
    [ 0.12,  0.62,  0.18,  0.48,  0.08, -0.12,  1.00,  0.15,  0.28,  0.05,  0.30,  0.06,  0.40,  0.35,  0.06],  # 6  leg_winner_next
    [ 0.35, -0.12,  0.20,  0.05,  0.18, -0.20,  0.15,  1.00,  0.18,  0.22, -0.08,  0.15, -0.10, -0.12,  0.04],  # 7  first_leg_winner
    [ 0.30,  0.35,  0.15,  0.65,  0.08, -0.15,  0.28,  0.18,  1.00,  0.10,  0.48,  0.12,  0.28,  0.18,  0.15],  # 8  most_180s
    [ 0.40, -0.08,  0.28, -0.10,  0.72, -0.28,  0.05,  0.22,  0.10,  1.00, -0.10,  0.55, -0.05, -0.06,  0.18],  # 9  highest_checkout
    [-0.18,  0.40, -0.05,  0.55, -0.12, -0.10,  0.30, -0.08,  0.48, -0.10,  1.00,  0.05,  0.32,  0.22,  0.12],  # 10 total_180s_band
    [ 0.25, -0.08,  0.22,  0.08,  0.60, -0.22,  0.06,  0.15,  0.12,  0.55,  0.05,  1.00, -0.05, -0.06,  0.20],  # 11 player_checkout_over
    [-0.20,  0.58,  0.15,  0.25, -0.08, -0.30,  0.40, -0.10,  0.28, -0.05,  0.32, -0.05,  1.00,  0.38,  0.06],  # 12 sets_over
    [-0.15,  0.42, -0.10,  0.20, -0.08, -0.05,  0.35, -0.12,  0.18, -0.06,  0.22, -0.06,  0.38,  1.00,  0.05],  # 13 break_of_throw
    [ 0.05,  0.08,  0.04,  0.18,  0.22, -0.05,  0.06,  0.04,  0.15,  0.18,  0.12,  0.20,  0.06,  0.05,  1.00],  # 14 nine_dart_finish
], dtype=np.float64)

# Competition-family shrinkage: fraction of empirical data to use
# (alpha = empirical weight; 1 - alpha = prior weight)
_FAMILY_SHRINKAGE_THRESHOLD = 500  # samples below this → strong shrinkage toward prior

# Market index names for logging — order must match the prior matrix above
MARKET_NAMES = [
    "match_win",           # 0
    "total_legs_over",     # 1
    "handicap",            # 2
    "180_over",            # 3
    "checkout_over",       # 4
    "exact_score",         # 5
    "leg_winner_next",     # 6
    "first_leg_winner",    # 7
    "most_180s",           # 8
    "highest_checkout",    # 9
    "total_180s_band",     # 10
    "player_checkout_over",# 11
    "sets_over",           # 12
    "break_of_throw",      # 13
    "nine_dart_finish",    # 14
]

# Number of market dimensions — must equal len(MARKET_NAMES)
N_MARKET_DIMS: int = len(MARKET_NAMES)


class DartsSGPCorrelationEstimator:
    """
    Estimate the SGP correlation matrix for a given competition.

    The estimator:
    1. Loads historical joint outcomes from the database (or accepts
       pre-computed Spearman correlations)
    2. Applies hierarchical shrinkage toward the global prior based on
       sample size
    3. Projects the result to the nearest PSD matrix via Higham (2002)
    """

    def __init__(
        self,
        prior: np.ndarray | None = None,
    ) -> None:
        """
        Parameters
        ----------
        prior:
            Global correlation prior.  Defaults to
            DARTS_GLOBAL_CORRELATION_PRIOR.
        """
        if prior is not None:
            expected = (N_MARKET_DIMS, N_MARKET_DIMS)
            if prior.shape != expected:
                raise DartsEngineError(
                    f"Prior must be {expected}, got {prior.shape}"
                )
            self._prior = prior.copy()
        else:
            self._prior = DARTS_GLOBAL_CORRELATION_PRIOR.copy()

        # Ensure prior is PSD
        self._prior = higham_psd(self._prior, eigenvalue_floor=1e-8)

    def estimate_for_competition(
        self,
        competition_code: str,
        min_samples: int = 500,
        empirical_spearman: np.ndarray | None = None,
        n_samples: int = 0,
    ) -> np.ndarray:
        """
        Estimate the SGP correlation matrix for a competition.

        When ``empirical_spearman`` is provided, it is used directly;
        otherwise the prior is returned (with shrinkage alpha = 0).

        Parameters
        ----------
        competition_code:
            Competition format code (e.g. ``"PDC_WC"``).
        min_samples:
            Minimum sample count for full empirical weight.
        empirical_spearman:
            Pre-computed Spearman rank correlation matrix (7x7).
            If None, returns the prior.
        n_samples:
            Number of joint outcome observations used in ``empirical_spearman``.

        Returns
        -------
        np.ndarray
            7x7 PSD correlation matrix.

        Raises
        ------
        DartsEngineError
            If empirical_spearman has wrong shape.
        """
        if empirical_spearman is not None:
            expected = (N_MARKET_DIMS, N_MARKET_DIMS)
            if empirical_spearman.shape != expected:
                raise DartsEngineError(
                    f"empirical_spearman must be {expected}, got {empirical_spearman.shape}"
                )
            # Compute shrinkage weight
            alpha = self._shrinkage_alpha(n_samples, min_samples)
            blended = self._shrink_toward_prior(
                empirical=empirical_spearman,
                alpha=alpha,
                prior=self._prior,
            )
        else:
            # No empirical data — use prior entirely
            alpha = 0.0
            blended = self._prior.copy()
            logger.info(
                "sgp_correlation_using_prior",
                competition_code=competition_code,
                reason="no_empirical_data",
            )

        # Project to PSD
        psd_matrix = self._higham_psd(blended)

        logger.info(
            "sgp_correlation_estimated",
            competition_code=competition_code,
            n_samples=n_samples,
            alpha=round(alpha, 4),
            min_eigenvalue=round(float(np.min(np.linalg.eigvalsh(psd_matrix))), 6),
        )

        return psd_matrix

    def estimate_from_joint_outcomes(
        self,
        joint_outcome_matrix: np.ndarray,
        competition_code: str,
        min_samples: int = 500,
    ) -> np.ndarray:
        """
        Estimate the correlation matrix directly from a joint-outcome data matrix.

        Parameters
        ----------
        joint_outcome_matrix:
            Array of shape (N, 7) where each row is one match's outcomes
            (binary or continuous) for the 7 market dimensions.
        competition_code:
            For logging.
        min_samples:
            Minimum N for full empirical weight.

        Returns
        -------
        np.ndarray
            7x7 PSD correlation matrix.

        Raises
        ------
        DartsEngineError
            If the matrix has wrong shape or insufficient variance.
        """
        if joint_outcome_matrix.ndim != 2 or joint_outcome_matrix.shape[1] != N_MARKET_DIMS:
            raise DartsEngineError(
                f"joint_outcome_matrix must be (N, {N_MARKET_DIMS}), "
                f"got {joint_outcome_matrix.shape}"
            )

        n_samples = joint_outcome_matrix.shape[0]

        if n_samples < 10:
            logger.warning(
                "sgp_too_few_samples_for_spearman",
                n_samples=n_samples,
                competition_code=competition_code,
            )
            return self.estimate_for_competition(
                competition_code=competition_code,
                min_samples=min_samples,
                empirical_spearman=None,
                n_samples=0,
            )

        # Compute Spearman rank correlations
        empirical = self._spearman_correlation(joint_outcome_matrix)

        return self.estimate_for_competition(
            competition_code=competition_code,
            min_samples=min_samples,
            empirical_spearman=empirical,
            n_samples=n_samples,
        )

    def _higham_psd(self, rho: np.ndarray) -> np.ndarray:
        """
        Higham (2002) alternating projections to nearest PSD matrix.

        Delegates to the standalone psd_enforcer module.
        """
        return higham_psd(rho, eigenvalue_floor=1e-8)

    def _shrink_toward_prior(
        self,
        empirical: np.ndarray,
        alpha: float,
        prior: np.ndarray,
    ) -> np.ndarray:
        """
        Linear shrinkage: alpha * empirical + (1 - alpha) * prior.

        Parameters
        ----------
        empirical:
            Empirical correlation matrix.
        alpha:
            Weight on empirical data [0, 1].  0 = all prior, 1 = all empirical.
        prior:
            Prior correlation matrix.

        Returns
        -------
        np.ndarray
            Blended correlation matrix (not necessarily PSD).
        """
        alpha = max(0.0, min(1.0, alpha))
        blended = alpha * empirical + (1.0 - alpha) * prior
        # Enforce unit diagonal after blending
        np.fill_diagonal(blended, 1.0)
        return blended

    @staticmethod
    def _shrinkage_alpha(n_samples: int, min_samples: int) -> float:
        """
        Compute the shrinkage weight alpha based on sample count.

        alpha = min(n_samples / min_samples, 1.0)

        This gives full weight to empirical data when n_samples >= min_samples,
        and linearly interpolates toward zero weight (all prior) as n → 0.
        """
        if min_samples <= 0:
            return 1.0
        return min(1.0, n_samples / min_samples)

    @staticmethod
    def _spearman_correlation(data: np.ndarray) -> np.ndarray:
        """
        Compute Spearman rank correlation matrix from data.

        Parameters
        ----------
        data:
            Array of shape (N, k).

        Returns
        -------
        np.ndarray
            k x k Spearman correlation matrix.
        """
        n, k = data.shape
        # Convert each column to ranks
        ranks = np.zeros_like(data, dtype=np.float64)
        for j in range(k):
            col = data[:, j]
            # Rank with average tie handling
            from scipy.stats import rankdata  # type: ignore
            ranks[:, j] = rankdata(col, method="average")

        # Pearson on ranks = Spearman
        # Subtract mean
        ranks_centered = ranks - ranks.mean(axis=0)
        # Compute covariance
        cov = (ranks_centered.T @ ranks_centered) / (n - 1)
        # Normalise to correlation
        std = np.sqrt(np.diag(cov))
        std[std < 1e-12] = 1.0  # avoid division by zero
        corr = cov / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)
        # Clip to [-1, 1]
        corr = np.clip(corr, -1.0, 1.0)
        return corr
