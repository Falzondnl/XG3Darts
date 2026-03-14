"""
Empirical Bayes skill estimation for darts players.

Per-player bivariate normal distribution over aim-point deviations.
Uses James-Stein / EB shrinkage to estimate individual player skill
parameters from limited throw data, pooling toward a population prior.

Reference:
    Haugh, M. & Wang, C. (2024).
    "An Empirical Bayes Approach for Estimating Skill Models for
     Professional Darts Players."

The model estimates:
    - Per-segment accuracy (P(hitting target segment))
    - Aim-point mean/variance (bivariate normal on the dartboard)
    - Shrinkage toward population-level prior for thin-data players
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog
from scipy.stats import norm

from models import DartsMLError

logger = structlog.get_logger(__name__)


# Dartboard segment layout: angular positions (degrees) for each numbered segment
# Standard dartboard clockwise from top: 20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
# 3, 19, 7, 16, 8, 11, 14, 9, 12, 5
_SEGMENT_ORDER: list[int] = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
                              3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# Radial boundaries (mm from center)
_BULL_RADIUS = 6.35           # inner bull (D25)
_OUTER_BULL_RADIUS = 15.9     # outer bull (25)
_TREBLE_INNER = 99.0          # inner edge of treble ring
_TREBLE_OUTER = 107.0         # outer edge of treble ring
_DOUBLE_INNER = 162.0         # inner edge of double ring
_DOUBLE_OUTER = 170.0         # outer edge of double ring

# Angular width of each segment
_SEGMENT_ANGLE = 2.0 * math.pi / 20  # 18 degrees

# Population prior parameters (derived from PDC tournament data)
_PRIOR_SIGMA_X = 15.0  # mm, population mean radial std
_PRIOR_SIGMA_Y = 18.0  # mm, population mean angular std
_PRIOR_MU_X = 0.0      # mm, population mean aim-point x
_PRIOR_MU_Y = 0.0      # mm, population mean aim-point y


@dataclass
class PlayerSkillPosterior:
    """
    Posterior skill parameters for a single player.

    Attributes
    ----------
    player_id:
        Canonical player identifier.
    mu_x:
        Posterior mean aim-point offset, x-axis (radial, mm).
    mu_y:
        Posterior mean aim-point offset, y-axis (angular, mm).
    sigma_x:
        Posterior aim-point std, x-axis (mm).
    sigma_y:
        Posterior aim-point std, y-axis (mm).
    shrinkage_factor:
        James-Stein shrinkage factor applied (0=full shrinkage, 1=no shrinkage).
    n_throws:
        Number of throws used for estimation.
    segment_accuracies:
        Estimated P(hitting each segment) from the posterior.
    """

    player_id: str
    mu_x: float
    mu_y: float
    sigma_x: float
    sigma_y: float
    shrinkage_factor: float
    n_throws: int
    segment_accuracies: dict[str, float] = field(default_factory=dict)


class EmpiricalBayesSkillModel:
    """
    Empirical Bayes skill model for darts players.

    Estimates per-player aim-point distributions using EB shrinkage.
    For players with limited data, estimates are shrunk toward the
    population prior. As data accumulates, the individual estimate
    dominates.

    Parameters
    ----------
    min_throws_for_individual:
        Minimum throws before individual estimate has nonzero weight.
    prior_sigma_x:
        Population prior std in x-direction (mm).
    prior_sigma_y:
        Population prior std in y-direction (mm).
    """

    def __init__(
        self,
        min_throws_for_individual: int = 50,
        prior_sigma_x: float = _PRIOR_SIGMA_X,
        prior_sigma_y: float = _PRIOR_SIGMA_Y,
    ) -> None:
        self.min_throws = min_throws_for_individual
        self.prior_sigma_x = prior_sigma_x
        self.prior_sigma_y = prior_sigma_y
        self._posteriors: dict[str, PlayerSkillPosterior] = {}
        self._log = logger.bind(component="EmpiricalBayesSkillModel")

    def fit(self, player_id: str, throw_data: list[dict]) -> dict:
        """
        Fit posterior parameters for a single player.

        Uses James-Stein / EB shrinkage to blend individual throw
        statistics with the population prior.

        Parameters
        ----------
        player_id:
            Canonical player ID.
        throw_data:
            List of throw records. Each dict must contain:
                - "target_x": intended x-position (mm from centre)
                - "target_y": intended y-position (mm from centre)
                - "actual_x": actual landing x-position
                - "actual_y": actual landing y-position
                - "segment": segment label (e.g. "T20", "D16", "S1")

        Returns
        -------
        dict
            Posterior parameter summary including shrinkage_factor and
            segment_accuracies.

        Raises
        ------
        DartsMLError
            If throw_data is empty or malformed.
        """
        if not throw_data:
            raise DartsMLError(
                f"Cannot fit EB model for player {player_id}: no throw data."
            )

        required_keys = {"target_x", "target_y", "actual_x", "actual_y", "segment"}
        for i, throw in enumerate(throw_data):
            missing = required_keys - set(throw.keys())
            if missing:
                raise DartsMLError(
                    f"Throw {i} missing keys: {missing}. "
                    f"Required: {required_keys}"
                )

        n = len(throw_data)

        # Compute individual MLE estimates
        deviations_x = np.array(
            [t["actual_x"] - t["target_x"] for t in throw_data],
            dtype=np.float64,
        )
        deviations_y = np.array(
            [t["actual_y"] - t["target_y"] for t in throw_data],
            dtype=np.float64,
        )

        mle_mu_x = float(np.mean(deviations_x))
        mle_mu_y = float(np.mean(deviations_y))
        mle_sigma_x = float(np.std(deviations_x, ddof=1)) if n > 1 else self.prior_sigma_x
        mle_sigma_y = float(np.std(deviations_y, ddof=1)) if n > 1 else self.prior_sigma_y

        # Empirical Bayes shrinkage factor
        # Based on relative precision: individual vs prior.
        # shrinkage = n / (n + kappa) where kappa controls the prior strength.
        # This is equivalent to the posterior mean under a normal-normal model:
        #   posterior_mu = (n * mle_mu / sigma^2 + kappa * prior_mu / tau^2)
        #                  / (n / sigma^2 + kappa / tau^2)
        # Simplified: data_weight = n / (n + min_throws) gives smooth transition.
        data_weight = n / (n + self.min_throws)

        # Also account for relative precision: if individual sigma is much
        # smaller than prior sigma, individual estimate is more informative.
        if mle_sigma_x > 0 and mle_sigma_y > 0:
            precision_ratio_x = (self.prior_sigma_x / mle_sigma_x) ** 2
            precision_ratio_y = (self.prior_sigma_y / mle_sigma_y) ** 2
            precision_factor = min(1.0, (precision_ratio_x + precision_ratio_y) / 2.0)
        else:
            precision_factor = 0.0

        shrinkage = data_weight * max(precision_factor, data_weight)

        # Posterior parameters: blend MLE with prior
        post_mu_x = shrinkage * mle_mu_x + (1.0 - shrinkage) * _PRIOR_MU_X
        post_mu_y = shrinkage * mle_mu_y + (1.0 - shrinkage) * _PRIOR_MU_Y
        post_sigma_x = (
            shrinkage * mle_sigma_x + (1.0 - shrinkage) * self.prior_sigma_x
        )
        post_sigma_y = (
            shrinkage * mle_sigma_y + (1.0 - shrinkage) * self.prior_sigma_y
        )

        # Compute segment accuracies from posterior
        segment_accs = self._compute_segment_accuracies(
            post_mu_x, post_mu_y, post_sigma_x, post_sigma_y
        )

        posterior = PlayerSkillPosterior(
            player_id=player_id,
            mu_x=post_mu_x,
            mu_y=post_mu_y,
            sigma_x=post_sigma_x,
            sigma_y=post_sigma_y,
            shrinkage_factor=shrinkage,
            n_throws=n,
            segment_accuracies=segment_accs,
        )

        self._posteriors[player_id] = posterior

        self._log.info(
            "eb_posterior_fitted",
            player_id=player_id,
            n_throws=n,
            shrinkage=round(shrinkage, 4),
            sigma_x=round(post_sigma_x, 2),
            sigma_y=round(post_sigma_y, 2),
        )

        return {
            "player_id": player_id,
            "shrinkage_factor": shrinkage,
            "sigma_x": post_sigma_x,
            "sigma_y": post_sigma_y,
            "mu_x": post_mu_x,
            "mu_y": post_mu_y,
            "n_throws": n,
            "segment_accuracies": segment_accs,
        }

    def _compute_segment_accuracies(
        self,
        mu_x: float,
        mu_y: float,
        sigma_x: float,
        sigma_y: float,
    ) -> dict[str, float]:
        """
        Compute P(hitting each key segment) from the posterior
        bivariate normal distribution.

        Uses the bivariate normal CDF over the segment geometry
        (simplified to rectangular approximation for efficiency).
        """
        accuracies: dict[str, float] = {}

        # Treble 20: target at (0, _TREBLE_MID) where the midpoint of
        # the treble ring is at ~103mm from center, at angle for segment 20 (top)
        treble_mid = (_TREBLE_INNER + _TREBLE_OUTER) / 2.0
        treble_width = _TREBLE_OUTER - _TREBLE_INNER  # 8mm radial
        angular_width_mm = treble_mid * _SEGMENT_ANGLE  # arc length at treble distance

        # P(hitting T20) = P(radial within treble ring) * P(angular within segment)
        # For T20 target at (treble_mid, 0) in polar coordinates:
        p_radial = self._p_in_band(
            target=treble_mid, lo=_TREBLE_INNER, hi=_TREBLE_OUTER,
            offset=mu_x, sigma=sigma_x,
        )
        p_angular = self._p_in_band(
            target=0.0, lo=-angular_width_mm / 2.0, hi=angular_width_mm / 2.0,
            offset=mu_y, sigma=sigma_y,
        )
        accuracies["T20"] = float(np.clip(p_radial * p_angular, 0.0, 1.0))

        # T19: similar geometry, adjacent segment
        accuracies["T19"] = float(np.clip(
            accuracies["T20"] * 0.95, 0.0, 1.0  # slight adjustment for off-centre
        ))

        # Bull (D25): circular target at centre
        p_bull = self._p_in_circle(
            radius=_BULL_RADIUS, mu_x=mu_x, mu_y=mu_y,
            sigma_x=sigma_x, sigma_y=sigma_y,
        )
        accuracies["D25"] = float(np.clip(p_bull, 0.0, 1.0))

        # Outer bull (25): annular ring
        p_outer_bull = (
            self._p_in_circle(_OUTER_BULL_RADIUS, mu_x, mu_y, sigma_x, sigma_y)
            - p_bull
        )
        accuracies["25"] = float(np.clip(p_outer_bull, 0.0, 1.0))

        # Double ring (generic): same geometry as treble but at double distance
        double_mid = (_DOUBLE_INNER + _DOUBLE_OUTER) / 2.0
        double_width = _DOUBLE_OUTER - _DOUBLE_INNER
        angular_width_double = double_mid * _SEGMENT_ANGLE

        p_double_radial = self._p_in_band(
            target=double_mid, lo=_DOUBLE_INNER, hi=_DOUBLE_OUTER,
            offset=mu_x, sigma=sigma_x,
        )
        p_double_angular = self._p_in_band(
            target=0.0, lo=-angular_width_double / 2.0, hi=angular_width_double / 2.0,
            offset=mu_y, sigma=sigma_y,
        )
        accuracies["D_generic"] = float(np.clip(
            p_double_radial * p_double_angular, 0.0, 1.0
        ))

        return accuracies

    @staticmethod
    def _p_in_band(
        target: float,
        lo: float,
        hi: float,
        offset: float,
        sigma: float,
    ) -> float:
        """
        P(random variable lands in [lo, hi]) given N(target + offset, sigma^2).
        """
        if sigma <= 0:
            return 1.0 if lo <= target + offset <= hi else 0.0
        z_lo = (lo - target - offset) / sigma
        z_hi = (hi - target - offset) / sigma
        return float(norm.cdf(z_hi) - norm.cdf(z_lo))

    @staticmethod
    def _p_in_circle(
        radius: float,
        mu_x: float,
        mu_y: float,
        sigma_x: float,
        sigma_y: float,
    ) -> float:
        """
        Approximate P(landing within circle of given radius) for a
        bivariate normal distribution centred at (mu_x, mu_y).

        Uses a Rayleigh-like approximation: P = 1 - exp(-r^2 / (2*sigma_avg^2))
        adjusted for offset.
        """
        sigma_avg = (sigma_x + sigma_y) / 2.0
        if sigma_avg <= 0:
            dist = math.sqrt(mu_x ** 2 + mu_y ** 2)
            return 1.0 if dist <= radius else 0.0
        dist_sq = mu_x ** 2 + mu_y ** 2
        effective_r_sq = max(0.0, radius ** 2 - dist_sq)
        return float(1.0 - math.exp(-effective_r_sq / (2.0 * sigma_avg ** 2)))

    def p_hit_segment(self, player_id: str, target_segment: str) -> float:
        """
        P(hitting segment) from the posterior for a given player.

        Parameters
        ----------
        player_id:
            Canonical player ID.
        target_segment:
            Segment label (e.g. "T20", "D25", "D_generic").

        Returns
        -------
        float
            Probability in [0, 1].

        Raises
        ------
        DartsMLError
            If no posterior exists for this player.
        """
        if player_id not in self._posteriors:
            raise DartsMLError(
                f"No EB posterior for player {player_id}. Call .fit() first."
            )
        posterior = self._posteriors[player_id]
        if target_segment not in posterior.segment_accuracies:
            raise DartsMLError(
                f"Unknown segment {target_segment!r} for player {player_id}. "
                f"Available: {list(posterior.segment_accuracies.keys())}"
            )
        return posterior.segment_accuracies[target_segment]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict match outcome probability as a base learner in R2 ensemble.

        Uses the ratio of EB skill estimates between the two players.
        The input features must include EB-derived columns for both players.

        Parameters
        ----------
        X:
            Feature array where columns include eb_t20_accuracy_p1/p2,
            eb_d_accuracy_p1/p2, eb_bull_accuracy_p1/p2.
            Shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            P(P1 wins) for each sample, shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        probas = np.zeros(n, dtype=np.float64)

        for i in range(n):
            row = X[i]
            # Extract EB features by position convention:
            # These correspond to the R2 feature columns for EB skill estimates.
            # In the full pipeline, this mapping is done by the feature builder.
            # Here we use a skill-ratio approach as a lightweight base learner.
            n_feat = len(row)
            if n_feat >= 6:
                # Use last 6 EB features: t20_p1, t20_p2, d_p1, d_p2, bull_p1, bull_p2
                # Weighted composite skill score
                p1_skill = 0.5 * row[-6] + 0.3 * row[-4] + 0.2 * row[-2]
                p2_skill = 0.5 * row[-5] + 0.3 * row[-3] + 0.2 * row[-1]
            else:
                # Fallback: equal skill assumption
                p1_skill = 0.5
                p2_skill = 0.5

            total_skill = p1_skill + p2_skill
            if total_skill > 0:
                probas[i] = p1_skill / total_skill
            else:
                probas[i] = 0.5

        return probas

    def get_posterior(self, player_id: str) -> Optional[PlayerSkillPosterior]:
        """Return the posterior for a player, or None if not fitted."""
        return self._posteriors.get(player_id)

    def has_posterior(self, player_id: str) -> bool:
        """Check if a posterior exists for a player."""
        return player_id in self._posteriors
