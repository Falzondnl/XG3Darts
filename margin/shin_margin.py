"""
Shin (1993) margin model for overround allocation.

Shin, H.S. (1993). "Measuring the Incidence of Insider Trading in a Market for
State-Contingent Claims." Economic Journal 103, pp. 1141-1153.

The Shin model allocates overround non-uniformly: underdogs receive a larger
share of the overround than favourites.  This reflects the empirical finding
that insiders concentrate bets on long-shot outcomes, forcing bookmakers to
shade those prices more aggressively.

Mathematical basis
------------------
Given true probabilities q_i (sum to 1) and a target overround O,
the Shin-adjusted market probabilities are:

    Step 1 (raw Shin prices, proportional to q_i):
        r_i = (sqrt(Z^2 + 4*(1-Z)*q_i) - Z) / (2*(1-Z))

    The raw r_i sum to a value that depends on Z and the distribution of q_i.

    Step 2 (scale to target overround):
        p_i = r_i * (1 + O) / sum(r_i)

This gives:
  - sum(p_i) = 1 + O   [desired overround]
  - p_i / q_i > 1 for underdogs (Z > 0 shifts relative margin toward long shots)
  - p_i / q_i < 1 for favourites (they absorb proportionally less overround)

Note: this is the widely-cited practical formulation of the Shin model as
used in sports-betting analytics (e.g. Franck, Verbeek, Nüesch 2010;
Baker and McHale 2013).  The original Shin paper derives it from the
perspective of equilibrium under insider trading.
"""
from __future__ import annotations

import math
from typing import Optional

import structlog

from engines.errors import DartsEngineError

logger = structlog.get_logger(__name__)


class ShinMarginModel:
    """
    Apply the Shin (1993) overround model to a set of true probabilities.

    The model allocates more margin to underdogs (long shots) than to
    favourites, mimicking the empirical pattern observed in sports betting.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply_shin_margin(
        self,
        true_probs: dict[str, float],
        target_margin: float,
        z_param: float = 0.08,
    ) -> dict[str, float]:
        """
        Return overround-adjusted market prices using the Shin model.

        Parameters
        ----------
        true_probs:
            Dict mapping outcome label → true probability.  Values must be
            non-negative and sum to 1.0 (within 1e-3 tolerance).
        target_margin:
            Desired bookmaker overround, e.g. 0.05 for 5 %.  Must be in [0, 0.20].
        z_param:
            Shin insider-trading parameter.  Typical range 0.04–0.12.
            Higher values push more overround onto longer shots.
            z_param=0 reduces to uniform pro-rata scaling.

        Returns
        -------
        dict[str, float]
            Market prices (implied probabilities) adjusted for overround.
            The values sum to approximately 1 + target_margin.

        Raises
        ------
        DartsEngineError
            If true_probs is empty, contains negative values, fails the
            normalisation check, or z_param is out of range.
        """
        if not true_probs:
            raise DartsEngineError("true_probs must not be empty")

        # Validate inputs
        self._validate_true_probs(true_probs)
        if not (0.0 <= target_margin <= 0.20):
            raise DartsEngineError(
                f"target_margin must be in [0, 0.20], got {target_margin:.4f}"
            )
        if not (0.0 <= z_param < 1.0):
            raise DartsEngineError(
                f"z_param must be in [0, 1), got {z_param:.4f}"
            )

        outcomes = list(true_probs.keys())
        qs = [true_probs[k] for k in outcomes]

        # Normalise to ensure exact sum = 1
        total_q = sum(qs)
        qs = [q / total_q for q in qs]

        # Special case: zero margin → return normalised true probabilities
        if target_margin < 1e-10:
            result = {k: q for k, q in zip(outcomes, qs)}
            logger.debug("shin_margin_zero_target", n_outcomes=len(outcomes))
            return result

        # Special case: z_param ≈ 0 → uniform (pro-rata) scaling
        if z_param < 1e-8:
            adjusted = {k: q * (1.0 + target_margin) for k, q in zip(outcomes, qs)}
            logger.debug("shin_margin_uniform_scaling", n_outcomes=len(outcomes))
            return adjusted

        # Compute raw Shin prices using the standard formulation
        # r_i = (sqrt(Z^2 + 4*(1-Z)*q_i) - Z) / (2*(1-Z))
        z = z_param
        raw_prices: list[float] = []
        for q in qs:
            inner = z * z + 4.0 * (1.0 - z) * q
            r_i = (math.sqrt(max(0.0, inner)) - z) / (2.0 * (1.0 - z))
            raw_prices.append(max(0.0, r_i))

        # Scale to target sum = 1 + target_margin
        raw_sum = sum(raw_prices)
        if raw_sum <= 0.0:
            raise DartsEngineError(
                "Shin model produced zero raw prices — all input probabilities are zero."
            )

        target_sum = 1.0 + target_margin
        scale = target_sum / raw_sum
        adjusted_probs = [r * scale for r in raw_prices]

        # Verify sum is close to target
        actual_sum = sum(adjusted_probs)
        if abs(actual_sum - target_sum) > 1e-4:
            logger.warning(
                "shin_margin_sum_deviation",
                target_sum=target_sum,
                actual_sum=actual_sum,
                deviation=abs(actual_sum - target_sum),
            )

        result = dict(zip(outcomes, adjusted_probs))

        logger.debug(
            "shin_margin_applied",
            n_outcomes=len(outcomes),
            target_margin=round(target_margin, 4),
            z_param=round(z_param, 4),
            actual_overround=round(actual_sum - 1.0, 5),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_true_probs(self, true_probs: dict[str, float]) -> None:
        """Validate probability values."""
        for k, v in true_probs.items():
            if not isinstance(v, (int, float)):
                raise DartsEngineError(
                    f"Probability for '{k}' is not numeric: {v!r}"
                )
            if v < 0.0:
                raise DartsEngineError(
                    f"Probability for '{k}' is negative: {v:.6f}"
                )

        total = sum(true_probs.values())
        if total <= 0.0:
            raise DartsEngineError(
                f"true_probs sum is {total:.6f} — all probabilities are zero"
            )
        if abs(total - 1.0) > 1e-3:
            raise DartsEngineError(
                f"true_probs must sum to 1.0 (got {total:.6f}, "
                f"deviation={abs(total-1.0):.6f})"
            )
