"""
Market-family calibration registry.

Seven separate calibrators -- one per market family.
NEVER use a single global calibrator. Each market family has distinct
probability distributions and calibration curves.

Market families:
    match_winner    - Match outcome probabilities
    leg_handicap    - Leg handicap line probabilities
    totals          - Total legs/sets over/under
    exact_score     - Exact score line markets
    props_180       - 180s and high-score propositions
    props_checkout  - Checkout percentage propositions
    outright        - Tournament outright winner
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import structlog

from calibration.beta_calibrator import BetaCalibrationError, BetaCalibrator

logger = structlog.get_logger(__name__)


class DartsCalibrationError(Exception):
    """Raised when market calibration encounters errors."""


# -----------------------------------------------------------------------
# Canonical market family names
# -----------------------------------------------------------------------

MARKET_FAMILIES: tuple[str, ...] = (
    "match_winner",
    "leg_handicap",
    "totals",
    "exact_score",
    "props_180",
    "props_checkout",
    "outright",
)


class MarketCalibrationRegistry:
    """
    Registry of per-market-family beta calibrators.

    Each market family has its own independently fitted BetaCalibrator.
    The registry enforces that calibrators are never shared or leaked
    across families.

    Usage:
        registry = MarketCalibrationRegistry()
        registry.fit("match_winner", raw_proba, y_true)
        calibrated = registry.calibrate("match_winner", raw_proba)
    """

    def __init__(self) -> None:
        self._calibrators: dict[str, BetaCalibrator] = {
            family: BetaCalibrator(market_family=family)
            for family in MARKET_FAMILIES
        }
        self._log = logger.bind(component="MarketCalibrationRegistry")

    def get_calibrator(self, market_family: str) -> BetaCalibrator:
        """
        Retrieve the calibrator for a specific market family.

        Parameters
        ----------
        market_family:
            One of the canonical market family names.

        Returns
        -------
        BetaCalibrator

        Raises
        ------
        DartsCalibrationError
            If the market family is unknown.
        """
        if market_family not in self._calibrators:
            raise DartsCalibrationError(
                f"Unknown market family {market_family!r}. "
                f"Valid families: {list(MARKET_FAMILIES)}"
            )
        return self._calibrators[market_family]

    def fit(
        self,
        market_family: str,
        raw_proba: np.ndarray,
        y_true: np.ndarray,
    ) -> None:
        """
        Fit the calibrator for one market family.

        Parameters
        ----------
        market_family:
            The market family to calibrate.
        raw_proba:
            Raw model probabilities.
        y_true:
            Binary ground truth labels.

        Raises
        ------
        DartsCalibrationError
            If the market family is unknown.
        BetaCalibrationError
            If calibration fitting fails.
        """
        calibrator = self.get_calibrator(market_family)
        calibrator.fit(raw_proba, y_true)
        self._log.info(
            "market_calibrator_fitted",
            market_family=market_family,
            a=calibrator.a_,
            b=calibrator.b_,
            n_samples=len(raw_proba),
        )

    def calibrate(
        self,
        market_family: str,
        raw_proba: np.ndarray,
    ) -> np.ndarray:
        """
        Calibrate raw probabilities for a market family.

        Parameters
        ----------
        market_family:
            The market family.
        raw_proba:
            Raw model probabilities.

        Returns
        -------
        np.ndarray
            Calibrated probabilities.

        Raises
        ------
        DartsCalibrationError
            If the market family is unknown or calibrator is not fitted.
        """
        calibrator = self.get_calibrator(market_family)
        if not calibrator.fitted_:
            raise DartsCalibrationError(
                f"Calibrator for {market_family!r} has not been fitted. "
                f"Call .fit() first."
            )
        return calibrator.predict_proba(raw_proba)

    def fit_all(
        self,
        data: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, bool]:
        """
        Fit all calibrators from a mapping of family -> (proba, y_true).

        Parameters
        ----------
        data:
            Mapping of market_family -> (raw_proba, y_true).

        Returns
        -------
        dict[str, bool]
            Mapping of market_family -> fit_success.
        """
        results: dict[str, bool] = {}
        for family, (proba, y_true) in data.items():
            try:
                self.fit(family, proba, y_true)
                results[family] = True
            except (BetaCalibrationError, DartsCalibrationError) as exc:
                self._log.warning(
                    "market_calibrator_fit_failed",
                    market_family=family,
                    error=str(exc),
                )
                results[family] = False
        return results

    def is_fitted(self, market_family: str) -> bool:
        """Check if a specific market family calibrator has been fitted."""
        calibrator = self.get_calibrator(market_family)
        return calibrator.fitted_

    def all_fitted(self) -> bool:
        """Check if all market family calibrators have been fitted."""
        return all(c.fitted_ for c in self._calibrators.values())

    def status(self) -> dict[str, dict[str, object]]:
        """
        Return status of all calibrators.

        Returns
        -------
        dict
            Mapping of market_family -> {fitted, a, b, n_train}.
        """
        return {
            family: cal.get_params()
            for family, cal in self._calibrators.items()
        }

    @property
    def families(self) -> tuple[str, ...]:
        """Return canonical market family names."""
        return MARKET_FAMILIES

    def __len__(self) -> int:
        return len(self._calibrators)

    def __repr__(self) -> str:
        fitted_count = sum(1 for c in self._calibrators.values() if c.fitted_)
        return (
            f"MarketCalibrationRegistry("
            f"{fitted_count}/{len(self._calibrators)} fitted)"
        )
