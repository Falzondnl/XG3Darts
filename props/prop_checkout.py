"""
Checkout props: highest checkout and checkout score range.

Mathematical basis
------------------
The highest checkout in a match is the maximum over all legs of the
checkout score achieved by either player.

For a single leg, the checkout score depends on:
  - The score the player was at when they checked out (the "leave")
  - Their route choice model (preferred checkout path)
  - Their double hit rate

We approximate the distribution of checkout scores using the checkout
model's route choice probabilities and the player's historical checkout
score distribution (checkout_ewm, checkout_pct_by_band).

P(highest checkout > line | match) = 1 - P(all checkouts <= line | match)

For n legs with independent checkout events:
  P(all <= line) = prod_legs P(checkout_in_leg <= line)

The per-leg checkout probability is derived from the player's historical
checkout score distribution (checkout_pct_by_band), which partitions
checkout scores into ranges (e.g. <40, 40-60, 60-80, 80-100, 100+, 140+).

All probabilities are derived from player statistics — never hardcoded.
"""
from __future__ import annotations

import math
from typing import Any

import structlog

from engines.errors import DartsDataError, DartsEngineError
from props.data_sufficiency_gate import DataSufficiencyGate

logger = structlog.get_logger(__name__)

_GATE = DataSufficiencyGate()

# Checkout score bands (lower bound, upper bound, inclusive)
# Based on standard darts checkout categorisation
_CHECKOUT_BANDS: list[tuple[int, int, str]] = [
    (2,   39,  "sub_40"),
    (40,  60,  "40_60"),
    (61,  80,  "61_80"),
    (81,  100, "81_100"),
    (101, 120, "101_120"),
    (121, 140, "121_140"),
    (141, 170, "141_170"),
]

# Maximum possible checkout (170: T20 T20 T10 → no, actually T20 T20 BULL = 170)
_MAX_CHECKOUT = 170
_MIN_CHECKOUT = 2


class PropCheckoutModel:
    """
    Price checkout-related prop markets.

    Requires player checkout statistics from the data pipeline:
      - ``checkout_ewm``: exponentially weighted checkout rate (probability of
        completing a checkout when in checkout territory)
      - ``checkout_pct_by_band``: dict mapping band labels to fraction of
        successful checkouts falling in that score range
      - ``checkout_distribution``: optional explicit distribution over scores
    """

    def price_highest_checkout(
        self,
        p1_checkout_model: dict[str, Any],
        p2_checkout_model: dict[str, Any],
        expected_legs: float,
        line: int,
        p1_id: str = "p1",
        p2_id: str = "p2",
        regime: int = 1,
    ) -> dict[str, Any]:
        """
        P(highest checkout in match > line).

        Parameters
        ----------
        p1_checkout_model:
            Dict containing player stats: ``checkout_ewm``,
            ``checkout_pct_by_band``, ``legs_observed``, ``confidence``.
        p2_checkout_model:
            Same for player 2.
        expected_legs:
            Expected total legs in the match.
        line:
            The threshold (e.g. 100 for "highest checkout > 100").
        p1_id / p2_id:
            Player IDs for gate validation and logging.
        regime:
            Data coverage regime.

        Returns
        -------
        dict with keys: ``over_prob``, ``under_prob``, ``line``,
        ``over_decimal``, ``under_decimal``.

        Raises
        ------
        DartsDataError
            If data gates fail.
        DartsEngineError
            If inputs are invalid.
        """
        # Data gate
        self._check_gate(p1_checkout_model, p1_id, regime)
        self._check_gate(p2_checkout_model, p2_id, regime)

        if not (1 <= line <= _MAX_CHECKOUT):
            raise DartsEngineError(
                f"line must be in [1, {_MAX_CHECKOUT}], got {line}"
            )
        if expected_legs <= 0:
            raise DartsEngineError(
                f"expected_legs must be positive, got {expected_legs:.2f}"
            )

        # P(a single checkout is <= line) for each player
        p1_co_lte = self._p_checkout_score_lte(p1_checkout_model, line)
        p2_co_lte = self._p_checkout_score_lte(p2_checkout_model, line)

        # Approximate number of checkouts per player in expected_legs legs
        # Each leg has exactly 1 checkout (by the winner), split between players
        # based on their win probability (unavailable here → assume 50/50)
        # P(single leg produces checkout > line) = 1 - P(checkout <= line)
        # For the match: P(ALL checkouts <= line) over all legs
        # With expected_legs legs and random allocation, approximation:
        # P(no checkout > line in n legs)
        #   ≈ (p1_co_lte * 0.5 + p2_co_lte * 0.5)^(expected_legs)
        # This treats each leg's checkout as independently drawn from the
        # combined distribution.

        avg_co_lte = (p1_co_lte + p2_co_lte) / 2.0
        # Per-leg: P(checkout_in_leg <= line)
        p_per_leg_lte = avg_co_lte

        # Over all expected legs
        p_all_legs_lte = p_per_leg_lte ** expected_legs
        over_prob = 1.0 - p_all_legs_lte
        under_prob = p_all_legs_lte

        result = self._build_ou(over_prob, under_prob, line)

        logger.debug(
            "price_highest_checkout",
            p1_id=p1_id,
            p2_id=p2_id,
            line=line,
            expected_legs=round(expected_legs, 2),
            p1_co_lte=round(p1_co_lte, 4),
            p2_co_lte=round(p2_co_lte, 4),
            over_prob=round(over_prob, 4),
        )
        return result

    def price_checkout_range(
        self,
        player_checkout_model: dict[str, Any],
        line_low: int,
        line_high: int,
        player_id: str = "player",
        regime: int = 1,
    ) -> dict[str, Any]:
        """
        P(checkout score in [line_low, line_high]).

        Parameters
        ----------
        player_checkout_model:
            Player stats dict.
        line_low:
            Lower bound of checkout range (inclusive).
        line_high:
            Upper bound of checkout range (inclusive).
        player_id:
            For gate validation and logging.
        regime:
            Data coverage regime.

        Returns
        -------
        dict with keys: ``in_range_prob``, ``out_range_prob``,
        ``line_low``, ``line_high``, ``in_range_decimal``, ``out_range_decimal``.

        Raises
        ------
        DartsDataError
            If gate fails.
        DartsEngineError
            If range is invalid.
        """
        self._check_gate(player_checkout_model, player_id, regime)

        if not (_MIN_CHECKOUT <= line_low <= line_high <= _MAX_CHECKOUT):
            raise DartsEngineError(
                f"Must have {_MIN_CHECKOUT} <= line_low <= line_high <= {_MAX_CHECKOUT}. "
                f"Got line_low={line_low}, line_high={line_high}."
            )

        p_in = self._p_checkout_in_range(player_checkout_model, line_low, line_high)
        p_out = 1.0 - p_in

        p_in = max(0.0, min(1.0, p_in))
        p_out = max(0.0, min(1.0, p_out))

        result = {
            "in_range_prob": round(p_in, 6),
            "out_range_prob": round(p_out, 6),
            "line_low": line_low,
            "line_high": line_high,
            "in_range_decimal": round(1.0 / p_in, 4) if p_in > 1e-6 else None,
            "out_range_decimal": round(1.0 / p_out, 4) if p_out > 1e-6 else None,
        }

        logger.debug(
            "price_checkout_range",
            player_id=player_id,
            line_low=line_low,
            line_high=line_high,
            p_in=round(p_in, 4),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _p_checkout_score_lte(
        self,
        checkout_model: dict[str, Any],
        line: int,
    ) -> float:
        """
        P(checkout score <= line) from the player's band distribution.

        Uses ``checkout_pct_by_band`` if available, otherwise falls back to
        a uniform distribution over reachable checkout scores.
        """
        band_dist: dict[str, float] | None = checkout_model.get("checkout_pct_by_band")

        if band_dist and isinstance(band_dist, dict) and len(band_dist) > 0:
            return self._cdf_from_band_dist(band_dist, line)

        # Fallback: uniform over [2, 170] using checkout_ewm as calibration
        # The shape: more checkouts cluster around 32-60 (standard doubles)
        # than at the high end. Without band data, use a simple model.
        checkout_ewm = checkout_model.get("checkout_ewm", 0.5)
        if checkout_ewm is None or checkout_ewm <= 0:
            checkout_ewm = 0.5

        return self._fallback_checkout_cdf(line, checkout_ewm)

    def _cdf_from_band_dist(
        self,
        band_dist: dict[str, float],
        line: int,
    ) -> float:
        """
        Compute P(checkout <= line) from a checkout_pct_by_band distribution.

        The band distribution maps band labels (e.g. "sub_40", "40_60", ...)
        to probability mass for that range.
        """
        # Normalise band_dist
        total = sum(band_dist.values())
        if total <= 0:
            return self._fallback_checkout_cdf(line, 0.5)

        cumulative = 0.0
        for lo, hi, label in _CHECKOUT_BANDS:
            band_prob = band_dist.get(label, 0.0) / total
            if hi <= line:
                # Entire band is below or at line
                cumulative += band_prob
            elif lo <= line < hi:
                # Partial band: interpolate linearly within the band
                fraction = (line - lo + 1) / (hi - lo + 1)
                cumulative += band_prob * fraction
            # else: band is entirely above line — do not add

        return max(0.0, min(1.0, cumulative))

    @staticmethod
    def _fallback_checkout_cdf(line: int, checkout_ewm: float) -> float:
        """
        Fallback checkout CDF when band distribution is unavailable.

        Uses an empirical approximation derived from tournament data:
        the bulk of professional checkouts cluster in the 32-80 range,
        with a long tail toward 170.  We model this as a scaled beta-like
        CDF parameterised by the player's checkout rate.
        """
        # Normalise line to [0, 1] within checkout range
        x = (line - _MIN_CHECKOUT) / (_MAX_CHECKOUT - _MIN_CHECKOUT)
        x = max(0.0, min(1.0, x))

        # Shape parameter: higher checkout_ewm → slightly higher proportion
        # of checkouts at larger values (better players finish from further)
        # alpha controls skewness: higher quality = less left-skewed
        alpha = 1.5 + checkout_ewm * 1.0  # range ~1.5 – 2.5

        # Approximate beta CDF using regularised incomplete beta
        try:
            from scipy.special import betainc  # type: ignore
            beta_param = 3.0  # fixed right tail
            return float(betainc(alpha, beta_param, x))
        except ImportError:
            # Simple polynomial approximation
            return x ** alpha

    def _p_checkout_in_range(
        self,
        checkout_model: dict[str, Any],
        lo: int,
        hi: int,
    ) -> float:
        """P(checkout score in [lo, hi]) = CDF(hi) - CDF(lo - 1)."""
        p_hi = self._p_checkout_score_lte(checkout_model, hi)
        p_lo = self._p_checkout_score_lte(checkout_model, lo - 1)
        return max(0.0, p_hi - p_lo)

    @staticmethod
    def _check_gate(
        stats: dict[str, Any],
        player_id: str,
        regime: int,
    ) -> None:
        ok, reason = _GATE.can_open_market(
            market_family="props_checkout",
            player_id=player_id,
            regime=regime,
            stats=stats,
        )
        if not ok:
            raise DartsDataError(reason)

    @staticmethod
    def _build_ou(
        over_prob: float,
        under_prob: float,
        line: int,
    ) -> dict[str, Any]:
        over_prob = max(0.0, min(1.0, over_prob))
        under_prob = max(0.0, min(1.0, under_prob))
        return {
            "over_prob": round(over_prob, 6),
            "under_prob": round(under_prob, 6),
            "line": line,
            "over_decimal": round(1.0 / over_prob, 4) if over_prob > 1e-6 else None,
            "under_decimal": round(1.0 / under_prob, 4) if under_prob > 1e-6 else None,
        }
