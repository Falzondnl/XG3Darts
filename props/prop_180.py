"""
180 frequency model using a Poisson process.

Mathematical basis
------------------
A player's probability of scoring 180 on a single 3-dart visit is modelled
as an independent Bernoulli trial with rate ``r = pct_180_ewm`` (the
exponentially weighted moving average of 180 hits per visit).

Over ``n`` independent visits the count of 180s follows:

    X ~ Poisson(lambda)   where  lambda = r * n

This is the standard Poisson approximation to the sum of independent
Bernoulli trials, valid when n is large and r is small.

Over/under lines are priced as:

    P(X > line) = 1 - CDF(floor(line), lambda)    [over]
    P(X <= line) = CDF(floor(line), lambda)        [under]

where CDF(k, lambda) = P(Poisson(lambda) <= k).

All rates are derived from player statistics — never hardcoded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import structlog

from engines.errors import DartsDataError, DartsEngineError
from props.data_sufficiency_gate import DataSufficiencyGate

logger = structlog.get_logger(__name__)

_GATE = DataSufficiencyGate()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poisson_cdf(k: int, lam: float) -> float:
    """
    P(Poisson(lam) <= k).

    Computed via the regularised incomplete gamma function identity:
        P(X <= k) = Q(k+1, lam)  = 1 - P(k+1, lam)

    Uses the log-sum-exp recurrence for numerical stability.

    Parameters
    ----------
    k:
        Non-negative integer threshold.
    lam:
        Poisson rate parameter (>= 0).

    Returns
    -------
    float in [0, 1].
    """
    if lam < 0.0:
        raise DartsEngineError(f"Poisson lambda must be >= 0, got {lam:.4f}")
    if k < 0:
        return 0.0
    if lam == 0.0:
        return 1.0

    # Use the regularised upper incomplete gamma function
    # P(X <= k) = Gamma(k+1, lambda) / Gamma(k+1) via scipy if available,
    # otherwise compute via direct summation (sufficient for typical darts values)
    try:
        from scipy.special import gammaincc  # type: ignore
        # scipy.special.gammaincc(a, x) is the regularised UPPER incomplete gamma Q(a, x).
        # For the Poisson CDF: P(Poisson(lam) <= k) = Q(k+1, lam) = gammaincc(k+1, lam).
        return float(gammaincc(k + 1, lam))
    except ImportError:
        pass

    # Fallback: direct summation for small k / lambda
    # P(X <= k) = sum_{j=0}^{k} exp(-lam) * lam^j / j!
    # Use log-space accumulation
    if lam > 700:
        # Very high lambda — practically all mass above any reasonable k
        return 0.0

    log_lam = math.log(lam)
    log_term = -lam  # log of P(X=0) = exp(-lam)
    total = math.exp(log_term)

    for j in range(1, k + 1):
        log_term += log_lam - math.log(j)
        # Guard against underflow / overflow
        if log_term < -700:
            break
        total += math.exp(log_term)
        if total >= 1.0 - 1e-14:
            return 1.0

    return min(1.0, max(0.0, total))


@dataclass
class OverUnderPrices:
    """Over/under prices for a prop market."""
    over_prob: float    # P(statistic > line)
    under_prob: float   # P(statistic <= line)
    line: float         # The O/U line
    lam: float          # Poisson lambda parameter
    over_decimal: float  # Decimal odds for over (no margin applied)
    under_decimal: float  # Decimal odds for under (no margin applied)


class Prop180Model:
    """
    Price over/under 180 count markets using a Poisson process.

    The model is parameterised entirely by player visit statistics
    (pct_180_ewm) from the data pipeline — no hardcoded rates.
    """

    def price_total_180s(
        self,
        p1_180_rate: float,
        p2_180_rate: float,
        expected_legs: float,
        expected_visits_per_leg: float,
        line: float,
        p1_stats: dict[str, Any] | None = None,
        p2_stats: dict[str, Any] | None = None,
        p1_id: str = "p1",
        p2_id: str = "p2",
        regime: int = 1,
    ) -> dict[str, Any]:
        """
        Price total 180s (both players combined) in a match.

        Parameters
        ----------
        p1_180_rate:
            P1 probability of scoring 180 on a single visit (from pct_180_ewm).
        p2_180_rate:
            P2 probability of scoring 180 on a single visit.
        expected_legs:
            Expected total legs in the match (from legs distribution).
        expected_visits_per_leg:
            Expected 3-dart visits per leg (per player).
        line:
            Over/under line (e.g. 6.5).
        p1_stats / p2_stats:
            Full stats dicts for data-gate validation (if provided).
        p1_id / p2_id:
            Player IDs for logging.
        regime:
            Data coverage regime (0/1/2).

        Returns
        -------
        dict with keys: ``over_prob``, ``under_prob``, ``line``, ``lam``,
        ``over_decimal``, ``under_decimal``, ``p1_lambda``, ``p2_lambda``.

        Raises
        ------
        DartsDataError
            If stats do not pass the 180 gate.
        DartsEngineError
            If rates are out of range.
        """
        # Data gate check
        if p1_stats is not None:
            self._check_gate(p1_stats, p1_id, regime)
        if p2_stats is not None:
            self._check_gate(p2_stats, p2_id, regime)

        self._validate_rate(p1_180_rate, "p1_180_rate")
        self._validate_rate(p2_180_rate, "p2_180_rate")
        self._validate_positive(expected_legs, "expected_legs")
        self._validate_positive(expected_visits_per_leg, "expected_visits_per_leg")

        p1_visits = self._expected_visits_in_match(
            expected_legs=expected_legs,
            expected_darts_per_leg=expected_visits_per_leg * 3.0,
        )
        p2_visits = p1_visits  # symmetric by default

        p1_lambda = p1_180_rate * p1_visits
        p2_lambda = p2_180_rate * p2_visits
        total_lambda = p1_lambda + p2_lambda

        # Poisson sum: X1 + X2 ~ Poisson(lam1 + lam2) for independent Poisson r.v.s
        k = int(math.floor(line))
        under_prob = _poisson_cdf(k, total_lambda)
        over_prob = 1.0 - under_prob

        result = self._build_result(
            over_prob=over_prob,
            under_prob=under_prob,
            line=line,
            lam=total_lambda,
        )
        result["p1_lambda"] = round(p1_lambda, 4)
        result["p2_lambda"] = round(p2_lambda, 4)

        logger.debug(
            "price_total_180s",
            p1_id=p1_id,
            p2_id=p2_id,
            p1_180_rate=round(p1_180_rate, 5),
            p2_180_rate=round(p2_180_rate, 5),
            expected_legs=round(expected_legs, 2),
            total_lambda=round(total_lambda, 4),
            line=line,
            over_prob=round(over_prob, 4),
        )
        return result

    def price_player_180s(
        self,
        player_180_rate: float,
        expected_visits: float,
        line: float,
        player_stats: dict[str, Any] | None = None,
        player_id: str = "player",
        regime: int = 1,
    ) -> dict[str, Any]:
        """
        Price a single player's 180 count.

        Parameters
        ----------
        player_180_rate:
            P(180 on a single visit).
        expected_visits:
            Expected number of 3-dart visits by this player in the match.
        line:
            Over/under line.
        player_stats:
            Full stats dict for gate validation.
        player_id:
            For logging.
        regime:
            Data coverage regime.

        Returns
        -------
        dict with keys: ``over_prob``, ``under_prob``, ``line``, ``lam``,
        ``over_decimal``, ``under_decimal``.
        """
        if player_stats is not None:
            self._check_gate(player_stats, player_id, regime)

        self._validate_rate(player_180_rate, "player_180_rate")
        self._validate_positive(expected_visits, "expected_visits")

        lam = player_180_rate * expected_visits
        k = int(math.floor(line))
        under_prob = _poisson_cdf(k, lam)
        over_prob = 1.0 - under_prob

        result = self._build_result(
            over_prob=over_prob,
            under_prob=under_prob,
            line=line,
            lam=lam,
        )

        logger.debug(
            "price_player_180s",
            player_id=player_id,
            rate=round(player_180_rate, 5),
            expected_visits=round(expected_visits, 2),
            lam=round(lam, 4),
            line=line,
            over_prob=round(over_prob, 4),
        )
        return result

    def _expected_visits_in_match(
        self,
        expected_legs: float,
        expected_darts_per_leg: float,
    ) -> float:
        """
        Expected number of 3-dart visits per player in the match.

        Each visit consists of exactly 3 darts.  The expected number of
        visits per leg for one player = expected_darts_per_leg / 3.
        Multiplied by expected legs.

        Parameters
        ----------
        expected_legs:
            Expected total number of legs played.
        expected_darts_per_leg:
            Expected total darts thrown per leg (both players).

        Returns
        -------
        float
            Expected 3-dart visits per player.
        """
        # Each player throws approximately half the darts in a leg
        darts_per_player_per_leg = expected_darts_per_leg / 2.0
        visits_per_player_per_leg = darts_per_player_per_leg / 3.0
        return visits_per_player_per_leg * expected_legs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_gate(
        stats: dict[str, Any],
        player_id: str,
        regime: int,
    ) -> None:
        ok, reason = _GATE.can_open_market(
            market_family="props_180",
            player_id=player_id,
            regime=regime,
            stats=stats,
        )
        if not ok:
            raise DartsDataError(reason)

    @staticmethod
    def _validate_rate(value: float, name: str) -> None:
        if not isinstance(value, (int, float)):
            raise DartsEngineError(f"{name} must be a float, got {type(value)}")
        if not (0.0 <= value <= 1.0):
            raise DartsEngineError(
                f"{name} must be in [0, 1], got {value:.4f}"
            )

    @staticmethod
    def _validate_positive(value: float, name: str) -> None:
        if not isinstance(value, (int, float)) or value <= 0.0:
            raise DartsEngineError(
                f"{name} must be positive, got {value}"
            )

    @staticmethod
    def _build_result(
        over_prob: float,
        under_prob: float,
        line: float,
        lam: float,
    ) -> dict[str, Any]:
        over_prob = max(0.0, min(1.0, over_prob))
        under_prob = max(0.0, min(1.0, under_prob))
        return {
            "over_prob": round(over_prob, 6),
            "under_prob": round(under_prob, 6),
            "line": line,
            "lam": round(lam, 5),
            "over_decimal": round(1.0 / over_prob, 4) if over_prob > 1e-6 else None,
            "under_decimal": round(1.0 / under_prob, 4) if under_prob > 1e-6 else None,
        }
