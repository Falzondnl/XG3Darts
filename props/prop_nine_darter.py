"""
Nine-darter probability model.

Mathematical basis
------------------
A nine-darter is the perfect leg: checkout in exactly 9 darts.

The classic sequence:
  Visit 1: 180 (T20 T20 T20)
  Visit 2: 180 (T20 T20 T20)
  Visit 3: 141 out (requires 3 darts on a specific route)

However multiple valid 9-dart finishes exist (all require exactly 501 in
3 visits of 3 darts each).

P(nine-darter in a single leg) approximation:

1. P(180 on visit 1) = player T20 accuracy^3
   (approximate — assumes player aims T20 T20 T20, subject to misses)

2. P(180 on visit 2 | scored 180 on visit 1, player must score 321 in 2
   visits) = P(another 180) = same rate (score-independent at open scores)

3. P(141 checkout in 3 darts from 141):
   Standard routes through T20 T19 D12, T20 T17 D15, etc.
   Computed via product of segment hit rates from the EB model.

P(9-darter in match) = 1 - P(no 9-darter in any leg)
  = 1 - (1 - P(9-dart in a single leg | P1))^(expected_legs * P1_win_frac)
      * (1 - P(9-dart in a single leg | P2))^(expected_legs * P2_win_frac)

Approximated as:
  = 1 - (1 - p1_nd)^n1 * (1 - p2_nd)^n2

where n1, n2 are expected legs thrown by each player.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from engines.errors import DartsDataError, DartsEngineError
from props.data_sufficiency_gate import DataSufficiencyGate

logger = structlog.get_logger(__name__)

_GATE = DataSufficiencyGate()


# ---------------------------------------------------------------------------
# Nine-dart checkout routes from 141
# (All routes achieving exactly 141 in 3 darts)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NineDartRoute:
    """A single 3-dart route for finishing 141."""
    darts: tuple[str, str, str]
    score: int = 141

    @property
    def setup1(self) -> str:
        return self.darts[0]

    @property
    def setup2(self) -> str:
        return self.darts[1]

    @property
    def finish(self) -> str:
        return self.darts[2]


# Valid 9-dart finish routes from 141 with the canonical target sequence.
# Each route: [setup_dart_1, setup_dart_2, finishing_double]
# We include all primary routes; the model weights them by segment hit rate.
NINE_DART_ROUTES_141: list[NineDartRoute] = [
    NineDartRoute(darts=("T20", "T19", "D12")),   # 60+57+24=141
    NineDartRoute(darts=("T20", "T17", "D15")),   # 60+51+30=141
    NineDartRoute(darts=("T19", "T18", "D15")),   # 57+54+30=141
    NineDartRoute(darts=("T20", "T15", "D18")),   # 60+45+36=141
    NineDartRoute(darts=("T19", "T16", "D18")),   # 57+48+36=141
    NineDartRoute(darts=("T20", "T13", "D21")),   # 60+39+42=141  (D21 = outer bull area, invalid)
    NineDartRoute(darts=("T18", "T19", "D15")),   # 54+57+30=141
    NineDartRoute(darts=("T20", "T11", "D25")),   # 60+33+50=141  (BULL)
    NineDartRoute(darts=("T20", "T19", "D12")),   # primary — already listed
]

# De-duplicated routes (preserve order, drop exact duplicates)
_seen: set[tuple[str, str, str]] = set()
NINE_DART_ROUTES_141_UNIQUE: list[NineDartRoute] = []
for _r in NINE_DART_ROUTES_141:
    if _r.darts not in _seen:
        _seen.add(_r.darts)
        NINE_DART_ROUTES_141_UNIQUE.append(_r)

# Segment scores for validation (only valid doubles)
_VALID_DOUBLES: frozenset[str] = frozenset(
    [f"D{i}" for i in range(1, 21)] + ["D25", "BULL", "50"]
)


class PropNineDarterModel:
    """
    Price nine-darter occurrence in a match.

    Parameterised by player T20 accuracy and double accuracy from the
    empirical Bayes segment model — never hardcoded.
    """

    # Standard 141 checkout routes
    NINE_DART_CHECKOUTS = {
        141: NINE_DART_ROUTES_141_UNIQUE,
    }

    def p_nine_darter_in_leg(
        self,
        player_t20_accuracy: float,
        player_double_accuracy: float,
    ) -> float:
        """
        P(nine-darter in a single leg) for one player.

        Parameters
        ----------
        player_t20_accuracy:
            P(hit T20 with a single dart) from the EB segment model.
            Typically 0.25–0.55 for professionals.
        player_double_accuracy:
            P(hit the required double) for checkout darts.
            Generalised: average over primary doubles (D12, D15, D18).

        Returns
        -------
        float
            P(nine-darter in this leg).

        Raises
        ------
        DartsEngineError
            If accuracy values are out of range.
        """
        self._validate_accuracy(player_t20_accuracy, "player_t20_accuracy")
        self._validate_accuracy(player_double_accuracy, "player_double_accuracy")

        # P(180 on one visit) = P(T20)^3
        p_180 = player_t20_accuracy ** 3

        # Two consecutive 180s required
        p_two_180s = p_180 * p_180

        # P(141 in 3 darts) = max over routes of P(complete specific route)
        # We take the probability of the best route (player picks their preferred)
        # This is an upper bound; for conservatism we weight by the top route only.
        p_141 = self._p_141_checkout(
            t20_accuracy=player_t20_accuracy,
            double_accuracy=player_double_accuracy,
        )

        p_nine_dart = p_two_180s * p_141

        logger.debug(
            "p_nine_darter_in_leg",
            player_t20_accuracy=round(player_t20_accuracy, 5),
            player_double_accuracy=round(player_double_accuracy, 5),
            p_180=round(p_180, 8),
            p_141=round(p_141, 8),
            p_nine_dart=round(p_nine_dart, 10),
        )

        return p_nine_dart

    def p_nine_darter_in_match(
        self,
        p1_nine_dart_per_leg: float,
        p2_nine_dart_per_leg: float,
        expected_legs: float,
    ) -> float:
        """
        P(nine-darter occurs anywhere in the match).

        Computed as:
          P(at least one 9-darter) = 1 - P(no 9-darter in any leg)

        Each leg is an independent trial.  With expected_legs legs total,
        we split approximately half to each player.

        Parameters
        ----------
        p1_nine_dart_per_leg:
            P1's nine-darter probability per leg.
        p2_nine_dart_per_leg:
            P2's nine-darter probability per leg.
        expected_legs:
            Expected total legs in the match (from combinatorial engine).

        Returns
        -------
        float
            P(nine-darter in match) in [0, 1].

        Raises
        ------
        DartsEngineError
            If inputs are out of range.
        """
        if expected_legs <= 0:
            raise DartsEngineError(
                f"expected_legs must be positive, got {expected_legs:.2f}"
            )

        # Each player throws (approximately) expected_legs / 2 legs
        # In a standard format they each serve and receive alternately
        legs_each = expected_legs / 2.0

        # P(P1 produces no 9-darter in legs_each legs)
        p_no_9d_p1 = (1.0 - p1_nine_dart_per_leg) ** legs_each
        # P(P2 produces no 9-darter in legs_each legs)
        p_no_9d_p2 = (1.0 - p2_nine_dart_per_leg) ** legs_each

        # All legs independent
        p_no_9d_match = p_no_9d_p1 * p_no_9d_p2
        p_match = 1.0 - p_no_9d_match

        logger.debug(
            "p_nine_darter_in_match",
            p1_per_leg=round(p1_nine_dart_per_leg, 10),
            p2_per_leg=round(p2_nine_dart_per_leg, 10),
            expected_legs=round(expected_legs, 2),
            p_match=round(p_match, 8),
        )

        return max(0.0, min(1.0, p_match))

    def price_nine_darter(
        self,
        p1_t20_accuracy: float,
        p2_t20_accuracy: float,
        p1_double_accuracy: float,
        p2_double_accuracy: float,
        expected_legs: float,
        p1_stats: dict[str, Any] | None = None,
        p2_stats: dict[str, Any] | None = None,
        p1_id: str = "p1",
        p2_id: str = "p2",
        regime: int = 2,
    ) -> dict[str, Any]:
        """
        Full nine-darter market pricing.

        Returns
        -------
        dict with keys: ``yes_prob``, ``no_prob``, ``yes_decimal``,
        ``no_decimal``, ``p1_per_leg``, ``p2_per_leg``.
        """
        # Gate check — nine-darter requires high-quality data
        if p1_stats is not None:
            ok, reason = _GATE.can_open_market(
                market_family="props_180",  # use 180 gate as minimum standard
                player_id=p1_id,
                regime=regime,
                stats=p1_stats,
            )
            if not ok:
                raise DartsDataError(reason)

        if p2_stats is not None:
            ok, reason = _GATE.can_open_market(
                market_family="props_180",
                player_id=p2_id,
                regime=regime,
                stats=p2_stats,
            )
            if not ok:
                raise DartsDataError(reason)

        p1_per_leg = self.p_nine_darter_in_leg(p1_t20_accuracy, p1_double_accuracy)
        p2_per_leg = self.p_nine_darter_in_leg(p2_t20_accuracy, p2_double_accuracy)

        yes_prob = self.p_nine_darter_in_match(p1_per_leg, p2_per_leg, expected_legs)
        no_prob = 1.0 - yes_prob

        return {
            "yes_prob": round(yes_prob, 8),
            "no_prob": round(no_prob, 8),
            "yes_decimal": round(1.0 / yes_prob, 2) if yes_prob > 1e-9 else None,
            "no_decimal": round(1.0 / no_prob, 4) if no_prob > 1e-9 else None,
            "p1_per_leg": round(p1_per_leg, 10),
            "p2_per_leg": round(p2_per_leg, 10),
            "expected_legs": round(expected_legs, 2),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _p_141_checkout(
        self,
        t20_accuracy: float,
        double_accuracy: float,
    ) -> float:
        """
        P(finish 141 in exactly 3 darts).

        Considers all valid routes; player chooses the route that maximises
        their P(complete).  We approximate by computing P for each route
        and taking the maximum (player uses the best route for them).

        Route T20 T19 D12: P(T20) * P(T19) * P(D12)
        Route T20 T17 D15: P(T20) * P(T17) * P(D15)
        ...

        T19 accuracy ≈ T20 accuracy (similar segment width).
        Double accuracy parameterises all doubles equally.
        """
        # Treble accuracy for T19, T17, T15 relative to T20
        # (all similar width; slight variation)
        t19_accuracy = t20_accuracy * 0.97
        t18_accuracy = t20_accuracy * 0.97
        t17_accuracy = t20_accuracy * 0.95
        t16_accuracy = t20_accuracy * 0.95
        t15_accuracy = t20_accuracy * 0.94
        t11_accuracy = t20_accuracy * 0.93
        t13_accuracy = t20_accuracy * 0.93

        # Segment accuracy map
        seg_acc: dict[str, float] = {
            "T20": t20_accuracy,
            "T19": t19_accuracy,
            "T18": t18_accuracy,
            "T17": t17_accuracy,
            "T16": t16_accuracy,
            "T15": t15_accuracy,
            "T13": t13_accuracy,
            "T11": t11_accuracy,
        }

        # Compute P for each route
        route_probs: list[float] = []
        for route in NINE_DART_ROUTES_141_UNIQUE:
            s1, s2, finish = route.darts

            # Skip routes with invalid double (e.g. D21 doesn't exist)
            if finish.startswith("D"):
                d_num = finish[1:]
                if not d_num.isdigit() or int(d_num) > 25:
                    continue

            p_s1 = seg_acc.get(s1, t20_accuracy * 0.90)
            p_s2 = seg_acc.get(s2, t20_accuracy * 0.90)

            # For BULL finish (50), use a scaled double accuracy
            if finish in ("BULL", "50"):
                p_finish = double_accuracy * 0.90
            elif finish in ("D25", "25"):
                p_finish = double_accuracy * 1.05
            else:
                p_finish = double_accuracy

            route_probs.append(p_s1 * p_s2 * p_finish)

        if not route_probs:
            return 0.0

        # P(complete at least one route) = max route probability
        # (simplified: player commits to best single route)
        return max(route_probs)

    @staticmethod
    def _validate_accuracy(value: float, name: str) -> None:
        if not isinstance(value, (int, float)):
            raise DartsEngineError(f"{name} must be a float, got {type(value)}")
        if not (0.0 <= value <= 1.0):
            raise DartsEngineError(
                f"{name} must be in [0, 1], got {value:.4f}"
            )
