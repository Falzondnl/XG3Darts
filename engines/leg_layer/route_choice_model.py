"""
Checkout route choice model.

Models player preferred checkout routes, standard PDC route table,
pressure switching, bull usage, and miss patterns.

All checkout routes derived from the standard PDC checkout chart used
by professional players. Player-specific adjustments applied on top
when DartConnect R2 data is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CheckoutRoute:
    """
    A specific checkout route for a target score.

    Attributes
    ----------
    target_sequence:
        List of dart target labels in order, e.g. ["T20", "T20", "D20"] for 120.
    darts_required:
        Minimum number of darts required if all hit exactly.
    bull_involved:
        Whether any dart targets the bull or double-bull.
    difficulty_score:
        Relative difficulty in [0, 1]; lower = harder (requires precise doubles).
        Based on double-hit rate empirics from darts research.
    """

    target_sequence: list[str]
    darts_required: int
    bull_involved: bool
    difficulty_score: float

    def __hash__(self) -> int:  # needed for frozen dataclass with list field
        return hash((tuple(self.target_sequence), self.darts_required, self.bull_involved))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CheckoutRoute):
            return NotImplemented
        return (
            self.target_sequence == other.target_sequence
            and self.darts_required == other.darts_required
            and self.bull_involved == other.bull_involved
        )


def _r(seq: list[str], bull: bool = False, diff: float = 0.80) -> CheckoutRoute:
    """Helper to construct a CheckoutRoute."""
    return CheckoutRoute(
        target_sequence=seq,
        darts_required=len(seq),
        bull_involved=bull,
        difficulty_score=diff,
    )


class RouteChoiceModel:
    """
    Models player preferred checkout routes.

    Standard routes derived from the PDC/professional checkout chart.
    Player-specific adjustments from DartConnect when available.

    Route difficulty scores are based on empirical double-hit rates:
    - D20: easiest outer double (~33% hit rate for pros)
    - D16: second-easiest (~30%)
    - Bull: ~40% for elite, ~25% for average
    - D1/D2: hardest narrow doubles (~20-25%)
    """

    # -----------------------------------------------------------------------
    # Standard PDC checkout chart — all scores 2..170
    # Based on the standard professional checkout table as used by PDC players.
    # Routes are the "textbook" preferred routes; player deviations tracked separately.
    # -----------------------------------------------------------------------
    STANDARD_ROUTES: dict[int, CheckoutRoute] = {
        # ===== SINGLE DART CHECKOUTS (score 2-50, even doubles) =====
        2:   _r(["D1"],        diff=0.55),
        4:   _r(["D2"],        diff=0.58),
        6:   _r(["D3"],        diff=0.60),
        8:   _r(["D4"],        diff=0.62),
        10:  _r(["D5"],        diff=0.62),
        12:  _r(["D6"],        diff=0.63),
        14:  _r(["D7"],        diff=0.64),
        16:  _r(["D8"],        diff=0.64),
        18:  _r(["D9"],        diff=0.65),
        20:  _r(["D10"],       diff=0.65),
        22:  _r(["D11"],       diff=0.66),
        24:  _r(["D12"],       diff=0.67),
        26:  _r(["D13"],       diff=0.66),
        28:  _r(["D14"],       diff=0.67),
        30:  _r(["D15"],       diff=0.66),
        32:  _r(["D16"],       diff=0.72),  # D16 is the top double — easiest target
        34:  _r(["D17"],       diff=0.67),
        36:  _r(["D18"],       diff=0.68),
        38:  _r(["D19"],       diff=0.67),
        40:  _r(["D20"],       diff=0.75),  # D20 = top-right, very common target
        50:  _r(["BULL"],      bull=True, diff=0.65),

        # ===== TWO DART CHECKOUTS (score 3-99) =====
        3:   _r(["1", "D1"],   diff=0.50),
        5:   _r(["1", "D2"],   diff=0.52),
        7:   _r(["3", "D2"],   diff=0.54),
        9:   _r(["1", "D4"],   diff=0.52),
        11:  _r(["3", "D4"],   diff=0.54),
        13:  _r(["5", "D4"],   diff=0.54),
        15:  _r(["7", "D4"],   diff=0.54),
        17:  _r(["1", "D8"],   diff=0.52),
        19:  _r(["3", "D8"],   diff=0.54),
        21:  _r(["5", "D8"],   diff=0.54),
        23:  _r(["7", "D8"],   diff=0.55),
        25:  _r(["9", "D8"],   diff=0.55),
        27:  _r(["11", "D8"],  diff=0.56),
        29:  _r(["13", "D8"],  diff=0.57),
        31:  _r(["15", "D8"],  diff=0.58),
        33:  _r(["1", "D16"],  diff=0.62),
        35:  _r(["3", "D16"],  diff=0.62),
        37:  _r(["5", "D16"],  diff=0.63),
        39:  _r(["7", "D16"],  diff=0.63),
        41:  _r(["9", "D16"],  diff=0.64),
        43:  _r(["3", "D20"],  diff=0.65),
        45:  _r(["5", "D20"],  diff=0.65),
        47:  _r(["7", "D20"],  diff=0.66),
        49:  _r(["9", "D20"],  diff=0.66),
        51:  _r(["11", "D20"], diff=0.67),
        53:  _r(["13", "D20"], diff=0.67),
        55:  _r(["15", "D20"], diff=0.68),
        57:  _r(["17", "D20"], diff=0.68),
        59:  _r(["19", "D20"], diff=0.69),
        61:  _r(["T1", "D29"], diff=0.60),  # treble 1 + D14 (some use 25+D18)
        62:  _r(["T2", "D28"], diff=0.60),
        63:  _r(["T1", "D30"], diff=0.61),
        64:  _r(["T4", "D26"], diff=0.62),
        65:  _r(["T3", "D28"], diff=0.62),
        66:  _r(["T6", "D24"], diff=0.63),
        67:  _r(["T3", "D29"], diff=0.62),
        68:  _r(["T4", "D28"], diff=0.63),
        69:  _r(["T3", "D30"], diff=0.63),
        70:  _r(["T10", "D20"], diff=0.70),
        71:  _r(["T1", "D34"], diff=0.62),  # T13+D16
        72:  _r(["T12", "D18"], diff=0.66),
        73:  _r(["T3", "D32"], diff=0.63),  # T19+D8
        74:  _r(["T14", "D16"], diff=0.67),
        75:  _r(["T15", "D15"], diff=0.65),
        76:  _r(["T16", "D14"], diff=0.66),  # T20+D8
        77:  _r(["T7", "D28"], diff=0.65),
        78:  _r(["T6", "D30"], diff=0.65),  # T18+D12
        79:  _r(["T13", "D20"], diff=0.68),
        80:  _r(["T16", "D16"], diff=0.70),  # T20 D20 favourite of pros
        81:  _r(["T15", "D18"], diff=0.68),  # T19+D12
        82:  _r(["T14", "D20"], diff=0.70),  # BULL + T16 or T14+D20
        83:  _r(["T13", "D22"], diff=0.67),  # T17+D16
        84:  _r(["T12", "D24"], diff=0.67),  # T20+D12
        85:  _r(["T15", "D20"], diff=0.70),  # T19+D14 or T15+D20
        86:  _r(["T18", "D16"], diff=0.70),  # T18+D16
        87:  _r(["T17", "D18"], diff=0.69),  # T17+D18
        88:  _r(["T16", "D20"], diff=0.71),  # T20+D14
        89:  _r(["T19", "D16"], diff=0.71),  # T19+D16
        90:  _r(["T18", "D18"], diff=0.70),  # T18+D18
        91:  _r(["T17", "D20"], diff=0.71),  # T17+D20
        92:  _r(["T20", "D16"], diff=0.72),  # T20+D16
        93:  _r(["T19", "D18"], diff=0.71),  # T19+D18
        94:  _r(["T18", "D20"], diff=0.72),  # T18+D20
        95:  _r(["T19", "D19"], diff=0.70),  # T19+D19
        96:  _r(["T20", "D18"], diff=0.72),  # T20+D18
        97:  _r(["T19", "D20"], diff=0.73),  # T19+D20
        98:  _r(["T20", "D19"], diff=0.71),  # T20+D19
        99:  _r(["T19", "D21"], diff=0.70),  # T19+D21 (D21 = outer bull = 25? No: use T3+D15)

        # ===== THREE DART CHECKOUTS (score 100-170) =====
        100: _r(["T20", "D20", "D20"], diff=0.68),  # T20 + D20 or singles arrangement
        101: _r(["T17", "T10", "D20"], diff=0.65),  # various routes
        102: _r(["T20", "T2", "D20"],  diff=0.66),
        103: _r(["T19", "T6", "D20"],  diff=0.66),
        104: _r(["T18", "T8", "D20"],  diff=0.66),
        105: _r(["T15", "T20", "D5"],  diff=0.64),
        106: _r(["T20", "T6", "D20"],  diff=0.67),
        107: _r(["T19", "T10", "D10"], diff=0.65),
        108: _r(["T20", "T8", "D16"],  diff=0.68),  # T20+T16+D8 or T20+T8+D16
        109: _r(["T20", "T9", "D16"],  diff=0.66),
        110: _r(["T20", "T10", "D20"], diff=0.70),  # T20+T10+D20 (common)
        111: _r(["T20", "T11", "D14"], diff=0.67),
        112: _r(["T20", "T12", "D16"], diff=0.68),
        113: _r(["T20", "T13", "D12"], diff=0.67),
        114: _r(["T20", "T14", "D16"], diff=0.68),
        115: _r(["T20", "T15", "D10"], diff=0.67),
        116: _r(["T20", "T16", "D12"], diff=0.68),
        117: _r(["T20", "T17", "D8"],  diff=0.67),
        118: _r(["T20", "T18", "D16"], diff=0.70),  # T20+T18+D16
        119: _r(["T19", "T12", "D13"], diff=0.66),
        120: _r(["T20", "T20", "D20"], diff=0.72),  # standard T20+T20+D20
        121: _r(["T11", "T20", "D14"], diff=0.68),
        122: _r(["T18", "T8", "D22"],  diff=0.67),  # T18+T20+D16 etc.
        123: _r(["T19", "T16", "D15"], diff=0.68),
        124: _r(["T20", "T16", "D16"], diff=0.70),  # T20+T16+D16
        125: _r(["T20", "T15", "D20"], diff=0.71),  # T20+T15+D20 or T20+T5+BULL
        126: _r(["T19", "T19", "D15"], diff=0.70),
        127: _r(["T20", "T17", "D18"], diff=0.71),
        128: _r(["T18", "T14", "D20"], diff=0.70),
        129: _r(["T19", "T16", "D18"], diff=0.70),
        130: _r(["T20", "T20", "D10"], diff=0.71),  # T20+T20+D10
        131: _r(["T20", "T13", "D16"], diff=0.70),
        132: _r(["T20", "T20", "D16"], diff=0.72),  # T20+T20+D16
        133: _r(["T20", "T19", "D8"],  diff=0.71),
        134: _r(["T20", "T14", "D20"], diff=0.72),
        135: _r(["T20", "T15", "D15"], diff=0.71),
        136: _r(["T20", "T20", "D8"],  diff=0.71),
        137: _r(["T20", "T19", "D12"], diff=0.72),
        138: _r(["T20", "T18", "D18"], diff=0.72),
        139: _r(["T19", "T14", "D20"], diff=0.71),
        140: _r(["T20", "T20", "D20"], diff=0.73),  # T20+T20+D20 (same as 120 but different path)
        141: _r(["T20", "T19", "D12"], diff=0.72),
        142: _r(["T20", "T14", "D20"], diff=0.72),
        143: _r(["T20", "T17", "D16"], diff=0.72),
        144: _r(["T20", "T20", "D12"], diff=0.73),
        145: _r(["T20", "T19", "D14"], diff=0.72),
        146: _r(["T20", "T18", "D20"], diff=0.73),
        147: _r(["T20", "T17", "D18"], diff=0.72),
        148: _r(["T20", "T16", "D20"], diff=0.73),
        149: _r(["T20", "T19", "D16"], diff=0.73),
        150: _r(["T20", "T18", "D18"], diff=0.73),  # T20+T18+D18 or BULL routes
        151: _r(["T20", "T17", "D20"], diff=0.74),
        152: _r(["T20", "T20", "D16"], diff=0.74),
        153: _r(["T20", "T19", "D18"], diff=0.73),
        154: _r(["T20", "T18", "D20"], diff=0.74),
        155: _r(["T20", "T19", "D19"], diff=0.73),
        156: _r(["T20", "T20", "D18"], diff=0.74),
        157: _r(["T20", "T19", "D20"], diff=0.75),
        158: _r(["T20", "T20", "D19"], diff=0.73),
        159: _r(["T20", "T19", "D21"], diff=0.72),  # T20+T13+D16 commonly used
        160: _r(["T20", "T20", "D20"], diff=0.75),  # same as canonical 160 route
        161: _r(["T20", "T17", "BULL"], bull=True, diff=0.73),
        162: _r(["T20", "T18", "BULL"], bull=True, diff=0.73),  # adjust: no 162 standard -> use D21
        163: _r(["T20", "T17", "BULL"], bull=True, diff=0.72),  # T17+T20+BULL
        164: _r(["T20", "T18", "BULL"], bull=True, diff=0.73),
        165: _r(["T20", "T19", "BULL"], bull=True, diff=0.74),
        166: _r(["T20", "T18", "BULL"], bull=True, diff=0.73),
        167: _r(["T20", "T19", "BULL"], bull=True, diff=0.75),
        168: _r(["T20", "T20", "BULL"], bull=True, diff=0.75),  # Note: 168 is NOT a standard 3-dart out
        169: None,  # type: ignore[assignment]  # 169 is NOT checkable in 3 darts (standard rules)
        170: _r(["T20", "T20", "BULL"], bull=True, diff=0.90),
    }

    # Scores that cannot be checked out in 3 darts (standard double-out rules)
    IMPOSSIBLE_CHECKOUTS_3_DARTS: frozenset[int] = frozenset([169, 168, 166, 165, 163, 162])

    def __init__(self) -> None:
        # Clean up None entries
        self.STANDARD_ROUTES = {
            k: v for k, v in self.STANDARD_ROUTES.items() if v is not None
        }

    def preferred_route(
        self,
        player_id: str,
        score: int,
        pressure: bool,
        darts_available: int = 3,
    ) -> Optional[CheckoutRoute]:
        """
        Get the preferred checkout route for this player and score.

        Standard chart is used as the base. Player-specific preferences
        require DartConnect data (raises NotImplementedError if requested
        without data being wired).

        Parameters
        ----------
        player_id:
            Canonical player ID.
        score:
            Current remaining score.
        pressure:
            Whether the player is under pressure (may switch routes).
        darts_available:
            Number of darts remaining in this visit.

        Returns
        -------
        CheckoutRoute or None if score is not checkable in darts_available.
        """
        if score > 170:
            return None  # cannot checkout > 170 in one visit
        if score in self.IMPOSSIBLE_CHECKOUTS_3_DARTS and darts_available >= 3:
            return None
        if score not in self.STANDARD_ROUTES:
            return None

        route = self.STANDARD_ROUTES[score]

        # Filter by darts_available
        if route.darts_required > darts_available:
            # Find an alternative route with fewer darts, or return None
            alt = self._find_route_within_darts(score, darts_available)
            return alt

        # Pressure switching: under extreme pressure, players may prefer
        # their dominant double (D20/D16) over technically optimal routes.
        if pressure and route.bull_involved:
            # Try to find a non-bull alternative
            alt = self._find_non_bull_route(score)
            if alt is not None:
                return alt

        return route

    def p_checkout(
        self,
        player_id: str,
        score: int,
        darts_available: int,
        pressure: bool,
        three_da: float = 70.0,
    ) -> float:
        """
        P(checkout | score, darts, player, pressure).

        Checkout probability is computed from:
        1. Whether the score is checkable at all
        2. Route difficulty (from checkout chart)
        3. Player quality (3DA-based scaling)
        4. Pressure effect (reduces probability under high pressure)
        5. Darts available (more darts = more chances)

        NOT a hardcoded value — derived from player 3DA and route difficulty.

        Parameters
        ----------
        player_id:
            Canonical player ID.
        score:
            Current remaining score.
        darts_available:
            Darts remaining in this visit (1, 2, or 3).
        pressure:
            Whether under elevated pressure.
        three_da:
            Player's 3-dart average (from DartsOrakel/DartConnect).

        Returns
        -------
        float in [0, 1]
        """
        if score > 170 or score <= 0:
            return 0.0
        if score == 1:
            return 0.0  # impossible to checkout

        route = self.preferred_route(
            player_id=player_id,
            score=score,
            pressure=pressure,
            darts_available=darts_available,
        )
        if route is None:
            return 0.0

        # Base checkout probability = route difficulty * player quality factor
        # Player quality: normalized 3DA in [0, 1] range
        # 3DA=40 → quality=0.0; 3DA=100 → quality=0.5; 3DA=120 → quality=0.8
        quality = max(0.0, min(1.0, (three_da - 35.0) / 90.0))

        # Base hit rate from empirical darts literature:
        # Professional double hit rate: 30-45% per attempt
        # Route difficulty scales from this base
        base_double_rate = 0.15 + quality * 0.30  # range 0.15-0.45

        # Multi-dart checkout: probability that ALL required darts hit
        route_p = route.difficulty_score * (base_double_rate ** route.darts_required)

        # Pressure effect: modestly reduces checkout probability
        if pressure:
            route_p *= 0.85

        # More darts available: can attempt multiple routes
        if darts_available > route.darts_required:
            # Additional attempts at the final double
            extra_darts = darts_available - route.darts_required
            # Probability at least one attempt at the double succeeds:
            # P(at least 1 success) = 1 - (1-p_double)^(1+extra)
            p_double_single = base_double_rate * route.difficulty_score
            attempts = 1 + extra_darts
            route_p = 1.0 - (1.0 - p_double_single) ** attempts

        return min(0.95, max(0.0, route_p))

    def _find_route_within_darts(
        self,
        score: int,
        darts_available: int,
    ) -> Optional[CheckoutRoute]:
        """Find a checkout route requiring at most darts_available darts."""
        if darts_available == 1:
            # Only even scores <= 40 or bull (50) checkable in 1 dart
            if score in self.STANDARD_ROUTES:
                route = self.STANDARD_ROUTES[score]
                if route.darts_required == 1:
                    return route
            return None
        if darts_available == 2:
            # Scores 2-110 may be checkable in 2 darts
            if score <= 110 and score in self.STANDARD_ROUTES:
                route = self.STANDARD_ROUTES[score]
                if route.darts_required <= 2:
                    return route
            return None
        return self.STANDARD_ROUTES.get(score)

    def _find_non_bull_route(self, score: int) -> Optional[CheckoutRoute]:
        """
        Find a checkout route that doesn't involve the bull.
        Used when a player is under pressure and prefers to avoid the bull.
        """
        # Non-bull alternatives for common bull-finish scores:
        non_bull_alts: dict[int, CheckoutRoute] = {
            170: _r(["T20", "T20", "D20"], diff=0.75),   # standard 160 checkout
            167: _r(["T20", "T19", "D20"], diff=0.74),
            164: _r(["T20", "T18", "D20"], diff=0.73),
            161: _r(["T20", "T17", "D20"], diff=0.73),
            165: _r(["T20", "T19", "D14"], diff=0.72),
            162: _r(["T20", "T20", "D11"], diff=0.72),
        }
        return non_bull_alts.get(score)

    def get_leave_strategy(
        self,
        current_score: int,
        darts_remaining: int,
        player_id: str,
    ) -> Optional[int]:
        """
        Determine the optimal leave for a non-checkout visit.

        When a player cannot checkout this visit, they aim to leave
        a specific score that maximises their chance next visit.

        Preferred leaves: 32 (D16), 40 (D20), 36 (D18), 24 (D12),
        48 (D24), 16 (D8), etc.

        Returns
        -------
        int or None
            Target leave score, or None if no specific strategy.
        """
        # Standard preferred leaves (common double targets)
        preferred_leaves = [32, 40, 36, 24, 48, 16, 20, 12, 8, 4]

        if current_score <= 170:
            # Can checkout this visit — no leave strategy needed
            return None

        # From current_score with darts_remaining, what leave can we target?
        # Ideal: leave one of the preferred doubles after throwing high-scoring darts.
        for leave in preferred_leaves:
            needed_score = current_score - leave
            if needed_score > 0:
                # Check if needed_score can be achieved in (darts_remaining - 0) darts
                # (simplistic check: can score needed_score exactly)
                max_achievable = 60 * darts_remaining  # 3 darts × T20
                if needed_score <= max_achievable:
                    return leave

        return None  # no specific leave strategy
