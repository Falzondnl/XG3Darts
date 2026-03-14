"""
Score state representation for the 501 Markov chain.

States: 0 (busted or finished) to 501.
Transitions depend on player visit distribution.
State 0 = leg finished (terminal win state).
State 1 = impossible to checkout (bust trap — no double equals 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DartsEngineError(Exception):
    """Base exception for all darts engine errors."""


class ThrowContext(Enum):
    OPEN = "open"          # score > 300: maximising T20/T19
    MIDDLE = "middle"      # 171-300: still scoring, can't finish in 1 visit
    SETUP = "setup"        # 100-170: route planning, may leave a double
    FINISH = "finish"      # 41-99: direct checkout attempt
    PRESSURE = "pressure"  # 2-40: high bust risk


# Maximum checkout score reachable in N darts (double-out rules).
# 1 dart: D25 (bull) = 50, D20 = 40 ... but T20+T20+D20 = 170 in 3 darts.
_MAX_CHECKOUT_1_DART: int = 50   # D25 (bull)
_MAX_CHECKOUT_2_DARTS: int = 110  # T20 + D25
_MAX_CHECKOUT_3_DARTS: int = 170  # T20 + T20 + D20 (or T20 + T20 + BULL for 170)


@dataclass(frozen=True)
class ScoreState:
    """
    Immutable score state in the 501 Markov chain.

    score=0   → leg finished (terminal win)
    score=1   → unreachable checkout; no double = 1 exists (terminal lose-path)
    score>1   → active state
    """

    score: int  # 0..501

    def __post_init__(self) -> None:
        if not (0 <= self.score <= 501):
            raise DartsEngineError(f"Invalid score state: {self.score}")

    @property
    def context(self) -> ThrowContext:
        """Classify this state into one of five throw-context bands."""
        s = self.score
        if s > 300:
            return ThrowContext.OPEN
        if s > 170:
            return ThrowContext.MIDDLE
        if s > 99:
            return ThrowContext.SETUP
        if s > 40:
            return ThrowContext.FINISH
        # 0 and 1 are terminal; for non-terminal scores 2..40 → PRESSURE
        return ThrowContext.PRESSURE

    @property
    def is_terminal(self) -> bool:
        """Score = 0 (won) or 1 (can't checkout — no double = 1)."""
        return self.score == 0 or self.score == 1

    def apply_visit(self, visit_score: int, bust: bool) -> ScoreState:
        """
        Apply a visit result and return the resulting state.

        If bust=True the state is unchanged (player stays at current score).
        Otherwise the new score is current - visit_score.

        Raises
        ------
        DartsEngineError
            If visit_score would produce a negative score (caller should have
            marked this as a bust).
        """
        if bust:
            return self
        new_score = self.score - visit_score
        if new_score < 0:
            raise DartsEngineError(
                f"Negative score result: {self.score} - {visit_score} = {new_score}. "
                "Caller should have flagged this as a bust."
            )
        return ScoreState(score=new_score)

    def can_checkout(self, darts: int = 3) -> bool:
        """
        True if this score is reachable as a checkout within darts_available darts.

        Rules (standard double-out):
        - Cannot checkout 1 (no double equals 1).
        - 1 dart:  max 50  (D25 / bull)
        - 2 darts: max 110 (T20 + D25)
        - 3 darts: max 170 (T20 + T20 + D20)
        Score 0 is already finished; returns True trivially.
        """
        if self.score == 0:
            return True
        if self.score == 1:
            return False
        if darts <= 0:
            return False
        if darts >= 3:
            return self.score <= _MAX_CHECKOUT_3_DARTS
        if darts == 2:
            return self.score <= _MAX_CHECKOUT_2_DARTS
        # darts == 1
        return self.score <= _MAX_CHECKOUT_1_DART and self.score % 2 == 0  # must be even (double)

    def remaining_darts_lower_bound(self) -> int:
        """
        Theoretical minimum darts to finish from this score.

        Used by the Markov chain to detect impossible paths.
        Returns 0 if already finished.
        """
        if self.score == 0:
            return 0
        if self.score == 1:
            return -1  # impossible
        # Each dart can score at most 60 (T20), final dart must be a double (max D25 = 50).
        # With 3 darts per visit: floor((score - 50) / 60) + 1 visits ≈ lower bound.
        # Minimum visits ceil((score) / (60 + 60 + 50)) = ceil(score / 170)
        import math
        return math.ceil(self.score / 170) * 3 - (2 if self.score <= 170 else 0)

    def __repr__(self) -> str:  # noqa: D105
        ctx = "terminal" if self.is_terminal else self.context.value
        return f"ScoreState(score={self.score}, context={ctx})"
