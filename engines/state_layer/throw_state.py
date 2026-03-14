"""
Throw state: tracks dart-in-hand sub-state (1/2/3 darts within a visit)
and pressure detection.

A "visit" in darts = up to 3 darts thrown at the board.
The throw state is the micro-state *within* a visit:
  dart_number ∈ {1, 2, 3}

Pressure is determined by a combination of:
  - current score (low score = high bust risk)
  - match situation (behind in legs, must-win leg)
  - opponent's score (if opponent is close to finishing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog

from engines.state_layer.score_state import ScoreState, ThrowContext

logger = structlog.get_logger(__name__)


class PressureLevel(Enum):
    """
    Classified pressure level for the current throw.

    Values represent relative intensity; used to modulate
    visit-distribution band selection and checkout model parameters.
    """

    NONE = "none"        # routine throw, no elevated pressure
    LOW = "low"          # slight awareness (e.g. opponent on double)
    MEDIUM = "medium"    # meaningful pressure (behind in legs/sets, checkout range)
    HIGH = "high"        # high pressure (must-win leg, final dart to win match)
    CRITICAL = "critical"  # last dart to win match in deciding leg


@dataclass(frozen=True)
class DartThrowState:
    """
    Sub-visit dart throw state.

    Attributes
    ----------
    dart_number:
        Which dart within the current visit (1, 2, or 3).
    score_before_visit:
        Score at the start of this visit (before any darts in this visit).
    darts_scored_this_visit:
        Running total scored in this visit so far (from dart 1..dart_number-1).
    player_index:
        0 = player 1, 1 = player 2.
    """

    dart_number: int          # 1, 2, or 3
    score_before_visit: int   # score at visit start
    darts_scored_this_visit: int  # accumulated score from earlier darts in visit
    player_index: int         # 0 or 1

    def __post_init__(self) -> None:
        if self.dart_number not in (1, 2, 3):
            raise ValueError(f"dart_number must be 1, 2 or 3; got {self.dart_number}")
        if self.score_before_visit < 0 or self.score_before_visit > 501:
            raise ValueError(f"score_before_visit out of range: {self.score_before_visit}")
        if self.darts_scored_this_visit < 0:
            raise ValueError(f"darts_scored_this_visit cannot be negative: {self.darts_scored_this_visit}")
        if self.player_index not in (0, 1):
            raise ValueError(f"player_index must be 0 or 1; got {self.player_index}")

    @property
    def current_score(self) -> int:
        """Score remaining after darts thrown so far in this visit."""
        return self.score_before_visit - self.darts_scored_this_visit

    @property
    def darts_remaining_in_visit(self) -> int:
        """How many more darts remain in this visit (including current)."""
        return 4 - self.dart_number  # dart 1→3 remaining, dart 2→2, dart 3→1

    @property
    def score_state(self) -> ScoreState:
        """Current ScoreState after darts thrown so far in this visit."""
        return ScoreState(score=self.current_score)

    def can_finish_this_visit(self) -> bool:
        """True if checkout is possible within remaining darts of this visit."""
        return self.score_state.can_checkout(darts=self.darts_remaining_in_visit)

    def advance(self, dart_score: int, bust: bool) -> "DartThrowState":
        """
        Produce the next DartThrowState after throwing one dart.

        Parameters
        ----------
        dart_score:
            Score from the dart just thrown (0–60, or 25/50 for bull).
        bust:
            Whether this throw caused a bust.

        Returns
        -------
        DartThrowState
            Next state (dart_number incremented, or visit reset if bust).
        """
        if bust:
            # Visit ends immediately on a bust; dart_number advances to signal end.
            return DartThrowState(
                dart_number=min(self.dart_number + 1, 3),
                score_before_visit=self.score_before_visit,
                darts_scored_this_visit=self.darts_scored_this_visit,  # unchanged
                player_index=self.player_index,
            )
        if dart_score < 0:
            raise ValueError(f"dart_score cannot be negative: {dart_score}")
        return DartThrowState(
            dart_number=min(self.dart_number + 1, 3),
            score_before_visit=self.score_before_visit,
            darts_scored_this_visit=self.darts_scored_this_visit + dart_score,
            player_index=self.player_index,
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"DartThrowState(dart={self.dart_number}/3, "
            f"score={self.current_score}, player={self.player_index})"
        )


@dataclass
class PressureDetector:
    """
    Detects and classifies pressure level for the current throw.

    Pressure is a composite signal from:
      1. Score band (PRESSURE context = high bust risk)
      2. Match situation (must-win leg)
      3. Opponent proximity (opponent can finish next visit)
    """

    def compute_pressure(
        self,
        throw_state: DartThrowState,
        legs_needed_by_thrower: int,
        legs_needed_by_opponent: int,
        opponent_current_score: int,
        is_deciding_leg: bool,
    ) -> PressureLevel:
        """
        Compute the pressure level for the current dart throw.

        Parameters
        ----------
        throw_state:
            Current micro-state (score, dart number, etc.).
        legs_needed_by_thrower:
            Legs the current thrower still needs to win the match.
        legs_needed_by_opponent:
            Legs the opponent still needs to win the match.
        opponent_current_score:
            Opponent's current score this leg.
        is_deciding_leg:
            True if this is the final possible leg (e.g. tied on max-1 legs).

        Returns
        -------
        PressureLevel
        """
        score = throw_state.current_score
        darts_left = throw_state.darts_remaining_in_visit
        score_ctx = throw_state.score_state.context

        # CRITICAL: last dart to win the match
        if (
            is_deciding_leg
            and legs_needed_by_thrower == 1
            and throw_state.dart_number == 3
            and ScoreState(score=score).can_checkout(darts=1)
        ):
            return PressureLevel.CRITICAL

        # HIGH: on checkout, must-win leg, deciding moment
        if is_deciding_leg and score_ctx in (ThrowContext.PRESSURE, ThrowContext.FINISH):
            return PressureLevel.HIGH

        # HIGH: opponent can checkout this visit (score <= 170)
        if opponent_current_score <= 170 and legs_needed_by_opponent == 1:
            return PressureLevel.HIGH

        # MEDIUM: in checkout range ourselves
        if score_ctx in (ThrowContext.FINISH, ThrowContext.PRESSURE):
            return PressureLevel.MEDIUM

        # MEDIUM: opponent could finish in 2 visits
        if opponent_current_score <= 340 and legs_needed_by_opponent == 1:
            return PressureLevel.MEDIUM

        # LOW: behind in legs/sets in a close match
        if legs_needed_by_thrower > legs_needed_by_opponent and is_deciding_leg:
            return PressureLevel.LOW

        return PressureLevel.NONE

    def is_high_bust_risk(
        self,
        current_score: int,
        darts_remaining: int,
    ) -> bool:
        """
        True when the player has elevated bust probability.

        Heuristic: score <= 50 with 2+ darts left, or score <= 32 with any darts.
        The exact bust probability is computed by the checkout model; this method
        is a fast boolean gate for routing logic.
        """
        if current_score <= 1:
            return True  # terminal or impossible state
        if current_score <= 32:
            return True  # all doubles territory — high bust risk
        if current_score <= 50 and darts_remaining >= 2:
            return True  # single dart from finishing but many darts left
        return False


@dataclass
class VisitResult:
    """
    Complete result of a 3-dart visit.

    Attributes
    ----------
    score_achieved:
        Total score reduction for this visit (0 on bust).
    is_bust:
        Whether the visit ended in a bust.
    darts_used:
        Number of darts actually thrown (1–3; can be < 3 on checkout).
    checkout_achieved:
        Whether the player checked out this visit (score_before - score_achieved == 0).
    score_after:
        Score remaining after this visit.
    """

    score_achieved: int
    is_bust: bool
    darts_used: int
    checkout_achieved: bool
    score_after: int

    def __post_init__(self) -> None:
        if self.is_bust and self.score_achieved != 0:
            raise ValueError("Bust visits must have score_achieved=0")
        if self.checkout_achieved and self.score_after != 0:
            raise ValueError("Checkout visits must have score_after=0")
        if self.darts_used not in (1, 2, 3):
            raise ValueError(f"darts_used must be 1, 2 or 3; got {self.darts_used}")
