"""
Draw result handling for darts matches.

Provides the canonical result types, score extraction, and ELO scoring
coefficients for each possible match outcome.  Only competitions whose
:class:`~competition.format_registry.DartsCompetitionFormat` includes
``"draw"`` in ``result_types`` will ever produce a drawn result; this
module enforces that constraint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from competition.format_registry import (
    DartsCompetitionFormat,
    DartsFormatError,
    get_format,
)


class DartsResultError(Exception):
    """Raised when a result is logically inconsistent with the match format."""


# Valid result type literals
RESULT_P1_WIN = "p1_win"
RESULT_P2_WIN = "p2_win"
RESULT_DRAW = "draw"

_VALID_RESULT_TYPES = frozenset({RESULT_P1_WIN, RESULT_P2_WIN, RESULT_DRAW})


@dataclass
class DartsMatchResult:
    """
    Canonical representation of a completed darts match result.

    Attributes
    ----------
    result_type:
        One of ``"p1_win"``, ``"p2_win"``, or ``"draw"``.
    p1_score:
        Sets or legs won by player/team 1.
    p2_score:
        Sets or legs won by player/team 2.
    format_code:
        The :attr:`~competition.format_registry.DartsCompetitionFormat.code`
        for the competition this result belongs to.
    round_name:
        The round within the competition (e.g. ``"Final"``).
    """

    result_type: str
    p1_score: int
    p2_score: int
    format_code: str
    round_name: str

    def __post_init__(self) -> None:
        if self.result_type not in _VALID_RESULT_TYPES:
            raise DartsResultError(
                f"result_type must be one of {sorted(_VALID_RESULT_TYPES)}, "
                f"got {self.result_type!r}"
            )
        if self.p1_score < 0 or self.p2_score < 0:
            raise DartsResultError(
                f"Scores must be non-negative, got ({self.p1_score}, {self.p2_score})"
            )
        # Validate draw consistency
        if self.result_type == RESULT_DRAW and self.p1_score != self.p2_score:
            raise DartsResultError(
                f"Draw declared but scores differ: {self.p1_score} vs {self.p2_score}"
            )
        # Validate win consistency
        if self.result_type == RESULT_P1_WIN and self.p1_score <= self.p2_score:
            raise DartsResultError(
                f"p1_win declared but p1_score ({self.p1_score}) "
                f"≤ p2_score ({self.p2_score})"
            )
        if self.result_type == RESULT_P2_WIN and self.p2_score <= self.p1_score:
            raise DartsResultError(
                f"p2_win declared but p2_score ({self.p2_score}) "
                f"≤ p1_score ({self.p1_score})"
            )
        # Validate the format permits this result type
        fmt = get_format(self.format_code)
        if self.result_type not in fmt.result_types:
            raise DartsResultError(
                f"Result type {self.result_type!r} is not valid for "
                f"{self.format_code} ({list(fmt.result_types)})"
            )
        # Validate the round exists
        fmt.get_round(self.round_name)

    @property
    def winner_index(self) -> Optional[int]:
        """
        Return ``1`` if player 1 won, ``2`` if player 2 won, ``None`` if draw.
        """
        if self.result_type == RESULT_P1_WIN:
            return 1
        if self.result_type == RESULT_P2_WIN:
            return 2
        return None

    @property
    def is_draw(self) -> bool:
        """True when the match ended in a draw."""
        return self.result_type == RESULT_DRAW


def elo_score_for_result(result_type: str) -> tuple[float, float]:
    """
    Return (score_p1, score_p2) for ELO calculation purposes.

    Standard ELO uses:
    - Win  → 1.0 for the winner, 0.0 for the loser
    - Draw → 0.5 for both players

    Parameters
    ----------
    result_type:
        One of ``"p1_win"``, ``"p2_win"``, or ``"draw"``.

    Returns
    -------
    tuple[float, float]
        ``(score_p1, score_p2)`` where each value is in ``{0.0, 0.5, 1.0}``.

    Raises
    ------
    DartsResultError
        If ``result_type`` is not a recognised value.
    """
    if result_type == RESULT_P1_WIN:
        return (1.0, 0.0)
    if result_type == RESULT_P2_WIN:
        return (0.0, 1.0)
    if result_type == RESULT_DRAW:
        return (0.5, 0.5)
    raise DartsResultError(
        f"Unknown result_type {result_type!r}. "
        f"Valid values: {sorted(_VALID_RESULT_TYPES)}"
    )


def classify_result_from_scores(
    p1_score: int,
    p2_score: int,
    format_code: str,
    round_name: str,
) -> DartsMatchResult:
    """
    Determine the result type from raw scores and format constraints.

    For formats that do not allow draws, equal scores raise an error (they
    indicate incomplete or corrupt data).  For draw-enabled formats, equal
    scores produce a ``"draw"`` result.

    Parameters
    ----------
    p1_score:
        Sets or legs won by player 1.
    p2_score:
        Sets or legs won by player 2.
    format_code:
        Competition format code.
    round_name:
        Round name within the competition.

    Returns
    -------
    DartsMatchResult

    Raises
    ------
    DartsResultError
        If equal scores occur in a non-draw format, or if scores are negative.
    DartsFormatError
        If the format or round name is unknown.
    """
    if p1_score < 0 or p2_score < 0:
        raise DartsResultError(
            f"Scores must be non-negative, got ({p1_score}, {p2_score})"
        )
    fmt = get_format(format_code)

    if p1_score > p2_score:
        result_type = RESULT_P1_WIN
    elif p2_score > p1_score:
        result_type = RESULT_P2_WIN
    else:
        # Equal scores
        if not fmt.allows_draw():
            raise DartsResultError(
                f"Scores are equal ({p1_score}-{p2_score}) but format "
                f"{format_code} does not allow draws. Data may be corrupt."
            )
        result_type = RESULT_DRAW

    return DartsMatchResult(
        result_type=result_type,
        p1_score=p1_score,
        p2_score=p2_score,
        format_code=format_code,
        round_name=round_name,
    )


def is_draw_possible(format_code: str) -> bool:
    """
    Return True if the competition format permits drawn results.

    Parameters
    ----------
    format_code:
        The competition format code to check.
    """
    return get_format(format_code).allows_draw()
