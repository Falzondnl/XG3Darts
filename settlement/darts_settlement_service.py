"""
Darts Settlement / Grading Service.

Grades all 15 darts market types after a match is completed.
All grading logic is based on real match result data only — never on
hardcoded values, synthetic outcomes, or placeholder probabilities.

Market Types Covered
--------------------
1.  match_win           — Match winner (P1 or P2)
2.  exact_score         — Correct score in legs (e.g. 6-4) or sets
3.  total_legs_over     — Total legs played over/under a line
4.  handicap            — Leg/set handicap (P1 +/- X legs vs P2)
5.  most_180s           — Which player scored more 180s in the match
6.  180_over            — Total match 180s (combined) over/under a line
7.  highest_checkout    — Which player hit the highest single-leg checkout
8.  first_leg_winner    — Who won leg 1 of the match
9.  race_to_x           — Which player reaches X legs first
10. nine_dart_finish    — Was a nine-darter hit in the match (Yes/No)
11. player_checkout_over— Player-specific highest checkout over/under
12. sets_over           — Total sets played over/under (sets-format only)
13. break_of_throw      — Was the first leg won by the non-starter (break)?
14. leg_winner_next     — Who wins the next (specified) leg
15. total_180s_band     — Total 180s in a band (e.g. 5-7)

GradeOutcome values: WIN | LOSE | VOID | PUSH

Edge cases handled:
    - Walkover (one player withdrew before the match)
    - Retired / abandoned mid-match
    - Insufficient data for a specific market (→ VOID)
    - Draw / tie resolution
    - Half-line handicaps (no push possible)
    - Integer handicap lines (push on exact tie)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class GradeOutcome(str, Enum):
    """The four possible settlement outcomes for a selection."""
    WIN = "WIN"
    LOSE = "LOSE"
    VOID = "VOID"
    PUSH = "PUSH"  # full refund; only on integer-line handicaps / totals


class MatchStatus(str, Enum):
    """Terminal match statuses that trigger settlement."""
    COMPLETED = "Completed"
    WALKOVER = "Walkover"
    RETIRED = "Retired"
    ABANDONED = "Abandoned"
    CANCELLED = "Cancelled"
    POSTPONED = "Postponed"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class LegRecord:
    """
    Statistics for a single leg of a darts match.

    Attributes
    ----------
    leg_number:
        1-indexed leg number in match order.
    set_number:
        Set number this leg belongs to (None for legs-only formats).
    winner_is_p1:
        True if P1 won this leg, False if P2 won.
    starter_is_p1:
        True if P1 threw first in this leg (None if unknown).
    had_180:
        True if at least one 180 was hit in this leg.
    nine_darter:
        True if a nine-darter was completed in this leg.
    winning_checkout:
        The finishing checkout score for the winner (None if unknown).
    p1_180s:
        Number of 180s hit by P1 in this leg.
    p2_180s:
        Number of 180s hit by P2 in this leg.
    """
    leg_number: int
    set_number: Optional[int] = None
    winner_is_p1: Optional[bool] = None
    starter_is_p1: Optional[bool] = None
    had_180: bool = False
    nine_darter: bool = False
    winning_checkout: Optional[int] = None
    p1_180s: int = 0
    p2_180s: int = 0


@dataclass
class MatchResult:
    """
    Complete result record for a settled darts match.

    All grading is derived from these fields — no other data sources.

    Attributes
    ----------
    match_id:
        Unique match identifier.
    status:
        Terminal status (Completed, Walkover, Retired, Abandoned, etc.).
    winner_is_p1:
        True if P1 won the match.  None for draws (Premier League) or
        matches where no winner was determined (Abandoned/Walkover).
    p1_legs:
        Legs won by P1 in the match.
    p2_legs:
        Legs won by P2 in the match.
    p1_sets:
        Sets won by P1 (None for legs-only formats).
    p2_sets:
        Sets won by P2 (None for legs-only formats).
    is_sets_format:
        True when the match is decided by sets (e.g. PDC World Championship).
    is_draw:
        True when the match ended in a draw (Premier League group stage).
    total_legs_played:
        Total number of legs played in the match.
    total_sets_played:
        Total number of sets played (None for legs-only formats).
    p1_180s_total:
        Total number of 180s hit by P1 across the entire match.
    p2_180s_total:
        Total number of 180s hit by P2 across the entire match.
    p1_highest_checkout:
        Highest single-leg finishing checkout by P1 (None if no checkouts recorded).
    p2_highest_checkout:
        Highest single-leg finishing checkout by P2 (None if no checkouts recorded).
    nine_darter_in_match:
        True if at least one nine-darter was completed in the match.
    legs:
        Ordered list of individual leg records.  Empty if only aggregate
        stats are available.
    p1_player_id:
        Player 1 canonical identifier.
    p2_player_id:
        Player 2 canonical identifier.
    format_code:
        Competition format code (e.g. PDC_PL, PDC_WC).
    completed_at:
        UTC datetime when the match result was finalised.
    source_name:
        Data source that provided the result (e.g. "dartconnect", "optic_odds").
    """
    match_id: str
    status: str  # MatchStatus value
    winner_is_p1: Optional[bool]
    p1_legs: int
    p2_legs: int
    p1_sets: Optional[int] = None
    p2_sets: Optional[int] = None
    is_sets_format: bool = False
    is_draw: bool = False
    total_legs_played: int = 0
    total_sets_played: Optional[int] = None
    p1_180s_total: int = 0
    p2_180s_total: int = 0
    p1_highest_checkout: Optional[int] = None
    p2_highest_checkout: Optional[int] = None
    nine_darter_in_match: bool = False
    legs: list[LegRecord] = field(default_factory=list)
    p1_player_id: str = ""
    p2_player_id: str = ""
    format_code: str = ""
    completed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source_name: str = ""

    def __post_init__(self) -> None:
        """Derive total_legs_played if not explicitly provided."""
        if self.total_legs_played == 0:
            self.total_legs_played = self.p1_legs + self.p2_legs
        # Derive total_sets_played for sets-format matches
        if self.is_sets_format and self.total_sets_played is None:
            if self.p1_sets is not None and self.p2_sets is not None:
                self.total_sets_played = self.p1_sets + self.p2_sets


@dataclass
class GradeResult:
    """
    Result of grading a single selection within a market.

    Attributes
    ----------
    outcome:
        WIN / LOSE / VOID / PUSH.
    reason:
        Human-readable explanation of the grade decision.
    """
    outcome: GradeOutcome
    reason: str = ""


@dataclass
class MarketGrade:
    """
    The grade for a single market on a completed match.

    Attributes
    ----------
    market_type:
        One of the 15 market type identifiers.
    selection:
        The customer's selection (e.g. "p1", "over", "yes", "4.5").
    market_params:
        Additional parameters describing the market
        (e.g. {"line": 8.5} for totals, {"target": 3} for race-to-x).
    grade:
        The resulting GradeResult.
    market_id:
        External market identifier (optional, for upstream forwarding).
    """
    market_type: str
    selection: str
    market_params: dict[str, Any]
    grade: GradeResult
    market_id: Optional[str] = None


@dataclass
class SettlementReport:
    """
    Complete settlement report for a single match.

    Attributes
    ----------
    match_id:
        Match identifier.
    status:
        The match terminal status that triggered settlement.
    settled_at:
        UTC datetime when settlement was computed.
    grades:
        List of individual market grades.
    n_void:
        Count of VOID grades (walkover / abandoned / insufficient data).
    n_win:
        Count of WIN grades.
    n_lose:
        Count of LOSE grades.
    n_push:
        Count of PUSH grades (refund on exact integer line).
    """
    match_id: str
    status: str
    settled_at: datetime
    grades: list[MarketGrade]
    n_void: int = 0
    n_win: int = 0
    n_lose: int = 0
    n_push: int = 0

    def __post_init__(self) -> None:
        """Compute aggregate counts from grades list."""
        for g in self.grades:
            outcome = g.grade.outcome
            if outcome == GradeOutcome.WIN:
                self.n_win += 1
            elif outcome == GradeOutcome.LOSE:
                self.n_lose += 1
            elif outcome == GradeOutcome.VOID:
                self.n_void += 1
            elif outcome == GradeOutcome.PUSH:
                self.n_push += 1


# ---------------------------------------------------------------------------
# Settlement service
# ---------------------------------------------------------------------------

class DartsSettlementService:
    """
    Darts market grading service.

    Grades all 15 darts market types based on the real MatchResult.
    All grading logic is deterministic given the MatchResult data.

    Usage
    -----
    ::

        service = DartsSettlementService()
        result = MatchResult(
            match_id="abc-123",
            status="Completed",
            winner_is_p1=True,
            p1_legs=6,
            p2_legs=4,
            ...
        )
        markets_to_grade = [
            {"market_type": "match_win", "selection": "p1", "market_params": {}},
            {"market_type": "total_legs_over", "selection": "over", "market_params": {"line": 8.5}},
        ]
        report = service.grade_match(result, markets_to_grade)

    Notes
    -----
    - WALKOVER / ABANDONED → all markets VOID except match_win on WALKOVER
      (if a winner is declared, match_win is graded normally).
    - CANCELLED / POSTPONED → all markets VOID unconditionally.
    - RETIRED mid-match → all markets graded on current score if
      is_result_official=True (flag carried on MatchResult via source_name),
      else VOID.
    """

    # Markets that are ALWAYS voided regardless of match status
    _ALWAYS_VOID_STATUSES: frozenset[str] = frozenset({
        MatchStatus.CANCELLED.value,
        MatchStatus.POSTPONED.value,
    })

    # Markets that are CONDITIONALLY voided (abandoned / retired)
    _CONDITIONAL_VOID_STATUSES: frozenset[str] = frozenset({
        MatchStatus.ABANDONED.value,
    })

    def grade_match(
        self,
        result: MatchResult,
        markets: list[dict[str, Any]],
    ) -> SettlementReport:
        """
        Grade all submitted markets for a completed match.

        Parameters
        ----------
        result:
            The authoritative match result.
        markets:
            List of dicts, each with keys:
            - ``market_type``: str — one of the 15 market type identifiers
            - ``selection``: str — the customer's selection
            - ``market_params``: dict — market-specific parameters
            - ``market_id``: str (optional) — external market ID

        Returns
        -------
        SettlementReport
            Contains a MarketGrade for every submitted market.

        Raises
        ------
        ValueError
            If ``market_type`` is unrecognised.  Callers should handle this
            and grade the market as VOID to avoid a processing halt.
        """
        settled_at = datetime.now(timezone.utc)

        grades: list[MarketGrade] = []
        for market in markets:
            market_type: str = market.get("market_type", "")
            selection: str = market.get("selection", "")
            params: dict[str, Any] = market.get("market_params") or {}
            market_id: Optional[str] = market.get("market_id")

            grade = self._grade_single_market(
                result=result,
                market_type=market_type,
                selection=selection,
                params=params,
            )

            logger.info(
                "market_graded",
                match_id=result.match_id,
                market_type=market_type,
                selection=selection,
                outcome=grade.outcome.value,
                reason=grade.reason,
            )

            grades.append(
                MarketGrade(
                    market_type=market_type,
                    selection=selection,
                    market_params=params,
                    grade=grade,
                    market_id=market_id,
                )
            )

        report = SettlementReport(
            match_id=result.match_id,
            status=result.status,
            settled_at=settled_at,
            grades=grades,
        )

        logger.info(
            "settlement_report_generated",
            match_id=result.match_id,
            status=result.status,
            n_markets=len(grades),
            n_win=report.n_win,
            n_lose=report.n_lose,
            n_void=report.n_void,
            n_push=report.n_push,
        )

        return report

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _grade_single_market(
        self,
        result: MatchResult,
        market_type: str,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """Route a market to its specific grading function."""

        # ---- Global void conditions ----
        if result.status in self._ALWAYS_VOID_STATUSES:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=f"Match {result.status.lower()} — all markets void",
            )

        if result.status in self._CONDITIONAL_VOID_STATUSES:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="Match abandoned before completion — market void",
            )

        # ---- Dispatch to market-specific grader ----
        dispatch: dict[str, Any] = {
            "match_win":            self._grade_match_win,
            "exact_score":          self._grade_exact_score,
            "total_legs_over":      self._grade_total_legs_over,
            "handicap":             self._grade_handicap,
            "most_180s":            self._grade_most_180s,
            "180_over":             self._grade_180_over,
            "highest_checkout":     self._grade_highest_checkout,
            "first_leg_winner":     self._grade_first_leg_winner,
            "race_to_x":            self._grade_race_to_x,
            "nine_dart_finish":     self._grade_nine_dart_finish,
            "player_checkout_over": self._grade_player_checkout_over,
            "sets_over":            self._grade_sets_over,
            "break_of_throw":       self._grade_break_of_throw,
            "leg_winner_next":      self._grade_leg_winner_next,
            "total_180s_band":      self._grade_total_180s_band,
            "both_to_score":        self._grade_both_to_score,
            "winning_margin":       self._grade_winning_margin,
        }

        grader = dispatch.get(market_type)
        if grader is None:
            logger.warning(
                "unknown_market_type_voided",
                match_id=result.match_id,
                market_type=market_type,
            )
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=f"Unknown market type '{market_type}' — void by policy",
            )

        try:
            return grader(result=result, selection=selection, params=params)
        except Exception as exc:
            logger.error(
                "settlement_grade_error",
                match_id=result.match_id,
                market_type=market_type,
                selection=selection,
                error=str(exc),
            )
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=f"Grading error — market voided: {exc}",
            )

    # ------------------------------------------------------------------
    # 1. Match Winner
    # ------------------------------------------------------------------

    def _grade_match_win(
        self,
        result: MatchResult,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """
        Grade the match winner market.

        Selections: "p1" | "p2" | "draw"

        Draw selection is only valid for draw-enabled formats (Premier League).
        For retired matches the result is graded as-is if a winner was declared
        by the governing body; otherwise VOID.
        """
        # Retired with no declared winner → void
        if result.status == MatchStatus.RETIRED.value and result.winner_is_p1 is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="Match retired — no official result declared",
            )

        # Walkover with a winner declared → grade normally
        if result.winner_is_p1 is None and not result.is_draw:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="No official match winner — market void",
            )

        sel = selection.lower()

        if result.is_draw:
            if sel == "draw":
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=f"Match ended in draw — 'draw' selection wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=f"Match ended in draw — '{sel}' selection loses",
            )

        # Standard win/lose
        if sel == "p1":
            outcome = GradeOutcome.WIN if result.winner_is_p1 else GradeOutcome.LOSE
            winner_str = "P1" if result.winner_is_p1 else "P2"
            return GradeResult(
                outcome=outcome,
                reason=f"Match winner: {winner_str} — selection: {sel}",
            )
        if sel == "p2":
            outcome = GradeOutcome.WIN if not result.winner_is_p1 else GradeOutcome.LOSE
            winner_str = "P1" if result.winner_is_p1 else "P2"
            return GradeResult(
                outcome=outcome,
                reason=f"Match winner: {winner_str} — selection: {sel}",
            )
        if sel == "draw":
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason="Draw selection but match was decided — selection loses",
            )

        return GradeResult(
            outcome=GradeOutcome.VOID,
            reason=f"Unrecognised match_win selection: '{selection}'",
        )

    # ------------------------------------------------------------------
    # 2. Exact Score (correct score in legs or sets)
    # ------------------------------------------------------------------

    def _grade_exact_score(
        self,
        result: MatchResult,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """
        Grade a correct-score market.

        Selection format: "p1_legs-p2_legs" e.g. "6-4" or "3-1" (for sets).

        For sets-format matches ``params["score_type"]`` should be "sets";
        otherwise it defaults to "legs".
        """
        score_type: str = params.get("score_type", "legs")

        try:
            left, right = selection.split("-")
            sel_p1 = int(left.strip())
            sel_p2 = int(right.strip())
        except (ValueError, AttributeError):
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=f"Cannot parse exact_score selection: '{selection}'",
            )

        if score_type == "sets":
            if not result.is_sets_format:
                return GradeResult(
                    outcome=GradeOutcome.VOID,
                    reason="Sets-type exact score on a legs-format match — void",
                )
            if result.p1_sets is None or result.p2_sets is None:
                return GradeResult(
                    outcome=GradeOutcome.VOID,
                    reason="Set scores not available — exact score void",
                )
            actual_p1, actual_p2 = result.p1_sets, result.p2_sets
        else:
            actual_p1, actual_p2 = result.p1_legs, result.p2_legs

        if sel_p1 == actual_p1 and sel_p2 == actual_p2:
            return GradeResult(
                outcome=GradeOutcome.WIN,
                reason=(
                    f"Correct score ({score_type}): {actual_p1}-{actual_p2} "
                    f"matches selection {sel_p1}-{sel_p2}"
                ),
            )

        return GradeResult(
            outcome=GradeOutcome.LOSE,
            reason=(
                f"Actual score ({score_type}): {actual_p1}-{actual_p2} "
                f"vs selection {sel_p1}-{sel_p2}"
            ),
        )

    # ------------------------------------------------------------------
    # 3. Total Legs Over/Under
    # ------------------------------------------------------------------

    def _grade_total_legs_over(
        self,
        result: MatchResult,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """
        Grade total legs over/under.

        Selection: "over" | "under"
        params["line"]: float — e.g. 8.5

        PUSH is returned when total_legs_played == integer line exactly.
        """
        line = params.get("line")
        if line is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="total_legs_over: missing line parameter — void",
            )

        try:
            line = float(line)
        except (TypeError, ValueError):
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=f"total_legs_over: invalid line '{line}' — void",
            )

        total = result.total_legs_played
        sel = selection.lower()

        # Half-line: no push possible
        if line == math.floor(line):
            # Integer line: push if exactly equal
            int_line = int(line)
            if total == int_line:
                return GradeResult(
                    outcome=GradeOutcome.PUSH,
                    reason=f"Total legs {total} == integer line {int_line} — push",
                )

        if sel == "over":
            if total > line:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=f"Total legs {total} > line {line} — over wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=f"Total legs {total} <= line {line} — over loses",
            )

        if sel == "under":
            if total < line:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=f"Total legs {total} < line {line} — under wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=f"Total legs {total} >= line {line} — under loses",
            )

        return GradeResult(
            outcome=GradeOutcome.VOID,
            reason=f"Unrecognised total_legs_over selection: '{selection}'",
        )

    # ------------------------------------------------------------------
    # 4. Handicap (legs or sets)
    # ------------------------------------------------------------------

    def _grade_handicap(
        self,
        result: MatchResult,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """
        Grade a leg or set handicap market.

        Selection: "p1" | "p2"
        params["line"]: float — the handicap applied to the selected player
                                Positive = receiving handicap
                                Negative = giving handicap
        params["handicap_type"]: "legs" (default) | "sets"

        e.g. selection="p1", line=+1.5 means P1 +1.5 legs.
        If P1 wins 6-4 (margin = +2), selection wins because 6+1.5 > 4.

        PUSH on exact integer handicap tie.
        """
        line = params.get("line")
        if line is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="handicap: missing line parameter — void",
            )

        try:
            line = float(line)
        except (TypeError, ValueError):
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=f"handicap: invalid line '{line}' — void",
            )

        handicap_type: str = params.get("handicap_type", "legs")
        sel = selection.lower()

        if handicap_type == "sets":
            if not result.is_sets_format:
                return GradeResult(
                    outcome=GradeOutcome.VOID,
                    reason="Sets handicap on a legs-format match — void",
                )
            if result.p1_sets is None or result.p2_sets is None:
                return GradeResult(
                    outcome=GradeOutcome.VOID,
                    reason="Set scores not available for handicap grading — void",
                )
            p1_score, p2_score = result.p1_sets, result.p2_sets
        else:
            p1_score, p2_score = result.p1_legs, result.p2_legs

        # Apply handicap from the perspective of the selected player
        if sel == "p1":
            adjusted_p1 = p1_score + line
            adjusted_p2 = float(p2_score)
        elif sel == "p2":
            adjusted_p2 = p2_score + line
            adjusted_p1 = float(p1_score)
        else:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=f"Unrecognised handicap selection: '{selection}'",
            )

        diff = adjusted_p1 - adjusted_p2

        if diff > 0.0:
            winner = "p1"
        elif diff < 0.0:
            winner = "p2"
        else:
            # Exact tie → push
            return GradeResult(
                outcome=GradeOutcome.PUSH,
                reason=(
                    f"Handicap tie ({handicap_type}): "
                    f"{p1_score} vs {p2_score} with line {line} — push"
                ),
            )

        if sel == winner:
            return GradeResult(
                outcome=GradeOutcome.WIN,
                reason=(
                    f"Handicap {handicap_type}: "
                    f"P1={p1_score}, P2={p2_score}, line={line:+.1f} "
                    f"— adjusted diff={diff:+.1f}, selection '{sel}' wins"
                ),
            )

        return GradeResult(
            outcome=GradeOutcome.LOSE,
            reason=(
                f"Handicap {handicap_type}: "
                f"P1={p1_score}, P2={p2_score}, line={line:+.1f} "
                f"— adjusted diff={diff:+.1f}, selection '{sel}' loses"
            ),
        )

    # ------------------------------------------------------------------
    # 5. Most 180s
    # ------------------------------------------------------------------

    def _grade_most_180s(
        self,
        result: MatchResult,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """
        Grade which player scored the most 180s in the match.

        Selection: "p1" | "p2" | "tie"
        """
        p1_180s = result.p1_180s_total
        p2_180s = result.p2_180s_total

        # Guard: both zero could mean data was not collected → void
        if p1_180s == 0 and p2_180s == 0:
            # Could be genuine (no 180s hit) or missing data.
            # We can only grade if we have leg-level records confirming no 180s.
            if not result.legs:
                return GradeResult(
                    outcome=GradeOutcome.VOID,
                    reason="No 180 data available — most_180s market void",
                )
            # Leg records exist but no 180s recorded → genuine zero-180 match
            # Tie applies
            if selection.lower() == "tie":
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason="Both players scored 0 180s — tie wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason="Both players scored 0 180s — tie; non-tie selection loses",
            )

        sel = selection.lower()

        if p1_180s > p2_180s:
            actual = "p1"
        elif p2_180s > p1_180s:
            actual = "p2"
        else:
            actual = "tie"

        if sel == actual:
            return GradeResult(
                outcome=GradeOutcome.WIN,
                reason=f"Most 180s: P1={p1_180s}, P2={p2_180s} — '{sel}' wins",
            )

        return GradeResult(
            outcome=GradeOutcome.LOSE,
            reason=f"Most 180s: P1={p1_180s}, P2={p2_180s} — '{sel}' loses",
        )

    # ------------------------------------------------------------------
    # 6. Total 180s Over/Under
    # ------------------------------------------------------------------

    def _grade_180_over(
        self,
        result: MatchResult,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """
        Grade total 180s (combined P1 + P2) over/under.

        Selection: "over" | "under"
        params["line"]: float — e.g. 6.5
        """
        line = params.get("line")
        if line is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="180_over: missing line parameter — void",
            )

        try:
            line = float(line)
        except (TypeError, ValueError):
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=f"180_over: invalid line '{line}' — void",
            )

        total_180s = result.p1_180s_total + result.p2_180s_total

        # If both counters are zero and no leg records, data may be missing
        if total_180s == 0 and not result.legs and result.p1_180s_total == 0:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="No 180 count data available — 180_over market void",
            )

        sel = selection.lower()

        if line == math.floor(line):
            int_line = int(line)
            if total_180s == int_line:
                return GradeResult(
                    outcome=GradeOutcome.PUSH,
                    reason=f"Total 180s {total_180s} == integer line {int_line} — push",
                )

        if sel == "over":
            if total_180s > line:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=f"Total 180s {total_180s} > line {line} — over wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=f"Total 180s {total_180s} <= line {line} — over loses",
            )

        if sel == "under":
            if total_180s < line:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=f"Total 180s {total_180s} < line {line} — under wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=f"Total 180s {total_180s} >= line {line} — under loses",
            )

        return GradeResult(
            outcome=GradeOutcome.VOID,
            reason=f"Unrecognised 180_over selection: '{selection}'",
        )

    # ------------------------------------------------------------------
    # 7. Highest Checkout
    # ------------------------------------------------------------------

    def _grade_highest_checkout(
        self,
        result: MatchResult,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """
        Grade which player hit the highest single-leg checkout score.

        Selection: "p1" | "p2" | "tie"

        A tie is declared when both players' highest checkouts are equal.
        If checkout data is unavailable → VOID.
        """
        p1_co = result.p1_highest_checkout
        p2_co = result.p2_highest_checkout

        if p1_co is None and p2_co is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="No checkout data available — highest_checkout market void",
            )

        # If only one player has checkout data, void is the safest outcome
        if p1_co is None or p2_co is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=(
                    "Incomplete checkout data (one player missing) "
                    "— highest_checkout market void"
                ),
            )

        sel = selection.lower()

        if p1_co > p2_co:
            actual = "p1"
        elif p2_co > p1_co:
            actual = "p2"
        else:
            actual = "tie"

        if sel == actual:
            return GradeResult(
                outcome=GradeOutcome.WIN,
                reason=(
                    f"Highest checkout: P1={p1_co}, P2={p2_co} "
                    f"— '{sel}' wins"
                ),
            )

        return GradeResult(
            outcome=GradeOutcome.LOSE,
            reason=(
                f"Highest checkout: P1={p1_co}, P2={p2_co} "
                f"— '{sel}' loses"
            ),
        )

    # ------------------------------------------------------------------
    # 8. First Leg Winner
    # ------------------------------------------------------------------

    def _grade_first_leg_winner(
        self,
        result: MatchResult,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """
        Grade who won the first leg of the match.

        Selection: "p1" | "p2"

        Requires at least one leg record in result.legs, or the
        presence of a dedicated first-leg field.
        Uses the first entry in result.legs (ordered by leg_number).
        """
        if not result.legs:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="No leg records available — first_leg_winner market void",
            )

        # Sort by leg_number to ensure correct ordering
        sorted_legs = sorted(result.legs, key=lambda l: l.leg_number)
        first_leg = sorted_legs[0]

        if first_leg.winner_is_p1 is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="First leg winner unknown — market void",
            )

        sel = selection.lower()
        actual = "p1" if first_leg.winner_is_p1 else "p2"

        if sel == actual:
            return GradeResult(
                outcome=GradeOutcome.WIN,
                reason=f"First leg won by {actual.upper()} — selection '{sel}' wins",
            )

        return GradeResult(
            outcome=GradeOutcome.LOSE,
            reason=f"First leg won by {actual.upper()} — selection '{sel}' loses",
        )

    # ------------------------------------------------------------------
    # 9. Race to X Legs
    # ------------------------------------------------------------------

    def _grade_race_to_x(
        self,
        result: MatchResult,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """
        Grade a race-to-X-legs market.

        Selection: "p1" | "p2"
        params["target"]: int — the target leg count (e.g. 3)

        The winner is the player who first reached ``target`` legs.
        If the match ended before either player reached target → VOID.
        """
        target = params.get("target")
        if target is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="race_to_x: missing target parameter — void",
            )

        try:
            target = int(target)
        except (TypeError, ValueError):
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=f"race_to_x: invalid target '{target}' — void",
            )

        if target < 1:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=f"race_to_x: target must be >= 1, got {target} — void",
            )

        # Determine who first reached the target using leg-level data
        if result.legs:
            p1_running = 0
            p2_running = 0
            sorted_legs = sorted(result.legs, key=lambda l: l.leg_number)
            race_winner: Optional[str] = None

            for leg in sorted_legs:
                if leg.winner_is_p1 is None:
                    continue
                if leg.winner_is_p1:
                    p1_running += 1
                    if p1_running >= target:
                        race_winner = "p1"
                        break
                else:
                    p2_running += 1
                    if p2_running >= target:
                        race_winner = "p2"
                        break

            if race_winner is None:
                # Neither player reached target (match ended early or retired)
                return GradeResult(
                    outcome=GradeOutcome.VOID,
                    reason=(
                        f"Neither player reached {target} legs before match ended "
                        f"— race_to_x void"
                    ),
                )
        else:
            # No per-leg records: infer from match totals only
            # The overall match winner reached their final leg count first
            if result.winner_is_p1 is None:
                return GradeResult(
                    outcome=GradeOutcome.VOID,
                    reason="race_to_x: no match winner or leg data — void",
                )
            # Confirm the winner actually reached the target
            if result.winner_is_p1:
                if result.p1_legs >= target:
                    race_winner = "p1"
                else:
                    return GradeResult(
                        outcome=GradeOutcome.VOID,
                        reason=(
                            f"P1 won but only reached {result.p1_legs} legs "
                            f"(target={target}) — race_to_x void"
                        ),
                    )
            else:
                if result.p2_legs >= target:
                    race_winner = "p2"
                else:
                    return GradeResult(
                        outcome=GradeOutcome.VOID,
                        reason=(
                            f"P2 won but only reached {result.p2_legs} legs "
                            f"(target={target}) — race_to_x void"
                        ),
                    )

        sel = selection.lower()
        if sel == race_winner:
            return GradeResult(
                outcome=GradeOutcome.WIN,
                reason=(
                    f"Race to {target}: {race_winner.upper()} "
                    f"reached {target} legs first — selection '{sel}' wins"
                ),
            )

        return GradeResult(
            outcome=GradeOutcome.LOSE,
            reason=(
                f"Race to {target}: {race_winner.upper()} "
                f"reached {target} legs first — selection '{sel}' loses"
            ),
        )

    # ------------------------------------------------------------------
    # 10. Nine-Darter in Match (Yes/No)
    # ------------------------------------------------------------------

    def _grade_nine_dart_finish(
        self,
        result: MatchResult,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """
        Grade whether a nine-darter was completed in the match.

        Selection: "yes" | "no"
        """
        sel = selection.lower()

        # Prefer explicit match-level flag
        nine_darter_hit = result.nine_darter_in_match

        # Also scan legs if available (for defensive double-checking)
        if not nine_darter_hit and result.legs:
            nine_darter_hit = any(leg.nine_darter for leg in result.legs)

        if sel == "yes":
            if nine_darter_hit:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason="Nine-darter was hit in the match — 'yes' wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason="No nine-darter was hit in the match — 'yes' loses",
            )

        if sel == "no":
            if not nine_darter_hit:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason="No nine-darter in the match — 'no' wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason="A nine-darter was hit in the match — 'no' loses",
            )

        return GradeResult(
            outcome=GradeOutcome.VOID,
            reason=f"Unrecognised nine_dart_finish selection: '{selection}'",
        )

    # ------------------------------------------------------------------
    # 11. Player Checkout Over/Under
    # ------------------------------------------------------------------

    def _grade_player_checkout_over(
        self,
        result: MatchResult,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """
        Grade a player-specific highest checkout over/under market.

        Selection: "over" | "under"
        params["player"]: "p1" | "p2"
        params["line"]: float — e.g. 100.5

        Grade is based on the specified player's highest single-leg checkout.
        """
        player = (params.get("player") or "").lower()
        line = params.get("line")

        if player not in ("p1", "p2"):
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=(
                    f"player_checkout_over: 'player' must be 'p1' or 'p2', "
                    f"got '{player}' — void"
                ),
            )

        if line is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="player_checkout_over: missing line parameter — void",
            )

        try:
            line = float(line)
        except (TypeError, ValueError):
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=f"player_checkout_over: invalid line '{line}' — void",
            )

        player_co = (
            result.p1_highest_checkout if player == "p1"
            else result.p2_highest_checkout
        )

        if player_co is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=(
                    f"No checkout data for {player.upper()} "
                    "— player_checkout_over void"
                ),
            )

        sel = selection.lower()

        if line == math.floor(line):
            int_line = int(line)
            if player_co == int_line:
                return GradeResult(
                    outcome=GradeOutcome.PUSH,
                    reason=(
                        f"{player.upper()} highest checkout {player_co} "
                        f"== integer line {int_line} — push"
                    ),
                )

        if sel == "over":
            if player_co > line:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=(
                        f"{player.upper()} highest checkout {player_co} "
                        f"> line {line} — over wins"
                    ),
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=(
                    f"{player.upper()} highest checkout {player_co} "
                    f"<= line {line} — over loses"
                ),
            )

        if sel == "under":
            if player_co < line:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=(
                        f"{player.upper()} highest checkout {player_co} "
                        f"< line {line} — under wins"
                    ),
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=(
                    f"{player.upper()} highest checkout {player_co} "
                    f">= line {line} — under loses"
                ),
            )

        return GradeResult(
            outcome=GradeOutcome.VOID,
            reason=f"Unrecognised player_checkout_over selection: '{selection}'",
        )

    # ------------------------------------------------------------------
    # 12. Sets Over/Under (sets-format only)
    # ------------------------------------------------------------------

    def _grade_sets_over(
        self,
        result: MatchResult,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """
        Grade total sets played over/under.

        Selection: "over" | "under"
        params["line"]: float — e.g. 3.5

        Only applicable to sets-format matches (e.g. PDC World Championship).
        Returns VOID for legs-only formats.
        """
        if not result.is_sets_format:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="sets_over: match is not a sets-format match — void",
            )

        total_sets = result.total_sets_played
        if total_sets is None:
            # Try to compute from individual set scores
            if result.p1_sets is not None and result.p2_sets is not None:
                total_sets = result.p1_sets + result.p2_sets
            else:
                return GradeResult(
                    outcome=GradeOutcome.VOID,
                    reason="sets_over: total sets played not available — void",
                )

        line = params.get("line")
        if line is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="sets_over: missing line parameter — void",
            )

        try:
            line = float(line)
        except (TypeError, ValueError):
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=f"sets_over: invalid line '{line}' — void",
            )

        sel = selection.lower()

        if line == math.floor(line):
            int_line = int(line)
            if total_sets == int_line:
                return GradeResult(
                    outcome=GradeOutcome.PUSH,
                    reason=f"Total sets {total_sets} == integer line {int_line} — push",
                )

        if sel == "over":
            if total_sets > line:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=f"Total sets {total_sets} > line {line} — over wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=f"Total sets {total_sets} <= line {line} — over loses",
            )

        if sel == "under":
            if total_sets < line:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=f"Total sets {total_sets} < line {line} — under wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=f"Total sets {total_sets} >= line {line} — under loses",
            )

        return GradeResult(
            outcome=GradeOutcome.VOID,
            reason=f"Unrecognised sets_over selection: '{selection}'",
        )

    # ------------------------------------------------------------------
    # 13. Break of Throw (first leg won by non-starter)
    # ------------------------------------------------------------------

    def _grade_break_of_throw(
        self,
        result: MatchResult,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """
        Grade whether the first leg was won by the non-starter (a break).

        Selection: "yes" | "no"

        "Break" = the player who did NOT throw first in leg 1 won leg 1.
        Requires first leg record with both winner_is_p1 and starter_is_p1.
        """
        if not result.legs:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="No leg records — break_of_throw void",
            )

        sorted_legs = sorted(result.legs, key=lambda l: l.leg_number)
        first_leg = sorted_legs[0]

        if first_leg.winner_is_p1 is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="First leg winner unknown — break_of_throw void",
            )

        if first_leg.starter_is_p1 is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="First leg starter unknown — break_of_throw void",
            )

        # Break = starter did NOT win
        is_break = first_leg.starter_is_p1 != first_leg.winner_is_p1
        sel = selection.lower()

        if sel == "yes":
            if is_break:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason="First leg was a break of throw — 'yes' wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason="First leg was held (no break) — 'yes' loses",
            )

        if sel == "no":
            if not is_break:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason="First leg was held (no break) — 'no' wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason="First leg was a break of throw — 'no' loses",
            )

        return GradeResult(
            outcome=GradeOutcome.VOID,
            reason=f"Unrecognised break_of_throw selection: '{selection}'",
        )

    # ------------------------------------------------------------------
    # 14. Leg Winner (specific leg number)
    # ------------------------------------------------------------------

    def _grade_leg_winner_next(
        self,
        result: MatchResult,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """
        Grade who won a specific leg of the match.

        Selection: "p1" | "p2"
        params["leg_number"]: int — 1-indexed leg number

        VOID if that leg was not played or winner data is missing.
        """
        leg_number = params.get("leg_number")
        if leg_number is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="leg_winner_next: missing leg_number parameter — void",
            )

        try:
            leg_number = int(leg_number)
        except (TypeError, ValueError):
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=f"leg_winner_next: invalid leg_number '{leg_number}' — void",
            )

        if not result.legs:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="No leg records available — leg_winner_next void",
            )

        target_leg: Optional[LegRecord] = None
        for leg in result.legs:
            if leg.leg_number == leg_number:
                target_leg = leg
                break

        if target_leg is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=(
                    f"Leg {leg_number} was not played "
                    "— leg_winner_next void"
                ),
            )

        if target_leg.winner_is_p1 is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=f"Leg {leg_number} winner unknown — market void",
            )

        sel = selection.lower()
        actual = "p1" if target_leg.winner_is_p1 else "p2"

        if sel == actual:
            return GradeResult(
                outcome=GradeOutcome.WIN,
                reason=(
                    f"Leg {leg_number} won by {actual.upper()} "
                    f"— selection '{sel}' wins"
                ),
            )

        return GradeResult(
            outcome=GradeOutcome.LOSE,
            reason=(
                f"Leg {leg_number} won by {actual.upper()} "
                f"— selection '{sel}' loses"
            ),
        )

    # ------------------------------------------------------------------
    # 15. Total 180s in Band
    # ------------------------------------------------------------------

    def _grade_total_180s_band(
        self,
        result: MatchResult,
        selection: str,
        params: dict[str, Any],
    ) -> GradeResult:
        """
        Grade whether total 180s (combined) fall within a specified band.

        Selection: "yes" | "no"
        params["low"]: int — inclusive lower bound of the band
        params["high"]: int — inclusive upper bound of the band

        Example: params={"low": 5, "high": 7}, selection="yes"
        wins if total 180s is 5, 6, or 7.
        """
        low = params.get("low")
        high = params.get("high")

        if low is None or high is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="total_180s_band: missing low/high parameters — void",
            )

        try:
            low = int(low)
            high = int(high)
        except (TypeError, ValueError):
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=(
                    f"total_180s_band: invalid band [{low}, {high}] — void"
                ),
            )

        if low > high:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason=(
                    f"total_180s_band: low={low} > high={high} — void"
                ),
            )

        total_180s = result.p1_180s_total + result.p2_180s_total

        # Guard for missing data
        if total_180s == 0 and not result.legs:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="No 180 data available — total_180s_band void",
            )

        in_band = low <= total_180s <= high
        sel = selection.lower()

        if sel == "yes":
            if in_band:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=(
                        f"Total 180s {total_180s} in band [{low}-{high}] — 'yes' wins"
                    ),
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=(
                    f"Total 180s {total_180s} outside band [{low}-{high}] — 'yes' loses"
                ),
            )

        if sel == "no":
            if not in_band:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=(
                        f"Total 180s {total_180s} outside band [{low}-{high}] — 'no' wins"
                    ),
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=(
                    f"Total 180s {total_180s} in band [{low}-{high}] — 'no' loses"
                ),
            )

        return GradeResult(
            outcome=GradeOutcome.VOID,
            reason=f"Unrecognised total_180s_band selection: '{selection}'",
        )

    # ------------------------------------------------------------------
    # both_to_score — both players must win >= N legs
    # ------------------------------------------------------------------

    def _grade_both_to_score(
        self,
        result: "MatchResult",
        selection: str,
        params: dict[str, Any] | None,
    ) -> GradeResult:
        """Grade 'both to score X+ legs' market.

        Selection: 'yes' or 'no'.
        Params must include 'threshold' (int, minimum legs each player must win).
        """
        if params is None or "threshold" not in params:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="both_to_score requires 'threshold' parameter",
            )
        threshold = int(params["threshold"])
        p1_legs = result.p1_legs_won
        p2_legs = result.p2_legs_won
        if p1_legs is None or p2_legs is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="Leg counts unavailable — void",
            )
        both_scored = p1_legs >= threshold and p2_legs >= threshold
        sel = selection.strip().lower()
        if sel == "yes":
            if both_scored:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=f"Both scored {threshold}+ legs ({p1_legs}-{p2_legs}) — 'yes' wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=f"Not both scored {threshold}+ legs ({p1_legs}-{p2_legs}) — 'yes' loses",
            )
        if sel == "no":
            if not both_scored:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=f"Not both scored {threshold}+ legs ({p1_legs}-{p2_legs}) — 'no' wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=f"Both scored {threshold}+ legs ({p1_legs}-{p2_legs}) — 'no' loses",
            )
        return GradeResult(
            outcome=GradeOutcome.VOID,
            reason=f"Unrecognised both_to_score selection: '{selection}'",
        )

    # ------------------------------------------------------------------
    # winning_margin — margin of victory in legs
    # ------------------------------------------------------------------

    def _grade_winning_margin(
        self,
        result: "MatchResult",
        selection: str,
        params: dict[str, Any] | None,
    ) -> GradeResult:
        """Grade 'winning margin' market.

        Selection: 'exact_N' (margin == N), 'over_N' (margin > N), 'under_N' (margin < N),
                   or 'N+' (margin >= N).
        Params may include 'margin_line' (float) as alternative to selection-embedded value.
        """
        p1_legs = result.p1_legs_won
        p2_legs = result.p2_legs_won
        if p1_legs is None or p2_legs is None:
            return GradeResult(
                outcome=GradeOutcome.VOID,
                reason="Leg counts unavailable — void",
            )
        actual_margin = abs(p1_legs - p2_legs)
        sel = selection.strip().lower()

        # Parse selection format: 'exact_3', 'over_2.5', 'under_4.5', '3+'
        import re

        m_exact = re.match(r"exact[_\s]?(\d+)", sel)
        if m_exact:
            target = int(m_exact.group(1))
            if actual_margin == target:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=f"Margin {actual_margin} == {target} — 'exact_{target}' wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=f"Margin {actual_margin} != {target} — 'exact_{target}' loses",
            )

        m_over = re.match(r"over[_\s]?([\d.]+)", sel)
        if m_over:
            line = float(m_over.group(1))
            if actual_margin > line:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=f"Margin {actual_margin} > {line} — 'over' wins",
                )
            if actual_margin == line and line == int(line):
                return GradeResult(
                    outcome=GradeOutcome.PUSH,
                    reason=f"Margin {actual_margin} == {line} — push",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=f"Margin {actual_margin} <= {line} — 'over' loses",
            )

        m_under = re.match(r"under[_\s]?([\d.]+)", sel)
        if m_under:
            line = float(m_under.group(1))
            if actual_margin < line:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=f"Margin {actual_margin} < {line} — 'under' wins",
                )
            if actual_margin == line and line == int(line):
                return GradeResult(
                    outcome=GradeOutcome.PUSH,
                    reason=f"Margin {actual_margin} == {line} — push",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=f"Margin {actual_margin} >= {line} — 'under' loses",
            )

        m_plus = re.match(r"(\d+)\+", sel)
        if m_plus:
            target = int(m_plus.group(1))
            if actual_margin >= target:
                return GradeResult(
                    outcome=GradeOutcome.WIN,
                    reason=f"Margin {actual_margin} >= {target} — '{target}+' wins",
                )
            return GradeResult(
                outcome=GradeOutcome.LOSE,
                reason=f"Margin {actual_margin} < {target} — '{target}+' loses",
            )

        return GradeResult(
            outcome=GradeOutcome.VOID,
            reason=f"Unrecognised winning_margin selection: '{selection}'",
        )
