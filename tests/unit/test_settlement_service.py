"""
Unit tests for DartsSettlementService — all 15 market types.

Tests cover:
- Normal WIN / LOSE outcomes for every market type
- VOID conditions (missing data, invalid params, cancelled/postponed)
- PUSH conditions (integer handicap/totals lines)
- Edge cases (walkover, retired, abandoned, draw, zero 180s, ties)
- All 15 market types from the monitoring engine map
"""
from __future__ import annotations

import pytest

from settlement.darts_settlement_service import (
    DartsSettlementService,
    GradeOutcome,
    LegRecord,
    MatchResult,
    SettlementReport,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(**kwargs) -> MatchResult:
    """Build a minimal completed MatchResult with overrides."""
    defaults = dict(
        match_id="test-match-001",
        status="Completed",
        winner_is_p1=True,
        p1_legs=6,
        p2_legs=4,
        p1_180s_total=3,
        p2_180s_total=2,
        p1_highest_checkout=121,
        p2_highest_checkout=96,
        nine_darter_in_match=False,
        legs=[],
    )
    defaults.update(kwargs)
    return MatchResult(**defaults)


def _make_legs(n: int, p1_wins: list[bool]) -> list[LegRecord]:
    """Build n leg records with specified winners."""
    legs = []
    for i in range(n):
        legs.append(LegRecord(
            leg_number=i + 1,
            winner_is_p1=p1_wins[i] if i < len(p1_wins) else None,
            starter_is_p1=(i % 2 == 0),  # alternating starts
            had_180=False,
            nine_darter=False,
            winning_checkout=40,
            p1_180s=1 if i == 0 else 0,
            p2_180s=0,
        ))
    return legs


_svc = DartsSettlementService()


def _grade(result: MatchResult, market_type: str, selection: str, **params) -> GradeOutcome:
    """Helper: grade a single market and return the outcome."""
    markets = [{
        "market_type": market_type,
        "selection": selection,
        "market_params": params,
    }]
    report = _svc.grade_match(result, markets)
    assert len(report.grades) == 1
    return report.grades[0].grade.outcome


# ===========================================================================
# 1. match_win
# ===========================================================================

class TestMatchWin:
    def test_p1_wins_selection_p1(self):
        r = _make_result(winner_is_p1=True)
        assert _grade(r, "match_win", "p1") == GradeOutcome.WIN

    def test_p1_wins_selection_p2(self):
        r = _make_result(winner_is_p1=True)
        assert _grade(r, "match_win", "p2") == GradeOutcome.LOSE

    def test_p2_wins_selection_p2(self):
        r = _make_result(winner_is_p1=False, p1_legs=4, p2_legs=6)
        assert _grade(r, "match_win", "p2") == GradeOutcome.WIN

    def test_draw_selection_draw(self):
        r = _make_result(winner_is_p1=None, is_draw=True, p1_legs=5, p2_legs=5)
        assert _grade(r, "match_win", "draw") == GradeOutcome.WIN

    def test_draw_selection_p1_loses(self):
        r = _make_result(winner_is_p1=None, is_draw=True, p1_legs=5, p2_legs=5)
        assert _grade(r, "match_win", "p1") == GradeOutcome.LOSE

    def test_walkover_winner_declared_grades_normally(self):
        r = _make_result(status="Walkover", winner_is_p1=True)
        assert _grade(r, "match_win", "p1") == GradeOutcome.WIN

    def test_walkover_no_winner_void(self):
        r = _make_result(status="Walkover", winner_is_p1=None)
        assert _grade(r, "match_win", "p1") == GradeOutcome.VOID

    def test_cancelled_always_void(self):
        r = _make_result(status="Cancelled")
        assert _grade(r, "match_win", "p1") == GradeOutcome.VOID

    def test_postponed_always_void(self):
        r = _make_result(status="Postponed")
        assert _grade(r, "match_win", "p2") == GradeOutcome.VOID

    def test_abandoned_void(self):
        r = _make_result(status="Abandoned")
        assert _grade(r, "match_win", "p1") == GradeOutcome.VOID

    def test_retired_with_winner_grades(self):
        r = _make_result(status="Retired", winner_is_p1=True)
        assert _grade(r, "match_win", "p1") == GradeOutcome.WIN

    def test_retired_no_winner_void(self):
        r = _make_result(status="Retired", winner_is_p1=None)
        assert _grade(r, "match_win", "p1") == GradeOutcome.VOID

    def test_unrecognised_selection_void(self):
        r = _make_result(winner_is_p1=True)
        assert _grade(r, "match_win", "invalid") == GradeOutcome.VOID

    def test_draw_selection_but_no_draw(self):
        r = _make_result(winner_is_p1=True)
        assert _grade(r, "match_win", "draw") == GradeOutcome.LOSE


# ===========================================================================
# 2. exact_score
# ===========================================================================

class TestExactScore:
    def test_correct_legs_score_wins(self):
        r = _make_result(p1_legs=6, p2_legs=4)
        assert _grade(r, "exact_score", "6-4") == GradeOutcome.WIN

    def test_wrong_legs_score_loses(self):
        r = _make_result(p1_legs=6, p2_legs=4)
        assert _grade(r, "exact_score", "6-3") == GradeOutcome.LOSE

    def test_correct_sets_score_wins(self):
        r = _make_result(
            is_sets_format=True,
            p1_sets=4,
            p2_sets=2,
        )
        assert _grade(r, "exact_score", "4-2", score_type="sets") == GradeOutcome.WIN

    def test_wrong_sets_score_loses(self):
        r = _make_result(
            is_sets_format=True,
            p1_sets=4,
            p2_sets=2,
        )
        assert _grade(r, "exact_score", "3-2", score_type="sets") == GradeOutcome.LOSE

    def test_sets_score_on_legs_format_void(self):
        r = _make_result(is_sets_format=False)
        assert _grade(r, "exact_score", "4-2", score_type="sets") == GradeOutcome.VOID

    def test_malformed_selection_void(self):
        r = _make_result()
        assert _grade(r, "exact_score", "not-a-score") == GradeOutcome.VOID

    def test_missing_sets_data_void(self):
        r = _make_result(is_sets_format=True, p1_sets=None, p2_sets=None)
        assert _grade(r, "exact_score", "4-2", score_type="sets") == GradeOutcome.VOID


# ===========================================================================
# 3. total_legs_over
# ===========================================================================

class TestTotalLegsOver:
    def test_over_wins(self):
        r = _make_result(p1_legs=6, p2_legs=4, total_legs_played=10)
        assert _grade(r, "total_legs_over", "over", line=8.5) == GradeOutcome.WIN

    def test_over_loses(self):
        r = _make_result(p1_legs=5, p2_legs=3, total_legs_played=8)
        assert _grade(r, "total_legs_over", "over", line=8.5) == GradeOutcome.LOSE

    def test_under_wins(self):
        r = _make_result(p1_legs=5, p2_legs=3, total_legs_played=8)
        assert _grade(r, "total_legs_over", "under", line=8.5) == GradeOutcome.WIN

    def test_under_loses(self):
        r = _make_result(p1_legs=6, p2_legs=4, total_legs_played=10)
        assert _grade(r, "total_legs_over", "under", line=8.5) == GradeOutcome.LOSE

    def test_integer_line_push(self):
        r = _make_result(p1_legs=5, p2_legs=3, total_legs_played=8)
        assert _grade(r, "total_legs_over", "over", line=8) == GradeOutcome.PUSH

    def test_missing_line_void(self):
        r = _make_result()
        assert _grade(r, "total_legs_over", "over") == GradeOutcome.VOID

    def test_invalid_selection_void(self):
        r = _make_result(total_legs_played=10)
        assert _grade(r, "total_legs_over", "neither", line=8.5) == GradeOutcome.VOID


# ===========================================================================
# 4. handicap
# ===========================================================================

class TestHandicap:
    def test_p1_plus_handicap_wins(self):
        # P1 wins 6-4; P1 +1.5 → adjusted 7.5 vs 4 → P1 wins
        r = _make_result(p1_legs=6, p2_legs=4)
        assert _grade(r, "handicap", "p1", line=1.5) == GradeOutcome.WIN

    def test_p1_minus_handicap_loses(self):
        # P1 wins 6-4; P1 -3.5 → adjusted 2.5 vs 4 → P2 wins
        r = _make_result(p1_legs=6, p2_legs=4)
        assert _grade(r, "handicap", "p1", line=-3.5) == GradeOutcome.LOSE

    def test_p2_plus_handicap_wins(self):
        # P2 loses 4-6; P2 +3.5 → adjusted P2=7.5 vs P1=6 → P2 wins
        r = _make_result(p1_legs=6, p2_legs=4)
        assert _grade(r, "handicap", "p2", line=3.5) == GradeOutcome.WIN

    def test_integer_line_push(self):
        # P1 6, P2 4; P1 -2 → adjusted 4 vs 4 → push
        r = _make_result(p1_legs=6, p2_legs=4)
        assert _grade(r, "handicap", "p1", line=-2) == GradeOutcome.PUSH

    def test_sets_handicap_wins(self):
        r = _make_result(is_sets_format=True, p1_sets=4, p2_sets=2)
        assert _grade(r, "handicap", "p1", line=1.5, handicap_type="sets") == GradeOutcome.WIN

    def test_sets_handicap_on_legs_format_void(self):
        r = _make_result(is_sets_format=False)
        assert _grade(r, "handicap", "p1", line=1.5, handicap_type="sets") == GradeOutcome.VOID

    def test_missing_line_void(self):
        r = _make_result()
        assert _grade(r, "handicap", "p1") == GradeOutcome.VOID

    def test_invalid_selection_void(self):
        r = _make_result()
        assert _grade(r, "handicap", "player3", line=1.5) == GradeOutcome.VOID


# ===========================================================================
# 5. most_180s
# ===========================================================================

class TestMost180s:
    def test_p1_most_wins(self):
        r = _make_result(p1_180s_total=5, p2_180s_total=2)
        assert _grade(r, "most_180s", "p1") == GradeOutcome.WIN

    def test_p2_most_wins(self):
        r = _make_result(p1_180s_total=2, p2_180s_total=5)
        assert _grade(r, "most_180s", "p2") == GradeOutcome.WIN

    def test_tie_selection_wins_on_equal(self):
        r = _make_result(p1_180s_total=3, p2_180s_total=3,
                         legs=[LegRecord(leg_number=1)])
        assert _grade(r, "most_180s", "tie") == GradeOutcome.WIN

    def test_p1_most_selection_p2_loses(self):
        r = _make_result(p1_180s_total=5, p2_180s_total=2)
        assert _grade(r, "most_180s", "p2") == GradeOutcome.LOSE

    def test_no_data_zero_zero_no_legs_void(self):
        r = _make_result(p1_180s_total=0, p2_180s_total=0, legs=[])
        assert _grade(r, "most_180s", "p1") == GradeOutcome.VOID

    def test_zero_zero_with_leg_records_tie_wins(self):
        r = _make_result(p1_180s_total=0, p2_180s_total=0,
                         legs=[LegRecord(leg_number=1, had_180=False)])
        assert _grade(r, "most_180s", "tie") == GradeOutcome.WIN


# ===========================================================================
# 6. 180_over
# ===========================================================================

class TestMatchTotal180sOver:
    def test_over_wins(self):
        r = _make_result(p1_180s_total=4, p2_180s_total=4)
        assert _grade(r, "180_over", "over", line=6.5) == GradeOutcome.WIN

    def test_over_loses(self):
        r = _make_result(p1_180s_total=2, p2_180s_total=2)
        assert _grade(r, "180_over", "over", line=6.5) == GradeOutcome.LOSE

    def test_under_wins(self):
        r = _make_result(p1_180s_total=2, p2_180s_total=2)
        assert _grade(r, "180_over", "under", line=6.5) == GradeOutcome.WIN

    def test_integer_line_push(self):
        r = _make_result(p1_180s_total=3, p2_180s_total=3,
                         legs=[LegRecord(leg_number=1)])
        assert _grade(r, "180_over", "over", line=6) == GradeOutcome.PUSH

    def test_missing_line_void(self):
        r = _make_result()
        assert _grade(r, "180_over", "over") == GradeOutcome.VOID

    def test_zero_totals_no_legs_void(self):
        r = _make_result(p1_180s_total=0, p2_180s_total=0, legs=[])
        assert _grade(r, "180_over", "over", line=4.5) == GradeOutcome.VOID


# ===========================================================================
# 7. highest_checkout
# ===========================================================================

class TestHighestCheckout:
    def test_p1_highest_wins(self):
        r = _make_result(p1_highest_checkout=121, p2_highest_checkout=96)
        assert _grade(r, "highest_checkout", "p1") == GradeOutcome.WIN

    def test_p2_highest_wins(self):
        r = _make_result(p1_highest_checkout=96, p2_highest_checkout=121)
        assert _grade(r, "highest_checkout", "p2") == GradeOutcome.WIN

    def test_tie_selection_wins_on_equal(self):
        r = _make_result(p1_highest_checkout=100, p2_highest_checkout=100)
        assert _grade(r, "highest_checkout", "tie") == GradeOutcome.WIN

    def test_no_checkout_data_void(self):
        r = _make_result(p1_highest_checkout=None, p2_highest_checkout=None)
        assert _grade(r, "highest_checkout", "p1") == GradeOutcome.VOID

    def test_partial_checkout_data_void(self):
        r = _make_result(p1_highest_checkout=100, p2_highest_checkout=None)
        assert _grade(r, "highest_checkout", "p1") == GradeOutcome.VOID


# ===========================================================================
# 8. first_leg_winner
# ===========================================================================

class TestFirstLegWinner:
    def test_p1_wins_leg1(self):
        legs = _make_legs(5, [True, False, True, False, True])
        r = _make_result(legs=legs)
        assert _grade(r, "first_leg_winner", "p1") == GradeOutcome.WIN

    def test_p2_wins_leg1(self):
        legs = _make_legs(5, [False, True, False, True, False])
        r = _make_result(legs=legs)
        assert _grade(r, "first_leg_winner", "p1") == GradeOutcome.LOSE

    def test_no_legs_void(self):
        r = _make_result(legs=[])
        assert _grade(r, "first_leg_winner", "p1") == GradeOutcome.VOID

    def test_first_leg_winner_unknown_void(self):
        legs = [LegRecord(leg_number=1, winner_is_p1=None)]
        r = _make_result(legs=legs)
        assert _grade(r, "first_leg_winner", "p1") == GradeOutcome.VOID


# ===========================================================================
# 9. race_to_x
# ===========================================================================

class TestRaceToX:
    def test_p1_wins_race_to_3(self):
        legs = _make_legs(6, [True, False, True, False, True, False])
        r = _make_result(p1_legs=3, p2_legs=3, legs=legs)
        assert _grade(r, "race_to_x", "p1", target=3) == GradeOutcome.WIN

    def test_p2_wins_race_to_3(self):
        legs = _make_legs(6, [False, True, False, True, False, True])
        r = _make_result(p1_legs=3, p2_legs=3, legs=legs)
        assert _grade(r, "race_to_x", "p2", target=3) == GradeOutcome.WIN

    def test_target_not_reached_void(self):
        # Only 2 legs played — neither player reached target of 5
        legs = _make_legs(2, [True, False])
        r = _make_result(p1_legs=1, p2_legs=1, legs=legs)
        assert _grade(r, "race_to_x", "p1", target=5) == GradeOutcome.VOID

    def test_no_legs_uses_match_totals(self):
        r = _make_result(winner_is_p1=True, p1_legs=6, p2_legs=4, legs=[])
        assert _grade(r, "race_to_x", "p1", target=6) == GradeOutcome.WIN

    def test_no_target_param_void(self):
        r = _make_result()
        assert _grade(r, "race_to_x", "p1") == GradeOutcome.VOID

    def test_target_zero_void(self):
        r = _make_result()
        assert _grade(r, "race_to_x", "p1", target=0) == GradeOutcome.VOID


# ===========================================================================
# 10. nine_dart_finish
# ===========================================================================

class TestNineDartFinish:
    def test_yes_wins_when_hit(self):
        r = _make_result(nine_darter_in_match=True)
        assert _grade(r, "nine_dart_finish", "yes") == GradeOutcome.WIN

    def test_yes_loses_when_not_hit(self):
        r = _make_result(nine_darter_in_match=False,
                         legs=[LegRecord(leg_number=1)])
        assert _grade(r, "nine_dart_finish", "yes") == GradeOutcome.LOSE

    def test_no_wins_when_not_hit(self):
        r = _make_result(nine_darter_in_match=False,
                         legs=[LegRecord(leg_number=1)])
        assert _grade(r, "nine_dart_finish", "no") == GradeOutcome.WIN

    def test_no_loses_when_hit(self):
        r = _make_result(nine_darter_in_match=True)
        assert _grade(r, "nine_dart_finish", "no") == GradeOutcome.LOSE

    def test_leg_level_nine_darter_detected(self):
        legs = [LegRecord(leg_number=1, nine_darter=True)]
        r = _make_result(nine_darter_in_match=False, legs=legs)
        assert _grade(r, "nine_dart_finish", "yes") == GradeOutcome.WIN

    def test_invalid_selection_void(self):
        r = _make_result(nine_darter_in_match=True)
        assert _grade(r, "nine_dart_finish", "maybe") == GradeOutcome.VOID


# ===========================================================================
# 11. player_checkout_over
# ===========================================================================

class TestPlayerCheckoutOver:
    def test_p1_over_wins(self):
        r = _make_result(p1_highest_checkout=121, p2_highest_checkout=96)
        assert _grade(r, "player_checkout_over", "over", player="p1", line=100.5) == GradeOutcome.WIN

    def test_p1_over_loses(self):
        r = _make_result(p1_highest_checkout=80, p2_highest_checkout=96)
        assert _grade(r, "player_checkout_over", "over", player="p1", line=100.5) == GradeOutcome.LOSE

    def test_p2_under_wins(self):
        r = _make_result(p1_highest_checkout=121, p2_highest_checkout=60)
        assert _grade(r, "player_checkout_over", "under", player="p2", line=100.5) == GradeOutcome.WIN

    def test_integer_line_push(self):
        r = _make_result(p1_highest_checkout=100, p2_highest_checkout=96)
        assert _grade(r, "player_checkout_over", "over", player="p1", line=100) == GradeOutcome.PUSH

    def test_missing_player_param_void(self):
        r = _make_result()
        assert _grade(r, "player_checkout_over", "over", line=100.5) == GradeOutcome.VOID

    def test_invalid_player_void(self):
        r = _make_result()
        assert _grade(r, "player_checkout_over", "over", player="p3", line=100.5) == GradeOutcome.VOID

    def test_missing_checkout_data_void(self):
        r = _make_result(p1_highest_checkout=None)
        assert _grade(r, "player_checkout_over", "over", player="p1", line=100.5) == GradeOutcome.VOID

    def test_missing_line_param_void(self):
        r = _make_result()
        assert _grade(r, "player_checkout_over", "over", player="p1") == GradeOutcome.VOID


# ===========================================================================
# 12. sets_over
# ===========================================================================

class TestSetsOver:
    def test_over_wins(self):
        r = _make_result(is_sets_format=True, p1_sets=4, p2_sets=2,
                         total_sets_played=6)
        assert _grade(r, "sets_over", "over", line=4.5) == GradeOutcome.WIN

    def test_under_wins(self):
        r = _make_result(is_sets_format=True, p1_sets=4, p2_sets=1,
                         total_sets_played=5)
        assert _grade(r, "sets_over", "under", line=5.5) == GradeOutcome.WIN

    def test_integer_line_push(self):
        r = _make_result(is_sets_format=True, p1_sets=3, p2_sets=1,
                         total_sets_played=4)
        assert _grade(r, "sets_over", "over", line=4) == GradeOutcome.PUSH

    def test_legs_format_void(self):
        r = _make_result(is_sets_format=False)
        assert _grade(r, "sets_over", "over", line=4.5) == GradeOutcome.VOID

    def test_missing_sets_data_void(self):
        r = _make_result(is_sets_format=True, p1_sets=None, p2_sets=None,
                         total_sets_played=None)
        assert _grade(r, "sets_over", "over", line=4.5) == GradeOutcome.VOID


# ===========================================================================
# 13. break_of_throw
# ===========================================================================

class TestBreakOfThrow:
    def test_yes_wins_on_break(self):
        # Leg 1: P1 starts, P2 wins → break
        legs = [LegRecord(leg_number=1, winner_is_p1=False, starter_is_p1=True)]
        r = _make_result(legs=legs)
        assert _grade(r, "break_of_throw", "yes") == GradeOutcome.WIN

    def test_no_wins_on_hold(self):
        # Leg 1: P1 starts, P1 wins → hold
        legs = [LegRecord(leg_number=1, winner_is_p1=True, starter_is_p1=True)]
        r = _make_result(legs=legs)
        assert _grade(r, "break_of_throw", "no") == GradeOutcome.WIN

    def test_yes_loses_on_hold(self):
        legs = [LegRecord(leg_number=1, winner_is_p1=True, starter_is_p1=True)]
        r = _make_result(legs=legs)
        assert _grade(r, "break_of_throw", "yes") == GradeOutcome.LOSE

    def test_no_legs_void(self):
        r = _make_result(legs=[])
        assert _grade(r, "break_of_throw", "yes") == GradeOutcome.VOID

    def test_unknown_starter_void(self):
        legs = [LegRecord(leg_number=1, winner_is_p1=True, starter_is_p1=None)]
        r = _make_result(legs=legs)
        assert _grade(r, "break_of_throw", "yes") == GradeOutcome.VOID

    def test_unknown_winner_void(self):
        legs = [LegRecord(leg_number=1, winner_is_p1=None, starter_is_p1=True)]
        r = _make_result(legs=legs)
        assert _grade(r, "break_of_throw", "yes") == GradeOutcome.VOID


# ===========================================================================
# 14. leg_winner_next (specific leg)
# ===========================================================================

class TestLegWinnerNext:
    def test_correct_leg_p1_wins(self):
        legs = _make_legs(5, [False, True, False, True, True])
        r = _make_result(legs=legs)
        # Leg 2: P1 wins (p1_wins[1] = True)
        assert _grade(r, "leg_winner_next", "p1", leg_number=2) == GradeOutcome.WIN

    def test_correct_leg_p2_wins(self):
        legs = _make_legs(5, [True, False, True, True, True])
        r = _make_result(legs=legs)
        # Leg 2: P2 wins (p1_wins[1] = False)
        assert _grade(r, "leg_winner_next", "p2", leg_number=2) == GradeOutcome.WIN

    def test_leg_not_played_void(self):
        legs = _make_legs(3, [True, False, True])
        r = _make_result(legs=legs)
        assert _grade(r, "leg_winner_next", "p1", leg_number=10) == GradeOutcome.VOID

    def test_no_legs_void(self):
        r = _make_result(legs=[])
        assert _grade(r, "leg_winner_next", "p1", leg_number=1) == GradeOutcome.VOID

    def test_missing_leg_number_void(self):
        r = _make_result(legs=_make_legs(3, [True, False, True]))
        assert _grade(r, "leg_winner_next", "p1") == GradeOutcome.VOID

    def test_leg_winner_unknown_void(self):
        legs = [LegRecord(leg_number=3, winner_is_p1=None)]
        r = _make_result(legs=legs)
        assert _grade(r, "leg_winner_next", "p1", leg_number=3) == GradeOutcome.VOID


# ===========================================================================
# 15. total_180s_band
# ===========================================================================

class TestTotal180sBand:
    def test_yes_wins_in_band(self):
        r = _make_result(p1_180s_total=3, p2_180s_total=3,
                         legs=[LegRecord(leg_number=1)])
        assert _grade(r, "total_180s_band", "yes", low=5, high=7) == GradeOutcome.WIN

    def test_yes_loses_outside_band(self):
        r = _make_result(p1_180s_total=1, p2_180s_total=1,
                         legs=[LegRecord(leg_number=1)])
        assert _grade(r, "total_180s_band", "yes", low=5, high=7) == GradeOutcome.LOSE

    def test_no_wins_outside_band(self):
        r = _make_result(p1_180s_total=1, p2_180s_total=0,
                         legs=[LegRecord(leg_number=1)])
        assert _grade(r, "total_180s_band", "no", low=5, high=7) == GradeOutcome.WIN

    def test_no_loses_in_band(self):
        r = _make_result(p1_180s_total=3, p2_180s_total=3,
                         legs=[LegRecord(leg_number=1)])
        assert _grade(r, "total_180s_band", "no", low=5, high=7) == GradeOutcome.LOSE

    def test_exactly_at_low_bound_wins(self):
        r = _make_result(p1_180s_total=3, p2_180s_total=2,
                         legs=[LegRecord(leg_number=1)])
        assert _grade(r, "total_180s_band", "yes", low=5, high=8) == GradeOutcome.WIN

    def test_exactly_at_high_bound_wins(self):
        r = _make_result(p1_180s_total=4, p2_180s_total=3,
                         legs=[LegRecord(leg_number=1)])
        assert _grade(r, "total_180s_band", "yes", low=5, high=7) == GradeOutcome.WIN

    def test_missing_params_void(self):
        r = _make_result()
        assert _grade(r, "total_180s_band", "yes") == GradeOutcome.VOID

    def test_low_greater_than_high_void(self):
        r = _make_result(p1_180s_total=3, p2_180s_total=3,
                         legs=[LegRecord(leg_number=1)])
        assert _grade(r, "total_180s_band", "yes", low=7, high=5) == GradeOutcome.VOID

    def test_zero_data_no_legs_void(self):
        r = _make_result(p1_180s_total=0, p2_180s_total=0, legs=[])
        assert _grade(r, "total_180s_band", "yes", low=0, high=2) == GradeOutcome.VOID


# ===========================================================================
# Global void / status tests
# ===========================================================================

class TestGlobalStatusHandling:
    def test_cancelled_all_markets_void(self):
        r = _make_result(status="Cancelled")
        markets = [
            {"market_type": "match_win", "selection": "p1", "market_params": {}},
            {"market_type": "total_legs_over", "selection": "over",
             "market_params": {"line": 8.5}},
        ]
        report = _svc.grade_match(r, markets)
        assert all(
            g.grade.outcome == GradeOutcome.VOID for g in report.grades
        )

    def test_abandoned_all_markets_void(self):
        r = _make_result(status="Abandoned")
        markets = [
            {"market_type": "match_win", "selection": "p1", "market_params": {}},
            {"market_type": "nine_dart_finish", "selection": "no", "market_params": {}},
        ]
        report = _svc.grade_match(r, markets)
        assert all(
            g.grade.outcome == GradeOutcome.VOID for g in report.grades
        )

    def test_unknown_market_type_void(self):
        r = _make_result()
        markets = [{"market_type": "alien_market", "selection": "yes",
                    "market_params": {}}]
        report = _svc.grade_match(r, markets)
        assert report.grades[0].grade.outcome == GradeOutcome.VOID

    def test_settlement_report_counts_are_correct(self):
        r = _make_result(p1_legs=6, p2_legs=4, total_legs_played=10,
                         p1_180s_total=5, p2_180s_total=2)
        markets = [
            # WIN
            {"market_type": "match_win", "selection": "p1", "market_params": {}},
            # LOSE
            {"market_type": "match_win", "selection": "p2", "market_params": {}},
            # PUSH (total=10, line=10)
            {"market_type": "total_legs_over", "selection": "over",
             "market_params": {"line": 10}},
            # VOID (cancelled — no, this is Completed; use missing line)
            {"market_type": "total_legs_over", "selection": "over",
             "market_params": {}},
        ]
        report = _svc.grade_match(r, markets)
        assert report.n_win == 1
        assert report.n_lose == 1
        assert report.n_push == 1
        assert report.n_void == 1
        assert report.n_win + report.n_lose + report.n_push + report.n_void == 4

    def test_empty_markets_produces_empty_report(self):
        r = _make_result()
        report = _svc.grade_match(r, [])
        assert report.grades == []
        assert report.n_win == 0
