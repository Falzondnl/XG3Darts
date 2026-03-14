"""
Unit tests for the state layer.

Tests:
  - test_score_state_context_bands
  - test_score_state_apply_visit_normal
  - test_score_state_apply_visit_bust
  - test_score_state_terminal
  - test_score_state_can_checkout
  - test_throw_state_advance
  - test_pressure_detector_levels
  - test_visit_result_validation
"""

from __future__ import annotations

import pytest

from engines.state_layer.score_state import (
    DartsEngineError,
    ScoreState,
    ThrowContext,
)
from engines.state_layer.throw_state import (
    DartThrowState,
    PressureDetector,
    PressureLevel,
    VisitResult,
)


# ---------------------------------------------------------------------------
# ScoreState tests
# ---------------------------------------------------------------------------


class TestScoreStateContextBands:
    """G: context bands are assigned correctly for all score ranges."""

    def test_open_band_high_score(self):
        """Scores > 300 should be OPEN context."""
        assert ScoreState(score=501).context == ThrowContext.OPEN
        assert ScoreState(score=400).context == ThrowContext.OPEN
        assert ScoreState(score=301).context == ThrowContext.OPEN

    def test_middle_band(self):
        """Scores 171-300 should be MIDDLE context."""
        assert ScoreState(score=300).context == ThrowContext.MIDDLE
        assert ScoreState(score=250).context == ThrowContext.MIDDLE
        assert ScoreState(score=171).context == ThrowContext.MIDDLE

    def test_setup_band(self):
        """Scores 100-170 should be SETUP context."""
        assert ScoreState(score=170).context == ThrowContext.SETUP
        assert ScoreState(score=140).context == ThrowContext.SETUP
        assert ScoreState(score=100).context == ThrowContext.SETUP

    def test_finish_band(self):
        """Scores 41-99 should be FINISH context."""
        assert ScoreState(score=99).context == ThrowContext.FINISH
        assert ScoreState(score=60).context == ThrowContext.FINISH
        assert ScoreState(score=41).context == ThrowContext.FINISH

    def test_pressure_band(self):
        """Scores 0-40 should be PRESSURE context."""
        assert ScoreState(score=40).context == ThrowContext.PRESSURE
        assert ScoreState(score=20).context == ThrowContext.PRESSURE
        assert ScoreState(score=2).context == ThrowContext.PRESSURE
        assert ScoreState(score=0).context == ThrowContext.PRESSURE  # terminal, but band is pressure

    def test_all_bands_cover_full_range(self):
        """Every score from 0 to 501 should be assigned a context."""
        for score in range(0, 502):
            state = ScoreState(score=score)
            assert state.context in list(ThrowContext), f"Score {score} has no context"

    def test_boundary_501(self):
        """Score 501 (starting state) is OPEN."""
        assert ScoreState(score=501).context == ThrowContext.OPEN

    def test_boundary_transition_300_301(self):
        """Exact band boundaries: 301=OPEN, 300=MIDDLE."""
        assert ScoreState(score=301).context == ThrowContext.OPEN
        assert ScoreState(score=300).context == ThrowContext.MIDDLE

    def test_boundary_transition_170_171(self):
        """171=MIDDLE, 170=SETUP."""
        assert ScoreState(score=171).context == ThrowContext.MIDDLE
        assert ScoreState(score=170).context == ThrowContext.SETUP

    def test_boundary_transition_99_100(self):
        """100=SETUP, 99=FINISH."""
        assert ScoreState(score=100).context == ThrowContext.SETUP
        assert ScoreState(score=99).context == ThrowContext.FINISH

    def test_boundary_transition_40_41(self):
        """41=FINISH, 40=PRESSURE."""
        assert ScoreState(score=41).context == ThrowContext.FINISH
        assert ScoreState(score=40).context == ThrowContext.PRESSURE


class TestScoreStateApplyVisit:
    """G: apply_visit transitions are correct."""

    def test_apply_visit_normal(self):
        """Normal visit reduces score correctly."""
        state = ScoreState(score=501)
        result = state.apply_visit(visit_score=60, bust=False)
        assert result.score == 441

    def test_apply_visit_to_zero(self):
        """Visit that reaches exactly 0 results in terminal state."""
        state = ScoreState(score=40)
        result = state.apply_visit(visit_score=40, bust=False)
        assert result.score == 0
        assert result.is_terminal

    def test_apply_visit_large_score(self):
        """Apply a large visit to a large score."""
        state = ScoreState(score=501)
        result = state.apply_visit(visit_score=180, bust=False)
        assert result.score == 321

    def test_apply_visit_bust(self):
        """Bust returns a state with the same score (score unchanged)."""
        state = ScoreState(score=32)
        result = state.apply_visit(visit_score=0, bust=True)
        assert result.score == 32  # unchanged
        # Note: frozen dataclasses with same value may be the same object (interned)
        assert result.score == state.score

    def test_apply_visit_bust_score_preserved(self):
        """Even with bust=True and non-zero visit_score, state is unchanged."""
        state = ScoreState(score=40)
        result = state.apply_visit(visit_score=999, bust=True)
        assert result.score == 40

    def test_apply_visit_raises_on_negative(self):
        """Negative resulting score raises DartsEngineError."""
        state = ScoreState(score=20)
        with pytest.raises(DartsEngineError, match="Negative score"):
            state.apply_visit(visit_score=50, bust=False)

    def test_apply_visit_sequential(self):
        """Multiple sequential visits reduce score correctly."""
        state = ScoreState(score=501)
        state = state.apply_visit(60, False)   # 501 - 60 = 441
        state = state.apply_visit(60, False)   # 441 - 60 = 381
        state = state.apply_visit(180, False)  # 381 - 180 = 201
        assert state.score == 201

    def test_apply_visit_immutable(self):
        """ScoreState is frozen; apply_visit returns a new instance."""
        state = ScoreState(score=200)
        result = state.apply_visit(50, False)
        assert state.score == 200  # original unchanged
        assert result.score == 150


class TestScoreStateTerminal:
    """G: terminal state detection."""

    def test_score_zero_is_terminal(self):
        """Score 0 is terminal (leg won)."""
        assert ScoreState(score=0).is_terminal

    def test_score_one_is_terminal(self):
        """Score 1 is terminal (impossible checkout — no double = 1)."""
        assert ScoreState(score=1).is_terminal

    def test_active_states_not_terminal(self):
        """Scores 2..501 are active (not terminal)."""
        for score in range(2, 502):
            assert not ScoreState(score=score).is_terminal, f"Score {score} falsely terminal"

    def test_invalid_score_raises(self):
        """Score outside 0..501 raises DartsEngineError."""
        with pytest.raises(DartsEngineError):
            ScoreState(score=-1)
        with pytest.raises(DartsEngineError):
            ScoreState(score=502)


class TestScoreStateCanCheckout:
    """G: checkout possibility detection."""

    def test_zero_always_checkable(self):
        """Score 0 (already done) is trivially checkable."""
        assert ScoreState(score=0).can_checkout(3)
        assert ScoreState(score=0).can_checkout(1)

    def test_score_one_not_checkable(self):
        """Score 1 is never checkable (no double = 1)."""
        assert not ScoreState(score=1).can_checkout(3)
        assert not ScoreState(score=1).can_checkout(1)

    def test_max_3_dart_checkout(self):
        """Score 170 is checkable in 3 darts (T20+T20+Bull)."""
        assert ScoreState(score=170).can_checkout(3)

    def test_score_171_not_3_dart_checkout(self):
        """Score 171 is not checkable in 3 darts."""
        assert not ScoreState(score=171).can_checkout(3)

    def test_max_2_dart_checkout(self):
        """Score 110 is the max 2-dart checkout (T20+D25)."""
        assert ScoreState(score=110).can_checkout(2)
        assert not ScoreState(score=111).can_checkout(2)

    def test_max_1_dart_checkout(self):
        """Score 50 (bull) is the max 1-dart checkout."""
        assert ScoreState(score=50).can_checkout(1)
        assert not ScoreState(score=51).can_checkout(1)

    def test_odd_scores_not_1_dart_checkout(self):
        """Odd scores require at least 2 darts (need final double)."""
        assert not ScoreState(score=3).can_checkout(1)
        assert not ScoreState(score=5).can_checkout(1)
        assert not ScoreState(score=49).can_checkout(1)

    def test_even_low_scores_1_dart_checkout(self):
        """Even scores <= 40 are 1-dart checkouts (doubles)."""
        for s in range(2, 42, 2):
            assert ScoreState(score=s).can_checkout(1), f"Score {s} should be 1-dart checkout"


# ---------------------------------------------------------------------------
# DartThrowState tests
# ---------------------------------------------------------------------------


class TestDartThrowState:
    """G: dart-level throw state transitions."""

    def test_advance_normal(self):
        """Advance by scoring a dart normally."""
        state = DartThrowState(
            dart_number=1,
            score_before_visit=501,
            darts_scored_this_visit=0,
            player_index=0,
        )
        new_state = state.advance(dart_score=60, bust=False)
        assert new_state.dart_number == 2
        assert new_state.darts_scored_this_visit == 60
        assert new_state.current_score == 441

    def test_advance_bust(self):
        """Advance on a bust: score unchanged, dart_number increments."""
        state = DartThrowState(
            dart_number=2,
            score_before_visit=32,
            darts_scored_this_visit=0,
            player_index=0,
        )
        new_state = state.advance(dart_score=0, bust=True)
        assert new_state.dart_number == 3
        assert new_state.darts_scored_this_visit == 0  # unchanged on bust

    def test_current_score_calculation(self):
        """current_score accounts for darts already thrown this visit."""
        state = DartThrowState(
            dart_number=3,
            score_before_visit=180,
            darts_scored_this_visit=120,  # two T20s already
            player_index=1,
        )
        assert state.current_score == 60  # 180 - 120

    def test_can_finish_this_visit(self):
        """can_finish_this_visit checks if checkout is possible with remaining darts."""
        # On dart 3 (1 remaining), score 40 → D20 (1-dart checkout)
        state = DartThrowState(
            dart_number=3,
            score_before_visit=40,
            darts_scored_this_visit=0,
            player_index=0,
        )
        assert state.can_finish_this_visit()

        # On dart 1 (3 remaining), score 170 → checkable
        state2 = DartThrowState(
            dart_number=1,
            score_before_visit=170,
            darts_scored_this_visit=0,
            player_index=0,
        )
        assert state2.can_finish_this_visit()

        # On dart 3 (1 remaining), score 100 → NOT checkable in 1 dart
        state3 = DartThrowState(
            dart_number=3,
            score_before_visit=100,
            darts_scored_this_visit=0,
            player_index=0,
        )
        assert not state3.can_finish_this_visit()

    def test_invalid_dart_number(self):
        """dart_number must be 1, 2, or 3."""
        with pytest.raises(ValueError):
            DartThrowState(dart_number=0, score_before_visit=501, darts_scored_this_visit=0, player_index=0)
        with pytest.raises(ValueError):
            DartThrowState(dart_number=4, score_before_visit=501, darts_scored_this_visit=0, player_index=0)

    def test_invalid_player_index(self):
        """player_index must be 0 or 1."""
        with pytest.raises(ValueError):
            DartThrowState(dart_number=1, score_before_visit=501, darts_scored_this_visit=0, player_index=2)


class TestPressureDetector:
    """G: pressure level detection."""

    def setup_method(self):
        self.detector = PressureDetector()

    def test_critical_last_dart_to_win(self):
        """Last dart to win the match is CRITICAL pressure."""
        state = DartThrowState(
            dart_number=3, score_before_visit=40, darts_scored_this_visit=0, player_index=0
        )
        pressure = self.detector.compute_pressure(
            throw_state=state,
            legs_needed_by_thrower=1,
            legs_needed_by_opponent=2,
            opponent_current_score=400,
            is_deciding_leg=True,
        )
        assert pressure == PressureLevel.CRITICAL

    def test_none_pressure_open_routine(self):
        """Routine high-score throw has no elevated pressure."""
        state = DartThrowState(
            dart_number=1, score_before_visit=501, darts_scored_this_visit=0, player_index=0
        )
        pressure = self.detector.compute_pressure(
            throw_state=state,
            legs_needed_by_thrower=5,
            legs_needed_by_opponent=3,
            opponent_current_score=450,
            is_deciding_leg=False,
        )
        assert pressure == PressureLevel.NONE


class TestVisitResult:
    """G: VisitResult validation."""

    def test_valid_checkout(self):
        """Valid checkout result."""
        result = VisitResult(
            score_achieved=40,
            is_bust=False,
            darts_used=2,
            checkout_achieved=True,
            score_after=0,
        )
        assert result.checkout_achieved
        assert not result.is_bust

    def test_bust_must_have_zero_score(self):
        """Bust visits must have score_achieved=0."""
        with pytest.raises(ValueError, match="Bust visits must have score_achieved=0"):
            VisitResult(
                score_achieved=20,  # non-zero on bust
                is_bust=True,
                darts_used=2,
                checkout_achieved=False,
                score_after=50,
            )

    def test_checkout_must_have_zero_after(self):
        """Checkout visits must have score_after=0."""
        with pytest.raises(ValueError, match="Checkout visits must have score_after=0"):
            VisitResult(
                score_achieved=40,
                is_bust=False,
                darts_used=2,
                checkout_achieved=True,
                score_after=1,  # non-zero after checkout
            )
