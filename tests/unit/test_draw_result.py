"""
Tests for competition/draw_result.py

Verifies draw detection, ELO scoring coefficients, and result classification
for Premier League, Grand Slam, and standard knockout formats.
"""
from __future__ import annotations

import pytest

from competition.draw_result import (
    RESULT_DRAW,
    RESULT_P1_WIN,
    RESULT_P2_WIN,
    DartsMatchResult,
    DartsResultError,
    classify_result_from_scores,
    elo_score_for_result,
    is_draw_possible,
)


class TestEloScoreForResult:
    """ELO score coefficients for all result types."""

    def test_p1_win_returns_1_0(self) -> None:
        s1, s2 = elo_score_for_result(RESULT_P1_WIN)
        assert s1 == 1.0
        assert s2 == 0.0

    def test_p2_win_returns_0_1(self) -> None:
        s1, s2 = elo_score_for_result(RESULT_P2_WIN)
        assert s1 == 0.0
        assert s2 == 1.0

    def test_draw_returns_half_half(self) -> None:
        s1, s2 = elo_score_for_result(RESULT_DRAW)
        assert s1 == 0.5
        assert s2 == 0.5

    def test_invalid_result_type_raises(self) -> None:
        with pytest.raises(DartsResultError):
            elo_score_for_result("unknown_result")

    def test_scores_sum_to_one_for_all_types(self) -> None:
        for result in (RESULT_P1_WIN, RESULT_P2_WIN, RESULT_DRAW):
            s1, s2 = elo_score_for_result(result)
            assert abs(s1 + s2 - 1.0) < 1e-9


class TestDrawPossibility:
    """is_draw_possible() for different formats."""

    def test_premier_league_allows_draw(self) -> None:
        assert is_draw_possible("PDC_PL") is True

    def test_grand_slam_allows_draw(self) -> None:
        assert is_draw_possible("PDC_GS") is True

    def test_world_championship_no_draw(self) -> None:
        assert is_draw_possible("PDC_WC") is False

    def test_world_matchplay_no_draw(self) -> None:
        assert is_draw_possible("PDC_WM") is False

    def test_grand_prix_no_draw(self) -> None:
        assert is_draw_possible("PDC_GP") is False


class TestDartsMatchResult:
    """DartsMatchResult construction and validation."""

    def test_valid_p1_win(self) -> None:
        result = DartsMatchResult(
            result_type=RESULT_P1_WIN,
            p1_score=7,
            p2_score=3,
            format_code="PDC_PL",
            round_name="League Night",
        )
        assert result.result_type == RESULT_P1_WIN
        assert result.winner_index == 1
        assert result.is_draw is False

    def test_valid_draw_in_pl(self) -> None:
        result = DartsMatchResult(
            result_type=RESULT_DRAW,
            p1_score=6,
            p2_score=6,
            format_code="PDC_PL",
            round_name="League Night",
        )
        assert result.result_type == RESULT_DRAW
        assert result.winner_index is None
        assert result.is_draw is True

    def test_draw_in_wc_raises(self) -> None:
        """Draws are not valid in World Championship."""
        with pytest.raises(DartsResultError):
            DartsMatchResult(
                result_type=RESULT_DRAW,
                p1_score=3,
                p2_score=3,
                format_code="PDC_WC",
                round_name="Final",
            )

    def test_inconsistent_p1_win_score_raises(self) -> None:
        """p1_win declared but p1_score ≤ p2_score."""
        with pytest.raises(DartsResultError):
            DartsMatchResult(
                result_type=RESULT_P1_WIN,
                p1_score=3,
                p2_score=7,
                format_code="PDC_PL",
                round_name="League Night",
            )

    def test_draw_with_unequal_scores_raises(self) -> None:
        with pytest.raises(DartsResultError):
            DartsMatchResult(
                result_type=RESULT_DRAW,
                p1_score=5,
                p2_score=4,
                format_code="PDC_PL",
                round_name="League Night",
            )

    def test_negative_score_raises(self) -> None:
        with pytest.raises(DartsResultError):
            DartsMatchResult(
                result_type=RESULT_P1_WIN,
                p1_score=-1,
                p2_score=0,
                format_code="PDC_PL",
                round_name="League Night",
            )


class TestClassifyResultFromScores:
    """classify_result_from_scores() correctness."""

    def test_p1_win_classification(self) -> None:
        result = classify_result_from_scores(7, 5, "PDC_PL", "League Night")
        assert result.result_type == RESULT_P1_WIN

    def test_p2_win_classification(self) -> None:
        result = classify_result_from_scores(4, 10, "PDC_WM", "Final")
        assert result.result_type == RESULT_P2_WIN

    def test_draw_classification_in_pl(self) -> None:
        result = classify_result_from_scores(6, 6, "PDC_PL", "League Night")
        assert result.result_type == RESULT_DRAW

    def test_equal_scores_in_knockout_raises(self) -> None:
        with pytest.raises(DartsResultError):
            classify_result_from_scores(3, 3, "PDC_WC", "Final")

    def test_grand_slam_group_draw(self) -> None:
        result = classify_result_from_scores(4, 4, "PDC_GS", "Group Stage")
        assert result.result_type == RESULT_DRAW
