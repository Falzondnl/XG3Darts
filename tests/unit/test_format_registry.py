"""
Tests for competition/format_registry.py

Sprint 1 target: verify all 22 formats load correctly with correct structural rules.
"""
from __future__ import annotations

import pytest

from competition.format_registry import (
    DartsCompetitionFormat,
    DartsFormatError,
    DartsRoundFormat,
    get_all_formats,
    get_format,
    list_formats,
)

# ---------------------------------------------------------------------------
# Expected format codes
# ---------------------------------------------------------------------------

EXPECTED_CODES = {
    "PDC_WC",
    "PDC_WC_ERA_2020",
    "PDC_PL",
    "PDC_WM",
    "PDC_GP",
    "PDC_UK",
    "PDC_GS",
    "PDC_ET",
    "PDC_PC",
    "PDC_PCF",
    "PDC_MASTERS",
    "PDC_WS",
    "PDC_WCUP",
    "PDC_WOM_SERIES",
    "PDC_WWM",
    "PDC_DEVTOUR",
    "PDC_CHALLENGE",
    "PDC_WYC",
    "WDF_WC",
    "WDF_EC",
    "WDF_OPEN",
    "TEAM_701",
}


class TestRegistryCompleteness:
    """All 22 formats must be registered."""

    def test_all_expected_codes_present(self) -> None:
        """Every expected format code is in the registry."""
        registered = set(list_formats())
        missing = EXPECTED_CODES - registered
        assert not missing, f"Missing format codes: {sorted(missing)}"

    def test_exactly_22_formats(self) -> None:
        """There are exactly 22 registered formats."""
        assert len(list_formats()) == 22

    def test_get_all_formats_returns_dict(self) -> None:
        """get_all_formats() returns a dict of all 22 entries."""
        all_fmts = get_all_formats()
        assert isinstance(all_fmts, dict)
        assert len(all_fmts) == 22

    def test_get_unknown_raises(self) -> None:
        """Requesting an unknown code raises DartsFormatError."""
        with pytest.raises(DartsFormatError):
            get_format("DOES_NOT_EXIST")


class TestPdcWorldChampionship:
    """PDC_WC format structural rules."""

    def test_starting_score(self) -> None:
        fmt = get_format("PDC_WC")
        assert fmt.starting_score == 501

    def test_no_double_start(self) -> None:
        fmt = get_format("PDC_WC")
        assert fmt.double_start_required is False

    def test_field_size(self) -> None:
        fmt = get_format("PDC_WC")
        assert fmt.field_size == 96

    def test_final_is_sets_format(self) -> None:
        fmt = get_format("PDC_WC")
        final = fmt.get_round("Final")
        assert final.is_sets_format is True
        assert final.sets_to_win == 7
        assert final.legs_per_set == 3

    def test_preliminary_is_legs_only(self) -> None:
        fmt = get_format("PDC_WC")
        prelim = fmt.get_round("Preliminary")
        assert prelim.is_sets_format is False
        assert prelim.legs_to_win == 2

    def test_no_draws(self) -> None:
        fmt = get_format("PDC_WC")
        assert fmt.allows_draw() is False
        assert "draw" not in fmt.result_types

    def test_era_2020_field_size_128(self) -> None:
        fmt = get_format("PDC_WC_ERA_2020")
        assert fmt.field_size == 128

    def test_unknown_round_raises(self) -> None:
        fmt = get_format("PDC_WC")
        with pytest.raises(DartsFormatError):
            fmt.get_round("NonExistentRound")


class TestPdcPremierLeague:
    """PDC_PL draw rules and round structure."""

    def test_draw_enabled_in_league_night(self) -> None:
        fmt = get_format("PDC_PL")
        rnd = fmt.get_round("League Night")
        assert rnd.draw_enabled is True
        assert rnd.group_stage is True

    def test_legs_to_win_league(self) -> None:
        fmt = get_format("PDC_PL")
        rnd = fmt.get_round("League Night")
        assert rnd.legs_to_win == 7

    def test_allows_draw(self) -> None:
        fmt = get_format("PDC_PL")
        assert fmt.allows_draw() is True
        assert "draw" in fmt.result_types

    def test_playoff_no_draw(self) -> None:
        fmt = get_format("PDC_PL")
        rnd = fmt.get_round("Play-Off Final")
        assert rnd.draw_enabled is False


class TestPdcWorldMatchplay:
    """PDC_WM two-clear-legs rule in the final."""

    def test_final_two_clear_legs(self) -> None:
        fmt = get_format("PDC_WM")
        final = fmt.get_round("Final")
        assert final.two_clear_legs is True
        assert final.two_clear_legs_max_extra is None

    def test_legs_only_format(self) -> None:
        fmt = get_format("PDC_WM")
        for round_name in ["Round 1", "Quarter-Final", "Semi-Final", "Final"]:
            rnd = fmt.get_round(round_name)
            assert rnd.is_sets_format is False


class TestPdcGrandPrix:
    """PDC_GP double-in required."""

    def test_double_start_required(self) -> None:
        fmt = get_format("PDC_GP")
        assert fmt.double_start_required is True

    def test_round1_is_legs_only(self) -> None:
        fmt = get_format("PDC_GP")
        r1 = fmt.get_round("Round 1")
        assert r1.is_sets_format is False
        assert r1.legs_to_win == 3


class TestPdcGrandSlam:
    """PDC_GS group stage draw and knockout structure."""

    def test_group_stage_draws(self) -> None:
        fmt = get_format("PDC_GS")
        gs = fmt.get_round("Group Stage")
        assert gs.draw_enabled is True
        assert gs.group_stage is True

    def test_knockout_no_draw(self) -> None:
        fmt = get_format("PDC_GS")
        rnd = fmt.get_round("Final")
        assert rnd.draw_enabled is False


class TestWorldCupDoubles:
    """PDC_WCUP doubles format detection."""

    def test_ecosystem_team_doubles(self) -> None:
        fmt = get_format("PDC_WCUP")
        assert fmt.ecosystem == "team_doubles"

    def test_doubles_rounds_have_correct_format_type(self) -> None:
        fmt = get_format("PDC_WCUP")
        doubles_rnd = fmt.get_round("Final Doubles")
        assert doubles_rnd.format_type == "doubles"

    def test_singles_rounds_are_singles(self) -> None:
        fmt = get_format("PDC_WCUP")
        singles_rnd = fmt.get_round("Final Singles")
        assert singles_rnd.format_type == "singles"


class TestTeam701:
    """TEAM_701 starting score = 701."""

    def test_starting_score_701(self) -> None:
        fmt = get_format("TEAM_701")
        assert fmt.starting_score == 701

    def test_doubles_format(self) -> None:
        fmt = get_format("TEAM_701")
        assert fmt.ecosystem == "team_doubles"


class TestWdfFormats:
    """WDF format structures."""

    def test_wdf_wc_sets_format(self) -> None:
        fmt = get_format("WDF_WC")
        final = fmt.get_round("Final")
        assert final.is_sets_format is True
        assert final.sets_to_win == 5

    def test_wdf_open_legs_format(self) -> None:
        fmt = get_format("WDF_OPEN")
        final = fmt.get_round("Final")
        assert final.is_sets_format is False
        assert final.legs_to_win == 5


class TestRoundFormatValidation:
    """DartsRoundFormat validation logic."""

    def test_must_have_legs_or_sets(self) -> None:
        with pytest.raises(DartsFormatError):
            DartsRoundFormat(name="Bad Round")

    def test_cannot_have_both(self) -> None:
        with pytest.raises(DartsFormatError):
            DartsRoundFormat(
                name="Bad Round",
                legs_to_win=6,
                sets_to_win=3,
                legs_per_set=3,
            )

    def test_invalid_format_type(self) -> None:
        with pytest.raises(DartsFormatError):
            DartsRoundFormat(name="Test", legs_to_win=6, format_type="triples")

    def test_total_legs_possible(self) -> None:
        rnd = DartsRoundFormat(name="Test", legs_to_win=6)
        assert rnd.total_legs_possible == 11

    def test_total_sets_possible(self) -> None:
        rnd = DartsRoundFormat(name="Test", sets_to_win=4, legs_per_set=3)
        assert rnd.total_sets_possible == 7
