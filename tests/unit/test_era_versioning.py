"""
Tests for competition/era_versioning.py

Verifies 96-player vs 128-player era detection and format resolution
for all registered competition base names.
"""
from __future__ import annotations

import pytest

from competition.era_versioning import (
    DartsEraError,
    EraWindow,
    get_field_size,
    list_era_windows,
    list_versioned_competitions,
    resolve_format,
)
from competition.format_registry import DartsCompetitionFormat


class TestPdcWcEraResolution:
    """PDC World Championship era versioning."""

    def test_2024_resolves_to_pdc_wc(self) -> None:
        """2024 season → 96-player format."""
        fmt = resolve_format("PDC_WC", 2024)
        assert fmt.code == "PDC_WC"
        assert fmt.field_size == 96

    def test_2020_resolves_to_pdc_wc(self) -> None:
        """2020 is the first year of the 96-player era."""
        fmt = resolve_format("PDC_WC", 2020)
        assert fmt.code == "PDC_WC"

    def test_2019_resolves_to_era_2020(self) -> None:
        """2019 season → 128-player era."""
        fmt = resolve_format("PDC_WC", 2019)
        assert fmt.code == "PDC_WC_ERA_2020"
        assert fmt.field_size == 128

    def test_2003_resolves_to_era_2020(self) -> None:
        """First year of 128-player era."""
        fmt = resolve_format("PDC_WC", 2003)
        assert fmt.code == "PDC_WC_ERA_2020"

    def test_1994_resolves_to_era_2020(self) -> None:
        """Earliest season uses pre-2020 format."""
        fmt = resolve_format("PDC_WC", 1994)
        assert fmt.code == "PDC_WC_ERA_2020"

    def test_return_type_is_competition_format(self) -> None:
        fmt = resolve_format("PDC_WC", 2023)
        assert isinstance(fmt, DartsCompetitionFormat)


class TestFieldSizeLookup:
    """get_field_size() returns correct draw sizes."""

    def test_field_size_96_for_recent(self) -> None:
        assert get_field_size("PDC_WC", 2022) == 96

    def test_field_size_128_for_pre_2020(self) -> None:
        assert get_field_size("PDC_WC", 2010) == 128

    def test_year_before_earliest_raises(self) -> None:
        with pytest.raises(DartsEraError):
            get_field_size("PDC_WC", 1990)  # before 1994


class TestEraWindowMethods:
    """EraWindow.contains_year() boundary conditions."""

    def test_contains_year_start_inclusive(self) -> None:
        window = EraWindow(
            format_code="PDC_WC",
            start_year=2020,
            end_year=None,
        )
        assert window.contains_year(2020) is True

    def test_contains_year_before_start(self) -> None:
        window = EraWindow(
            format_code="PDC_WC",
            start_year=2020,
            end_year=None,
        )
        assert window.contains_year(2019) is False

    def test_contains_year_at_end_exclusive(self) -> None:
        window = EraWindow(
            format_code="PDC_WC_ERA_2020",
            start_year=1994,
            end_year=2020,
        )
        assert window.contains_year(2020) is False
        assert window.contains_year(2019) is True


class TestRegisteredVersionedCompetitions:
    """All registered base names can be queried."""

    def test_pdc_wc_is_registered(self) -> None:
        assert "PDC_WC" in list_versioned_competitions()

    def test_pdc_pl_is_registered(self) -> None:
        assert "PDC_PL" in list_versioned_competitions()

    def test_list_era_windows_returns_list(self) -> None:
        windows = list_era_windows("PDC_WC")
        assert isinstance(windows, list)
        assert len(windows) >= 2

    def test_unknown_base_raises(self) -> None:
        with pytest.raises(DartsEraError):
            resolve_format("UNKNOWN_COMPETITION", 2024)
