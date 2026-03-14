"""
Tests for ML regime selection.

Tests that the regime selector correctly picks R0/R1/R2 based on
player data availability, and uses the minimum of the two players.
"""
from __future__ import annotations

import pytest

from data.coverage_regime import CoverageSignals
from models import DartsMLError
from models.regime_selector import DartsMLRegime, RegimeSelector


def _signals(
    player_id: str = "player_001",
    has_dartconnect: bool = False,
    has_mastercaller: bool = False,
    has_dartsorakel: bool = False,
    has_pdc_stats: bool = False,
    dc_legs: int = 0,
    mc_legs: int = 0,
    match_count: int = 0,
) -> CoverageSignals:
    return CoverageSignals(
        player_id=player_id,
        competition_id=None,
        has_dartconnect_data=has_dartconnect,
        has_mastercaller_data=has_mastercaller,
        has_dartsorakel_stats=has_dartsorakel,
        has_pdc_match_stats=has_pdc_stats,
        dartconnect_leg_count=dc_legs,
        mastercaller_leg_count=mc_legs,
        match_count=match_count,
    )


class TestR0WhenNoStats:
    """Players with no statistics should get R0."""

    def test_r0_when_no_stats(self) -> None:
        selector = RegimeSelector()
        p1 = _signals(player_id="p1")
        p2 = _signals(player_id="p2")

        regime = selector.select_regime_from_signals(p1, p2)
        assert regime == DartsMLRegime.R0

    def test_r0_integer_value(self) -> None:
        assert DartsMLRegime.R0 == 0

    def test_r0_with_some_matches(self) -> None:
        """Players with match count but no stats are still R0."""
        selector = RegimeSelector()
        p1 = _signals(player_id="p1", match_count=50)
        p2 = _signals(player_id="p2", match_count=30)

        regime = selector.select_regime_from_signals(p1, p2)
        assert regime == DartsMLRegime.R0


class TestR1WhenOrakelStats:
    """Players with DartsOrakel stats should get R1."""

    def test_r1_when_orakel_stats(self) -> None:
        selector = RegimeSelector()
        p1 = _signals(player_id="p1", has_dartsorakel=True, match_count=10)
        p2 = _signals(player_id="p2", has_dartsorakel=True, match_count=10)

        regime = selector.select_regime_from_signals(p1, p2)
        assert regime == DartsMLRegime.R1

    def test_r1_integer_value(self) -> None:
        assert DartsMLRegime.R1 == 1


class TestR2WhenDartConnectLinked:
    """Players with DartConnect visit data should get R2."""

    def test_r2_when_dartconnect_linked(self) -> None:
        selector = RegimeSelector()
        p1 = _signals(
            player_id="p1", has_dartconnect=True, dc_legs=20, match_count=10,
        )
        p2 = _signals(
            player_id="p2", has_dartconnect=True, dc_legs=15, match_count=10,
        )

        regime = selector.select_regime_from_signals(p1, p2)
        assert regime == DartsMLRegime.R2

    def test_r2_integer_value(self) -> None:
        assert DartsMLRegime.R2 == 2


class TestRegimeMinimumOfTwoPlayers:
    """The selected regime must be the minimum of the two players."""

    def test_regime_minimum_of_two_players(self) -> None:
        """If P1 qualifies for R2 but P2 only R0, result should be R0."""
        selector = RegimeSelector()
        p1 = _signals(
            player_id="p1", has_dartconnect=True, dc_legs=50, match_count=20,
        )
        p2 = _signals(player_id="p2")  # R0 only

        regime = selector.select_regime_from_signals(p1, p2)
        assert regime == DartsMLRegime.R0

    def test_regime_min_r2_r1(self) -> None:
        """If P1 is R2 and P2 is R1, result should be R1."""
        selector = RegimeSelector()
        p1 = _signals(
            player_id="p1", has_dartconnect=True, dc_legs=50, match_count=20,
        )
        p2 = _signals(
            player_id="p2", has_dartsorakel=True, match_count=10,
        )

        regime = selector.select_regime_from_signals(p1, p2)
        assert regime == DartsMLRegime.R1

    def test_regime_symmetric(self) -> None:
        """Order of players should not matter."""
        selector = RegimeSelector()
        p1_r2 = _signals(
            player_id="p1", has_dartconnect=True, dc_legs=50, match_count=20,
        )
        p2_r0 = _signals(player_id="p2")

        regime_a = selector.select_regime_from_signals(p1_r2, p2_r0)
        regime_b = selector.select_regime_from_signals(p2_r0, p1_r2)
        assert regime_a == regime_b


class TestDartsMLRegimeEnum:
    """Test DartsMLRegime enum behaviour."""

    def test_regime_ordering(self) -> None:
        assert DartsMLRegime.R0 < DartsMLRegime.R1
        assert DartsMLRegime.R1 < DartsMLRegime.R2

    def test_from_string(self) -> None:
        assert DartsMLRegime.from_string("R0") == DartsMLRegime.R0
        assert DartsMLRegime.from_string("R1") == DartsMLRegime.R1
        assert DartsMLRegime.from_string("R2") == DartsMLRegime.R2

    def test_from_string_invalid_raises(self) -> None:
        with pytest.raises(DartsMLError, match="Unknown regime"):
            DartsMLRegime.from_string("R3")

    def test_min_of_regimes(self) -> None:
        assert min(DartsMLRegime.R2, DartsMLRegime.R0) == DartsMLRegime.R0
        assert min(DartsMLRegime.R1, DartsMLRegime.R2) == DartsMLRegime.R1


class TestRegimeSelectorCache:
    """Test regime selector caching."""

    def test_cache_clear(self) -> None:
        selector = RegimeSelector()
        # Populate cache via select_regime (no store -> defaults to R0)
        regime = selector.select_regime("p1", "p2", "comp1")
        assert regime == DartsMLRegime.R0
        assert len(selector._cache) > 0

        selector.clear_cache()
        assert len(selector._cache) == 0
