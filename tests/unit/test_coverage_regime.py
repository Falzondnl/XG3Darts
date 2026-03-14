"""
Tests for data/coverage_regime.py

Verifies R0/R1/R2 regime detection based on available data signals.
"""
from __future__ import annotations

import pytest

from data.coverage_regime import (
    REGIME_R0,
    REGIME_R1,
    REGIME_R2,
    CoverageSignals,
    DartsCoverageError,
    RegimeResult,
    batch_detect_regimes,
    detect_regime,
    regime_from_db_flags,
)


def _signals(
    *,
    player_id: str = "test_player",
    competition_id: str | None = None,
    has_dartconnect: bool = False,
    has_mastercaller: bool = False,
    has_dartsorakel: bool = False,
    has_pdc_stats: bool = False,
    dc_legs: int = 0,
    mc_legs: int = 0,
    da: float | None = None,
    match_count: int = 0,
) -> CoverageSignals:
    return CoverageSignals(
        player_id=player_id,
        competition_id=competition_id,
        has_dartconnect_data=has_dartconnect,
        has_mastercaller_data=has_mastercaller,
        has_dartsorakel_stats=has_dartsorakel,
        has_pdc_match_stats=has_pdc_stats,
        dartconnect_leg_count=dc_legs,
        mastercaller_leg_count=mc_legs,
        dartsorakel_3da=da,
        match_count=match_count,
    )


class TestR0Detection:
    """Only result data available."""

    def test_no_data_is_r0(self) -> None:
        sig = _signals()
        result = detect_regime(sig)
        assert result.regime == REGIME_R0

    def test_r0_has_result_only_flag(self) -> None:
        sig = _signals()
        result = detect_regime(sig)
        assert result.has_result_only is True
        assert result.has_visit_data is False
        assert result.has_match_stats is False

    def test_low_match_count_r0(self) -> None:
        sig = _signals(match_count=3)
        result = detect_regime(sig)
        assert result.regime == REGIME_R0

    def test_r0_regime_score_in_range(self) -> None:
        sig = _signals(match_count=25)
        result = detect_regime(sig)
        assert 0.0 <= result.regime_score < 1.0


class TestR1Detection:
    """Match-level statistics available."""

    def test_dartsorakel_stats_gives_r1(self) -> None:
        sig = _signals(
            has_dartsorakel=True,
            da=98.5,
            match_count=10,
        )
        result = detect_regime(sig)
        assert result.regime == REGIME_R1

    def test_r1_has_match_stats_flag(self) -> None:
        sig = _signals(has_dartsorakel=True, da=95.0, match_count=10)
        result = detect_regime(sig)
        assert result.has_match_stats is True
        assert result.has_dartsorakel is True

    def test_r1_regime_score_in_range(self) -> None:
        sig = _signals(has_dartsorakel=True, da=100.0, match_count=10)
        result = detect_regime(sig)
        assert 0.0 < result.regime_score <= 1.0

    def test_r1_reasoning_mentions_dartsorakel(self) -> None:
        sig = _signals(has_dartsorakel=True, da=99.0, match_count=5)
        result = detect_regime(sig)
        assert "DartsOrakel" in result.reasoning or "R1" in result.reasoning


class TestR2Detection:
    """Full visit-level data available."""

    def test_sufficient_dartconnect_legs_gives_r2(self) -> None:
        sig = _signals(has_dartconnect=True, dc_legs=20, match_count=10)
        result = detect_regime(sig)
        assert result.regime == REGIME_R2

    def test_sufficient_mastercaller_legs_gives_r2(self) -> None:
        sig = _signals(has_mastercaller=True, mc_legs=15, match_count=8)
        result = detect_regime(sig)
        assert result.regime == REGIME_R2

    def test_combined_legs_reach_r2_threshold(self) -> None:
        sig = _signals(
            has_dartconnect=True,
            has_mastercaller=True,
            dc_legs=6,
            mc_legs=5,
            match_count=5,
        )
        result = detect_regime(sig)
        assert result.regime == REGIME_R2

    def test_r2_has_visit_data_flag(self) -> None:
        sig = _signals(has_dartconnect=True, dc_legs=20)
        result = detect_regime(sig)
        assert result.has_visit_data is True

    def test_r2_regime_score_above_1(self) -> None:
        sig = _signals(has_dartconnect=True, dc_legs=50)
        result = detect_regime(sig)
        assert result.regime_score > 1.0

    def test_few_legs_stays_r1_not_r2(self) -> None:
        sig = _signals(
            has_dartconnect=True,
            dc_legs=3,  # below R2 threshold of 10
            has_dartsorakel=True,
            da=98.0,
            match_count=10,
        )
        result = detect_regime(sig)
        assert result.regime == REGIME_R1


class TestCoverageSignalsValidation:
    """Invalid signal inputs raise DartsCoverageError."""

    def test_negative_dc_legs_raises(self) -> None:
        with pytest.raises(DartsCoverageError):
            _signals(dc_legs=-1)

    def test_negative_mc_legs_raises(self) -> None:
        with pytest.raises(DartsCoverageError):
            _signals(mc_legs=-1)

    def test_negative_match_count_raises(self) -> None:
        with pytest.raises(DartsCoverageError):
            _signals(match_count=-1)


class TestBatchDetection:
    """batch_detect_regimes() processes multiple signals."""

    def test_batch_returns_correct_count(self) -> None:
        signals = [
            _signals(player_id=f"p{i}", match_count=i)
            for i in range(10)
        ]
        results = batch_detect_regimes(signals)
        assert len(results) == 10

    def test_batch_order_preserved(self) -> None:
        signals = [
            _signals(player_id=f"p{i}", match_count=i * 5)
            for i in range(5)
        ]
        results = batch_detect_regimes(signals)
        for i, result in enumerate(results):
            assert result.player_id == f"p{i}"


class TestRegimeFromDbFlags:
    """regime_from_db_flags() convenience wrapper."""

    def test_visit_data_gives_r2(self) -> None:
        result = regime_from_db_flags(
            "player1",
            None,
            has_visit_data=True,
            has_match_stats=True,
            has_dartsorakel=True,
            has_dartconnect=True,
            visit_leg_count=20,
            match_count=10,
        )
        assert result.regime == REGIME_R2

    def test_no_data_gives_r0(self) -> None:
        result = regime_from_db_flags(
            "player1",
            None,
            has_visit_data=False,
            has_match_stats=False,
            has_dartsorakel=False,
            has_dartconnect=False,
        )
        assert result.regime == REGIME_R0
