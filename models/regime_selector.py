"""
Determines which ML regime (R0/R1/R2) to use for a given player/match.

Based on the coverage_regimes table. The selected regime is the MINIMUM
of the two players' individual regimes -- both players must qualify for
a regime to use it.

R0: official metadata only (Elo, rankings, format) -- logistic regression
R1: + Darts Orakel stats (3DA, checkout%, rolling) -- LightGBM stacking
R2: + DartConnect visit data (NOT DEFAULT) -- full ensemble
"""
from __future__ import annotations

import enum
from typing import Optional

import structlog

from data.coverage_regime import (
    REGIME_R0,
    REGIME_R1,
    REGIME_R2,
    CoverageSignals,
    RegimeResult,
    detect_regime,
)
from models import DartsMLError

logger = structlog.get_logger(__name__)


class DartsMLRegime(enum.IntEnum):
    """ML model regime, ordered by data richness."""
    R0 = 0  # official metadata only
    R1 = 1  # + Darts Orakel stats
    R2 = 2  # + DartConnect visit data (NOT DEFAULT)

    @classmethod
    def from_string(cls, regime_str: str) -> DartsMLRegime:
        """Convert a string regime label to the enum."""
        mapping = {"R0": cls.R0, "R1": cls.R1, "R2": cls.R2}
        if regime_str not in mapping:
            raise DartsMLError(
                f"Unknown regime string {regime_str!r}. Valid: {list(mapping.keys())}"
            )
        return mapping[regime_str]


class RegimeSelector:
    """
    Selects the appropriate ML regime for a match.

    The regime is determined by the minimum of the two players'
    individual regime qualifications. Both players must have sufficient
    data for a given tier.

    Parameters
    ----------
    coverage_store:
        Optional callable that retrieves CoverageSignals for a player.
        Signature: (player_id, competition_id) -> CoverageSignals.
        If None, a default store must be provided via set_store().
    """

    def __init__(
        self,
        coverage_store: Optional[object] = None,
    ) -> None:
        self._coverage_store = coverage_store
        self._cache: dict[str, DartsMLRegime] = {}
        self._log = logger.bind(component="RegimeSelector")

    def select_regime(
        self,
        player1_id: str,
        player2_id: str,
        competition_id: str,
    ) -> DartsMLRegime:
        """
        Select the ML regime for a match between two players.

        Both players must qualify for a regime. The minimum of the
        two players' regimes is used.

        Parameters
        ----------
        player1_id:
            Canonical player 1 ID.
        player2_id:
            Canonical player 2 ID.
        competition_id:
            Competition ID for context-specific regime detection.

        Returns
        -------
        DartsMLRegime
            The selected regime (minimum of the two players).

        Raises
        ------
        DartsMLError
            If regime detection fails for either player.
        """
        r1 = self._get_player_regime(player1_id, competition_id)
        r2 = self._get_player_regime(player2_id, competition_id)
        selected = min(r1, r2)

        self._log.info(
            "regime_selected",
            player1_id=player1_id,
            player2_id=player2_id,
            competition_id=competition_id,
            player1_regime=r1.name,
            player2_regime=r2.name,
            selected_regime=selected.name,
        )

        return selected

    def select_regime_from_signals(
        self,
        p1_signals: CoverageSignals,
        p2_signals: CoverageSignals,
    ) -> DartsMLRegime:
        """
        Select regime directly from coverage signals (no store lookup).

        Parameters
        ----------
        p1_signals:
            Coverage signals for player 1.
        p2_signals:
            Coverage signals for player 2.

        Returns
        -------
        DartsMLRegime
        """
        r1_result = detect_regime(p1_signals)
        r2_result = detect_regime(p2_signals)

        r1 = DartsMLRegime.from_string(r1_result.regime)
        r2 = DartsMLRegime.from_string(r2_result.regime)

        return min(r1, r2)

    def _get_player_regime(
        self,
        player_id: str,
        competition_id: str,
    ) -> DartsMLRegime:
        """
        Get or compute the regime for a single player.

        Uses cache keyed by (player_id, competition_id).
        """
        cache_key = f"{player_id}:{competition_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        signals = self._load_coverage_signals(player_id, competition_id)
        result = detect_regime(signals)
        regime = DartsMLRegime.from_string(result.regime)

        self._cache[cache_key] = regime
        return regime

    def _load_coverage_signals(
        self,
        player_id: str,
        competition_id: str,
    ) -> CoverageSignals:
        """
        Load coverage signals for a player.

        If a coverage store is available, uses it. Otherwise falls back
        to a default R0 signal (result-only).
        """
        if self._coverage_store is not None and hasattr(
            self._coverage_store, "get_signals"
        ):
            signals = self._coverage_store.get_signals(  # type: ignore[union-attr]
                player_id, competition_id
            )
            if signals is not None:
                return signals

        # Default: R0 (result-only) when no store or no data found
        return CoverageSignals(
            player_id=player_id,
            competition_id=competition_id,
            has_dartconnect_data=False,
            has_mastercaller_data=False,
            has_dartsorakel_stats=False,
            has_pdc_match_stats=False,
            dartconnect_leg_count=0,
            mastercaller_leg_count=0,
            match_count=0,
        )

    def clear_cache(self) -> None:
        """Clear the regime cache."""
        self._cache.clear()
