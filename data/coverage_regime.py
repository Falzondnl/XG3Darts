"""
Data coverage regime detection.

Determines which modelling tier (R0/R1/R2) applies for a given player
or player-competition combination, based on what data is available:

R0 — Result-only
    Only match results (win/loss/draw) are available.  No per-visit or
    per-match stats.  Uses a baseline logit model.

R1 — Match-level statistics
    Match-level aggregates available: 3DA, first-9 average, checkout%,
    hold/break rates.  Derived from DartsOrakel or aggregated PDC data.
    Uses a LightGBM model.

R2 — Visit-level detail
    Full per-visit data from DartConnect or Mastercaller: individual dart
    scores, checkout sequences, exact darts-at-double counts.
    Uses the full stacking ensemble.

The regime is computed from a :class:`CoverageSignals` record and stored
in the ``darts_coverage_regimes`` database table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog


logger = structlog.get_logger(__name__)


class DartsCoverageError(Exception):
    """Raised when coverage regime detection encounters invalid inputs."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGIME_R0 = "R0"
REGIME_R1 = "R1"
REGIME_R2 = "R2"

VALID_REGIMES = frozenset({REGIME_R0, REGIME_R1, REGIME_R2})

# Minimum thresholds required to qualify for each regime
_R1_MIN_STAT_OBSERVATIONS = 5   # minimum matches with 3DA data
_R2_MIN_VISIT_LEGS = 10         # minimum legs with visit-level data


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CoverageSignals:
    """
    Input signals for regime detection, one record per player/competition.

    Attributes
    ----------
    player_id:
        Canonical internal player UUID.
    competition_id:
        Competition UUID, or None for career-level coverage.
    has_dartconnect_data:
        True if DartConnect visit-level data exists for this player.
    has_mastercaller_data:
        True if Mastercaller visit data exists.
    has_dartsorakel_stats:
        True if DartsOrakel 3DA/checkout stats exist.
    has_pdc_match_stats:
        True if PDC-level match statistics (averages) are available.
    dartconnect_leg_count:
        Number of legs with DartConnect visit data. 0 if none.
    mastercaller_leg_count:
        Number of legs with Mastercaller data. 0 if none.
    dartsorakel_3da:
        DartsOrakel 3-dart average, if available.
    match_count:
        Number of completed matches recorded for this player
        (in scope — career or within competition).
    """

    player_id: str
    competition_id: Optional[str]
    has_dartconnect_data: bool
    has_mastercaller_data: bool
    has_dartsorakel_stats: bool
    has_pdc_match_stats: bool
    dartconnect_leg_count: int = 0
    mastercaller_leg_count: int = 0
    dartsorakel_3da: Optional[float] = None
    match_count: int = 0

    def __post_init__(self) -> None:
        if self.dartconnect_leg_count < 0:
            raise DartsCoverageError(
                f"dartconnect_leg_count must be >= 0, got {self.dartconnect_leg_count}"
            )
        if self.mastercaller_leg_count < 0:
            raise DartsCoverageError(
                f"mastercaller_leg_count must be >= 0, got {self.mastercaller_leg_count}"
            )
        if self.match_count < 0:
            raise DartsCoverageError(
                f"match_count must be >= 0, got {self.match_count}"
            )


@dataclass
class RegimeResult:
    """
    Output from regime detection.

    Attributes
    ----------
    player_id:
        Canonical player UUID.
    competition_id:
        Competition UUID, or None for career-level.
    regime:
        Detected regime: ``"R0"``, ``"R1"``, or ``"R2"``.
    regime_score:
        Continuous score in [0, 2] where 0=R0, 1=R1, 2=R2.
        Used to rank-order players within a tier.
    has_visit_data:
        True if any visit-level data is present.
    has_match_stats:
        True if match-level statistics are present.
    has_result_only:
        True when only result data is available (regime is R0).
    has_dartsorakel:
        True if DartsOrakel stats contributed to R1+.
    has_dartconnect:
        True if DartConnect data contributed to R2.
    reasoning:
        Human-readable explanation of the regime assignment.
    """

    player_id: str
    competition_id: Optional[str]
    regime: str
    regime_score: float
    has_visit_data: bool
    has_match_stats: bool
    has_result_only: bool
    has_dartsorakel: bool
    has_dartconnect: bool
    reasoning: str


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------

def detect_regime(signals: CoverageSignals) -> RegimeResult:
    """
    Determine the data coverage regime for a player/competition.

    The decision tree is:
    1. If visit-level data exceeds the R2 threshold → R2
    2. Else if match-level stats are present → R1
    3. Else → R0

    Parameters
    ----------
    signals:
        Coverage signals for the player/competition combination.

    Returns
    -------
    RegimeResult
        The detected regime and supporting metadata.

    Raises
    ------
    DartsCoverageError
        If signals contain invalid values.
    """
    visit_legs = signals.dartconnect_leg_count + signals.mastercaller_leg_count
    has_visit_data = visit_legs >= _R2_MIN_VISIT_LEGS

    has_match_stats = (
        signals.has_dartsorakel_stats
        or signals.has_pdc_match_stats
        and signals.match_count >= _R1_MIN_STAT_OBSERVATIONS
    )

    # R2: full visit data
    if has_visit_data:
        regime = REGIME_R2
        # Continuous score: 1.0 base + up to 1.0 additional for volume
        regime_score = 1.0 + min(visit_legs / 100.0, 1.0)
        reasoning = (
            f"R2: {visit_legs} visit-level legs available "
            f"(DartConnect={signals.dartconnect_leg_count}, "
            f"Mastercaller={signals.mastercaller_leg_count}). "
            f"Full stacking ensemble applicable."
        )

    # R1: match stats but no sufficient visit detail
    elif has_match_stats:
        regime = REGIME_R1
        # Continuous score: 0.0–1.0 range
        stat_score = 1.0 if signals.has_dartsorakel_stats else 0.5
        regime_score = stat_score * min(
            signals.match_count / _R1_MIN_STAT_OBSERVATIONS, 1.0
        )
        sources = []
        if signals.has_dartsorakel_stats:
            sources.append("DartsOrakel")
        if signals.has_pdc_match_stats:
            sources.append("PDC")
        reasoning = (
            f"R1: match-level statistics available from {', '.join(sources)}. "
            f"{signals.match_count} matches recorded. "
            f"LightGBM model applicable."
        )

    # R0: result only
    else:
        regime = REGIME_R0
        # Continuous score: 0.0–1.0, based on match count
        regime_score = min(signals.match_count / 50.0, 1.0) * 0.9
        reasoning = (
            f"R0: only match result data available. "
            f"{signals.match_count} matches recorded. "
            f"Baseline logit model only."
        )

    result = RegimeResult(
        player_id=signals.player_id,
        competition_id=signals.competition_id,
        regime=regime,
        regime_score=round(regime_score, 4),
        has_visit_data=has_visit_data,
        has_match_stats=has_match_stats,
        has_result_only=(regime == REGIME_R0),
        has_dartsorakel=signals.has_dartsorakel_stats,
        has_dartconnect=signals.has_dartconnect_data,
        reasoning=reasoning,
    )

    logger.debug(
        "coverage_regime_detected",
        player_id=signals.player_id,
        competition_id=signals.competition_id,
        regime=regime,
        regime_score=result.regime_score,
    )
    return result


def batch_detect_regimes(
    all_signals: list[CoverageSignals],
) -> list[RegimeResult]:
    """
    Detect regimes for a batch of player/competition combinations.

    Parameters
    ----------
    all_signals:
        List of coverage signal records.

    Returns
    -------
    list[RegimeResult]
        One result per input signal, in the same order.
    """
    results: list[RegimeResult] = []
    r0_count = r1_count = r2_count = 0

    for signals in all_signals:
        result = detect_regime(signals)
        results.append(result)
        if result.regime == REGIME_R0:
            r0_count += 1
        elif result.regime == REGIME_R1:
            r1_count += 1
        else:
            r2_count += 1

    logger.info(
        "batch_regime_detection_complete",
        total=len(results),
        r0=r0_count,
        r1=r1_count,
        r2=r2_count,
    )
    return results


def regime_from_db_flags(
    player_id: str,
    competition_id: Optional[str],
    has_visit_data: bool,
    has_match_stats: bool,
    has_dartsorakel: bool,
    has_dartconnect: bool,
    visit_leg_count: int = 0,
    match_count: int = 0,
) -> RegimeResult:
    """
    Convenience wrapper to build a :class:`RegimeResult` from database flag columns.

    Parameters
    ----------
    player_id:
        Player UUID.
    competition_id:
        Competition UUID or None.
    has_visit_data:
        ``darts_coverage_regimes.has_visit_data`` column value.
    has_match_stats:
        ``darts_coverage_regimes.has_match_stats`` column value.
    has_dartsorakel:
        ``darts_coverage_regimes.has_dartsorakel`` column value.
    has_dartconnect:
        ``darts_coverage_regimes.has_dartconnect`` column value.
    visit_leg_count:
        Total visit-level legs for regime score calculation.
    match_count:
        Total matches for regime score calculation.

    Returns
    -------
    RegimeResult
    """
    signals = CoverageSignals(
        player_id=player_id,
        competition_id=competition_id,
        has_dartconnect_data=has_dartconnect,
        has_mastercaller_data=False,  # flags already merged into has_visit_data
        has_dartsorakel_stats=has_dartsorakel,
        has_pdc_match_stats=has_match_stats,
        dartconnect_leg_count=visit_leg_count if has_dartconnect else 0,
        mastercaller_leg_count=visit_leg_count if (has_visit_data and not has_dartconnect) else 0,
        match_count=match_count,
    )
    return detect_regime(signals)
