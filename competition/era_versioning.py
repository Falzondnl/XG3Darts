"""
Competition era versioning.

Handles format changes within the same competition over time.  The primary
use-case for Sprint 1 is the PDC World Championship's draw-size change:

- **Pre-2003**: 16-player / 32-player (early PDC era, BDO split)
- **2003–2019**: 128-player draw  → code ``PDC_WC_ERA_2020``
- **2020+**:     96-player draw   → code ``PDC_WC``

The registry maps ``(base_code, season_year) → format_code`` so that
downstream components can look up the correct format for historical matches
without embedding year logic everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from competition.format_registry import (
    DartsCompetitionFormat,
    DartsFormatError,
    get_format,
)


class DartsEraError(Exception):
    """Raised when era versioning cannot resolve a format for a given year."""


@dataclass(frozen=True)
class EraWindow:
    """
    A half-open interval [start_year, end_year) associated with a format code.

    Attributes
    ----------
    format_code:
        The :attr:`~competition.format_registry.DartsCompetitionFormat.code`
        that applies during this era.
    start_year:
        First season year (inclusive) in which this format was used.
    end_year:
        First season year (exclusive) in which this format was *no longer*
        used.  ``None`` means this era is open-ended (current format).
    field_size:
        The draw size during this era. Informational only.
    notes:
        Free-text notes for documentation purposes.
    """

    format_code: str
    start_year: int
    end_year: Optional[int]
    field_size: Optional[int] = None
    notes: str = ""

    def contains_year(self, season_year: int) -> bool:
        """Return True if *season_year* falls within this era window."""
        if season_year < self.start_year:
            return False
        if self.end_year is not None and season_year >= self.end_year:
            return False
        return True


# ---------------------------------------------------------------------------
# Era registry
# Key = base competition name (not a format code)
# Value = ordered list of EraWindow from earliest to latest
# ---------------------------------------------------------------------------

_ERA_REGISTRY: dict[str, list[EraWindow]] = {}


def _register_era(base_name: str, windows: list[EraWindow]) -> None:
    """Register a list of era windows for a base competition name."""
    # Validate windows are ordered and non-overlapping
    for i in range(len(windows) - 1):
        current = windows[i]
        nxt = windows[i + 1]
        if current.end_year is None:
            raise DartsEraError(
                f"Era window for {current.format_code!r} has end_year=None "
                f"but is not the last window."
            )
        if current.end_year != nxt.start_year:
            raise DartsEraError(
                f"Era windows for {base_name!r} are not contiguous: "
                f"{current.end_year} ≠ {nxt.start_year}"
            )
    _ERA_REGISTRY[base_name] = windows


# ---------------------------------------------------------------------------
# PDC World Championship eras
# ---------------------------------------------------------------------------
_register_era(
    "PDC_WC",
    [
        EraWindow(
            format_code="PDC_WC_ERA_2020",
            start_year=1994,
            end_year=2020,
            field_size=128,
            notes=(
                "Original PDC World Championship era. Draw grew from 16 to 128 "
                "players across this period; the 2003 expansion to 128 is the "
                "canonical cutpoint used throughout the system."
            ),
        ),
        EraWindow(
            format_code="PDC_WC",
            start_year=2020,
            end_year=None,
            field_size=96,
            notes=(
                "96-player draw introduced for the 2020 World Championship. "
                "32 first-round byes eliminated, replaced by preliminary round."
            ),
        ),
    ],
)

# ---------------------------------------------------------------------------
# PDC Premier League eras (format has been stable since 2005 in essentials;
# the number of players changed from 8 to 10 and back — noting here for
# future expansion but using a single era for Sprint 1)
# ---------------------------------------------------------------------------
_register_era(
    "PDC_PL",
    [
        EraWindow(
            format_code="PDC_PL",
            start_year=2005,
            end_year=None,
            field_size=8,
            notes=(
                "Premier League launched 2005. Player count has varied (8, 10, "
                "back to 8) but core draw-enabled leg format is unchanged."
            ),
        ),
    ],
)

# ---------------------------------------------------------------------------
# PDC Grand Slam eras (group stage draw format stable since 2007)
# ---------------------------------------------------------------------------
_register_era(
    "PDC_GS",
    [
        EraWindow(
            format_code="PDC_GS",
            start_year=2007,
            end_year=None,
            field_size=32,
            notes="Grand Slam of Darts launched 2007.",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_format(base_name: str, season_year: int) -> DartsCompetitionFormat:
    """
    Resolve the correct :class:`~competition.format_registry.DartsCompetitionFormat`
    for a competition based on the season year.

    Parameters
    ----------
    base_name:
        The competition base name registered via :func:`_register_era`
        (e.g. ``"PDC_WC"``).
    season_year:
        The four-digit season year of the match or tournament.

    Returns
    -------
    DartsCompetitionFormat

    Raises
    ------
    DartsEraError
        If no era window covers the given year for this base competition.
    DartsFormatError
        If the resolved format code is not in the format registry.
    """
    if base_name not in _ERA_REGISTRY:
        raise DartsEraError(
            f"No era windows registered for {base_name!r}. "
            f"Registered competitions: {sorted(_ERA_REGISTRY.keys())}"
        )
    for window in _ERA_REGISTRY[base_name]:
        if window.contains_year(season_year):
            return get_format(window.format_code)

    windows = _ERA_REGISTRY[base_name]
    raise DartsEraError(
        f"No era window covers season_year={season_year} for {base_name!r}. "
        f"Earliest registered year: {windows[0].start_year}"
    )


def get_field_size(base_name: str, season_year: int) -> Optional[int]:
    """
    Return the draw field size for a competition in a given season year.

    Parameters
    ----------
    base_name:
        The competition base name (e.g. ``"PDC_WC"``).
    season_year:
        The season year.

    Returns
    -------
    int | None
        The field size, or ``None`` if unknown / variable.

    Raises
    ------
    DartsEraError
        If no era window covers the given year.
    """
    if base_name not in _ERA_REGISTRY:
        raise DartsEraError(
            f"No era windows registered for {base_name!r}."
        )
    for window in _ERA_REGISTRY[base_name]:
        if window.contains_year(season_year):
            return window.field_size
    raise DartsEraError(
        f"No era window covers season_year={season_year} for {base_name!r}."
    )


def list_era_windows(base_name: str) -> list[EraWindow]:
    """
    Return the ordered list of era windows for a competition.

    Parameters
    ----------
    base_name:
        The competition base name.

    Raises
    ------
    DartsEraError
        If no windows are registered for ``base_name``.
    """
    if base_name not in _ERA_REGISTRY:
        raise DartsEraError(
            f"No era windows registered for {base_name!r}."
        )
    return list(_ERA_REGISTRY[base_name])


def list_versioned_competitions() -> list[str]:
    """Return the base names of all competitions with era versioning."""
    return sorted(_ERA_REGISTRY.keys())
