"""
Darts competition format registry.

Defines every known competition format with complete structural metadata:
starting score, double-in requirements, bull-off rules, per-round formats,
draw rules, and result types. This is the single source of truth for all
format-dependent downstream calculations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class DartsFormatError(Exception):
    """Raised when a competition format is invalid or unknown."""


@dataclass(frozen=True)
class DartsRoundFormat:
    """
    Encodes the structural rules for a single round within a competition.

    Attributes
    ----------
    name:
        Human-readable round label (e.g. ``"Final"``, ``"Last 16"``).
    legs_to_win:
        Number of legs required to win the match (legs-only formats).
    sets_to_win:
        Number of sets required to win the match (sets formats).
    legs_per_set:
        Number of legs required to win one set (sets formats).
    draw_enabled:
        Whether a drawn result is a valid outcome (Premier League group stage,
        Grand Slam group stage).
    two_clear_legs:
        Whether a two-clear-legs rule is active (World Matchplay final).
    two_clear_legs_max_extra:
        Cap on extra legs under the two-clear rule. ``None`` means no cap.
    group_stage:
        Whether this round is a group stage (affects ELO draw handling).
    format_type:
        ``"singles"`` or ``"doubles"`` (World Cup of Darts).
    """

    name: str
    legs_to_win: Optional[int] = None
    sets_to_win: Optional[int] = None
    legs_per_set: Optional[int] = None
    draw_enabled: bool = False
    two_clear_legs: bool = False
    two_clear_legs_max_extra: Optional[int] = None
    group_stage: bool = False
    format_type: str = "singles"

    def __post_init__(self) -> None:
        if self.format_type not in ("singles", "doubles"):
            raise DartsFormatError(
                f"format_type must be 'singles' or 'doubles', got {self.format_type!r}"
            )
        has_legs = self.legs_to_win is not None
        has_sets = self.sets_to_win is not None and self.legs_per_set is not None
        if not has_legs and not has_sets:
            raise DartsFormatError(
                f"Round '{self.name}' must define either legs_to_win or "
                f"(sets_to_win + legs_per_set)."
            )
        if has_legs and has_sets:
            raise DartsFormatError(
                f"Round '{self.name}' cannot define both legs_to_win and sets format."
            )

    @property
    def is_sets_format(self) -> bool:
        """True when the round uses sets-and-legs structure."""
        return self.sets_to_win is not None

    @property
    def total_legs_possible(self) -> int:
        """Maximum legs that can occur in a legs-only round."""
        if self.is_sets_format:
            raise DartsFormatError(
                "total_legs_possible is only valid for legs-only rounds."
            )
        assert self.legs_to_win is not None
        return 2 * self.legs_to_win - 1

    @property
    def total_sets_possible(self) -> int:
        """Maximum sets that can occur in a sets round."""
        if not self.is_sets_format:
            raise DartsFormatError(
                "total_sets_possible is only valid for sets rounds."
            )
        assert self.sets_to_win is not None
        return 2 * self.sets_to_win - 1


@dataclass(frozen=True)
class DartsCompetitionFormat:
    """
    Complete structural description of a darts competition.

    Attributes
    ----------
    code:
        Unique short identifier used throughout the system (e.g. ``"PDC_WC"``).
    name:
        Full human-readable competition name.
    organiser:
        Organising body (``"PDC"`` | ``"WDF"`` | ``"BDO"`` | ``"Mixed"``).
    ecosystem:
        Betting/statistical ecosystem classification
        (``"pdc_mens"`` | ``"pdc_womens"`` | ``"wdf_open"`` | ``"development"``
        | ``"team_doubles"``).
    starting_score:
        Starting score for each leg (almost always 501, occasionally 701).
    double_start_required:
        Whether players must hit a double to begin scoring.
    bull_off_first_leg:
        Whether the opening leg of each set/match is determined by bull-off.
    alternating_starts:
        Whether throw order alternates each leg (standard PDC rule).
    result_types:
        Tuple of valid result strings. Typically ``("p1_win", "p2_win")``; add
        ``"draw"`` where draws are possible.
    per_round:
        Mapping of round label → :class:`DartsRoundFormat`.
    season_year:
        The season year this format definition applies from (used with era
        versioning for formats that change over time).
    field_size:
        Number of players/pairs in the draw (``None`` if open/variable).
    """

    code: str
    name: str
    organiser: str
    ecosystem: str
    starting_score: int
    double_start_required: bool
    bull_off_first_leg: bool
    alternating_starts: bool
    result_types: tuple
    per_round: dict
    season_year: int
    field_size: Optional[int] = None

    def __post_init__(self) -> None:
        valid_ecosystems = {
            "pdc_mens", "pdc_womens", "wdf_open", "development", "team_doubles"
        }
        if self.ecosystem not in valid_ecosystems:
            raise DartsFormatError(
                f"Unknown ecosystem {self.ecosystem!r}. "
                f"Valid: {sorted(valid_ecosystems)}"
            )
        if self.starting_score not in (501, 701):
            raise DartsFormatError(
                f"starting_score must be 501 or 701, got {self.starting_score}"
            )
        for result in self.result_types:
            if result not in ("p1_win", "p2_win", "draw"):
                raise DartsFormatError(f"Invalid result_type {result!r}")

    def allows_draw(self) -> bool:
        """Return True if a drawn result is possible in this competition."""
        return "draw" in self.result_types

    def get_round(self, round_name: str) -> DartsRoundFormat:
        """
        Retrieve the format spec for a named round.

        Parameters
        ----------
        round_name:
            The round label to look up.

        Raises
        ------
        DartsFormatError
            If the round name is not defined for this competition.
        """
        if round_name not in self.per_round:
            raise DartsFormatError(
                f"Round {round_name!r} not defined for {self.code}. "
                f"Available: {list(self.per_round.keys())}"
            )
        return self.per_round[round_name]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, DartsCompetitionFormat] = {}


def _register(fmt: DartsCompetitionFormat) -> DartsCompetitionFormat:
    _REGISTRY[fmt.code] = fmt
    return fmt


# ---------------------------------------------------------------------------
# PDC World Championship  (2020+ era — 96 players)
# era_versioning handles the 128-player era separately
# ---------------------------------------------------------------------------
PDC_WC = _register(DartsCompetitionFormat(
    code="PDC_WC",
    name="PDC World Championship",
    organiser="PDC",
    ecosystem="pdc_mens",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=2020,
    field_size=96,
    per_round={
        "Preliminary": DartsRoundFormat(name="Preliminary", legs_to_win=2),
        "Round 1":     DartsRoundFormat(name="Round 1",     sets_to_win=3, legs_per_set=3),
        "Round 2":     DartsRoundFormat(name="Round 2",     sets_to_win=3, legs_per_set=3),
        "Round 3":     DartsRoundFormat(name="Round 3",     sets_to_win=4, legs_per_set=3),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", sets_to_win=4, legs_per_set=3),
        "Semi-Final":  DartsRoundFormat(name="Semi-Final",  sets_to_win=5, legs_per_set=3),
        "Final":       DartsRoundFormat(name="Final",       sets_to_win=7, legs_per_set=3),
    },
))

# ---------------------------------------------------------------------------
# PDC World Championship — pre-2020 era (128-player draw)
# ---------------------------------------------------------------------------
PDC_WC_ERA_2020 = _register(DartsCompetitionFormat(
    code="PDC_WC_ERA_2020",
    name="PDC World Championship (128-player era)",
    organiser="PDC",
    ecosystem="pdc_mens",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=2003,
    field_size=128,
    per_round={
        "Preliminary": DartsRoundFormat(name="Preliminary", legs_to_win=2),
        "Round 1":     DartsRoundFormat(name="Round 1",     sets_to_win=3, legs_per_set=3),
        "Round 2":     DartsRoundFormat(name="Round 2",     sets_to_win=3, legs_per_set=3),
        "Round 3":     DartsRoundFormat(name="Round 3",     sets_to_win=4, legs_per_set=3),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", sets_to_win=4, legs_per_set=3),
        "Semi-Final":  DartsRoundFormat(name="Semi-Final",  sets_to_win=5, legs_per_set=3),
        "Final":       DartsRoundFormat(name="Final",       sets_to_win=7, legs_per_set=3),
    },
))

# ---------------------------------------------------------------------------
# PDC Premier League (group stage has draws; play-offs do not)
# ---------------------------------------------------------------------------
PDC_PL = _register(DartsCompetitionFormat(
    code="PDC_PL",
    name="PDC Premier League",
    organiser="PDC",
    ecosystem="pdc_mens",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=False,  # PL uses winner-throws-next (not alternating)
    result_types=("p1_win", "p2_win", "draw"),
    season_year=2005,
    field_size=8,
    per_round={
        "League Night":   DartsRoundFormat(
            name="League Night", legs_to_win=7,
            draw_enabled=True, group_stage=True,
        ),
        "Play-Off Quarter-Final": DartsRoundFormat(
            name="Play-Off Quarter-Final", legs_to_win=10,
        ),
        "Play-Off Semi-Final": DartsRoundFormat(
            name="Play-Off Semi-Final", legs_to_win=10,
        ),
        "Play-Off Final": DartsRoundFormat(
            name="Play-Off Final", legs_to_win=10,
        ),
    },
))

# ---------------------------------------------------------------------------
# PDC World Matchplay
# ---------------------------------------------------------------------------
PDC_WM = _register(DartsCompetitionFormat(
    code="PDC_WM",
    name="PDC World Matchplay",
    organiser="PDC",
    ecosystem="pdc_mens",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=1994,
    field_size=32,
    per_round={
        "Round 1":       DartsRoundFormat(name="Round 1",       legs_to_win=10),
        "Round 2":       DartsRoundFormat(name="Round 2",       legs_to_win=11),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", legs_to_win=13),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    legs_to_win=15),
        "Final":         DartsRoundFormat(
            name="Final", legs_to_win=18,
            two_clear_legs=True, two_clear_legs_max_extra=None,
        ),
    },
))

# ---------------------------------------------------------------------------
# PDC Grand Prix
# ---------------------------------------------------------------------------
PDC_GP = _register(DartsCompetitionFormat(
    code="PDC_GP",
    name="PDC Grand Prix",
    organiser="PDC",
    ecosystem="pdc_mens",
    starting_score=501,
    double_start_required=True,   # double-in required
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=1995,
    field_size=32,
    per_round={
        "Round 1":       DartsRoundFormat(name="Round 1",       legs_to_win=3),
        "Round 2":       DartsRoundFormat(name="Round 2",       sets_to_win=3, legs_per_set=3),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", sets_to_win=3, legs_per_set=3),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    sets_to_win=3, legs_per_set=3),
        "Final":         DartsRoundFormat(name="Final",         sets_to_win=4, legs_per_set=3),
    },
))

# ---------------------------------------------------------------------------
# PDC UK Open
# ---------------------------------------------------------------------------
PDC_UK = _register(DartsCompetitionFormat(
    code="PDC_UK",
    name="PDC UK Open",
    organiser="PDC",
    ecosystem="pdc_mens",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=2003,
    field_size=256,
    per_round={
        "Round 1":       DartsRoundFormat(name="Round 1",       legs_to_win=4),
        "Round 2":       DartsRoundFormat(name="Round 2",       legs_to_win=4),
        "Round 3":       DartsRoundFormat(name="Round 3",       legs_to_win=5),
        "Round 4":       DartsRoundFormat(name="Round 4",       legs_to_win=5),
        "Round 5":       DartsRoundFormat(name="Round 5",       legs_to_win=6),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", legs_to_win=6),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    legs_to_win=6),
        "Final":         DartsRoundFormat(name="Final",         legs_to_win=6),
    },
))

# ---------------------------------------------------------------------------
# PDC Grand Slam of Darts  (group stage has draws)
# ---------------------------------------------------------------------------
PDC_GS = _register(DartsCompetitionFormat(
    code="PDC_GS",
    name="PDC Grand Slam of Darts",
    organiser="PDC",
    ecosystem="pdc_mens",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win", "draw"),
    season_year=2007,
    field_size=32,
    per_round={
        "Group Stage":   DartsRoundFormat(
            name="Group Stage", legs_to_win=5,
            draw_enabled=True, group_stage=True,
        ),
        "Round of 16":   DartsRoundFormat(name="Round of 16",   legs_to_win=10),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", legs_to_win=10),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    legs_to_win=11),
        "Final":         DartsRoundFormat(name="Final",         legs_to_win=11),
    },
))

# ---------------------------------------------------------------------------
# PDC European Tour
# ---------------------------------------------------------------------------
PDC_ET = _register(DartsCompetitionFormat(
    code="PDC_ET",
    name="PDC European Tour",
    organiser="PDC",
    ecosystem="pdc_mens",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=2012,
    field_size=48,
    per_round={
        "Round 1":       DartsRoundFormat(name="Round 1",       legs_to_win=6),
        "Round 2":       DartsRoundFormat(name="Round 2",       legs_to_win=6),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", legs_to_win=6),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    legs_to_win=6),
        "Final":         DartsRoundFormat(name="Final",         legs_to_win=7),
    },
))

# ---------------------------------------------------------------------------
# PDC Players Championship
# ---------------------------------------------------------------------------
PDC_PC = _register(DartsCompetitionFormat(
    code="PDC_PC",
    name="PDC Players Championship",
    organiser="PDC",
    ecosystem="pdc_mens",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=2005,
    field_size=128,
    per_round={
        "Round 1":       DartsRoundFormat(name="Round 1",       legs_to_win=4),
        "Round 2":       DartsRoundFormat(name="Round 2",       legs_to_win=4),
        "Round 3":       DartsRoundFormat(name="Round 3",       legs_to_win=4),
        "Round 4":       DartsRoundFormat(name="Round 4",       legs_to_win=4),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", legs_to_win=5),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    legs_to_win=5),
        "Final":         DartsRoundFormat(name="Final",         legs_to_win=5),
    },
))

# ---------------------------------------------------------------------------
# PDC Players Championship Finals
# ---------------------------------------------------------------------------
PDC_PCF = _register(DartsCompetitionFormat(
    code="PDC_PCF",
    name="PDC Players Championship Finals",
    organiser="PDC",
    ecosystem="pdc_mens",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=2010,
    field_size=64,
    per_round={
        "Round 1":       DartsRoundFormat(name="Round 1",       legs_to_win=6),
        "Round 2":       DartsRoundFormat(name="Round 2",       legs_to_win=6),
        "Round 3":       DartsRoundFormat(name="Round 3",       legs_to_win=6),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", legs_to_win=8),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    legs_to_win=8),
        "Final":         DartsRoundFormat(name="Final",         legs_to_win=10),
    },
))

# ---------------------------------------------------------------------------
# PDC Masters (formerly The Masters)
# ---------------------------------------------------------------------------
PDC_MASTERS = _register(DartsCompetitionFormat(
    code="PDC_MASTERS",
    name="PDC Masters",
    organiser="PDC",
    ecosystem="pdc_mens",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=2012,
    field_size=16,
    per_round={
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", legs_to_win=6),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    legs_to_win=7),
        "Final":         DartsRoundFormat(name="Final",         legs_to_win=8),
    },
))

# ---------------------------------------------------------------------------
# PDC World Series of Darts Finals
# ---------------------------------------------------------------------------
PDC_WS = _register(DartsCompetitionFormat(
    code="PDC_WS",
    name="PDC World Series of Darts Finals",
    organiser="PDC",
    ecosystem="pdc_mens",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=2014,
    field_size=16,
    per_round={
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", sets_to_win=3, legs_per_set=5),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    sets_to_win=4, legs_per_set=5),
        "Final":         DartsRoundFormat(name="Final",         sets_to_win=5, legs_per_set=5),
    },
))

# ---------------------------------------------------------------------------
# PDC World Cup of Darts  (doubles format)
# ---------------------------------------------------------------------------
PDC_WCUP = _register(DartsCompetitionFormat(
    code="PDC_WCUP",
    name="PDC World Cup of Darts",
    organiser="PDC",
    ecosystem="team_doubles",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=2010,
    field_size=32,
    per_round={
        "Round 1 Singles":   DartsRoundFormat(
            name="Round 1 Singles", legs_to_win=6, format_type="singles",
        ),
        "Round 1 Doubles":   DartsRoundFormat(
            name="Round 1 Doubles", legs_to_win=4, format_type="doubles",
        ),
        "Quarter-Final Singles": DartsRoundFormat(
            name="Quarter-Final Singles", legs_to_win=7, format_type="singles",
        ),
        "Quarter-Final Doubles": DartsRoundFormat(
            name="Quarter-Final Doubles", legs_to_win=5, format_type="doubles",
        ),
        "Semi-Final Singles": DartsRoundFormat(
            name="Semi-Final Singles", legs_to_win=7, format_type="singles",
        ),
        "Semi-Final Doubles": DartsRoundFormat(
            name="Semi-Final Doubles", legs_to_win=5, format_type="doubles",
        ),
        "Final Singles":     DartsRoundFormat(
            name="Final Singles", legs_to_win=8, format_type="singles",
        ),
        "Final Doubles":     DartsRoundFormat(
            name="Final Doubles", legs_to_win=6, format_type="doubles",
        ),
    },
))

# ---------------------------------------------------------------------------
# PDC Women's Series
# ---------------------------------------------------------------------------
PDC_WOM_SERIES = _register(DartsCompetitionFormat(
    code="PDC_WOM_SERIES",
    name="PDC Women's Series",
    organiser="PDC",
    ecosystem="pdc_womens",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=2019,
    field_size=None,
    per_round={
        "Round 1":       DartsRoundFormat(name="Round 1",       legs_to_win=3),
        "Round 2":       DartsRoundFormat(name="Round 2",       legs_to_win=3),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", legs_to_win=4),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    legs_to_win=4),
        "Final":         DartsRoundFormat(name="Final",         legs_to_win=5),
    },
))

# ---------------------------------------------------------------------------
# PDC Women's World Matchplay
# ---------------------------------------------------------------------------
PDC_WWM = _register(DartsCompetitionFormat(
    code="PDC_WWM",
    name="PDC Women's World Matchplay",
    organiser="PDC",
    ecosystem="pdc_womens",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=2023,
    field_size=16,
    per_round={
        "Round 1":       DartsRoundFormat(name="Round 1",       legs_to_win=5),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", legs_to_win=7),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    legs_to_win=8),
        "Final":         DartsRoundFormat(name="Final",         legs_to_win=9),
    },
))

# ---------------------------------------------------------------------------
# PDC Development Tour
# ---------------------------------------------------------------------------
PDC_DEVTOUR = _register(DartsCompetitionFormat(
    code="PDC_DEVTOUR",
    name="PDC Development Tour",
    organiser="PDC",
    ecosystem="development",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=2006,
    field_size=None,
    per_round={
        "Round 1":       DartsRoundFormat(name="Round 1",       legs_to_win=4),
        "Round 2":       DartsRoundFormat(name="Round 2",       legs_to_win=4),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", legs_to_win=4),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    legs_to_win=4),
        "Final":         DartsRoundFormat(name="Final",         legs_to_win=5),
    },
))

# ---------------------------------------------------------------------------
# PDC Challenge Tour
# ---------------------------------------------------------------------------
PDC_CHALLENGE = _register(DartsCompetitionFormat(
    code="PDC_CHALLENGE",
    name="PDC Challenge Tour",
    organiser="PDC",
    ecosystem="development",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=2012,
    field_size=None,
    per_round={
        "Round 1":       DartsRoundFormat(name="Round 1",       legs_to_win=4),
        "Round 2":       DartsRoundFormat(name="Round 2",       legs_to_win=4),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", legs_to_win=4),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    legs_to_win=4),
        "Final":         DartsRoundFormat(name="Final",         legs_to_win=5),
    },
))

# ---------------------------------------------------------------------------
# PDC World Youth Championship
# ---------------------------------------------------------------------------
PDC_WYC = _register(DartsCompetitionFormat(
    code="PDC_WYC",
    name="PDC World Youth Championship",
    organiser="PDC",
    ecosystem="development",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=2010,
    field_size=None,
    per_round={
        "Round 1":       DartsRoundFormat(name="Round 1",       legs_to_win=4),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", legs_to_win=5),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    legs_to_win=6),
        "Final":         DartsRoundFormat(name="Final",         legs_to_win=7),
    },
))

# ---------------------------------------------------------------------------
# WDF World Championship
# ---------------------------------------------------------------------------
WDF_WC = _register(DartsCompetitionFormat(
    code="WDF_WC",
    name="WDF World Championship",
    organiser="WDF",
    ecosystem="wdf_open",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=2020,
    field_size=64,
    per_round={
        "Round 1":       DartsRoundFormat(name="Round 1",       sets_to_win=3, legs_per_set=3),
        "Round 2":       DartsRoundFormat(name="Round 2",       sets_to_win=3, legs_per_set=3),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", sets_to_win=4, legs_per_set=3),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    sets_to_win=4, legs_per_set=3),
        "Final":         DartsRoundFormat(name="Final",         sets_to_win=5, legs_per_set=3),
    },
))

# ---------------------------------------------------------------------------
# WDF European Championship
# ---------------------------------------------------------------------------
WDF_EC = _register(DartsCompetitionFormat(
    code="WDF_EC",
    name="WDF European Championship",
    organiser="WDF",
    ecosystem="wdf_open",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=1978,
    field_size=None,
    per_round={
        "Round 1":       DartsRoundFormat(name="Round 1",       legs_to_win=4),
        "Round 2":       DartsRoundFormat(name="Round 2",       legs_to_win=4),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", legs_to_win=5),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    legs_to_win=6),
        "Final":         DartsRoundFormat(name="Final",         legs_to_win=7),
    },
))

# ---------------------------------------------------------------------------
# WDF Open tournaments (generic)
# ---------------------------------------------------------------------------
WDF_OPEN = _register(DartsCompetitionFormat(
    code="WDF_OPEN",
    name="WDF Open",
    organiser="WDF",
    ecosystem="wdf_open",
    starting_score=501,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=1990,
    field_size=None,
    per_round={
        "Round 1":       DartsRoundFormat(name="Round 1",       legs_to_win=3),
        "Round 2":       DartsRoundFormat(name="Round 2",       legs_to_win=3),
        "Quarter-Final": DartsRoundFormat(name="Quarter-Final", legs_to_win=4),
        "Semi-Final":    DartsRoundFormat(name="Semi-Final",    legs_to_win=4),
        "Final":         DartsRoundFormat(name="Final",         legs_to_win=5),
    },
))

# ---------------------------------------------------------------------------
# 701 Team (Doubles — used in some World Cup legs and exhibition formats)
# ---------------------------------------------------------------------------
TEAM_701 = _register(DartsCompetitionFormat(
    code="TEAM_701",
    name="701 Team Doubles",
    organiser="Mixed",
    ecosystem="team_doubles",
    starting_score=701,
    double_start_required=False,
    bull_off_first_leg=True,
    alternating_starts=True,
    result_types=("p1_win", "p2_win"),
    season_year=1990,
    field_size=None,
    per_round={
        "Match": DartsRoundFormat(
            name="Match", legs_to_win=3, format_type="doubles",
        ),
        "Final": DartsRoundFormat(
            name="Final", legs_to_win=5, format_type="doubles",
        ),
    },
))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_format(code: str) -> DartsCompetitionFormat:
    """
    Retrieve a competition format by its unique code.

    Parameters
    ----------
    code:
        The format code string (e.g. ``"PDC_WC"``).

    Returns
    -------
    DartsCompetitionFormat

    Raises
    ------
    DartsFormatError
        If the code is not registered.
    """
    if code not in _REGISTRY:
        raise DartsFormatError(
            f"Unknown format code {code!r}. "
            f"Registered codes: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[code]


def list_formats() -> list[str]:
    """Return all registered format codes, sorted."""
    return sorted(_REGISTRY.keys())


def get_all_formats() -> dict[str, DartsCompetitionFormat]:
    """Return a shallow copy of the full registry."""
    return dict(_REGISTRY)
