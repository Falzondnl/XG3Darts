"""
Player entity resolution.

Matches player identities across data sources:
- PDC (``pdc_id``, name, slug)
- DartsOrakel (``player_key``, ``player_name``, country)
- DartsDatabase (``dartsdatabase_id``, name)
- DartConnect (``dartconnect_id``, name)
- WDF (``wdf_id``, name)

Resolution strategy (priority order):
1. Exact match on a known cross-reference ID (if a mapping table entry exists)
2. Exact name match (full name, case-insensitive, normalised)
3. Fuzzy name match with confidence scoring (edit distance + token overlap)
4. Slug match (PDC slugs are fairly canonical)

Each resolved match gets a ``source_confidence`` score in [0, 1] and
a ``primary_source`` tag indicating which source is most trusted.

Source confidence hierarchy:
    PDC    → 0.95  (authoritative for PDC players)
    DartsOrakel → 0.85 (cross-checked stats, good name coverage)
    DartsDatabase → 0.75 (historical, manually maintained)
    DartConnect → 0.90 (live, highly accurate for tour events)
    WDF     → 0.80 (authoritative for WDF players)
    Mastercaller → 0.70 (community-maintained, smaller coverage)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import structlog


logger = structlog.get_logger(__name__)


class DartsEntityResolutionError(Exception):
    """Raised when entity resolution encounters an unresolvable conflict."""


# ---------------------------------------------------------------------------
# Source confidence scores
# ---------------------------------------------------------------------------

SOURCE_CONFIDENCE: dict[str, float] = {
    "pdc": 0.95,
    "dartconnect": 0.90,
    "dartsorakel": 0.85,
    "wdf": 0.80,
    "dartsdatabase": 0.75,
    "mastercaller": 0.70,
}

# Resolution confidence thresholds
_EXACT_MATCH_CONFIDENCE = 1.0
_SLUG_MATCH_CONFIDENCE = 0.95
_FUZZY_HIGH_CONFIDENCE = 0.88
_FUZZY_LOW_CONFIDENCE = 0.70
_MIN_ACCEPTABLE_CONFIDENCE = 0.65


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def normalise_name(name: str) -> str:
    """
    Normalise a player name for comparison.

    Steps:
    1. Unicode NFC normalisation
    2. Strip diacritics (ä→a, ö→o, ü→u, etc.)
    3. Lowercase
    4. Remove non-alphanumeric characters except spaces and hyphens
    5. Collapse multiple spaces

    Parameters
    ----------
    name:
        Raw player name string.

    Returns
    -------
    str
        Normalised name.
    """
    if not name:
        return ""

    # NFC normalisation
    normalised = unicodedata.normalize("NFC", name)

    # Strip diacritics by decomposing and filtering combining chars
    nfd = unicodedata.normalize("NFD", normalised)
    stripped = "".join(
        c for c in nfd if unicodedata.category(c) != "Mn"
    )

    # Lowercase and remove non-alphanumeric (except space/hyphen)
    cleaned = re.sub(r"[^a-z0-9 \-]", "", stripped.lower())

    # Collapse spaces
    return re.sub(r"\s+", " ", cleaned).strip()


def name_to_slug(name: str) -> str:
    """
    Convert a normalised name to a URL slug.

    Parameters
    ----------
    name:
        Player name (will be normalised internally).

    Returns
    -------
    str
        Hyphenated slug.
    """
    normalised = normalise_name(name)
    return re.sub(r"[\s\-]+", "-", normalised).strip("-")


def _token_sort_ratio(a: str, b: str) -> float:
    """
    Simple token-sort similarity in [0, 1].

    Tokens of both strings are sorted and compared with character-level
    edit distance approximation.  Pure Python — no external dep.
    """
    a_tokens = sorted(a.split())
    b_tokens = sorted(b.split())
    a_sorted = " ".join(a_tokens)
    b_sorted = " ".join(b_tokens)

    # Jaccard similarity on character n-grams (bigrams)
    def bigrams(s: str) -> set[str]:
        return {s[i: i + 2] for i in range(len(s) - 1)}

    bg_a = bigrams(a_sorted)
    bg_b = bigrams(b_sorted)
    if not bg_a and not bg_b:
        return 1.0
    if not bg_a or not bg_b:
        return 0.0
    intersection = len(bg_a & bg_b)
    union = len(bg_a | bg_b)
    return intersection / union


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlayerRecord:
    """
    A player record from a single data source.

    Attributes
    ----------
    source:
        Source system name (``"pdc"`` | ``"dartsorakel"`` | etc.).
    source_player_id:
        The ID or key within the source system (string-coerced).
    first_name:
        First name from the source.
    last_name:
        Last name from the source.
    full_name:
        Full name as provided by the source (used as primary for matching).
    slug:
        URL slug if the source provides one.
    country_code:
        ISO country code (2-letter).
    extra:
        Any additional source-specific fields (dict).
    """

    source: str
    source_player_id: str
    first_name: str
    last_name: str
    full_name: str
    slug: Optional[str] = None
    country_code: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in SOURCE_CONFIDENCE:
            raise DartsEntityResolutionError(
                f"Unknown source {self.source!r}. "
                f"Valid sources: {sorted(SOURCE_CONFIDENCE.keys())}"
            )

    @property
    def normalised_name(self) -> str:
        """Normalised full name for fuzzy matching."""
        return normalise_name(self.full_name)

    @property
    def computed_slug(self) -> str:
        """Slug from full name if source slug is absent."""
        return self.slug or name_to_slug(self.full_name)


@dataclass
class ResolvedPlayer:
    """
    The output of entity resolution — a canonical player record.

    Attributes
    ----------
    canonical_id:
        Internal UUID assigned by the resolution engine.
    full_name:
        Best-quality full name (from primary source).
    first_name:
        First name from primary source.
    last_name:
        Last name from primary source.
    slug:
        Canonical URL slug.
    country_code:
        ISO country code.
    source_ids:
        Mapping of source → source_player_id.
    source_confidence:
        Overall confidence score for this resolution in [0, 1].
    primary_source:
        The highest-confidence source that contributed data.
    match_method:
        How the match was achieved: ``"exact_id"`` | ``"exact_name"`` |
        ``"slug"`` | ``"fuzzy"`` | ``"single_source"``.
    """

    canonical_id: str
    full_name: str
    first_name: str
    last_name: str
    slug: str
    country_code: Optional[str]
    source_ids: dict[str, str]
    source_confidence: float
    primary_source: str
    match_method: str


# ---------------------------------------------------------------------------
# Resolution engine
# ---------------------------------------------------------------------------

class EntityResolver:
    """
    Resolve player identities across multiple data sources.

    Parameters
    ----------
    known_mappings:
        Optional pre-loaded cross-reference table:
        ``{(source_a, id_a): (source_b, id_b)}``.
        When a cross-reference is found, entity resolution is bypassed
        and the mapping is used directly with full confidence.
    """

    def __init__(
        self,
        known_mappings: Optional[dict[tuple[str, str], tuple[str, str]]] = None,
    ) -> None:
        self._known_mappings: dict[tuple[str, str], tuple[str, str]] = (
            known_mappings or {}
        )
        self._log = logger.bind(component="EntityResolver")

    def resolve_pair(
        self,
        record_a: PlayerRecord,
        record_b: PlayerRecord,
    ) -> tuple[bool, float, str]:
        """
        Determine if two player records refer to the same person.

        Parameters
        ----------
        record_a:
            First player record.
        record_b:
            Second player record.

        Returns
        -------
        tuple[bool, float, str]
            ``(is_match, confidence, method)``.
        """
        # 1. Known mapping table
        key_a = (record_a.source, record_a.source_player_id)
        key_b = (record_b.source, record_b.source_player_id)
        if key_a in self._known_mappings and self._known_mappings[key_a] == key_b:
            return True, _EXACT_MATCH_CONFIDENCE, "exact_id"
        if key_b in self._known_mappings and self._known_mappings[key_b] == key_a:
            return True, _EXACT_MATCH_CONFIDENCE, "exact_id"

        # 2. Slug match
        if record_a.computed_slug and record_b.computed_slug:
            if record_a.computed_slug == record_b.computed_slug:
                return True, _SLUG_MATCH_CONFIDENCE, "slug"

        # 3. Exact normalised name match
        name_a = record_a.normalised_name
        name_b = record_b.normalised_name
        if name_a and name_b and name_a == name_b:
            return True, _EXACT_MATCH_CONFIDENCE, "exact_name"

        # 4. Fuzzy name match
        if name_a and name_b:
            ratio = _token_sort_ratio(name_a, name_b)
            if ratio >= _FUZZY_HIGH_CONFIDENCE:
                return True, ratio * 0.95, "fuzzy"
            if ratio >= _FUZZY_LOW_CONFIDENCE:
                # Low-confidence — only accept if country codes match
                if (
                    record_a.country_code
                    and record_b.country_code
                    and record_a.country_code.upper() == record_b.country_code.upper()
                ):
                    return True, ratio * 0.90, "fuzzy"

        return False, 0.0, "no_match"

    def build_canonical(
        self,
        records: list[PlayerRecord],
        canonical_id: str,
    ) -> ResolvedPlayer:
        """
        Build a canonical player from a list of resolved matching records.

        The primary source is chosen by highest ``SOURCE_CONFIDENCE`` score.
        Name and slug are taken from the primary source.

        Parameters
        ----------
        records:
            All source records that have been resolved to the same entity.
        canonical_id:
            The UUID to assign to this canonical player.

        Returns
        -------
        ResolvedPlayer

        Raises
        ------
        DartsEntityResolutionError
            If ``records`` is empty.
        """
        if not records:
            raise DartsEntityResolutionError(
                "Cannot build canonical player from empty record list."
            )

        # Sort records by source confidence — primary source is first
        sorted_records = sorted(
            records,
            key=lambda r: SOURCE_CONFIDENCE.get(r.source, 0.0),
            reverse=True,
        )
        primary = sorted_records[0]

        source_ids = {r.source: r.source_player_id for r in records}

        # Overall confidence = weighted mean of source confidences
        total_conf = sum(
            SOURCE_CONFIDENCE.get(r.source, 0.5) for r in records
        )
        avg_conf = total_conf / len(records)
        # Boost for multi-source agreement
        if len(records) > 1:
            avg_conf = min(avg_conf * 1.05, 1.0)

        # Country code: prefer primary source, fall back to any non-None
        country_code = primary.country_code or next(
            (r.country_code for r in sorted_records if r.country_code), None
        )

        match_method = "multi_source" if len(records) > 1 else "single_source"

        resolved = ResolvedPlayer(
            canonical_id=canonical_id,
            full_name=primary.full_name,
            first_name=primary.first_name,
            last_name=primary.last_name,
            slug=primary.computed_slug,
            country_code=country_code,
            source_ids=source_ids,
            source_confidence=round(avg_conf, 4),
            primary_source=primary.source,
            match_method=match_method,
        )

        self._log.debug(
            "canonical_built",
            canonical_id=canonical_id,
            primary_source=primary.source,
            source_count=len(records),
            confidence=resolved.source_confidence,
        )
        return resolved

    def resolve_from_dartsorakel(
        self,
        dartsorakel_entry: dict,
    ) -> PlayerRecord:
        """
        Build a :class:`PlayerRecord` from a DartsOrakel JSON entry.

        Expected keys: ``player_key``, ``player_name``, ``country``,
        ``stat``, ``rank``, ``player_profile_url``.

        Parameters
        ----------
        dartsorakel_entry:
            Dict as found in ``stats_player.json``.

        Returns
        -------
        PlayerRecord

        Raises
        ------
        DartsEntityResolutionError
            If required keys are missing.
        """
        required = {"player_key", "player_name"}
        missing = required - dartsorakel_entry.keys()
        if missing:
            raise DartsEntityResolutionError(
                f"DartsOrakel entry missing required keys: {missing}"
            )

        full_name = dartsorakel_entry["player_name"]
        parts = full_name.strip().rsplit(" ", 1)
        first_name = parts[0] if len(parts) == 2 else ""
        last_name = parts[-1]

        return PlayerRecord(
            source="dartsorakel",
            source_player_id=str(dartsorakel_entry["player_key"]),
            first_name=first_name,
            last_name=last_name,
            full_name=full_name,
            country_code=dartsorakel_entry.get("country"),
            extra={
                "stat": dartsorakel_entry.get("stat"),
                "rank": dartsorakel_entry.get("rank"),
                "profile_url": dartsorakel_entry.get("player_profile_url"),
            },
        )

    def resolve_from_pdc_participant(
        self,
        participant_row: dict,
    ) -> PlayerRecord:
        """
        Build a :class:`PlayerRecord` from a PDC participants CSV row.

        Expected keys from ``participants.csv``:
        ``id``, ``first_name``, ``last_name``, ``nickname``,
        ``participant_slug``, ``country_code``.

        Parameters
        ----------
        participant_row:
            Dict from CSV row.

        Returns
        -------
        PlayerRecord

        Raises
        ------
        DartsEntityResolutionError
            If required keys are missing.
        """
        required = {"id", "first_name", "last_name"}
        missing = required - participant_row.keys()
        if missing:
            raise DartsEntityResolutionError(
                f"PDC participant row missing required keys: {missing}"
            )

        first = str(participant_row.get("first_name", "")).strip()
        last = str(participant_row.get("last_name", "")).strip()
        full_name = f"{first} {last}".strip()

        return PlayerRecord(
            source="pdc",
            source_player_id=str(participant_row["id"]),
            first_name=first,
            last_name=last,
            full_name=full_name,
            slug=participant_row.get("participant_slug"),
            country_code=participant_row.get("country_code"),
            extra={
                "nickname": participant_row.get("nickname"),
                "ranking": participant_row.get("ranking"),
                "prize_money": participant_row.get("prize_money"),
                "tour_card_holder": participant_row.get("tour_card_holder"),
            },
        )

    def find_best_match(
        self,
        query: PlayerRecord,
        candidates: list[PlayerRecord],
    ) -> Optional[tuple[PlayerRecord, float, str]]:
        """
        Find the best-matching candidate for a query record.

        Parameters
        ----------
        query:
            The record to match.
        candidates:
            Pool of candidate records to search.

        Returns
        -------
        Optional[tuple[PlayerRecord, float, str]]
            ``(best_candidate, confidence, method)`` or ``None`` if no
            match exceeds ``_MIN_ACCEPTABLE_CONFIDENCE``.
        """
        best_candidate: Optional[PlayerRecord] = None
        best_confidence = 0.0
        best_method = "no_match"

        for candidate in candidates:
            if candidate.source == query.source and candidate.source_player_id == query.source_player_id:
                continue  # skip self
            is_match, confidence, method = self.resolve_pair(query, candidate)
            if is_match and confidence > best_confidence:
                best_confidence = confidence
                best_candidate = candidate
                best_method = method

        if best_candidate is None or best_confidence < _MIN_ACCEPTABLE_CONFIDENCE:
            return None

        return best_candidate, best_confidence, best_method
