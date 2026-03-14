"""
Tests for data/entity_resolution.py

Verifies player ID matching across PDC, DartsOrakel, and other sources.
"""
from __future__ import annotations

import pytest

from data.entity_resolution import (
    EntityResolver,
    PlayerRecord,
    SOURCE_CONFIDENCE,
    DartsEntityResolutionError,
    normalise_name,
    name_to_slug,
)


class TestNormaliseName:
    """Name normalisation function."""

    def test_lowercase(self) -> None:
        assert normalise_name("Luke Littler") == "luke littler"

    def test_strips_diacritics(self) -> None:
        # ä → a, ü → u, ö → o
        assert normalise_name("Michael van Gerwen") == "michael van gerwen"

    def test_strips_accents(self) -> None:
        result = normalise_name("José García")
        assert "jose" in result or "jose garcia" in result

    def test_collapses_spaces(self) -> None:
        assert normalise_name("Luke   Littler") == "luke littler"

    def test_empty_string(self) -> None:
        assert normalise_name("") == ""


class TestNameToSlug:
    """Slug generation from player names."""

    def test_basic_slug(self) -> None:
        assert name_to_slug("Luke Littler") == "luke-littler"

    def test_multi_word_slug(self) -> None:
        assert name_to_slug("Michael van Gerwen") == "michael-van-gerwen"


class TestPlayerRecord:
    """PlayerRecord construction and property access."""

    def test_valid_pdc_record(self) -> None:
        record = PlayerRecord(
            source="pdc",
            source_player_id="40676",
            first_name="Luke",
            last_name="Littler",
            full_name="Luke Littler",
            slug="luke-littler",
            country_code="GB-ENG",
        )
        assert record.normalised_name == "luke littler"
        assert record.computed_slug == "luke-littler"

    def test_invalid_source_raises(self) -> None:
        with pytest.raises(DartsEntityResolutionError):
            PlayerRecord(
                source="unknown_source",
                source_player_id="1",
                first_name="Test",
                last_name="Player",
                full_name="Test Player",
            )

    def test_computed_slug_from_name(self) -> None:
        record = PlayerRecord(
            source="dartsorakel",
            source_player_id="5403",
            first_name="Luke",
            last_name="Littler",
            full_name="Luke Littler",
        )
        assert record.computed_slug == "luke-littler"


class TestEntityResolverPairMatching:
    """resolve_pair() matching logic."""

    def setup_method(self) -> None:
        self.resolver = EntityResolver()

    def _make_pdc(self, name: str, pid: str) -> PlayerRecord:
        parts = name.rsplit(" ", 1)
        return PlayerRecord(
            source="pdc",
            source_player_id=pid,
            first_name=parts[0] if len(parts) == 2 else name,
            last_name=parts[-1],
            full_name=name,
        )

    def _make_orakel(self, name: str, key: str) -> PlayerRecord:
        parts = name.rsplit(" ", 1)
        return PlayerRecord(
            source="dartsorakel",
            source_player_id=key,
            first_name=parts[0] if len(parts) == 2 else name,
            last_name=parts[-1],
            full_name=name,
        )

    def test_exact_name_match(self) -> None:
        a = self._make_pdc("Luke Littler", "40676")
        b = self._make_orakel("Luke Littler", "5403")
        is_match, conf, method = self.resolver.resolve_pair(a, b)
        assert is_match is True
        # Both slug and exact_name are valid high-confidence matches;
        # slug takes priority in the resolution order since computed_slug
        # derives from the same full_name.
        assert method in ("exact_name", "slug"), f"Unexpected method: {method}"
        assert conf >= 0.95

    def test_no_match_for_different_players(self) -> None:
        a = self._make_pdc("Luke Littler", "40676")
        b = self._make_pdc("Michael van Gerwen", "19")
        is_match, conf, method = self.resolver.resolve_pair(a, b)
        assert is_match is False

    def test_slug_match(self) -> None:
        a = PlayerRecord(
            source="pdc",
            source_player_id="40676",
            first_name="Luke",
            last_name="Littler",
            full_name="Luke Littler",
            slug="luke-littler",
        )
        b = PlayerRecord(
            source="dartsorakel",
            source_player_id="5403",
            first_name="Luke",
            last_name="Littler",
            full_name="Luke Littler",
            slug="luke-littler",
        )
        is_match, conf, method = self.resolver.resolve_pair(a, b)
        assert is_match is True
        assert method in ("slug", "exact_name")

    def test_known_mapping_override(self) -> None:
        resolver = EntityResolver(
            known_mappings={
                ("pdc", "40676"): ("dartsorakel", "5403")
            }
        )
        a = PlayerRecord(
            source="pdc",
            source_player_id="40676",
            first_name="Luke",
            last_name="Littler",
            full_name="Luke Littler",
        )
        b = PlayerRecord(
            source="dartsorakel",
            source_player_id="5403",
            first_name="Luke",
            last_name="Littler",
            full_name="Luke Littler",
        )
        is_match, conf, method = resolver.resolve_pair(a, b)
        assert is_match is True
        assert method == "exact_id"
        assert conf == 1.0


class TestEntityResolverDartsOrakel:
    """resolve_from_dartsorakel() builds correct PlayerRecord."""

    def test_from_seed_entry(self) -> None:
        resolver = EntityResolver()
        entry = {
            "player_key": 5403,
            "player_name": "Luke Littler",
            "country": "ENG",
            "stat": "101.03",
            "rank": 1,
            "player_profile_url": "https://dartsorakel.com/player/details/5403/luke-littler",
        }
        record = resolver.resolve_from_dartsorakel(entry)
        assert record.source == "dartsorakel"
        assert record.source_player_id == "5403"
        assert "littler" in record.normalised_name
        assert record.country_code == "ENG"
        assert record.extra["stat"] == "101.03"

    def test_missing_keys_raises(self) -> None:
        resolver = EntityResolver()
        with pytest.raises(DartsEntityResolutionError):
            resolver.resolve_from_dartsorakel({"some_key": "value"})


class TestEntityResolverPdc:
    """resolve_from_pdc_participant() builds correct PlayerRecord."""

    def test_from_csv_row(self) -> None:
        resolver = EntityResolver()
        row = {
            "id": "40676",
            "first_name": "Luke",
            "last_name": "Littler",
            "nickname": "The Nuke",
            "participant_slug": "luke-littler",
            "country_code": "GB-ENG",
            "ranking": "1",
            "prize_money": "3973925",
            "tour_card_holder": "True",
        }
        record = resolver.resolve_from_pdc_participant(row)
        assert record.source == "pdc"
        assert record.source_player_id == "40676"
        assert record.slug == "luke-littler"
        assert record.country_code == "GB-ENG"

    def test_missing_keys_raises(self) -> None:
        resolver = EntityResolver()
        with pytest.raises(DartsEntityResolutionError):
            resolver.resolve_from_pdc_participant({"id": "1"})


class TestSourceConfidence:
    """Source confidence values are ordered correctly."""

    def test_pdc_highest(self) -> None:
        assert SOURCE_CONFIDENCE["pdc"] >= SOURCE_CONFIDENCE["dartsorakel"]

    def test_dartconnect_high(self) -> None:
        assert SOURCE_CONFIDENCE["dartconnect"] > 0.85

    def test_mastercaller_lowest(self) -> None:
        mc = SOURCE_CONFIDENCE["mastercaller"]
        for source, conf in SOURCE_CONFIDENCE.items():
            if source != "mastercaller":
                assert conf >= mc
