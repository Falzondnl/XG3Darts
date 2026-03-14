"""
Tests for compliance/gdpr_anonymizer.py

Verifies anonymization of player data, consent checking, and right-to-erasure.
"""
from __future__ import annotations

import os

import pytest

from compliance.gdpr_anonymizer import (
    PERSONAL_DATA_FIELDS,
    PUBLIC_STATS_FIELDS,
    AnonymizationResult,
    ConsentRecord,
    ConsentRequiredError,
    DartsGdprError,
    anonymize_player_dict,
    apply_right_to_erasure,
    check_consent,
    filter_personal_fields,
    validate_name_for_pii,
)


@pytest.fixture()
def sample_player() -> dict:
    return {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "first_name": "Luke",
        "last_name": "Littler",
        "nickname": "The Nuke",
        "dob": "2007-01-21",
        "country_code": "GB-ENG",
        "slug": "luke-littler",
        "pdc_id": 40676,
        "dartsorakel_key": 5403,
        "dartsdatabase_id": None,
        "dartconnect_id": None,
        "wdf_id": None,
        "three_dart_average": 101.03,
        "pdc_ranking": 1,
        "tour_card_holder": True,
        "gdpr_anonymized": False,
        "gdpr_anonymized_at": None,
    }


class TestAnonymizePlayerDict:
    """anonymize_player_dict() replaces personal fields."""

    def test_first_last_name_replaced(self, sample_player: dict) -> None:
        anon, result = anonymize_player_dict(sample_player)
        assert anon["first_name"] == "ANONYMIZED"
        assert anon["last_name"] == "PLAYER"

    def test_gdpr_anonymized_flag_set(self, sample_player: dict) -> None:
        anon, result = anonymize_player_dict(sample_player)
        assert anon["gdpr_anonymized"] is True
        assert anon["gdpr_anonymized_at"] is not None

    def test_stats_retained_by_default(self, sample_player: dict) -> None:
        anon, result = anonymize_player_dict(sample_player)
        assert anon["three_dart_average"] == 101.03
        assert anon["pdc_ranking"] == 1

    def test_stats_cleared_when_retain_false(self, sample_player: dict) -> None:
        anon, result = anonymize_player_dict(sample_player, retain_stats=False)
        assert anon.get("three_dart_average") is None

    def test_id_preserved(self, sample_player: dict) -> None:
        anon, result = anonymize_player_dict(sample_player)
        assert anon["id"] == sample_player["id"]

    def test_returns_anonymization_result(self, sample_player: dict) -> None:
        anon, result = anonymize_player_dict(sample_player)
        assert isinstance(result, AnonymizationResult)
        assert result.operation == "anonymize"
        assert result.player_id == sample_player["id"]
        assert len(result.fields_cleared) > 0

    def test_fields_cleared_list_contains_names(self, sample_player: dict) -> None:
        anon, result = anonymize_player_dict(sample_player)
        assert "first_name" in result.fields_cleared
        assert "last_name" in result.fields_cleared

    def test_missing_id_raises(self) -> None:
        with pytest.raises(DartsGdprError):
            anonymize_player_dict({"first_name": "Luke"})

    def test_original_not_mutated(self, sample_player: dict) -> None:
        original_name = sample_player["first_name"]
        anonymize_player_dict(sample_player)
        assert sample_player["first_name"] == original_name


class TestRightToErasure:
    """apply_right_to_erasure() full erasure including stats."""

    def test_stats_cleared_in_erasure(self, sample_player: dict) -> None:
        erased, result = apply_right_to_erasure(sample_player)
        assert erased.get("three_dart_average") is None

    def test_erasure_operation_label(self, sample_player: dict) -> None:
        erased, result = apply_right_to_erasure(sample_player)
        assert result.operation == "erase"

    def test_gdpr_flag_set_after_erasure(self, sample_player: dict) -> None:
        erased, result = apply_right_to_erasure(sample_player)
        assert erased["gdpr_anonymized"] is True


class TestCheckConsent:
    """check_consent() enforcement."""

    def test_consent_given_returns_record(self) -> None:
        record = check_consent("player-uuid", True, "v1.0")
        assert isinstance(record, ConsentRecord)
        assert record.consent_given is True

    def test_no_consent_raises(self) -> None:
        with pytest.raises(ConsentRequiredError):
            check_consent("player-uuid", False, None)

    def test_consent_version_recorded(self) -> None:
        record = check_consent("player-uuid", True, "v2.0")
        assert record.consent_version == "v2.0"


class TestFilterPersonalFields:
    """filter_personal_fields() removes PII for non-consented access."""

    def test_with_consent_returns_all(self, sample_player: dict) -> None:
        filtered = filter_personal_fields(sample_player, has_consent=True)
        assert filtered["first_name"] == sample_player["first_name"]

    def test_without_consent_removes_personal(self, sample_player: dict) -> None:
        filtered = filter_personal_fields(sample_player, has_consent=False)
        for field in PERSONAL_DATA_FIELDS:
            assert field not in filtered

    def test_without_consent_retains_stats(self, sample_player: dict) -> None:
        filtered = filter_personal_fields(sample_player, has_consent=False)
        assert "three_dart_average" in filtered
        assert filtered["three_dart_average"] == 101.03


class TestValidateNameForPii:
    """validate_name_for_pii() heuristic."""

    def test_proper_name_detected(self) -> None:
        assert validate_name_for_pii("Luke Littler") is True

    def test_single_word_not_name(self) -> None:
        assert validate_name_for_pii("Littler") is False

    def test_all_lowercase_not_detected(self) -> None:
        # Our heuristic requires capitalised words
        assert validate_name_for_pii("luke littler") is False

    def test_stat_value_not_detected(self) -> None:
        assert validate_name_for_pii("101.03") is False
