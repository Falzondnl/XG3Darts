"""
GDPR compliance module.

Handles:
- Anonymization of personal data in player records
- Consent checking before returning personal data
- Right-to-erasure (Article 17 GDPR) handling
- Pseudonymization of player identifiers in exported datasets

All operations are audit-logged via structlog.  No personal data is
ever written to log files — only internal UUIDs and operation timestamps.

Definitions
-----------
Personal data fields for darts players (Article 4 GDPR):
    - ``first_name``, ``last_name`` (identifiable individually or combined)
    - ``nickname`` (often publicly known but still personal data)
    - ``dob`` (date of birth)
    - ``country_code`` (when combined with name, adds precision)
    - ``slug`` (derived from name, identifiable)

Anonymization replaces these fields with non-reversible tokens or removes
them entirely.  The ``id`` (UUID) is retained so referential integrity
within the database is preserved, but the UUID itself is pseudonymous.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import structlog


logger = structlog.get_logger(__name__)


class DartsGdprError(Exception):
    """Raised when a GDPR operation cannot be completed safely."""


class ConsentRequiredError(DartsGdprError):
    """Raised when personal data is requested but consent has not been given."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fields considered personal under GDPR Article 4 for this domain
PERSONAL_DATA_FIELDS: frozenset[str] = frozenset(
    {
        "first_name",
        "last_name",
        "nickname",
        "dob",
        "country_code",
        "slug",
        "pdc_id",
        "dartsorakel_key",
        "dartsdatabase_id",
        "dartconnect_id",
        "wdf_id",
    }
)

# Fields that are publicly available statistics — not personal data under
# GDPR when the player record is anonymized (no name linkage)
PUBLIC_STATS_FIELDS: frozenset[str] = frozenset(
    {
        "three_dart_average",
        "first_nine_average",
        "checkout_percentage",
        "hold_rate",
        "break_rate",
        "count_180",
        "matches_played",
        "matches_won",
        "dartsorakel_3da",
        "dartsorakel_rank",
        "pdc_ranking",
    }
)

_ANONYMIZED_FIRST_NAME = "ANONYMIZED"
_ANONYMIZED_LAST_NAME = "PLAYER"
_ANONYMIZED_NICKNAME = None
_ANONYMIZED_DOB = None
_ANONYMIZED_COUNTRY_CODE = None
_ANONYMIZED_SLUG = None

# HMAC secret for pseudonymization — loaded from env; never hardcoded.
# The secret must be set via the GDPR_PSEUDONYM_SECRET environment variable.
_PSEUDONYM_SECRET: Optional[str] = None


def _get_pseudonym_secret() -> bytes:
    """
    Return the HMAC secret for pseudonymization.

    Raises
    ------
    DartsGdprError
        If the secret has not been configured.
    """
    import os  # lazy import to avoid module-level env dependency

    secret = os.environ.get("GDPR_PSEUDONYM_SECRET")
    if not secret:
        raise DartsGdprError(
            "GDPR_PSEUDONYM_SECRET environment variable is not set. "
            "Pseudonymization cannot proceed without a secure secret."
        )
    return secret.encode("utf-8")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AnonymizationResult:
    """
    Result of an anonymization operation.

    Attributes
    ----------
    player_id:
        Internal UUID of the player record (retained for referential integrity).
    fields_cleared:
        Names of fields that were set to None or replaced with tokens.
    anonymized_at:
        UTC timestamp of the operation.
    operation:
        ``"anonymize"`` | ``"pseudonymize"`` | ``"erase"``
    """

    player_id: str
    fields_cleared: list[str]
    anonymized_at: datetime
    operation: str


@dataclass
class ConsentRecord:
    """
    Lightweight consent check result.

    Attributes
    ----------
    player_id:
        Internal UUID of the player.
    consent_given:
        Whether active, non-withdrawn consent exists.
    consent_version:
        The consent version string on record.
    checked_at:
        UTC timestamp of the check.
    """

    player_id: str
    consent_given: bool
    consent_version: Optional[str]
    checked_at: datetime


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def anonymize_player_dict(
    player_dict: dict[str, Any],
    *,
    retain_stats: bool = True,
) -> tuple[dict[str, Any], AnonymizationResult]:
    """
    Anonymize personal data fields in a player dictionary.

    The original dictionary is not mutated.  A new dictionary is returned
    with personal fields cleared.  Non-personal statistical fields are
    retained if ``retain_stats=True``.

    Parameters
    ----------
    player_dict:
        A dict representing a player record (matches :class:`db.models.DartsPlayer`
        column names).
    retain_stats:
        If True, retain aggregate statistics fields (3DA, checkout%, etc.)
        which are not personal data in isolation.

    Returns
    -------
    tuple[dict[str, Any], AnonymizationResult]
        The anonymized dict and an audit record.

    Raises
    ------
    DartsGdprError
        If ``player_id`` is missing from the input dict.
    """
    player_id = player_dict.get("id")
    if not player_id:
        raise DartsGdprError(
            "player_dict must contain 'id' field for GDPR audit trail."
        )

    anonymized = dict(player_dict)
    fields_cleared: list[str] = []

    # Replace personal fields
    _replacements = {
        "first_name": _ANONYMIZED_FIRST_NAME,
        "last_name": _ANONYMIZED_LAST_NAME,
        "nickname": _ANONYMIZED_NICKNAME,
        "dob": _ANONYMIZED_DOB,
        "country_code": _ANONYMIZED_COUNTRY_CODE,
        "slug": _ANONYMIZED_SLUG,
        # Source IDs are personal when they allow re-identification via
        # external systems
        "pdc_id": None,
        "dartsorakel_key": None,
        "dartsdatabase_id": None,
        "dartconnect_id": None,
        "wdf_id": None,
    }

    for field_name, replacement in _replacements.items():
        if field_name in anonymized:
            original_value = anonymized[field_name]
            if original_value is not None:
                anonymized[field_name] = replacement
                fields_cleared.append(field_name)

    if not retain_stats:
        # Remove stats too (e.g., for full erasure)
        for stat_field in PUBLIC_STATS_FIELDS:
            if stat_field in anonymized and anonymized[stat_field] is not None:
                anonymized[stat_field] = None
                fields_cleared.append(stat_field)

    anonymized["gdpr_anonymized"] = True
    now = datetime.now(tz=timezone.utc)
    anonymized["gdpr_anonymized_at"] = now

    result = AnonymizationResult(
        player_id=str(player_id),
        fields_cleared=fields_cleared,
        anonymized_at=now,
        operation="anonymize",
    )

    logger.info(
        "gdpr_anonymize",
        player_id=str(player_id),
        fields_cleared_count=len(fields_cleared),
        retain_stats=retain_stats,
    )
    return anonymized, result


def pseudonymize_player_id(player_id: str) -> str:
    """
    Return a deterministic pseudonym for a player UUID.

    Uses HMAC-SHA256 with a secret key so the pseudonym is:
    - Deterministic (same input → same output with same key)
    - Non-reversible without the key
    - Safe for use in exported datasets and logs

    Parameters
    ----------
    player_id:
        The internal UUID string of the player.

    Returns
    -------
    str
        A hex-encoded pseudonym (64 characters).

    Raises
    ------
    DartsGdprError
        If the HMAC secret is not configured.
    """
    secret = _get_pseudonym_secret()
    mac = hmac.new(secret, player_id.encode("utf-8"), hashlib.sha256)
    pseudonym = mac.hexdigest()
    logger.debug("gdpr_pseudonymize", pseudonym_prefix=pseudonym[:8])
    return pseudonym


def check_consent(
    player_id: str,
    consent_given: bool,
    consent_version: Optional[str],
) -> ConsentRecord:
    """
    Check whether a player has given valid GDPR consent.

    Parameters
    ----------
    player_id:
        Internal UUID of the player.
    consent_given:
        Whether the consent record shows consent was given and not withdrawn.
    consent_version:
        The consent version string on record.

    Returns
    -------
    ConsentRecord

    Raises
    ------
    ConsentRequiredError
        If consent has not been given.
    """
    record = ConsentRecord(
        player_id=player_id,
        consent_given=consent_given,
        consent_version=consent_version,
        checked_at=datetime.now(tz=timezone.utc),
    )
    if not consent_given:
        raise ConsentRequiredError(
            f"Player {player_id} has not given GDPR consent (or consent "
            "has been withdrawn). Personal data cannot be returned."
        )
    return record


def apply_right_to_erasure(
    player_dict: dict[str, Any],
) -> tuple[dict[str, Any], AnonymizationResult]:
    """
    Apply Article 17 GDPR right-to-erasure to a player record.

    This is a stronger operation than anonymization — ALL personal fields
    AND statistics are cleared.  The record remains in the database to
    preserve referential integrity of historical match records, but
    becomes non-identifiable.

    Parameters
    ----------
    player_dict:
        A dict representing a player record.

    Returns
    -------
    tuple[dict[str, Any], AnonymizationResult]
        The erased dict and an audit record.
    """
    # Anonymize with stats also cleared
    erased, result = anonymize_player_dict(player_dict, retain_stats=False)
    # Override operation label
    result.operation = "erase"

    logger.info(
        "gdpr_right_to_erasure",
        player_id=result.player_id,
        fields_cleared_count=len(result.fields_cleared),
    )
    return erased, result


def filter_personal_fields(
    player_dict: dict[str, Any],
    *,
    has_consent: bool,
) -> dict[str, Any]:
    """
    Return a version of the player dict safe to return to callers.

    If ``has_consent`` is False, all personal data fields are removed from
    the response dict.  Statistical fields are always returned.

    Parameters
    ----------
    player_dict:
        Raw player dict from the database.
    has_consent:
        Whether the requesting context has consent to view personal data.

    Returns
    -------
    dict[str, Any]
        Filtered player dict.
    """
    if has_consent:
        return dict(player_dict)

    # Remove personal fields for non-consented access
    filtered = {}
    for key, value in player_dict.items():
        if key not in PERSONAL_DATA_FIELDS:
            filtered[key] = value

    logger.debug(
        "gdpr_filter_personal_fields",
        fields_removed=[k for k in player_dict if k in PERSONAL_DATA_FIELDS],
        has_consent=False,
    )
    return filtered


def validate_name_for_pii(name: str) -> bool:
    """
    Return True if the string looks like it contains personal name data.

    Used as a guard in logging pipelines to prevent accidental PII leakage
    to log files.

    Parameters
    ----------
    name:
        A string to check.

    Returns
    -------
    bool
        True if the string appears to be a proper name.
    """
    # Simple heuristic: proper name has ≥2 alphabetic words, each capitalised
    words = name.strip().split()
    if len(words) < 2:
        return False
    return all(
        bool(re.match(r"^[A-Z][a-z]+", w)) for w in words if w.isalpha()
    )
