"""Shared cross-sport SGP / Bet Builder standard contracts.

Defines the canonical interfaces, data models, validation taxonomy,
and pricing authority rules that all sport-specific SGP engines must
conform to for platform coherence.

This module does NOT contain sport-specific math (correlation matrices,
copula parameters, simulation engines). Those remain in each sport's
dedicated SGP engine. This module provides:

1. Canonical leg identity model
2. Validation result taxonomy
3. Pricing authority hierarchy rules
4. Output schema contract
5. Common utility functions

Usage::

    from shared.sgp_standard import (
        SGPLeg, SGPResult, ValidationResult, ValidationCode,
        validate_common_rules, clip_probability,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Canonical Leg Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SGPLeg:
    """Canonical same-game parlay leg identity.

    Every sport-specific SGP engine must accept legs in this format
    or provide a mapping from its native format to this standard.
    """

    event_id: str
    market_family: str        # e.g., "moneyline", "spread", "total", "team_total"
    market_subtype: str = ""  # e.g., "first_half", "quarter_1", "regulation"
    period: str = "full"      # "full", "first_half", "second_half", "quarter_N", "set_N"
    participant: str = ""     # team/player identity where relevant
    side: str = ""            # "home", "away", "over", "under", "yes", "no", "p1", "p2"
    line: Optional[float] = None  # spread/total line value
    probability: float = 0.0  # fair probability (no vig)
    source: str = "model"     # "model", "blended", "external"
    is_stale: bool = False
    is_suspended: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_key(self) -> str:
        """Unique identity key for dedup/conflict detection."""
        parts = [self.event_id, self.market_family, self.market_subtype,
                 self.period, self.participant, self.side]
        if self.line is not None:
            parts.append(f"L{self.line}")
        return "|".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Validation Taxonomy
# ---------------------------------------------------------------------------


class ValidationCode(str, Enum):
    """Standardized SGP validation result codes."""

    VALID = "valid"
    DUPLICATE_LEG = "duplicate_leg"
    CONTRADICTORY = "contradictory"
    SAME_MARKET_CONFLICT = "same_market_conflict"
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    UNSUPPORTED_FAMILY = "unsupported_family"
    STALE_LEG = "stale_leg"
    SUSPENDED_LEG = "suspended_leg"
    MIXED_EVENT = "mixed_event"
    TOO_MANY_LEGS = "too_many_legs"
    TOO_FEW_LEGS = "too_few_legs"
    PROBABILITY_OUT_OF_RANGE = "probability_out_of_range"
    PROP_IDENTITY_INSUFFICIENT = "prop_identity_insufficient"
    SHADOW_ONLY = "shadow_only"


@dataclass(frozen=True)
class ValidationResult:
    """Result of SGP leg validation."""

    valid: bool
    code: ValidationCode
    detail: str = ""
    affected_legs: tuple[int, ...] = ()  # indices of problematic legs


# ---------------------------------------------------------------------------
# Output Schema
# ---------------------------------------------------------------------------


@dataclass
class SGPResult:
    """Standardized SGP pricing result.

    All sport-specific engines should return this or a superset.
    """

    valid: bool
    rejection_reasons: list[str] = field(default_factory=list)

    # Pricing
    joint_probability: float = 0.0
    naive_probability: float = 0.0
    correlation_adjustment: float = 1.0  # joint / naive
    fair_decimal_odds: float = 0.0

    # Audit
    legs: list[dict[str, Any]] = field(default_factory=list)
    method: str = ""  # "gaussian_copula", "student_t", "markov", "analytical", "mc"
    correlation_matrix_used: list[list[float]] = field(default_factory=list)
    n_simulations: int = 0
    latency_ms: float = 0.0

    # Authority
    price_source: str = "model"  # "model", "blended", "live_blended"
    live_coherent: bool = False  # True if live SGP uses same authority as straight markets

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "rejection_reasons": self.rejection_reasons,
            "joint_probability": round(self.joint_probability, 8),
            "naive_probability": round(self.naive_probability, 8),
            "correlation_adjustment": round(self.correlation_adjustment, 4),
            "fair_decimal_odds": round(self.fair_decimal_odds, 2),
            "method": self.method,
            "price_source": self.price_source,
            "live_coherent": self.live_coherent,
            "n_simulations": self.n_simulations,
            "latency_ms": round(self.latency_ms, 2),
            "n_legs": len(self.legs),
        }


# ---------------------------------------------------------------------------
# Common Validation Utilities
# ---------------------------------------------------------------------------


def clip_probability(p: float, lo: float = 0.02, hi: float = 0.98) -> float:
    """Clip probability to safe range."""
    return max(lo, min(hi, p))


def validate_common_rules(
    legs: list[SGPLeg],
    max_legs: int = 12,
    min_legs: int = 2,
    supported_families: Optional[set[str]] = None,
) -> list[ValidationResult]:
    """Run common cross-sport validation rules.

    Sport-specific engines should call this first, then add their own rules.
    """
    results: list[ValidationResult] = []

    # Leg count
    if len(legs) < min_legs:
        results.append(ValidationResult(
            valid=False,
            code=ValidationCode.TOO_FEW_LEGS,
            detail=f"Need at least {min_legs} legs, got {len(legs)}",
        ))
        return results

    if len(legs) > max_legs:
        results.append(ValidationResult(
            valid=False,
            code=ValidationCode.TOO_MANY_LEGS,
            detail=f"Maximum {max_legs} legs allowed, got {len(legs)}",
        ))
        return results

    # Mixed events
    event_ids = {leg.event_id for leg in legs}
    if len(event_ids) > 1:
        results.append(ValidationResult(
            valid=False,
            code=ValidationCode.MIXED_EVENT,
            detail=f"SGP requires same event, found {len(event_ids)} events",
        ))
        return results

    # Stale / suspended legs
    for i, leg in enumerate(legs):
        if leg.is_stale:
            results.append(ValidationResult(
                valid=False,
                code=ValidationCode.STALE_LEG,
                detail=f"Leg {i} ({leg.market_family}) is stale",
                affected_legs=(i,),
            ))
        if leg.is_suspended:
            results.append(ValidationResult(
                valid=False,
                code=ValidationCode.SUSPENDED_LEG,
                detail=f"Leg {i} ({leg.market_family}) is suspended",
                affected_legs=(i,),
            ))

    # Probability range
    for i, leg in enumerate(legs):
        if leg.probability <= 0.0 or leg.probability >= 1.0:
            results.append(ValidationResult(
                valid=False,
                code=ValidationCode.PROBABILITY_OUT_OF_RANGE,
                detail=f"Leg {i} probability {leg.probability} out of (0, 1)",
                affected_legs=(i,),
            ))

    # Unsupported families
    if supported_families:
        for i, leg in enumerate(legs):
            if leg.market_family not in supported_families:
                results.append(ValidationResult(
                    valid=False,
                    code=ValidationCode.UNSUPPORTED_FAMILY,
                    detail=f"Leg {i} family '{leg.market_family}' not supported",
                    affected_legs=(i,),
                ))

    # Duplicate detection
    keys_seen: dict[str, int] = {}
    for i, leg in enumerate(legs):
        key = leg.canonical_key
        if key in keys_seen:
            results.append(ValidationResult(
                valid=False,
                code=ValidationCode.DUPLICATE_LEG,
                detail=f"Leg {i} duplicates leg {keys_seen[key]}",
                affected_legs=(keys_seen[key], i),
            ))
        else:
            keys_seen[key] = i

    # Contradictory sides (same market, opposite sides)
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            a, b = legs[i], legs[j]
            if (a.market_family == b.market_family
                    and a.market_subtype == b.market_subtype
                    and a.period == b.period
                    and a.line == b.line
                    and a.side != b.side
                    and a.side and b.side):
                # Check for contradictory pairs
                contra_pairs = [
                    ("home", "away"), ("over", "under"),
                    ("yes", "no"), ("p1", "p2"),
                ]
                for s1, s2 in contra_pairs:
                    if {a.side, b.side} == {s1, s2}:
                        results.append(ValidationResult(
                            valid=False,
                            code=ValidationCode.CONTRADICTORY,
                            detail=f"Legs {i} and {j}: {a.side} vs {b.side} on {a.market_family}",
                            affected_legs=(i, j),
                        ))

    return results


# ---------------------------------------------------------------------------
# Pricing Authority Hierarchy
# ---------------------------------------------------------------------------


class PricingAuthority(str, Enum):
    """Source authority level for SGP leg probabilities."""

    LIVE_BLENDED = "live_blended"      # Best: live model + external anchor
    PREMATCH_BLENDED = "prematch_blended"  # Good: pre-match model + Pinnacle
    LIVE_MODEL_ONLY = "live_model_only"  # OK: live model without anchor
    PREMATCH_MODEL_ONLY = "prematch_model_only"  # Acceptable
    STALE = "stale"                    # Reject
    UNKNOWN = "unknown"                # Reject


def resolve_leg_authority(leg: SGPLeg) -> PricingAuthority:
    """Determine the pricing authority level of a leg."""
    if leg.is_stale or leg.is_suspended:
        return PricingAuthority.STALE
    if leg.source == "live_blended":
        return PricingAuthority.LIVE_BLENDED
    if leg.source == "blended":
        return PricingAuthority.PREMATCH_BLENDED
    if leg.source == "live_model":
        return PricingAuthority.LIVE_MODEL_ONLY
    if leg.source == "model":
        return PricingAuthority.PREMATCH_MODEL_ONLY
    return PricingAuthority.UNKNOWN


def check_authority_coherence(legs: list[SGPLeg]) -> tuple[bool, str]:
    """Check that all legs use coherent pricing authority.

    Returns (coherent, reason).
    SGP should not mix live-blended legs with stale pre-match legs.
    """
    authorities = [resolve_leg_authority(leg) for leg in legs]

    if PricingAuthority.STALE in authorities:
        return False, "One or more legs have stale pricing authority"

    if PricingAuthority.UNKNOWN in authorities:
        return False, "One or more legs have unknown pricing authority"

    # Mixed live+prematch is acceptable but should be documented
    has_live = any(a in (PricingAuthority.LIVE_BLENDED, PricingAuthority.LIVE_MODEL_ONLY) for a in authorities)
    has_prematch = any(a in (PricingAuthority.PREMATCH_BLENDED, PricingAuthority.PREMATCH_MODEL_ONLY) for a in authorities)

    if has_live and has_prematch:
        return True, "mixed_live_prematch"

    return True, "coherent"
