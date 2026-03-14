"""
Data sufficiency gates for prop markets.

Before any prop market is opened, the gate verifies that the player-specific
statistics meet the minimum requirements for the regime and market family.

Gate checks
-----------
1. Minimum legs observed threshold
2. Required stat fields are present (non-None, non-NaN)
3. Minimum confidence score
4. R2-required markets are blocked in R0/R1 regimes
5. WDF / women's margin multiplier is annotated on the result

All checks are pure functions over the provided ``stats`` dict and
``regime`` integer (0/1/2) — no database calls.
"""
from __future__ import annotations

import math
from typing import Any

import structlog

from engines.errors import DartsDataError

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------------------

DATA_SUFFICIENCY_GATES: dict[str, dict[str, Any]] = {
    "props_180": {
        "min_legs_observed": 50,
        "required_coverage": ["pct_180_ewm"],
        "min_confidence": 0.7,
    },
    "props_checkout": {
        "min_legs_observed": 30,
        "required_coverage": ["checkout_ewm", "checkout_pct_by_band"],
        "min_confidence": 0.6,
        "r2_preferred": True,
    },
    "props_double_segment": {
        "min_legs_observed": 100,
        "required_coverage": ["double_hit_rate", "eb_double_accuracy"],
        "min_confidence": 0.8,
        "r2_required": True,
    },
    "props_next_visit": {
        "min_legs_observed": 200,
        "r2_required": True,
        "min_confidence": 0.85,
    },
    "wdf_womens_props": {
        "min_legs_observed": 20,
        "margin_multiplier": 1.30,
    },
}


class DataSufficiencyGate:
    """
    Evaluate data sufficiency gates before opening a prop market.

    Usage::

        gate = DataSufficiencyGate()
        ok, reason = gate.can_open_market(
            market_family="props_180",
            player_id="player_uuid",
            regime=1,
            stats={
                "legs_observed": 80,
                "pct_180_ewm": 0.042,
                "confidence": 0.75,
            },
        )
        if not ok:
            raise DartsMarketClosedError(reason)
    """

    def can_open_market(
        self,
        market_family: str,
        player_id: str,
        regime: int,
        stats: dict[str, Any],
    ) -> tuple[bool, str]:
        """
        Determine whether a prop market can be opened for this player.

        Parameters
        ----------
        market_family:
            Gate key from DATA_SUFFICIENCY_GATES (e.g. ``"props_180"``).
        player_id:
            Canonical player identifier (for logging).
        regime:
            Data coverage regime: 0 = R0, 1 = R1, 2 = R2.
        stats:
            Player statistics dict.  Expected keys depend on the gate
            definition (see DATA_SUFFICIENCY_GATES).

        Returns
        -------
        (can_open, reason):
            can_open is True when all gate checks pass.
            reason is a human-readable explanation (empty string when passing).

        Raises
        ------
        DartsDataError
            If market_family is not registered.
        """
        if market_family not in DATA_SUFFICIENCY_GATES:
            raise DartsDataError(
                f"Unknown market family {market_family!r}. "
                f"Known: {sorted(DATA_SUFFICIENCY_GATES.keys())}"
            )

        if regime not in (0, 1, 2):
            raise DartsDataError(
                f"regime must be 0, 1, or 2; got {regime!r}"
            )

        gate = DATA_SUFFICIENCY_GATES[market_family]

        # ------------------------------------------------------------------
        # Check 1: R2 required — block in R0/R1
        # ------------------------------------------------------------------
        if gate.get("r2_required", False) and regime < 2:
            reason = (
                f"Market '{market_family}' requires R2 (visit-level) data. "
                f"Player '{player_id}' is at regime R{regime}. Market blocked."
            )
            logger.info(
                "prop_gate_r2_required_failed",
                market_family=market_family,
                player_id=player_id,
                regime=regime,
            )
            return False, reason

        # ------------------------------------------------------------------
        # Check 2: Minimum legs observed
        # ------------------------------------------------------------------
        min_legs = gate.get("min_legs_observed", 0)
        if min_legs > 0:
            observed = stats.get("legs_observed", 0)
            if not isinstance(observed, (int, float)) or observed < min_legs:
                reason = (
                    f"Market '{market_family}': insufficient legs observed for "
                    f"player '{player_id}'. Required: {min_legs}, "
                    f"got: {observed}."
                )
                logger.info(
                    "prop_gate_insufficient_legs",
                    market_family=market_family,
                    player_id=player_id,
                    min_legs=min_legs,
                    observed=observed,
                )
                return False, reason

        # ------------------------------------------------------------------
        # Check 3: Required coverage fields present and non-null
        # ------------------------------------------------------------------
        required_fields: list[str] = gate.get("required_coverage", [])
        for field_name in required_fields:
            value = stats.get(field_name)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                reason = (
                    f"Market '{market_family}': required stat '{field_name}' is "
                    f"missing or NaN for player '{player_id}'."
                )
                logger.info(
                    "prop_gate_missing_field",
                    market_family=market_family,
                    player_id=player_id,
                    missing_field=field_name,
                )
                return False, reason

        # ------------------------------------------------------------------
        # Check 4: Minimum confidence
        # ------------------------------------------------------------------
        min_confidence = gate.get("min_confidence", 0.0)
        if min_confidence > 0.0:
            confidence = stats.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)):
                confidence = 0.0
            if confidence < min_confidence:
                reason = (
                    f"Market '{market_family}': confidence too low for "
                    f"player '{player_id}'. Required: {min_confidence:.2f}, "
                    f"got: {confidence:.2f}."
                )
                logger.info(
                    "prop_gate_confidence_failed",
                    market_family=market_family,
                    player_id=player_id,
                    min_confidence=min_confidence,
                    actual_confidence=confidence,
                )
                return False, reason

        logger.debug(
            "prop_gate_passed",
            market_family=market_family,
            player_id=player_id,
            regime=regime,
        )
        return True, ""

    def get_margin_multiplier(self, market_family: str) -> float:
        """
        Return the margin multiplier for a given market family.

        Returns 1.0 if no multiplier is defined.

        Parameters
        ----------
        market_family:
            Gate key from DATA_SUFFICIENCY_GATES.

        Raises
        ------
        DartsDataError
            If market_family is not registered.
        """
        if market_family not in DATA_SUFFICIENCY_GATES:
            raise DartsDataError(
                f"Unknown market family {market_family!r}."
            )
        return DATA_SUFFICIENCY_GATES[market_family].get("margin_multiplier", 1.0)

    def is_r2_required(self, market_family: str) -> bool:
        """Return True when the market family requires R2 data."""
        if market_family not in DATA_SUFFICIENCY_GATES:
            raise DartsDataError(
                f"Unknown market family {market_family!r}."
            )
        return bool(DATA_SUFFICIENCY_GATES[market_family].get("r2_required", False))
