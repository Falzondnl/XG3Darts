"""
Probabilistic starter inference.

When starter data is missing (no DartConnect or PDC official feed),
infer the leg starter from the alternating_starts rule plus the
first-leg starter (which may itself be uncertain).

Confidence propagates into margin widening downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LegStarterInfo:
    """
    Describes who starts a specific leg and how confident we are.

    Attributes
    ----------
    leg_number:
        1-based leg number within the current set/match.
    starter_player_id:
        The player_id of the inferred/confirmed starter, or None if truly unknown.
    source:
        Where this information came from.
        "dartconnect"  — DartConnect R2 per-dart feed (highest trust)
        "pdc_official" — PDC official XML/API feed
        "inferred"     — deterministic inference from alternating_starts rule
        "unknown"      — no basis for inference
    confidence:
        Float in [0, 1] representing confidence that starter_player_id is correct.
        1.0 = confirmed by authoritative feed
        0.9 = inferred with high certainty (leg 2 from confirmed leg 1 starter)
        0.5 = 50/50, no information
        <0.5 = possible but weak
    is_confirmed:
        True only if source is "dartconnect" or "pdc_official".
    """

    leg_number: int
    starter_player_id: Optional[str]
    source: str  # "dartconnect" | "pdc_official" | "inferred" | "unknown"
    confidence: float  # [0, 1]
    is_confirmed: bool

    def __post_init__(self) -> None:
        valid_sources = {"dartconnect", "pdc_official", "inferred", "unknown"}
        if self.source not in valid_sources:
            raise ValueError(f"Invalid source {self.source!r}. Valid: {valid_sources}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1]; got {self.confidence}")
        if self.is_confirmed and self.source not in ("dartconnect", "pdc_official"):
            raise ValueError(
                f"is_confirmed=True requires an authoritative source; got {self.source!r}"
            )


class StarterInferenceEngine:
    """
    Infers who starts each leg given available information.

    Algorithm
    ---------
    1. If a confirmed starter is known for any leg in the match, use it as anchor.
    2. Apply alternating_starts rule from the anchor forward and backward.
    3. Confidence decays with each inference step away from the anchor.
    4. If no anchor exists, return confidence=0.5 (uniform prior).

    Confidence decay model:
        confidence(n) = anchor_confidence * decay_factor ^ |n - anchor_leg|
    where decay_factor = 0.95 per leg.
    """

    INFERENCE_DECAY_FACTOR: float = 0.95
    """Per-leg confidence decay when inferring from an anchor."""

    def infer_starter(
        self,
        leg_number: int,
        players: tuple[str, str],
        alternating_starts: bool,
        previous_starters: Optional[Sequence[LegStarterInfo]] = None,
        feed_starter_id: Optional[str] = None,
        feed_confidence: float = 1.0,
        feed_source: str = "unknown",
    ) -> LegStarterInfo:
        """
        Infer the starter for a given leg number.

        Parameters
        ----------
        leg_number:
            The (1-based) leg number to infer.
        players:
            Tuple of (player1_id, player2_id).
        alternating_starts:
            Whether the format uses alternating-starts rule.
        previous_starters:
            Already-resolved starters for earlier/later legs (anchor pool).
        feed_starter_id:
            If a data feed has reported a starter for this exact leg, pass it here.
        feed_confidence:
            Confidence in the feed data (1.0 for DartConnect, 0.95 for PDC XML).
        feed_source:
            Source label of the feed ("dartconnect", "pdc_official", etc.).

        Returns
        -------
        LegStarterInfo
        """
        p1_id, p2_id = players

        # --- Direct feed confirmation ---
        if feed_starter_id is not None and feed_starter_id in (p1_id, p2_id):
            is_confirmed = feed_source in ("dartconnect", "pdc_official")
            logger.debug(
                "starter_from_feed",
                leg=leg_number,
                starter=feed_starter_id,
                source=feed_source,
                confidence=feed_confidence,
            )
            return LegStarterInfo(
                leg_number=leg_number,
                starter_player_id=feed_starter_id,
                source=feed_source,
                confidence=feed_confidence,
                is_confirmed=is_confirmed,
            )

        # --- No alternating starts: cannot infer ---
        if not alternating_starts:
            logger.debug(
                "starter_unknown_no_alternating",
                leg=leg_number,
            )
            return LegStarterInfo(
                leg_number=leg_number,
                starter_player_id=None,
                source="unknown",
                confidence=0.5,
                is_confirmed=False,
            )

        # --- Find best anchor from previous starters ---
        anchor = self._find_best_anchor(
            leg_number=leg_number,
            previous_starters=previous_starters or [],
        )

        if anchor is None:
            # No anchor available: uniform prior
            logger.debug("starter_no_anchor", leg=leg_number)
            return LegStarterInfo(
                leg_number=leg_number,
                starter_player_id=None,
                source="unknown",
                confidence=0.5,
                is_confirmed=False,
            )

        # --- Infer by alternating from anchor ---
        distance = abs(leg_number - anchor.leg_number)
        inferred_conf = anchor.confidence * (self.INFERENCE_DECAY_FACTOR ** distance)

        # Determine which player starts this leg based on parity from anchor.
        # If anchor player started leg A, the same player starts leg A+2k,
        # the other player starts leg A+2k+1.
        if anchor.starter_player_id is None:
            # Anchor itself is uncertain
            inferred_starter = None
            inferred_conf = 0.5
        else:
            if distance % 2 == 0:
                inferred_starter = anchor.starter_player_id
            else:
                inferred_starter = p2_id if anchor.starter_player_id == p1_id else p1_id

        logger.debug(
            "starter_inferred",
            leg=leg_number,
            inferred_starter=inferred_starter,
            anchor_leg=anchor.leg_number,
            distance=distance,
            confidence=round(inferred_conf, 4),
        )

        return LegStarterInfo(
            leg_number=leg_number,
            starter_player_id=inferred_starter,
            source="inferred",
            confidence=inferred_conf,
            is_confirmed=False,
        )

    def _find_best_anchor(
        self,
        leg_number: int,
        previous_starters: Sequence[LegStarterInfo],
    ) -> Optional[LegStarterInfo]:
        """
        Select the best anchor from the pool of known starters.

        Preference: confirmed anchors > inferred anchors.
        Tie-break: closest leg distance.
        """
        if not previous_starters:
            return None

        confirmed = [s for s in previous_starters if s.is_confirmed]
        candidates = confirmed if confirmed else list(previous_starters)

        # Sort by (confidence DESC, distance ASC)
        candidates.sort(
            key=lambda s: (-s.confidence, abs(s.leg_number - leg_number))
        )
        return candidates[0]

    def infer_match_starters(
        self,
        num_legs: int,
        players: tuple[str, str],
        alternating_starts: bool,
        confirmed_starters: Optional[dict[int, tuple[str, str, float]]] = None,
    ) -> list[LegStarterInfo]:
        """
        Infer starters for all legs in a match at once.

        Parameters
        ----------
        num_legs:
            Total number of legs to resolve.
        players:
            (player1_id, player2_id).
        alternating_starts:
            Whether format uses alternating starts.
        confirmed_starters:
            Dict mapping leg_number → (starter_id, source, confidence)
            for any legs where the starter is known from a feed.

        Returns
        -------
        list[LegStarterInfo]
            Length = num_legs, indexed from leg 1..num_legs.
        """
        confirmed_starters = confirmed_starters or {}
        resolved: list[LegStarterInfo] = []

        for leg in range(1, num_legs + 1):
            if leg in confirmed_starters:
                starter_id, source, conf = confirmed_starters[leg]
                info = self.infer_starter(
                    leg_number=leg,
                    players=players,
                    alternating_starts=alternating_starts,
                    previous_starters=resolved,
                    feed_starter_id=starter_id,
                    feed_confidence=conf,
                    feed_source=source,
                )
            else:
                info = self.infer_starter(
                    leg_number=leg,
                    players=players,
                    alternating_starts=alternating_starts,
                    previous_starters=resolved,
                )
            resolved.append(info)

        return resolved

    @staticmethod
    def margin_widening_factor(starter_confidence: float) -> float:
        """
        Compute the margin-widening multiplier for uncertain starter information.

        At full confidence (1.0) no widening is applied.
        At zero confidence (0.0) a 30% widening is applied.
        Interpolates linearly between these points.

        Parameters
        ----------
        starter_confidence:
            Float in [0, 1].

        Returns
        -------
        float
            Multiplier >= 1.0.

        Examples
        --------
        >>> StarterInferenceEngine.margin_widening_factor(1.0)
        1.0
        >>> StarterInferenceEngine.margin_widening_factor(0.5)
        1.15
        >>> StarterInferenceEngine.margin_widening_factor(0.0)
        1.3
        """
        if not (0.0 <= starter_confidence <= 1.0):
            raise ValueError(f"starter_confidence must be in [0, 1]; got {starter_confidence}")
        return 1.0 + (1.0 - starter_confidence) * 0.30

    def aggregate_match_confidence(
        self, starters: Sequence[LegStarterInfo]
    ) -> float:
        """
        Aggregate confidence across all legs in a match.

        Uses the minimum confidence (weakest link) as the conservative
        estimate for margin widening purposes.
        """
        if not starters:
            return 0.5
        return min(s.confidence for s in starters)
