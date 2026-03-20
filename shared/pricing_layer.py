"""Shared cross-sport Pinnacle / exchange logit-space blend layer.

Provides the canonical pricing hierarchy used by all XG3 sport microservices:

    Internal ML model  →  authoritative core probability
    Pinnacle / sharp   →  primary external anchor
    Softbook / weaker  →  secondary reference (lower weight)

Blend happens in LOGIT SPACE (not probability space) per the Tennis MS
standard.  This preserves multiplicative structure and avoids the well-known
bias of linear probability averaging near 0 and 1.

Usage::

    from shared.pricing_layer import PricingLayer, BlendConfig

    layer = PricingLayer(
        sport="basketball",
        optic_odds_api_key=os.getenv("OPTIC_ODDS_API_KEY", ""),
    )

    # Pre-match: fetch Pinnacle odds and blend with model
    result = await layer.blend_pre_match(
        fixture_id="abc123",
        model_prob=0.62,       # P(home/p1 wins) from ML model
        market_type="moneyline",
    )
    # result.blended_prob → 0.608  (logit-blended)
    # result.model_prob   → 0.62   (raw model)
    # result.market_prob  → 0.59   (devigged Pinnacle)
    # result.source       → "pinnacle"
    # result.blend_weight → 0.20   (model weight)

Architecture notes:
    - Internal model remains authoritative (weight 0.20-0.40 depending on sport/tier)
    - Pinnacle weight is 0.60-0.80 (sharp market is the anchor, model is the edge)
    - When Pinnacle unavailable: model-only with protective clamp [0.05, 0.95]
    - Derivatives MUST consume the blended probability, not the raw model probability
    - SGP / bet builder MUST use the same blended pricing ecosystem
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPTIC_ODDS_BASE = "https://api.opticodds.com"
_DEFAULT_TIMEOUT = 8.0
_STALE_THRESHOLD_S = 300.0  # 5 minutes
_CLAMP_LO = 0.02
_CLAMP_HI = 0.98
_FALLBACK_CLAMP_LO = 0.05
_FALLBACK_CLAMP_HI = 0.95


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlendConfig:
    """Per-sport blend configuration."""

    sport: str
    model_weight: float = 0.20  # Tennis standard: 20% model, 80% Pinnacle
    pinnacle_weight: float = 0.80
    softbook_weight: float = 0.65  # If Pinnacle unavailable, softbook gets less trust
    stale_threshold_s: float = _STALE_THRESHOLD_S
    fallback_clamp_lo: float = _FALLBACK_CLAMP_LO
    fallback_clamp_hi: float = _FALLBACK_CLAMP_HI
    enabled: bool = True


@dataclass(frozen=True)
class BlendResult:
    """Output of a pricing layer blend operation."""

    blended_prob: float
    model_prob: float
    market_prob: Optional[float]
    source: str  # "pinnacle", "softbook", "model_only"
    blend_weight: float  # model weight used
    raw_odds_home: Optional[float]
    raw_odds_away: Optional[float]
    raw_odds_draw: Optional[float]
    stale: bool
    reason: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "blended_prob": round(self.blended_prob, 6),
            "model_prob": round(self.model_prob, 6),
            "market_prob": round(self.market_prob, 6) if self.market_prob is not None else None,
            "source": self.source,
            "blend_weight": self.blend_weight,
            "raw_odds_home": self.raw_odds_home,
            "raw_odds_away": self.raw_odds_away,
            "raw_odds_draw": self.raw_odds_draw,
            "stale": self.stale,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def logit(p: float) -> float:
    """Log-odds transform. Clamps to avoid ±inf."""
    p = max(_CLAMP_LO, min(_CLAMP_HI, p))
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """Inverse logit (sigmoid)."""
    if x > 20.0:
        return 1.0 - 1e-9
    if x < -20.0:
        return 1e-9
    return 1.0 / (1.0 + math.exp(-x))


def logit_blend(prob_a: float, prob_b: float, weight_a: float) -> float:
    """Blend two probabilities in logit space.

    result = sigmoid(weight_a * logit(a) + (1 - weight_a) * logit(b))

    This is the canonical Tennis MS blend method.
    """
    weight_b = 1.0 - weight_a
    blended_logit = weight_a * logit(prob_a) + weight_b * logit(prob_b)
    return sigmoid(blended_logit)


def devig_2way(odds_1: float, odds_2: float) -> tuple[float, float]:
    """Remove vig from 2-way market odds. Returns fair probabilities.

    Uses multiplicative normalisation:
        implied_i = 1 / odds_i
        fair_i = implied_i / sum(implied)
    """
    if odds_1 <= 1.0 or odds_2 <= 1.0:
        raise ValueError(f"Invalid odds: {odds_1}, {odds_2} (must be > 1.0)")
    imp_1 = 1.0 / odds_1
    imp_2 = 1.0 / odds_2
    total = imp_1 + imp_2
    return imp_1 / total, imp_2 / total


def devig_3way(odds_1: float, odds_draw: float, odds_2: float) -> tuple[float, float, float]:
    """Remove vig from 3-way market odds (home/draw/away)."""
    if odds_1 <= 1.0 or odds_draw <= 1.0 or odds_2 <= 1.0:
        raise ValueError(f"Invalid odds: {odds_1}, {odds_draw}, {odds_2}")
    imp_1 = 1.0 / odds_1
    imp_d = 1.0 / odds_draw
    imp_2 = 1.0 / odds_2
    total = imp_1 + imp_d + imp_2
    return imp_1 / total, imp_d / total, imp_2 / total


# ---------------------------------------------------------------------------
# Optic Odds market fetcher
# ---------------------------------------------------------------------------


async def fetch_pinnacle_odds(
    fixture_id: str,
    sport: str,
    api_key: str,
    market_type: str = "moneyline",
    timeout: float = _DEFAULT_TIMEOUT,
) -> Optional[dict[str, Any]]:
    """Fetch Pinnacle pre-match odds from Optic Odds REST API.

    Returns dict with keys: home, away, draw (optional), source, timestamp.
    Returns None if Pinnacle odds unavailable or on error.
    """
    if not api_key:
        return None

    try:
        import httpx

        url = f"{_OPTIC_ODDS_BASE}/api/v3/fixtures/{fixture_id}/odds"
        headers = {"x-api-key": api_key}
        params = {"sportsbook": "pinnacle", "market": market_type}

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers, params=params)

        if resp.status_code != 200:
            logger.debug(
                "pinnacle_odds_fetch_failed: sport=%s fixture=%s status=%d",
                sport, fixture_id, resp.status_code,
            )
            return None

        data = resp.json()

        # Extract Pinnacle odds from Optic Odds response structure
        # Optic Odds returns: { "data": [ { "sportsbook": "pinnacle", "odds": {...} } ] }
        odds_list = data.get("data", [])
        if not odds_list:
            return None

        for entry in odds_list:
            if entry.get("sportsbook", "").lower() == "pinnacle":
                odds = entry.get("odds", {})
                return {
                    "home": odds.get("home"),
                    "away": odds.get("away"),
                    "draw": odds.get("draw"),
                    "source": "pinnacle",
                    "timestamp": time.time(),
                    "raw": entry,
                }

        return None

    except Exception as exc:
        logger.warning(
            "pinnacle_odds_fetch_error: sport=%s fixture=%s error=%s",
            sport, fixture_id, exc,
        )
        return None


# ---------------------------------------------------------------------------
# PricingLayer — the main class
# ---------------------------------------------------------------------------


class PricingLayer:
    """Cross-sport pricing layer with Pinnacle logit blend.

    Instantiate once per sport at service startup. Thread-safe.
    """

    def __init__(
        self,
        sport: str,
        optic_odds_api_key: str = "",
        config: Optional[BlendConfig] = None,
    ):
        self.sport = sport
        self._api_key = optic_odds_api_key or os.getenv("OPTIC_ODDS_API_KEY", "")
        self._config = config or BlendConfig(sport=sport)
        self._cache: dict[str, dict[str, Any]] = {}  # fixture_id → cached odds

    async def blend_pre_match(
        self,
        fixture_id: str,
        model_prob: float,
        market_type: str = "moneyline",
        force_model_only: bool = False,
    ) -> BlendResult:
        """Blend model probability with Pinnacle market anchor.

        This is the PRIMARY entry point for all pre-match pricing.
        Derivatives should consume the returned blended_prob.
        """
        now = time.time()

        # Guard: disabled or forced model-only
        if not self._config.enabled or force_model_only or not self._api_key:
            return self._model_only_result(model_prob, now, "blend_disabled_or_no_key")

        # Fetch Pinnacle odds (with cache)
        odds = await self._get_cached_odds(fixture_id, market_type)

        if odds is None:
            return self._model_only_result(model_prob, now, "pinnacle_unavailable")

        # Check staleness
        age = now - odds.get("timestamp", 0)
        if age > self._config.stale_threshold_s:
            return self._model_only_result(model_prob, now, f"pinnacle_stale_{age:.0f}s")

        # Extract and devig
        home_odds = odds.get("home")
        away_odds = odds.get("away")
        draw_odds = odds.get("draw")

        if home_odds is None or away_odds is None:
            return self._model_only_result(model_prob, now, "pinnacle_odds_incomplete")

        try:
            if draw_odds and draw_odds > 1.0:
                fair_home, _fair_draw, _fair_away = devig_3way(home_odds, draw_odds, away_odds)
            else:
                fair_home, _fair_away = devig_2way(home_odds, away_odds)
        except ValueError as e:
            return self._model_only_result(model_prob, now, f"devig_error: {e}")

        # Logit blend
        blended = logit_blend(
            prob_a=model_prob,
            prob_b=fair_home,
            weight_a=self._config.model_weight,
        )

        return BlendResult(
            blended_prob=blended,
            model_prob=model_prob,
            market_prob=fair_home,
            source=odds.get("source", "pinnacle"),
            blend_weight=self._config.model_weight,
            raw_odds_home=home_odds,
            raw_odds_away=away_odds,
            raw_odds_draw=draw_odds,
            stale=False,
            reason="logit_blend_success",
            timestamp=now,
        )

    async def _get_cached_odds(
        self, fixture_id: str, market_type: str,
    ) -> Optional[dict[str, Any]]:
        """Fetch with simple in-memory cache (60s TTL)."""
        key = f"{fixture_id}:{market_type}"
        cached = self._cache.get(key)
        if cached and (time.time() - cached.get("timestamp", 0)) < 60:
            return cached

        odds = await fetch_pinnacle_odds(
            fixture_id=fixture_id,
            sport=self.sport,
            api_key=self._api_key,
            market_type=market_type,
        )
        if odds:
            self._cache[key] = odds
        return odds

    def _model_only_result(
        self, model_prob: float, timestamp: float, reason: str,
    ) -> BlendResult:
        """Fallback: model-only with protective clamp."""
        clamped = max(
            self._config.fallback_clamp_lo,
            min(self._config.fallback_clamp_hi, model_prob),
        )
        return BlendResult(
            blended_prob=clamped,
            model_prob=model_prob,
            market_prob=None,
            source="model_only",
            blend_weight=1.0,
            raw_odds_home=None,
            raw_odds_away=None,
            raw_odds_draw=None,
            stale=False,
            reason=reason,
            timestamp=timestamp,
        )

    def blend_sync(
        self,
        model_prob: float,
        pinnacle_home_odds: Optional[float],
        pinnacle_away_odds: Optional[float],
        pinnacle_draw_odds: Optional[float] = None,
    ) -> BlendResult:
        """Synchronous blend when Pinnacle odds are already available.

        Use this when odds have been pre-fetched (e.g., from feed cache or DB).
        """
        now = time.time()

        if pinnacle_home_odds is None or pinnacle_away_odds is None:
            return self._model_only_result(model_prob, now, "no_pinnacle_odds_provided")

        if pinnacle_home_odds <= 1.0 or pinnacle_away_odds <= 1.0:
            return self._model_only_result(model_prob, now, "invalid_pinnacle_odds")

        try:
            if pinnacle_draw_odds and pinnacle_draw_odds > 1.0:
                fair_home, _, _ = devig_3way(
                    pinnacle_home_odds, pinnacle_draw_odds, pinnacle_away_odds,
                )
            else:
                fair_home, _ = devig_2way(pinnacle_home_odds, pinnacle_away_odds)
        except ValueError as e:
            return self._model_only_result(model_prob, now, f"devig_error: {e}")

        blended = logit_blend(
            prob_a=model_prob,
            prob_b=fair_home,
            weight_a=self._config.model_weight,
        )

        return BlendResult(
            blended_prob=blended,
            model_prob=model_prob,
            market_prob=fair_home,
            source="pinnacle_pre_fetched",
            blend_weight=self._config.model_weight,
            raw_odds_home=pinnacle_home_odds,
            raw_odds_away=pinnacle_away_odds,
            raw_odds_draw=pinnacle_draw_odds,
            stale=False,
            reason="logit_blend_success",
            timestamp=now,
        )


# ---------------------------------------------------------------------------
# Default sport configs
# ---------------------------------------------------------------------------

SPORT_CONFIGS: dict[str, BlendConfig] = {
    "basketball": BlendConfig(sport="basketball", model_weight=0.25, pinnacle_weight=0.75),
    "soccer": BlendConfig(sport="soccer", model_weight=0.20, pinnacle_weight=0.80),
    "table_tennis": BlendConfig(sport="table_tennis", model_weight=0.20, pinnacle_weight=0.80),
    "darts": BlendConfig(sport="darts", model_weight=0.30, pinnacle_weight=0.70),
    "snooker": BlendConfig(sport="snooker", model_weight=0.30, pinnacle_weight=0.70),
    "golf": BlendConfig(sport="golf", model_weight=0.35, pinnacle_weight=0.65),
    "f1": BlendConfig(sport="f1", model_weight=0.35, pinnacle_weight=0.65),
    "american_football": BlendConfig(sport="american_football", model_weight=0.25, pinnacle_weight=0.75),
    "rugby_league": BlendConfig(sport="rugby_league", model_weight=0.30, pinnacle_weight=0.70),
    "ice_hockey": BlendConfig(sport="ice_hockey", model_weight=0.25, pinnacle_weight=0.75),
    "volleyball": BlendConfig(sport="volleyball", model_weight=0.30, pinnacle_weight=0.70),
}


def get_pricing_layer(sport: str) -> PricingLayer:
    """Factory: return a PricingLayer with sport-appropriate config."""
    config = SPORT_CONFIGS.get(sport, BlendConfig(sport=sport))
    return PricingLayer(
        sport=sport,
        config=config,
    )
