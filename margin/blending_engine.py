"""
5-factor margin blending framework for XG3 Darts.

The five factors applied sequentially:
  1. Regime widening  — R0 markets are most uncertain, R2 least
  2. Starter-confidence widening — uncertain throw-order inflates margin
  3. Source-confidence widening — low data quality inflates margin
  4. Model-disagreement widening — ensemble divergence inflates margin
  5. Ecosystem / liquidity widening — women's / development / low-liquidity markets

All multipliers are compounded, then the result is hard-capped at 15 %.
"""
from __future__ import annotations

import structlog

from engines.errors import DartsEngineError

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regime widening multipliers (R0 = result-only, R1 = match stats, R2 = visit data)
_REGIME_WIDENING: dict[int, float] = {
    0: 1.30,  # R0: highest uncertainty → widen most
    1: 1.10,  # R1: moderate data quality
    2: 1.00,  # R2: full visit data → no regime widening
}

# Low-liquidity ecosystems that attract an extra premium
_LOW_LIQUIDITY_ECOSYSTEMS = frozenset({
    "pdc_womens",
    "wdf",          # generic WDF prefix match
    "wdf_open",
    "development",
    "challenge",
})

# Hard cap on final margin
_MARGIN_HARD_CAP = 0.15


class DartsMarginEngine:
    """
    5-factor margin computation engine.

    The engine accepts calibrated inputs describing uncertainty along five
    orthogonal dimensions and returns a final margin capped at 15 %.

    All factors are documented on the individual parameter; callers are
    expected to populate them from live data and model outputs — never
    from hardcoded values.
    """

    def compute_margin(
        self,
        base_margin: float,
        regime: int,
        starter_confidence: float,
        source_confidence: float,
        model_agreement: float,
        market_liquidity: str,
        ecosystem: str,
    ) -> float:
        """
        Compute the final margin from five multiplicative factors.

        Parameters
        ----------
        base_margin:
            Starting margin before any widening, e.g. 0.05 for 5 %.
            Must be in (0, 0.15].
        regime:
            Data coverage regime (0 = R0, 1 = R1, 2 = R2).
        starter_confidence:
            Confidence in throw-order assignment [0, 1].
            1.0 = certain; 0.0 = completely unknown.
        source_confidence:
            Confidence in the underlying data sources [0, 1].
            1.0 = high quality; 0.0 = unreliable / imputed.
        model_agreement:
            Agreement between model outputs (e.g. ensemble variance).
            1.0 = full agreement; 0.0 = maximum disagreement.
        market_liquidity:
            Expected market liquidity tier: ``"high"`` | ``"medium"`` | ``"low"``.
        ecosystem:
            Competition ecosystem string from the format registry (e.g.
            ``"pdc_mens"``, ``"pdc_womens"``, ``"wdf_open"``, ``"development"``).

        Returns
        -------
        float
            Final margin, capped at 15 %.

        Raises
        ------
        DartsEngineError
            If any input is out of range or the regime is invalid.
        """
        self._validate_inputs(
            base_margin=base_margin,
            regime=regime,
            starter_confidence=starter_confidence,
            source_confidence=source_confidence,
            model_agreement=model_agreement,
            market_liquidity=market_liquidity,
        )

        margin = base_margin

        # Factor 1: Regime widening
        regime_mult = _REGIME_WIDENING[regime]
        margin *= regime_mult

        # Factor 2: Starter uncertainty widening
        # starter_confidence=1 → no widening; starter_confidence=0 → 30% wider
        starter_mult = 1.0 + (1.0 - starter_confidence) * 0.30
        margin *= starter_mult

        # Factor 3: Source confidence widening
        # source_confidence=1 → no widening; source_confidence=0 → 20% wider
        source_mult = 1.0 + (1.0 - source_confidence) * 0.20
        margin *= source_mult

        # Factor 4: Model disagreement widening
        # model_agreement=1 → no widening; model_agreement=0 → 25% wider
        agreement_mult = 1.0 + (1.0 - model_agreement) * 0.25
        margin *= agreement_mult

        # Factor 5a: Ecosystem widening for low-liquidity / development competitions
        eco_lower = ecosystem.lower()
        if eco_lower in _LOW_LIQUIDITY_ECOSYSTEMS or eco_lower.startswith("wdf"):
            margin *= 1.20

        # Factor 5b: Explicit low-liquidity flag
        if market_liquidity == "low":
            margin *= 1.15

        # Hard cap
        final_margin = min(margin, _MARGIN_HARD_CAP)

        logger.debug(
            "margin_computed",
            base_margin=round(base_margin, 5),
            regime=regime,
            regime_mult=round(regime_mult, 4),
            starter_mult=round(starter_mult, 4),
            source_mult=round(source_mult, 4),
            agreement_mult=round(agreement_mult, 4),
            ecosystem=ecosystem,
            market_liquidity=market_liquidity,
            final_margin=round(final_margin, 5),
            capped=(final_margin == _MARGIN_HARD_CAP),
        )

        return final_margin

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def compute_margin_for_regime(
        self,
        base_margin: float,
        regime: int,
        ecosystem: str = "pdc_mens",
    ) -> float:
        """
        Simplified margin computation using regime only (neutral other factors).

        Useful for quick estimates and unit tests.
        """
        return self.compute_margin(
            base_margin=base_margin,
            regime=regime,
            starter_confidence=1.0,
            source_confidence=1.0,
            model_agreement=1.0,
            market_liquidity="high",
            ecosystem=ecosystem,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        base_margin: float,
        regime: int,
        starter_confidence: float,
        source_confidence: float,
        model_agreement: float,
        market_liquidity: str,
    ) -> None:
        if not (0.0 < base_margin <= _MARGIN_HARD_CAP):
            raise DartsEngineError(
                f"base_margin must be in (0, {_MARGIN_HARD_CAP}], got {base_margin:.4f}"
            )
        if regime not in _REGIME_WIDENING:
            raise DartsEngineError(
                f"regime must be 0, 1, or 2; got {regime!r}"
            )
        for name, val in [
            ("starter_confidence", starter_confidence),
            ("source_confidence", source_confidence),
            ("model_agreement", model_agreement),
        ]:
            if not (0.0 <= val <= 1.0):
                raise DartsEngineError(
                    f"{name} must be in [0, 1], got {val:.4f}"
                )
        if market_liquidity not in ("high", "medium", "low"):
            raise DartsEngineError(
                f"market_liquidity must be 'high', 'medium', or 'low'; got {market_liquidity!r}"
            )
