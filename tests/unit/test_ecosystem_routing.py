"""
Unit tests for ecosystem-aware pricing routing.

Tests
-----
- test_pdc_mens_uses_r0_r1_r2
- test_wdf_uses_r0_primarily
- test_development_r0_only
- test_team_doubles_routed_to_worldcup
- test_ecosystem_max_regime_mapping
- test_margin_engine_wdf_widening
- test_margin_engine_development_widening
- test_womens_margin_wider_than_mens
- test_ecosystem_endpoint_caps_regime
"""
from __future__ import annotations

import pytest

from competition.format_registry import get_format, get_all_formats
from margin.blending_engine import DartsMarginEngine


# ---------------------------------------------------------------------------
# test_pdc_mens_uses_r0_r1_r2
# ---------------------------------------------------------------------------

class TestPDCMensEcosystem:
    """PDC mens ecosystem allows full R0/R1/R2 model path."""

    def test_pdc_mens_max_regime_is_2(self):
        from app.routes.prematch import _ECOSYSTEM_MAX_REGIME
        assert _ECOSYSTEM_MAX_REGIME["pdc_mens"] == 2

    def test_pdc_mens_formats_have_pdc_mens_ecosystem(self):
        """All core PDC mens competition formats use the pdc_mens ecosystem."""
        pdc_mens_codes = [
            "PDC_WC", "PDC_WC_ERA_2020", "PDC_PL", "PDC_WM",
            "PDC_GP", "PDC_UK", "PDC_GS", "PDC_ET", "PDC_PC",
            "PDC_PCF", "PDC_MASTERS", "PDC_WS",
        ]
        all_fmts = get_all_formats()
        for code in pdc_mens_codes:
            if code in all_fmts:
                fmt = all_fmts[code]
                assert fmt.ecosystem == "pdc_mens", (
                    f"{code} has ecosystem={fmt.ecosystem!r}, expected 'pdc_mens'"
                )

    def test_pdc_mens_margin_no_ecosystem_widening(self):
        """PDC mens does not trigger ecosystem widening in the margin engine."""
        engine = DartsMarginEngine()
        margin = engine.compute_margin(
            base_margin=0.05,
            regime=2,
            starter_confidence=1.0,
            source_confidence=1.0,
            model_agreement=1.0,
            market_liquidity="high",
            ecosystem="pdc_mens",
        )
        # R2 regime, all factors neutral, mens ecosystem -> base margin unchanged
        expected = 0.05 * 1.0  # regime_mult=1.0, all others neutral, no eco widening
        assert margin == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# test_wdf_uses_r0_primarily
# ---------------------------------------------------------------------------

class TestWDFEcosystem:
    """WDF ecosystem uses R0 primarily due to limited data."""

    def test_wdf_max_regime_is_0(self):
        from app.routes.prematch import _ECOSYSTEM_MAX_REGIME
        assert _ECOSYSTEM_MAX_REGIME["wdf_open"] == 0

    def test_wdf_formats_have_wdf_open_ecosystem(self):
        """WDF competition formats should use wdf_open ecosystem."""
        wdf_codes = ["WDF_WC", "WDF_EC", "WDF_OPEN"]
        all_fmts = get_all_formats()
        for code in wdf_codes:
            if code in all_fmts:
                fmt = all_fmts[code]
                assert fmt.ecosystem == "wdf_open", (
                    f"{code} has ecosystem={fmt.ecosystem!r}, expected 'wdf_open'"
                )

    def test_wdf_margin_has_ecosystem_widening(self):
        """WDF ecosystem should trigger 1.20x widening in the margin engine."""
        engine = DartsMarginEngine()
        mens_margin = engine.compute_margin(
            base_margin=0.05,
            regime=0,
            starter_confidence=1.0,
            source_confidence=1.0,
            model_agreement=1.0,
            market_liquidity="high",
            ecosystem="pdc_mens",
        )
        wdf_margin = engine.compute_margin(
            base_margin=0.05,
            regime=0,
            starter_confidence=1.0,
            source_confidence=1.0,
            model_agreement=1.0,
            market_liquidity="high",
            ecosystem="wdf_open",
        )
        assert wdf_margin > mens_margin, (
            f"WDF margin {wdf_margin:.5f} should exceed mens margin {mens_margin:.5f}"
        )

    def test_wdf_regime_capped_at_0(self):
        """Requesting regime=2 for WDF should be capped at 0."""
        from app.routes.prematch import _ECOSYSTEM_MAX_REGIME
        effective = min(2, _ECOSYSTEM_MAX_REGIME["wdf_open"])
        assert effective == 0

    def test_wdf_open_format_rounds(self):
        fmt = get_format("WDF_OPEN")
        assert "Final" in fmt.per_round
        assert fmt.ecosystem == "wdf_open"


# ---------------------------------------------------------------------------
# test_development_r0_only
# ---------------------------------------------------------------------------

class TestDevelopmentEcosystem:
    """Development ecosystem (PDC Development / Challenge Tour) is R0 only."""

    def test_development_max_regime_is_0(self):
        from app.routes.prematch import _ECOSYSTEM_MAX_REGIME
        assert _ECOSYSTEM_MAX_REGIME["development"] == 0

    def test_development_formats_have_development_ecosystem(self):
        """PDC Development and Challenge Tour use the development ecosystem."""
        dev_codes = ["PDC_DEVTOUR", "PDC_CHALLENGE", "PDC_WYC"]
        all_fmts = get_all_formats()
        for code in dev_codes:
            if code in all_fmts:
                fmt = all_fmts[code]
                assert fmt.ecosystem == "development", (
                    f"{code} has ecosystem={fmt.ecosystem!r}, expected 'development'"
                )

    def test_development_margin_has_ecosystem_widening(self):
        """Development ecosystem should trigger 1.20x widening."""
        engine = DartsMarginEngine()
        mens_margin = engine.compute_margin(
            base_margin=0.05,
            regime=0,
            starter_confidence=1.0,
            source_confidence=1.0,
            model_agreement=1.0,
            market_liquidity="high",
            ecosystem="pdc_mens",
        )
        dev_margin = engine.compute_margin(
            base_margin=0.05,
            regime=0,
            starter_confidence=1.0,
            source_confidence=1.0,
            model_agreement=1.0,
            market_liquidity="high",
            ecosystem="development",
        )
        assert dev_margin > mens_margin

    def test_development_format_challenge_tour_rounds(self):
        fmt = get_format("PDC_CHALLENGE")
        expected_rounds = {"Round 1", "Round 2", "Quarter-Final", "Semi-Final", "Final"}
        assert set(fmt.per_round.keys()) == expected_rounds

    def test_development_format_devtour_rounds(self):
        fmt = get_format("PDC_DEVTOUR")
        assert "Final" in fmt.per_round
        assert fmt.ecosystem == "development"


# ---------------------------------------------------------------------------
# Ecosystem max regime mapping completeness
# ---------------------------------------------------------------------------

class TestEcosystemMaxRegimeMapping:
    """Verify the _ECOSYSTEM_MAX_REGIME dict covers all known ecosystems."""

    def test_all_ecosystems_have_regime_mapping(self):
        """Every ecosystem used in any format should have a regime cap."""
        from app.routes.prematch import _ECOSYSTEM_MAX_REGIME
        all_fmts = get_all_formats()
        used_ecosystems = {fmt.ecosystem for fmt in all_fmts.values()}
        for eco in used_ecosystems:
            assert eco in _ECOSYSTEM_MAX_REGIME, (
                f"Ecosystem {eco!r} is not in _ECOSYSTEM_MAX_REGIME"
            )

    def test_regime_caps_are_valid(self):
        """All regime caps should be 0, 1, or 2."""
        from app.routes.prematch import _ECOSYSTEM_MAX_REGIME
        for eco, cap in _ECOSYSTEM_MAX_REGIME.items():
            assert cap in (0, 1, 2), f"Ecosystem {eco!r} has invalid cap {cap}"


# ---------------------------------------------------------------------------
# Cross-ecosystem margin comparison
# ---------------------------------------------------------------------------

class TestCrossEcosystemMargin:
    """Comparative margin tests across all ecosystems."""

    def test_margin_ordering(self):
        """
        Expected order (at same base and regime):
        pdc_mens < pdc_womens ≈ wdf_open ≈ development
        (all low-liquidity ecosystems get 1.20x widening)
        """
        engine = DartsMarginEngine()

        def get_margin(eco):
            return engine.compute_margin(
                base_margin=0.05,
                regime=0,
                starter_confidence=1.0,
                source_confidence=1.0,
                model_agreement=1.0,
                market_liquidity="high",
                ecosystem=eco,
            )

        m_mens = get_margin("pdc_mens")
        m_womens = get_margin("pdc_womens")
        m_wdf = get_margin("wdf_open")
        m_dev = get_margin("development")

        assert m_mens < m_womens
        assert m_mens < m_wdf
        assert m_mens < m_dev

    def test_all_margins_within_cap(self):
        """No margin should exceed the hard cap of 0.15."""
        engine = DartsMarginEngine()
        ecosystems = ["pdc_mens", "pdc_womens", "wdf_open", "development"]
        for eco in ecosystems:
            margin = engine.compute_margin(
                base_margin=0.12,
                regime=0,
                starter_confidence=0.5,
                source_confidence=0.5,
                model_agreement=0.5,
                market_liquidity="low",
                ecosystem=eco,
            )
            assert margin <= 0.15, (
                f"Ecosystem {eco!r} margin {margin:.5f} exceeds hard cap 0.15"
            )
