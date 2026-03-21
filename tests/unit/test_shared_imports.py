"""
tests/unit/test_shared_imports.py
==================================
Regression guard: verifies all 5 shared platform modules are importable
and expose their canonical public symbols.

These tests run without a database — pure import-level checks.
If any test here fails, the shared module stack has drifted or is broken.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# shared.__init__
# ---------------------------------------------------------------------------
class TestSharedInit:
    def test_shared_package_importable(self):
        import shared  # noqa: F401


# ---------------------------------------------------------------------------
# shared.autonomy_worker
# ---------------------------------------------------------------------------
class TestAutonomyWorker:
    def test_module_importable(self):
        import shared.autonomy_worker  # noqa: F401

    def test_AutonomyWorker_present(self):
        from shared.autonomy_worker import AutonomyWorker
        assert AutonomyWorker is not None

    def test_WorkerConfig_present(self):
        from shared.autonomy_worker import WorkerConfig
        assert WorkerConfig is not None

    def test_EventState_present(self):
        from shared.autonomy_worker import EventState
        assert EventState is not None


# ---------------------------------------------------------------------------
# shared.pricing_layer
# ---------------------------------------------------------------------------
class TestPricingLayer:
    def test_module_importable(self):
        import shared.pricing_layer  # noqa: F401

    def test_PricingLayer_present(self):
        from shared.pricing_layer import PricingLayer
        assert PricingLayer is not None

    def test_BlendConfig_present(self):
        from shared.pricing_layer import BlendConfig
        assert BlendConfig is not None

    def test_BlendResult_present(self):
        from shared.pricing_layer import BlendResult
        assert BlendResult is not None


# ---------------------------------------------------------------------------
# shared.sgp_standard
# ---------------------------------------------------------------------------
class TestSGPStandard:
    def test_module_importable(self):
        import shared.sgp_standard  # noqa: F401

    def test_SGPLeg_present(self):
        from shared.sgp_standard import SGPLeg
        assert SGPLeg is not None

    def test_SGPResult_present(self):
        from shared.sgp_standard import SGPResult
        assert SGPResult is not None

    def test_ValidationResult_present(self):
        from shared.sgp_standard import ValidationResult
        assert ValidationResult is not None

    def test_PricingAuthority_present(self):
        from shared.sgp_standard import PricingAuthority
        assert PricingAuthority is not None


# ---------------------------------------------------------------------------
# shared.stale_price_guard
# ---------------------------------------------------------------------------
class TestStalePriceGuard:
    def test_module_importable(self):
        import shared.stale_price_guard  # noqa: F401

    def test_StalePriceGuard_present(self):
        from shared.stale_price_guard import StalePriceGuard
        assert StalePriceGuard is not None

    def test_StalePriceStatus_present(self):
        from shared.stale_price_guard import StalePriceStatus
        assert StalePriceStatus is not None

    def test_FreshnessLevel_present(self):
        from shared.stale_price_guard import FreshnessLevel
        assert FreshnessLevel is not None
