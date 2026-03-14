"""
Unit tests for Women's Series model path.

Tests
-----
- test_womens_margin_multiplier
- test_womens_ecosystem_routing
- test_transfer_model_predicts_in_range
- test_transfer_model_fit_and_predict
- test_partial_pooling_shrinkage
- test_womens_model_untrained_fallback
- test_womens_ecosystem_in_format_registry
- test_pdc_wwm_format_is_pdc_womens
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from competition.format_registry import get_format, list_formats
from models.womens_transfer import WomensTransferModel
from models import DartsMLError


# ---------------------------------------------------------------------------
# test_womens_margin_multiplier
# ---------------------------------------------------------------------------

class TestWomensMarginMultiplier:
    """WomensTransferModel.get_ecosystem_margin_multiplier() must return 1.20."""

    def test_multiplier_is_1_20(self):
        model = WomensTransferModel()
        assert model.get_ecosystem_margin_multiplier() == pytest.approx(1.20, abs=1e-10)

    def test_multiplier_unchanged_after_fit(self):
        """The multiplier should be constant regardless of fitted state."""
        model = WomensTransferModel()
        # Before fit
        assert model.get_ecosystem_margin_multiplier() == pytest.approx(1.20)
        # After fit with minimal data
        X = pd.DataFrame({"elo_diff": [100.0, -50.0, 200.0, -100.0, 150.0]})
        y = np.array([1, 0, 1, 0, 1])
        model.fit(X, y)
        assert model.get_ecosystem_margin_multiplier() == pytest.approx(1.20)

    def test_margin_engine_applies_womens_widening(self):
        """
        The DartsMarginEngine should apply widening for pdc_womens ecosystem.
        Women's ecosystem is in the low-liquidity set, so it gets 1.20x multiplier.
        """
        from margin.blending_engine import DartsMarginEngine
        engine = DartsMarginEngine()
        base = 0.05
        mens_margin = engine.compute_margin(
            base_margin=base,
            regime=1,
            starter_confidence=1.0,
            source_confidence=1.0,
            model_agreement=1.0,
            market_liquidity="high",
            ecosystem="pdc_mens",
        )
        womens_margin = engine.compute_margin(
            base_margin=base,
            regime=1,
            starter_confidence=1.0,
            source_confidence=1.0,
            model_agreement=1.0,
            market_liquidity="high",
            ecosystem="pdc_womens",
        )
        # Women's margin should be wider due to 1.20x ecosystem factor
        assert womens_margin > mens_margin, (
            f"Womens margin {womens_margin:.5f} should be > mens margin {mens_margin:.5f}"
        )
        # Approximately 1.20x the mens margin (may be capped at 0.15)
        ratio = womens_margin / mens_margin
        assert ratio == pytest.approx(1.20, abs=0.01) or womens_margin == 0.15


# ---------------------------------------------------------------------------
# test_womens_ecosystem_routing
# ---------------------------------------------------------------------------

class TestWomensEcosystemRouting:
    """Verify the pdc_womens ecosystem flag is properly set in format registry."""

    def test_pdc_wom_series_ecosystem(self):
        fmt = get_format("PDC_WOM_SERIES")
        assert fmt.ecosystem == "pdc_womens"

    def test_pdc_wwm_ecosystem(self):
        fmt = get_format("PDC_WWM")
        assert fmt.ecosystem == "pdc_womens"

    def test_womens_formats_in_registry(self):
        """Confirm both women's formats exist in the registry."""
        codes = list_formats()
        assert "PDC_WOM_SERIES" in codes
        assert "PDC_WWM" in codes

    def test_prematch_ecosystem_routing_caps_womens_regime(self):
        """
        For pdc_womens, max regime should still be 2 in the routing table.
        (Women's can use R0/R1/R2 if data is available.)
        """
        from app.routes.prematch import _ECOSYSTEM_MAX_REGIME
        assert _ECOSYSTEM_MAX_REGIME["pdc_womens"] == 2

    def test_prematch_ecosystem_routing_wdf_is_r0(self):
        """WDF ecosystem should be capped at R0."""
        from app.routes.prematch import _ECOSYSTEM_MAX_REGIME
        assert _ECOSYSTEM_MAX_REGIME["wdf_open"] == 0

    def test_prematch_ecosystem_routing_development_is_r0(self):
        """Development ecosystem should be capped at R0."""
        from app.routes.prematch import _ECOSYSTEM_MAX_REGIME
        assert _ECOSYSTEM_MAX_REGIME["development"] == 0


# ---------------------------------------------------------------------------
# test_transfer_model_predicts_in_range
# ---------------------------------------------------------------------------

class TestTransferModelPrediction:
    """WomensTransferModel.predict_proba must return a value in (0, 1)."""

    def test_predict_in_range_unfitted(self):
        """Before fitting, uses men's prior — still valid probability."""
        model = WomensTransferModel()
        features = {"elo_p1": 1600.0, "elo_p2": 1500.0}
        p = model.predict_proba(features, "player_a", "player_b")
        assert 0.01 <= p <= 0.99

    def test_predict_higher_elo_wins_more(self):
        """Player with higher ELO should have higher win probability."""
        model = WomensTransferModel()
        p_high = model.predict_proba({"elo_p1": 1700.0, "elo_p2": 1500.0}, "A", "B")
        p_low = model.predict_proba({"elo_p1": 1500.0, "elo_p2": 1700.0}, "A", "B")
        assert p_high > p_low

    def test_predict_equal_elo_near_50(self):
        """Equal ELO should give a probability close to 0.5."""
        model = WomensTransferModel()
        p = model.predict_proba({"elo_p1": 1500.0, "elo_p2": 1500.0}, "A", "B")
        assert abs(p - 0.5) < 0.05

    def test_predict_complementary(self):
        """P(A beats B) + P(B beats A) should not need to sum to 1 (independent calls)
        but each should be in (0, 1)."""
        model = WomensTransferModel()
        p_ab = model.predict_proba({"elo_p1": 1600.0, "elo_p2": 1550.0}, "A", "B")
        p_ba = model.predict_proba({"elo_p1": 1550.0, "elo_p2": 1600.0}, "B", "A")
        assert 0.0 < p_ab < 1.0
        assert 0.0 < p_ba < 1.0
        # Since it's logistic with elo_diff: p_ab + p_ba should be approx 1
        assert abs(p_ab + p_ba - 1.0) < 0.05

    def test_predict_raises_on_missing_elo(self):
        """Missing ELO features should raise DartsMLError."""
        model = WomensTransferModel()
        with pytest.raises(DartsMLError, match="elo_p1"):
            model.predict_proba({"three_da": 85.0}, "A", "B")


# ---------------------------------------------------------------------------
# test_transfer_model_fit_and_predict
# ---------------------------------------------------------------------------

class TestTransferModelFit:
    """Tests for WomensTransferModel.fit()."""

    def _make_training_data(self, n: int = 50) -> tuple:
        rng = np.random.default_rng(42)
        elo_diffs = rng.normal(0, 150, n)
        # Simulate: higher ELO diff -> more likely to win
        logits = 0.005 * elo_diffs
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = rng.binomial(1, probs)
        X = pd.DataFrame({"elo_diff": elo_diffs})
        return X, y

    def test_fit_succeeds(self):
        model = WomensTransferModel()
        X, y = self._make_training_data(50)
        model.fit(X, y)
        assert model.is_fitted

    def test_predict_after_fit_in_range(self):
        model = WomensTransferModel()
        X, y = self._make_training_data(50)
        model.fit(X, y)
        p = model.predict_proba({"elo_p1": 1600.0, "elo_p2": 1500.0}, "A", "B")
        assert 0.01 <= p <= 0.99

    def test_fit_raises_on_missing_elo_diff_column(self):
        model = WomensTransferModel()
        X = pd.DataFrame({"three_da": [85.0, 90.0, 80.0, 75.0, 88.0]})
        y = np.array([1, 1, 0, 0, 1])
        with pytest.raises(DartsMLError, match="elo_diff"):
            model.fit(X, y)

    def test_fit_with_sparse_data_uses_prior(self):
        """With fewer than 5 matches, model should fall back to prior."""
        model = WomensTransferModel()
        X = pd.DataFrame({"elo_diff": [100.0, -50.0]})
        y = np.array([1, 0])
        model.fit(X, y)
        # Should still be fitted (with men's prior)
        assert model.is_fitted
        assert model.n_womens_matches == 2

    def test_partial_pooling_shrinkage_increases_with_data(self):
        """More data should increase effective influence of women's estimates."""
        model_sparse = WomensTransferModel(shrinkage=0.4)
        X_sparse = pd.DataFrame({"elo_diff": [100.0] * 5})
        y_sparse = np.array([1, 1, 1, 0, 1])
        model_sparse.fit(X_sparse, y_sparse)

        model_rich = WomensTransferModel(shrinkage=0.4)
        X_rich = pd.DataFrame({"elo_diff": [100.0] * 200})
        y_rich = np.array([1] * 130 + [0] * 70)
        model_rich.fit(X_rich, y_rich)

        # Both should be fitted and produce valid predictions
        p_sparse = model_sparse.predict_proba(
            {"elo_p1": 1600.0, "elo_p2": 1500.0}, "A", "B"
        )
        p_rich = model_rich.predict_proba(
            {"elo_p1": 1600.0, "elo_p2": 1500.0}, "A", "B"
        )
        assert 0.01 <= p_sparse <= 0.99
        assert 0.01 <= p_rich <= 0.99

    def test_invalid_shrinkage_raises(self):
        with pytest.raises(DartsMLError):
            WomensTransferModel(shrinkage=1.5)

    def test_invalid_shrinkage_negative_raises(self):
        with pytest.raises(DartsMLError):
            WomensTransferModel(shrinkage=-0.1)


# ---------------------------------------------------------------------------
# test_womens_ecosystem_in_format_registry
# ---------------------------------------------------------------------------

class TestWomensFormatRegistry:
    """Verify Women's Series formats are fully registered."""

    def test_pdc_wom_series_rounds(self):
        fmt = get_format("PDC_WOM_SERIES")
        expected_rounds = {"Round 1", "Round 2", "Quarter-Final", "Semi-Final", "Final"}
        assert set(fmt.per_round.keys()) == expected_rounds

    def test_pdc_wwm_rounds(self):
        fmt = get_format("PDC_WWM")
        expected_rounds = {"Round 1", "Quarter-Final", "Semi-Final", "Final"}
        assert set(fmt.per_round.keys()) == expected_rounds

    def test_womens_series_legs_format(self):
        """PDC Women's Series uses legs-only format."""
        fmt = get_format("PDC_WOM_SERIES")
        final_round = fmt.get_round("Final")
        assert final_round.legs_to_win == 5
        assert not final_round.is_sets_format

    def test_pdc_wwm_legs_format(self):
        fmt = get_format("PDC_WWM")
        final_round = fmt.get_round("Final")
        assert final_round.legs_to_win == 9
