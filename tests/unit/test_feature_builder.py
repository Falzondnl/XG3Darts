"""
Tests for the feature builder.

Tests R0/R1 feature building, EWM computation, temporal leak detection,
and edge cases.
"""
from __future__ import annotations

import pytest

from models import DartsMLError
from models.features.feature_builder import DartsFeatureBuilder
from models.r0_logit import R0_FEATURES
from models.r1_lgbm import R1_FEATURES


def _sample_match() -> dict:
    """Create a sample match dict for testing."""
    return {
        "player1_id": "player_001",
        "player2_id": "player_002",
        "competition_id": "comp_pdc_wc",
        "format_code": "PDC_WC",
        "round_name": "Quarter-Final",
        "match_date": "2024-12-20",
        "ecosystem": "pdc_mens",
        "short_format": False,
    }


def _sample_player_stats() -> dict:
    """Create sample player stats."""
    return {
        "player_001": {
            "ranking": 5,
            "three_da_pdc": 98.5,
            "checkout_pct_pdc": 42.1,
        },
        "player_002": {
            "ranking": 12,
            "three_da_pdc": 95.2,
            "checkout_pct_pdc": 38.7,
        },
    }


def _sample_elo_ratings() -> dict:
    """Create sample Elo ratings."""
    return {
        "player_001": 1680.0,
        "player_002": 1620.0,
    }


def _sample_rolling_stats() -> dict:
    """Create sample rolling stats."""
    return {
        "player_001": {
            "ewm_form": 99.1,
            "win_rate": 0.72,
            "rolling_3da": 97.8,
            "rolling_checkout_pct": 41.5,
            "hold_rate": 0.78,
            "break_rate": 0.32,
            "stage_3da": 100.2,
            "floor_3da": 96.5,
            "throw_first_win_rate": 0.65,
            "opp_adj_form": 2.5,
            "days_since_last_match": 3,
        },
        "player_002": {
            "ewm_form": 94.3,
            "win_rate": 0.61,
            "rolling_3da": 94.1,
            "rolling_checkout_pct": 37.8,
            "hold_rate": 0.72,
            "break_rate": 0.28,
            "stage_3da": 95.0,
            "floor_3da": 93.5,
            "throw_first_win_rate": 0.58,
            "opp_adj_form": -1.2,
            "days_since_last_match": 5,
        },
    }


def _sample_h2h_stats() -> dict:
    """Create sample H2H stats."""
    return {
        "p1_win_rate": 0.6,
        "total_matches": 10,
    }


class TestR0FeatureBuild:
    """Test R0 feature building."""

    def test_r0_feature_build(self) -> None:
        builder = DartsFeatureBuilder()
        features = builder.build_r0_features(
            match=_sample_match(),
            player_stats=_sample_player_stats(),
            elo_ratings=_sample_elo_ratings(),
        )

        # All 14 R0 features must be present
        assert set(R0_FEATURES) == set(features.keys()) & set(R0_FEATURES)
        for feat in R0_FEATURES:
            assert feat in features, f"Missing R0 feature: {feat}"

    def test_r0_feature_values(self) -> None:
        builder = DartsFeatureBuilder()
        features = builder.build_r0_features(
            match=_sample_match(),
            player_stats=_sample_player_stats(),
            elo_ratings=_sample_elo_ratings(),
        )

        assert features["elo_p1"] == 1680.0
        assert features["elo_p2"] == 1620.0
        assert features["elo_diff"] == 60.0
        assert features["three_da_p1_pdc"] == 98.5
        assert features["stage_floor"] == 1.0  # Quarter-Final is stage

    def test_r0_missing_player_raises(self) -> None:
        builder = DartsFeatureBuilder()
        match = _sample_match()
        match["player1_id"] = ""
        with pytest.raises(DartsMLError):
            builder.build_r0_features(
                match=match,
                player_stats=_sample_player_stats(),
                elo_ratings=_sample_elo_ratings(),
            )

    def test_r0_default_values_for_missing_stats(self) -> None:
        """Players without stats should get sensible defaults."""
        builder = DartsFeatureBuilder()
        features = builder.build_r0_features(
            match=_sample_match(),
            player_stats={},  # no stats for anyone
            elo_ratings={},    # no ratings
        )

        # Defaults: Elo=1500, ranking=128, three_da=80, checkout=35
        assert features["elo_p1"] == 1500.0
        assert features["ranking_p1"] == 128.0
        assert features["three_da_p1_pdc"] == 80.0
        assert features["checkout_pct_p1_pdc"] == 35.0


class TestR1FeatureBuild:
    """Test R1 feature building."""

    def test_r1_feature_build(self) -> None:
        builder = DartsFeatureBuilder()
        features = builder.build_r1_features(
            match=_sample_match(),
            player_stats=_sample_player_stats(),
            elo_ratings=_sample_elo_ratings(),
            rolling_stats=_sample_rolling_stats(),
            h2h_stats=_sample_h2h_stats(),
        )

        # All 38 R1 features must be present
        for feat in R1_FEATURES:
            assert feat in features, f"Missing R1 feature: {feat}"
        assert len([f for f in features if f in R1_FEATURES]) == 38

    def test_r1_includes_r0_features(self) -> None:
        builder = DartsFeatureBuilder()
        features = builder.build_r1_features(
            match=_sample_match(),
            player_stats=_sample_player_stats(),
            elo_ratings=_sample_elo_ratings(),
            rolling_stats=_sample_rolling_stats(),
            h2h_stats=_sample_h2h_stats(),
        )

        for feat in R0_FEATURES:
            assert feat in features, f"R1 missing R0 feature: {feat}"


class TestEWMFormComputation:
    """Test EWM form computation."""

    def test_ewm_form_computation(self) -> None:
        """EWM should weight recent observations more heavily."""
        historical = [80.0, 85.0, 90.0, 95.0, 100.0]
        result = DartsFeatureBuilder._compute_ewm_form(historical, decay=0.05)

        # Result should be closer to 100 (most recent) than 80 (oldest)
        assert result > 90.0
        assert result < 100.0

    def test_ewm_single_value(self) -> None:
        result = DartsFeatureBuilder._compute_ewm_form([95.0])
        assert result == 95.0

    def test_ewm_no_data_raises(self) -> None:
        with pytest.raises(RuntimeError, match="No historical 3DA data"):
            DartsFeatureBuilder._compute_ewm_form([])

    def test_ewm_decay_effect(self) -> None:
        """Higher decay should put more weight on recent values."""
        historical = [60.0, 70.0, 80.0, 90.0, 100.0]
        low_decay = DartsFeatureBuilder._compute_ewm_form(historical, decay=0.01)
        high_decay = DartsFeatureBuilder._compute_ewm_form(historical, decay=0.15)

        # Higher decay -> more weight on recent -> higher result
        assert high_decay > low_decay


class TestNoFutureFeatures:
    """Test temporal leak detection."""

    def test_no_future_features(self) -> None:
        """Features using future data must raise DartsMLError."""
        builder = DartsFeatureBuilder()
        features = {"elo_p1": 1600.0, "rolling_3da_p1": 95.0}

        # Feature computed from data AFTER match
        feature_dates = {
            "elo_p1": "2024-01-01",       # before match: OK
            "rolling_3da_p1": "2024-06-15",  # after match: BAD
        }

        with pytest.raises(DartsMLError, match="Temporal leakage"):
            builder.verify_no_future_features(
                features=features,
                match_date="2024-06-01",
                feature_dates=feature_dates,
            )

    def test_features_before_match_pass(self) -> None:
        """Features computed before match date should pass."""
        builder = DartsFeatureBuilder()
        features = {"elo_p1": 1600.0, "rolling_3da_p1": 95.0}

        feature_dates = {
            "elo_p1": "2024-01-01",
            "rolling_3da_p1": "2024-05-01",
        }

        # Should not raise
        builder.verify_no_future_features(
            features=features,
            match_date="2024-06-01",
            feature_dates=feature_dates,
        )


class TestRollingStats:
    """Test rolling statistics computation."""

    def test_rolling_stats_before_date(self) -> None:
        """Only matches before the cutoff date should be included."""
        match_history = [
            {
                "match_date": "2024-01-10",
                "three_da": 85.0,
                "checkout_pct": 35.0,
                "won": True,
                "throw_first": True,
                "is_stage": False,
                "opponent_elo": 1500,
                "held": True,
                "broke": False,
            },
            {
                "match_date": "2024-02-15",
                "three_da": 95.0,
                "checkout_pct": 42.0,
                "won": True,
                "throw_first": False,
                "is_stage": True,
                "opponent_elo": 1600,
                "held": False,
                "broke": True,
            },
            {
                "match_date": "2024-06-01",  # AFTER cutoff
                "three_da": 105.0,
                "checkout_pct": 50.0,
                "won": True,
                "throw_first": True,
                "is_stage": True,
                "opponent_elo": 1700,
                "held": True,
                "broke": False,
            },
        ]

        stats = DartsFeatureBuilder._compute_rolling_stats(
            match_history=match_history,
            before_date="2024-03-01",
            window=200,
        )

        # Should only include 2 matches (before March 1)
        assert stats["win_rate"] == 1.0  # both wins
        assert stats["rolling_3da"] == 90.0  # mean of 85 and 95

    def test_rolling_stats_empty_history(self) -> None:
        stats = DartsFeatureBuilder._compute_rolling_stats(
            match_history=[],
            before_date="2024-03-01",
        )
        assert stats == {}
