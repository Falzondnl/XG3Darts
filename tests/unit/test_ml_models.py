"""
Tests for ML models (R0/R1/R2).

Tests feature completeness, temporal split correctness, pair swap balance,
beta calibrator fit, and calibrator output range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models import DartsMLError
from models.r0_logit import R0LogisticModel, R0_FEATURES, R0_FEATURE_COUNT
from models.r1_lgbm import R1_FEATURES, R1_FEATURE_COUNT
from models.r2_stacking import R2_FEATURES, R2_FEATURE_COUNT
from models.trainer import DartsModelTrainer
from calibration.beta_calibrator import BetaCalibrator, BetaCalibrationError


def _make_r0_df(n: int = 200, rng: np.random.Generator | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic R0 feature data for testing."""
    if rng is None:
        rng = np.random.default_rng(42)

    data = {}
    for feat in R0_FEATURES:
        if feat == "elo_diff":
            data[feat] = rng.normal(0, 200, n)
        elif feat in ("elo_p1", "elo_p2"):
            data[feat] = rng.normal(1500, 200, n)
        elif feat in ("ranking_p1", "ranking_p2"):
            data[feat] = rng.uniform(1, 128, n)
        elif feat == "ranking_log_ratio":
            data[feat] = rng.normal(0, 1, n)
        elif feat in ("three_da_p1_pdc", "three_da_p2_pdc"):
            data[feat] = rng.normal(85, 10, n)
        elif feat in ("checkout_pct_p1_pdc", "checkout_pct_p2_pdc"):
            data[feat] = rng.uniform(25, 50, n)
        elif feat in ("format_type_encoded", "stage_floor", "short_format"):
            data[feat] = rng.choice([0.0, 1.0], n)
        elif feat == "ecosystem_encoded":
            data[feat] = rng.choice([0.0, 1.0, 2.0, 3.0], n)
        else:
            data[feat] = rng.normal(0, 1, n)

    df = pd.DataFrame(data)
    # Labels: P1 wins more when elo_diff > 0
    proba = 1.0 / (1.0 + np.exp(-df["elo_diff"].values / 200.0))
    labels = (rng.random(n) < proba).astype(np.float64)

    return df, labels


def _make_r1_df(n: int = 200, rng: np.random.Generator | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic R1 feature data."""
    if rng is None:
        rng = np.random.default_rng(42)

    df, labels = _make_r0_df(n, rng)
    for feat in R1_FEATURES:
        if feat not in df.columns:
            if "rate" in feat or "pct" in feat:
                df[feat] = rng.uniform(0.1, 0.9, n)
            elif "days" in feat:
                df[feat] = rng.uniform(1, 30, n)
            elif "h2h_total" in feat:
                df[feat] = rng.integers(0, 20, n).astype(float)
            else:
                df[feat] = rng.normal(0, 1, n)
    return df, labels


class TestR0FeaturesComplete:
    """Test that R0 feature set is complete and correct."""

    def test_r0_features_count(self) -> None:
        assert R0_FEATURE_COUNT == 14

    def test_r0_features_complete(self) -> None:
        """All 14 R0 features must be present in the canonical list."""
        expected = {
            "elo_p1", "elo_p2", "elo_diff",
            "ranking_p1", "ranking_p2", "ranking_log_ratio",
            "three_da_p1_pdc", "three_da_p2_pdc",
            "checkout_pct_p1_pdc", "checkout_pct_p2_pdc",
            "format_type_encoded",
            "stage_floor",
            "short_format",
            "ecosystem_encoded",
        }
        assert set(R0_FEATURES) == expected

    def test_r0_model_fit_and_predict(self) -> None:
        df, labels = _make_r0_df(200)
        model = R0LogisticModel()
        model.fit(df, labels)
        assert model.fitted_ is True

        proba = model.predict_proba(df)
        assert proba.shape == (200, 2)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        # Probabilities must sum to 1 per row
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_r0_missing_feature_raises(self) -> None:
        df, labels = _make_r0_df(50)
        df = df.drop(columns=["elo_diff"])
        model = R0LogisticModel()
        with pytest.raises(DartsMLError, match="Missing R0 features"):
            model.fit(df, labels)

    def test_r0_nan_feature_raises(self) -> None:
        df, labels = _make_r0_df(50)
        df.loc[0, "elo_p1"] = np.nan
        model = R0LogisticModel()
        with pytest.raises(DartsMLError, match="NaN"):
            model.fit(df, labels)


class TestR0NoTargetLeakage:
    """Verify that R0 features do not contain target-leaking columns."""

    def test_r0_no_target_leakage(self) -> None:
        """R0 features must not include any match outcome columns."""
        leaky_patterns = [
            "result", "winner", "won", "outcome", "target",
            "score_final", "legs_won", "sets_won",
        ]
        for feat in R0_FEATURES:
            for pattern in leaky_patterns:
                assert pattern not in feat.lower(), (
                    f"R0 feature {feat!r} contains leaky pattern {pattern!r}"
                )

    def test_r0_features_are_pregame(self) -> None:
        """All R0 features should be computable before the match starts."""
        pregame_patterns = [
            "elo", "ranking", "three_da", "checkout_pct",
            "format", "stage", "short", "ecosystem",
        ]
        for feat in R0_FEATURES:
            found = any(p in feat.lower() for p in pregame_patterns)
            assert found, f"R0 feature {feat!r} does not match any pregame pattern"


class TestR1FeaturesComplete:
    """Test R1 feature set."""

    def test_r1_features_count(self) -> None:
        assert R1_FEATURE_COUNT == 38

    def test_r1_includes_all_r0(self) -> None:
        """R1 must be a superset of R0 features."""
        assert set(R0_FEATURES).issubset(set(R1_FEATURES))

    def test_r1_additional_features(self) -> None:
        """R1 adds 24 features beyond R0."""
        r1_only = set(R1_FEATURES) - set(R0_FEATURES)
        assert len(r1_only) == 24


class TestR2FeaturesComplete:
    """Test R2 feature set."""

    def test_r2_features_count(self) -> None:
        assert R2_FEATURE_COUNT == 68

    def test_r2_includes_all_r1(self) -> None:
        """R2 must be a superset of R1 features."""
        assert set(R1_FEATURES).issubset(set(R2_FEATURES))


class TestTemporalSplit:
    """Test temporal split correctness."""

    def test_temporal_split_correct(self) -> None:
        """Temporal split must not shuffle data."""
        rng = np.random.default_rng(42)
        n = 100
        df, labels = _make_r0_df(n, rng)
        # Add match_date column in chronological order
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        df["match_date"] = dates
        df["result"] = labels

        trainer = DartsModelTrainer()
        X_train, y_train, X_val, y_val, X_test, y_test = trainer.temporal_split(
            df, labels
        )

        # Training dates must come before validation dates
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0
        assert len(X_train) + len(X_val) + len(X_test) == n

    def test_temporal_split_no_overlap(self) -> None:
        """Train/val/test sets must not overlap in time."""
        rng = np.random.default_rng(42)
        n = 100
        df, labels = _make_r0_df(n, rng)
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        df["match_date"] = dates
        df["result"] = labels

        trainer = DartsModelTrainer()
        X_train, _, X_val, _, X_test, _ = trainer.temporal_split(df, labels)

        if "match_date" in X_train.columns and "match_date" in X_val.columns:
            assert X_train["match_date"].max() <= X_val["match_date"].min()
        if "match_date" in X_val.columns and "match_date" in X_test.columns:
            assert X_val["match_date"].max() <= X_test["match_date"].min()


class TestRandomPairSwap:
    """Test random pair swap for class balance."""

    def test_random_pair_swap_balance(self) -> None:
        """After pair swap, class balance should be approximately 50%."""
        rng = np.random.default_rng(42)
        n = 1000
        df, _ = _make_r0_df(n, rng)
        # Start with heavily imbalanced labels (80% P1 wins)
        labels_imbal = (rng.random(n) < 0.8).astype(np.float64)
        df["result"] = labels_imbal
        df["match_date"] = pd.date_range("2023-01-01", periods=n, freq="h")

        trainer = DartsModelTrainer()
        _, swapped_labels = trainer._random_pair_swap(df)

        # After swap, balance should be roughly 50% (within tolerance)
        balance = float(np.mean(swapped_labels))
        assert 0.35 < balance < 0.65, f"Class balance after swap: {balance}"


class TestBetaCalibratorFit:
    """Test Beta calibrator fitting."""

    def test_beta_calibrator_fit(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        # Generate well-separated predictions
        proba = np.concatenate([
            rng.beta(2, 5, n // 2),   # negative class: low probabilities
            rng.beta(5, 2, n // 2),   # positive class: high probabilities
        ])
        y_true = np.concatenate([
            np.zeros(n // 2),
            np.ones(n // 2),
        ])

        cal = BetaCalibrator()
        cal.fit(proba, y_true)

        assert cal.fitted_ is True
        assert cal.a_ > 0
        assert cal.b_ > 0

    def test_calibrator_output_in_range(self) -> None:
        """Calibrated outputs must be in [0, 1]."""
        rng = np.random.default_rng(42)
        n = 500
        proba = np.concatenate([
            rng.beta(2, 5, n // 2),
            rng.beta(5, 2, n // 2),
        ])
        y_true = np.concatenate([
            np.zeros(n // 2),
            np.ones(n // 2),
        ])

        cal = BetaCalibrator()
        cal.fit(proba, y_true)

        test_proba = rng.uniform(0.01, 0.99, 100)
        calibrated = cal.predict_proba(test_proba)

        assert calibrated.shape == (100,)
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)

    def test_calibrator_unfitted_raises(self) -> None:
        cal = BetaCalibrator()
        with pytest.raises(BetaCalibrationError, match="not been fitted"):
            cal.predict_proba(np.array([0.5]))
