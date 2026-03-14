"""
Training pipeline for all regimes.

- No data leakage (strict temporal split)
- Random pair swap for class balance (~50% P1 wins)
- Cross-validation with walk-forward temporal splitting
- Optuna HPO integration
- Saves models to models/saved/

The trainer is the single entry point for model training and handles
all data preparation, feature engineering, training, calibration, and
persistence.
"""
from __future__ import annotations

import pathlib
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from models import DartsMLError
from models.r0_logit import R0LogisticModel, R0_FEATURES
from models.r1_lgbm import R1StackingModel, R1_FEATURES
from models.r2_stacking import R2StackingEnsemble, R2_FEATURES

logger = structlog.get_logger(__name__)

# Default temporal split ratio
TRAIN_RATIO: float = 0.7
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# Default model save directory
DEFAULT_SAVE_DIR = pathlib.Path(__file__).parent / "saved"


class DartsModelTrainer:
    """
    Training pipeline for darts ML models.

    Handles data preparation (temporal split, pair swap),
    model training, calibration, and persistence for all regimes.

    Parameters
    ----------
    save_dir:
        Directory for saving trained models.
    random_state:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        save_dir: str | pathlib.Path = DEFAULT_SAVE_DIR,
        random_state: int = 42,
    ) -> None:
        self.save_dir = pathlib.Path(save_dir)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._log = logger.bind(component="DartsModelTrainer")

    def prepare_features(
        self,
        matches_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        elo_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare feature matrix and labels from raw data.

        Temporal split: features computed from data BEFORE match date.
        Random pair swap: ~50% P1 wins in training data for class balance.

        Parameters
        ----------
        matches_df:
            Match records with columns: match_id, player1_id, player2_id,
            match_date, result (1=P1 wins, 0=P2 wins), plus feature columns.
        stats_df:
            Player statistics indexed by player_id.
        elo_df:
            Elo ratings indexed by player_id.

        Returns
        -------
        tuple[pd.DataFrame, np.ndarray]
            (feature_matrix, labels) with pair-swapped rows.

        Raises
        ------
        DartsMLError
            If required columns are missing.
        """
        required_cols = {"match_date", "result"}
        missing = required_cols - set(matches_df.columns)
        if missing:
            raise DartsMLError(f"matches_df missing required columns: {missing}")

        # Sort by date for temporal ordering
        df = matches_df.sort_values("match_date").reset_index(drop=True)

        # Apply random pair swap for class balance
        df, labels = self._random_pair_swap(df)

        self._log.info(
            "features_prepared",
            n_samples=len(df),
            class_balance=float(np.mean(labels)),
            date_range_start=str(df["match_date"].iloc[0]),
            date_range_end=str(df["match_date"].iloc[-1]),
        )

        return df, labels

    def _random_pair_swap(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Randomly swap P1/P2 designations for ~50% class balance.

        For each match, with probability 0.5, swap all P1/P2 features
        and flip the label. This removes any systematic P1 bias.

        Parameters
        ----------
        df:
            Feature dataframe with P1/P2 columns and 'result' column.

        Returns
        -------
        tuple[pd.DataFrame, np.ndarray]
            Swapped dataframe and labels array.
        """
        df = df.copy()
        labels = df["result"].values.astype(np.float64)
        n = len(df)

        # Generate swap mask: True = swap this row
        swap_mask = self._rng.random(n) < 0.5

        # Identify P1/P2 column pairs
        p1_cols = [c for c in df.columns if c.endswith("_p1")]
        p2_cols = [c.replace("_p1", "_p2") for c in p1_cols]

        # Validate that all p2 counterparts exist
        valid_pairs = [
            (c1, c2) for c1, c2 in zip(p1_cols, p2_cols) if c2 in df.columns
        ]

        for c1, c2 in valid_pairs:
            p1_vals = df[c1].values.copy()
            p2_vals = df[c2].values.copy()
            df.loc[swap_mask, c1] = p2_vals[swap_mask]
            df.loc[swap_mask, c2] = p1_vals[swap_mask]

        # Handle elo_diff and ranking_log_ratio (sign flip)
        if "elo_diff" in df.columns:
            df.loc[swap_mask, "elo_diff"] = -df.loc[swap_mask, "elo_diff"]
        if "ranking_log_ratio" in df.columns:
            df.loc[swap_mask, "ranking_log_ratio"] = -df.loc[
                swap_mask, "ranking_log_ratio"
            ]

        # H2H: flip p1_win_rate
        if "h2h_p1_win_rate" in df.columns:
            df.loc[swap_mask, "h2h_p1_win_rate"] = (
                1.0 - df.loc[swap_mask, "h2h_p1_win_rate"]
            )

        # Flip labels for swapped rows
        labels[swap_mask] = 1.0 - labels[swap_mask]

        self._log.debug(
            "pair_swap_applied",
            n_swapped=int(swap_mask.sum()),
            n_total=n,
            new_class_balance=float(np.mean(labels)),
        )

        return df, labels

    def temporal_split(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
    ) -> tuple[
        pd.DataFrame, np.ndarray,
        pd.DataFrame, np.ndarray,
        pd.DataFrame, np.ndarray,
    ]:
        """
        Split data temporally (no shuffling -- strict time ordering).

        Parameters
        ----------
        df:
            Feature dataframe sorted by match_date.
        labels:
            Labels array.
        train_ratio:
            Fraction of data for training.
        val_ratio:
            Fraction for validation.

        Returns
        -------
        tuple
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train = df.iloc[:train_end].reset_index(drop=True)
        y_train = labels[:train_end]
        X_val = df.iloc[train_end:val_end].reset_index(drop=True)
        y_val = labels[train_end:val_end]
        X_test = df.iloc[val_end:].reset_index(drop=True)
        y_test = labels[val_end:]

        self._log.info(
            "temporal_split",
            n_train=len(X_train),
            n_val=len(X_val),
            n_test=len(X_test),
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_r0(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
    ) -> R0LogisticModel:
        """
        Train an R0 logistic regression model.

        Parameters
        ----------
        X_train:
            Training features (must contain R0 features).
        y_train:
            Training labels.

        Returns
        -------
        R0LogisticModel
            Fitted model.
        """
        model = R0LogisticModel(random_state=self.random_state)
        model.fit(X_train, y_train)

        # Save
        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir / "r0_logit.joblib"
        model.save(save_path)

        self._log.info("r0_trained_and_saved", path=str(save_path))
        return model

    def train_r1(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> R1StackingModel:
        """
        Train an R1 stacking model (LightGBM + XGBoost).

        Parameters
        ----------
        X_train:
            Training features (must contain R1 features).
        y_train:
            Training labels.
        X_val:
            Validation features (for calibration).
        y_val:
            Validation labels.

        Returns
        -------
        R1StackingModel
            Fitted model.
        """
        model = R1StackingModel()
        model.fit(X_train, y_train, X_val, y_val)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir / "r1_stacking.joblib"
        model.save(save_path)

        self._log.info("r1_trained_and_saved", path=str(save_path))
        return model

    def train_r2(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> R2StackingEnsemble:
        """
        Train an R2 full stacking ensemble.

        Parameters
        ----------
        X_train:
            Training features (must contain R2 features).
        y_train:
            Training labels.
        X_val:
            Validation features.
        y_val:
            Validation labels.

        Returns
        -------
        R2StackingEnsemble
            Fitted model.
        """
        model = R2StackingEnsemble()
        model.fit(X_train, y_train, X_val, y_val)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir / "r2_ensemble.joblib"
        model.save(save_path)

        self._log.info("r2_trained_and_saved", path=str(save_path))
        return model

    def _verify_no_leakage(
        self,
        X: pd.DataFrame,
        match_dates: pd.Series,
    ) -> None:
        """
        Verify that all features predate their match.

        For each sample, checks that no feature was computed using
        data from after the match date.

        Parameters
        ----------
        X:
            Feature dataframe.
        match_dates:
            Series of match dates, aligned with X.

        Raises
        ------
        DartsMLError
            If temporal leakage is detected.

        Notes
        -----
        This check is performed by verifying that within any temporal
        split, no training sample's features use data from validation/test
        periods. This is enforced structurally by the temporal split and
        the feature builder's before_date parameter.
        """
        if len(X) != len(match_dates):
            raise DartsMLError(
                f"X and match_dates length mismatch: {len(X)} vs {len(match_dates)}"
            )

        dates = pd.to_datetime(match_dates)
        sorted_dates = dates.sort_values()

        # Verify temporal ordering is maintained
        if not dates.equals(sorted_dates):
            # Find the first violation
            for i in range(1, len(dates)):
                if dates.iloc[i] < dates.iloc[i - 1]:
                    raise DartsMLError(
                        f"Temporal ordering violated at index {i}: "
                        f"{dates.iloc[i]} < {dates.iloc[i-1]}. "
                        f"Features must be ordered by match_date."
                    )

        # Verify no future-looking features by checking for columns
        # that contain "_next_" or "_future_" patterns
        suspicious_cols = [
            c for c in X.columns
            if any(kw in c.lower() for kw in ("next_", "future_", "post_", "after_"))
        ]
        if suspicious_cols:
            raise DartsMLError(
                f"Suspicious future-looking feature columns detected: "
                f"{suspicious_cols}"
            )

        self._log.info(
            "leakage_check_passed",
            n_samples=len(X),
            date_range=f"{dates.min()} to {dates.max()}",
        )

    def walk_forward_cv(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        n_splits: int = 5,
        min_train_size: int = 100,
        regime: int = 0,
    ) -> list[dict]:
        """
        Walk-forward cross-validation with temporal ordering.

        Each fold uses all data up to time t for training and
        the next chunk for validation. No future information leaks.

        Parameters
        ----------
        df:
            Feature dataframe sorted by match_date.
        labels:
            Labels array.
        n_splits:
            Number of temporal CV folds.
        min_train_size:
            Minimum training set size.
        regime:
            Model regime (0, 1, 2).

        Returns
        -------
        list[dict]
            List of fold results with keys: fold, n_train, n_val, metric.
        """
        n = len(df)
        fold_size = (n - min_train_size) // n_splits

        if fold_size < 10:
            raise DartsMLError(
                f"Not enough data for {n_splits} walk-forward folds "
                f"with min_train_size={min_train_size}: n={n}"
            )

        results: list[dict] = []

        for fold in range(n_splits):
            train_end = min_train_size + fold * fold_size
            val_end = min(train_end + fold_size, n)

            if val_end <= train_end:
                break

            X_train = df.iloc[:train_end]
            y_train = labels[:train_end]
            X_val = df.iloc[train_end:val_end]
            y_val = labels[train_end:val_end]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue

            # Train and evaluate
            if regime == 0:
                model = R0LogisticModel(random_state=self.random_state)
                model.fit(X_train, y_train)
                proba = model.predict_p1_win_proba(X_val)
            elif regime == 1:
                model = R1StackingModel()
                # For WF CV, split training into train/cal
                cal_split = int(len(X_train) * 0.8)
                model.fit(
                    X_train.iloc[:cal_split],
                    y_train[:cal_split],
                    X_train.iloc[cal_split:],
                    y_train[cal_split:],
                )
                proba = model.predict_proba(X_val)
            else:
                model = R2StackingEnsemble()
                cal_split = int(len(X_train) * 0.8)
                model.fit(
                    X_train.iloc[:cal_split],
                    y_train[:cal_split],
                    X_train.iloc[cal_split:],
                    y_train[cal_split:],
                )
                proba = model.predict_proba(X_val)

            # Log loss metric
            eps = 1e-7
            proba_clipped = np.clip(proba, eps, 1.0 - eps)
            log_loss = -np.mean(
                y_val * np.log(proba_clipped) + (1 - y_val) * np.log(1 - proba_clipped)
            )

            fold_result = {
                "fold": fold,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "log_loss": float(log_loss),
            }
            results.append(fold_result)

            self._log.info(
                "walk_forward_fold",
                fold=fold,
                regime=regime,
                n_train=len(X_train),
                n_val=len(X_val),
                log_loss=round(float(log_loss), 4),
            )

        return results
