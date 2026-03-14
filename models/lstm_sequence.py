"""
PyTorch LSTM for sequential visit modeling.

Input: sequence of (visit_score, is_180, is_checkout_attempt, result, ...)
Output: momentum_state vector that feeds into R2 features.

The LSTM captures within-match momentum dynamics that static features
cannot model. The hidden state at the end of a sequence encodes
the player's recent form trajectory.

Architecture:
    Input -> LSTM (2 layers, 64 hidden) -> Linear -> momentum_vector (4-dim)

The 4-dimensional output represents:
    [momentum_level, momentum_volatility, scoring_trend, pressure_response]
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import structlog

from models import DartsMLError

logger = structlog.get_logger(__name__)

# Input features per visit step
VISIT_INPUT_SIZE: int = 8
# Hidden state dimensionality
HIDDEN_SIZE: int = 64
# Number of LSTM layers
NUM_LAYERS: int = 2
# Output momentum vector size
MOMENTUM_SIZE: int = 4


class DartsLSTMSequenceModel:
    """
    LSTM model for sequential visit momentum encoding.

    Processes a sequence of visit-level features and produces a
    fixed-size momentum state vector that summarises the player's
    recent form trajectory.

    The model is lazily loaded -- PyTorch is only imported when
    the model is actually used, allowing the rest of the system
    to run without a GPU.

    Parameters
    ----------
    input_size:
        Number of features per visit step.
    hidden_size:
        LSTM hidden state dimensionality.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout rate between LSTM layers.
    """

    def __init__(
        self,
        input_size: int = VISIT_INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = 0.1,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self._model: Optional[object] = None
        self._torch_available: bool = False
        self.fitted_: bool = False
        self._log = logger.bind(component="DartsLSTMSequenceModel")
        self._init_torch()

    def _init_torch(self) -> None:
        """Lazily initialise PyTorch model."""
        try:
            import torch
            import torch.nn as nn

            self._torch_available = True

            class _LSTMModule(nn.Module):
                def __init__(
                    self_inner,
                    input_size: int,
                    hidden_size: int,
                    num_layers: int,
                    dropout: float,
                ) -> None:
                    super().__init__()
                    self_inner.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=dropout if num_layers > 1 else 0.0,
                    )
                    self_inner.fc = nn.Linear(hidden_size, MOMENTUM_SIZE)
                    self_inner.activation = nn.Tanh()

                def forward(
                    self_inner, x: torch.Tensor
                ) -> torch.Tensor:
                    # x: (batch, seq_len, input_size)
                    lstm_out, (h_n, _) = self_inner.lstm(x)
                    # Use final hidden state from last layer
                    final_hidden = h_n[-1]  # (batch, hidden_size)
                    output = self_inner.activation(self_inner.fc(final_hidden))
                    return output  # (batch, MOMENTUM_SIZE)

            self._model = _LSTMModule(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
            self._log.info("lstm_torch_initialised")

        except ImportError:
            self._torch_available = False
            self._log.warning(
                "lstm_torch_unavailable",
                message="PyTorch not available. LSTM features will use fallback.",
            )

    def forward(self, visit_sequence: np.ndarray) -> np.ndarray:
        """
        Compute momentum state from a visit sequence.

        Parameters
        ----------
        visit_sequence:
            Array of shape (seq_len, input_size) or (batch, seq_len, input_size).
            Each row is a visit feature vector:
                [visit_score, is_180, is_checkout_attempt, checkout_success,
                 darts_thrown, score_remaining, leg_number, is_throw_first]

        Returns
        -------
        np.ndarray
            Momentum state vector of shape (MOMENTUM_SIZE,) or (batch, MOMENTUM_SIZE).
            Values in [-1, 1] (tanh output).

        Raises
        ------
        DartsMLError
            If the input shape is invalid.
        """
        visit_sequence = np.asarray(visit_sequence, dtype=np.float32)

        if visit_sequence.ndim == 2:
            visit_sequence = visit_sequence[np.newaxis, :, :]  # add batch dim
            squeeze = True
        elif visit_sequence.ndim == 3:
            squeeze = False
        else:
            raise DartsMLError(
                f"visit_sequence must be 2D or 3D, got {visit_sequence.ndim}D"
            )

        if visit_sequence.shape[2] != self.input_size:
            raise DartsMLError(
                f"Expected input_size={self.input_size}, "
                f"got {visit_sequence.shape[2]}"
            )

        if self._torch_available and self._model is not None:
            import torch

            with torch.no_grad():
                x = torch.from_numpy(visit_sequence).float()
                self._model.eval()  # type: ignore[union-attr]
                output = self._model(x).numpy()  # type: ignore[union-attr]
        else:
            # Fallback: compute simple momentum features without LSTM
            output = self._fallback_momentum(visit_sequence)

        if squeeze:
            return output[0]
        return output

    def _fallback_momentum(self, visit_sequence: np.ndarray) -> np.ndarray:
        """
        Compute momentum features without PyTorch (heuristic fallback).

        Uses exponentially weighted statistics over the visit sequence
        as a proxy for LSTM hidden state.

        Parameters
        ----------
        visit_sequence:
            Shape (batch, seq_len, input_size).

        Returns
        -------
        np.ndarray
            Shape (batch, MOMENTUM_SIZE).
        """
        batch_size = visit_sequence.shape[0]
        seq_len = visit_sequence.shape[1]
        output = np.zeros((batch_size, MOMENTUM_SIZE), dtype=np.float32)

        for b in range(batch_size):
            scores = visit_sequence[b, :, 0]  # visit_score column

            if seq_len == 0:
                continue

            # Momentum: EWM trend of recent scoring
            decay = 0.1
            weights = np.array(
                [(1.0 - decay) ** (seq_len - 1 - i) for i in range(seq_len)],
                dtype=np.float32,
            )
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum

            ewm_mean = float(np.sum(weights * scores))
            overall_mean = float(np.mean(scores))

            # momentum_level: normalised scoring trend
            output[b, 0] = np.tanh((ewm_mean - overall_mean) / max(overall_mean, 1.0))
            # momentum_volatility: variance of recent scores
            output[b, 1] = np.tanh(float(np.std(scores)) / 60.0 - 0.5)
            # scoring_trend: linear trend coefficient
            if seq_len > 1:
                x_t = np.arange(seq_len, dtype=np.float32)
                corr = np.corrcoef(x_t, scores)
                if corr.shape == (2, 2) and not np.isnan(corr[0, 1]):
                    output[b, 2] = np.tanh(float(corr[0, 1]))
            # pressure_response: performance in later visits vs earlier
            if seq_len >= 4:
                first_half = float(np.mean(scores[: seq_len // 2]))
                second_half = float(np.mean(scores[seq_len // 2:]))
                if first_half > 0:
                    output[b, 3] = np.tanh((second_half - first_half) / first_half)

        return output

    def fit(
        self,
        sequences: list[np.ndarray],
        labels: np.ndarray,
        n_epochs: int = 50,
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ) -> dict:
        """
        Train the LSTM on labelled visit sequences.

        Parameters
        ----------
        sequences:
            List of arrays, each shape (seq_len_i, input_size).
            Sequences may have different lengths (will be padded).
        labels:
            Binary labels, shape (n_samples,).
        n_epochs:
            Number of training epochs.
        learning_rate:
            Adam optimizer learning rate.
        batch_size:
            Training batch size.

        Returns
        -------
        dict
            Training history with loss per epoch.

        Raises
        ------
        DartsMLError
            If PyTorch is not available or inputs are invalid.
        """
        if not self._torch_available:
            raise DartsMLError(
                "PyTorch not available. Cannot train LSTM model."
            )

        import torch
        import torch.nn as nn
        from torch.optim import Adam

        labels = np.asarray(labels, dtype=np.float32)

        if len(sequences) != len(labels):
            raise DartsMLError(
                f"sequences and labels length mismatch: "
                f"{len(sequences)} vs {len(labels)}"
            )

        # Pad sequences to same length
        max_len = max(s.shape[0] for s in sequences)
        padded = np.zeros(
            (len(sequences), max_len, self.input_size),
            dtype=np.float32,
        )
        for i, seq in enumerate(sequences):
            padded[i, :seq.shape[0], :] = seq

        X_tensor = torch.from_numpy(padded)
        y_tensor = torch.from_numpy(labels).unsqueeze(1)

        model = self._model
        model.train()  # type: ignore[union-attr]
        optimizer = Adam(model.parameters(), lr=learning_rate)  # type: ignore[union-attr]
        criterion = nn.MSELoss()

        history: list[float] = []
        n_batches = max(1, len(sequences) // batch_size)

        for epoch in range(n_epochs):
            # Shuffle
            perm = torch.randperm(len(sequences))
            X_shuffled = X_tensor[perm]
            y_shuffled = y_tensor[perm]

            epoch_loss = 0.0
            for batch_start in range(0, len(sequences), batch_size):
                batch_end = min(batch_start + batch_size, len(sequences))
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]

                optimizer.zero_grad()
                momentum = model(X_batch)  # type: ignore[union-attr]
                # Use first momentum dimension as prediction target
                loss = criterion(momentum[:, 0:1], y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / n_batches
            history.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                self._log.debug(
                    "lstm_training_epoch",
                    epoch=epoch + 1,
                    loss=round(avg_loss, 6),
                )

        self.fitted_ = True
        self._log.info(
            "lstm_model_fitted",
            n_sequences=len(sequences),
            n_epochs=n_epochs,
            final_loss=round(history[-1], 6),
        )

        return {"loss_history": history}

    def save(self, path: str) -> None:
        """Save the LSTM model state."""
        if not self._torch_available or self._model is None:
            raise DartsMLError("Cannot save: PyTorch model not available.")

        import torch
        torch.save(self._model.state_dict(), path)  # type: ignore[union-attr]
        self._log.info("lstm_model_saved", path=path)

    def load(self, path: str) -> None:
        """Load a previously saved LSTM model state."""
        if not self._torch_available or self._model is None:
            raise DartsMLError("Cannot load: PyTorch model not available.")

        import torch
        self._model.load_state_dict(torch.load(path, weights_only=True))  # type: ignore[union-attr]
        self.fitted_ = True
        self._log.info("lstm_model_loaded", path=path)
