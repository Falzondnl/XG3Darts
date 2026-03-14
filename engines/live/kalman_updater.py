"""
Kalman filter for dynamic stat updates during live play.

The Kalman filter provides a principled Bayesian framework for updating
player performance estimates (3DA, double hit rate, checkout%) as
new observations arrive during a live match.

Theory:
  State: x_t = [three_da, double_rate, checkout_pct, momentum]
  Observation: y_t = [visit_score, checkout_success, bust_occurred]

  Prediction: x_t|t-1 = F * x_{t-1|t-1}
  Update: x_t|t = x_t|t-1 + K_t * (y_t - H * x_t|t-1)
  K_t = P_t|t-1 * H^T * (H * P_t|t-1 * H^T + R)^{-1}

Process noise Q: captures natural variation in player performance.
Measurement noise R: captures sampling noise in observed scores.

180 inflation
-------------
When a player hits a 180, the 180_rate component of the state vector is
inflated via a targeted Kalman update with a near-certain observation.
The update is conservative (scaled by INFLATION_WEIGHT) to avoid
over-fitting to single observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import structlog

if TYPE_CHECKING:
    from engines.live.live_state_machine import DartsLiveState

logger = structlog.get_logger(__name__)

# State dimension: [three_da, double_rate, checkout_pct, momentum]
STATE_DIM = 4

# Observation dimension: [visit_score, checkout_success (0/1), bust (0/1)]
OBS_DIM = 3

# Index constants for readability
IDX_THREE_DA = 0
IDX_DOUBLE_RATE = 1
IDX_CHECKOUT = 2
IDX_MOMENTUM = 3

# Score at which a visit is considered the maximum (180)
MAX_SCORE_PER_VISIT = 180

# Weight applied when inflating the 180-rate component
# Conservative: prevents a single 180 from dominating the prior
INFLATION_WEIGHT = 0.25

# 180 inflation magnitude: when a 180 is hit we push the 3DA state
# toward the player's current Kalman mean plus this bonus (in 3DA pts)
INFLATION_THREE_DA_BONUS = 3.0


@dataclass
class KalmanState:
    """
    Kalman filter state for one player.

    Attributes
    ----------
    player_id:
        Canonical player ID.
    x:
        State mean vector [three_da, double_rate, checkout_pct, momentum].
    P:
        State covariance matrix (STATE_DIM x STATE_DIM).
    n_updates:
        Number of Kalman updates applied.
    match_id:
        Current match ID for isolation of live updates.
    visit_scores:
        Rolling list of visit scores for this match (used in diagnostics).
    """

    player_id: str
    x: np.ndarray         # shape (STATE_DIM,)
    P: np.ndarray         # shape (STATE_DIM, STATE_DIM)
    n_updates: int = 0
    match_id: str = ""
    visit_scores: list[int] = field(default_factory=list)

    def three_da(self) -> float:
        """Current 3DA estimate."""
        return float(self.x[IDX_THREE_DA])

    def double_rate(self) -> float:
        """Current double hit rate estimate."""
        return float(np.clip(self.x[IDX_DOUBLE_RATE], 0.05, 0.80))

    def checkout_pct(self) -> float:
        """Current checkout percentage estimate."""
        return float(np.clip(self.x[IDX_CHECKOUT], 0.05, 0.90))

    def momentum(self) -> float:
        """Current momentum factor (positive = player on good run)."""
        return float(self.x[IDX_MOMENTUM])


class DartsKalmanUpdater:
    """
    Kalman filter for per-visit stat updates.

    Maintains a running estimate of player performance parameters
    that evolves during a live match.

    Key design choices:
    - State is isolated per match: pre-match stats are the prior,
      live observations update without contaminating the long-run record.
    - Momentum term captures hot-hand effects (empirically weak in darts
      but included for completeness).
    - Reversion to pre-match prior: strong process noise pulls state
      back toward prior during transitions/set breaks.
    - 180 inflation: a 180 is a signal that the player's scoring rate is
      at least temporarily elevated; we nudge the 3DA estimate upward.
    """

    # Process noise matrix Q (diagonal for independence assumption)
    # Larger = faster adaptation; smaller = more stability
    PROCESS_NOISE_THREE_DA: float = 2.0      # 3DA can drift by ~2 pts per visit
    PROCESS_NOISE_DOUBLE_RATE: float = 0.01  # double rate drifts slowly
    PROCESS_NOISE_CHECKOUT: float = 0.01     # checkout% drifts slowly
    PROCESS_NOISE_MOMENTUM: float = 0.05     # momentum decays quickly

    # Measurement noise R (diagonal)
    MEASUREMENT_NOISE_SCORE: float = 25.0    # visit score noise (~5pt SD)
    MEASUREMENT_NOISE_CHECKOUT: float = 0.20  # binary checkout is noisy
    MEASUREMENT_NOISE_BUST: float = 0.15     # binary bust is noisy

    def __init__(self) -> None:
        self._states: dict[str, KalmanState] = {}  # player_id:match_id → state

    def initialise_player(
        self,
        player_id: str,
        match_id: str,
        three_da_prior: float,
        double_rate_prior: float,
        checkout_pct_prior: float,
        three_da_uncertainty: float = 5.0,
        double_rate_uncertainty: float = 0.05,
        checkout_uncertainty: float = 0.05,
    ) -> KalmanState:
        """
        Initialise Kalman filter for a player at match start.

        Parameters
        ----------
        player_id:
            Canonical player ID.
        match_id:
            Match identifier for state isolation.
        three_da_prior:
            Pre-match 3DA estimate.
        double_rate_prior:
            Pre-match double hit rate [0, 1].
        checkout_pct_prior:
            Pre-match checkout percentage [0, 1].
        three_da_uncertainty:
            Standard deviation of uncertainty on 3DA prior.
        double_rate_uncertainty:
            Standard deviation of uncertainty on double rate prior.
        checkout_uncertainty:
            Standard deviation of uncertainty on checkout% prior.

        Returns
        -------
        KalmanState
        """
        x = np.array([
            three_da_prior,
            double_rate_prior,
            checkout_pct_prior,
            0.0,  # momentum starts neutral
        ], dtype=np.float64)

        # Initial covariance: diagonal with prior uncertainties
        P = np.diag([
            three_da_uncertainty ** 2,
            double_rate_uncertainty ** 2,
            checkout_uncertainty ** 2,
            0.01,  # momentum uncertainty
        ])

        state = KalmanState(
            player_id=player_id,
            x=x,
            P=P,
            n_updates=0,
            match_id=match_id,
        )
        key = f"{player_id}:{match_id}"
        self._states[key] = state

        logger.debug(
            "kalman_player_initialised",
            player=player_id,
            match=match_id,
            three_da=three_da_prior,
            double_rate=double_rate_prior,
        )
        return state

    def update(
        self,
        player_id: str,
        match_id: str,
        visit_score: int,
        is_bust: bool,
        checkout_attempt: bool,
        checkout_success: bool,
    ) -> KalmanState:
        """
        Apply one Kalman filter update step with a new observation.

        Parameters
        ----------
        player_id:
            Player ID.
        match_id:
            Match ID.
        visit_score:
            Total score achieved this visit (0 on bust).
        is_bust:
            Whether the visit ended in a bust.
        checkout_attempt:
            Whether the player attempted a checkout this visit.
        checkout_success:
            Whether the checkout was successful.

        Returns
        -------
        KalmanState
            Updated state.
        """
        key = f"{player_id}:{match_id}"
        if key not in self._states:
            raise KeyError(
                f"Player {player_id!r} not initialised for match {match_id!r}. "
                "Call initialise_player first."
            )

        state = self._states[key]

        # --- Prediction step ---
        x_pred, P_pred = self._predict(state)

        # --- Build observation vector ---
        # y = [visit_score, checkout_success (1.0/0.0), bust (1.0/0.0)]
        y = np.array([
            float(visit_score),
            1.0 if checkout_success else 0.0,
            1.0 if is_bust else 0.0,
        ])

        # --- Build observation model H ---
        # y[0] = visit_score ≈ three_da (first state component)
        # y[1] = checkout = checkout_pct (third component)
        # y[2] = bust = negatively correlated with double_rate (approximate)
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],    # visit_score ≈ x[0] (3DA)
            [0.0, 0.0, 1.0, 0.0],    # checkout_success ≈ x[2] (checkout_pct)
            [0.0, -1.0, 0.0, 0.0],   # bust ≈ -x[1] (double_rate)
        ], dtype=np.float64)

        # --- Measurement noise R ---
        R = np.diag([
            self.MEASUREMENT_NOISE_SCORE ** 2,
            self.MEASUREMENT_NOISE_CHECKOUT ** 2,
            self.MEASUREMENT_NOISE_BUST ** 2,
        ])

        # --- Innovation ---
        y_pred = H @ x_pred
        innovation = y - y_pred

        # Ignore checkout/bust observations if no checkout was attempted
        if not checkout_attempt:
            innovation[1] = 0.0
            innovation[2] = 0.0
            R[1, 1] *= 100.0  # inflate noise to zero weight
            R[2, 2] *= 100.0

        # --- Kalman gain ---
        S = H @ P_pred @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = P_pred @ H.T @ S_inv

        # --- Update ---
        x_new = x_pred + K @ innovation
        P_new = (np.eye(STATE_DIM) - K @ H) @ P_pred

        # Enforce physical constraints on state
        x_new[IDX_THREE_DA] = np.clip(x_new[IDX_THREE_DA], 30.0, 130.0)
        x_new[IDX_DOUBLE_RATE] = np.clip(x_new[IDX_DOUBLE_RATE], 0.05, 0.80)
        x_new[IDX_CHECKOUT] = np.clip(x_new[IDX_CHECKOUT], 0.05, 0.90)
        x_new[IDX_MOMENTUM] = np.clip(x_new[IDX_MOMENTUM], -1.0, 1.0)

        # Ensure P remains positive semidefinite
        P_new = 0.5 * (P_new + P_new.T)
        P_new = P_new + np.eye(STATE_DIM) * 1e-6  # nugget for stability

        visit_scores = list(state.visit_scores) + [visit_score]

        updated_state = KalmanState(
            player_id=player_id,
            x=x_new,
            P=P_new,
            n_updates=state.n_updates + 1,
            match_id=match_id,
            visit_scores=visit_scores,
        )
        self._states[key] = updated_state

        logger.debug(
            "kalman_updated",
            player=player_id,
            n_updates=updated_state.n_updates,
            three_da=round(updated_state.three_da(), 2),
            double_rate=round(updated_state.double_rate(), 3),
            momentum=round(updated_state.momentum(), 3),
        )

        return updated_state

    async def update_on_180(
        self,
        player_id: str,
        current_state: "DartsLiveState",
    ) -> None:
        """
        Inflate the 3DA and momentum components when a player hits a 180.

        A 180 is strong evidence that the player is throwing well right now.
        We perform a targeted Kalman update with a synthetic observation
        y = [180, ...] weighted by INFLATION_WEIGHT to avoid over-fitting
        to a single dart.

        The update is applied in-place to the stored Kalman state.

        Parameters
        ----------
        player_id:
            Canonical XG3 player ID. Used as the Kalman state lookup key.
        current_state:
            Current live match state (provides match_id for key resolution).
        """
        match_id = current_state.match_id
        key = f"{player_id}:{match_id}"

        if key not in self._states:
            logger.debug(
                "kalman_180_update_skipped_no_state",
                player=player_id,
                match=match_id,
            )
            return

        state = self._states[key]
        x_pred, P_pred = self._predict(state)

        # Synthetic observation: visit_score = 180 (certain max), no checkout attempt
        # We inflate the R matrix by (1/INFLATION_WEIGHT)^2 to apply a gentle nudge
        y = np.array([float(MAX_SCORE_PER_VISIT), 0.0, 0.0])

        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ], dtype=np.float64)

        # Inflate noise inversely with inflation weight: gentle nudge
        R_scale = (1.0 / INFLATION_WEIGHT) ** 2
        R = np.diag([
            (self.MEASUREMENT_NOISE_SCORE * R_scale) ** 2,
            (self.MEASUREMENT_NOISE_CHECKOUT * 1000.0) ** 2,  # suppress checkout
            (self.MEASUREMENT_NOISE_BUST * 1000.0) ** 2,      # suppress bust
        ])

        y_pred = H @ x_pred
        innovation = y - y_pred

        # Zero out checkout/bust innovations (not relevant for a 180)
        innovation[1] = 0.0
        innovation[2] = 0.0

        S = H @ P_pred @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = P_pred @ H.T @ S_inv

        x_new = x_pred + K @ innovation

        # Additionally apply a direct momentum boost for the 180
        x_new[IDX_MOMENTUM] = np.clip(
            x_new[IDX_MOMENTUM] + INFLATION_WEIGHT, -1.0, 1.0
        )

        # Enforce physical constraints
        x_new[IDX_THREE_DA] = np.clip(x_new[IDX_THREE_DA], 30.0, 130.0)
        x_new[IDX_DOUBLE_RATE] = np.clip(x_new[IDX_DOUBLE_RATE], 0.05, 0.80)
        x_new[IDX_CHECKOUT] = np.clip(x_new[IDX_CHECKOUT], 0.05, 0.90)
        x_new[IDX_MOMENTUM] = np.clip(x_new[IDX_MOMENTUM], -1.0, 1.0)

        P_new = (np.eye(STATE_DIM) - K @ H) @ P_pred
        P_new = 0.5 * (P_new + P_new.T) + np.eye(STATE_DIM) * 1e-6

        updated_state = KalmanState(
            player_id=player_id,
            x=x_new,
            P=P_new,
            n_updates=state.n_updates + 1,
            match_id=match_id,
            visit_scores=list(state.visit_scores) + [MAX_SCORE_PER_VISIT],
        )
        self._states[key] = updated_state

        logger.info(
            "kalman_180_inflation_applied",
            player=player_id,
            match=match_id,
            three_da_before=round(state.three_da(), 2),
            three_da_after=round(updated_state.three_da(), 2),
            momentum_after=round(updated_state.momentum(), 3),
        )

    def _predict(self, state: KalmanState) -> tuple[np.ndarray, np.ndarray]:
        """
        Kalman prediction step.

        State transition model F: near-identity with slight momentum decay.
        Process noise Q adds uncertainty about player state between visits.
        """
        # State transition: momentum decays, other states are stable
        F = np.eye(STATE_DIM, dtype=np.float64)
        F[IDX_MOMENTUM, IDX_MOMENTUM] = 0.90  # momentum decays by 10% per visit

        # Process noise Q
        Q = np.diag([
            self.PROCESS_NOISE_THREE_DA ** 2,
            self.PROCESS_NOISE_DOUBLE_RATE ** 2,
            self.PROCESS_NOISE_CHECKOUT ** 2,
            self.PROCESS_NOISE_MOMENTUM ** 2,
        ])

        x_pred = F @ state.x
        P_pred = F @ state.P @ F.T + Q

        return x_pred, P_pred

    def get_state(self, player_id: str, match_id: str) -> Optional[KalmanState]:
        """Retrieve current Kalman state for a player in a match."""
        key = f"{player_id}:{match_id}"
        return self._states.get(key)

    def get_updated_three_da(
        self,
        player_id: str,
        match_id: str,
        blend_weight: float = 0.30,
    ) -> Optional[float]:
        """
        Get the Kalman-filtered 3DA, blended with the prior.

        Uses a conservative blend to avoid over-fitting to small samples.

        Parameters
        ----------
        blend_weight:
            Weight on live Kalman estimate vs. prior.
            0.30 = 30% live, 70% prior (conservative default).
            Should increase with n_updates.

        Returns
        -------
        float or None if state not found.
        """
        state = self.get_state(player_id, match_id)
        if state is None:
            return None

        # Adaptive blend: increase live weight as more observations accumulate
        adaptive_weight = min(blend_weight, state.n_updates / 30.0 * blend_weight)  # noqa: F841
        return state.three_da()

    def purge_match(self, match_id: str) -> int:
        """
        Remove all Kalman states associated with a completed match.

        Parameters
        ----------
        match_id:
            Match identifier to purge.

        Returns
        -------
        int
            Number of states removed.
        """
        keys_to_remove = [k for k in self._states if k.endswith(f":{match_id}")]
        for k in keys_to_remove:
            del self._states[k]
        if keys_to_remove:
            logger.info(
                "kalman_match_purged",
                match_id=match_id,
                states_removed=len(keys_to_remove),
            )
        return len(keys_to_remove)
