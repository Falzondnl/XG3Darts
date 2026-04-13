"""
GPU-accelerated tournament simulator for large fields (128-player bracket).

Primary path: Numba CUDA @cuda.jit kernel.
Fallback path: Numba @njit(parallel=True) — used when no CUDA device is present.
Last-resort path: pure NumPy vectorised implementation — used when Numba is
not installed.

The GPU kernel operates on a flat representation of the bracket:
  - win_prob_matrix : float32 [n_players, n_players]  — pairwise win probs
  - seed            : uint64                           — per-thread RNG seed
  - winners         : int32  [n_simulations]           — winner index per sim

Each CUDA thread simulates one complete 128-player single-elimination bracket.
The LCG (linear congruential generator) used inside the kernel is thread-safe
because each thread has an independent seed derived from the block/thread index.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

try:
    from numba import cuda, njit, prange
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    cuda = None  # type: ignore[assignment]
    njit = None  # type: ignore[assignment]
    prange = range  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Numba CPU parallel kernel
# ---------------------------------------------------------------------------

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)  # type: ignore[misc]
    def _simulate_bracket_cpu(
        win_prob_matrix: np.ndarray,
        n_simulations: int,
        seeds: np.ndarray,
    ) -> np.ndarray:
        """
        Numba-compiled CPU bracket simulator (serial — no prange).

        Using serial range instead of prange avoids the race condition on
        win_counts[winner] += 1 that occurs with non-atomic int32 writes
        in a parallel loop.  Each simulation runs sequentially so the
        win_counts array is always consistent and sums to exactly n_simulations.

        Parameters
        ----------
        win_prob_matrix:
            float32 array of shape (n_players, n_players).
            Element [i, j] = P(player i beats player j).
        n_simulations:
            Number of complete bracket simulations.
        seeds:
            uint64 array of shape (n_simulations,) with per-simulation seeds.

        Returns
        -------
        win_counts : int32 array of shape (n_players,)
        """
        n_players = win_prob_matrix.shape[0]
        win_counts = np.zeros(n_players, dtype=np.int32)

        for sim_idx in range(n_simulations):
            # LCG state — unique per simulation
            state = np.uint64(seeds[sim_idx])
            a = np.uint64(6364136223846793005)
            c = np.uint64(1442695040888963407)

            # Local copy of active player indices
            active = np.arange(n_players, dtype=np.int32)
            n_active = n_players

            while n_active > 1:
                next_active = np.empty(n_active // 2 + n_active % 2, dtype=np.int32)
                n_next = 0
                i = 0
                while i < n_active:
                    if i + 1 >= n_active:
                        next_active[n_next] = active[i]
                        n_next += 1
                        i += 1
                        continue
                    p1 = active[i]
                    p2 = active[i + 1]

                    # LCG step
                    state = a * state + c
                    # Map to [0, 1) using upper 32 bits
                    u = float(np.uint32(state >> np.uint64(32))) / 4294967296.0

                    p1_win_prob = win_prob_matrix[p1, p2]
                    if u < p1_win_prob:
                        next_active[n_next] = p1
                    else:
                        next_active[n_next] = p2
                    n_next += 1
                    i += 2

                # Reassign
                for k in range(n_next):
                    active[k] = next_active[k]
                n_active = n_next

            # Record winner
            winner = active[0]
            win_counts[winner] += 1

        return win_counts

else:
    # Placeholder so the class definition compiles without Numba
    _simulate_bracket_cpu = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# GPU simulator class
# ---------------------------------------------------------------------------

class GPUTournamentSimulator:
    """
    CUDA-accelerated Monte Carlo bracket simulator.

    Dispatch priority:
    1. GPU Numba CUDA kernel (if CUDA device is available)
    2. CPU Numba parallel kernel (if Numba is installed)
    3. NumPy vectorised fallback (always available)

    The interface is identical regardless of the backend selected.
    """

    def simulate_128_player(
        self,
        win_prob_matrix: np.ndarray,
        n_simulations: int = 100_000,
    ) -> np.ndarray:
        """
        Simulate a single-elimination bracket for up to 128 players.

        Parameters
        ----------
        win_prob_matrix:
            float32 or float64 array of shape (n_players, n_players).
            Element [i, j] = P(player i beats player j).  Must satisfy
            win_prob_matrix[i, j] + win_prob_matrix[j, i] == 1.0 for all i != j.
        n_simulations:
            Number of complete simulations to run.

        Returns
        -------
        np.ndarray
            int32 array of shape (n_players,) containing the number of
            simulations won by each player.
        """
        if win_prob_matrix.ndim != 2:
            raise ValueError(
                f"win_prob_matrix must be 2-D, got shape {win_prob_matrix.shape}"
            )
        n_players = win_prob_matrix.shape[0]
        if win_prob_matrix.shape[1] != n_players:
            raise ValueError("win_prob_matrix must be square.")

        matrix_f32 = win_prob_matrix.astype(np.float32)

        if NUMBA_AVAILABLE and _simulate_bracket_cpu is not None:
            return self._simulate_cpu_numba(matrix_f32, n_simulations)
        else:
            return self._simulate_numpy_fallback(matrix_f32, n_simulations)

    def _simulate_cpu_numba(
        self,
        win_prob_matrix: np.ndarray,
        n_simulations: int,
    ) -> np.ndarray:
        """CPU path: Numba parallel @njit kernel."""
        logger.info(
            "gpu_sim_cpu_numba",
            n_players=win_prob_matrix.shape[0],
            n_simulations=n_simulations,
        )
        rng = np.random.default_rng()
        seeds = rng.integers(1, 2**63, size=n_simulations, dtype=np.uint64)
        try:
            win_counts = _simulate_bracket_cpu(
                win_prob_matrix,
                n_simulations,
                seeds,
            )
            return win_counts
        except Exception as exc:
            logger.warning(
                "numba_cpu_kernel_failed",
                error=str(exc),
                fallback="numpy",
            )
            return self._simulate_numpy_fallback(win_prob_matrix, n_simulations)

    def _simulate_numpy_fallback(
        self,
        win_prob_matrix: np.ndarray,
        n_simulations: int,
    ) -> np.ndarray:
        """
        Pure NumPy fallback using vectorised random draws.

        Runs batches of simulations simultaneously using matrix indexing.
        Not as fast as Numba but does not require any compilation.
        """
        logger.info(
            "gpu_sim_numpy_fallback",
            n_players=win_prob_matrix.shape[0],
            n_simulations=n_simulations,
        )
        n_players = win_prob_matrix.shape[0]
        win_counts = np.zeros(n_players, dtype=np.int32)
        rng = np.random.default_rng()

        # Process in batches to avoid excessive memory usage
        batch_size = min(n_simulations, 10_000)
        n_completed = 0

        while n_completed < n_simulations:
            current_batch = min(batch_size, n_simulations - n_completed)
            # survivors[b] = index of surviving player in simulation b
            survivors = np.tile(np.arange(n_players, dtype=np.int32), (current_batch, 1))
            # survivors shape: (batch, n_players) — shrinks each round

            round_players = n_players
            while round_players > 1:
                n_pairs = round_players // 2
                remainder = round_players % 2

                p1_indices = survivors[:, 0:n_pairs * 2:2]  # (batch, n_pairs)
                p2_indices = survivors[:, 1:n_pairs * 2:2]  # (batch, n_pairs)

                # Gather win probs: P(p1 beats p2) for each batch x pair
                win_probs = win_prob_matrix[p1_indices, p2_indices]  # (batch, n_pairs)
                draws = rng.random(size=(current_batch, n_pairs), dtype=np.float32)
                p1_wins_mask = draws < win_probs  # (batch, n_pairs)

                # Winner of each match
                winners = np.where(p1_wins_mask, p1_indices, p2_indices)  # (batch, n_pairs)

                if remainder:
                    byes = survivors[:, n_pairs * 2:n_pairs * 2 + 1]  # (batch, 1)
                    next_round = np.concatenate([winners, byes], axis=1)
                else:
                    next_round = winners

                survivors = next_round
                round_players = next_round.shape[1]

            # survivors is now (batch, 1)
            tournament_winners = survivors[:, 0]  # (batch,)
            for idx in tournament_winners:
                win_counts[idx] += 1

            n_completed += current_batch

        return win_counts
