"""
Ray parallel CPU simulator for multi-core tournament simulation.

Used as the CPU fallback when:
  - No GPU is available, OR
  - Field size < 64 players (GPU kernel overhead not worth it).

When Ray is not installed the module falls back gracefully to the
single-threaded DartsTournamentSimulator without raising an import error.

Architecture
------------
The simulation is split into n_workers equal-sized chunks.  Each chunk runs
as a Ray remote task on a separate CPU core.  Results are aggregated by
summing the per-player win/top4 counts.

If Ray is unavailable the simulate() method transparently delegates to
DartsTournamentSimulator, which has its own antithetic Monte Carlo loop.
"""
from __future__ import annotations

import structlog

from engines.errors import DartsEngineError
from outrights.tournament_simulator import (
    DartsTournamentSimulator,
    OutrightSimResult,
    TournamentField,
)

logger = structlog.get_logger(__name__)

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    ray = None  # type: ignore[assignment]
    RAY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Ray remote worker function
# ---------------------------------------------------------------------------

def _worker_func(
    field: TournamentField,
    n_simulations: int,
    worker_seed: int,
) -> dict:
    """
    Standalone function run inside a Ray remote task.

    Returns a dict with ``win_counts`` and ``top4_counts`` keyed by player_id
    and ``n_sims`` with the actual number of simulations completed.
    """
    import numpy as np
    from outrights.tournament_simulator import DartsTournamentSimulator

    sim = DartsTournamentSimulator()
    rng = np.random.default_rng(worker_seed)

    win_counts: dict[str, int] = {pid: 0 for pid in field.players}
    top4_counts: dict[str, int] = {pid: 0 for pid in field.players}

    for _ in range(n_simulations):
        placements = sim._simulate_bracket_once(field, rng)
        for pid, place in placements.items():
            if place == 1:
                win_counts[pid] += 1
            if place <= 4:
                top4_counts[pid] += 1

    return {
        "win_counts": win_counts,
        "top4_counts": top4_counts,
        "n_sims": n_simulations,
    }


# Create a Ray remote version only when Ray is available.
if RAY_AVAILABLE:
    _ray_worker = ray.remote(_worker_func)
else:
    _ray_worker = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Simulator class
# ---------------------------------------------------------------------------

class RayTournamentSimulator:
    """
    Multi-core parallel tournament simulator backed by Ray.

    Splits n_simulations evenly across n_workers Ray remote tasks, then
    aggregates results.  Falls back to the single-threaded
    DartsTournamentSimulator when Ray is not installed.
    """

    def simulate(
        self,
        field: TournamentField,
        n_simulations: int = 100_000,
        n_workers: int = 4,
    ) -> OutrightSimResult:
        """
        Simulate a tournament using multiple CPU cores.

        Parameters
        ----------
        field:
            Fully populated TournamentField.
        n_simulations:
            Total number of simulations to run (split across workers).
        n_workers:
            Number of parallel Ray workers.  Ignored when Ray is unavailable.

        Returns
        -------
        OutrightSimResult
        """
        if not field.players:
            raise DartsEngineError("TournamentField.players cannot be empty.")

        if RAY_AVAILABLE:
            return self._simulate_ray(field, n_simulations, n_workers)
        else:
            logger.info(
                "ray_not_available_fallback",
                fallback="DartsTournamentSimulator",
                n_simulations=n_simulations,
            )
            return DartsTournamentSimulator().simulate(
                competition_id=field.competition_id,
                field=field,
                n_simulations=n_simulations,
                use_gpu=False,
                use_antithetic=True,
            )

    def _simulate_ray(
        self,
        field: TournamentField,
        n_simulations: int,
        n_workers: int,
    ) -> OutrightSimResult:
        """Execute parallel simulation using Ray remote tasks."""
        import numpy as np

        n_workers = max(1, min(n_workers, n_simulations))
        sims_per_worker = n_simulations // n_workers
        remainder = n_simulations % n_workers

        # Generate distinct seeds for each worker
        rng_seed = np.random.default_rng()
        seeds = rng_seed.integers(1, 2**63, size=n_workers, dtype=np.uint64)

        # Distribute simulation counts
        worker_sim_counts = [sims_per_worker] * n_workers
        for i in range(remainder):
            worker_sim_counts[i] += 1

        # Launch tasks
        refs = []
        for i in range(n_workers):
            refs.append(
                _ray_worker.remote(field, worker_sim_counts[i], int(seeds[i]))  # type: ignore[attr-defined]
            )

        # Gather results
        results = ray.get(refs)

        # Aggregate
        win_counts: dict[str, int] = {pid: 0 for pid in field.players}
        top4_counts: dict[str, int] = {pid: 0 for pid in field.players}
        total_sims = 0

        for res in results:
            total_sims += res["n_sims"]
            for pid in field.players:
                win_counts[pid] += res["win_counts"].get(pid, 0)
                top4_counts[pid] += res["top4_counts"].get(pid, 0)

        win_probs = {
            pid: win_counts[pid] / total_sims
            for pid in field.players
        }
        top4_probs = {
            pid: top4_counts[pid] / total_sims
            for pid in field.players
        }

        # Confidence intervals via base simulator
        base_sim = DartsTournamentSimulator()
        ci = {
            pid: base_sim._compute_confidence_intervals(
                np.array([win_probs[pid]]),
                total_sims,
            )
            for pid in field.players
        }

        logger.info(
            "ray_sim_complete",
            competition_id=field.competition_id,
            n_workers=n_workers,
            n_simulations=total_sims,
            top_player=max(win_probs, key=win_probs.get),
            top_win_prob=round(max(win_probs.values()), 4),
        )

        return OutrightSimResult(
            competition_id=field.competition_id,
            player_win_probs=win_probs,
            top4_probs=top4_probs,
            n_simulations=total_sims,
            confidence_intervals=ci,
        )
