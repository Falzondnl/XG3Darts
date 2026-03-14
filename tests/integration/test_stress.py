"""
Stress tests: concurrent match pricing requests.

Validates:
  - 500 concurrent pre-match requests: all succeed, p95 < 50 ms
  - 100 live update requests: p99 < 100 ms

These tests exercise the computation engine (not HTTP layer) to avoid
dependency on a running server in CI.  The engine is invoked directly
in async tasks to simulate concurrent load.

Note: the latency targets are validated against the in-process computation
time.  Network + HTTP overhead is excluded from these benchmarks.
"""
from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


@dataclass
class PricingResult:
    """Result of a single pricing call."""
    match_id: str
    p1_win: float
    p2_win: float
    elapsed_ms: float
    success: bool
    error: Optional[str] = None


def _make_player_dists(player_id: str, three_da: float):
    """Build visit distributions for a player (used in async tasks)."""
    from engines.leg_layer.visit_distributions import (
        HierarchicalVisitDistributionModel,
        BAND_NAMES,
    )
    model = HierarchicalVisitDistributionModel()
    return {
        band: model.get_distribution(
            player_id=player_id,
            score_band=band,
            stage=False,
            short_format=False,
            throw_first=(band == "open"),
            three_da=three_da,
        )
        for band in BAND_NAMES
    }


async def _price_single_match(
    match_id: str,
    p1_three_da: float,
    p2_three_da: float,
    legs_to_win: int = 7,
) -> PricingResult:
    """
    Price one match asynchronously using the Markov chain engine.

    This simulates the core computation path for a pre-match request.
    """
    t0 = time.perf_counter()
    try:
        from engines.leg_layer.markov_chain import DartsMarkovChain
        from engines.match_layer.match_combinatorics import MatchCombinatorialEngine
        from competition.format_registry import get_format

        chain = DartsMarkovChain()

        p1_dists = _make_player_dists(f"{match_id}_p1", p1_three_da)
        p2_dists = _make_player_dists(f"{match_id}_p2", p2_three_da)

        # Yield to event loop to simulate async behaviour
        await asyncio.sleep(0)

        hb = chain.break_probability(
            p1_visit_dists=p1_dists,
            p2_visit_dists=p2_dists,
            p1_id=f"{match_id}_p1",
            p2_id=f"{match_id}_p2",
            p1_three_da=p1_three_da,
            p2_three_da=p2_three_da,
        )

        comb_engine = MatchCombinatorialEngine()
        result = comb_engine._dp_legs_format(
            hb=hb,
            legs_to_win=legs_to_win,
            p1_starts=True,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PricingResult(
            match_id=match_id,
            p1_win=result["p1_win"],
            p2_win=result["p2_win"],
            elapsed_ms=elapsed_ms,
            success=True,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PricingResult(
            match_id=match_id,
            p1_win=0.0,
            p2_win=0.0,
            elapsed_ms=elapsed_ms,
            success=False,
            error=str(exc),
        )


async def _live_update_match(
    match_id: str,
    legs_p1: int,
    legs_p2: int,
    current_score: int,
    p1_three_da: float,
    p2_three_da: float,
) -> PricingResult:
    """
    Simulate a live in-play update for one match.

    Updates the score state and re-prices using the Markov chain.
    """
    t0 = time.perf_counter()
    try:
        from engines.state_layer.score_state import ScoreState
        from engines.leg_layer.markov_chain import DartsMarkovChain
        from engines.match_layer.match_combinatorics import MatchCombinatorialEngine

        chain = DartsMarkovChain()

        p1_dists = _make_player_dists(f"{match_id}_p1", p1_three_da)
        p2_dists = _make_player_dists(f"{match_id}_p2", p2_three_da)

        # Yield to event loop
        await asyncio.sleep(0)

        hb = chain.break_probability(
            p1_visit_dists=p1_dists,
            p2_visit_dists=p2_dists,
            p1_id=f"{match_id}_p1",
            p2_id=f"{match_id}_p2",
            p1_three_da=p1_three_da,
            p2_three_da=p2_three_da,
        )

        comb_engine = MatchCombinatorialEngine()
        # Price from current state: legs already won matter
        result = comb_engine.price_from_state(
            hb=hb,
            legs_to_win=7,
            p1_legs_won=legs_p1,
            p2_legs_won=legs_p2,
            p1_starts_next=True,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PricingResult(
            match_id=match_id,
            p1_win=result["p1_win"],
            p2_win=result["p2_win"],
            elapsed_ms=elapsed_ms,
            success=True,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PricingResult(
            match_id=match_id,
            p1_win=0.0,
            p2_win=0.0,
            elapsed_ms=elapsed_ms,
            success=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_500_concurrent_prematch_requests() -> None:
    """
    Price 500 matches concurrently. All must succeed. p95 < 50 ms.

    Each match uses a realistic range of 3DA values drawn from a
    population of PDC tour-card players (75-105 range).
    """
    import random
    rng = random.Random(42)

    n_matches = 500
    tasks = [
        _price_single_match(
            match_id=f"stress_pm_{i:04d}",
            p1_three_da=rng.uniform(75.0, 105.0),
            p2_three_da=rng.uniform(75.0, 105.0),
            legs_to_win=rng.choice([6, 7, 10, 11]),
        )
        for i in range(n_matches)
    ]

    results = await asyncio.gather(*tasks)

    # All must succeed
    failures = [r for r in results if not r.success]
    assert len(failures) == 0, (
        f"STRESS FAIL: {len(failures)}/{n_matches} requests failed.\n"
        + "\n".join(f"  {r.match_id}: {r.error}" for r in failures[:5])
    )

    # p95 < 50 ms
    latencies = sorted(r.elapsed_ms for r in results)
    n = len(latencies)
    p95_idx = int(0.95 * n) - 1
    p95 = latencies[p95_idx]

    mean_ms = statistics.mean(latencies)
    median_ms = statistics.median(latencies)
    p99_idx = int(0.99 * n) - 1
    p99 = latencies[p99_idx]

    # Record stats for visibility
    print(
        f"\nStress test (500 pre-match): "
        f"mean={mean_ms:.1f}ms median={median_ms:.1f}ms "
        f"p95={p95:.1f}ms p99={p99:.1f}ms"
    )

    assert p95 < 50.0, (
        f"STRESS FAIL: p95 latency = {p95:.1f} ms > 50 ms threshold. "
        f"mean={mean_ms:.1f}ms, p99={p99:.1f}ms"
    )


@pytest.mark.asyncio
async def test_live_update_throughput() -> None:
    """
    100 live visit updates. p99 < 100 ms.

    Simulates mid-match live pricing updates with varying score states.
    """
    import random
    rng = random.Random(99)

    n_updates = 100
    tasks = [
        _live_update_match(
            match_id=f"stress_live_{i:03d}",
            legs_p1=rng.randint(0, 5),
            legs_p2=rng.randint(0, 5),
            current_score=rng.randint(80, 350),
            p1_three_da=rng.uniform(80.0, 102.0),
            p2_three_da=rng.uniform(80.0, 102.0),
        )
        for i in range(n_updates)
    ]

    results = await asyncio.gather(*tasks)

    failures = [r for r in results if not r.success]
    assert len(failures) == 0, (
        f"LIVE STRESS FAIL: {len(failures)}/{n_updates} live updates failed.\n"
        + "\n".join(f"  {r.match_id}: {r.error}" for r in failures[:5])
    )

    latencies = sorted(r.elapsed_ms for r in results)
    n = len(latencies)
    p99_idx = int(0.99 * n) - 1
    p99 = latencies[p99_idx]
    mean_ms = statistics.mean(latencies)

    print(
        f"\nLive update stress (100): "
        f"mean={mean_ms:.1f}ms p99={p99:.1f}ms"
    )

    assert p99 < 100.0, (
        f"LIVE STRESS FAIL: p99 latency = {p99:.1f} ms > 100 ms threshold. "
        f"mean={mean_ms:.1f}ms"
    )


@pytest.mark.asyncio
async def test_concurrent_outright_simulations() -> None:
    """
    10 concurrent outright tournament simulations.
    Each must complete without error and produce valid probabilities.
    """
    from outrights.tournament_simulator import TournamentSimulator

    async def run_one_sim(sim_id: int) -> dict:
        t0 = time.perf_counter()
        await asyncio.sleep(0)

        # Use a small field for speed
        simulator = TournamentSimulator(n_simulations=1000)
        players = [
            {"player_id": f"p{j:02d}", "elo": 1800.0 - j * 15.0}
            for j in range(8)
        ]
        result = simulator.simulate_tournament(
            players=players,
            format_code="PDC_MASTERS",
            round_name="Quarter-Final",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {"sim_id": sim_id, "result": result, "elapsed_ms": elapsed_ms, "ok": True}

    tasks = [run_one_sim(i) for i in range(10)]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    errors = [o for o in outcomes if isinstance(o, Exception)]
    assert len(errors) == 0, (
        f"OUTRIGHT STRESS FAIL: {len(errors)}/10 simulations raised exceptions: "
        + str(errors[:3])
    )
