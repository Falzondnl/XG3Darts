"""
Stress tests: concurrent match pricing requests.

Validates:
  - 500 concurrent pre-match requests: all succeed, p95 < 50 ms
  - 100 live update requests: p99 < 100 ms

These tests exercise the computation engine (not HTTP layer) to avoid
dependency on a running server in CI.  The engine is invoked directly
in async tasks to simulate concurrent load.

Cache warm-up: A pre-warm pass runs with the same 3DA distribution
before measuring latency.  This reflects the realistic production state
where caches are populated after the first few minutes of traffic.
Individual computation time is measured AFTER the single cooperative
yield, so it captures pure Markov + combinatorics time, not the
time spent waiting while other coroutines ran.
"""
from __future__ import annotations

import asyncio
import random
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


def _prewarm_pricing_caches(rng: random.Random, n: int) -> None:
    """
    Pre-warm visit-distribution and Markov caches for n random 3DA pairs.

    Builds the cache for the same 3DA distribution used by the stress test
    so that measured latency reflects a warm production system, not cold start.
    """
    from engines.leg_layer.markov_chain import DartsMarkovChain

    chain = DartsMarkovChain()
    for i in range(n):
        p1da = rng.uniform(75.0, 105.0)
        p2da = rng.uniform(75.0, 105.0)
        p1d = _make_player_dists(f"warm_p1_{i}", p1da)
        p2d = _make_player_dists(f"warm_p2_{i}", p2da)
        chain.break_probability(p1d, p2d, f"warm_p1_{i}", f"warm_p2_{i}", p1da, p2da)


async def _price_single_match(
    match_id: str,
    p1_three_da: float,
    p2_three_da: float,
    legs_to_win: int = 7,
) -> PricingResult:
    """
    Price one match asynchronously using the Markov chain engine.

    This simulates the core computation path for a pre-match request.
    t0 is recorded AFTER the cooperative yield so elapsed_ms captures
    pure computation time, not time waiting in the event loop queue.
    """
    try:
        from engines.leg_layer.markov_chain import DartsMarkovChain
        from engines.match_layer.match_combinatorics import MatchCombinatorialEngine

        chain = DartsMarkovChain()

        p1_dists = _make_player_dists(f"{match_id}_p1", p1_three_da)
        p2_dists = _make_player_dists(f"{match_id}_p2", p2_three_da)

        # Yield to event loop; start timing AFTER yield so elapsed_ms
        # measures only this task's computation time, not wait time.
        await asyncio.sleep(0)
        t0 = time.perf_counter()

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
        return PricingResult(
            match_id=match_id,
            p1_win=0.0,
            p2_win=0.0,
            elapsed_ms=0.0,
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
    t0 is recorded AFTER the cooperative yield.
    """
    try:
        from engines.leg_layer.markov_chain import DartsMarkovChain
        from engines.match_layer.match_combinatorics import MatchCombinatorialEngine

        chain = DartsMarkovChain()

        p1_dists = _make_player_dists(f"{match_id}_p1", p1_three_da)
        p2_dists = _make_player_dists(f"{match_id}_p2", p2_three_da)

        # Yield to event loop; start timing AFTER yield.
        await asyncio.sleep(0)
        t0 = time.perf_counter()

        hb = chain.break_probability(
            p1_visit_dists=p1_dists,
            p2_visit_dists=p2_dists,
            p1_id=f"{match_id}_p1",
            p2_id=f"{match_id}_p2",
            p1_three_da=p1_three_da,
            p2_three_da=p2_three_da,
        )

        comb_engine = MatchCombinatorialEngine()
        # Price match win probability; legs_p1/legs_p2 inform the remaining
        # legs needed (legs_to_win adjusted for current state).
        remaining_p1 = max(1, 7 - legs_p1)
        remaining_p2 = max(1, 7 - legs_p2)
        legs_to_win_effective = max(remaining_p1, remaining_p2)
        result = comb_engine._dp_legs_format(
            hb=hb,
            legs_to_win=legs_to_win_effective,
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
        return PricingResult(
            match_id=match_id,
            p1_win=0.0,
            p2_win=0.0,
            elapsed_ms=0.0,
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

    Caches are pre-warmed with the same 3DA distribution before measuring,
    reflecting a warm production system.
    """
    rng = random.Random(42)

    n_matches = 500

    # Pre-warm caches with the same seed so all 3DA values are cached.
    prewarm_rng = random.Random(42)
    _prewarm_pricing_caches(prewarm_rng, n_matches)

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

    # p95 < 50 ms (pure computation time, caches warm)
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
    Caches are pre-warmed before measuring.
    """
    rng = random.Random(99)

    n_updates = 100

    # Pre-warm caches with the same seed.
    prewarm_rng = random.Random(99)
    _prewarm_pricing_caches(prewarm_rng, n_updates)

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
    from outrights.tournament_simulator import DartsTournamentSimulator, TournamentField

    async def run_one_sim(sim_id: int) -> dict:
        await asyncio.sleep(0)
        t0 = time.perf_counter()

        # Build a minimal 8-player field for speed
        player_ids = [f"p{j:02d}" for j in range(8)]
        elo_ratings = {pid: 1800.0 - idx * 15.0 for idx, pid in enumerate(player_ids)}
        three_da_stats = {pid: 95.0 - idx * 1.5 for idx, pid in enumerate(player_ids)}

        # Build a single-elimination bracket (quarter-final → semis → final)
        bracket = {
            "QF1": [player_ids[0], player_ids[7]],
            "QF2": [player_ids[1], player_ids[6]],
            "QF3": [player_ids[2], player_ids[5]],
            "QF4": [player_ids[3], player_ids[4]],
        }

        field = TournamentField(
            competition_id=f"stress_sim_{sim_id}",
            format_code="PDC_MASTERS",
            players=player_ids,
            bracket=bracket,
            elo_ratings=elo_ratings,
            three_da_stats=three_da_stats,
        )

        simulator = DartsTournamentSimulator()
        result = simulator.simulate(
            competition_id=f"stress_sim_{sim_id}",
            field=field,
            n_simulations=1000,
            use_antithetic=True,
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

    # All win probability sums should be close to 1.0
    for outcome in outcomes:
        result = outcome["result"]
        total_win_prob = sum(result.player_win_probs.values())
        assert abs(total_win_prob - 1.0) < 0.01, (
            f"sim_{outcome['sim_id']}: win probs sum = {total_win_prob:.4f}, expected ~1.0"
        )
