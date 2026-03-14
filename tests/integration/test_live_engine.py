"""
Integration tests for the live engine (Sprint 5).

These tests exercise the full visit processing pipeline, Redis state
persistence (mocked), and multi-visit match sequences.

Integration tests use async fixtures and mock the Redis client so they
can run without a live Redis instance.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from engines.live.live_state_machine import DartsLiveEngine, DartsLiveState


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_state(**kwargs) -> DartsLiveState:
    """Minimal DartsLiveState factory."""
    defaults = dict(
        match_id="integration-match-001",
        score_p1=501,
        score_p2=501,
        current_thrower=0,
        legs_p1=0,
        legs_p2=0,
        sets_p1=0,
        sets_p2=0,
        lwp_current=0.5,
        regime=1,
        double_start=False,
        draw_enabled=False,
        two_clear_legs=False,
        format_code="PDC_WC_QF",
        leg_starter="player-1",
        leg_starter_confidence=0.9,
        is_pressure_state=False,
        current_dart_number=1,
        dartconnect_feed_lag_ms=0,
        last_updated=datetime.now(timezone.utc),
        p1_player_id="player-1",
        p2_player_id="player-2",
        p1_three_da=95.0,
        p2_three_da=92.0,
    )
    defaults.update(kwargs)
    return DartsLiveState(**defaults)


class MockRedis:
    """
    In-memory mock of redis.asyncio.Redis.

    Stores values in a dict and honours EX (TTL) through metadata only
    (TTL expiry is not enforced in tests).
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str):
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None):
        self._store[key] = value

    async def delete(self, key: str) -> int:
        removed = key in self._store
        self._store.pop(key, None)
        return int(removed)


@pytest.fixture
def mock_redis() -> MockRedis:
    return MockRedis()


@pytest.fixture
def engine(mock_redis: MockRedis) -> DartsLiveEngine:
    return DartsLiveEngine(redis_client=mock_redis)


# ---------------------------------------------------------------------------
# test_redis_state_persistence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_redis_state_persistence(
    engine: DartsLiveEngine, mock_redis: MockRedis
) -> None:
    """State saved to Redis should be retrievable by get_state."""
    state = make_state(score_p1=240, score_p2=180)
    await engine.save_state("integration-match-001", state)

    loaded = await engine.get_state("integration-match-001")
    assert loaded is not None
    assert loaded.score_p1 == 240
    assert loaded.score_p2 == 180
    assert loaded.match_id == "integration-match-001"


@pytest.mark.asyncio
async def test_redis_state_missing_returns_none(
    engine: DartsLiveEngine,
) -> None:
    """get_state returns None for an unknown match_id."""
    result = await engine.get_state("nonexistent-match")
    assert result is None


@pytest.mark.asyncio
async def test_redis_state_overwrite(
    engine: DartsLiveEngine, mock_redis: MockRedis
) -> None:
    """Saving state twice with different scores should update correctly."""
    state1 = make_state(score_p1=501)
    await engine.save_state("integration-match-001", state1)

    state2 = make_state(score_p1=300)
    await engine.save_state("integration-match-001", state2)

    loaded = await engine.get_state("integration-match-001")
    assert loaded is not None
    assert loaded.score_p1 == 300


@pytest.mark.asyncio
async def test_redis_state_ttl_key_set(
    engine: DartsLiveEngine, mock_redis: MockRedis
) -> None:
    """State should be stored under the expected Redis key prefix."""
    state = make_state()
    await engine.save_state("my-match", state)

    key = "xg3:live:state:my-match"
    assert key in mock_redis._store, f"Expected Redis key {key!r} not found"


# ---------------------------------------------------------------------------
# test_full_live_match_sequence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_live_match_sequence(engine: DartsLiveEngine) -> None:
    """
    Simulate a short multi-visit sequence and verify state evolution.

    Sequence:
      Visit 1: P1 scores 180 (501→321)
      Visit 2: P2 scores 100 (501→401)
      Visit 3: P1 scores 140 (321→181)
      Visit 4: P2 scores 140 (401→261)
    """
    state = make_state()

    # Visit 1: P1 scores 180
    state = engine._apply_visit_to_state(state, visit_score=180, is_bust=False)
    assert state.score_p1 == 321
    assert state.current_thrower == 1

    # Visit 2: P2 scores 100
    state = engine._apply_visit_to_state(state, visit_score=100, is_bust=False)
    assert state.score_p2 == 401
    assert state.current_thrower == 0

    # Visit 3: P1 scores 140
    state = engine._apply_visit_to_state(state, visit_score=140, is_bust=False)
    assert state.score_p1 == 181
    assert state.current_thrower == 1

    # Visit 4: P2 scores 140
    state = engine._apply_visit_to_state(state, visit_score=140, is_bust=False)
    assert state.score_p2 == 261
    assert state.current_thrower == 0

    # Visit counts should track correctly
    assert state.visit_count_p1 == 2
    assert state.visit_count_p2 == 2


@pytest.mark.asyncio
async def test_full_sequence_with_bust(engine: DartsLiveEngine) -> None:
    """A bust mid-sequence should not change the busting player's score."""
    state = make_state(score_p1=42, score_p2=501, current_thrower=0)

    # P1 at 42, attempts checkout but busts
    state = engine._apply_visit_to_state(state, visit_score=0, is_bust=True)
    assert state.score_p1 == 42, "Bust must not change P1 score"
    assert state.current_thrower == 1

    # P2 visits
    state = engine._apply_visit_to_state(state, visit_score=60, is_bust=False)
    assert state.score_p2 == 441
    assert state.current_thrower == 0


@pytest.mark.asyncio
async def test_leg_transition_resets_scores(engine: DartsLiveEngine) -> None:
    """When a player reaches 0, the next leg should start with scores at 501."""
    state = make_state(score_p1=60, score_p2=200, legs_p1=0, legs_p2=0)

    # P1 checkouts (reduces to 0)
    state = engine._apply_visit_to_state(state, visit_score=60, is_bust=False)
    assert state.score_p1 == 0
    assert state.leg_complete()

    # Handle leg completion
    state = await engine._handle_leg_completion(state)
    assert state.score_p1 == 501
    assert state.score_p2 == 501
    assert state.legs_p1 == 1
    assert state.legs_p2 == 0


@pytest.mark.asyncio
async def test_full_sequence_with_redis_persistence(
    engine: DartsLiveEngine, mock_redis: MockRedis
) -> None:
    """Multi-visit sequence with Redis writes on each leg change."""
    state = make_state()

    # Persist initial state
    await engine.save_state(state.match_id, state)

    # Apply a visit
    state = engine._apply_visit_to_state(state, 140, False)
    await engine.save_state(state.match_id, state)

    # Load from Redis and verify
    loaded = await engine.get_state(state.match_id)
    assert loaded is not None
    assert loaded.score_p1 == 361


@pytest.mark.asyncio
async def test_on_visit_scored_without_prior_state_raises_404(
    engine: DartsLiveEngine,
) -> None:
    """
    on_visit_scored is not called without a state — but the route layer
    returns 404 before calling the engine. Verify get_state returns None
    when no prior state exists.
    """
    result = await engine.get_state("no-such-match")
    assert result is None


@pytest.mark.asyncio
async def test_delete_state_removes_from_redis(
    engine: DartsLiveEngine, mock_redis: MockRedis
) -> None:
    """After delete_state, get_state should return None."""
    state = make_state()
    await engine.save_state(state.match_id, state)

    # Confirm saved
    assert await engine.get_state(state.match_id) is not None

    await engine.delete_state(state.match_id)
    assert await engine.get_state(state.match_id) is None
