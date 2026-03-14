"""
Unit tests for the live engine state machine (Sprint 5).

Tests cover:
- visit score application (scoring and bust)
- leg completion detection
- pressure state detection
- staleness check
- 180 Kalman inflation path
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engines.live.live_state_machine import DartsLiveEngine, DartsLiveState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(
    score_p1: int = 501,
    score_p2: int = 501,
    current_thrower: int = 0,
    legs_p1: int = 0,
    legs_p2: int = 0,
    sets_p1: int = 0,
    sets_p2: int = 0,
    dartconnect_feed_lag_ms: int = 0,
    last_updated: datetime | None = None,
) -> DartsLiveState:
    """Factory for DartsLiveState test fixtures."""
    if last_updated is None:
        last_updated = datetime.now(timezone.utc)
    return DartsLiveState(
        match_id="test-match-001",
        score_p1=score_p1,
        score_p2=score_p2,
        current_thrower=current_thrower,
        legs_p1=legs_p1,
        legs_p2=legs_p2,
        sets_p1=sets_p1,
        sets_p2=sets_p2,
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
        dartconnect_feed_lag_ms=dartconnect_feed_lag_ms,
        last_updated=last_updated,
        p1_player_id="player-1",
        p2_player_id="player-2",
        p1_three_da=95.0,
        p2_three_da=92.0,
    )


@pytest.fixture
def engine() -> DartsLiveEngine:
    """Return a DartsLiveEngine with no Redis (memory-only mode)."""
    return DartsLiveEngine(redis_client=None)


# ---------------------------------------------------------------------------
# test_visit_update_changes_score
# ---------------------------------------------------------------------------

def test_visit_update_changes_score(engine: DartsLiveEngine) -> None:
    """Applying a valid visit should reduce the current thrower's score."""
    state = make_state(score_p1=501, score_p2=501, current_thrower=0)
    new_state = engine._apply_visit_to_state(state, visit_score=60, is_bust=False)

    assert new_state.score_p1 == 441, "P1 score should decrease by 60"
    assert new_state.score_p2 == 501, "P2 score should be unchanged"
    assert new_state.current_thrower == 1, "Thrower should alternate to P2"
    assert new_state.visit_count_p1 == 1


def test_visit_update_p2_changes_score(engine: DartsLiveEngine) -> None:
    """Applying a visit for P2 should reduce P2's score only."""
    state = make_state(score_p1=200, score_p2=350, current_thrower=1)
    new_state = engine._apply_visit_to_state(state, visit_score=100, is_bust=False)

    assert new_state.score_p2 == 250
    assert new_state.score_p1 == 200
    assert new_state.current_thrower == 0


# ---------------------------------------------------------------------------
# test_bust_keeps_score_same
# ---------------------------------------------------------------------------

def test_bust_keeps_score_same(engine: DartsLiveEngine) -> None:
    """A busted visit must not change the thrower's score."""
    state = make_state(score_p1=32, score_p2=240, current_thrower=0)
    new_state = engine._apply_visit_to_state(state, visit_score=0, is_bust=True)

    assert new_state.score_p1 == 32, "Bust must not change P1 score"
    assert new_state.score_p2 == 240
    # Thrower still alternates after a bust visit
    assert new_state.current_thrower == 1


def test_bust_with_nonzero_score_ignored(engine: DartsLiveEngine) -> None:
    """If is_bust=True, visit_score payload is ignored."""
    state = make_state(score_p1=100, score_p2=100, current_thrower=0)
    new_state = engine._apply_visit_to_state(state, visit_score=60, is_bust=True)
    # Score unchanged despite non-zero visit_score
    assert new_state.score_p1 == 100


# ---------------------------------------------------------------------------
# test_leg_complete_on_zero
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_leg_complete_on_zero(engine: DartsLiveEngine) -> None:
    """When P1 reduces score to 0 the leg_complete flag must be True."""
    state = make_state(score_p1=60, score_p2=200, current_thrower=0)
    new_state = engine._apply_visit_to_state(state, visit_score=60, is_bust=False)

    assert new_state.score_p1 == 0
    assert new_state.leg_complete() is True


@pytest.mark.asyncio
async def test_leg_completion_increments_legs(engine: DartsLiveEngine) -> None:
    """After leg completion the winner's leg count increments."""
    # P1 just hit 0; simulate via direct _apply then _handle_leg_completion
    state = make_state(score_p1=0, score_p2=120, legs_p1=0, legs_p2=0)
    new_state = await engine._handle_leg_completion(state)

    assert new_state.legs_p1 == 1
    assert new_state.legs_p2 == 0
    # Scores reset to 501 for the new leg
    assert new_state.score_p1 == 501
    assert new_state.score_p2 == 501
    # Visit counts reset
    assert new_state.visit_count_p1 == 0
    assert new_state.visit_count_p2 == 0


@pytest.mark.asyncio
async def test_leg_completion_p2_winner(engine: DartsLiveEngine) -> None:
    """P2 winning the leg increments P2's leg count."""
    state = make_state(score_p1=120, score_p2=0, legs_p1=0, legs_p2=0)
    new_state = await engine._handle_leg_completion(state)

    assert new_state.legs_p2 == 1
    assert new_state.legs_p1 == 0


# ---------------------------------------------------------------------------
# test_pressure_state_detection
# ---------------------------------------------------------------------------

def test_pressure_state_detected_low_score(engine: DartsLiveEngine) -> None:
    """Scores ≤ 40 should trigger is_pressure_state=True."""
    state = make_state(score_p1=32, score_p2=400, current_thrower=0)
    updated = engine._update_pressure_flag(state)
    assert updated.is_pressure_state is True


def test_pressure_state_detected_finish_range(engine: DartsLiveEngine) -> None:
    """Scores 41-99 (FINISH context) should also set is_pressure_state=True."""
    state = make_state(score_p1=72, score_p2=400, current_thrower=0)
    updated = engine._update_pressure_flag(state)
    assert updated.is_pressure_state is True


def test_pressure_state_off_high_score(engine: DartsLiveEngine) -> None:
    """Scores > 300 (OPEN context) should have is_pressure_state=False."""
    state = make_state(score_p1=450, score_p2=300, current_thrower=0)
    updated = engine._update_pressure_flag(state)
    assert updated.is_pressure_state is False


def test_pressure_state_p2_thrower(engine: DartsLiveEngine) -> None:
    """Pressure flag is computed from current thrower's score, not P1 always."""
    # P2 is current thrower with high score; P1 has low score (irrelevant)
    state = make_state(score_p1=12, score_p2=501, current_thrower=1)
    updated = engine._update_pressure_flag(state)
    # P2 is at 501 → OPEN context → no pressure
    assert updated.is_pressure_state is False


# ---------------------------------------------------------------------------
# test_live_state_staleness
# ---------------------------------------------------------------------------

def test_live_state_not_stale_fresh(engine: DartsLiveEngine) -> None:
    """A freshly created state should not be stale."""
    state = make_state(last_updated=datetime.now(timezone.utc))
    assert state.is_stale(max_age_ms=30_000) is False


def test_live_state_stale_old(engine: DartsLiveEngine) -> None:
    """A state updated 60 seconds ago is stale with 30 s threshold."""
    old_time = datetime.now(timezone.utc) - timedelta(seconds=60)
    state = make_state(last_updated=old_time)
    assert state.is_stale(max_age_ms=30_000) is True


def test_live_state_stale_custom_threshold(engine: DartsLiveEngine) -> None:
    """Custom threshold: 10 s old state not stale with 20 s threshold."""
    slightly_old = datetime.now(timezone.utc) - timedelta(seconds=10)
    state = make_state(last_updated=slightly_old)
    assert state.is_stale(max_age_ms=20_000) is False
    assert state.is_stale(max_age_ms=5_000) is True


def test_live_state_stale_naive_datetime() -> None:
    """A naive (timezone-unaware) last_updated datetime is handled gracefully."""
    naive_time = datetime.utcnow() - timedelta(seconds=60)  # naive UTC
    state = make_state(last_updated=naive_time)
    # Should not raise; staleness should be detected
    result = state.is_stale(max_age_ms=30_000)
    assert result is True


# ---------------------------------------------------------------------------
# Redis persistence (memory-only tests)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_state_returns_none_without_redis() -> None:
    """get_state returns None when Redis is not configured."""
    engine = DartsLiveEngine(redis_client=None)
    result = await engine.get_state("some-match-id")
    assert result is None


@pytest.mark.asyncio
async def test_save_state_noop_without_redis() -> None:
    """save_state is a no-op when Redis is not configured."""
    engine = DartsLiveEngine(redis_client=None)
    state = make_state()
    # Should not raise
    await engine.save_state("some-match-id", state)


# ---------------------------------------------------------------------------
# State serialisation round-trip
# ---------------------------------------------------------------------------

def test_state_redis_roundtrip() -> None:
    """State serialised to dict and deserialised should match."""
    state = make_state(score_p1=241, score_p2=180, legs_p1=2, legs_p2=1)
    d = state.to_redis_dict()
    restored = DartsLiveState.from_redis_dict(d)

    assert restored.match_id == state.match_id
    assert restored.score_p1 == state.score_p1
    assert restored.score_p2 == state.score_p2
    assert restored.legs_p1 == state.legs_p1
    assert restored.legs_p2 == state.legs_p2
    assert restored.p1_three_da == state.p1_three_da
    assert restored.format_code == state.format_code
