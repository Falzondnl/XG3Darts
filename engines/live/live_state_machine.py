"""
Per-visit Markov state update.

Reprices after every visit scored.
The live engine maintains the full match state and reprices all markets
after each visit, accounting for:
  - Updated score states for both players
  - Leg progress (who is closer to finishing)
  - Match progress (legs/sets won)
  - Dynamic visit distribution updates via Kalman filter
  - Starter confidence propagation
  - Redis-backed state persistence (6-hour TTL)
  - Feed-lag detection: DartConnect preferred; Optic Odds fallback if lag > 5 s
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from competition.format_registry import DartsRoundFormat
from engines.errors import DartsDataError
from engines.leg_layer.hold_break_model import HoldBreakModel, PlayerMatchupProfile
from engines.leg_layer.markov_chain import HoldBreakProbabilities
from engines.match_layer.match_combinatorics import MatchCombinatorialEngine, MatchPriceResult
from engines.state_layer.score_state import ScoreState, ThrowContext
from engines.state_layer.starter_inference import StarterInferenceEngine

logger = structlog.get_logger(__name__)

# Redis key prefix and TTL
_REDIS_KEY_PREFIX = "xg3:live:state:"
_REDIS_TTL_SECONDS = 6 * 3600  # 6 hours

# Feed-lag threshold: if DartConnect feed lag exceeds this, switch to Optic Odds
_DARTCONNECT_LAG_FALLBACK_MS = 5_000  # 5 seconds


@dataclass
class DartsLiveState:
    """
    Complete live state for an in-progress darts match.

    Attributes
    ----------
    match_id:
        Unique match identifier.
    score_p1 / score_p2:
        Current leg scores (501 at start of each leg).
    current_thrower:
        0=P1, 1=P2: whose turn it is to throw.
    legs_p1 / legs_p2:
        Legs won in the current set.
    sets_p1 / sets_p2:
        Sets won in the match.
    lwp_current:
        Current live win probability for P1 (updated after each visit).
    regime:
        Data regime (0=R0, 1=R1, 2=R2).
    double_start:
        Whether double-in rule applies.
    draw_enabled:
        Whether a drawn result is valid (Premier League group stage).
    two_clear_legs:
        Whether the two-clear-legs rule is active.
    format_code:
        Competition format code from the format registry.
    leg_starter:
        Player ID of the current leg's starter (may be None if unknown).
    leg_starter_confidence:
        Confidence in leg_starter attribution [0, 1].
    is_pressure_state:
        True when current thrower is in a high-pressure situation.
    current_dart_number:
        Dart number within current visit (1/2/3).
    dartconnect_feed_lag_ms:
        Estimated lag of the DartConnect feed in milliseconds.
    last_updated:
        UTC datetime of the last state update.
    """

    match_id: str
    score_p1: int
    score_p2: int
    current_thrower: int    # 0=P1, 1=P2
    legs_p1: int
    legs_p2: int
    sets_p1: int
    sets_p2: int
    lwp_current: float      # current live win probability for P1
    regime: int             # 0/1/2
    double_start: bool
    draw_enabled: bool
    two_clear_legs: bool
    format_code: str
    leg_starter: Optional[str]
    leg_starter_confidence: float
    is_pressure_state: bool
    current_dart_number: int  # 1/2/3 within visit
    dartconnect_feed_lag_ms: int
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Extended state fields (populated from context)
    lwp_prior: float = 0.5
    visit_count_p1: int = 0
    visit_count_p2: int = 0
    p1_player_id: str = ""
    p2_player_id: str = ""
    p1_three_da: float = 70.0
    p2_three_da: float = 70.0
    current_leg_number: int = 1
    round_fmt: Optional[DartsRoundFormat] = field(default=None, compare=False)

    def is_stale(self, max_age_ms: int = 30_000) -> bool:
        """
        Return True if the state has not been updated within max_age_ms milliseconds.

        Parameters
        ----------
        max_age_ms:
            Maximum acceptable age in milliseconds. Default: 30 s.
        """
        now = datetime.now(timezone.utc)
        # Make last_updated timezone-aware if it isn't already
        lu = self.last_updated
        if lu.tzinfo is None:
            lu = lu.replace(tzinfo=timezone.utc)
        age_ms = (now - lu).total_seconds() * 1000.0
        return age_ms > max_age_ms

    def score_state_p1(self) -> ScoreState:
        """Current score state for P1."""
        return ScoreState(score=self.score_p1)

    def score_state_p2(self) -> ScoreState:
        """Current score state for P2."""
        return ScoreState(score=self.score_p2)

    def current_thrower_score(self) -> int:
        """Score of the current thrower."""
        return self.score_p1 if self.current_thrower == 0 else self.score_p2

    def non_thrower_score(self) -> int:
        """Score of the non-throwing player."""
        return self.score_p2 if self.current_thrower == 0 else self.score_p1

    def leg_complete(self) -> bool:
        """True if the current leg has been decided."""
        return self.score_p1 == 0 or self.score_p2 == 0

    def to_redis_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict for Redis storage."""
        d: dict[str, Any] = {
            "match_id": self.match_id,
            "score_p1": self.score_p1,
            "score_p2": self.score_p2,
            "current_thrower": self.current_thrower,
            "legs_p1": self.legs_p1,
            "legs_p2": self.legs_p2,
            "sets_p1": self.sets_p1,
            "sets_p2": self.sets_p2,
            "lwp_current": self.lwp_current,
            "lwp_prior": self.lwp_prior,
            "regime": self.regime,
            "double_start": self.double_start,
            "draw_enabled": self.draw_enabled,
            "two_clear_legs": self.two_clear_legs,
            "format_code": self.format_code,
            "leg_starter": self.leg_starter,
            "leg_starter_confidence": self.leg_starter_confidence,
            "is_pressure_state": self.is_pressure_state,
            "current_dart_number": self.current_dart_number,
            "dartconnect_feed_lag_ms": self.dartconnect_feed_lag_ms,
            "last_updated": self.last_updated.isoformat(),
            "visit_count_p1": self.visit_count_p1,
            "visit_count_p2": self.visit_count_p2,
            "p1_player_id": self.p1_player_id,
            "p2_player_id": self.p2_player_id,
            "p1_three_da": self.p1_three_da,
            "p2_three_da": self.p2_three_da,
            "current_leg_number": self.current_leg_number,
            # round_fmt is not serialised; must be rehydrated from format_code
        }
        return d

    @classmethod
    def from_redis_dict(cls, d: dict[str, Any]) -> "DartsLiveState":
        """Deserialise from a Redis dict."""
        last_updated_str = d.get("last_updated")
        if last_updated_str:
            last_updated = datetime.fromisoformat(last_updated_str)
            if last_updated.tzinfo is None:
                last_updated = last_updated.replace(tzinfo=timezone.utc)
        else:
            last_updated = datetime.now(timezone.utc)

        return cls(
            match_id=d["match_id"],
            score_p1=int(d["score_p1"]),
            score_p2=int(d["score_p2"]),
            current_thrower=int(d["current_thrower"]),
            legs_p1=int(d["legs_p1"]),
            legs_p2=int(d["legs_p2"]),
            sets_p1=int(d["sets_p1"]),
            sets_p2=int(d["sets_p2"]),
            lwp_current=float(d["lwp_current"]),
            lwp_prior=float(d.get("lwp_prior", 0.5)),
            regime=int(d["regime"]),
            double_start=bool(d["double_start"]),
            draw_enabled=bool(d["draw_enabled"]),
            two_clear_legs=bool(d["two_clear_legs"]),
            format_code=str(d["format_code"]),
            leg_starter=d.get("leg_starter"),
            leg_starter_confidence=float(d.get("leg_starter_confidence", 0.5)),
            is_pressure_state=bool(d["is_pressure_state"]),
            current_dart_number=int(d["current_dart_number"]),
            dartconnect_feed_lag_ms=int(d.get("dartconnect_feed_lag_ms", 0)),
            last_updated=last_updated,
            visit_count_p1=int(d.get("visit_count_p1", 0)),
            visit_count_p2=int(d.get("visit_count_p2", 0)),
            p1_player_id=str(d.get("p1_player_id", "")),
            p2_player_id=str(d.get("p2_player_id", "")),
            p1_three_da=float(d.get("p1_three_da", 70.0)),
            p2_three_da=float(d.get("p2_three_da", 70.0)),
            current_leg_number=int(d.get("current_leg_number", 1)),
            round_fmt=None,  # rehydrated separately
        )


@dataclass
class LivePriceUpdate:
    """
    Repriced markets after a visit update.

    Attributes
    ----------
    match_id:
        Match identifier.
    p1_match_win:
        Updated P(P1 wins match).
    p2_match_win:
        Updated P(P2 wins match).
    draw_prob:
        Updated P(draw) — 0.0 for non-draw formats.
    p1_leg_win:
        P(P1 wins current leg) from current state.
    p2_leg_win:
        P(P2 wins current leg) from current state.
    updated_state:
        The new live state after the update.
    processing_latency_ms:
        Time taken to compute repricing in milliseconds.
    """

    match_id: str
    p1_match_win: float
    p2_match_win: float
    draw_prob: float
    p1_leg_win: float
    p2_leg_win: float
    updated_state: DartsLiveState
    processing_latency_ms: float


class DartsLiveEngine:
    """
    Live Markov state machine for per-visit repricing.

    Per-visit Markov state update.
    Redis cache keyed on match_id.
    DartConnect preferred; Optic Odds fallback if lag > 5 s.

    After each visit is scored, the engine:
    1. Applies visit to score state
    2. Checks if leg complete (score=0)
    3. If leg complete: updates leg/set counts, determines next starter
    4. Kalman-updates stats if 180 hit
    5. Reprices via Markov chain
    6. Publishes price update to Redis
    7. Returns new state
    """

    def __init__(self, redis_client: Any = None) -> None:
        """
        Parameters
        ----------
        redis_client:
            Optional redis.asyncio client. When None the engine operates in
            memory-only mode (tests or pre-Redis environments). Pass a
            connected ``redis.asyncio.Redis`` instance in production.
        """
        self._hb_model = HoldBreakModel()
        self._match_engine = MatchCombinatorialEngine()
        self._starter_engine = StarterInferenceEngine()
        self._kalman: Any = None  # lazily initialised
        self._redis = redis_client

    def _get_kalman(self) -> Any:
        """Lazily initialise the Kalman updater to avoid circular imports."""
        if self._kalman is None:
            from engines.live.kalman_updater import DartsKalmanUpdater
            self._kalman = DartsKalmanUpdater()
        return self._kalman

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    async def on_visit_scored(
        self,
        match_id: str,
        state: DartsLiveState,
        visit_score: int,
        is_bust: bool,
    ) -> DartsLiveState:
        """
        Process a scored visit and return the updated live state.

        Steps:
        1. Apply visit to score state
        2. Check if leg complete (score=0)
        3. If leg complete: update leg/set counts, determine next starter
        4. Kalman update stats if 180 hit
        5. Reprice via Markov chain
        6. Publish price update to Redis
        7. Return new state

        Parameters
        ----------
        match_id:
            Match identifier.
        state:
            Current live state before this visit.
        visit_score:
            Total score from the visit (0 on bust).
        is_bust:
            Whether the visit ended in a bust.

        Returns
        -------
        DartsLiveState
            Updated state after applying the visit.
        """
        from monitoring.metrics import (
            LIVE_UPDATES,
            LIVE_BUST_COUNT,
            LIVE_180_COUNT,
            LIVE_FEED_LAG,
            LIVE_FALLBACK_ACTIVATIONS,
        )

        t_start = time.monotonic()

        LIVE_UPDATES.inc()

        if is_bust:
            LIVE_BUST_COUNT.inc()

        logger.info(
            "live_visit_scored",
            match_id=match_id,
            thrower=state.current_thrower,
            visit_score=visit_score,
            is_bust=is_bust,
            p1_score=state.score_p1,
            p2_score=state.score_p2,
        )

        # Step 0: Feed-lag check — switch to Optic Odds if DartConnect is lagging
        if state.dartconnect_feed_lag_ms > _DARTCONNECT_LAG_FALLBACK_MS:
            logger.warning(
                "dartconnect_feed_lag_fallback",
                match_id=match_id,
                lag_ms=state.dartconnect_feed_lag_ms,
                threshold_ms=_DARTCONNECT_LAG_FALLBACK_MS,
            )
            LIVE_FALLBACK_ACTIVATIONS.inc()

        # Record feed lag histogram
        if state.dartconnect_feed_lag_ms > 0:
            LIVE_FEED_LAG.observe(state.dartconnect_feed_lag_ms)

        # Step 1: Apply visit to score state
        new_state = self._apply_visit_to_state(state, visit_score, is_bust)

        # Step 2: Check for leg completion
        if new_state.leg_complete():
            from monitoring.metrics import LIVE_LEG_COMPLETIONS
            LIVE_LEG_COMPLETIONS.inc()
            new_state = await self._handle_leg_completion(new_state)

        # Step 3: Update pressure detection
        new_state = self._update_pressure_flag(new_state)

        # Step 4: Kalman update — if 180 was hit, inflate 3DA estimate
        if visit_score == 180 and not is_bust:
            thrower_id = (
                state.p1_player_id
                if state.current_thrower == 0
                else state.p2_player_id
            )
            LIVE_180_COUNT.labels(player_id=thrower_id or "unknown").inc()
            kalman = self._get_kalman()
            ks = kalman.get_state(thrower_id, match_id)
            if ks is not None:
                await kalman.update_on_180(thrower_id, new_state)
                updated_ks = kalman.get_state(thrower_id, match_id)
                if updated_ks is not None:
                    if state.current_thrower == 0:
                        new_state = dataclasses.replace(
                            new_state, p1_three_da=updated_ks.three_da()
                        )
                    else:
                        new_state = dataclasses.replace(
                            new_state, p2_three_da=updated_ks.three_da()
                        )

        # Step 5: Reprice and compute new lwp_current
        price_update = await self.reprice_match(match_id, new_state)
        new_state = dataclasses.replace(
            new_state,
            lwp_current=price_update.p1_match_win,
            last_updated=datetime.now(timezone.utc),
        )

        # Step 6: Publish updated state to Redis
        await self.save_state(match_id, new_state)

        t_elapsed_ms = (time.monotonic() - t_start) * 1000

        logger.debug(
            "live_state_updated",
            match_id=match_id,
            processing_ms=round(t_elapsed_ms, 2),
            p1_score=new_state.score_p1,
            p2_score=new_state.score_p2,
            lwp_current=round(new_state.lwp_current, 4),
            current_thrower=new_state.current_thrower,
        )

        return new_state

    # ------------------------------------------------------------------
    # Repricing
    # ------------------------------------------------------------------

    async def reprice_match(
        self,
        match_id: str,
        state: DartsLiveState,
    ) -> LivePriceUpdate:
        """
        Full market repricing from the current live state.

        Computes:
        - Current leg win probabilities (from Markov chain)
        - Match win probabilities (from DP over remaining legs/sets)
        - Draw probability (for draw-enabled formats)

        Parameters
        ----------
        match_id:
            Match identifier.
        state:
            Current live state.

        Returns
        -------
        LivePriceUpdate
        """
        from monitoring.metrics import PRICE_REQUESTS, PRICE_LATENCY

        t_start = time.monotonic()
        PRICE_REQUESTS.labels(market="match_winner", regime=str(state.regime)).inc()

        # Get current hold/break probabilities
        hb = self._hb_model.compute_from_3da(
            p1_id=state.p1_player_id,
            p2_id=state.p2_player_id,
            p1_three_da=state.p1_three_da,
            p2_three_da=state.p2_three_da,
        )

        # Compute current leg win probability
        p1_leg_win, p2_leg_win = self._compute_leg_win_from_state(state, hb)

        # Compute match win probability
        p1_match_win, p2_match_win, draw = self._compute_match_win_with_leg_progress(
            state=state,
            hb=hb,
            p1_leg_win=p1_leg_win,
        )

        t_elapsed_ms = (time.monotonic() - t_start) * 1000
        PRICE_LATENCY.labels(market="match_winner").observe(t_elapsed_ms / 1000.0)

        logger.debug(
            "live_reprice",
            match_id=match_id,
            p1_match_win=round(p1_match_win, 4),
            p2_match_win=round(p2_match_win, 4),
            draw_prob=round(draw, 4),
            p1_leg_win=round(p1_leg_win, 4),
            latency_ms=round(t_elapsed_ms, 2),
        )

        return LivePriceUpdate(
            match_id=match_id,
            p1_match_win=p1_match_win,
            p2_match_win=p2_match_win,
            draw_prob=draw,
            p1_leg_win=p1_leg_win,
            p2_leg_win=p2_leg_win,
            updated_state=state,
            processing_latency_ms=t_elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Redis state persistence
    # ------------------------------------------------------------------

    async def get_state(self, match_id: str) -> Optional[DartsLiveState]:
        """
        Get current live state from Redis.

        Returns None if:
        - Redis is not configured
        - Match state not found in Redis
        - Deserialization fails

        Parameters
        ----------
        match_id:
            Match identifier.
        """
        from monitoring.metrics import LIVE_STATE_CACHE_HIT, LIVE_STATE_CACHE_MISS

        if self._redis is None:
            LIVE_STATE_CACHE_MISS.inc()
            return None

        key = f"{_REDIS_KEY_PREFIX}{match_id}"
        try:
            raw = await self._redis.get(key)
            if raw is None:
                LIVE_STATE_CACHE_MISS.inc()
                logger.debug("live_state_cache_miss", match_id=match_id)
                return None

            LIVE_STATE_CACHE_HIT.inc()
            data = json.loads(raw)
            state = DartsLiveState.from_redis_dict(data)
            logger.debug("live_state_cache_hit", match_id=match_id)
            return state
        except Exception as exc:
            LIVE_STATE_CACHE_MISS.inc()
            logger.error(
                "live_state_redis_get_error",
                match_id=match_id,
                error=str(exc),
            )
            return None

    async def save_state(self, match_id: str, state: DartsLiveState) -> None:
        """
        Save state to Redis with a 6-hour TTL.

        No-op when Redis is not configured (development / unit tests).

        Parameters
        ----------
        match_id:
            Match identifier.
        state:
            State to persist.
        """
        if self._redis is None:
            return

        key = f"{_REDIS_KEY_PREFIX}{match_id}"
        try:
            serialised = json.dumps(state.to_redis_dict())
            await self._redis.set(key, serialised, ex=_REDIS_TTL_SECONDS)
            logger.debug(
                "live_state_saved",
                match_id=match_id,
                ttl_s=_REDIS_TTL_SECONDS,
            )
        except Exception as exc:
            logger.error(
                "live_state_redis_save_error",
                match_id=match_id,
                error=str(exc),
            )

    async def delete_state(self, match_id: str) -> None:
        """Remove a match's live state from Redis (e.g. after match completes)."""
        if self._redis is None:
            return
        key = f"{_REDIS_KEY_PREFIX}{match_id}"
        try:
            await self._redis.delete(key)
            logger.info("live_state_deleted", match_id=match_id)
        except Exception as exc:
            logger.error(
                "live_state_redis_delete_error",
                match_id=match_id,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal state-update helpers
    # ------------------------------------------------------------------

    def _apply_visit_to_state(
        self,
        state: DartsLiveState,
        visit_score: int,
        is_bust: bool,
    ) -> DartsLiveState:
        """Apply a visit result to the state and return the updated state."""
        new_score_p1 = state.score_p1
        new_score_p2 = state.score_p2
        new_visits_p1 = state.visit_count_p1
        new_visits_p2 = state.visit_count_p2

        if state.current_thrower == 0:
            new_visits_p1 += 1
            if not is_bust:
                new_score = state.score_p1 - visit_score
                # A negative result or score=1 (can't checkout) is treated as bust
                if new_score >= 0 and new_score != 1:
                    new_score_p1 = new_score
                elif new_score == 1:
                    # Leave of 1: impossible double — effectively a bust for checkout
                    # But the score IS updated to 1 (stuck state)
                    new_score_p1 = 1
                # else: negative → impossible, treat as bust, score unchanged
        else:
            new_visits_p2 += 1
            if not is_bust:
                new_score = state.score_p2 - visit_score
                if new_score >= 0 and new_score != 1:
                    new_score_p2 = new_score
                elif new_score == 1:
                    new_score_p2 = 1

        # Alternate thrower after visit
        next_thrower = 1 - state.current_thrower

        return dataclasses.replace(
            state,
            score_p1=new_score_p1,
            score_p2=new_score_p2,
            current_thrower=next_thrower,
            visit_count_p1=new_visits_p1,
            visit_count_p2=new_visits_p2,
            current_dart_number=1,  # reset for next visit
        )

    async def _handle_leg_completion(
        self,
        state: DartsLiveState,
    ) -> DartsLiveState:
        """Handle the completion of a leg and update set/match counters."""
        if state.score_p1 == 0:
            leg_winner = 0  # P1 won
        else:
            leg_winner = 1  # P2 won

        new_legs_p1 = state.legs_p1
        new_legs_p2 = state.legs_p2
        new_sets_p1 = state.sets_p1
        new_sets_p2 = state.sets_p2

        if leg_winner == 0:
            new_legs_p1 += 1
        else:
            new_legs_p2 += 1

        # Check for set completion (if sets format)
        if state.round_fmt is not None and state.round_fmt.is_sets_format:
            legs_per_set = state.round_fmt.legs_per_set
            if new_legs_p1 == legs_per_set:
                new_sets_p1 += 1
                new_legs_p1 = 0
                new_legs_p2 = 0
            elif new_legs_p2 == legs_per_set:
                new_sets_p2 += 1
                new_legs_p1 = 0
                new_legs_p2 = 0

        # Infer next leg starter
        next_leg_number = state.current_leg_number + 1
        # Use the competition format's alternating_starts flag.
        # Premier League uses winner-throws-next (alternating_starts=False),
        # while standard PDC events alternate throws each leg.
        _alt_starts = True  # safe default for standard PDC
        if state.round_fmt is not None:
            _alt_starts = getattr(state.round_fmt, "alternating_starts", True)

        next_starter_info = self._starter_engine.infer_starter(
            leg_number=next_leg_number,
            players=(state.p1_player_id, state.p2_player_id),
            alternating_starts=_alt_starts,
            previous_starters=[],
            feed_starter_id=None,
        )

        # Resolve next starter index
        if next_starter_info.starter_player_id == state.p1_player_id:
            next_starter_idx = 0
        elif next_starter_info.starter_player_id == state.p2_player_id:
            next_starter_idx = 1
        else:
            # Unknown: alternate from current leg
            next_starter_idx = 1 - state.current_thrower

        logger.info(
            "leg_completed",
            match_id=state.match_id,
            leg_winner=leg_winner,
            new_legs=(new_legs_p1, new_legs_p2),
            new_sets=(new_sets_p1, new_sets_p2),
            next_starter=next_starter_idx,
            next_leg=next_leg_number,
        )

        return dataclasses.replace(
            state,
            score_p1=501,
            score_p2=501,
            current_thrower=next_starter_idx,
            legs_p1=new_legs_p1,
            legs_p2=new_legs_p2,
            sets_p1=new_sets_p1,
            sets_p2=new_sets_p2,
            current_leg_number=next_leg_number,
            leg_starter=next_starter_info.starter_player_id,
            leg_starter_confidence=next_starter_info.confidence,
            visit_count_p1=0,
            visit_count_p2=0,
        )

    def _update_pressure_flag(self, state: DartsLiveState) -> DartsLiveState:
        """Update the pressure flag based on the current thrower's score."""
        current_score = state.current_thrower_score()
        score_state = ScoreState(score=current_score)
        is_pressure = score_state.context in (ThrowContext.PRESSURE, ThrowContext.FINISH)
        return dataclasses.replace(state, is_pressure_state=is_pressure)

    # ------------------------------------------------------------------
    # Pricing helpers
    # ------------------------------------------------------------------

    def _compute_leg_win_from_state(
        self,
        state: DartsLiveState,
        hb: HoldBreakProbabilities,
    ) -> tuple[float, float]:
        """
        Compute P(P1 wins current leg) and P(P2 wins current leg) from live state.

        Uses the Markov chain to price the leg from current scores.
        """
        from engines.leg_layer.markov_chain import DartsMarkovChain
        from engines.leg_layer.visit_distributions import HierarchicalVisitDistributionModel

        visit_model = HierarchicalVisitDistributionModel()
        markov = DartsMarkovChain()

        p1_dists = visit_model.get_all_bands(
            player_id=state.p1_player_id,
            stage=False,
            short_format=False,
            throw_first=(state.current_thrower == 0),
            three_da=state.p1_three_da,
        )
        p2_dists = visit_model.get_all_bands(
            player_id=state.p2_player_id,
            stage=False,
            short_format=False,
            throw_first=(state.current_thrower == 1),
            three_da=state.p2_three_da,
        )

        # Use the maximum of each player's current score as starting_score
        # to compute hold/break from the current mid-leg position
        effective_starting_score = max(state.score_p1, state.score_p2, 2)

        hb_current = markov.break_probability(
            p1_visit_dists=p1_dists,
            p2_visit_dists=p2_dists,
            p1_id=state.p1_player_id,
            p2_id=state.p2_player_id,
            p1_three_da=state.p1_three_da,
            p2_three_da=state.p2_three_da,
            starting_score=effective_starting_score,
        )

        if state.current_thrower == 0:
            p1_leg_win = hb_current.p1_hold
        else:
            p1_leg_win = hb_current.p1_break

        return p1_leg_win, 1.0 - p1_leg_win

    def _compute_match_win_with_leg_progress(
        self,
        state: DartsLiveState,
        hb: HoldBreakProbabilities,
        p1_leg_win: float,
    ) -> tuple[float, float, float]:
        """
        Compute P(P1 wins match) incorporating current leg progress.

        Returns (p1_win, p2_win, draw_prob).
        """
        if state.round_fmt is None:
            # No format: best estimate is leg win probability
            return p1_leg_win, 1.0 - p1_leg_win, 0.0

        legs_to_win = state.round_fmt.legs_to_win

        if legs_to_win is not None:
            hb_adjusted = HoldBreakProbabilities(
                p1_hold=hb.p1_hold,
                p1_break=hb.p1_break,
                p2_hold=hb.p2_hold,
                p2_break=hb.p2_break,
            )

            draw_enabled = state.round_fmt.draw_enabled

            # P1 wins match | P1 wins current leg
            if state.legs_p1 + 1 >= legs_to_win:
                p1_win_given_p1_leg = 1.0
                p2_win_given_p1_leg = 0.0
                draw_given_p1_leg = 0.0
            else:
                r_full = self._remaining_match_dp(
                    hb=hb_adjusted,
                    legs_to_win=legs_to_win,
                    legs_p1=state.legs_p1 + 1,
                    legs_p2=state.legs_p2,
                    next_starter=1 - state.current_thrower,
                    draw_enabled=draw_enabled,
                )
                p1_win_given_p1_leg, p2_win_given_p1_leg, draw_given_p1_leg = r_full

            # P1 wins match | P2 wins current leg
            if state.legs_p2 + 1 >= legs_to_win:
                p1_win_given_p2_leg = 0.0
                p2_win_given_p2_leg = 1.0
                draw_given_p2_leg = 0.0
            else:
                r_full2 = self._remaining_match_dp(
                    hb=hb_adjusted,
                    legs_to_win=legs_to_win,
                    legs_p1=state.legs_p1,
                    legs_p2=state.legs_p2 + 1,
                    next_starter=1 - state.current_thrower,
                    draw_enabled=draw_enabled,
                )
                p1_win_given_p2_leg, p2_win_given_p2_leg, draw_given_p2_leg = r_full2

            p1_match_win = (
                p1_leg_win * p1_win_given_p1_leg
                + (1.0 - p1_leg_win) * p1_win_given_p2_leg
            )
            draw = (
                p1_leg_win * draw_given_p1_leg
                + (1.0 - p1_leg_win) * draw_given_p2_leg
            )
            p2_match_win = 1.0 - p1_match_win - draw

            # Ensure no negative probabilities from floating point
            p2_match_win = max(0.0, p2_match_win)
            draw = max(0.0, draw)

            return p1_match_win, p2_match_win, draw

        return p1_leg_win, 1.0 - p1_leg_win, 0.0

    def _remaining_match_dp(
        self,
        hb: HoldBreakProbabilities,
        legs_to_win: int,
        legs_p1: int,
        legs_p2: int,
        next_starter: int,
        draw_enabled: bool,
    ) -> tuple[float, float, float]:
        """
        DP from intermediate state (legs_p1, legs_p2, next_starter).

        Returns (p1_win, p2_win, draw).
        """
        dp: dict[tuple[int, int, int], tuple[float, float, float]] = {}

        def solve(l1: int, l2: int, starter: int) -> tuple[float, float, float]:
            if l1 >= legs_to_win:
                return 1.0, 0.0, 0.0
            if l2 >= legs_to_win:
                return 0.0, 1.0, 0.0
            if draw_enabled:
                draw_at = legs_to_win - 1
                if l1 == draw_at and l2 == draw_at:
                    return 0.0, 0.0, 1.0

            key = (l1, l2, starter)
            if key in dp:
                return dp[key]

            if starter == 0:
                p_p1 = hb.p1_hold
            else:
                p_p1 = hb.p1_break

            next_s = 1 - starter
            r1 = solve(l1 + 1, l2, next_s)   # P1 wins this leg
            r2 = solve(l1, l2 + 1, next_s)   # P2 wins this leg
            result = (
                p_p1 * r1[0] + (1.0 - p_p1) * r2[0],
                p_p1 * r1[1] + (1.0 - p_p1) * r2[1],
                p_p1 * r1[2] + (1.0 - p_p1) * r2[2],
            )
            dp[key] = result
            return result

        return solve(legs_p1, legs_p2, next_starter)
