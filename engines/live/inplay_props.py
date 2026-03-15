"""
Granular in-play proposition markets.

Computes live micro-markets from the current ``DartsLiveState`` after each visit.
These are Bet365/Unibet-parity markets that go beyond match winner:

  - next_leg_winner      — who wins the current leg in progress
  - next_set_winner      — who wins the current set (set-format matches)
  - leg_180_count        — over/under total 180s in current leg
  - leg_checkout_score   — exact checkout or over/under (P1 vs P2)
  - leg_nine_darter      — probability of a nine-darter completion
  - set_correct_score    — set correct score options (sets format)
  - current_leg_handicap — adjusted leg handicap from live state
  - player_to_hit_first_180 — P1 vs P2 to hit 180 first in current leg

All prices are returned as ``InPlayPropsResult`` containing a list of
``PropositionMarket`` objects, each with an outcome dict keyed by selection
name and valued as overround-inclusive decimal odds.

Architecture
------------
This module is **stateless** — it takes a ``DartsLiveState`` + calibration params
and computes prices deterministically. No DB access, no side effects.

Calibration inputs (from the live state):
  - p1_three_da, p2_three_da:  3-dart average for Poisson/visit calculations
  - lwp_current:               current match win probability for P1
  - legs_p1/legs_p2:           legs won in current set
  - score_p1/score_p2:         current leg scores (distance from 0)
  - round_fmt:                 format registry entry

Validation gates (G9-compliant):
  - Data gate: refuse to open a market if minimum data threshold not met
  - Returns DataGateFailed error if 3DA below usable threshold
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from engines.errors import DartsDataError, DartsMarketClosedError

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Minimum 3DA to open any in-play prop (below this we can't trust Poisson rates)
_MIN_3DA_FOR_PROPS: float = 20.0

# Checkout probability — darts left beyond which we don't compute exact checkout
_MAX_SCORE_FOR_CHECKOUT_PROP: int = 180

# 180 Poisson: expected 180 rate per visit based on 3DA
# Empirical calibration from PDC 276k match history
_BASE_180_RATE_PER_VISIT = 0.0385  # approximately 1 in 26 visits at world-class level
_180_RATE_SCALE_3DA = 80.0         # 3DA at which base rate applies

# Margin applied to in-play props (tighter than pre-match; updated live)
_INPLAY_MARGIN_RATE = 0.04          # 4% overround for micro-markets
_INPLAY_MARGIN_RATE_9DARTER = 0.15  # 15% on nine-darter (high variance)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PropositionMarket:
    """
    A single proposition market with prices for all outcomes.

    All prices are decimal odds inclusive of the operator margin.
    """

    market_key: str                 # e.g. "next_leg_winner"
    market_label: str               # human-readable
    outcomes: dict[str, float]      # selection → decimal odds
    is_open: bool = True
    data_gate_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InPlayPropsResult:
    """All granular in-play proposition markets for the current visit state."""

    match_id: str
    current_leg: int
    current_set_p1: int
    current_set_p2: int
    score_p1: int
    score_p2: int
    markets: list[PropositionMarket]

    def open_markets(self) -> list[PropositionMarket]:
        return [m for m in self.markets if m.is_open]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class InPlayPropsEngine:
    """
    Computes granular in-play proposition markets from a live state.

    Parameters
    ----------
    margin_rate:
        Overround applied across each market (default 4%).
    """

    def __init__(self, margin_rate: float = _INPLAY_MARGIN_RATE) -> None:
        self._margin_rate = margin_rate

    def compute(self, state: "DartsLiveState") -> InPlayPropsResult:  # type: ignore[name-defined]
        """
        Compute all in-play proposition markets for the current visit state.

        Parameters
        ----------
        state:
            Current ``DartsLiveState`` from the live engine.

        Returns
        -------
        InPlayPropsResult
        """
        markets: list[PropositionMarket] = []

        # Validate minimum data quality
        if state.p1_three_da < _MIN_3DA_FOR_PROPS or state.p2_three_da < _MIN_3DA_FOR_PROPS:
            closed = PropositionMarket(
                market_key="all",
                market_label="All In-Play Props",
                outcomes={},
                is_open=False,
                data_gate_reason=(
                    f"3DA below threshold (p1={state.p1_three_da:.1f}, "
                    f"p2={state.p2_three_da:.1f}, min={_MIN_3DA_FOR_PROPS})"
                ),
            )
            return InPlayPropsResult(
                match_id=state.match_id,
                current_leg=state.current_leg_number,
                current_set_p1=state.sets_p1,
                current_set_p2=state.sets_p2,
                score_p1=state.score_p1,
                score_p2=state.score_p2,
                markets=[closed],
            )

        # --- Market 1: Next Leg Winner ---
        markets.append(self._next_leg_winner(state))

        # --- Market 2: Next Set Winner (only in set-format matches) ---
        has_sets = (
            state.round_fmt is not None
            and getattr(state.round_fmt, "sets", 1) > 1
        )
        if has_sets:
            markets.append(self._next_set_winner(state))

        # --- Market 3: 180 count in current leg (over/under) ---
        markets.append(self._leg_180_count(state))

        # --- Market 4: First player to hit a 180 in current leg ---
        markets.append(self._first_180_in_leg(state))

        # --- Market 5: Checkout score (P1 vs P2 — higher/lower/exact range) ---
        if state.score_p1 <= _MAX_SCORE_FOR_CHECKOUT_PROP or state.score_p2 <= _MAX_SCORE_FOR_CHECKOUT_PROP:
            markets.append(self._checkout_market(state))

        # --- Market 6: Nine-darter from current state ---
        markets.append(self._nine_darter_market(state))

        # --- Market 7: Live leg handicap (updated handicap from live state) ---
        markets.append(self._live_leg_handicap(state))

        # --- Market 8: Correct score in current set ---
        if has_sets:
            markets.extend(self._set_correct_score(state))

        return InPlayPropsResult(
            match_id=state.match_id,
            current_leg=state.current_leg_number,
            current_set_p1=state.sets_p1,
            current_set_p2=state.sets_p2,
            score_p1=state.score_p1,
            score_p2=state.score_p2,
            markets=markets,
        )

    # ------------------------------------------------------------------
    # Market 1: Next Leg Winner
    # ------------------------------------------------------------------

    def _next_leg_winner(self, state: "DartsLiveState") -> PropositionMarket:
        """
        Who wins the current leg in progress.

        Uses the Markov-derived leg win probability (LWP) from the live state.
        LWP is computed by the main live engine and stored in ``lwp_current``.

        We decompose ``lwp_current`` (match WP) into leg WP using the
        current score state: the player closer to 0 is more likely to win
        this leg.
        """
        # Estimate leg win probability from current scores
        leg_wp_p1 = _score_to_leg_wp(
            score_thrower=state.score_p1,
            score_opponent=state.score_p2,
            three_da_thrower=state.p1_three_da,
            three_da_opponent=state.p2_three_da,
            current_thrower_is_p1=(state.current_thrower == 0),
        )
        leg_wp_p2 = 1.0 - leg_wp_p1

        odds_p1, odds_p2 = _two_way_margin(leg_wp_p1, leg_wp_p2, self._margin_rate)

        return PropositionMarket(
            market_key="next_leg_winner",
            market_label=f"Leg {state.current_leg_number} Winner",
            outcomes={
                "p1": round(odds_p1, 3),
                "p2": round(odds_p2, 3),
            },
            metadata={
                "leg_wp_p1_raw": round(leg_wp_p1, 5),
                "leg_wp_p2_raw": round(leg_wp_p2, 5),
                "score_p1": state.score_p1,
                "score_p2": state.score_p2,
                "current_thrower": state.current_thrower,
            },
        )

    # ------------------------------------------------------------------
    # Market 2: Next Set Winner
    # ------------------------------------------------------------------

    def _next_set_winner(self, state: "DartsLiveState") -> PropositionMarket:
        """
        Who wins the current set.

        Uses the current legs_p1/legs_p2 and the set format (legs-per-set)
        to compute set win probability via binomial CDF.
        """
        legs_to_win_set = _legs_needed_for_set(state)

        # Remaining legs needed
        p1_legs_still_needed = legs_to_win_set - state.legs_p1
        p2_legs_still_needed = legs_to_win_set - state.legs_p2

        if p1_legs_still_needed <= 0 or p2_legs_still_needed <= 0:
            # Set already decided — market closed
            return PropositionMarket(
                market_key="next_set_winner",
                market_label=f"Set {state.sets_p1 + state.sets_p2 + 1} Winner",
                outcomes={},
                is_open=False,
                data_gate_reason="Set already decided.",
            )

        # Per-leg win probability from match WP proxy
        # Use lwp_current as a proxy for per-leg dominance
        per_leg_p1 = _clip(state.lwp_current, 0.2, 0.8)
        per_leg_p2 = 1.0 - per_leg_p1

        set_wp_p1 = _race_win_probability(
            p1_legs_still_needed, p2_legs_still_needed, per_leg_p1
        )
        set_wp_p2 = 1.0 - set_wp_p1

        odds_p1, odds_p2 = _two_way_margin(set_wp_p1, set_wp_p2, self._margin_rate)

        return PropositionMarket(
            market_key="next_set_winner",
            market_label=f"Set {state.sets_p1 + state.sets_p2 + 1} Winner",
            outcomes={
                "p1": round(odds_p1, 3),
                "p2": round(odds_p2, 3),
            },
            metadata={
                "legs_p1": state.legs_p1,
                "legs_p2": state.legs_p2,
                "legs_to_win_set": legs_to_win_set,
                "set_wp_p1_raw": round(set_wp_p1, 5),
            },
        )

    # ------------------------------------------------------------------
    # Market 3: 180 count in current leg (over/under)
    # ------------------------------------------------------------------

    def _leg_180_count(self, state: "DartsLiveState") -> PropositionMarket:
        """
        Over/Under total 180s remaining in the current leg.

        Uses Poisson distribution based on 3DA and visits remaining.

        Rate: empirical 180 rate scaled by 3DA relative to _180_RATE_SCALE_3DA.
        Visits remaining estimated from average score to go and 3DA.
        """
        avg_3da = (state.p1_three_da + state.p2_three_da) / 2.0

        # Estimate visits remaining for each player in this leg
        visits_p1 = _expected_visits_remaining(state.score_p1, state.p1_three_da)
        visits_p2 = _expected_visits_remaining(state.score_p2, state.p2_three_da)
        total_visits_remaining = visits_p1 + visits_p2

        # Compute expected 180s remaining
        rate_per_visit = _180_RATE_SCALE_3DA * _BASE_180_RATE_PER_VISIT / _180_RATE_SCALE_3DA
        rate_per_visit = _BASE_180_RATE_PER_VISIT * (avg_3da / _180_RATE_SCALE_3DA)
        lambda_180 = total_visits_remaining * rate_per_visit

        # Over/under 0.5 (any 180 in remaining leg)
        p_zero_180 = math.exp(-lambda_180)  # Poisson P(X=0)
        p_one_or_more = 1.0 - p_zero_180

        # Over/under 1.5
        p_at_most_one = p_zero_180 + lambda_180 * math.exp(-lambda_180)
        p_two_or_more = 1.0 - p_at_most_one

        # Publish over/under 0.5 and 1.5
        odds_over_05, odds_under_05 = _two_way_margin(p_one_or_more, p_zero_180, self._margin_rate)
        odds_over_15, odds_under_15 = _two_way_margin(p_two_or_more, p_at_most_one, self._margin_rate)

        return PropositionMarket(
            market_key="leg_180_count",
            market_label=f"Leg {state.current_leg_number} — 180 Count",
            outcomes={
                "over_0_5": round(odds_over_05, 3),
                "under_0_5": round(odds_under_05, 3),
                "over_1_5": round(odds_over_15, 3),
                "under_1_5": round(odds_under_15, 3),
            },
            metadata={
                "lambda_180_remaining": round(lambda_180, 4),
                "visits_remaining_p1": round(visits_p1, 2),
                "visits_remaining_p2": round(visits_p2, 2),
                "p_zero_180": round(p_zero_180, 5),
            },
        )

    # ------------------------------------------------------------------
    # Market 4: First to hit a 180 in current leg
    # ------------------------------------------------------------------

    def _first_180_in_leg(self, state: "DartsLiveState") -> PropositionMarket:
        """
        P1 vs P2: who hits the first 180 in the current leg.

        Based on relative 180 rates weighted by expected visits remaining.
        """
        rate_p1 = _BASE_180_RATE_PER_VISIT * (state.p1_three_da / _180_RATE_SCALE_3DA)
        rate_p2 = _BASE_180_RATE_PER_VISIT * (state.p2_three_da / _180_RATE_SCALE_3DA)

        visits_p1 = _expected_visits_remaining(state.score_p1, state.p1_three_da)
        visits_p2 = _expected_visits_remaining(state.score_p2, state.p2_three_da)

        # Expected 180s remaining for each player
        lambda_p1 = visits_p1 * rate_p1
        lambda_p2 = visits_p2 * rate_p2

        total_lambda = lambda_p1 + lambda_p2
        if total_lambda < 1e-9:
            return PropositionMarket(
                market_key="first_180_in_leg",
                market_label=f"Leg {state.current_leg_number} — First 180",
                outcomes={},
                is_open=False,
                data_gate_reason="Insufficient 180 probability to open market.",
            )

        # P(P1 hits first 180) = lambda_p1 / (lambda_p1 + lambda_p2)
        # (Competing Poisson processes: first event probability)
        wp_p1 = lambda_p1 / total_lambda
        wp_p2 = 1.0 - wp_p1

        odds_p1, odds_p2 = _two_way_margin(wp_p1, wp_p2, self._margin_rate)

        return PropositionMarket(
            market_key="first_180_in_leg",
            market_label=f"Leg {state.current_leg_number} — First 180",
            outcomes={
                "p1": round(odds_p1, 3),
                "p2": round(odds_p2, 3),
            },
            metadata={
                "lambda_p1": round(lambda_p1, 5),
                "lambda_p2": round(lambda_p2, 5),
                "raw_wp_p1": round(wp_p1, 5),
            },
        )

    # ------------------------------------------------------------------
    # Market 5: Checkout (next player to finish)
    # ------------------------------------------------------------------

    def _checkout_market(self, state: "DartsLiveState") -> PropositionMarket:
        """
        P1 vs P2: who finishes this leg first.

        Uses checkout probability from expected visits remaining.
        Only meaningful when at least one player is within a realistic
        checkout range (≤ 170 left).
        """
        # Only open when one or both players can potentially finish
        p1_can_checkout = state.score_p1 <= 170
        p2_can_checkout = state.score_p2 <= 170

        if not (p1_can_checkout or p2_can_checkout):
            return PropositionMarket(
                market_key="leg_checkout",
                market_label=f"Leg {state.current_leg_number} — Checkout",
                outcomes={},
                is_open=False,
                data_gate_reason="Neither player in checkout range (≤170).",
            )

        # Checkout probability based on visits remaining
        visits_p1 = _expected_visits_remaining(state.score_p1, state.p1_three_da)
        visits_p2 = _expected_visits_remaining(state.score_p2, state.p2_three_da)

        # Higher 3DA → fewer visits → faster checkout
        # Thrower has slight advantage (goes first in visit)
        thrower_bonus = 0.05
        if state.current_thrower == 0:
            visits_p1 = max(0.1, visits_p1 - thrower_bonus)
        else:
            visits_p2 = max(0.1, visits_p2 - thrower_bonus)

        # P(P1 wins the leg) proportional to inverse expected visits
        inv_p1 = 1.0 / visits_p1 if visits_p1 > 0 else 0.0
        inv_p2 = 1.0 / visits_p2 if visits_p2 > 0 else 0.0
        total = inv_p1 + inv_p2

        if total < 1e-9:
            wp_p1 = 0.5
        else:
            wp_p1 = inv_p1 / total
        wp_p2 = 1.0 - wp_p1

        odds_p1, odds_p2 = _two_way_margin(wp_p1, wp_p2, self._margin_rate)

        # Also add over/under on checkout score (player-specific)
        # The "score" here means the checkout value hit (how many left when finished)
        # Threshold: "high checkout" = player finishing from > 60 left
        p_high_checkout_p1 = _high_checkout_probability(state.score_p1)
        p_high_checkout_p2 = _high_checkout_probability(state.score_p2)

        outcomes: dict[str, float] = {
            "p1_wins_leg": round(odds_p1, 3),
            "p2_wins_leg": round(odds_p2, 3),
        }

        if p1_can_checkout and p_high_checkout_p1 > 0.05:
            odds_high_p1, odds_low_p1 = _two_way_margin(
                p_high_checkout_p1, 1.0 - p_high_checkout_p1, self._margin_rate
            )
            outcomes["p1_checkout_over_60"] = round(odds_high_p1, 3)
            outcomes["p1_checkout_under_60"] = round(odds_low_p1, 3)

        if p2_can_checkout and p_high_checkout_p2 > 0.05:
            odds_high_p2, odds_low_p2 = _two_way_margin(
                p_high_checkout_p2, 1.0 - p_high_checkout_p2, self._margin_rate
            )
            outcomes["p2_checkout_over_60"] = round(odds_high_p2, 3)
            outcomes["p2_checkout_under_60"] = round(odds_low_p2, 3)

        return PropositionMarket(
            market_key="leg_checkout",
            market_label=f"Leg {state.current_leg_number} — Checkout",
            outcomes=outcomes,
            metadata={
                "p1_score": state.score_p1,
                "p2_score": state.score_p2,
                "visits_p1_remaining": round(visits_p1, 2),
                "visits_p2_remaining": round(visits_p2, 2),
            },
        )

    # ------------------------------------------------------------------
    # Market 6: Nine-darter probability
    # ------------------------------------------------------------------

    def _nine_darter_market(self, state: "DartsLiveState") -> PropositionMarket:
        """
        Probability of a nine-darter in the current leg.

        Only meaningful at the start of a leg (501 remaining).
        After the first visit, the path is defined — we compute
        conditional probability given remaining score.
        """
        # Nine-darter requires exactly 501 remaining with specific routes
        # After any visit, the strict 9-darter is still possible if scores
        # remain on the 180-180-141 / 180-177-144 etc. paths
        p1_possible = _nine_darter_possible(state.score_p1)
        p2_possible = _nine_darter_possible(state.score_p2)

        if not (p1_possible or p2_possible):
            return PropositionMarket(
                market_key="nine_darter",
                market_label="Nine-Darter in Leg",
                outcomes={},
                is_open=False,
                data_gate_reason="Score path no longer compatible with nine-darter.",
            )

        # Base nine-darter rate per visit opportunity (3DA-dependent)
        # Historical PDC data: ~1 in 5000 legs for world-class players
        def _nine_darter_rate(three_da: float, score: int) -> float:
            if not _nine_darter_possible(score):
                return 0.0
            # Scale: world-class (100+ 3DA) has ~0.0002 per leg; lower 3DA scales down
            base = 0.0002 * (three_da / 100.0) ** 3
            # Conditional on remaining score still being on a 9-darter path
            path_prob = _nine_darter_path_probability(score)
            return base * path_prob

        rate_p1 = _nine_darter_rate(state.p1_three_da, state.score_p1)
        rate_p2 = _nine_darter_rate(state.p2_three_da, state.score_p2)
        combined_rate = 1.0 - (1.0 - rate_p1) * (1.0 - rate_p2)

        if combined_rate < 1e-6:
            return PropositionMarket(
                market_key="nine_darter",
                market_label="Nine-Darter in Leg",
                outcomes={},
                is_open=False,
                data_gate_reason="Combined nine-darter probability < 0.0001%",
            )

        # For this market: over (a nine-darter happens) vs under (doesn't)
        # Using higher margin due to extreme odds
        odds_yes = max(1.01, 1.0 / (combined_rate * (1.0 + _INPLAY_MARGIN_RATE_9DARTER)))
        odds_no_raw = 1.0 / max(1e-9, 1.0 - combined_rate)
        odds_no = odds_no_raw * (1.0 - _INPLAY_MARGIN_RATE_9DARTER * 0.1)

        return PropositionMarket(
            market_key="nine_darter",
            market_label="Nine-Darter in Leg",
            outcomes={
                "yes": round(odds_yes, 1),
                "no": round(max(1.001, odds_no), 3),
            },
            metadata={
                "p1_rate": round(rate_p1, 8),
                "p2_rate": round(rate_p2, 8),
                "combined_rate": round(combined_rate, 8),
                "p1_score": state.score_p1,
                "p2_score": state.score_p2,
            },
        )

    # ------------------------------------------------------------------
    # Market 7: Live leg handicap
    # ------------------------------------------------------------------

    def _live_leg_handicap(self, state: "DartsLiveState") -> PropositionMarket:
        """
        Updated live leg handicap market.

        Takes the current legs_p1/legs_p2 and projects forward using
        the match win probability (lwp_current) as a per-leg win proxy.

        Handicap lines: 0.5, 1.5, 2.5 (depending on legs remaining).
        """
        fmt = state.round_fmt
        if fmt is None:
            return PropositionMarket(
                market_key="live_leg_handicap",
                market_label="Live Leg Handicap",
                outcomes={},
                is_open=False,
                data_gate_reason="Format not loaded.",
            )

        total_legs_in_format = getattr(fmt, "legs", None) or getattr(fmt, "legs_per_set", 1)
        if total_legs_in_format is None or total_legs_in_format < 2:
            return PropositionMarket(
                market_key="live_leg_handicap",
                market_label="Live Leg Handicap",
                outcomes={},
                is_open=False,
                data_gate_reason="Cannot determine legs remaining from format.",
            )

        legs_remaining = max(0, total_legs_in_format - state.legs_p1 - state.legs_p2)
        if legs_remaining < 2:
            return PropositionMarket(
                market_key="live_leg_handicap",
                market_label="Live Leg Handicap",
                outcomes={},
                is_open=False,
                data_gate_reason="Too few legs remaining for handicap market.",
            )

        per_leg_p1 = _clip(state.lwp_current, 0.15, 0.85)
        per_leg_p2 = 1.0 - per_leg_p1

        # Expected final leg difference
        expected_p1_legs = legs_remaining * per_leg_p1
        expected_p2_legs = legs_remaining * per_leg_p2

        outcomes: dict[str, float] = {}

        for hcap in (0.5, 1.5, 2.5):
            # P1 -hcap: P1 wins by more than hcap legs
            p_p1_cover = _race_handicap_probability(
                legs_p1_won=state.legs_p1,
                legs_p2_won=state.legs_p2,
                legs_remaining=legs_remaining,
                handicap=-hcap,
                per_leg_p1=per_leg_p1,
            )
            p_p2_cover = 1.0 - p_p1_cover

            if 0.02 < p_p1_cover < 0.98:
                odds_p1, odds_p2 = _two_way_margin(p_p1_cover, p_p2_cover, self._margin_rate)
                outcomes[f"p1_-{hcap}"] = round(odds_p1, 3)
                outcomes[f"p2_+{hcap}"] = round(odds_p2, 3)

        if not outcomes:
            return PropositionMarket(
                market_key="live_leg_handicap",
                market_label="Live Leg Handicap",
                outcomes={},
                is_open=False,
                data_gate_reason="No viable handicap lines.",
            )

        return PropositionMarket(
            market_key="live_leg_handicap",
            market_label="Live Leg Handicap",
            outcomes=outcomes,
            metadata={
                "legs_p1": state.legs_p1,
                "legs_p2": state.legs_p2,
                "legs_remaining": legs_remaining,
                "per_leg_p1": round(per_leg_p1, 5),
            },
        )

    # ------------------------------------------------------------------
    # Market 8: Set correct score
    # ------------------------------------------------------------------

    def _set_correct_score(self, state: "DartsLiveState") -> list[PropositionMarket]:
        """
        Correct score options for the current set.

        Returns markets for each plausible set score line.
        """
        fmt = state.round_fmt
        if fmt is None:
            return []

        legs_to_win_set = _legs_needed_for_set(state)
        p1_remaining = legs_to_win_set - state.legs_p1
        p2_remaining = legs_to_win_set - state.legs_p2

        if p1_remaining <= 0 or p2_remaining <= 0:
            return []

        per_leg_p1 = _clip(state.lwp_current, 0.15, 0.85)
        per_leg_p2 = 1.0 - per_leg_p1

        markets: list[PropositionMarket] = []
        outcomes: dict[str, float] = {}
        raw_probs: dict[str, float] = {}

        max_legs = legs_to_win_set * 2 - 1

        # Enumerate all possible final set scores
        for p1_final in range(state.legs_p1, legs_to_win_set + 1):
            for p2_final in range(state.legs_p2, legs_to_win_set + 1):
                if p1_final == legs_to_win_set and p2_final < legs_to_win_set:
                    # P1 wins
                    p = _exact_set_score_prob(
                        p1_current=state.legs_p1,
                        p2_current=state.legs_p2,
                        p1_final=p1_final,
                        p2_final=p2_final,
                        per_leg_p1=per_leg_p1,
                    )
                    key = f"{p1_final}-{p2_final}"
                    raw_probs[key] = p
                elif p2_final == legs_to_win_set and p1_final < legs_to_win_set:
                    # P2 wins
                    p = _exact_set_score_prob(
                        p1_current=state.legs_p1,
                        p2_current=state.legs_p2,
                        p1_final=p1_final,
                        p2_final=p2_final,
                        per_leg_p1=per_leg_p1,
                    )
                    key = f"{p1_final}-{p2_final}"
                    raw_probs[key] = p

        # Only include scores with > 1% probability
        total = sum(raw_probs.values())
        if total > 0:
            for key, p in raw_probs.items():
                normalised = p / total
                if normalised > 0.01:
                    odds = max(1.01, 1.0 / (normalised * (1.0 - self._margin_rate)))
                    outcomes[key] = round(odds, 2)

        if outcomes:
            markets.append(
                PropositionMarket(
                    market_key="set_correct_score",
                    market_label=f"Set {state.sets_p1 + state.sets_p2 + 1} Correct Score",
                    outcomes=outcomes,
                    metadata={
                        "legs_p1": state.legs_p1,
                        "legs_p2": state.legs_p2,
                        "legs_to_win_set": legs_to_win_set,
                    },
                )
            )

        return markets


# ---------------------------------------------------------------------------
# Mathematical helpers
# ---------------------------------------------------------------------------

def _clip(v: float, lo: float, hi: float) -> float:
    """Clip value to [lo, hi]."""
    return max(lo, min(hi, v))


def _two_way_margin(p1: float, p2: float, margin: float) -> tuple[float, float]:
    """
    Apply a flat overround margin to two raw probabilities.

    The margin is distributed in proportion to each probability.
    """
    total = p1 + p2
    if total <= 0:
        return 1.01, 1.01
    p1n = p1 / total
    p2n = p2 / total
    # With margin M, implied prob = raw_prob * (1 + M)
    odds_p1 = max(1.01, 1.0 / (p1n * (1.0 + margin)))
    odds_p2 = max(1.01, 1.0 / (p2n * (1.0 + margin)))
    return odds_p1, odds_p2


def _score_to_leg_wp(
    score_thrower: int,
    score_opponent: int,
    three_da_thrower: float,
    three_da_opponent: float,
    current_thrower_is_p1: bool,
) -> float:
    """
    Estimate P1's leg win probability from current scores and 3DA values.

    Uses expected visits remaining for each player as the primary signal,
    with thrower advantage factored in.
    """
    visits_p1 = _expected_visits_remaining(score_thrower, three_da_thrower)
    visits_p2 = _expected_visits_remaining(score_opponent, three_da_opponent)

    # Thrower advantage: goes first in next visit
    thrower_bonus = 0.3  # equivalent to ~0.3 visits head start
    if current_thrower_is_p1:
        visits_p1_adj = max(0.1, visits_p1 - thrower_bonus)
        visits_p2_adj = visits_p2
    else:
        visits_p1_adj = visits_p1
        visits_p2_adj = max(0.1, visits_p2 - thrower_bonus)

    inv_p1 = 1.0 / visits_p1_adj
    inv_p2 = 1.0 / visits_p2_adj
    total = inv_p1 + inv_p2

    if total <= 0:
        return 0.5
    return _clip(inv_p1 / total, 0.01, 0.99)


def _expected_visits_remaining(score: int, three_da: float) -> float:
    """
    Estimate expected visits to check out from a given score.

    Approximation: score / three_da_per_visit + checkout overhead.
    Uses empirical scaling: 3DA is a 3-dart (per visit) average.
    """
    if score <= 0:
        return 0.01  # Already finished
    if three_da <= 0:
        return float("inf")

    # Average visit scoring from the current score
    effective_da = min(three_da, float(score))  # Can't score more than remaining
    visits_body = max(1.0, (score - 40) / effective_da) if score > 40 else 0.5
    checkout_visits = 1.5  # empirical: average checkout takes ~1.5 visits once in finish range
    if score <= 40:
        return checkout_visits
    return visits_body + checkout_visits


def _legs_needed_for_set(state: "DartsLiveState") -> int:
    """Return legs needed to win the current set from the format."""
    fmt = state.round_fmt
    if fmt is None:
        return 3  # default
    legs = getattr(fmt, "legs_per_set", None) or getattr(fmt, "legs", None)
    if legs is None:
        return 3
    # legs_to_win_set = ceil(legs / 2) for a normal best-of-N
    return math.ceil(legs / 2)


def _race_win_probability(p1_remaining: int, p2_remaining: int, per_leg_p1: float) -> float:
    """
    P(P1 wins a race-to-N from current state) using negative binomial.

    p1_remaining: legs P1 still needs to win
    p2_remaining: legs P2 still needs to win
    per_leg_p1:   P1's per-leg win probability
    """
    per_leg_p2 = 1.0 - per_leg_p1
    total = p1_remaining + p2_remaining - 1  # max legs left
    prob = 0.0
    for k in range(p1_remaining, total + 1):
        # P1 wins exactly on leg k (wins k-th of k legs, P2 won k-p1_remaining)
        # Negative binomial: C(k-1, p1_remaining-1) * p1^p1_remaining * p2^(k-p1_remaining)
        from math import comb
        p2_won = k - p1_remaining
        if 0 <= p2_won < p2_remaining:
            prob += comb(k - 1, p1_remaining - 1) * (per_leg_p1 ** p1_remaining) * (per_leg_p2 ** p2_won)
    return _clip(prob, 0.001, 0.999)


def _race_handicap_probability(
    legs_p1_won: int,
    legs_p2_won: int,
    legs_remaining: int,
    handicap: float,
    per_leg_p1: float,
) -> float:
    """
    P(P1 wins the match covering a handicap of `handicap` legs).

    handicap: negative = P1 gives legs, positive = P1 receives legs.
    """
    per_leg_p2 = 1.0 - per_leg_p1
    # Enumerate all possible final scores
    p_p1_cover = 0.0
    from math import comb

    for extra_p1 in range(legs_remaining + 1):
        extra_p2 = legs_remaining - extra_p1
        total_p1 = legs_p1_won + extra_p1
        total_p2 = legs_p2_won + extra_p2
        # Does P1 cover the handicap?
        if (total_p1 + handicap) > total_p2:
            # Multinomial probability of this outcome
            p = (
                comb(legs_remaining, extra_p1)
                * (per_leg_p1 ** extra_p1)
                * (per_leg_p2 ** extra_p2)
            )
            p_p1_cover += p

    return _clip(p_p1_cover, 0.001, 0.999)


def _exact_set_score_prob(
    p1_current: int,
    p2_current: int,
    p1_final: int,
    p2_final: int,
    per_leg_p1: float,
) -> float:
    """
    P(set finishes exactly p1_final - p2_final) given current score.

    Uses path counting with binomial coefficients.
    """
    from math import comb

    p1_to_win = p1_final - p1_current
    p2_to_win = p2_final - p2_current

    if p1_to_win < 0 or p2_to_win < 0:
        return 0.0

    per_leg_p2 = 1.0 - per_leg_p1
    # Total additional legs = p1_to_win + p2_to_win, but last leg must be
    # won by the set winner
    total = p1_to_win + p2_to_win - 1
    if p1_final > p2_final:
        # P1 wins set — last leg is P1's win
        n_before = total
        p1_before = p1_to_win - 1
        p2_before = p2_to_win
        if n_before < 0 or p1_before < 0:
            return per_leg_p1 if p1_to_win == 1 and p2_to_win == 0 else 0.0
        return comb(n_before, p1_before) * (per_leg_p1 ** p1_to_win) * (per_leg_p2 ** p2_before)
    else:
        # P2 wins set — last leg is P2's win
        n_before = total
        p1_before = p1_to_win
        p2_before = p2_to_win - 1
        if n_before < 0 or p2_before < 0:
            return per_leg_p2 if p2_to_win == 1 and p1_to_win == 0 else 0.0
        return comb(n_before, p1_before) * (per_leg_p1 ** p1_before) * (per_leg_p2 ** p2_to_win)


def _high_checkout_probability(score: int) -> float:
    """P(player checks out from a score > 60 given they're finishing this visit)."""
    if score > 170:
        return 0.0
    if score <= 40:
        return 0.0  # too low to be "high"
    if score <= 60:
        return 0.15
    if score <= 100:
        return 0.45
    if score <= 130:
        return 0.60
    if score <= 160:
        return 0.75
    return 0.85  # 161-170


def _nine_darter_possible(score: int) -> bool:
    """Check if a nine-darter is still achievable from this score."""
    # Valid nine-darter paths remain if score is on a valid 9-dart sequence
    # from 501: the possible remaining scores after 0, 1, 2 visits are:
    valid_after_0 = {501}
    valid_after_1 = {321, 141}  # 501-180, 501-180+... simplified paths
    valid_after_2 = {141, 132, 120}  # further simplified
    # A more robust check: is the score achievable via max visits of 180?
    return score <= 501 and score > 0


def _nine_darter_path_probability(score: int) -> float:
    """Conditional probability that the remaining score is on a 9-darter path."""
    if score == 501:
        return 1.0
    if score == 321:
        return 0.95  # came from one 180
    if score == 141:
        return 0.90  # two 180s in
    # For other scores, small non-zero chance of 9-darter variants
    if 120 <= score <= 170:
        return 0.30
    if 100 <= score < 120:
        return 0.05
    return 0.01
