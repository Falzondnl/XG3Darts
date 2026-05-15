"""
Darts Tier 2B Reverse-Engineering Module
=========================================
LOCK-DARTS-TIER-2B-REVERSE-ENGINEER-001

Inverts Pinnacle's de-vigged player-A win probability into the underlying
three-dart-average (3DA) pair (3da_a, 3da_b) via:

  1. Pin player A's 3DA from ELO + tour baseline (analytical prior).
  2. Solve for player B's 3DA via brentq on the Markov leg-win-probability
     function such that:
         markov_match_prob(3da_a_prior, 3da_b) = p_pinnacle_a

Markov leg-win probability derivation:
  For 501 darts, P(player wins leg | serving) is approximately:
      P(hold) ≈ f(3DA_holder, 3DA_breaker)

  Rather than running the full 501 Markov chain (O(501 * V) per evaluation
  which is too slow for brentq), we use the analytical approximation derived
  from the Markov engine:

      expected_darts_to_finish = 501 / (3da / 3)  [simplified: darts/visit * visits_to_finish]

  More precisely:
      avg_visits_to_finish = 501 / (3da / 3)    [3 darts per visit]
      P(player A finishes on visit k) ≈ Geometric(1/avg_visits)

  P(A wins leg | A throws first) = sum_k P(A finishes on k-th visit, B hasn't finished yet)

  Using the exact convolution formula:
      Let p_a = 1 / avg_visits_a  (P A finishes any given visit)
      Let p_b = 1 / avg_visits_b  (P B finishes any given visit)

      P(A wins | A goes first) = sum_{k=1}^{inf} p_a*(1-p_a)^(k-1) * (1-p_b)^(k-1)
                                = p_a / (1 - (1-p_a)*(1-p_b))

  This is the standard geometric series competition formula.
  It is a closed-form monotone function of 3DA_b, enabling brentq convergence.

  Match format: P(A wins best-of-N legs or sets) via negative-binomial.

Author: War Room Wave II.4 (2026-05-15)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from scipy.optimize import brentq

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 3DA feasible range for brentq
_3DA_LO: float = 25.0   # Minimum realistic 3DA (development-level player)
_3DA_HI: float = 125.0  # Maximum realistic 3DA (van Gerwen peak ~105 typical; 125 covers
                         # exhibition records and extreme mismatches for brentq bracketing

# PDC tour baselines by ecosystem (median 3DA from DartConnect/PDC data)
_TOUR_3DA_BASELINES: dict[str, float] = {
    "pdc_mens": 96.0,       # PDC World Championship median qualifier
    "pdc_womens": 74.0,
    "wdf_open": 68.0,
    "development": 55.0,
}

# ELO scale for 3DA prior (empirically calibrated: 100 ELO pts ~ 2 3DA pts)
_ELO_SCALE: float = 400.0
_3DA_ELO_SENSITIVITY: float = 8.0  # 3DA units per 400 ELO

# Brentq convergence parameters
_BRENTQ_XTOL: float = 1e-5
_BRENTQ_RTOL: float = 1e-5
_BRENTQ_MAXITER: int = 30

# Parity tolerance (0.5pp)
_PARITY_TOLERANCE: float = 0.005


# ---------------------------------------------------------------------------
# Leg-win probability (analytical closed-form)
# ---------------------------------------------------------------------------

def _avg_visits_to_finish(three_da: float) -> float:
    """
    Average number of 3-dart visits to finish a 501 leg.

    three_da: three-dart average (e.g. 96.0 = 96 points per 3 darts).
    Returns: expected visits to close the leg.

    501 / (3da / 3) is the naive estimate (3 darts per visit, 3da/3 per dart).
    We apply a small correction for the checkout requiring a double:
      - Roughly 20% of the average 3DA is wasted on sub-optimal checkout routes.
      - Effective 3DA for finish = 3da * 0.80 (conservative calibration).
    """
    if three_da <= 0:
        raise ValueError(f"three_da must be > 0, got {three_da}")
    effective_3da = three_da * 0.80
    darts_per_visit = 3.0
    visits = 501.0 / (effective_3da / darts_per_visit)
    return max(visits, 4.0)  # physical floor: at least 4 visits (12 darts = 501 min)


def _p_visit_finish(three_da: float) -> float:
    """P(player finishes leg on any given visit), geometric approximation."""
    avg_v = _avg_visits_to_finish(three_da)
    return 1.0 / avg_v


def _leg_win_prob(three_da_a: float, three_da_b: float, a_starts: bool = True) -> float:
    """
    P(player A wins a leg) in a 501 darts competition.

    Uses the exact geometric competition formula:
      P(A wins | A goes first) = p_a / (1 - (1-p_a)(1-p_b))

    where p_a, p_b = P(player finishes on any given visit).

    When B starts first:
      P(A wins | B goes first) = p_a * (1-p_b) / (1 - (1-p_a)(1-p_b))
    """
    p_a = _p_visit_finish(three_da_a)
    p_b = _p_visit_finish(three_da_b)
    common_denom = 1.0 - (1.0 - p_a) * (1.0 - p_b)
    if common_denom < 1e-10:
        return 0.5
    if a_starts:
        return p_a / common_denom
    else:
        # B starts: A must wait for B's first visit to fail, then win from visit 2
        return (p_a * (1.0 - p_b)) / common_denom


def _match_prob_darts(
    three_da_a: float,
    three_da_b: float,
    legs_to_win: int = 7,
    sets_to_win: Optional[int] = None,
) -> float:
    """
    P(A wins the match) for a standard PDC darts format.

    For leg-based formats (no sets): binomial over legs.
    For set-based formats: binomial over sets, each set via leg composition.

    Default: best-of-13-legs (PDC standard, first to 7) — uses leg_win_prob
    with alternating starts (A starts first leg of each set).

    Parameters
    ----------
    three_da_a : float
    three_da_b : float
    legs_to_win : int
        Legs needed to win the match (for leg-only format) or legs per set.
    sets_to_win : Optional[int]
        If provided, match is best-of-(2*sets_to_win-1) sets, each set is
        best-of-(2*legs_to_win-1) legs.

    Returns
    -------
    float
        P(A wins match).
    """
    # Average leg-win probability (symmetric starts, averaged 50/50 for match-level)
    p_leg_a = 0.5 * (_leg_win_prob(three_da_a, three_da_b, a_starts=True)
                     + _leg_win_prob(three_da_a, three_da_b, a_starts=False))

    if sets_to_win is None:
        # Leg-based match
        return _negative_binomial_win_prob(p_leg_a, legs_to_win)

    # Set-based match: each set is a NegBin(p_leg_a, legs_to_win) outcome
    p_set_a = _negative_binomial_win_prob(p_leg_a, legs_to_win)
    return _negative_binomial_win_prob(p_set_a, sets_to_win)


def _negative_binomial_win_prob(p_win_unit: float, units_to_win: int) -> float:
    """
    P(A wins best-of-(2T-1) contest) where P(A wins each unit) = p_win_unit.
    T = units_to_win.

    Uses negative-binomial: P(A wins exactly T units, B wins j units, j < T)
      = C(T+j-1, j) * p^T * (1-p)^j
    summed over j = 0..T-1.
    """
    from math import comb
    T = units_to_win
    p = p_win_unit
    q = 1.0 - p
    total = 0.0
    for j in range(T):
        total += comb(T + j - 1, j) * (p ** T) * (q ** j)
    return total


# ---------------------------------------------------------------------------
# ELO -> 3DA prior
# ---------------------------------------------------------------------------

def _elo_to_3da_prior(
    elo_a: float,
    elo_b: float,
    ecosystem: str = "pdc_mens",
) -> float:
    """
    Compute player A's 3DA prior from ELO ratings + tour baseline.

    Maps ELO difference to 3DA adjustment: 400 ELO pts -> 8 3DA pts shift
    from the tour baseline. Anchored to player A's ELO position relative
    to the baseline (1500 ELO = tour median).
    """
    baseline = _TOUR_3DA_BASELINES.get(ecosystem, 68.0)
    elo_diff_from_baseline_a = elo_a - 1500.0
    three_da_prior = baseline + (elo_diff_from_baseline_a / _ELO_SCALE) * _3DA_ELO_SENSITIVITY
    return max(_3DA_LO + 1.0, min(_3DA_HI - 1.0, three_da_prior))


# ---------------------------------------------------------------------------
# Core Tier 2B solve
# ---------------------------------------------------------------------------

@dataclass
class DartsTier2BResult:
    """
    Full Tier 2B output for a darts fixture.

    prediction_source is ALWAYS "market_scrape_reverse_engineered" when
    converged=True, per TIER 2B platform contract.
    """
    three_da_a: float
    three_da_b: float
    p_leg_a: float
    p_match_a: float
    p_match_b: float
    ecosystem: str
    legs_to_win: int
    three_da_a_prior_used: float
    prediction_source: str
    model_available: bool
    converged: bool
    solve_residual: float
    solve_iterations: int
    tier_2b_restricted: bool


def reverse_engineer_darts(
    pinnacle_a_prob: float,
    elo_a: float,
    elo_b: float,
    ecosystem: str = "pdc_mens",
    legs_to_win: int = 7,
    sets_to_win: Optional[int] = None,
    fixture_id: str = "",
    correlation_id: str = "",
) -> DartsTier2BResult:
    """
    LOCK-DARTS-TIER-2B-REVERSE-ENGINEER-001

    Invert Pinnacle's de-vigged player-A win probability into (3da_a, 3da_b).

    Algorithm:
      1. Compute 3da_a_prior from ELO + tour baseline.
      2. brentq: find 3da_b in [3DA_LO, 3DA_HI] s.t.
             match_prob_darts(3da_a_prior, 3da_b, ...) = pinnacle_a_prob
      3. Round-trip validate: |repriced - pinnacle| <= 0.5pp.
      4. Return full feature set.

    Parameters
    ----------
    pinnacle_a_prob : float
        De-vigged Pinnacle player-A win probability.
    elo_a : float
        Player A ELO (from darts_elo_ratings DB).
    elo_b : float
        Player B ELO.
    ecosystem : str
        PDC tour ecosystem — used for 3DA baseline lookup.
    legs_to_win : int
        Legs required to win (or legs per set in set-based format).
    sets_to_win : Optional[int]
        If provided, match is set-based.
    fixture_id : str
        For logging only.
    correlation_id : str
        For logging only.

    Returns
    -------
    DartsTier2BResult
        converged=True when solve succeeded and round-trip passes.
        converged=False falls through to Tier 2A.
    """
    if not (0.0 < pinnacle_a_prob < 1.0):
        raise ValueError(
            f"pinnacle_a_prob must be in (0,1), got {pinnacle_a_prob}"
        )

    three_da_a_prior = _elo_to_3da_prior(elo_a, elo_b, ecosystem)

    def _obj(three_da_b: float) -> float:
        try:
            return _match_prob_darts(
                three_da_a_prior, three_da_b, legs_to_win, sets_to_win
            ) - pinnacle_a_prob
        except Exception:
            return float("nan")

    f_lo = _obj(_3DA_LO + 0.5)
    f_hi = _obj(_3DA_HI - 0.5)

    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        log.warning(
            "tier2b_darts_bracket_nan fixture_id=%s corr=%s",
            fixture_id, correlation_id,
        )
        return DartsTier2BResult(
            three_da_a=three_da_a_prior, three_da_b=three_da_a_prior,
            p_leg_a=0.5, p_match_a=pinnacle_a_prob, p_match_b=1.0 - pinnacle_a_prob,
            ecosystem=ecosystem, legs_to_win=legs_to_win,
            three_da_a_prior_used=three_da_a_prior,
            prediction_source="market_scrape_reverse_engineered",
            model_available=False, converged=False, solve_residual=999.0,
            solve_iterations=0, tier_2b_restricted=True,
        )

    if f_lo * f_hi >= 0:
        # Bracket failure: ELO-derived prior for A is too extreme.
        # Fall back to tour baseline — this maximises the achievable range for 3da_b.
        # E.g. for a heavy underdog (p_a=0.35), a tour-median 3da_a allows
        # brentq to find a higher 3da_b solution.
        three_da_a_prior = _TOUR_3DA_BASELINES.get(ecosystem, 68.0)
        f_lo = _obj(_3DA_LO + 0.5)
        f_hi = _obj(_3DA_HI - 0.5)
        if f_lo * f_hi >= 0:
            log.warning(
                "tier2b_darts_no_bracket fixture_id=%s p_a=%.4f 3da_prior=%.2f",
                fixture_id, pinnacle_a_prob, three_da_a_prior,
            )
            return DartsTier2BResult(
                three_da_a=three_da_a_prior, three_da_b=three_da_a_prior,
                p_leg_a=0.5, p_match_a=pinnacle_a_prob, p_match_b=1.0 - pinnacle_a_prob,
                ecosystem=ecosystem, legs_to_win=legs_to_win,
                three_da_a_prior_used=three_da_a_prior,
                prediction_source="market_scrape_reverse_engineered",
                model_available=False, converged=False, solve_residual=999.0,
                solve_iterations=0, tier_2b_restricted=True,
            )

    solve_iters = [0]
    def _obj_counted(x: float) -> float:
        solve_iters[0] += 1
        return _obj(x)

    try:
        three_da_b_solved, result_obj = brentq(
            _obj_counted,
            _3DA_LO + 0.5,
            _3DA_HI - 0.5,
            xtol=_BRENTQ_XTOL,
            rtol=_BRENTQ_RTOL,
            maxiter=_BRENTQ_MAXITER,
            full_output=True,
        )
    except ValueError as exc:
        log.warning(
            "tier2b_darts_brentq_failed fixture_id=%s error=%s",
            fixture_id, str(exc),
        )
        return DartsTier2BResult(
            three_da_a=three_da_a_prior, three_da_b=three_da_a_prior,
            p_leg_a=0.5, p_match_a=pinnacle_a_prob, p_match_b=1.0 - pinnacle_a_prob,
            ecosystem=ecosystem, legs_to_win=legs_to_win,
            three_da_a_prior_used=three_da_a_prior,
            prediction_source="market_scrape_reverse_engineered",
            model_available=False, converged=False, solve_residual=999.0,
            solve_iterations=solve_iters[0], tier_2b_restricted=True,
        )

    # Round-trip validation
    p_repriced = _match_prob_darts(
        three_da_a_prior, three_da_b_solved, legs_to_win, sets_to_win
    )
    residual = abs(p_repriced - pinnacle_a_prob)
    converged = residual <= _PARITY_TOLERANCE

    p_leg_a = 0.5 * (_leg_win_prob(three_da_a_prior, three_da_b_solved, a_starts=True)
                     + _leg_win_prob(three_da_a_prior, three_da_b_solved, a_starts=False))

    if not converged:
        log.warning(
            "tier2b_darts_parity_fail fixture_id=%s residual=%.6f p_a=%.4f repriced=%.4f",
            fixture_id, residual, pinnacle_a_prob, p_repriced,
        )

    log.info(
        "tier2b_darts_solved fixture_id=%s 3da_a=%.2f 3da_b=%.2f "
        "residual=%.6f iters=%d converged=%s",
        fixture_id, three_da_a_prior, three_da_b_solved,
        residual, solve_iters[0], converged,
    )

    return DartsTier2BResult(
        three_da_a=round(three_da_a_prior, 4),
        three_da_b=round(three_da_b_solved, 4),
        p_leg_a=round(p_leg_a, 6),
        p_match_a=round(p_repriced, 6),
        p_match_b=round(1.0 - p_repriced, 6),
        ecosystem=ecosystem,
        legs_to_win=legs_to_win,
        three_da_a_prior_used=round(three_da_a_prior, 4),
        prediction_source="market_scrape_reverse_engineered",
        model_available=False,
        converged=converged,
        solve_residual=round(residual, 8),
        solve_iterations=solve_iters[0],
        tier_2b_restricted=not converged,
    )
