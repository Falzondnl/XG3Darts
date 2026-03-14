"""
Race-to-X legs engine.

Computes the probability that player 1 reaches exactly X legs before
player 2, given per-leg hold and break probabilities.

The computation uses exact dynamic programming (no simulation):
  state = (p1_legs, p2_legs, server)   server ∈ {1, 2}
  value = P(p1 reaches target before p2 | state)

With target ≤ 14 legs the state space is tiny (~400 states) and the
recursion terminates in microseconds.
"""
from __future__ import annotations

import math
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class RaceResult:
    p1_win: float
    p2_win: float
    target: int
    p1_hold: float
    p2_hold: float

    @property
    def p1_odds(self) -> Optional[float]:
        return 1.0 / self.p1_win if self.p1_win > 1e-9 else None

    @property
    def p2_odds(self) -> Optional[float]:
        return 1.0 / self.p2_win if self.p2_win > 1e-9 else None


def race_to_x(
    target: int,
    p1_hold: float,
    p2_hold: float,
    p1_starts: bool = True,
) -> RaceResult:
    """
    Compute race-to-X probabilities.

    Parameters
    ----------
    target:
        Number of legs required to win the race (e.g. 3 for "race to 3").
    p1_hold:
        Probability player 1 wins a leg they serve.
    p2_hold:
        Probability player 2 wins a leg they serve.
    p1_starts:
        True if player 1 serves leg 1.

    Returns
    -------
    RaceResult
        p1_win and p2_win (sum to 1.0 by construction).
    """
    if target < 1:
        raise ValueError(f"target must be >= 1, got {target}")
    if not (0.0 <= p1_hold <= 1.0 and 0.0 <= p2_hold <= 1.0):
        raise ValueError("Hold probabilities must be in [0, 1]")

    p1_break = 1.0 - p2_hold
    p2_break = 1.0 - p1_hold

    @lru_cache(maxsize=None)
    def _p1_wins(a: int, b: int, server: int) -> float:
        """P(p1 reaches target first) from state (a, b, server)."""
        if a >= target:
            return 1.0
        if b >= target:
            return 0.0
        if server == 1:
            p_win_leg = p1_hold
        else:
            p_win_leg = p1_break
        return p_win_leg * _p1_wins(a + 1, b, 2 if server == 1 else 1) + \
               (1.0 - p_win_leg) * _p1_wins(a, b + 1, 2 if server == 1 else 1)

    first_server = 1 if p1_starts else 2
    p1 = _p1_wins(0, 0, first_server)
    _p1_wins.cache_clear()

    return RaceResult(
        p1_win=p1,
        p2_win=1.0 - p1,
        target=target,
        p1_hold=p1_hold,
        p2_hold=p2_hold,
    )


def apply_margin(result: RaceResult, margin: float) -> dict[str, float]:
    """
    Apply a balanced margin to true probabilities.

    Parameters
    ----------
    result:
        Race result with true probabilities.
    margin:
        Overround to apply (e.g. 0.05 for 5%).

    Returns
    -------
    dict with keys p1_win, p2_win (adjusted probs, sum to 1 + margin).
    """
    half = margin / 2.0
    p1_adj = result.p1_win + half
    p2_adj = result.p2_win + half
    # Normalise to exactly (1 + margin)
    total = p1_adj + p2_adj
    scale = (1.0 + margin) / total
    return {
        "p1_win": round(p1_adj * scale, 6),
        "p2_win": round(p2_adj * scale, 6),
    }
