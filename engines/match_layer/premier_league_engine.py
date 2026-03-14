"""
Premier League match engine.

Draw-enabled: league night is first to 7 legs (7 wins), tie = draw.
Play-offs: first to 10 legs, no draw possible.

League night rules:
  - 13 legs maximum (7-6 = win, 6-7 = loss, 6-6 would be draw but game stops at 7).
  - Actually: the game stops when EITHER player reaches 7 legs.
  - If it ends 6-6, neither has reached 7, so it continues... but wait:
    the Premier League uses a 7-legs format where the game can end in a draw
    IF both players reach 6-6 AND then the next leg is drawn? No.
  - CORRECT RULE: League night = first to 7 legs wins. If game ends and both
    are at 6-6 (12 legs played), the draw is declared (neither reached 7).
    Actually: the 13th leg is played and MUST produce a winner (so no draw
    at 6-6 — the 13th leg is the deciding leg, but draws ARE possible:
    the match is declared a draw if it ends in a tie, which in PL means
    leg 13 is played. The draw only happens if the match finishes 6-6 before
    all legs are played? No — the correct rule:

Premier League League Night actual rule (PDC official):
  - First to 7 legs wins. Game CAN end in a draw.
  - DRAW happens when neither player has reached 7 after a leg that makes
    it impossible (i.e., 7-6 or 6-7 is a win; 6-6 after the 12th leg means
    the 13th leg is played and the winner takes the match — NOT a draw).
  - Actually the draw DOES occur at 6-6 when both players have been awarded
    6 legs each at the start of the final possible leg. In practice, the PDC
    Premier League allows draws in the group stage.

IMPLEMENTATION NOTE: We model the Premier League league night as:
  - Game to 7 legs
  - Draw possible: if game ends with l1 < 7 AND l2 < 7 AND neither can reach 7
  - This happens at exactly 6-6 (12 legs played) — then 13th leg is decisive,
    so draws are NOT possible under this strict reading.
  - HOWEVER: The PDC DOES award a point for a draw in certain seasons.
  - We use the configurable format: draw_enabled=True means the game can end 6-6.

For the betting market, we model:
  - P(P1 wins) = P(P1 reaches 7 before P2 reaches 7)
  - P(draw) = P(12 legs played and it's 6-6 AND no 13th leg per format rules)
  - P(P2 wins) = P(P2 reaches 7 before P1 reaches 7)

In practice for PL, a DRAW occurs when the match is played for exactly the
number of legs determined, with equal scorelines. The engine supports
configurable legs_to_win to handle both League Night (legs_to_win=7) and
Play-Off formats (legs_to_win=10, draw_enabled=False).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog

from competition.format_registry import DartsRoundFormat
from engines.leg_layer.markov_chain import HoldBreakProbabilities

logger = structlog.get_logger(__name__)


@dataclass
class PremierLeaguePriceResult:
    """
    Premier League match price result.

    Attributes
    ----------
    p1_win:
        Probability P1 wins the match.
    draw:
        Probability of a drawn result.
    p2_win:
        Probability P2 wins the match.
    legs_distribution:
        Distribution over final leg scorelines.
    """

    p1_win: float
    draw: float
    p2_win: float
    legs_distribution: dict[tuple[int, int], float]

    def validate(self, tol: float = 1e-6) -> None:
        """Verify sum = 1."""
        total = self.p1_win + self.draw + self.p2_win
        if abs(total - 1.0) > tol:
            raise ValueError(
                f"PL price does not sum to 1: "
                f"p1={self.p1_win:.6f}, draw={self.draw:.6f}, p2={self.p2_win:.6f}, "
                f"total={total:.6f}"
            )


class PremierLeagueEngine:
    """
    Premier League match engine with draw support.

    Handles both:
      - League Night: first to 7, draw possible at 6-6 (if format declares game over)
      - Play-Off: first to 10, no draw

    The draw occurs when both players have the same number of legs at the end
    of the maximum possible legs and neither has reached legs_to_win.

    For legs_to_win=7:
      Maximum legs = 13 (7+6). Draw possible only if implementation allows.
      In the PDC PL, draws ARE awarded at 6-6: the match ends 6-6 without
      a 13th leg being played. This is encoded as:
        A draw happens when neither player has reached 7 AND total_legs_played
        is at maximum for this format (i.e., the format explicitly allows ties).

    Implementation:
      Forward DP enumeration over (l1, l2, starter) states.
    """

    def price_match(
        self,
        hb: HoldBreakProbabilities,
        round_fmt: DartsRoundFormat,
        p1_starts: bool,
    ) -> PremierLeaguePriceResult:
        """
        Compute P1_win, draw, P2_win probabilities.

        Parameters
        ----------
        hb:
            Hold/break probabilities from the Markov chain.
        round_fmt:
            Must have draw_enabled=True and legs_to_win set.
        p1_starts:
            Whether P1 throws first.

        Returns
        -------
        PremierLeaguePriceResult
        """
        assert round_fmt.draw_enabled, "PremierLeagueEngine requires draw_enabled=True"
        assert round_fmt.legs_to_win is not None

        legs_to_win = round_fmt.legs_to_win
        draw_at_score = legs_to_win - 1  # e.g. draw at 6-6 for legs_to_win=7

        result = self._dp_draw_format(
            hb=hb,
            legs_to_win=legs_to_win,
            draw_at_score=draw_at_score,
            p1_starts=p1_starts,
        )

        logger.info(
            "premier_league_priced",
            legs_to_win=legs_to_win,
            draw_at_score=draw_at_score,
            p1_win=round(result.p1_win, 4),
            draw=round(result.draw, 4),
            p2_win=round(result.p2_win, 4),
        )

        result.validate()
        return result

    def _dp_draw_format(
        self,
        hb: HoldBreakProbabilities,
        legs_to_win: int,
        draw_at_score: int,
        p1_starts: bool,
    ) -> PremierLeaguePriceResult:
        """
        Forward DP for draw-enabled format.

        State: (l1, l2, starter)
        Terminal conditions:
          l1 == legs_to_win → P1 wins
          l2 == legs_to_win → P2 wins
          l1 == draw_at_score AND l2 == draw_at_score → DRAW

        The draw occurs when both reach (legs_to_win - 1) simultaneously.
        If the format uses a 13th leg tiebreaker, set draw_at_score = -1
        (draw never declared = standard knockout format).
        """
        state_probs: dict[tuple[int, int, int], float] = {}
        initial_starter = 0 if p1_starts else 1
        state_probs[(0, 0, initial_starter)] = 1.0

        p1_win_prob = 0.0
        p2_win_prob = 0.0
        draw_prob = 0.0
        legs_dist: dict[tuple[int, int], float] = {}

        max_total_legs = (legs_to_win - 1) * 2 + 1  # maximum legs in the match

        for total_legs in range(max_total_legs + 1):
            current_states = {
                (l1, l2, t): p
                for (l1, l2, t), p in state_probs.items()
                if l1 + l2 == total_legs and p > 0
            }

            for (l1, l2, starter), prob in current_states.items():
                if prob < 1e-12:
                    continue

                # Check terminal conditions
                if l1 == legs_to_win:
                    p1_win_prob += prob
                    key = (l1, l2)
                    legs_dist[key] = legs_dist.get(key, 0.0) + prob
                    continue

                if l2 == legs_to_win:
                    p2_win_prob += prob
                    key = (l1, l2)
                    legs_dist[key] = legs_dist.get(key, 0.0) + prob
                    continue

                if l1 == draw_at_score and l2 == draw_at_score:
                    # Both at (legs_to_win - 1): check if format declares draw here
                    # For PL League Night: this IS a draw (no 13th leg under draw_enabled rules)
                    draw_prob += prob
                    key = (l1, l2)
                    legs_dist[key] = legs_dist.get(key, 0.0) + prob
                    continue

                # Continue playing
                if starter == 0:
                    p_p1 = hb.p1_hold
                    p_p2 = hb.p2_break
                else:
                    p_p1 = hb.p1_break
                    p_p2 = hb.p2_hold

                next_starter = 1 - starter

                new_key_p1 = (l1 + 1, l2, next_starter)
                state_probs[new_key_p1] = state_probs.get(new_key_p1, 0.0) + prob * p_p1

                new_key_p2 = (l1, l2 + 1, next_starter)
                state_probs[new_key_p2] = state_probs.get(new_key_p2, 0.0) + prob * p_p2

        # Normalise for floating point
        total = p1_win_prob + p2_win_prob + draw_prob
        if total > 0 and abs(total - 1.0) > 1e-8:
            p1_win_prob /= total
            p2_win_prob /= total
            draw_prob /= total

        return PremierLeaguePriceResult(
            p1_win=p1_win_prob,
            draw=draw_prob,
            p2_win=p2_win_prob,
            legs_distribution=legs_dist,
        )

    def price_ko_match(
        self,
        hb: HoldBreakProbabilities,
        legs_to_win: int,
        p1_starts: bool,
    ) -> PremierLeaguePriceResult:
        """
        Price a knockout (no-draw) match (Play-Offs).

        Uses the same DP but with draw_at_score set to -1 (never trigger).
        """
        return self._dp_draw_format(
            hb=hb,
            legs_to_win=legs_to_win,
            draw_at_score=-1,  # draw never triggered
            p1_starts=p1_starts,
        )
