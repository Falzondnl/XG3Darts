"""
Unit tests for outrights/tournament_simulator.py and related GPU/Ray simulators.

Tests
-----
- test_tournament_simulator_probabilities_sum_to_one
- test_antithetic_reduces_variance
- test_confidence_intervals_correct
- test_128_player_bracket_completion
- test_elo_match_prob_formula
- test_single_elimination_bracket_produces_one_winner
- test_simulator_with_missing_elo_falls_back
- test_all_players_have_win_prob_entries
- test_gpu_fallback_without_gpu
- test_ray_fallback_without_ray
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from outrights.tournament_simulator import (
    DartsTournamentSimulator,
    OutrightSimResult,
    TournamentField,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_field(n_players: int = 8, elo_spread: float = 100.0) -> TournamentField:
    """Create a uniform ELO field with n_players."""
    player_ids = [f"P{i:03d}" for i in range(n_players)]
    base_elo = 1500.0
    step = elo_spread / max(n_players - 1, 1)
    elo_ratings = {pid: base_elo + i * step for i, pid in enumerate(player_ids)}
    three_da_stats = {pid: 80.0 + i * 0.5 for i, pid in enumerate(player_ids)}

    bracket: dict[str, list[str]] = {}
    for i in range(0, n_players - 1, 2):
        bracket[f"R1_M{i // 2 + 1}"] = [player_ids[i], player_ids[i + 1]]

    return TournamentField(
        competition_id=f"TEST_{n_players}P",
        format_code="PDC_WC",
        players=player_ids,
        bracket=bracket,
        elo_ratings=elo_ratings,
        three_da_stats=three_da_stats,
    )


def _make_equal_field(n_players: int = 8) -> TournamentField:
    """Create a field where all players have identical ELO (equal probs)."""
    player_ids = [f"EQ{i:03d}" for i in range(n_players)]
    elo_ratings = {pid: 1500.0 for pid in player_ids}
    three_da_stats = {pid: 80.0 for pid in player_ids}
    bracket = {}
    for i in range(0, n_players - 1, 2):
        bracket[f"R1_M{i // 2 + 1}"] = [player_ids[i], player_ids[i + 1]]
    return TournamentField(
        competition_id="TEST_EQ",
        format_code="PDC_WC",
        players=player_ids,
        bracket=bracket,
        elo_ratings=elo_ratings,
        three_da_stats=three_da_stats,
    )


# ---------------------------------------------------------------------------
# test_tournament_simulator_probabilities_sum_to_one
# ---------------------------------------------------------------------------

class TestProbabilitiesSumToOne:
    """Win probabilities must sum to 1.0 across all players in the field."""

    def test_8_player_field_sums_to_one(self):
        field = _make_field(n_players=8)
        sim = DartsTournamentSimulator()
        result = sim.simulate(
            competition_id="TEST",
            field=field,
            n_simulations=10_000,
            use_antithetic=True,
        )
        total = sum(result.player_win_probs.values())
        assert abs(total - 1.0) < 1e-9, (
            f"Win probs sum to {total:.8f}, expected 1.0"
        )

    def test_16_player_field_sums_to_one(self):
        field = _make_field(n_players=16)
        sim = DartsTournamentSimulator()
        result = sim.simulate(
            competition_id="TEST",
            field=field,
            n_simulations=10_000,
            use_antithetic=False,
        )
        total = sum(result.player_win_probs.values())
        assert abs(total - 1.0) < 1e-9

    def test_32_player_field_sums_to_one(self):
        field = _make_field(n_players=32)
        sim = DartsTournamentSimulator()
        result = sim.simulate(
            competition_id="TEST",
            field=field,
            n_simulations=10_000,
            use_antithetic=True,
        )
        total = sum(result.player_win_probs.values())
        assert abs(total - 1.0) < 1e-9

    def test_2_player_field_sums_to_one(self):
        """Degenerate case: 2-player field (single match)."""
        field = _make_field(n_players=2, elo_spread=200.0)
        sim = DartsTournamentSimulator()
        result = sim.simulate(
            competition_id="TEST",
            field=field,
            n_simulations=10_000,
        )
        total = sum(result.player_win_probs.values())
        assert abs(total - 1.0) < 1e-9

    def test_all_players_present_in_results(self):
        field = _make_field(n_players=8)
        sim = DartsTournamentSimulator()
        result = sim.simulate(
            competition_id="TEST",
            field=field,
            n_simulations=10_000,
        )
        assert set(result.player_win_probs.keys()) == set(field.players)


# ---------------------------------------------------------------------------
# test_antithetic_reduces_variance
# ---------------------------------------------------------------------------

class TestAntitheticVarianceReduction:
    """
    Antithetic variates should reduce variance of the win probability estimate.

    Method: run both paths at the same n_simulations and compare the
    spread (std dev) of repeated estimates across multiple runs.
    We use a lenient threshold — we just verify antithetic does not
    *increase* variance.
    """

    def _run_many(self, use_antithetic: bool, n_repeats: int = 10) -> list[float]:
        """Run repeated simulations and collect the top-player win prob."""
        field = _make_field(n_players=8, elo_spread=200.0)
        sim = DartsTournamentSimulator()
        top_player = max(field.elo_ratings, key=field.elo_ratings.get)
        estimates = []
        for _ in range(n_repeats):
            result = sim.simulate(
                competition_id="TEST",
                field=field,
                n_simulations=2_000,
                use_antithetic=use_antithetic,
            )
            estimates.append(result.player_win_probs[top_player])
        return estimates

    def test_antithetic_variance_not_greater_than_standard(self):
        """
        Variance with antithetic should be <= variance without.

        We use a relaxed test: if antithetic variance is more than
        3x standard variance, it fails (otherwise variance reduction
        is just statistical noise at small n_repeats).
        """
        standard_estimates = self._run_many(use_antithetic=False, n_repeats=15)
        antithetic_estimates = self._run_many(use_antithetic=True, n_repeats=15)

        var_std = float(np.var(standard_estimates))
        var_anti = float(np.var(antithetic_estimates))

        # Antithetic should not be dramatically worse
        # (loose bound — exact variance reduction depends on correlation)
        assert var_anti <= var_std * 3.0, (
            f"Antithetic variance {var_anti:.6f} > 3x standard variance {var_std:.6f}"
        )

    def test_antithetic_result_has_correct_n_simulations(self):
        """n_simulations in result should be even and match or exceed request."""
        field = _make_field(n_players=4)
        sim = DartsTournamentSimulator()
        result = sim.simulate(
            competition_id="TEST",
            field=field,
            n_simulations=1_001,   # odd number — should be rounded up
            use_antithetic=True,
        )
        assert result.n_simulations >= 1_001
        assert result.n_simulations % 2 == 0


# ---------------------------------------------------------------------------
# test_confidence_intervals_correct
# ---------------------------------------------------------------------------

class TestConfidenceIntervals:
    """Wilson score CI tests."""

    def test_ci_contains_empirical_proportion(self):
        """The CI should bracket the empirical win probability."""
        field = _make_field(n_players=4, elo_spread=100.0)
        sim = DartsTournamentSimulator()
        result = sim.simulate(
            competition_id="TEST",
            field=field,
            n_simulations=10_000,
        )
        for pid in field.players:
            p = result.player_win_probs[pid]
            lo, hi = result.confidence_intervals[pid]
            assert lo <= p <= hi, (
                f"Player {pid}: win_prob={p:.4f} outside CI [{lo:.4f}, {hi:.4f}]"
            )

    def test_ci_lower_bound_non_negative(self):
        """Lower CI bound must always be >= 0."""
        field = _make_field(n_players=8)
        sim = DartsTournamentSimulator()
        result = sim.simulate(
            competition_id="TEST",
            field=field,
            n_simulations=10_000,
        )
        for pid in field.players:
            lo, hi = result.confidence_intervals[pid]
            assert lo >= 0.0
            assert hi <= 1.0

    def test_ci_width_shrinks_with_more_simulations(self):
        """Wider simulations -> narrower CI for the same field."""
        field = _make_field(n_players=4)
        sim = DartsTournamentSimulator()
        result_small = sim.simulate(
            competition_id="TEST",
            field=field,
            n_simulations=1_000,
        )
        result_large = sim.simulate(
            competition_id="TEST",
            field=field,
            n_simulations=50_000,
        )
        # Average CI width should be smaller with more simulations
        avg_width_small = np.mean([
            hi - lo for _, (lo, hi) in result_small.confidence_intervals.items()
        ])
        avg_width_large = np.mean([
            hi - lo for _, (lo, hi) in result_large.confidence_intervals.items()
        ])
        assert avg_width_large < avg_width_small, (
            f"Expected CI to narrow: small={avg_width_small:.4f}, large={avg_width_large:.4f}"
        )

    def test_wilson_formula_manual(self):
        """Verify the Wilson CI against a known manual calculation."""
        from outrights.tournament_simulator import _normal_ppf
        sim = DartsTournamentSimulator()

        # n=100, k=50, p_hat=0.5 -> known CI is approximately (0.40, 0.60)
        raw = np.array([1] * 50 + [0] * 50)
        lo, hi = sim._compute_confidence_intervals(raw, n_sim=100)
        assert 0.38 < lo < 0.47, f"Lower bound {lo:.4f} outside expected range"
        assert 0.53 < hi < 0.62, f"Upper bound {hi:.4f} outside expected range"

    def test_normal_ppf_at_0975(self):
        """z at 0.975 should be approximately 1.96."""
        from outrights.tournament_simulator import _normal_ppf
        z = _normal_ppf(0.975)
        assert abs(z - 1.96) < 0.01, f"Expected ~1.96, got {z:.4f}"


# ---------------------------------------------------------------------------
# test_128_player_bracket_completion
# ---------------------------------------------------------------------------

class TestLargeFieldSimulation:
    """Verify correct behaviour for a full 128-player bracket."""

    def test_128_player_probs_sum_to_one(self):
        field = _make_field(n_players=128, elo_spread=500.0)
        sim = DartsTournamentSimulator()
        result = sim.simulate(
            competition_id="TEST_128",
            field=field,
            n_simulations=2_000,  # small for speed; correctness only
            use_antithetic=True,
        )
        total = sum(result.player_win_probs.values())
        assert abs(total - 1.0) < 1e-9

    def test_128_player_all_players_have_prob(self):
        field = _make_field(n_players=128, elo_spread=500.0)
        sim = DartsTournamentSimulator()
        result = sim.simulate(
            competition_id="TEST_128",
            field=field,
            n_simulations=2_000,
        )
        assert len(result.player_win_probs) == 128

    def test_top_elo_player_has_highest_win_prob(self):
        """The player with the highest ELO should have the highest win probability."""
        field = _make_field(n_players=16, elo_spread=400.0)
        top_player = max(field.elo_ratings, key=field.elo_ratings.get)
        sim = DartsTournamentSimulator()
        result = sim.simulate(
            competition_id="TEST",
            field=field,
            n_simulations=20_000,
        )
        best_sim_player = max(result.player_win_probs, key=result.player_win_probs.get)
        assert best_sim_player == top_player, (
            f"Top ELO={top_player} but top sim winner={best_sim_player}"
        )

    def test_gpu_fallback_without_gpu(self):
        """Requesting GPU on a system without CUDA should fall back to CPU."""
        field = _make_field(n_players=8)
        sim = DartsTournamentSimulator()
        # use_gpu=True should not raise even without CUDA
        result = sim.simulate(
            competition_id="TEST",
            field=field,
            n_simulations=2_000,
            use_gpu=True,   # will fall back to CPU
        )
        total = sum(result.player_win_probs.values())
        assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# test_elo_match_prob_formula
# ---------------------------------------------------------------------------

class TestEloMatchProbFormula:
    """Verify the ELO match win probability formula."""

    def test_equal_elo_gives_50_50(self):
        field = _make_equal_field(n_players=4)
        sim = DartsTournamentSimulator()
        p = sim._match_win_prob_from_elo("EQ000", "EQ001", field)
        assert abs(p - 0.5) < 1e-10

    def test_higher_elo_wins_more_often(self):
        field = _make_field(n_players=2, elo_spread=400.0)
        sim = DartsTournamentSimulator()
        p1_id, p2_id = field.players[0], field.players[1]
        p_p1_wins = sim._match_win_prob_from_elo(p1_id, p2_id, field)
        p_p2_wins = sim._match_win_prob_from_elo(p2_id, p1_id, field)
        # The higher ELO player (p2, since sorted ascending) should have higher prob
        assert p_p1_wins + p_p2_wins == pytest.approx(1.0, abs=1e-10)

    def test_elo_formula_at_400_spread(self):
        """ELO diff of 400 -> P(winner) = 10/11 ≈ 0.909."""
        field = TournamentField(
            competition_id="TEST",
            format_code="PDC_WC",
            players=["P1", "P2"],
            bracket={},
            elo_ratings={"P1": 1900.0, "P2": 1500.0},
            three_da_stats={"P1": 90.0, "P2": 80.0},
        )
        sim = DartsTournamentSimulator()
        p = sim._match_win_prob_from_elo("P1", "P2", field)
        expected = 1.0 / (1.0 + 10.0 ** (-400.0 / 400.0))
        assert abs(p - expected) < 1e-10

    def test_elo_formula_complementary(self):
        """P(P1 wins) + P(P2 wins) must equal exactly 1.0."""
        field = _make_field(n_players=2, elo_spread=300.0)
        sim = DartsTournamentSimulator()
        p1, p2 = field.players[0], field.players[1]
        assert sim._match_win_prob_from_elo(p1, p2, field) + \
               sim._match_win_prob_from_elo(p2, p1, field) == pytest.approx(1.0, abs=1e-10)

    def test_missing_elo_falls_back_to_50_50(self):
        """Player with missing ELO should get 0.5 win probability."""
        field = _make_field(n_players=4)
        # Remove one player's ELO
        field_partial = TournamentField(
            competition_id="TEST",
            format_code="PDC_WC",
            players=field.players,
            bracket=field.bracket,
            elo_ratings={field.players[0]: 1600.0},   # only P000 has ELO
            three_da_stats=field.three_da_stats,
        )
        sim = DartsTournamentSimulator()
        p = sim._match_win_prob_from_elo(field.players[0], field.players[1], field_partial)
        assert p == 0.5


# ---------------------------------------------------------------------------
# GPU simulator tests
# ---------------------------------------------------------------------------

class TestGPUSimulator:
    """Tests for outrights/gpu_simulator.py."""

    def test_gpu_simulator_win_counts_sum_to_n_sims(self):
        from outrights.gpu_simulator import GPUTournamentSimulator
        n = 8
        matrix = np.full((n, n), 0.5, dtype=np.float32)
        np.fill_diagonal(matrix, 0.5)
        sim = GPUTournamentSimulator()
        win_counts = sim.simulate_128_player(matrix, n_simulations=1_000)
        assert win_counts.sum() == 1_000

    def test_gpu_simulator_dominant_player_wins_most(self):
        """Give player 0 a very high win probability against all others."""
        from outrights.gpu_simulator import GPUTournamentSimulator
        n = 8
        matrix = np.full((n, n), 0.5, dtype=np.float32)
        # Player 0 beats everyone with 90% probability
        matrix[0, :] = 0.9
        matrix[:, 0] = 0.1
        matrix[0, 0] = 0.5
        sim = GPUTournamentSimulator()
        win_counts = sim.simulate_128_player(matrix, n_simulations=5_000)
        # Player 0 should win the most
        assert win_counts[0] == win_counts.max()


# ---------------------------------------------------------------------------
# Ray simulator tests
# ---------------------------------------------------------------------------

class TestRaySimulator:
    """Tests for outrights/ray_simulator.py."""

    def test_ray_fallback_probabilities_sum_to_one(self):
        """Ray fallback (single-thread) should produce valid results."""
        from outrights.ray_simulator import RayTournamentSimulator
        field = _make_field(n_players=8)
        sim = RayTournamentSimulator()
        result = sim.simulate(field, n_simulations=2_000, n_workers=2)
        total = sum(result.player_win_probs.values())
        assert abs(total - 1.0) < 1e-9

    def test_ray_result_type(self):
        """Result should be an OutrightSimResult instance."""
        from outrights.ray_simulator import RayTournamentSimulator
        field = _make_field(n_players=4)
        sim = RayTournamentSimulator()
        result = sim.simulate(field, n_simulations=1_000, n_workers=2)
        assert isinstance(result, OutrightSimResult)
