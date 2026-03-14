"""
Tests for elo/elo_pipeline.py

Verifies ELO updates for wins, losses, and draws; K-factor selection by tier;
and multi-pool pipeline routing.
"""
from __future__ import annotations

from datetime import date

import pytest

from elo.elo_pipeline import (
    DEFAULT_RATING,
    ELO_SCALE,
    DartsEloError,
    EloMatchInput,
    EloPipeline,
    EloPool,
    EloRating,
    EloUpdateResult,
    POOL_PDC_MENS,
    POOL_PDC_WOMENS,
    POOL_WDF_OPEN,
    POOL_DEVELOPMENT,
    POOL_TEAM_DOUBLES,
    expected_score,
    k_factor_for_rank,
)

_TODAY = date(2025, 6, 1)


def _make_match(
    p1: str = "player_a",
    p2: str = "player_b",
    result: str = "p1_win",
    pool: str = POOL_PDC_MENS,
    format_code: str = "PDC_WC",
    round_name: str = "Final",
    p1_rank: int | None = None,
    p2_rank: int | None = None,
) -> EloMatchInput:
    return EloMatchInput(
        player1_id=p1,
        player2_id=p2,
        result_type=result,
        pool=pool,
        format_code=format_code,
        round_name=round_name,
        match_date=_TODAY,
        player1_rank=p1_rank,
        player2_rank=p2_rank,
    )


class TestExpectedScore:
    """expected_score() mathematical properties."""

    def test_equal_ratings_returns_half(self) -> None:
        assert abs(expected_score(1500, 1500) - 0.5) < 1e-9

    def test_higher_rated_favoured(self) -> None:
        assert expected_score(1600, 1400) > 0.5

    def test_lower_rated_less_than_half(self) -> None:
        assert expected_score(1400, 1600) < 0.5

    def test_symmetric(self) -> None:
        e_a = expected_score(1600, 1400)
        e_b = expected_score(1400, 1600)
        assert abs(e_a + e_b - 1.0) < 1e-9


class TestKFactorByTier:
    """K-factor selection by pool and ranking tier."""

    def test_elite_rank_1(self) -> None:
        k = k_factor_for_rank(POOL_PDC_MENS, 1)
        assert k == 16.0

    def test_elite_rank_16(self) -> None:
        k = k_factor_for_rank(POOL_PDC_MENS, 16)
        assert k == 16.0

    def test_established_rank_17(self) -> None:
        k = k_factor_for_rank(POOL_PDC_MENS, 17)
        assert k == 24.0

    def test_established_rank_64(self) -> None:
        k = k_factor_for_rank(POOL_PDC_MENS, 64)
        assert k == 24.0

    def test_challenger_rank_65(self) -> None:
        k = k_factor_for_rank(POOL_PDC_MENS, 65)
        assert k == 32.0

    def test_newcomer_no_rank(self) -> None:
        k = k_factor_for_rank(POOL_PDC_MENS, None)
        assert k == 40.0

    def test_newcomer_rank_200(self) -> None:
        k = k_factor_for_rank(POOL_PDC_MENS, 200)
        assert k == 40.0

    def test_development_pool_higher_k(self) -> None:
        k_dev = k_factor_for_rank(POOL_DEVELOPMENT, 1)
        k_pdc = k_factor_for_rank(POOL_PDC_MENS, 1)
        assert k_dev > k_pdc

    def test_unknown_pool_raises(self) -> None:
        with pytest.raises(DartsEloError):
            k_factor_for_rank("nonexistent_pool", 1)


class TestEloPoolUpdate:
    """EloPool single-match update mechanics."""

    def test_winner_rating_increases(self) -> None:
        pool = EloPool(POOL_PDC_MENS)
        match = _make_match(result="p1_win")
        result = pool.update(match)
        assert result.new_rating_p1 > result.old_rating_p1

    def test_loser_rating_decreases(self) -> None:
        pool = EloPool(POOL_PDC_MENS)
        match = _make_match(result="p1_win")
        result = pool.update(match)
        assert result.new_rating_p2 < result.old_rating_p2

    def test_deltas_equal_in_magnitude_for_equal_k(self) -> None:
        """With equal K-factors, delta magnitudes should be equal."""
        pool = EloPool(POOL_PDC_MENS)
        # Both players at default rating, no rank provided → same K
        match = _make_match(result="p1_win", p1_rank=None, p2_rank=None)
        result = pool.update(match)
        assert abs(abs(result.delta_p1) - abs(result.delta_p2)) < 1e-9

    def test_draw_both_ratings_change_symmetrically(self) -> None:
        """Draw with equal ratings → both ratings unchanged (symmetric expected score)."""
        pool = EloPool(POOL_PDC_MENS)
        match = _make_match(
            result="draw",
            format_code="PDC_PL",
            round_name="League Night",
            p1_rank=None,
            p2_rank=None,
        )
        result = pool.update(match)
        # With equal ratings, expected = 0.5, actual = 0.5 → delta = 0
        assert abs(result.delta_p1) < 1e-9
        assert abs(result.delta_p2) < 1e-9

    def test_games_played_increments(self) -> None:
        pool = EloPool(POOL_PDC_MENS)
        match = _make_match()
        pool.update(match)
        rating = pool.get_or_create("player_a")
        assert rating.games_played == 1

    def test_provisional_flag_cleared_after_20_games(self) -> None:
        pool = EloPool(POOL_PDC_MENS)
        for i in range(20):
            pool.update(_make_match(p1="player_a", p2=f"opponent_{i}"))
        r = pool.get_or_create("player_a")
        assert r.provisional is False

    def test_wrong_pool_raises(self) -> None:
        pool = EloPool(POOL_PDC_MENS)
        match = _make_match(pool=POOL_PDC_WOMENS)
        with pytest.raises(DartsEloError):
            pool.update(match)

    def test_default_rating_for_new_player(self) -> None:
        pool = EloPool(POOL_PDC_MENS)
        assert pool.get_rating("brand_new_player") == DEFAULT_RATING

    def test_top_n_returns_sorted(self) -> None:
        pool = EloPool(POOL_PDC_MENS)
        # Give player_a more wins
        for i in range(10):
            pool.update(_make_match(p1="player_a", p2=f"opp_{i}", result="p1_win"))
        top5 = pool.top_n(5)
        assert top5[0].player_id == "player_a"


class TestEloPipeline:
    """EloPipeline routing to correct pools."""

    def test_process_match_routes_correctly(self) -> None:
        pipeline = EloPipeline()
        match = _make_match(pool=POOL_PDC_MENS)
        result = pipeline.process_match(match)
        assert isinstance(result, EloUpdateResult)

    def test_process_batch_returns_all_results(self) -> None:
        pipeline = EloPipeline()
        matches = [
            _make_match(p1="p1", p2=f"opp_{i}", pool=POOL_PDC_MENS)
            for i in range(5)
        ]
        results = pipeline.process_batch(matches)
        assert len(results) == 5

    def test_different_pools_are_independent(self) -> None:
        pipeline = EloPipeline()
        # Update pdc_mens pool
        pipeline.process_match(_make_match(pool=POOL_PDC_MENS))
        # womens pool should not be affected
        womens_rating = pipeline.get_rating(POOL_PDC_WOMENS, "player_a")
        assert womens_rating == DEFAULT_RATING

    def test_unknown_pool_raises(self) -> None:
        pipeline = EloPipeline()
        with pytest.raises(DartsEloError):
            pipeline.get_rating("bad_pool", "player_a")

    def test_pool_stats_has_all_pools(self) -> None:
        pipeline = EloPipeline()
        stats = pipeline.pool_stats()
        for pool in (
            POOL_PDC_MENS,
            POOL_PDC_WOMENS,
            POOL_WDF_OPEN,
            POOL_DEVELOPMENT,
            POOL_TEAM_DOUBLES,
        ):
            assert pool in stats
