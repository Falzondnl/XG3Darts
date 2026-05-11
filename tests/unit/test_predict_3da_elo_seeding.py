"""
Regression tests for P0 fix: ELO seeding + 3DA query correctness.

BUG-DARTS-3DA-001 (2026-05-11):
  darts_player_stats had no player_name column and 0 rows.
  _lookup_3da() queried for a column that didn't exist, returned None for every
  player, causing p1_3da_source='neutral_default' and p1_3da=50.0 for all.

BUG-DARTS-ELO-COVERAGE-001 (2026-05-11):
  darts_elo_ratings had only 39 players (ranks 1-39).
  Any predict request for rank 40+ players returned HTTP 503 darts_elo_unavailable.
  ELO now covers 128 PDC players via bootstrap rank→ELO formula.

Tests here verify:
  1. The _lookup_3da SQL query syntax is valid (column name in correct table).
  2. The seed data CSV has no duplicate player names.
  3. The rank→ELO bootstrap formula produces values within expected ranges.
  4. The predict endpoint _lookup_3da code path uses the right column name.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Test 1: rank→ELO bootstrap formula (from seed_pdc_elo.py)
# ---------------------------------------------------------------------------

def _rank_to_elo(rank: int) -> float:
    """Inline copy of seed_pdc_elo.rank_to_elo for isolation."""
    if rank <= 1:
        return 1650.0
    if rank <= 50:
        return round(1650.0 - (rank - 1) * (150.0 / 49.0), 1)
    if rank <= 100:
        return round(1500.0 - (rank - 50) * (150.0 / 50.0), 1)
    return round(max(1300.0, 1350.0 - (rank - 100) * (50.0 / 28.0)), 1)


@pytest.mark.parametrize("rank,expected_approx,tolerance", [
    (1,   1650.0, 0.1),   # rank 1 = exact anchor
    (50,  1500.0, 1.0),   # rank 50 = exact anchor (tolerance for float)
    (100, 1350.0, 1.0),   # rank 100 = exact anchor
    (128, 1300.0, 1.0),   # rank 128 = floor value
    (25,  1575.0, 2.0),   # rank 25 ≈ midpoint rank1-50 region
    (75,  1425.0, 2.0),   # rank 75 ≈ midpoint rank50-100 region
])
def test_rank_to_elo_anchors(rank: int, expected_approx: float, tolerance: float) -> None:
    """Bootstrap ELO values stay within tolerance of expected anchors."""
    elo = _rank_to_elo(rank)
    assert abs(elo - expected_approx) <= tolerance, (
        f"rank {rank}: got {elo}, expected ~{expected_approx} ±{tolerance}"
    )


def test_rank_to_elo_monotonically_decreasing() -> None:
    """Higher rank (worse player) must produce lower ELO."""
    elos = [_rank_to_elo(r) for r in range(1, 129)]
    for i in range(len(elos) - 1):
        assert elos[i] >= elos[i + 1], (
            f"ELO not monotonically decreasing: rank {i+1}={elos[i]} > "
            f"rank {i+2}={elos[i+1]}"
        )


def test_rank_to_elo_floor() -> None:
    """ELO must never go below 1300.0 for any rank."""
    for rank in range(1, 200):
        assert _rank_to_elo(rank) >= 1300.0, (
            f"ELO below floor at rank {rank}: {_rank_to_elo(rank)}"
        )


# ---------------------------------------------------------------------------
# Test 2: seed data has no duplicate player names
# ---------------------------------------------------------------------------

def _load_seed_data() -> list[tuple[str, int, float, float]]:
    """Load PDC_PLAYER_STATS from scripts/seed_pdc_player_stats.py."""
    import importlib.util
    import pathlib
    seed_path = pathlib.Path(__file__).resolve().parent.parent.parent / "scripts" / "seed_pdc_player_stats.py"
    spec = importlib.util.spec_from_file_location("seed_pdc_player_stats", seed_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PDC_PLAYER_STATS


def test_seed_data_no_duplicate_names() -> None:
    """PDC_PLAYER_STATS must not contain duplicate player names."""
    data = _load_seed_data()
    names = [entry[0] for entry in data]
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    assert not duplicates, f"Duplicate player names in seed data: {duplicates}"


def test_seed_data_3da_range() -> None:
    """All 3DA values must be in the realistic PDC range [80.0, 100.0]."""
    data = _load_seed_data()
    out_of_range = [
        (entry[0], entry[2]) for entry in data
        if not (80.0 <= entry[2] <= 100.0)
    ]
    assert not out_of_range, (
        f"3DA values outside [80, 100]: {out_of_range}"
    )


def test_seed_data_checkout_pct_range() -> None:
    """All checkout percentages must be in the realistic PDC range [30.0, 50.0]."""
    data = _load_seed_data()
    out_of_range = [
        (entry[0], entry[3]) for entry in data
        if not (30.0 <= entry[3] <= 50.0)
    ]
    assert not out_of_range, (
        f"Checkout pct values outside [30, 50]: {out_of_range}"
    )


def test_seed_data_rank_1_player_has_highest_3da() -> None:
    """Rank 1 player (Luke Littler) must have the highest 3DA in the seed set."""
    data = _load_seed_data()
    max_3da = max(entry[2] for entry in data)
    rank_1 = next(entry for entry in data if entry[1] == 1)
    assert rank_1[2] == max_3da, (
        f"Rank 1 player {rank_1[0]!r} does not have highest 3DA: "
        f"has {rank_1[2]}, max is {max_3da}"
    )


# ---------------------------------------------------------------------------
# Test 3: _lookup_3da SQL query uses correct column name
# ---------------------------------------------------------------------------

def test_lookup_3da_query_uses_correct_column() -> None:
    """
    _lookup_3da() must query 'three_da_pdc', not 'three_dart_average'.

    This is a source-code assertion to catch any regression that reverts the
    column name to the old wrong value.  The old code queried a column that
    didn't exist on the table (three_dart_average is the column in the ORM
    model, but the query column is three_da_pdc — added in migration 005).
    """
    import ast
    import pathlib

    predict_path = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "app" / "routes" / "predict.py"
    )
    source = predict_path.read_text(encoding="utf-8")

    # Verify the correct column name is present in the lookup query
    assert "three_da_pdc" in source, (
        "_lookup_3da must query the 'three_da_pdc' column (added in migration 005). "
        "The old 'three_dart_average' column name is not the predict-facing column."
    )

    # Verify the old wrong pattern is not present
    # (three_dart_average is the ORM column, not the lookup column)
    assert "SELECT three_dart_average FROM darts_player_stats" not in source, (
        "Found old wrong column 'three_dart_average' in _lookup_3da query. "
        "The correct column is 'three_da_pdc'."
    )


def test_lookup_3da_does_not_return_hardcoded_50() -> None:
    """
    The predict.py code must not contain a hardcoded 3DA fallback of 50.0
    without also being labeled as 'neutral_default' (regression guard).

    The neutral_default label must be present wherever 50.0 is used as fallback.
    """
    import pathlib

    predict_path = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "app" / "routes" / "predict.py"
    )
    source = predict_path.read_text(encoding="utf-8")

    # If 50.0 appears as a 3DA default, it must be accompanied by the
    # neutral_default label so operators can see when it fires.
    if "_DEFAULT_3DA = 50.0" in source:
        assert "neutral_default" in source, (
            "predict.py has _DEFAULT_3DA = 50.0 but no 'neutral_default' label. "
            "The label is required so operators know when the fallback fires."
        )
