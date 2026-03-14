"""
End-to-end validation tests for all 12 QA gates.

These are the canonical acceptance tests that MUST pass before any price
is published.  Every gate tests a specific architectural invariant.

G1  — Markov probability sums to 1.00 ± 0.001
G2  — P1/P2 swap produces consistent prices ± 0.01%
G3  — hold_p1 + break_p2 = 1.0 ± 0.001
G4  — Format loaded from registry, not hardcoded
G5  — Starter confidence flag always set
G6  — Draw market only opens if format.draw_enabled = True
G7  — Two-clear-legs engine used for World Matchplay
G8  — Regime loaded from coverage_regimes table (not hardcoded)
G9  — Data sufficiency gates enforced for all props
G10 — Market-specific calibrator (not global)
G11 — 5-factor margin computed
G12 — World Cup uses doubles engine
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# G1: Markov probability sums to 1.00 ± 0.001
# ---------------------------------------------------------------------------


def test_g1_markov_probability_sums_to_one() -> None:
    """G1: Markov chain total probability = 1.00 ± 0.001."""
    from engines.leg_layer.markov_chain import DartsMarkovChain, MARKOV_TOTAL_TOLERANCE
    from engines.leg_layer.visit_distributions import ConditionalVisitDistribution

    chain = DartsMarkovChain()

    # Build a realistic visit distribution — total must be within tolerance
    dist = ConditionalVisitDistribution(
        player_id="test_player_g1",
        score_band="open",
        stage=False,
        short_format=False,
        throw_first=True,
        visit_probs={
            60: 0.05,
            80: 0.10,
            100: 0.25,
            120: 0.30,
            140: 0.15,
            160: 0.05,
            180: 0.02,
        },
        bust_prob=0.08,
        data_source="derived",
        n_observations=500,
        confidence=0.8,
    )

    total = sum(dist.visit_probs.values()) + dist.bust_prob
    assert abs(total - 1.0) <= MARKOV_TOTAL_TOLERANCE, (
        f"G1 FAIL: Markov distribution total = {total:.6f}, "
        f"expected 1.0 ± {MARKOV_TOTAL_TOLERANCE}"
    )

    ok = chain.validate_markov_totals(dist)
    assert ok, "G1 FAIL: validate_markov_totals() returned False"


def test_g1_multiple_bands_all_sum_to_one() -> None:
    """G1: All score bands in the hierarchical model sum to 1.00 ± 0.001."""
    from engines.leg_layer.visit_distributions import (
        HierarchicalVisitDistributionModel,
        BAND_NAMES,
    )
    from engines.leg_layer.markov_chain import MARKOV_TOTAL_TOLERANCE

    model = HierarchicalVisitDistributionModel()
    for band in BAND_NAMES:
        dist = model.get_distribution(
            player_id="qa_gate_g1",
            score_band=band,
            stage=False,
            short_format=False,
            throw_first=True,
            three_da=92.0,
        )
        total = sum(dist.visit_probs.values()) + dist.bust_prob
        assert abs(total - 1.0) <= MARKOV_TOTAL_TOLERANCE, (
            f"G1 FAIL: band={band!r}, total={total:.6f}"
        )


# ---------------------------------------------------------------------------
# G2: P1/P2 swap produces consistent prices ± 0.01%
# ---------------------------------------------------------------------------


def test_g2_p1_p2_swap_symmetry() -> None:
    """G2: P1/P2 swap produces consistent prices ±0.01%."""
    from engines.leg_layer.markov_chain import DartsMarkovChain
    from engines.leg_layer.visit_distributions import HierarchicalVisitDistributionModel, BAND_NAMES

    model = HierarchicalVisitDistributionModel()
    chain = DartsMarkovChain()

    # Build two distinct players across all bands
    p1_dists = {
        band: model.get_distribution("player_A", band, False, False, True, 98.0)
        for band in BAND_NAMES
    }
    p2_dists = {
        band: model.get_distribution("player_B", band, False, False, False, 85.0)
        for band in BAND_NAMES
    }

    hb_normal = chain.break_probability(
        p1_visit_dists=p1_dists,
        p2_visit_dists=p2_dists,
        p1_id="player_A",
        p2_id="player_B",
        p1_three_da=98.0,
        p2_three_da=85.0,
    )

    hb_swapped = chain.break_probability(
        p1_visit_dists=p2_dists,
        p2_visit_dists=p1_dists,
        p1_id="player_B",
        p2_id="player_A",
        p1_three_da=85.0,
        p2_three_da=98.0,
    )

    # Symmetry: p1_hold(A,B) == p2_hold(B,A)
    tolerance = 0.0001  # ±0.01%
    assert abs(hb_normal.p1_hold - hb_swapped.p2_hold) <= tolerance, (
        f"G2 FAIL: p1_hold(A,B)={hb_normal.p1_hold:.6f} != "
        f"p2_hold(B,A)={hb_swapped.p2_hold:.6f}"
    )
    assert abs(hb_normal.p2_hold - hb_swapped.p1_hold) <= tolerance, (
        f"G2 FAIL: p2_hold(A,B)={hb_normal.p2_hold:.6f} != "
        f"p1_hold(B,A)={hb_swapped.p1_hold:.6f}"
    )


# ---------------------------------------------------------------------------
# G3: hold_p1 + break_p2 = 1.0 ± 0.001
# ---------------------------------------------------------------------------


def test_g3_hold_break_consistency() -> None:
    """G3: hold_p1 + break_p2 = 1.0 ± 0.001."""
    from engines.leg_layer.markov_chain import DartsMarkovChain
    from engines.leg_layer.visit_distributions import HierarchicalVisitDistributionModel, BAND_NAMES

    model = HierarchicalVisitDistributionModel()
    chain = DartsMarkovChain()

    p1_dists = {
        band: model.get_distribution("p1_g3", band, False, False, True, 95.0)
        for band in BAND_NAMES
    }
    p2_dists = {
        band: model.get_distribution("p2_g3", band, False, False, False, 88.0)
        for band in BAND_NAMES
    }

    hb = chain.break_probability(
        p1_visit_dists=p1_dists,
        p2_visit_dists=p2_dists,
        p1_id="p1_g3",
        p2_id="p2_g3",
        p1_three_da=95.0,
        p2_three_da=88.0,
    )

    tol = 0.001

    # Core G3 constraints
    assert abs(hb.p1_hold + hb.p2_break - 1.0) <= tol, (
        f"G3 FAIL: p1_hold={hb.p1_hold:.6f} + p2_break={hb.p2_break:.6f} = "
        f"{hb.p1_hold + hb.p2_break:.6f} != 1.0"
    )
    assert abs(hb.p2_hold + hb.p1_break - 1.0) <= tol, (
        f"G3 FAIL: p2_hold={hb.p2_hold:.6f} + p1_break={hb.p1_break:.6f} = "
        f"{hb.p2_hold + hb.p1_break:.6f} != 1.0"
    )

    # validate() must not raise
    hb.validate()

    # consistency_check() must return True
    assert hb.consistency_check(tol=tol), "G3 FAIL: consistency_check() returned False"


def test_g3_direct_construction_validates() -> None:
    """G3: HoldBreakProbabilities.validate() enforces constraints."""
    from engines.leg_layer.markov_chain import HoldBreakProbabilities

    hb = HoldBreakProbabilities(
        p1_hold=0.60,
        p2_break=0.40,  # p1_hold + p2_break = 1.0 ✓
        p2_hold=0.55,
        p1_break=0.45,  # p2_hold + p1_break = 1.0 ✓
    )
    hb.validate()  # must not raise

    # Violated constraint must raise
    with pytest.raises(ValueError):
        bad = HoldBreakProbabilities(
            p1_hold=0.70,
            p2_break=0.40,  # sum = 1.10 — violation
            p2_hold=0.55,
            p1_break=0.45,
        )
        bad.validate()


# ---------------------------------------------------------------------------
# G4: Format loaded from registry, not hardcoded
# ---------------------------------------------------------------------------


def test_g4_format_from_registry() -> None:
    """G4: Format loaded from registry, not hardcoded."""
    from competition.format_registry import get_format, list_formats, DartsFormatError

    # Every known format must be loadable from the registry by code
    codes = list_formats()
    assert len(codes) > 0, "G4 FAIL: format registry is empty"

    for code in codes:
        fmt = get_format(code)
        # Validate structural integrity
        assert fmt.code == code, f"G4 FAIL: code mismatch {fmt.code!r} != {code!r}"
        assert len(fmt.per_round) > 0, f"G4 FAIL: {code} has no rounds defined"
        assert fmt.ecosystem in {
            "pdc_mens", "pdc_womens", "wdf_open", "development", "team_doubles"
        }, f"G4 FAIL: invalid ecosystem {fmt.ecosystem!r} for {code}"

    # Unregistered code must raise DartsFormatError (not KeyError)
    with pytest.raises(DartsFormatError):
        get_format("NONEXISTENT_FORMAT_CODE_XYZ")


def test_g4_key_formats_are_registered() -> None:
    """G4: All mandatory competition codes are present in the registry."""
    from competition.format_registry import get_format

    mandatory = [
        "PDC_WC", "PDC_PL", "PDC_WM", "PDC_GP", "PDC_GS",
        "PDC_WCUP", "WDF_WC",
    ]
    for code in mandatory:
        fmt = get_format(code)
        assert fmt is not None, f"G4 FAIL: mandatory format {code!r} not registered"


# ---------------------------------------------------------------------------
# G5: Starter confidence flag always set
# ---------------------------------------------------------------------------


def test_g5_starter_confidence_flag() -> None:
    """G5: Starter confidence flag always set."""
    from engines.state_layer.starter_inference import StarterInferenceEngine

    engine = StarterInferenceEngine()

    # Case 1: Official feed confirmation — confidence must be 1.0
    result = engine.infer_starter(
        leg_number=1,
        players=("player_a", "player_b"),
        alternating_starts=True,
        feed_starter_id="player_a",
        feed_confidence=1.0,
        feed_source="dartconnect",
    )
    assert result.confidence is not None, "G5 FAIL: confidence is None for dartconnect source"
    assert result.confidence == 1.0, (
        f"G5 FAIL: dartconnect confidence={result.confidence}, expected 1.0"
    )
    assert result.is_confirmed is True, "G5 FAIL: is_confirmed should be True for dartconnect"

    # Case 2: Inferred — confidence is in (0.5, 1.0)
    result_inferred = engine.infer_starter(
        leg_number=3,
        players=("player_a", "player_b"),
        alternating_starts=True,
        previous_starters=[],
        feed_starter_id=None,
    )
    assert result_inferred.confidence is not None, (
        "G5 FAIL: confidence is None when inferring"
    )
    # Confidence must always be a float — never None
    assert isinstance(result_inferred.confidence, float), (
        "G5 FAIL: inferred confidence is not float"
    )


def test_g5_match_starters_all_have_confidence() -> None:
    """G5: infer_match_starters() always assigns confidence to every leg."""
    from engines.state_layer.starter_inference import StarterInferenceEngine

    engine = StarterInferenceEngine()

    starters = engine.infer_match_starters(
        num_legs=11,  # typical match length
        players=("player_x", "player_y"),
        alternating_starts=True,
        confirmed_starters={1: ("player_x", "dartconnect", 1.0)},
    )

    assert len(starters) == 11, f"G5 FAIL: expected 11 starter records, got {len(starters)}"
    for info in starters:
        assert info.confidence is not None, (
            f"G5 FAIL: confidence is None for leg {info.leg_number}"
        )
        assert 0.0 <= info.confidence <= 1.0, (
            f"G5 FAIL: confidence={info.confidence} for leg {info.leg_number} out of range"
        )


# ---------------------------------------------------------------------------
# G6: Draw market only opens if format.draw_enabled = True
# ---------------------------------------------------------------------------


def test_g6_draw_market_only_for_draw_formats() -> None:
    """G6: Draw market only opens if format.draw_enabled=True."""
    from competition.format_registry import get_format
    from competition.draw_result import DartsMatchResult, DartsResultError

    # Premier League group stage — draws enabled
    pl_fmt = get_format("PDC_PL")
    pl_round = pl_fmt.get_round("League Night")
    assert pl_round.draw_enabled is True, "G6 setup: PL League Night should allow draws"

    # A draw result must be valid for PDC_PL
    draw_result = DartsMatchResult(
        result_type="draw",
        p1_score=6,
        p2_score=6,
        format_code="PDC_PL",
        round_name="League Night",
    )
    assert draw_result.is_draw, "G6 FAIL: PDC_PL draw result not recognised as draw"

    # World Championship final — draws NOT enabled
    wc_fmt = get_format("PDC_WC")
    wc_round = wc_fmt.get_round("Final")
    assert wc_round.draw_enabled is False, "G6 setup: WC Final should not allow draws"

    # A draw result must be INVALID for PDC_WC
    with pytest.raises((DartsResultError, Exception)):
        invalid_draw = DartsMatchResult(
            result_type="draw",
            p1_score=6,
            p2_score=6,
            format_code="PDC_WC",
            round_name="Final",
        )


def test_g6_grand_slam_group_stage_draws() -> None:
    """G6: Grand Slam group stage also allows draws."""
    from competition.format_registry import get_format
    from competition.draw_result import DartsMatchResult, RESULT_DRAW

    gs_fmt = get_format("PDC_GS")
    gs_round = gs_fmt.get_round("Group Stage")
    assert gs_round.draw_enabled is True, "G6 FAIL: Grand Slam Group Stage should allow draws"
    assert gs_fmt.allows_draw() is True, "G6 FAIL: PDC_GS.allows_draw() should return True"

    # Draw result valid for Grand Slam Group Stage
    draw_r = DartsMatchResult(
        result_type=RESULT_DRAW,
        p1_score=4,
        p2_score=4,
        format_code="PDC_GS",
        round_name="Group Stage",
    )
    assert draw_r.is_draw

    # Knockout rounds do NOT allow draws
    ks_round = gs_fmt.get_round("Final")
    assert ks_round.draw_enabled is False, "G6 FAIL: Grand Slam Final should not allow draws"


# ---------------------------------------------------------------------------
# G7: Two-clear-legs engine used for World Matchplay
# ---------------------------------------------------------------------------


def test_g7_two_clear_engine_activated() -> None:
    """G7: Two-clear-legs engine used for World Matchplay."""
    from competition.format_registry import get_format
    from engines.match_layer.world_matchplay_engine import WorldMatchplayEngine
    from engines.leg_layer.markov_chain import HoldBreakProbabilities

    wm_fmt = get_format("PDC_WM")
    wm_final = wm_fmt.get_round("Final")

    # The Final uses two_clear_legs
    assert wm_final.two_clear_legs is True, (
        "G7 FAIL: World Matchplay Final should have two_clear_legs=True"
    )

    # WorldMatchplayEngine must price this format
    engine = WorldMatchplayEngine()
    hb = HoldBreakProbabilities(
        p1_hold=0.62,
        p2_break=0.38,
        p2_hold=0.58,
        p1_break=0.42,
    )

    result = engine.price_match(
        round_fmt=wm_final,
        hb=hb,
        p1_starts=True,
    )

    assert result is not None, "G7 FAIL: WorldMatchplayEngine returned None"
    # price_match returns a dict with keys p1_win, p2_win
    p1_win = result["p1_win"]
    p2_win = result["p2_win"]
    assert abs(p1_win + p2_win - 1.0) <= 0.001, (
        f"G7 FAIL: p1_win + p2_win = {p1_win + p2_win:.6f} != 1.0"
    )
    assert 0.0 < p1_win < 1.0, "G7 FAIL: p1_win out of (0, 1)"


def test_g7_non_matchplay_format_does_not_use_two_clear() -> None:
    """G7: Standard formats do NOT have two_clear_legs."""
    from competition.format_registry import get_format

    for code in ["PDC_WC", "PDC_GS", "PDC_UK"]:
        fmt = get_format(code)
        for round_name, rnd in fmt.per_round.items():
            assert not rnd.two_clear_legs, (
                f"G7 FAIL: {code}/{round_name} should not have two_clear_legs"
            )


# ---------------------------------------------------------------------------
# G8: Regime from coverage_regimes table
# ---------------------------------------------------------------------------


def test_g8_regime_from_coverage_table() -> None:
    """G8: Regime from coverage_regimes table."""
    from data.coverage_regime import (
        CoverageSignals,
        detect_regime,
        REGIME_R0,
        REGIME_R1,
        REGIME_R2,
    )

    # R2: full visit data via DartConnect (>= 10 legs)
    signals_r2 = CoverageSignals(
        player_id="test_r2",
        competition_id=None,
        has_dartconnect_data=True,
        has_mastercaller_data=False,
        has_dartsorakel_stats=True,
        has_pdc_match_stats=True,
        dartconnect_leg_count=50,
        match_count=30,
    )
    result_r2 = detect_regime(signals_r2)
    assert result_r2.regime == REGIME_R2, (
        f"G8 FAIL: expected R2 for full visit data, got {result_r2.regime!r}"
    )

    # R1: only match-level stats, no visit data
    signals_r1 = CoverageSignals(
        player_id="test_r1",
        competition_id=None,
        has_dartconnect_data=False,
        has_mastercaller_data=False,
        has_dartsorakel_stats=True,
        has_pdc_match_stats=True,
        dartconnect_leg_count=0,
        match_count=30,
    )
    result_r1 = detect_regime(signals_r1)
    assert result_r1.regime == REGIME_R1, (
        f"G8 FAIL: expected R1 for match stats, got {result_r1.regime!r}"
    )

    # R0: result only
    signals_r0 = CoverageSignals(
        player_id="test_r0",
        competition_id=None,
        has_dartconnect_data=False,
        has_mastercaller_data=False,
        has_dartsorakel_stats=False,
        has_pdc_match_stats=False,
        dartconnect_leg_count=0,
        match_count=10,
    )
    result_r0 = detect_regime(signals_r0)
    assert result_r0.regime == REGIME_R0, (
        f"G8 FAIL: expected R0 for result-only, got {result_r0.regime!r}"
    )


def test_g8_regime_values_are_canonical() -> None:
    """G8: REGIME constants are exactly 'R0', 'R1', 'R2'."""
    from data.coverage_regime import REGIME_R0, REGIME_R1, REGIME_R2, VALID_REGIMES

    assert REGIME_R0 == "R0", f"G8 FAIL: REGIME_R0={REGIME_R0!r}"
    assert REGIME_R1 == "R1", f"G8 FAIL: REGIME_R1={REGIME_R1!r}"
    assert REGIME_R2 == "R2", f"G8 FAIL: REGIME_R2={REGIME_R2!r}"
    assert {"R0", "R1", "R2"} == set(VALID_REGIMES), "G8 FAIL: VALID_REGIMES mismatch"


# ---------------------------------------------------------------------------
# G9: Data sufficiency gates enforced for all props
# ---------------------------------------------------------------------------


def test_g9_data_gate_enforcement() -> None:
    """G9: Data sufficiency gates enforced for all props."""
    from props.data_sufficiency_gate import DataSufficiencyGate

    gate = DataSufficiencyGate()

    # 180 prop: requires minimum legs observed (50)
    ok_fail, reason_fail = gate.can_open_market(
        market_family="props_180",
        player_id="test_player",
        regime=0,   # R0 — result only
        stats={
            "legs_observed": 0,   # insufficient
            "confidence": 0.5,
        },
    )
    assert not ok_fail, f"G9 FAIL: gate should block with 0 legs but returned ok=True"

    # Sufficient data — must pass
    ok_pass, _ = gate.can_open_market(
        market_family="props_180",
        player_id="test_player",
        regime=1,
        stats={
            "legs_observed": 100,
            "pct_180_ewm": 0.042,
            "confidence": 0.80,
        },
    )
    assert ok_pass, f"G9 FAIL: gate blocked despite sufficient data: {_!r}"


def test_g9_all_gate_families_defined() -> None:
    """G9: DATA_SUFFICIENCY_GATES defines at least the four canonical prop families."""
    from props.data_sufficiency_gate import DATA_SUFFICIENCY_GATES

    required = {"props_180", "props_checkout", "props_double_segment", "props_next_visit"}
    defined = set(DATA_SUFFICIENCY_GATES.keys())
    missing = required - defined
    assert not missing, (
        f"G9 FAIL: DATA_SUFFICIENCY_GATES missing families: {missing}"
    )


def test_g9_r2_required_gates_block_lower_regimes() -> None:
    """G9: Markets requiring R2 are blocked in R0 and R1."""
    from props.data_sufficiency_gate import DataSufficiencyGate

    gate = DataSufficiencyGate()

    # props_double_segment requires R2
    for regime in (0, 1):
        ok, reason = gate.can_open_market(
            market_family="props_double_segment",
            player_id="test_player",
            regime=regime,
            stats={"legs_observed": 200, "double_hit_rate": 0.42, "eb_double_accuracy": 0.5, "confidence": 0.9},
        )
        assert not ok, (
            f"G9 FAIL: props_double_segment should be blocked at R{regime}, got ok=True"
        )


# ---------------------------------------------------------------------------
# G10: Market-specific calibrator (not global)
# ---------------------------------------------------------------------------


def test_g10_market_specific_calibrator() -> None:
    """G10: Market-specific calibrator (not global)."""
    from calibration.market_calibrators import MarketCalibrationRegistry, MARKET_FAMILIES

    registry = MarketCalibrationRegistry()

    # Each market family must have its own calibrator instance
    seen_ids: set[int] = set()
    for family in MARKET_FAMILIES:
        calibrator = registry.get_calibrator(family)
        assert calibrator is not None, (
            f"G10 FAIL: no calibrator registered for market family {family!r}"
        )
        cal_id = id(calibrator)
        assert cal_id not in seen_ids, (
            f"G10 FAIL: {family!r} shares calibrator instance with another market family"
        )
        seen_ids.add(cal_id)


def test_g10_unknown_family_raises() -> None:
    """G10: Requesting an unknown market family raises DartsCalibrationError."""
    from calibration.market_calibrators import MarketCalibrationRegistry, DartsCalibrationError

    registry = MarketCalibrationRegistry()
    with pytest.raises(DartsCalibrationError):
        registry.get_calibrator("nonexistent_family_xyz")


# ---------------------------------------------------------------------------
# G11: 5-factor margin computed
# ---------------------------------------------------------------------------


def test_g11_5_factor_margin() -> None:
    """G11: 5-factor margin computed."""
    from margin.blending_engine import DartsMarginEngine

    engine = DartsMarginEngine()

    # Use regime=2 (R2) to avoid capping
    margin = engine.compute_margin(
        base_margin=0.04,
        regime=2,
        starter_confidence=0.90,
        source_confidence=0.95,
        model_agreement=0.90,
        market_liquidity="high",
        ecosystem="pdc_mens",
    )

    assert isinstance(margin, float), f"G11 FAIL: expected float, got {type(margin).__name__}"
    assert 0.0 < margin <= 0.15, f"G11 FAIL: margin={margin:.4f} out of (0, 0.15]"
    # Margin should be larger than base due to uncertainty widening
    assert margin >= 0.04, f"G11 FAIL: margin={margin:.4f} < base_margin=0.04"


def test_g11_all_five_factors_contribute() -> None:
    """G11: All 5 factors individually affect the margin."""
    from margin.blending_engine import DartsMarginEngine

    engine = DartsMarginEngine()

    # Baseline: all factors at neutral
    base = engine.compute_margin(
        base_margin=0.04,
        regime=2,
        starter_confidence=1.0,
        source_confidence=1.0,
        model_agreement=1.0,
        market_liquidity="high",
        ecosystem="pdc_mens",
    )

    # Factor 1: Regime
    m_regime = engine.compute_margin(
        base_margin=0.04,
        regime=0,  # R0 → 30% wider
        starter_confidence=1.0,
        source_confidence=1.0,
        model_agreement=1.0,
        market_liquidity="high",
        ecosystem="pdc_mens",
    )
    assert m_regime > base, "G11 FAIL: regime factor has no effect"

    # Factor 2: Starter confidence
    m_starter = engine.compute_margin(
        base_margin=0.04,
        regime=2,
        starter_confidence=0.5,  # uncertain → wider
        source_confidence=1.0,
        model_agreement=1.0,
        market_liquidity="high",
        ecosystem="pdc_mens",
    )
    assert m_starter > base, "G11 FAIL: starter_confidence factor has no effect"

    # Factor 3: Source confidence
    m_source = engine.compute_margin(
        base_margin=0.04,
        regime=2,
        starter_confidence=1.0,
        source_confidence=0.5,  # uncertain → wider
        model_agreement=1.0,
        market_liquidity="high",
        ecosystem="pdc_mens",
    )
    assert m_source > base, "G11 FAIL: source_confidence factor has no effect"

    # Factor 4: Model agreement
    m_agreement = engine.compute_margin(
        base_margin=0.04,
        regime=2,
        starter_confidence=1.0,
        source_confidence=1.0,
        model_agreement=0.5,  # disagreement → wider
        market_liquidity="high",
        ecosystem="pdc_mens",
    )
    assert m_agreement > base, "G11 FAIL: model_agreement factor has no effect"

    # Factor 5: Ecosystem / liquidity (women's / development → wider)
    m_eco = engine.compute_margin(
        base_margin=0.04,
        regime=2,
        starter_confidence=1.0,
        source_confidence=1.0,
        model_agreement=1.0,
        market_liquidity="low",
        ecosystem="pdc_womens",
    )
    assert m_eco > base, "G11 FAIL: ecosystem/liquidity factor has no effect"


# ---------------------------------------------------------------------------
# G12: World Cup uses doubles engine
# ---------------------------------------------------------------------------


def test_g12_world_cup_doubles_engine() -> None:
    """G12: World Cup uses doubles engine."""
    from competition.format_registry import get_format
    from engines.doubles.world_cup_pricer import WorldCupPricer
    from engines.doubles.team_visit_model import DoublesTeam

    wcup_fmt = get_format("PDC_WCUP")

    # Verify doubles rounds exist
    doubles_rounds = [
        r for name, r in wcup_fmt.per_round.items()
        if r.format_type == "doubles"
    ]
    assert len(doubles_rounds) > 0, "G12 FAIL: PDC_WCUP has no doubles rounds"

    # WorldCupPricer must produce valid prices
    pricer = WorldCupPricer()
    team1 = DoublesTeam(
        team_id="england",
        player_a_id="england_a",
        player_b_id="england_b",
        player_a_three_da=99.0,
        player_b_three_da=93.0,
    )
    team2 = DoublesTeam(
        team_id="netherlands",
        player_a_id="netherlands_a",
        player_b_id="netherlands_b",
        player_a_three_da=101.0,
        player_b_three_da=89.0,
    )

    doubles_round = wcup_fmt.get_round("Round 1 Doubles")
    result = pricer.price_doubles_leg(
        team1=team1,
        team2=team2,
        round_fmt=doubles_round,
        team1_starts=True,
    )

    assert result is not None, "G12 FAIL: WorldCupPricer returned None"
    assert abs(result.p1_win + result.p2_win - 1.0) <= 0.001, (
        f"G12 FAIL: doubles p1_win + p2_win = {result.p1_win + result.p2_win:.6f}"
    )
    assert 0.0 < result.p1_win < 1.0, "G12 FAIL: doubles p1_win out of (0, 1)"


def test_g12_world_cup_format_is_team_doubles_ecosystem() -> None:
    """G12: PDC_WCUP is in the team_doubles ecosystem."""
    from competition.format_registry import get_format

    wcup = get_format("PDC_WCUP")
    assert wcup.ecosystem == "team_doubles", (
        f"G12 FAIL: PDC_WCUP ecosystem={wcup.ecosystem!r}, expected 'team_doubles'"
    )
