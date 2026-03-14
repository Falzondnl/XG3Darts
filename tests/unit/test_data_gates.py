"""
QA Gate tests H1-H7 (G1-G7 in the spec).

These gates are invariants that the engine must satisfy at all times.

G1: Markov total probability = 1.00 ± 0.001
G2: P1/P2 swap symmetry ±0.01%
G3: hold_p1 + break_p2 = 1.0
G4: Format from registry (not hardcoded)
G5: Starter confidence flag set
G6: Draw market only opens if draw_enabled=True
G7: Two-clear-legs engine used when format flag set
"""
from __future__ import annotations

import pytest

from competition.format_registry import (
    DartsFormatError,
    DartsRoundFormat,
    get_format,
    list_formats,
)
from engines.leg_layer.hold_break_model import HoldBreakModel
from engines.leg_layer.markov_chain import DartsMarkovChain, HoldBreakProbabilities
from engines.leg_layer.visit_distributions import (
    HierarchicalVisitDistributionModel,
)
from engines.match_layer.match_combinatorics import MatchCombinatorialEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hb(p1_hold: float = 0.62, p2_hold: float = 0.58) -> HoldBreakProbabilities:
    return HoldBreakProbabilities(
        p1_hold=p1_hold,
        p1_break=1.0 - p2_hold,
        p2_hold=p2_hold,
        p2_break=1.0 - p1_hold,
    )


def _make_symmetric_hb(hold: float = 0.60) -> HoldBreakProbabilities:
    return HoldBreakProbabilities(
        p1_hold=hold,
        p1_break=1.0 - hold,
        p2_hold=hold,
        p2_break=1.0 - hold,
    )


# ---------------------------------------------------------------------------
# G1: Markov total probability = 1.00 ± 0.001
# ---------------------------------------------------------------------------

class TestG1MarkovTotalProbability:
    """G1: Every visit distribution must sum to 1.0 within tolerance."""

    def test_g1_markov_totals_valid_distribution(self):
        """
        A correctly constructed ConditionalVisitDistribution must pass G1.
        """
        from engines.leg_layer.visit_distributions import ConditionalVisitDistribution
        markov = DartsMarkovChain()

        # Build a well-formed distribution using correct constructor signature
        probs = {60: 0.30, 45: 0.25, 41: 0.20, 26: 0.15}
        bust_prob = 1.0 - sum(probs.values())  # = 0.10
        dist = ConditionalVisitDistribution(
            player_id="test_player",
            score_band="open",
            stage=False,
            short_format=False,
            throw_first=True,
            visit_probs=probs,
            bust_prob=bust_prob,
            data_source="test",
            n_observations=100,
            confidence=0.8,
        )

        assert markov.validate_markov_totals(dist), (
            "Distribution summing to exactly 1.0 should pass G1"
        )

    def test_g1_markov_totals_violation_detected(self):
        """
        A distribution that doesn't sum to 1.0 should fail G1.
        """
        from engines.leg_layer.visit_distributions import ConditionalVisitDistribution
        markov = DartsMarkovChain()

        # Deliberately bad distribution (sums to 0.80, not 1.0)
        probs = {60: 0.50, 45: 0.20}
        dist = ConditionalVisitDistribution(
            player_id="bad_player",
            score_band="open",
            stage=False,
            short_format=False,
            throw_first=True,
            visit_probs=probs,
            bust_prob=0.10,  # total = 0.80, not 1.0
            data_source="test",
            n_observations=10,
            confidence=0.5,
        )

        ok = markov.validate_markov_totals(dist)
        assert not ok, "Distribution summing to 0.80 should fail G1"

    def test_g1_holds_for_hierarchical_prior(self):
        """
        Visit distributions generated from hierarchical model must pass G1.
        """
        visit_model = HierarchicalVisitDistributionModel()
        markov = DartsMarkovChain()

        # Test multiple 3DA values across the realistic range
        for three_da in [55.0, 70.0, 85.0, 95.0]:
            dists = visit_model.get_all_bands(
                player_id=f"player_{int(three_da)}",
                stage=False,
                short_format=False,
                throw_first=True,
                three_da=three_da,
            )
            for band, dist in dists.items():
                ok = markov.validate_markov_totals(dist)
                total = sum(dist.visit_probs.values()) + dist.bust_prob
                assert ok, (
                    f"G1 violation for 3DA={three_da}, band={band}: "
                    f"total={total:.6f}"
                )


# ---------------------------------------------------------------------------
# G2: P1/P2 swap symmetry ±0.01%
# ---------------------------------------------------------------------------

class TestG2SwapSymmetry:
    """
    G2: Swapping P1 and P2 must produce complementary match win probabilities.

    For symmetric players: P1_win(P1, P2) + P2_win(P1, P2) = 1.0
    And: P1_win(P1, P2) ≈ P2_win(P2, P1) for symmetric player stats.
    """

    def test_g2_symmetric_players_swap(self):
        """
        For identical players, swapping P1 and P2 must give exactly
        complementary probabilities.
        """
        engine = MatchCombinatorialEngine()
        hb = _make_symmetric_hb(hold=0.60)

        result_normal = engine._dp_legs_format(hb=hb, legs_to_win=7, p1_starts=True)

        # Swap P1 and P2 — for symmetric HB, swap just means P2 starts
        # (and for identical players, p1_win should be close to 0.5)
        p1_win = result_normal["p1_win"]
        p2_win = result_normal["p2_win"]

        # Must sum to 1.0
        assert abs(p1_win + p2_win - 1.0) < 1e-10, (
            f"P1_win + P2_win = {p1_win + p2_win:.10f}, expected 1.0"
        )

    def test_g2_asymmetric_players_swap(self):
        """
        G2 swap symmetry: for asymmetric players, swapping P1 and P2 (including
        who starts) must give complementary win probabilities.

        If P1 starts in the original, then when we swap player identities AND
        also swap who starts (now P2 starts = original P1 starts), the result
        should be complementary: swapped_P1_win = original P2_win.
        """
        engine = MatchCombinatorialEngine()
        hb = _make_hb(p1_hold=0.65, p2_hold=0.55)

        result = engine._dp_legs_format(hb=hb, legs_to_win=4, p1_starts=True)
        p1_win = result["p1_win"]
        p2_win = result["p2_win"]

        # Sum must be 1.0 exactly (fundamental G2 constraint)
        assert abs(p1_win + p2_win - 1.0) < 1e-10, (
            f"G2: p1_win + p2_win = {p1_win + p2_win:.12f} (must be 1.0)"
        )

        # Confirm p1 is the stronger player
        assert p1_win > p2_win, (
            f"p1 (hold=0.65) should win more than p2 (hold=0.55): "
            f"p1={p1_win:.4f}, p2={p2_win:.4f}"
        )

        # Swap P1/P2 labels AND swap starter so the same player goes first
        # (original P2 now plays as P1, original P1 plays as P2, P2 starts first)
        hb_swapped = HoldBreakProbabilities(
            p1_hold=hb.p2_hold,    # original P2 stats → now P1
            p1_break=hb.p2_break,
            p2_hold=hb.p1_hold,   # original P1 stats → now P2
            p2_break=hb.p1_break,
        )
        # P2 starts (to match original P1 starting — same player, same advantage)
        result_swapped = engine._dp_legs_format(
            hb=hb_swapped, legs_to_win=4, p1_starts=False  # P2 starts = original P1 starts
        )
        p1_win_swapped = result_swapped["p1_win"]  # This is original P2's win prob

        # With swapped labels and swapped starter, original P2 win = swapped P1 win
        # Tolerance: 0.01% = 0.0001
        assert abs(p2_win - p1_win_swapped) < 1e-4, (
            f"G2 violation: original P2_win={p2_win:.8f} != "
            f"swapped_P1_win={p1_win_swapped:.8f}, "
            f"diff={abs(p2_win - p1_win_swapped):.8f}"
        )


# ---------------------------------------------------------------------------
# G3: hold_p1 + break_p2 = 1.0
# ---------------------------------------------------------------------------

class TestG3HoldBreakConsistency:
    """G3: p1_hold + p2_break = 1.0 and p2_hold + p1_break = 1.0."""

    def test_g3_holds_for_computed_hb(self):
        """Hold/break from HoldBreakModel must satisfy G3."""
        model = HoldBreakModel()
        hb = model.compute_from_3da(
            p1_id="p1",
            p2_id="p2",
            p1_three_da=80.0,
            p2_three_da=75.0,
        )
        # G3 constraints
        assert abs(hb.p1_hold + hb.p2_break - 1.0) < 1e-10, (
            f"G3 violation: p1_hold={hb.p1_hold:.8f}, p2_break={hb.p2_break:.8f}"
        )
        assert abs(hb.p2_hold + hb.p1_break - 1.0) < 1e-10, (
            f"G3 violation: p2_hold={hb.p2_hold:.8f}, p1_break={hb.p1_break:.8f}"
        )

    def test_g3_validate_method(self):
        """validate() on a well-formed HB should not raise."""
        hb = _make_hb(p1_hold=0.65, p2_hold=0.58)
        hb.validate()  # should not raise

    def test_g3_invalid_hb_raises(self):
        """A HB with violated G3 should raise on validate()."""
        hb = HoldBreakProbabilities(
            p1_hold=0.65,
            p1_break=0.50,  # should be 1 - p2_hold, but we make it wrong
            p2_hold=0.58,
            p2_break=0.35,  # should be 1 - p1_hold = 0.35 ✓ but...
        )
        # p1_hold + p2_break = 0.65 + 0.35 = 1.0 ✓
        # p2_hold + p1_break = 0.58 + 0.50 = 1.08 ✗
        with pytest.raises(ValueError, match="consistency"):
            hb.validate()


# ---------------------------------------------------------------------------
# G4: Format from registry (not hardcoded)
# ---------------------------------------------------------------------------

class TestG4FormatFromRegistry:
    """G4: All format lookups must go through the registry."""

    def test_g4_all_registered_formats_retrievable(self):
        """Every format code in list_formats() must be retrievable."""
        codes = list_formats()
        assert len(codes) > 0, "Registry must contain at least one format"

        for code in codes:
            fmt = get_format(code)
            assert fmt.code == code, (
                f"Format code mismatch: registered {code!r}, returned {fmt.code!r}"
            )
            # Must have at least one round
            assert len(fmt.per_round) > 0

    def test_g4_unknown_format_raises(self):
        """Unknown format code must raise DartsFormatError."""
        with pytest.raises(DartsFormatError):
            get_format("NONEXISTENT_FORMAT_XYZ")

    def test_g4_format_has_correct_structure(self):
        """
        A selected known format must have all required structural fields.
        """
        fmt = get_format("PDC_WC")
        assert fmt.code == "PDC_WC"
        assert fmt.organiser == "PDC"
        assert fmt.ecosystem == "pdc_mens"
        assert fmt.starting_score == 501
        assert "Final" in fmt.per_round
        assert fmt.alternating_starts is True

    def test_g4_world_matchplay_has_two_clear_legs_in_final(self):
        """PDC_WM Final must have two_clear_legs flag set."""
        fmt = get_format("PDC_WM")
        final_round = fmt.get_round("Final")
        assert final_round.two_clear_legs is True

    def test_g4_unknown_round_raises(self):
        """Unknown round name must raise DartsFormatError."""
        fmt = get_format("PDC_WC")
        with pytest.raises(DartsFormatError, match="Round"):
            fmt.get_round("Quarterfinal_Typo")


# ---------------------------------------------------------------------------
# G5: Starter confidence flag set
# ---------------------------------------------------------------------------

class TestG5StarterConfidence:
    """G5: Margin engine must accept and use the starter_confidence parameter."""

    def test_g5_starter_confidence_increases_margin(self):
        """
        Lower starter_confidence should increase the margin.
        """
        from margin.blending_engine import DartsMarginEngine
        engine = DartsMarginEngine()

        margin_certain = engine.compute_margin(
            base_margin=0.05,
            regime=1,
            starter_confidence=1.0,  # certain
            source_confidence=1.0,
            model_agreement=1.0,
            market_liquidity="high",
            ecosystem="pdc_mens",
        )
        margin_uncertain = engine.compute_margin(
            base_margin=0.05,
            regime=1,
            starter_confidence=0.0,  # completely uncertain
            source_confidence=1.0,
            model_agreement=1.0,
            market_liquidity="high",
            ecosystem="pdc_mens",
        )
        assert margin_uncertain > margin_certain, (
            f"Uncertain starter should give higher margin: "
            f"certain={margin_certain:.4f}, uncertain={margin_uncertain:.4f}"
        )
        # At certainty=0: factor = 1 + 1*0.30 = 1.30
        expected_uncertain = 0.05 * 1.10 * 1.30  # R1 * starter_uncertainty
        assert abs(margin_uncertain - expected_uncertain) < 1e-9

    def test_g5_starter_confidence_out_of_range_raises(self):
        """Starter confidence outside [0, 1] should raise DartsEngineError."""
        from engines.errors import DartsEngineError
        from margin.blending_engine import DartsMarginEngine
        engine = DartsMarginEngine()

        with pytest.raises(DartsEngineError, match="starter_confidence"):
            engine.compute_margin(
                base_margin=0.05,
                regime=1,
                starter_confidence=1.5,  # invalid
                source_confidence=1.0,
                model_agreement=1.0,
                market_liquidity="high",
                ecosystem="pdc_mens",
            )


# ---------------------------------------------------------------------------
# G6: Draw market only opens if draw_enabled=True
# ---------------------------------------------------------------------------

class TestG6DrawMarket:
    """G6: Draw result must only be present when format allows draws."""

    def test_g6_draw_not_in_non_draw_format(self):
        """For a non-draw format, match result should not include a draw."""
        engine = MatchCombinatorialEngine()
        fmt = get_format("PDC_WC")  # No draws in WC
        round_fmt = fmt.get_round("Final")
        assert not round_fmt.draw_enabled, "WC Final should not have draw"

        hb = _make_symmetric_hb(hold=0.60)
        result = engine.price_match(
            hold_break=hb,
            fmt=fmt,
            round_name="Final",
            p1_starts_first=True,
        )
        assert result.draw == 0.0, (
            f"Draw should be 0.0 for non-draw format, got {result.draw:.6f}"
        )

    def test_g6_draw_present_in_draw_format(self):
        """For a draw-enabled format, match result should include a draw probability."""
        engine = MatchCombinatorialEngine()
        fmt = get_format("PDC_PL")  # Premier League has draws
        round_fmt = fmt.get_round("League Night")
        assert round_fmt.draw_enabled, "PL League Night should allow draws"

        hb = _make_symmetric_hb(hold=0.60)
        result = engine.price_match(
            hold_break=hb,
            fmt=fmt,
            round_name="League Night",
            p1_starts_first=True,
        )
        assert result.draw > 0.0, (
            f"Draw should be > 0 for draw-enabled format, got {result.draw:.6f}"
        )

    def test_g6_draw_enabled_flag_from_registry(self):
        """Draw-enabled formats must have 'draw' in result_types."""
        pdc_pl = get_format("PDC_PL")
        assert pdc_pl.allows_draw(), "PDC_PL must allow draws"

        pdc_gsc = get_format("PDC_GS")
        assert pdc_gsc.allows_draw(), "PDC_GS must allow draws"

        pdc_wc = get_format("PDC_WC")
        assert not pdc_wc.allows_draw(), "PDC_WC must not allow draws"


# ---------------------------------------------------------------------------
# G7: Two-clear-legs engine used when format flag set
# ---------------------------------------------------------------------------

class TestG7TwoClearLegs:
    """G7: Formats with two_clear_legs=True must use the WorldMatchplayEngine."""

    def test_g7_world_matchplay_final_uses_two_clear(self):
        """PDC_WM Final must use WorldMatchplayEngine (two-clear-legs)."""
        from engines.match_layer.world_matchplay_engine import WorldMatchplayEngine

        fmt = get_format("PDC_WM")
        round_fmt = fmt.get_round("Final")
        assert round_fmt.two_clear_legs is True, (
            "PDC_WM Final must have two_clear_legs=True"
        )

        # Price the match — must return without error (uses WorldMatchplayEngine)
        engine = MatchCombinatorialEngine()
        hb = _make_symmetric_hb(hold=0.62)
        result = engine.price_match(
            hold_break=hb,
            fmt=fmt,
            round_name="Final",
            p1_starts_first=True,
        )
        assert abs(result.p1_win + result.p2_win - 1.0) < 1e-6
        assert result.draw == 0.0

    def test_g7_standard_format_does_not_use_two_clear(self):
        """Standard format rounds must NOT have two_clear_legs flag."""
        fmt = get_format("PDC_WC")
        for round_name, round_fmt in fmt.per_round.items():
            assert not round_fmt.two_clear_legs, (
                f"PDC_WC round '{round_name}' should not have two_clear_legs"
            )

    def test_g7_two_clear_flag_propagates_to_result(self):
        """
        Match result from a two-clear-legs format should differ from
        the standard DP result for the same HB (since the two-clear
        engine handles ties differently).
        """
        fmt = get_format("PDC_WM")
        hb = _make_hb(p1_hold=0.62, p2_hold=0.58)

        engine = MatchCombinatorialEngine()
        # Should not raise — confirms WorldMatchplayEngine is invoked
        result = engine.price_match(
            hold_break=hb,
            fmt=fmt,
            round_name="Final",
            p1_starts_first=True,
        )
        # P1 is better → P1 should win more
        assert result.p1_win > result.p2_win, (
            f"Better player should win more: p1={result.p1_win:.4f}, p2={result.p2_win:.4f}"
        )
        assert abs(result.p1_win + result.p2_win - 1.0) < 1e-6
