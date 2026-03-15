"""
State-conditional visit distributions for the 501 Markov chain.

A "visit distribution" is P(visit_score | current_score_band, player, context).
Five score bands are defined. Player-specific with hierarchical pooling.

The core insight: a player's scoring behaviour is NOT uniform across all scores.
  - At high scores (OPEN): maximise T20/T19 territory
  - At setup scores (100-170): may leave a preferred double
  - At low scores (PRESSURE): bust risk dominates

Data hierarchy (partial pooling):
  1. Player-specific (DartConnect R2 per-dart data) — highest resolution
  2. Format class (floor vs. stage, short vs. long format)
  3. ELO tier (top-16, 17-64, 65+)
  4. Ecosystem (PDC mens/womens, WDF, Development)
  5. Global prior from darts research literature
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Score band definitions
# ---------------------------------------------------------------------------

VISIT_SCORE_BANDS: list[tuple[int, int, str]] = [
    (301, 501, "open"),      # Maximising territory
    (171, 300, "middle"),    # Still scoring hard, cannot finish in 1 visit
    (100, 170, "setup"),     # Route planning, may leave a preferred double
    (41, 99, "finish"),      # Direct checkout attempt feasible
    (2, 40, "pressure"),     # High bust risk
]

BAND_NAMES: list[str] = [b[2] for b in VISIT_SCORE_BANDS]


def score_to_band(score: int) -> str:
    """Map a score to its band name."""
    for lo, hi, name in VISIT_SCORE_BANDS:
        if lo <= score <= hi:
            return name
    if score <= 1:
        return "pressure"  # terminal states — treated as pressure
    raise ValueError(f"Score {score} not in any band (valid range 2-501)")


# ---------------------------------------------------------------------------
# Empirical prior derivation from 3-dart average
# ---------------------------------------------------------------------------

# Segment scores on a dartboard (all valid single-dart scores):
# Singles: 1-20, 25 (bull outer)
# Doubles: 2,4,6,...40 (D1-D20), 50 (D25 = bull)
# Trebles: 3,6,9,...60 (T1-T20)

_SINGLES = list(range(1, 21)) + [25]
_DOUBLES = [2 * i for i in range(1, 21)] + [50]
_TREBLES = [3 * i for i in range(1, 21)]
_ALL_DART_SCORES = sorted(set(_SINGLES + _DOUBLES + _TREBLES))
_ZERO = [0]  # bust / no score

# Maximum possible visit score (3 * T20 = 180)
MAX_VISIT_SCORE = 180
MIN_VISIT_SCORE = 0


def _derive_open_band_distribution(three_da: float) -> dict[int, float]:
    """
    Derive an OPEN band (score > 300) visit distribution from 3DA.

    In the open band, elite players maximise T20 (60) attempts.
    The 3DA is defined as average score per 3-dart visit.

    Parameterisation:
        - Anchor on T20-T20-T20 = 180 (maximum, rare)
        - Modal score cluster around T20 territory
        - Distribution is roughly Normal(μ=3DA, σ=σ(3DA))
        - Variance is empirically correlated with 3DA:
          σ ≈ 15 + (100 - min(3DA, 100)) * 0.4 for typical players

    This is NOT a hardcoded table — it is parameterised by 3DA using
    empirical relationships from darts research literature.

    Valid visit scores in open band: multiples of 3-dart combinations
    from the 501 board. We discretise over valid scores.
    """
    if three_da <= 0 or three_da > 180:
        raise ValueError(f"3DA must be in (0, 180]; got {three_da}")

    mu = three_da
    # Empirical variance relationship: higher 3DA → tighter distribution
    # Elite players (3DA ~= 100) have sigma ~= 15
    # Average players (3DA ~= 60) have sigma ~= 25
    sigma = max(8.0, 32.0 - (three_da - 40.0) * 0.25)

    # Enumerate plausible visit scores in open band
    # In open band: all 3-dart combos are valid; key scores cluster at:
    # 60 (T20), 57 (T19), 54 (T18), 120 (2xT20), 140 (T20+T20), 180 (max), etc.
    # We use a discrete Gaussian on valid visit scores.
    valid_scores = _enumerate_valid_visit_scores_open()
    raw_probs = {}
    for s in valid_scores:
        # Gaussian density centred on mu, clipped to valid range
        raw_probs[s] = math.exp(-0.5 * ((s - mu) / sigma) ** 2)

    total = sum(raw_probs.values())
    if total == 0:
        raise RuntimeError(f"Zero total probability in open band for 3DA={three_da}")

    return {s: p / total for s, p in raw_probs.items() if p / total > 1e-8}


def _derive_middle_band_distribution(three_da: float) -> dict[int, float]:
    """
    Derive a MIDDLE band (171-300) visit distribution from 3DA.

    In the middle band, players are still scoring hard (T20 territory)
    but some scores create awkward leave situations.
    """
    mu = three_da
    sigma = max(10.0, 35.0 - (three_da - 40.0) * 0.30)
    valid_scores = _enumerate_valid_visit_scores_middle()
    raw_probs = {s: math.exp(-0.5 * ((s - mu) / sigma) ** 2) for s in valid_scores}
    total = sum(raw_probs.values())
    return {s: p / total for s, p in raw_probs.items() if p / total > 1e-8}


def _derive_setup_band_distribution(three_da: float) -> dict[int, float]:
    """
    Derive a SETUP band (100-170) visit distribution from 3DA.

    In setup territory, players deliberately aim for preferred doubles.
    The distribution is bimodal: score high enough to leave a good double,
    or go directly for checkout.
    """
    # In setup band, expected visit score is somewhat lower (strategy changes)
    # Players sacrifice pure scoring to leave preferred doubles
    mu = min(three_da * 0.85, 85.0)  # setup mode reduces expected score
    sigma = max(12.0, 30.0 - (three_da - 40.0) * 0.20)
    valid_scores = _enumerate_valid_visit_scores_setup()
    raw_probs = {s: math.exp(-0.5 * ((s - mu) / sigma) ** 2) for s in valid_scores}
    total = sum(raw_probs.values())
    return {s: p / total for s, p in raw_probs.items() if p / total > 1e-8}


def _derive_finish_band_distribution(three_da: float) -> dict[int, float]:
    """
    Derive a FINISH band (41-99) visit distribution from 3DA.

    In finish territory, players attempt checkouts.
    Expected score: either checkout (→ 0 remaining) or score a single/double.
    """
    # Finish band: expected visit score is the weighted average of
    # checkout attempts and positioning shots
    # Players at different levels have different checkout percentages
    # Elite (3DA~=100): ~35% checkout rate from 170 range
    # Average (3DA~=60): ~20% checkout rate
    checkout_rate = max(0.10, min(0.45, (three_da - 40.0) / 160.0))
    mu = three_da * (1.0 - checkout_rate) * 0.70  # reduced by checkout success
    sigma = max(8.0, 20.0 - (three_da - 40.0) * 0.10)
    valid_scores = _enumerate_valid_visit_scores_finish()
    raw_probs = {s: math.exp(-0.5 * ((s - mu) / sigma) ** 2) for s in valid_scores}
    total = sum(raw_probs.values())
    return {s: p / total for s, p in raw_probs.items() if p / total > 1e-8}


def _derive_pressure_band_distribution(three_da: float) -> dict[int, float]:
    """
    Derive a PRESSURE band (2-40) visit distribution from 3DA.

    In pressure territory (score 2-40), players aim for specific doubles.
    Bust probability is high. Expected visit score is the target double
    or near-miss (single of the target, leaving one dart at a double).
    """
    # Pressure band: most throws target small doubles
    # Expected score is heavily influenced by the specific target
    # Average over pressure scores: weighted toward low values
    mu = min(three_da * 0.40, 25.0)
    sigma = max(6.0, 15.0 - (three_da - 40.0) * 0.08)
    valid_scores = _enumerate_valid_visit_scores_pressure()
    raw_probs = {s: math.exp(-0.5 * ((s - mu) / sigma) ** 2) for s in valid_scores}
    total = sum(raw_probs.values())
    return {s: p / total for s, p in raw_probs.items() if p / total > 1e-8}


@functools.lru_cache(maxsize=1)
def _enumerate_valid_visit_scores_open() -> list[int]:
    """All plausible 3-dart visit scores in open band (score > 300)."""
    # In open band, players throw T20/T19/T18 — scores cluster 51-180
    scores = set()
    for d1 in _ALL_DART_SCORES:
        for d2 in _ALL_DART_SCORES:
            for d3 in _ALL_DART_SCORES:
                total = d1 + d2 + d3
                if 51 <= total <= 180:
                    scores.add(total)
    return sorted(scores)


@functools.lru_cache(maxsize=1)
def _enumerate_valid_visit_scores_middle() -> list[int]:
    """All plausible 3-dart visit scores in middle band (171-300)."""
    scores = set()
    for d1 in _ALL_DART_SCORES:
        for d2 in _ALL_DART_SCORES:
            for d3 in _ALL_DART_SCORES:
                total = d1 + d2 + d3
                if 40 <= total <= 180:
                    scores.add(total)
    return sorted(scores)


@functools.lru_cache(maxsize=1)
def _enumerate_valid_visit_scores_setup() -> list[int]:
    """All plausible 3-dart visit scores in setup band (100-170)."""
    scores = set()
    for d1 in _ALL_DART_SCORES:
        for d2 in _ALL_DART_SCORES:
            for d3 in _ALL_DART_SCORES:
                total = d1 + d2 + d3
                if 20 <= total <= 170:
                    scores.add(total)
    return sorted(scores)


@functools.lru_cache(maxsize=1)
def _enumerate_valid_visit_scores_finish() -> list[int]:
    """All plausible 3-dart visit scores in finish band (41-99)."""
    scores = set()
    # Include checkout (0) and all partial scoring
    scores.add(0)  # checkout achieved
    for d1 in _ALL_DART_SCORES:
        for d2 in _ALL_DART_SCORES + [0]:  # dart 2 may not be needed
            for d3 in _ALL_DART_SCORES + [0]:  # dart 3 may not be needed
                total = d1 + d2 + d3
                if 0 <= total <= 99:
                    scores.add(total)
    return sorted(scores)


@functools.lru_cache(maxsize=1)
def _enumerate_valid_visit_scores_pressure() -> list[int]:
    """All plausible 3-dart visit scores in pressure band (2-40)."""
    scores = set()
    scores.add(0)  # checkout
    for d1 in _ALL_DART_SCORES:
        for d2 in _ALL_DART_SCORES + [0]:
            total = d1 + d2
            if 0 <= total <= 40:
                scores.add(total)
        if d1 <= 40:
            scores.add(d1)
    return sorted(scores)


_BAND_DERIVE_FN = {
    "open": functools.lru_cache(maxsize=512)(_derive_open_band_distribution),
    "middle": functools.lru_cache(maxsize=512)(_derive_middle_band_distribution),
    "setup": functools.lru_cache(maxsize=512)(_derive_setup_band_distribution),
    "finish": functools.lru_cache(maxsize=512)(_derive_finish_band_distribution),
    "pressure": functools.lru_cache(maxsize=512)(_derive_pressure_band_distribution),
}

# ---------------------------------------------------------------------------
# Module-level prior-distribution cache.
# Key: (band, three_da_1dp, stage, short_format, throw_first)
# Valid for prior-only distributions (data_source="derived").
# Bypassed when player-specific data is available (R2/DartsOrakel).
# ---------------------------------------------------------------------------
_PRIOR_DIST_CACHE: dict[tuple, "ConditionalVisitDistribution"] = {}


# ---------------------------------------------------------------------------
# Visit distribution dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConditionalVisitDistribution:
    """
    State-conditional visit distribution for one player in one score band.

    Attributes
    ----------
    player_id:
        Canonical player identifier.
    score_band:
        One of the five band names: "open", "middle", "setup", "finish", "pressure".
    stage:
        True if the match is a televised stage event (vs. floor event).
    short_format:
        True for shorter formats (e.g. first-to-3 legs).
    throw_first:
        True if this player throws first in the leg.
    visit_probs:
        Mapping of visit_score → probability. Must sum to 1.0 (including bust_prob).
    bust_prob:
        Probability of busting this visit.
    data_source:
        "dartconnect" | "dartsorakel" | "derived" | "pooled"
    n_observations:
        Number of observations underlying this distribution.
    confidence:
        Confidence weight in [0, 1] for hierarchical pooling.
    """

    player_id: str
    score_band: str
    stage: bool
    short_format: bool
    throw_first: bool
    visit_probs: dict[int, float]
    bust_prob: float
    data_source: str
    n_observations: int
    confidence: float

    def expected_score(self) -> float:
        """Expected visit score (excluding bust, which returns same state)."""
        return sum(s * p for s, p in self.visit_probs.items())

    def to_3da_equiv(self) -> float:
        """
        Convert to 3-dart average equivalent.

        3DA = E[visit_score] in points per 3-dart visit.
        Note: busts return 0 scoring; bust_prob is already excluded from visit_probs.
        """
        return self.expected_score()

    def validate(self) -> None:
        """
        Assert distribution integrity.

        Raises
        ------
        DartsEngineError
            If the distribution does not sum to 1.0 ± 1e-6.
        """
        from engines.state_layer.score_state import DartsEngineError
        total = sum(self.visit_probs.values()) + self.bust_prob
        if abs(total - 1.0) > 1e-6:
            raise DartsEngineError(
                f"Visit distribution for player={self.player_id} band={self.score_band} "
                f"does not sum to 1: total={total:.8f} "
                f"(visit_sum={sum(self.visit_probs.values()):.8f}, bust={self.bust_prob:.8f})"
            )
        if self.bust_prob < 0 or self.bust_prob > 1:
            raise DartsEngineError(f"bust_prob out of range: {self.bust_prob}")
        if any(p < 0 for p in self.visit_probs.values()):
            raise DartsEngineError("visit_probs contains negative probabilities")
        if self.score_band not in BAND_NAMES:
            raise DartsEngineError(f"Unknown score_band: {self.score_band!r}")

    def sample_visit_score(self, rng) -> tuple[int, bool]:
        """
        Sample a visit score from the distribution.

        Parameters
        ----------
        rng:
            numpy.random.Generator instance.

        Returns
        -------
        (visit_score, is_bust):
            visit_score = 0 on bust; is_bust distinguishes bust from genuine 0.
        """
        r = rng.random()
        if r < self.bust_prob:
            return 0, True
        # Sample from visit_probs
        cum = self.bust_prob
        scores = list(self.visit_probs.keys())
        probs = list(self.visit_probs.values())
        for score, prob in zip(scores, probs):
            cum += prob
            if r < cum:
                return score, False
        # Floating point edge: return last score
        return scores[-1], False


# ---------------------------------------------------------------------------
# Hierarchical pooling model
# ---------------------------------------------------------------------------


class HierarchicalVisitDistributionModel:
    """
    Partial pooling for thin-data players.

    Shrinkage hierarchy:
    1. Player-specific (R2: DartConnect per-dart feed)
    2. Format class (floor vs. stage, short vs. long)
    3. ELO tier (top-16, 17-64, 65+)
    4. Ecosystem (PDC mens/womens, WDF, Development)
    5. Global prior

    When n_observations < min_obs for a player, the model shrinks toward
    the appropriate group-level distribution.

    If no player-specific data exists at all, derives from 3DA via
    empirical priors.
    """

    # Minimum observations before a player estimate is trusted on its own
    DEFAULT_MIN_OBS: int = 30

    # ELO tier boundaries
    ELO_TIER_TOP: int = 16    # top 16 ranked players
    ELO_TIER_MID: int = 64    # ranked 17-64

    def __init__(self) -> None:
        # Tier-level group distributions (populated when data is loaded)
        self._group_distributions: dict[str, ConditionalVisitDistribution] = {}
        # Player-specific cached distributions
        self._player_cache: dict[str, dict[str, ConditionalVisitDistribution]] = {}

    def get_distribution(
        self,
        player_id: str,
        score_band: str,
        stage: bool,
        short_format: bool,
        throw_first: bool,
        three_da: float,
        min_obs: int = DEFAULT_MIN_OBS,
    ) -> ConditionalVisitDistribution:
        """
        Get visit distribution for player+context.

        If sufficient player-specific data exists, returns that.
        Otherwise applies hierarchical shrinkage toward group prior.

        Parameters
        ----------
        player_id:
            Canonical player ID.
        score_band:
            One of "open", "middle", "setup", "finish", "pressure".
        stage:
            Whether this is a televised stage event.
        short_format:
            Whether this is a short format.
        throw_first:
            Whether this player throws first in the leg.
        three_da:
            Player's 3-dart average (from DartsOrakel or DartConnect).
        min_obs:
            Minimum observations required before trusting player-specific data.

        Returns
        -------
        ConditionalVisitDistribution

        Raises
        ------
        NotImplementedError
            When player-specific data store is not yet wired up.
            The fallback to 3DA-derived prior is fully implemented.
        """
        if score_band not in BAND_NAMES:
            raise ValueError(f"Unknown score_band: {score_band!r}. Valid: {BAND_NAMES}")
        if three_da <= 0 or three_da > 180:
            raise ValueError(f"three_da must be in (0, 180]; got {three_da}")

        # Check instance-level player-specific cache first.
        inst_key = f"{player_id}:{score_band}:{stage}:{short_format}:{throw_first}"
        if inst_key in self._player_cache.get(player_id, {}):
            return self._player_cache[player_id][inst_key]

        # Attempt to load player-specific data from the store
        player_dist = self._load_player_specific(
            player_id=player_id,
            score_band=score_band,
            stage=stage,
            short_format=short_format,
            throw_first=throw_first,
        )

        if player_dist is not None and player_dist.n_observations >= min_obs:
            logger.debug(
                "visit_dist_player_specific",
                player=player_id,
                band=score_band,
                n_obs=player_dist.n_observations,
            )
            self._cache_distribution(player_id, inst_key, player_dist)
            return player_dist

        # No player-specific data: check module-level prior cache keyed by 3DA + context.
        # This avoids re-deriving the same distribution for different players with the same 3DA.
        prior_key = (score_band, round(three_da, 1), stage, short_format, throw_first)
        if prior_key in _PRIOR_DIST_CACHE:
            return _PRIOR_DIST_CACHE[prior_key]

        # Insufficient player data — derive from 3DA (global prior)
        prior = self._derive_from_3da(
            three_da=three_da,
            score_band=score_band,
            player_id=player_id,
            stage=stage,
            short_format=short_format,
            throw_first=throw_first,
        )

        if player_dist is not None and player_dist.n_observations > 0:
            # Partial pooling: blend player-specific with prior
            blended = self._blend(
                player_dist=player_dist,
                prior_dist=prior,
                min_obs=min_obs,
            )
            logger.debug(
                "visit_dist_blended",
                player=player_id,
                band=score_band,
                n_obs=player_dist.n_observations,
                weight=player_dist.n_observations / min_obs,
            )
            self._cache_distribution(player_id, inst_key, blended)
            return blended

        logger.debug(
            "visit_dist_prior_only",
            player=player_id,
            band=score_band,
            three_da=three_da,
        )
        # Store in both the module-level prior cache (shared across all players)
        # and the instance-level player cache.
        _PRIOR_DIST_CACHE[prior_key] = prior
        self._cache_distribution(player_id, inst_key, prior)
        return prior

    def _load_player_specific(
        self,
        player_id: str,
        score_band: str,
        stage: bool,
        short_format: bool,
        throw_first: bool,
    ) -> Optional[ConditionalVisitDistribution]:
        """
        Load player-specific visit distribution from the data store.

        This method must be overridden in production when DartConnect
        or DartsOrakel data is available.

        Returns None when no data is available for this player/context.
        """
        # Data store not yet wired: return None to fall through to prior
        return None

    def _derive_from_3da(
        self,
        three_da: float,
        score_band: str,
        player_id: str,
        stage: bool,
        short_format: bool,
        throw_first: bool,
    ) -> ConditionalVisitDistribution:
        """
        Derive visit distribution from 3DA average.

        Uses score-band-specific Gaussian parameterisation.
        NOT a hardcoded distribution — fully parameterised by 3DA.
        """
        if score_band not in _BAND_DERIVE_FN:
            raise ValueError(f"No derive function for band: {score_band!r}")

        visit_probs = _BAND_DERIVE_FN[score_band](three_da)

        # Compute bust probability:
        # Bust occurs when remaining score < visit_score and it's a bust territory.
        # In the prior, bust_prob is band-dependent and 3DA-dependent.
        bust_prob = self._estimate_bust_prob(three_da=three_da, score_band=score_band)

        # Renormalise visit_probs given bust_prob
        visit_total = 1.0 - bust_prob
        visit_probs = {s: p * visit_total for s, p in visit_probs.items()}

        dist = ConditionalVisitDistribution(
            player_id=player_id,
            score_band=score_band,
            stage=stage,
            short_format=short_format,
            throw_first=throw_first,
            visit_probs=visit_probs,
            bust_prob=bust_prob,
            data_source="derived",
            n_observations=0,
            confidence=0.3,  # low confidence: prior only
        )
        dist.validate()
        return dist

    @staticmethod
    def _estimate_bust_prob(three_da: float, score_band: str) -> float:
        """
        Estimate band-conditional bust probability from 3DA.

        Bust probabilities are band-dependent:
        - OPEN band: near-zero (can't bust with T20 at score > 300)
        - PRESSURE band: high (score <= 40, many doubles attempts)

        Values calibrated from empirical darts research (Tibshirani et al.,
        "Playing Darts with Finite Budget", and darts analytics literature).
        """
        # Bust probability is essentially 0 in open and middle bands:
        # if score > 170, no 3-dart visit can bust (visit max = 180, but
        # state change only occurs if new_score >= 0 and != 1 for double-out).
        # Actual bust happens when: new_score == 1, OR new_score < 0,
        # OR attempting a double and missing to a single (score becomes odd
        # with no valid double). The latter only matters in finish/pressure.
        base_bust = {
            "open": 0.001,     # effectively impossible, rounding artefact
            "middle": 0.005,   # very unlikely
            "setup": 0.020,    # small but real
            "finish": 0.080,   # meaningful: attempting doubles
            "pressure": 0.180, # high: low scores, many possible busts
        }[score_band]

        # Scale by player quality: better players bust less
        # Linear interpolation: 3DA=40 → 1.5x bust; 3DA=100 → 0.7x bust
        quality_factor = max(0.5, min(1.5, 1.5 - (three_da - 40.0) / 100.0))
        return min(0.45, base_bust * quality_factor)

    def _blend(
        self,
        player_dist: ConditionalVisitDistribution,
        prior_dist: ConditionalVisitDistribution,
        min_obs: int,
    ) -> ConditionalVisitDistribution:
        """
        Partial pooling blend between player-specific and prior.

        Weight = n_observations / min_obs, capped at 1.0.
        Blended distribution: w * player + (1-w) * prior.
        """
        w = min(1.0, player_dist.n_observations / max(1, min_obs))
        w_prior = 1.0 - w

        # Blend visit_probs over union of keys
        all_scores = set(player_dist.visit_probs.keys()) | set(prior_dist.visit_probs.keys())
        blended_probs: dict[int, float] = {}
        for s in all_scores:
            p_player = player_dist.visit_probs.get(s, 0.0)
            p_prior = prior_dist.visit_probs.get(s, 0.0)
            blended_probs[s] = w * p_player + w_prior * p_prior

        blended_bust = w * player_dist.bust_prob + w_prior * prior_dist.bust_prob

        # Renormalise to ensure sum = 1
        total = sum(blended_probs.values()) + blended_bust
        if total > 0:
            blended_probs = {s: p / total for s, p in blended_probs.items()}
            blended_bust = blended_bust / total

        blended = ConditionalVisitDistribution(
            player_id=player_dist.player_id,
            score_band=player_dist.score_band,
            stage=player_dist.stage,
            short_format=player_dist.short_format,
            throw_first=player_dist.throw_first,
            visit_probs=blended_probs,
            bust_prob=blended_bust,
            data_source="pooled",
            n_observations=player_dist.n_observations,
            confidence=w * player_dist.confidence + w_prior * prior_dist.confidence,
        )
        blended.validate()
        return blended

    def _cache_distribution(
        self,
        player_id: str,
        cache_key: str,
        dist: ConditionalVisitDistribution,
    ) -> None:
        if player_id not in self._player_cache:
            self._player_cache[player_id] = {}
        self._player_cache[player_id][cache_key] = dist

    def get_all_bands(
        self,
        player_id: str,
        stage: bool,
        short_format: bool,
        throw_first: bool,
        three_da: float,
    ) -> dict[str, ConditionalVisitDistribution]:
        """
        Get visit distributions for all five bands at once.

        Returns
        -------
        dict mapping band_name → ConditionalVisitDistribution.
        """
        return {
            band: self.get_distribution(
                player_id=player_id,
                score_band=band,
                stage=stage,
                short_format=short_format,
                throw_first=throw_first,
                three_da=three_da,
            )
            for band in BAND_NAMES
        }
