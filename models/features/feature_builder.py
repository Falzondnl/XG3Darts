"""
Feature engineering for darts ML models.

Builds feature vectors for R0/R1/R2 models.
All features computed from data BEFORE match date (no leakage).

Feature provenance:
    R0 (14): Elo, rankings, format metadata, PDC career stats
    R1 (38): R0 + rolling stats, hold/break, stage/floor, H2H, form
    R2 (68): R1 + visit distributions, route choice, EB skills, LSTM state
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Optional

import numpy as np
import structlog

from models import DartsMLError
from models.r0_logit import R0_FEATURES
from models.r1_lgbm import R1_FEATURES
from models.r2_stacking import R2_FEATURES

logger = structlog.get_logger(__name__)


# Ecosystem encoding map
_ECOSYSTEM_ENCODING: dict[str, int] = {
    "pdc_mens": 0,
    "pdc_womens": 1,
    "wdf_open": 2,
    "development": 3,
    "team_doubles": 4,
}


class DartsFeatureBuilder:
    """
    Builds feature vectors for R0/R1/R2 models.

    All features are computed from data BEFORE match date (no leakage).
    The builder enforces temporal ordering and raises errors when
    feature computation would require future data.
    """

    def __init__(self) -> None:
        self._log = logger.bind(component="DartsFeatureBuilder")

    def build_r0_features(
        self,
        match: dict,
        player_stats: dict,
        elo_ratings: dict,
    ) -> dict:
        """
        Build 14 R0 features from official metadata.

        Parameters
        ----------
        match:
            Match metadata dict with keys:
                player1_id, player2_id, competition_id, format_code,
                round_name, match_date, ecosystem
        player_stats:
            Dict with player stats keyed by player_id, containing:
                ranking, three_da_pdc, checkout_pct_pdc
        elo_ratings:
            Dict with Elo ratings keyed by player_id.

        Returns
        -------
        dict
            Feature dict with all 14 R0 feature names.

        Raises
        ------
        DartsMLError
            If required data is missing.
        """
        p1_id = match.get("player1_id")
        p2_id = match.get("player2_id")

        if not p1_id or not p2_id:
            raise DartsMLError("Match must contain player1_id and player2_id.")

        elo_p1 = float(elo_ratings.get(p1_id, 1500.0))
        elo_p2 = float(elo_ratings.get(p2_id, 1500.0))

        p1_stats = player_stats.get(p1_id, {})
        p2_stats = player_stats.get(p2_id, {})

        ranking_p1 = float(p1_stats.get("ranking", 128))
        ranking_p2 = float(p2_stats.get("ranking", 128))

        # Log ratio of rankings (lower rank = better)
        # Avoid log(0) by clamping to [1, ...]
        ranking_log_ratio = math.log(max(1.0, ranking_p2)) - math.log(max(1.0, ranking_p1))

        three_da_p1 = float(p1_stats.get("three_da_pdc", 80.0))
        three_da_p2 = float(p2_stats.get("three_da_pdc", 80.0))

        checkout_p1 = float(p1_stats.get("checkout_pct_pdc", 35.0))
        checkout_p2 = float(p2_stats.get("checkout_pct_pdc", 35.0))

        # Format encoding
        format_code = match.get("format_code", "")
        is_sets = 1.0 if "WC" in format_code or "GP" in format_code else 0.0

        # Stage/floor: 1=stage (televised), 0=floor
        round_name = match.get("round_name", "")
        stage_floor = 1.0 if any(
            kw in round_name.lower()
            for kw in ("final", "semi", "quarter", "round 3", "round 4")
        ) else 0.0

        # Short format: 1 if best-of < 7 legs
        short_format = 1.0 if match.get("short_format", False) else 0.0

        # Ecosystem encoding
        ecosystem = match.get("ecosystem", "pdc_mens")
        ecosystem_encoded = float(_ECOSYSTEM_ENCODING.get(ecosystem, 0))

        features = {
            "elo_p1": elo_p1,
            "elo_p2": elo_p2,
            "elo_diff": elo_p1 - elo_p2,
            "ranking_p1": ranking_p1,
            "ranking_p2": ranking_p2,
            "ranking_log_ratio": ranking_log_ratio,
            "three_da_p1_pdc": three_da_p1,
            "three_da_p2_pdc": three_da_p2,
            "checkout_pct_p1_pdc": checkout_p1,
            "checkout_pct_p2_pdc": checkout_p2,
            "format_type_encoded": is_sets,
            "stage_floor": stage_floor,
            "short_format": short_format,
            "ecosystem_encoded": ecosystem_encoded,
        }

        # Verify all R0 features are present
        missing = set(R0_FEATURES) - set(features.keys())
        if missing:
            raise DartsMLError(f"Failed to build R0 features: missing {missing}")

        return features

    def build_r1_features(
        self,
        match: dict,
        player_stats: dict,
        elo_ratings: dict,
        rolling_stats: dict,
        h2h_stats: dict,
    ) -> dict:
        """
        Build 38 R1 features.

        R0 features + rolling stats, hold/break, stage/floor splits,
        throw-first/second, opponent-adjusted form, rest/travel, H2H.

        Parameters
        ----------
        match:
            Match metadata.
        player_stats:
            Player-level statistics.
        elo_ratings:
            Elo ratings.
        rolling_stats:
            Rolling window statistics keyed by player_id.
        h2h_stats:
            Head-to-head statistics dict with keys:
                p1_win_rate, total_matches.

        Returns
        -------
        dict
            Feature dict with all 38 R1 feature names.
        """
        # Start with R0 features
        features = self.build_r0_features(match, player_stats, elo_ratings)

        p1_id = match["player1_id"]
        p2_id = match["player2_id"]

        p1_rolling = rolling_stats.get(p1_id, {})
        p2_rolling = rolling_stats.get(p2_id, {})

        # Rolling stats
        features["ewm_form_p1"] = float(p1_rolling.get("ewm_form", 80.0))
        features["ewm_form_p2"] = float(p2_rolling.get("ewm_form", 80.0))
        features["rolling_win_rate_p1"] = float(p1_rolling.get("win_rate", 0.5))
        features["rolling_win_rate_p2"] = float(p2_rolling.get("win_rate", 0.5))
        features["rolling_3da_p1"] = float(p1_rolling.get("rolling_3da", 80.0))
        features["rolling_3da_p2"] = float(p2_rolling.get("rolling_3da", 80.0))
        features["rolling_checkout_pct_p1"] = float(
            p1_rolling.get("rolling_checkout_pct", 35.0)
        )
        features["rolling_checkout_pct_p2"] = float(
            p2_rolling.get("rolling_checkout_pct", 35.0)
        )

        # Hold/break rates
        features["hold_rate_p1"] = float(p1_rolling.get("hold_rate", 0.65))
        features["hold_rate_p2"] = float(p2_rolling.get("hold_rate", 0.65))
        features["break_rate_p1"] = float(p1_rolling.get("break_rate", 0.25))
        features["break_rate_p2"] = float(p2_rolling.get("break_rate", 0.25))

        # Stage/floor splits
        features["stage_3da_p1"] = float(p1_rolling.get("stage_3da", 80.0))
        features["stage_3da_p2"] = float(p2_rolling.get("stage_3da", 80.0))
        features["floor_3da_p1"] = float(p1_rolling.get("floor_3da", 80.0))
        features["floor_3da_p2"] = float(p2_rolling.get("floor_3da", 80.0))

        # Throw-first/second
        features["throw_first_win_rate_p1"] = float(
            p1_rolling.get("throw_first_win_rate", 0.55)
        )
        features["throw_first_win_rate_p2"] = float(
            p2_rolling.get("throw_first_win_rate", 0.55)
        )

        # Opponent-adjusted form
        features["opp_adj_form_p1"] = float(
            p1_rolling.get("opp_adj_form", 0.0)
        )
        features["opp_adj_form_p2"] = float(
            p2_rolling.get("opp_adj_form", 0.0)
        )

        # Rest/travel
        features["days_since_last_match_p1"] = float(
            p1_rolling.get("days_since_last_match", 7.0)
        )
        features["days_since_last_match_p2"] = float(
            p2_rolling.get("days_since_last_match", 7.0)
        )

        # H2H
        features["h2h_p1_win_rate"] = float(h2h_stats.get("p1_win_rate", 0.5))
        features["h2h_total_matches"] = float(h2h_stats.get("total_matches", 0))

        # Verify all R1 features
        missing = set(R1_FEATURES) - set(features.keys())
        if missing:
            raise DartsMLError(f"Failed to build R1 features: missing {missing}")

        return features

    def build_r2_features(
        self,
        match: dict,
        player_stats: dict,
        elo_ratings: dict,
        rolling_stats: dict,
        h2h_stats: dict,
        visit_distributions: dict,
        route_params: dict,
    ) -> dict:
        """
        Build 68 R2 features.

        R1 features + visit distributions, route choice, EB skills,
        LSTM momentum, segment accuracies.

        Parameters
        ----------
        match:
            Match metadata.
        player_stats:
            Player statistics.
        elo_ratings:
            Elo ratings.
        rolling_stats:
            Rolling window stats.
        h2h_stats:
            H2H stats.
        visit_distributions:
            Visit distribution stats keyed by player_id.
        route_params:
            Route choice and EB/LSTM parameters keyed by player_id.

        Returns
        -------
        dict
            Feature dict with all 68 R2 feature names.
        """
        # Start with R1 features
        features = self.build_r1_features(
            match, player_stats, elo_ratings, rolling_stats, h2h_stats
        )

        p1_id = match["player1_id"]
        p2_id = match["player2_id"]

        p1_visit = visit_distributions.get(p1_id, {})
        p2_visit = visit_distributions.get(p2_id, {})
        p1_route = route_params.get(p1_id, {})
        p2_route = route_params.get(p2_id, {})

        # Visit distributions by band
        features["visit_mean_open_p1"] = float(p1_visit.get("mean_open", 80.0))
        features["visit_mean_open_p2"] = float(p2_visit.get("mean_open", 80.0))
        features["visit_mean_setup_p1"] = float(p1_visit.get("mean_setup", 50.0))
        features["visit_mean_setup_p2"] = float(p2_visit.get("mean_setup", 50.0))
        features["visit_std_open_p1"] = float(p1_visit.get("std_open", 20.0))
        features["visit_std_open_p2"] = float(p2_visit.get("std_open", 20.0))
        features["visit_bust_rate_pressure_p1"] = float(
            p1_visit.get("bust_rate_pressure", 0.15)
        )
        features["visit_bust_rate_pressure_p2"] = float(
            p2_visit.get("bust_rate_pressure", 0.15)
        )
        features["visit_180_rate_p1"] = float(p1_visit.get("rate_180", 0.10))
        features["visit_180_rate_p2"] = float(p2_visit.get("rate_180", 0.10))

        # Route choice parameters
        features["preferred_double_pct_p1"] = float(
            p1_route.get("preferred_double_pct", 40.0)
        )
        features["preferred_double_pct_p2"] = float(
            p2_route.get("preferred_double_pct", 40.0)
        )
        features["route_aggression_p1"] = float(
            p1_route.get("route_aggression", 0.5)
        )
        features["route_aggression_p2"] = float(
            p2_route.get("route_aggression", 0.5)
        )

        # EB skill estimates
        features["eb_t20_accuracy_p1"] = float(
            p1_route.get("eb_t20_accuracy", 0.40)
        )
        features["eb_t20_accuracy_p2"] = float(
            p2_route.get("eb_t20_accuracy", 0.40)
        )
        features["eb_d_accuracy_p1"] = float(
            p1_route.get("eb_d_accuracy", 0.35)
        )
        features["eb_d_accuracy_p2"] = float(
            p2_route.get("eb_d_accuracy", 0.35)
        )
        features["eb_bull_accuracy_p1"] = float(
            p1_route.get("eb_bull_accuracy", 0.20)
        )
        features["eb_bull_accuracy_p2"] = float(
            p2_route.get("eb_bull_accuracy", 0.20)
        )

        # LSTM momentum state
        features["lstm_momentum_p1"] = float(
            p1_route.get("lstm_momentum", 0.0)
        )
        features["lstm_momentum_p2"] = float(
            p2_route.get("lstm_momentum", 0.0)
        )
        features["lstm_volatility_p1"] = float(
            p1_route.get("lstm_volatility", 0.0)
        )
        features["lstm_volatility_p2"] = float(
            p2_route.get("lstm_volatility", 0.0)
        )

        # Segment accuracies
        features["segment_t20_hit_rate_p1"] = float(
            p1_route.get("segment_t20_hit_rate", 0.40)
        )
        features["segment_t20_hit_rate_p2"] = float(
            p2_route.get("segment_t20_hit_rate", 0.40)
        )
        features["segment_t19_hit_rate_p1"] = float(
            p1_route.get("segment_t19_hit_rate", 0.38)
        )
        features["segment_t19_hit_rate_p2"] = float(
            p2_route.get("segment_t19_hit_rate", 0.38)
        )
        features["segment_d_hit_rate_p1"] = float(
            p1_route.get("segment_d_hit_rate", 0.35)
        )
        features["segment_d_hit_rate_p2"] = float(
            p2_route.get("segment_d_hit_rate", 0.35)
        )

        # Verify all R2 features
        missing = set(R2_FEATURES) - set(features.keys())
        if missing:
            raise DartsMLError(f"Failed to build R2 features: missing {missing}")

        return features

    @staticmethod
    def _compute_ewm_form(
        historical_3da: list[float],
        decay: float = 0.05,
    ) -> float:
        """
        Compute exponentially weighted moving average of 3-dart averages.

        Parameters
        ----------
        historical_3da:
            List of 3DA values in chronological order (oldest first).
        decay:
            Decay rate per observation.

        Returns
        -------
        float
            EWM 3DA value.

        Raises
        ------
        RuntimeError
            If no historical 3DA data is available.
        """
        if not historical_3da:
            raise RuntimeError("No historical 3DA data available")

        weights = [(1.0 - decay) ** i for i in range(len(historical_3da))]
        weight_sum = sum(weights)
        if weight_sum == 0:
            raise RuntimeError("Weight sum is zero in EWM computation")

        weights = [w / weight_sum for w in weights]
        return sum(w * v for w, v in zip(weights, reversed(historical_3da)))

    @staticmethod
    def _compute_rolling_stats(
        match_history: list[dict],
        before_date: str,
        window: int = 200,
    ) -> dict:
        """
        Compute rolling window stats strictly before match date.

        Parameters
        ----------
        match_history:
            List of historical match records in chronological order.
            Each record must contain: match_date, three_da, checkout_pct,
            won (bool), throw_first (bool), is_stage (bool),
            opponent_elo, held (bool), broke (bool).
        before_date:
            ISO date string. Only matches before this date are used.
        window:
            Maximum number of recent matches to consider.

        Returns
        -------
        dict
            Rolling statistics dict.

        Raises
        ------
        DartsMLError
            If date parsing fails.
        """
        try:
            cutoff = datetime.fromisoformat(before_date).date() if isinstance(
                before_date, str
            ) else before_date
        except (ValueError, TypeError) as exc:
            raise DartsMLError(f"Invalid before_date: {before_date}") from exc

        # Filter to matches before cutoff date
        eligible = []
        for m in match_history:
            m_date = m.get("match_date")
            if m_date is None:
                continue
            if isinstance(m_date, str):
                m_date = datetime.fromisoformat(m_date).date()
            if m_date < cutoff:
                eligible.append(m)

        # Take most recent `window` matches
        recent = eligible[-window:] if len(eligible) > window else eligible

        if not recent:
            return {}

        # Compute stats
        three_das = [m["three_da"] for m in recent if "three_da" in m and m["three_da"] is not None]
        wins = [1.0 if m.get("won", False) else 0.0 for m in recent]
        checkouts = [
            m["checkout_pct"] for m in recent
            if "checkout_pct" in m and m["checkout_pct"] is not None
        ]

        hold_opportunities = [m for m in recent if m.get("throw_first") is True]
        break_opportunities = [m for m in recent if m.get("throw_first") is False]

        stage_matches = [m for m in recent if m.get("is_stage") is True]
        floor_matches = [m for m in recent if m.get("is_stage") is False]
        throw_first_matches = [m for m in recent if m.get("throw_first") is True]

        stats: dict = {}

        # EWM form
        if three_das:
            stats["ewm_form"] = DartsFeatureBuilder._compute_ewm_form(three_das)
            stats["rolling_3da"] = sum(three_das) / len(three_das)

        # Win rate
        stats["win_rate"] = sum(wins) / len(wins) if wins else 0.5

        # Checkout percentage
        if checkouts:
            stats["rolling_checkout_pct"] = sum(checkouts) / len(checkouts)

        # Hold/break rates
        if hold_opportunities:
            holds = [1.0 if m.get("held", False) else 0.0 for m in hold_opportunities]
            stats["hold_rate"] = sum(holds) / len(holds)

        if break_opportunities:
            breaks = [1.0 if m.get("broke", False) else 0.0 for m in break_opportunities]
            stats["break_rate"] = sum(breaks) / len(breaks)

        # Stage/floor 3DA splits
        stage_3das = [
            m["three_da"] for m in stage_matches
            if "three_da" in m and m["three_da"] is not None
        ]
        floor_3das = [
            m["three_da"] for m in floor_matches
            if "three_da" in m and m["three_da"] is not None
        ]
        if stage_3das:
            stats["stage_3da"] = sum(stage_3das) / len(stage_3das)
        if floor_3das:
            stats["floor_3da"] = sum(floor_3das) / len(floor_3das)

        # Throw-first win rate
        if throw_first_matches:
            tf_wins = [1.0 if m.get("won", False) else 0.0 for m in throw_first_matches]
            stats["throw_first_win_rate"] = sum(tf_wins) / len(tf_wins)

        # Opponent-adjusted form: mean(3DA - opponent_avg_elo_expected_3da)
        opp_adj_vals = []
        for m in recent:
            if "three_da" in m and m["three_da"] is not None and "opponent_elo" in m:
                opp_elo = m["opponent_elo"]
                # Higher opponent Elo -> expected to score lower against them
                # Adjustment: 3DA - baseline * (opp_elo / 1500)
                baseline_3da = stats.get("rolling_3da", 80.0)
                expected_adj = baseline_3da * (opp_elo / 1500.0)
                opp_adj_vals.append(m["three_da"] - expected_adj)
        if opp_adj_vals:
            stats["opp_adj_form"] = sum(opp_adj_vals) / len(opp_adj_vals)

        # Days since last match
        if recent:
            last_match = recent[-1]
            last_date = last_match.get("match_date")
            if last_date is not None:
                if isinstance(last_date, str):
                    last_date = datetime.fromisoformat(last_date).date()
                stats["days_since_last_match"] = (cutoff - last_date).days

        return stats

    def verify_no_future_features(
        self,
        features: dict,
        match_date: str,
        feature_dates: dict[str, str],
    ) -> None:
        """
        Verify that no feature was computed using data from after match_date.

        Parameters
        ----------
        features:
            Feature dict to verify.
        match_date:
            The match date (ISO string).
        feature_dates:
            Mapping of feature_name -> latest_data_date used in computation.

        Raises
        ------
        DartsMLError
            If any feature uses future data.
        """
        try:
            match_dt = datetime.fromisoformat(match_date).date()
        except (ValueError, TypeError) as exc:
            raise DartsMLError(f"Invalid match_date: {match_date}") from exc

        violations: list[str] = []
        for feat_name, data_date_str in feature_dates.items():
            if feat_name not in features:
                continue
            try:
                data_dt = datetime.fromisoformat(data_date_str).date()
            except (ValueError, TypeError):
                continue
            if data_dt >= match_dt:
                violations.append(
                    f"{feat_name}: data_date={data_date_str} >= match_date={match_date}"
                )

        if violations:
            raise DartsMLError(
                f"Temporal leakage detected in {len(violations)} features: "
                f"{violations[:5]}"
            )
