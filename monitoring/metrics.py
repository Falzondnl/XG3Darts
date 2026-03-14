"""
Prometheus metrics for the XG3 Darts pricing engine.

All metrics are registered at module import and shared across the service.
Labels allow slicing by market family and regime.

Import this module early (in app/main.py or engine init) so counters
are zero-initialised before the first request arrives.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Pricing request counters
# ---------------------------------------------------------------------------

PRICE_REQUESTS = Counter(
    "darts_price_requests_total",
    "Total pricing requests handled",
    ["market", "regime"],
)

PRICE_ERRORS = Counter(
    "darts_price_errors_total",
    "Total pricing errors (engine failures, data errors, etc.)",
    ["market", "error_type"],
)

# ---------------------------------------------------------------------------
# Pricing latency
# ---------------------------------------------------------------------------

PRICE_LATENCY = Histogram(
    "darts_price_latency_seconds",
    "End-to-end pricing latency from request receipt to response",
    ["market"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

MARKOV_LATENCY = Histogram(
    "darts_markov_chain_latency_seconds",
    "Time spent computing hold/break probabilities via Markov chain",
    ["player_pair"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# ---------------------------------------------------------------------------
# Live engine counters
# ---------------------------------------------------------------------------

LIVE_UPDATES = Counter(
    "darts_live_updates_total",
    "Total live state updates received (visit-scored events)",
)

LIVE_LEG_COMPLETIONS = Counter(
    "darts_live_leg_completions_total",
    "Total legs completed during live matches",
)

LIVE_BUST_COUNT = Counter(
    "darts_live_bust_count_total",
    "Total bust visits observed in live matches",
)

LIVE_180_COUNT = Counter(
    "darts_live_180_count_total",
    "Total 180 maximum scores observed in live matches",
    ["player_id"],
)

LIVE_FEED_LAG = Histogram(
    "darts_live_feed_lag_milliseconds",
    "DartConnect/Optic Odds feed lag observed at update time",
    buckets=(100, 250, 500, 1000, 2000, 5000, 10000, 30000),
)

LIVE_FALLBACK_ACTIVATIONS = Counter(
    "darts_live_fallback_activations_total",
    "Times the Optic Odds fallback was activated due to DartConnect lag > 5 s",
)

LIVE_STATE_CACHE_HIT = Counter(
    "darts_live_state_cache_hits_total",
    "Redis cache hits when loading live match state",
)

LIVE_STATE_CACHE_MISS = Counter(
    "darts_live_state_cache_misses_total",
    "Redis cache misses when loading live match state (state not found)",
)

# ---------------------------------------------------------------------------
# Brier score gauges (updated by MarkovValidationMonitor)
# ---------------------------------------------------------------------------

BRIER_GAUGE = Gauge(
    "darts_brier_score",
    "Current rolling Brier score per market family",
    ["market_family"],
)

# ---------------------------------------------------------------------------
# Calibration / ECE gauges
# ---------------------------------------------------------------------------

ECE_GAUGE = Gauge(
    "darts_ece",
    "Expected Calibration Error per market family",
    ["market_family"],
)

AUC_GAUGE = Gauge(
    "darts_auc",
    "ROC-AUC per market family (binary markets only)",
    ["market_family"],
)

# ---------------------------------------------------------------------------
# Margin gauges (updated by CLVMonitor after auto-adjust)
# ---------------------------------------------------------------------------

MARGIN_GAUGE = Gauge(
    "darts_margin",
    "Current applied margin per market family",
    ["market_family"],
)

# ---------------------------------------------------------------------------
# CLV gauge
# ---------------------------------------------------------------------------

CLV_GAUGE = Gauge(
    "darts_clv",
    "Closing Line Value (log scale) per market family",
    ["market_family"],
)

# ---------------------------------------------------------------------------
# PSI / drift gauges
# ---------------------------------------------------------------------------

PSI_GAUGE = Gauge(
    "darts_psi",
    "Population Stability Index for feature drift detection",
    ["feature"],
)

DRIFT_ALERT_COUNT = Counter(
    "darts_drift_alerts_total",
    "Total drift alerts triggered (Brier > threshold or PSI > 0.2)",
    ["alert_type"],
)

# ---------------------------------------------------------------------------
# Kalman filter gauges
# ---------------------------------------------------------------------------

KALMAN_THREE_DA_GAUGE = Gauge(
    "darts_kalman_three_da",
    "Kalman-filtered 3-dart average for active live matches",
    ["player_id", "match_id"],
)

KALMAN_MOMENTUM_GAUGE = Gauge(
    "darts_kalman_momentum",
    "Kalman momentum factor for active live matches",
    ["player_id", "match_id"],
)

# ---------------------------------------------------------------------------
# Active live matches
# ---------------------------------------------------------------------------

LIVE_ACTIVE_MATCHES = Gauge(
    "darts_live_active_matches",
    "Number of currently active live matches tracked in Redis",
)
