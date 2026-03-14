# XG3 Darts Pricing Engine — API Reference

Version: 0.1.0
Base path: `/api/v1/darts`
Content-Type: `application/json`

---

## Authentication

All endpoints require a Bearer token in the `Authorization` header:

```
Authorization: Bearer <API_KEY>
```

API keys are issued per B2B consumer. Contact XG3 to obtain a key.

---

## Pre-Match Pricing

### POST /prematch/price

Compute pre-match match winner probabilities.

**Request body:**
```json
{
  "competition_code": "PDC_WM",
  "round_name": "Final",
  "p1_id": "player-uuid-1",
  "p2_id": "player-uuid-2",
  "p1_three_da": 98.5,
  "p2_three_da": 93.2,
  "p1_starts": true,
  "regime": 1,
  "ecosystem": "pdc_mens"
}
```

**Response 200:**
```json
{
  "match_id": null,
  "competition_code": "PDC_WM",
  "round_name": "Final",
  "p1_win": 0.6234,
  "p2_win": 0.3766,
  "margin": 0.0412,
  "regime": "R1",
  "starter_confidence": 1.0
}
```

**Errors:**
- `400` — Invalid competition format or round name
- `422` — Validation error (missing/invalid fields)
- `503` — Engine computation failed

---

### POST /prematch/exact-score

Price the correct score market (e.g. 10-7, 10-8).

**Request:** Same as `/prematch/price` plus optional `max_score_diff` parameter.

**Response 200:**
```json
{
  "scores": {
    "10-7": 0.1823,
    "10-8": 0.1456,
    "10-9": 0.1102,
    "9-10": 0.0987
  },
  "total_probability": 1.0
}
```

---

### POST /prematch/handicap

Price leg handicap markets.

**Request:**
```json
{
  "competition_code": "PDC_WC",
  "round_name": "Final",
  "p1_id": "player-uuid-1",
  "p2_id": "player-uuid-2",
  "p1_three_da": 99.1,
  "p2_three_da": 91.0,
  "handicap_line": -2.5
}
```

**Response 200:**
```json
{
  "p1_covers": 0.6812,
  "p2_covers": 0.3188,
  "handicap_line": -2.5
}
```

---

### POST /prematch/totals

Price total legs over/under markets.

**Request:**
```json
{
  "competition_code": "PDC_WC",
  "round_name": "Final",
  "p1_three_da": 99.1,
  "p2_three_da": 91.0,
  "total_line": 23.5
}
```

**Response 200:**
```json
{
  "over": 0.5234,
  "under": 0.4766,
  "expected_legs": 23.2
}
```

---

### GET /prematch/markets/{match_id}

Retrieve all pre-computed markets for a fixture.

**Parameters:**
- `match_id` (path) — fixture UUID

**Response 200:**
```json
{
  "match_id": "uuid",
  "markets": {
    "match_winner": {"p1_win": 0.62, "p2_win": 0.38},
    "total_legs": {"over_23.5": 0.52, "under_23.5": 0.48},
    "180s": {"over_5.5": 0.55, "under_5.5": 0.45}
  },
  "computed_at": "2026-03-14T12:00:00Z"
}
```

---

## Live Pricing

### POST /live/update

Process a live visit update and re-price.

**Request body:**
```json
{
  "match_id": "match-uuid",
  "competition_code": "PDC_WM",
  "round_name": "Final",
  "p1_id": "player-uuid-1",
  "p2_id": "player-uuid-2",
  "p1_three_da": 98.5,
  "p2_three_da": 93.2,
  "score_p1": 200,
  "score_p2": 380,
  "legs_p1": 5,
  "legs_p2": 4,
  "p1_throws_next": true,
  "visit_score": 180
}
```

**Response 200:**
```json
{
  "match_id": "match-uuid",
  "p1_win": 0.7123,
  "p2_win": 0.2877,
  "legs_p1": 5,
  "legs_p2": 4,
  "score_p1": 20,
  "score_p2": 380,
  "updated_at": "2026-03-14T14:23:01Z"
}
```

---

## Outrights

### POST /outrights/simulate

Run tournament simulation to compute outright winner probabilities.

**Request body:**
```json
{
  "competition_code": "PDC_MASTERS",
  "players": [
    {"player_id": "player-uuid-1", "elo_rating": 1920.5},
    {"player_id": "player-uuid-2", "elo_rating": 1845.0}
  ],
  "n_simulations": 10000,
  "seed": 42
}
```

**Response 200:**
```json
{
  "competition_code": "PDC_MASTERS",
  "n_simulations": 10000,
  "win_probabilities": {
    "player-uuid-1": 0.2341,
    "player-uuid-2": 0.1892
  },
  "computed_at": "2026-03-14T12:00:00Z"
}
```

---

## Same-Game Parlay (SGP)

### POST /sgp/price

Price a multi-leg same-game parlay.

**Request body:**
```json
{
  "match_id": "match-uuid",
  "competition_code": "PDC_WM",
  "round_name": "Final",
  "p1_id": "player-uuid-1",
  "p2_id": "player-uuid-2",
  "p1_three_da": 98.5,
  "p2_three_da": 93.2,
  "legs": [
    {"market": "match_winner", "selection": "p1_win", "raw_prob": 0.62},
    {"market": "total_legs", "selection": "over", "line": 23.5, "raw_prob": 0.52}
  ]
}
```

**Response 200:**
```json
{
  "match_id": "match-uuid",
  "combined_probability": 0.3224,
  "correlation_adjustment": 0.9812,
  "legs": [
    {"market": "match_winner", "selection": "p1_win", "probability": 0.62},
    {"market": "total_legs", "selection": "over", "probability": 0.52}
  ]
}
```

---

## Props

### POST /props/180s

Price 180s count prop for a match.

**Request body:**
```json
{
  "p1_id": "player-uuid-1",
  "p2_id": "player-uuid-2",
  "p1_three_da": 98.5,
  "p2_three_da": 93.2,
  "competition_code": "PDC_WM",
  "regime": 2
}
```

**Response 200:**
```json
{
  "expected_180s": 5.82,
  "lines": {
    "over_4.5": 0.7123,
    "under_4.5": 0.2877,
    "over_5.5": 0.5834,
    "over_6.5": 0.4123
  }
}
```

---

## Players

### GET /players/{player_id}

Return player profile and statistical summary.

**Query parameters:**
- `include_personal` (bool, default `false`) — include GDPR-protected fields

**Response 200:**
```json
{
  "player_id": "uuid",
  "slug": "michael-van-gerwen",
  "nickname": "MvG",
  "pdc_ranking": 1,
  "tour_card_holder": true,
  "dartsorakel_3da": 99.5,
  "dartsorakel_rank": 1,
  "source_confidence": 0.98,
  "primary_source": "dartsorakel",
  "gdpr_anonymized": false,
  "_links": {
    "self": "https://api.xg3.ai/api/v1/darts/players/uuid",
    "regime": "https://api.xg3.ai/api/v1/darts/players/uuid/regime",
    "elo": "https://api.xg3.ai/api/v1/darts/players/uuid/elo"
  }
}
```

**Errors:**
- `403` — Personal data requested without consent
- `404` — Player not found

---

### GET /players/{player_id}/regime

Return the active data coverage regime for a player.

**Response 200:**
```json
{
  "player_id": "uuid",
  "regime": "R1",
  "regime_score": 0.85,
  "has_visit_data": false,
  "has_match_stats": true,
  "has_dartsorakel": true,
  "has_dartconnect": false
}
```

Regimes:
- `R0` — result-only, baseline logit model
- `R1` — match-level stats, LightGBM model
- `R2` — full visit data, stacking ensemble

---

### POST /players/explain

Return SHAP feature importance for a player's pricing contribution.

**Request body:**
```json
{
  "player_id": "uuid",
  "opponent_id": "opponent-uuid",
  "competition_code": "PDC_WC",
  "round_name": "Final"
}
```

**Response 200:**
```json
{
  "player_id": "uuid",
  "shap_values": {
    "dartsorakel_3da": 0.142,
    "elo_rating": 0.087,
    "checkout_pct": 0.063,
    "form_last_5": 0.041
  },
  "features_used": ["dartsorakel_3da", "elo_rating", "checkout_pct", "form_last_5"]
}
```

**Errors:**
- `501` — SHAP not available for R0 regime

---

### GET /players/{player_id}/elo

Return ELO rating history.

**Query parameters:**
- `pool` (string) — filter by ELO pool
- `limit` (int, default `50`) — max rows returned

**Response 200:**
```json
{
  "player_id": "uuid",
  "current_ratings": {
    "pdc_mens": 1892.4,
    "pdc_womens": null
  },
  "history": [
    {
      "pool": "pdc_mens",
      "rating_before": 1881.2,
      "rating_after": 1892.4,
      "delta": 11.2,
      "match_date": "2026-01-12"
    }
  ]
}
```

---

### GET /players/search

Search players by name.

**Query parameters:**
- `q` (string, required, min 2 chars) — search query
- `limit` (int, default `20`) — max results

**Response 200:**
```json
{
  "query": "gerwen",
  "count": 1,
  "players": [
    {
      "player_id": "uuid",
      "slug": "michael-van-gerwen",
      "nickname": "MvG",
      "pdc_ranking": 1
    }
  ]
}
```

---

## World Cup

### POST /worldcup/price

Price a World Cup of Darts team matchup (singles + doubles).

**Request body:**
```json
{
  "round_name": "Round 1 Doubles",
  "team1": {
    "country": "Netherlands",
    "player_a_id": "mvg-uuid",
    "player_b_id": "van-den-bergh-uuid",
    "player_a_three_da": 99.5,
    "player_b_three_da": 96.2
  },
  "team2": {
    "country": "England",
    "player_a_id": "littler-uuid",
    "player_b_id": "smith-uuid",
    "player_a_three_da": 102.1,
    "player_b_three_da": 95.0
  },
  "team1_starts": true
}
```

**Response 200:**
```json
{
  "team1_win": 0.4892,
  "team2_win": 0.5108,
  "format_type": "doubles"
}
```

---

## Monitoring

### GET /monitoring/health

Internal health check with component status.

**Response 200:**
```json
{
  "status": "ok",
  "components": {
    "markov_engine": "ok",
    "calibration": "ok",
    "elo_pipeline": "ok"
  }
}
```

### GET /monitoring/metrics

Prometheus-compatible metrics endpoint.

---

## Health & Readiness

### GET /health

Simple health check (used by load balancers).

**Response 200:**
```json
{"status": "ok", "service": "xg3-darts", "version": "0.1.0"}
```

### GET /ready

Readiness probe — checks database and Redis.

**Response 200 (ready):**
```json
{"status": "ready", "checks": {"database": "ok", "redis": "ok"}}
```

**Response 503 (degraded):**
```json
{"status": "degraded", "checks": {"database": "unavailable", "redis": "ok"}}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": "error_code",
  "message": "Human-readable description",
  "detail": "Optional technical detail"
}
```

| Status | Meaning |
|--------|---------|
| 400 | Bad request — malformed input |
| 403 | Forbidden — insufficient permissions or consent |
| 404 | Not found |
| 422 | Validation error — input fields fail Pydantic validation |
| 500 | Internal server error |
| 501 | Not implemented (feature flag disabled) |
| 503 | Service unavailable (database/Redis down) |

---

## Rate Limits

| Tier | Requests/minute | Burst |
|------|----------------|-------|
| B2B Standard | 600 | 100 |
| B2B Premium | 3000 | 500 |
| Internal | Unlimited | — |

Rate limit headers are included in all responses:
```
X-RateLimit-Limit: 600
X-RateLimit-Remaining: 543
X-RateLimit-Reset: 1710418800
```
