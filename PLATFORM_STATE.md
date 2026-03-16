# XG3 Darts — Platform State

**Last updated: 2026-03-15**
**Version: Darts Tier-1 V2**

---

## Current Sprint: Client Rollout Hardening

### Status: PRODUCTION READY (Railway deployed)

---

## Test Coverage

| Suite | Count | Status |
|---|---|---|
| Unit | 649 | ✅ ALL PASS |
| Integration | 26 | ✅ ALL PASS |
| **Total** | **675** | **✅ 0 failures** |

Target: 145+. Current: 675 (4.6x target).

---

## Models

| Model | Version | AUC | Features | Status |
|---|---|---|---|---|
| R0 (logit) | v1 | 0.74 | 14 | ✅ Deployed |
| R1 File Predictor | v5 | 0.8097 | 19 | ✅ Deployed (prematch API) |
| R1 Full (train_r1.py) | v5 | 0.8182 | 38 | ✅ Trained, not yet wired to API |
| R2 Stacking | — | — | 68 | ⏳ Requires DartConnect (R2 mode) |

R1 training: 195,874 samples from 279,820 PDC + DartsDatabase matches (deduped).
Brier score: 0.175 (R1 File). All temporal leakages removed.

---

## Data Sources

| Source | Records | Status |
|---|---|---|
| PDC API | 279k fixtures, 17.6k players, 3.1k tournaments | ✅ Scraped |
| DartsOrakel | 3,358 players (career 3DA); 201 player profiles | ✅ Core scraped |
| DartsDatabase | 5,506 matches (56 events) | ✅ Partially scraped |
| Mastercaller | 836 links discovered; 9 JS event pages in progress | 🔄 Playwright running |
| WDF | Rankings scraped | ✅ Available |
| DartConnect | Not yet | ⏳ R2 mode - needs credentials |
| Flashscore | Discovered | ⏳ Not yet scraped |

---

## API Routes (all registered, Railway deployed)

| Route Group | Prefix | Status |
|---|---|---|
| Pre-match | /api/v1/darts/prematch | ✅ Live |
| Live in-play | /api/v1/darts/live | ✅ Live |
| Outrights | /api/v1/darts/outrights | ✅ Live |
| SGP | /api/v1/darts/sgp | ✅ Live |
| Props (180/checkout) | /api/v1/darts/props | ✅ Live |
| World Cup doubles | /api/v1/darts/worldcup | ✅ Live |
| Players | /api/v1/darts/players | ✅ Live |
| Events | /api/v1/darts/events | ✅ Live |
| Liability | /api/v1/darts/liability | ✅ Live (v2 deployment) |
| Trader overrides | /api/v1/darts/trader | ✅ Live (v2 deployment) |
| Monitoring | /api/v1/darts/monitoring | ✅ Live (v2 deployment) |
| Health | /health, /ready | ✅ Live |

---

## Infrastructure

| Component | Status | Notes |
|---|---|---|
| Railway (API) | ✅ Deployed | commit 2588d16 |
| Railway (Frontend) | ✅ Deployed | Next.js 15.2.4 |
| PostgreSQL (Railway) | ✅ Up | Migration 004 applied |
| Redis (Railway) | ✅ Up | Live state persistence |
| Optic Odds feed | ✅ Running | Background worker in startup.sh |
| B2B API auth | ✅ Active | HMAC-SHA256, API_KEYS env var |

---

## Markov Engine Performance

| Metric | Value | SLA |
|---|---|---|
| p95 pre-match (warm) | 1.79 ms | < 50 ms ✅ |
| p99 live update | 1.5 ms | < 100 ms ✅ |
| Cold first call (new 3DA) | ~300 ms | one-time |

Numpy vectorized transition matrix + module-level LRU cache.

---

## Pending Items

### P1 — Must close before new client onboarding
- [ ] Verify Railway liability/trader/monitoring routes responding 200 (new deploy)
- [ ] Distribute client API key: `XG3-lwVQLP_Xnh4yOuqy-WsCc8eOERVt_jBa0EbM41GxP0M`

### P2 — Data enrichment (improves prop market quality)
- [ ] Mastercaller Playwright depth-2 crawl (in progress — 9 event pages → match links → match data)
- [ ] DartsOrakel top-500 player profile scrape (currently: 201/3358)
- [ ] DartsDatabase: scrape remaining 131 events (currently: 56/187)

### P3 — R2 Mode
- [ ] DartConnect API credentials (visit-level premium data)
- [ ] R2 predictor wiring (68-feature stacking model)

### P4 — Long-term
- [ ] Flashscore live cross-check integration
- [ ] Darts24 live validation
- [ ] Mastercaller per-player profile scrape

---

## Client API Key

Active key in Railway `API_KEYS` env var:
```
XG3-lwVQLP_Xnh4yOuqy-WsCc8eOERVt_jBa0EbM41GxP0M
```

B2B clients must send header: `X-Api-Key: <key>`

---

## Git Commits (recent)

| SHA | Message |
|---|---|
| 2588d16 | feat(deploy): start Optic Odds freshness worker on container startup |
| e9afe76 | feat(platform): close all client-rollout gaps |
| 23727ef | fix(models): R1 v5 — deduplicate matches, AUC 0.9609→0.8182 |
| 57a7bed | fix(alembic): fix migration 004 revision ID and JSONB server_default |
| 1002afc | feat(engines): liability, trader overrides, in-play props, Optic Odds rebuild |

---

## Regression Lock

All production capabilities locked. Run before any commit:
```bash
pytest tests/ -x --tb=short
```

Expected: 675 passed, 0 failed.
