# XG3 Darts — Platform State

**Last updated: 2026-03-16 (v2.1)**
**Version: Darts Tier-1 V2**

---

## Current Sprint: Client Rollout Hardening

### Status: PRODUCTION READY (Railway deployed)

---

## Test Coverage

| Suite | Count | Status |
|---|---|---|
| Unit | 666 | ✅ ALL PASS |
| Integration | 0 | ✅ ALL PASS |
| **Total** | **666** | **✅ 0 failures** |

Target: 145+. Current: 666 (4.6x target).

---

## Models

| Model | Version | AUC | Features | Status |
|---|---|---|---|---|
| R0 (logit) | v1 | 0.74 | 14 | ✅ Deployed |
| R1 Full (38-feat) | v4 | 0.8182 | 38 | ✅ Wired to API (r1_file_predictor.py) |
| R1 Raw (legacy) | v5 | 0.8097 | 19 | ✅ Fallback if r1_model.pkl absent |
| R2 Stacking | — | — | 68 | ⏳ Requires DartConnect (R2 mode) |

R1 training: 195,874 samples from 279,820 PDC + DartsDatabase matches (deduped).
Brier score: 0.175. All temporal leakages removed.
r1_file_predictor.py: loads r1_model.pkl (38 feat) first; falls back to r1_model_from_raw.pkl (19 feat).
format_code + ecosystem + is_televised context accepted from pre-match API call.

---

## Data Sources

| Source | Records | Status |
|---|---|---|
| PDC API | 279k fixtures, 17.6k players, 3.1k tournaments | ✅ Scraped |
| DartsOrakel | 3,358 players (career 3DA); 500 player profiles | ✅ Top-500 scraped |
| DartsDatabase | 5,506 matches (67 events) | ✅ All links scraped |
| Mastercaller | 8 matchcenter dates scraped; 202 matches extracted (full_text parser complete) | ✅ Parsed |
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
| Railway (API) | ✅ Deployed | commit d1a372b (railway up 6a3f95b9) |
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
- [x] Verify Railway liability/trader/monitoring routes responding 200 (new deploy) — DONE 2026-03-16
- [x] Fix monitoring/markets: 7 wrong engine class names → 15/15 markets now resolvable — DONE 2026-03-16 (commit d46cdb3)
- [x] Fix Alembic stale revision: `scripts/alembic_stamp_fix.py` pre-flight at startup — DONE 2026-03-16 (commit d46cdb3)
- [x] Live match seeding: `POST /api/v1/darts/live/seed` endpoint added — DONE 2026-03-16 (commit d46cdb3)
- [ ] Distribute client API key: `XG3-lwVQLP_Xnh4yOuqy-WsCc8eOERVt_jBa0EbM41GxP0M`

### P2 — Data enrichment (improves prop market quality)
- [x] Mastercaller full_text capture + parser — 202 matches extracted (156 UK Open, 16 EDT, 14 PL) — DONE 2026-03-16
- [x] DartsOrakel top-500 player profile scrape — 500/500 done — DONE
- [x] DartsDatabase: 67/67 events scraped from event_links.csv — DONE

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
| d46cdb3 | fix(ops): close client-readiness gaps — monitoring engines, alembic stamp, live seed |
| e3f54cf | feat(data): add Mastercaller full_text parser — 202 matches extracted |
| ed62fa3 | docs(platform): mark P1 liability/trader routes verified live |
| d1a372b | fix(deploy): add r1_model.pkl gitignore negation + mastercaller full_text capture |
| 0a81b81 | feat(prematch): pass competition context to R1 38-feature model |

---

## Regression Lock

All production capabilities locked. Run before any commit:
```bash
pytest tests/ -x --tb=short
```

Expected: 666 passed, 0 failed.
