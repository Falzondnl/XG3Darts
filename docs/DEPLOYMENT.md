# XG3 Darts — Deployment Guide (Railway)

## Prerequisites

- Railway account with a project set up
- GitHub repository connected to Railway
- PostgreSQL database provisioned on Railway
- Redis instance provisioned on Railway

---

## Repository Setup

### 1. Create GitHub Repository

```bash
cd E:/DF/XG3V10/darts/XG3Darts
git init
git add .
git commit -m "Initial commit: XG3 Darts pricing engine Sprint 7"
git branch -M main
git remote add origin https://github.com/YOUR_ORG/xg3-darts.git
git push -u origin main
```

### 2. Branch Protection

In GitHub Settings > Branches:
- Protect `main`: require PR reviews, require CI to pass
- Protect `develop`: require CI to pass

---

## Railway Setup

### 1. Create Railway Project

1. Log in to Railway: https://railway.app
2. Click "New Project" > "Deploy from GitHub repo"
3. Select your `xg3-darts` repository
4. Railway detects `Dockerfile` automatically

### 2. Add PostgreSQL

In the Railway project:
1. Click "New" > "Database" > "PostgreSQL"
2. Railway provisions the database and injects `DATABASE_URL` automatically

### 3. Add Redis

1. Click "New" > "Database" > "Redis"
2. Railway injects `REDIS_URL` automatically

### 4. Configure Environment Variables

In Railway project settings > Variables, add:

```
ENVIRONMENT=production
LOG_LEVEL=INFO
SERVICE_NAME=xg3-darts
SERVICE_VERSION=0.1.0

# Database (auto-injected by Railway PostgreSQL plugin)
# DATABASE_URL=postgresql+asyncpg://...

# Redis (auto-injected by Railway Redis plugin)
# REDIS_URL=redis://...

# GDPR
GDPR_PSEUDONYM_SECRET=<generate-64-char-random-string>

# API Keys
DARTCONNECT_API_KEY=<your-dartconnect-key>
OPTIC_ODDS_API_KEY=<your-optic-odds-key>

# Feature flags
ENABLE_LIVE_PRICING=true
ENABLE_GPU_SIMULATION=false

# Enterprise proxy
DARTS_SERVICE_URL=https://xg3-darts.railway.app/api/v1/darts
```

**Critical:** `GDPR_PSEUDONYM_SECRET` must be at least 32 characters and stored in Railway's encrypted variable store, never in source code.

### 5. Configure railway.json

The `railway.json` file controls the deployment configuration. The existing file should contain:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 30,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
```

---

## Database Migrations

Run Alembic migrations after deployment:

```bash
# In Railway CLI (or Railway console)
alembic upgrade head
```

Or add to the startup command:

```
alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Run Initial Migration

```bash
# Local (with production DATABASE_URL set)
alembic upgrade 001_initial_schema
```

### Create New Migration

```bash
alembic revision --autogenerate -m "description_of_change"
alembic upgrade head
```

---

## ELO Backfill (Post-Deployment)

After deploying and migrating the database, run the ELO backfill:

```bash
# From CSV fixture file (does not require live DB access during computation)
python scripts/backfill_elo.py --csv --from-year 2003 --dry-run

# Full backfill to database
python scripts/backfill_elo.py --csv --from-year 2003

# Single pool only
python scripts/backfill_elo.py --csv --pool pdc_mens --from-year 2010
```

---

## Regression Lock (Post-Sprint)

After completing a sprint, lock the checksums of all critical modules:

```bash
python tests/regression/regression_lock_manager.py lock
git add lock_state.json
git commit -m "chore: update regression lock after Sprint 7"
```

CI will verify the lock on every push.

---

## Health Monitoring

### Health Check

```
GET https://xg3-darts.railway.app/health
```

Expected response: `{"status": "ok", "service": "xg3-darts"}`

### Readiness Check

```
GET https://xg3-darts.railway.app/ready
```

Returns 200 when database + Redis are connected. Returns 503 when degraded.

### Metrics (Prometheus)

```
GET https://xg3-darts.railway.app/api/v1/darts/monitoring/metrics
```

---

## Scaling

Railway auto-scales based on resource usage.

For production load:
- Set `--workers 4` in the start command (2 × CPU cores)
- Enable Railway's horizontal scaling (Pro plan)
- Consider separate Railway services for the data ingestion workers

---

## Logging

All logs are structured JSON (structlog) and available in the Railway dashboard.

Log levels:
- `production`: INFO and above
- `staging`: DEBUG
- `development`: DEBUG + console renderer

---

## Environment Secrets Checklist

Before going live, verify all secrets are set:

- [ ] `GDPR_PSEUDONYM_SECRET` — min 32 chars, cryptographically random
- [ ] `DARTCONNECT_API_KEY` — from DartConnect partner portal
- [ ] `OPTIC_ODDS_API_KEY` — from Optic Odds partner portal
- [ ] `DATABASE_URL` — Railway auto-injects (verify format: `postgresql+asyncpg://`)
- [ ] `REDIS_URL` — Railway auto-injects
- [ ] `ENVIRONMENT=production`

---

## Rollback

Railway supports instant rollback to any previous deployment:

1. Go to Railway project > Deployments
2. Select the last stable deployment
3. Click "Redeploy"

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push:

1. **Lint** — ruff code style check
2. **Test** — pytest unit + integration tests (598+ tests must pass)
3. **Regression lock** — verify critical module checksums unchanged
4. **Docker build** — validate container builds successfully (main branch only)
5. **Stress tests** — 500 concurrent requests (main/develop only)

Railway deploys automatically when CI passes on `main`.
