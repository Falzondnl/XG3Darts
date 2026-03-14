# XG3 Darts Pricing Engine

Tier-1 sports betting microservice for PDC and WDF darts markets.

## Sprint 1 — Foundation

### What is built
- Full competition format registry (22 formats)
- ELO pipeline (5 pools, K-factor by tier, draw support)
- Database models (SQLAlchemy async)
- Entity resolution across PDC/DartsOrakel/WDF/DartConnect
- R0/R1/R2 coverage regime detection
- GDPR compliance layer
- DartsOrakel + Mastercaller scrapers
- PDC ingest pipeline (279k fixture rows)
- FastAPI app with all route stubs
- 30 unit tests

### Quick start

```bash
cp .env.example .env
# Edit .env — set DATABASE_URL, REDIS_URL

pip install -r requirements.txt

# Run migrations (once DB is up)
alembic upgrade head

# Ingest PDC data
python scripts/ingest_pdc.py

# Ingest DartsOrakel stats
python scripts/ingest_dartsorakel.py

# Backfill ELO ratings
python scripts/backfill_elo.py

# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run tests

```bash
pytest tests/unit/ -v
```

### API base URL
`/api/v1/darts`

### Docs (development only)
`http://localhost:8000/docs`
