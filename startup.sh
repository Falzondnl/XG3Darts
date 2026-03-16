#!/bin/bash
# XG3 Darts startup script
# Attempts Alembic migration with fast retries; starts app regardless.
set -e

MAX_ATTEMPTS=3
attempt=1

echo "[startup] Running alembic version stamp fix..."
python scripts/alembic_stamp_fix.py

echo "[startup] Attempting DB migration (max ${MAX_ATTEMPTS} tries)..."
while [ $attempt -le $MAX_ATTEMPTS ]; do
    if timeout 15 python -m alembic upgrade head 2>&1; then
        echo "[startup] Migration succeeded."
        break
    fi
    echo "[startup] Migration attempt $attempt/$MAX_ATTEMPTS failed."
    if [ $attempt -lt $MAX_ATTEMPTS ]; then
        sleep 5
    fi
    attempt=$((attempt + 1))
done

if [ $attempt -gt $MAX_ATTEMPTS ]; then
    echo "[startup] WARNING: Migration failed after $MAX_ATTEMPTS attempts. Starting app anyway."
fi

echo "[startup] Starting Optic Odds freshness worker (background)..."
if [ -n "${OPTIC_ODDS_API_KEY}" ]; then
    python data/sources/optic_odds_freshness.py \
        --interval 60 \
        --db-url "${DATABASE_URL}" \
        >> /tmp/optic_odds_freshness.log 2>&1 &
    echo "[startup] Optic Odds freshness worker started (PID $!)."
else
    echo "[startup] OPTIC_ODDS_API_KEY not set — skipping freshness worker."
fi

echo "[startup] Starting API (port ${PORT:-8000})..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 2 \
    --log-level info \
    --access-log
