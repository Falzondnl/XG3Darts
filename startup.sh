#!/bin/bash
# XG3 Darts startup script — runs DB migrations then starts the API
set -e

echo "[startup] Running Alembic migrations (with retry)..."

MAX_ATTEMPTS=5
WAIT_SECONDS=8
attempt=1

while [ $attempt -le $MAX_ATTEMPTS ]; do
    if python -m alembic upgrade head 2>&1; then
        echo "[startup] Migrations complete."
        break
    fi
    echo "[startup] Migration attempt $attempt/$MAX_ATTEMPTS failed. Waiting ${WAIT_SECONDS}s..."
    sleep $WAIT_SECONDS
    attempt=$((attempt + 1))
    WAIT_SECONDS=$((WAIT_SECONDS * 2))
done

if [ $attempt -gt $MAX_ATTEMPTS ]; then
    echo "[startup] WARNING: All migration attempts failed. Starting app anyway — schema may be stale."
fi

echo "[startup] Starting API..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 2 \
    --log-level info \
    --access-log
