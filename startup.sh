#!/bin/bash
# XG3 Darts startup script
# Attempts Alembic migration with fast retries; starts app regardless.
set -e

MAX_ATTEMPTS=3
attempt=1

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

echo "[startup] Starting API (port ${PORT:-8000})..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 2 \
    --log-level info \
    --access-log
