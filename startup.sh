#!/bin/bash
# XG3 Darts startup script — runs DB migrations then starts the API
set -e

echo "[startup] Running Alembic migrations..."
python -m alembic upgrade head

echo "[startup] Migrations complete. Starting API..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 2 \
    --log-level info \
    --access-log
