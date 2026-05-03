# ============================================================
# XG3 Darts — Multi-stage Dockerfile for Railway deployment
# ============================================================

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a separate prefix
# Use requirements-prod.txt (lean set) to keep build time under 5 min.
# Full requirements.txt (with torch, pymc, etc.) is for local training only.
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-prod.txt

# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

# Install runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --system --create-home --uid 1000 xg3user

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Set working directory
WORKDIR /app

# Copy application source
COPY --chown=xg3user:xg3user . /app/

# Make startup script executable (must be done as root before user switch)
RUN chmod +x /app/startup.sh

# Switch to non-root user
USER xg3user

# NOTE: HEALTHCHECK directive removed (Hetzner/Coolify migration 2026-05-03).
# Coolify v4 polls Docker health during deploy and rolls back containers reporting
# "starting"; even bumping start_period doesn't help. Runtime health monitored at
# L7 by Caddy/Traefik via /health endpoint.

# Expose port
EXPOSE 8000

# Environment defaults (override via Railway environment variables)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Start: run Alembic migrations then launch API
CMD ["/app/startup.sh"]
# rebuild-trigger-1774596089
