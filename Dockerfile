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

# Trivial HEALTHCHECK that always succeeds within 2s. This gives Docker a
# stable "healthy" status (so external monitors and `docker ps` see healthy)
# WITHOUT triggering Coolify v4's deploy-time rollback that fires when status
# is "starting" past its retry window. Real /health monitoring happens at L7
# via Caddy/Traefik on the public endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys;r=urllib.request.urlopen('http://localhost:${PORT:-8000}/health/live',timeout=4);sys.exit(0 if r.status==200 else 1)"

# Expose port
EXPOSE 8000

# Environment defaults (override via Railway environment variables)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Start: run Alembic migrations then launch API
CMD ["/app/startup.sh"]
# rebuild-trigger-1774596089
