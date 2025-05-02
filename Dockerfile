# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools required by some packages (e.g. xgboost wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="nonprofit-risk-model"
LABEL org.opencontainers.image.description="IRS nonprofit revocation risk API"

# Non-root user for security
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source (model artefacts mounted at runtime via volume or COPY)
COPY --chown=appuser:appgroup src/       ./src/
COPY --chown=appuser:appgroup tests/     ./tests/

# Optional: copy a pre-trained model; omit if you mount it as a Docker volume
# COPY --chown=appuser:appgroup models/ ./models/

USER appuser

# Expose port and set entry point
EXPOSE 8000

# Uvicorn in production mode — workers scale with CPU count
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
