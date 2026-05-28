# =============================================================================
# Multi-Agent Orchestrator v6.0.2 — Production Dockerfile
# =============================================================================
# Multi-stage build for the FastAPI orchestrator API.
#
# Usage:
#   docker build -t multi-agent-orchestrator .
#   docker run -p 8000:8000 --env-file .env multi-agent-orchestrator
#
# NOTE: Ollama must be accessible from the container. Either:
#   - Run Ollama on the host and use --network host
#   - Use docker-compose with an Ollama service
# =============================================================================

# --------------- Stage 1: Builder ---------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system deps for native extensions (numpy, scipy, lxml, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# --------------- Stage 2: Runtime ---------------
FROM python:3.11-slim

LABEL maintainer="Multi-Agent Orchestrator Team"
LABEL description="FastAPI Multi-Agent Pipeline with MiroFish ABM Grandmaster"
LABEL version="6.0.2"

WORKDIR /app

# Install minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 libxslt1.1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Remove sensitive files that shouldn't be in the image
RUN rm -f .env credentials.json token.json client_secret*.json || true

# Create directories for runtime data
RUN mkdir -p reports/output logs data/cache

# Health check — verify FastAPI responds
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose API port
EXPOSE 8000

# Run with uvicorn (production settings)
CMD ["python", "-m", "uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--timeout-keep-alive", "120"]
