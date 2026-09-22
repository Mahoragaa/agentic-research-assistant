# ── Agentic Research Assistant ─────────────────────────────────────────
# Multi-stage Docker build for production deployment

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install system dependencies ──────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────────────
COPY app/ app/
COPY tests/ tests/

# ── Create data directory for ChromaDB persistence ───────────────────
RUN mkdir -p /app/data/chroma_db

# ── Expose Gradio default port ───────────────────────────────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# ── Launch the application ───────────────────────────────────────────
CMD ["python", "-m", "app.main"]
