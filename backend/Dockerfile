# ── Nexus AI Orchestrator – Backend ──────────────────────────────
# Build context must be the project root (set via docker-compose)
# so that `from backend.app.*` imports resolve correctly.
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps for faiss-cpu and general build
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend source + root .env
COPY backend/ /app/backend/
COPY .env /app/.env

# Expose FastAPI port
EXPOSE 8000

# Run from project root so `backend.app.*` imports work
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
