# ── Nexus AI Orchestrator – Frontend (Streamlit) ────────────────
# Lightweight image – no AI/ML dependencies needed.
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy frontend source
COPY app.py /app/app.py

# Expose Streamlit port
EXPOSE 8501

# Streamlit config: disable CORS warning, set port
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
