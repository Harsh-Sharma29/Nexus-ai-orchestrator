# Nexus AI Orchestrator — Production-Ready Multi-Container AI Service Mesh

[![Production](https://img.shields.io/badge/status-live-success?style=for-the-badge)](https://nexus-ai-orchestrator.scholarme.in/)
[![Docker](https://img.shields.io/badge/orchestration-docker--compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](./docker-compose.yml)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](./backend/app/main.py)
[![LangGraph](https://img.shields.io/badge/workflow-LangGraph-1C3C3C?style=for-the-badge)](./orchestrator/graph.py)

> **🌐 Live App:** [https://nexus-ai-orchestrator.scholarme.in/](https://nexus-ai-orchestrator.scholarme.in/)

---

## 💡 High-Level System Overview

**Nexus AI Orchestrator** is a centralized **microservices orchestration layer** that separates concerns between presentation and intelligence. The platform decouples the **Streamlit UI runtime** from heavyweight, **asynchronous graph-engineering pipelines** powered by **FastAPI** and **LangGraph**.

| Layer | Responsibility |
|-------|----------------|
| **Frontend (Streamlit)** | Enterprise UI, workspace management, document upload, session UX — communicates only via REST |
| **Backend (FastAPI)** | Async API gateway, request lifecycle, health checks, CORS, persistence orchestration |
| **Graph Engine (LangGraph)** | Intent routing, multi-agent execution (RAG, SQL, Code, Research, Chat), retries, approval gates |
| **Data Plane** | SQLite chat/workspaces, FAISS vector indexes, per-tenant document isolation |

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend  (:8501)                           │
│              Thin client — no LangGraph / LLM imports                     │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  REST (API_BASE_URL)
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend  (:8000)                              │
│         Endpoints · Services · SQLite · RAG · LLM routing                │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              LangGraph Orchestrator (StateGraph workflow)                │
│   IntentRouter → Agent execution → Approval / Retry → Persist context    │
└──────────────────────────────────────────────────────────────────────────┘
```

This architecture keeps the UI **thread-safe and responsive** while graph nodes, embeddings, and LLM calls execute in the **background worker process** — the pattern required for production-grade AI platforms at scale.

---

## 🐳 Production-Grade Architecture Features

### Multi-Container Orchestration

The entire application mesh is **fully containerized** and managed through **`docker-compose`**. Two services — `backend` (FastAPI + LangGraph) and `frontend` (Streamlit) — are built, networked, and started as a single declarative stack. Persistent volumes mount SQLite and FAISS data; shared `./workspaces` binds document uploads across containers.

### Microservice Isolation

**Independent layer separation** ensures the Streamlit UI never loads LangGraph, FAISS, or agent runtimes locally in production. The frontend issues HTTP requests to the API; the backend owns all AI orchestration. This boundary guarantees **thread-safe UI behavior** and eliminates blocking the Streamlit event loop on long-running graph invocations.

### Live Cloud Infrastructure Deployment

Nexus AI is deployed on a **production AWS EC2** instance (`t3.medium`), sized for concurrent API + UI workloads with predictable CPU and memory headroom. Traffic is **reverse-proxied** under a secure **custom corporate subdomain** (`nexus-ai-orchestrator.scholarme.in`), providing TLS termination, stable public routing, and enterprise-ready network mapping without exposing raw container ports to end users.

| Concern | Implementation |
|---------|----------------|
| Compute | AWS EC2 `t3.medium` |
| Packaging | Docker multi-stage builds per service |
| Orchestration | `docker-compose` with health checks & `depends_on` |
| Persistence | Named volume `backend-data` + host `workspaces/` |
| Public access | Reverse proxy → custom subdomain (HTTPS) |

---

## 📂 Repository Directory Layout

```
nexus-ai-orchestrator/
├── docker-compose.yml          # Multi-container stack (backend + frontend)
├── .dockerignore
├── .streamlit/
│   └── config.toml             # Streamlit theme / server settings
├── backend/
│   ├── Dockerfile              # FastAPI + LangGraph worker image
│   ├── requirements.txt
│   └── app/
│       ├── main.py             # FastAPI entrypoint, CORS, routers
│       ├── config.py           # Pydantic settings (.env loading)
│       ├── api/
│       │   └── endpoints.py    # REST API surface
│       ├── agents/
│       │   ├── graph.py        # LangGraph StateGraph (production path)
│       │   └── state.py        # Typed orchestrator state
│       └── services/
│           ├── llm_router.py     # Gemini + Hugging Face fallback
│           ├── rag_service.py    # FAISS / document pipeline
│           └── storage.py        # SQLite persistence layer
├── frontend/
│   ├── Dockerfile              # Lightweight Streamlit-only image
│   ├── requirements.txt
│   └── app.py                  # Decoupled UI (REST client only)
├── agents/                     # Specialized execution agents
│   ├── intent_router.py
│   ├── rag_agent.py
│   ├── sql_agent.py
│   ├── code_agent.py
│   ├── research_agent.py
│   └── chat_agent.py
├── orchestrator/
│   └── graph.py                # Core LangGraph workflow definition
├── state/                      # State schemas & normalization
├── storage/                    # SQLite store utilities
├── llm/                        # LLM routing helpers
├── config/                     # Tenant / workspace configuration
├── workspaces/                 # Per-user document & FAISS storage
├── streamlit_app.py            # Legacy monolithic UI (local dev reference)
├── requirements.txt            # Root dependencies (monolith / scripts)
└── README.md
```

---

## 🛠️ Local Development Setup (Quickstart)

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & Docker Compose v2+
- [Git](https://git-scm.com/)
- Google Gemini API key (required)
- Hugging Face token (optional — enables LLM fallback)

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/nexus-ai-orchestrator.git
cd nexus-ai-orchestrator
```

### 2. Configure environment variables

Create a `.env` file in the **project root** (same directory as `docker-compose.yml`). The backend loads this file automatically; it is listed in `.gitignore` and must never be committed.

```env
# ── Required ──────────────────────────────────────────────────────────
GOOGLE_API_KEY=your-google-gemini-api-key

# ── Optional — LLM fallback when Gemini is unavailable ───────────────
HUGGINGFACEHUB_API_TOKEN=your-huggingface-token
HF_FALLBACK_MODEL=HuggingFaceH4/zephyr-7b-beta

# ── Optional — LangSmith observability ───────────────────────────────
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=autonomous-ai-orchestrator

# ── Optional — tuning (defaults shown) ───────────────────────────────
PRIMARY_LLM_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=models/embedding-001
SQLITE_DB_PATH=/app/data/memory.db
CORS_ORIGINS=*
DEBUG=false
```

| Variable | Required | Purpose |
|----------|----------|---------|
| `GOOGLE_API_KEY` | **Yes** | Primary Gemini LLM access |
| `HUGGINGFACEHUB_API_TOKEN` | No | Fallback LLM via Hugging Face Hub |
| `LANGCHAIN_API_KEY` | No | LangSmith tracing |
| `SQLITE_DB_PATH` | No | Overridden in Compose to `/app/data/memory.db` |

> **Note:** `API_BASE_URL` is set inside `docker-compose.yml` for the frontend (`http://backend:8000`). For bare-metal local UI development against a running API, use `http://localhost:8000`.

### 3. Build and run the stack

```bash
docker-compose up --build
```

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:8501 | Streamlit UI |
| **Backend** | http://localhost:8000 | FastAPI + OpenAPI |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger UI |
| **Health** | http://localhost:8000/api/health | Liveness probe (used by Compose) |

**Useful commands:**

```bash
docker-compose up -d              # Detached mode
docker-compose logs -f backend  # Tail backend logs
docker-compose down             # Stop and remove containers
```

On first boot, the backend waits until its health check passes before the frontend starts — preventing race conditions where the UI calls an API that is not yet ready.

---

<p align="center">
  <strong>Nexus AI Orchestrator</strong> — Enterprise graph orchestration, production container mesh, live on AWS.<br/>
  Built by <strong>Harsh Sharma</strong>
</p>
