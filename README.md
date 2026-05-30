# Nexus AI Orchestrator — Production-Ready Multi-Container AI Service Mesh

[![Production](https://img.shields.io/badge/status-live-success?style=for-the-badge)](https://nexus-ai-orchestrator.scholarme.in/)
[![Docker](https://img.shields.io/badge/orchestration-docker--compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](./docker-compose.yml)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](./backend/app/main.py)
[![LangGraph](https://img.shields.io/badge/workflow-LangGraph-1C3C3C?style=for-the-badge)](./backend/app/agents/graph.py)

> **🌐 Live App:** [https://nexus-ai-orchestrator.scholarme.in/](https://nexus-ai-orchestrator.scholarme.in/)

### ✨ UI & Execution Preview

![Nexus AI Orchestrator Dashboard](./workspaces/preview.png)

The production dashboard surfaces **real-time asynchronous graph state updates** as LangGraph nodes execute in the FastAPI worker. **Multi-agent intent routing** classifies each query and dispatches it to the appropriate specialist path (RAG, SQL, Code, Research, or Chat) without blocking the Streamlit UI thread.

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
│         LangGraph Orchestrator  (backend/app — StateGraph workflow)      │
│   IntentRouter → Agent execution → Approval / Retry → Persist context    │
└──────────────────────────────────────────────────────────────────────────┘
```

This architecture keeps the UI **thread-safe and responsive** while graph nodes, embeddings, and LLM calls execute in the **background worker process** — the pattern required for production-grade AI platforms at scale.

---

## 🐳 Production-Grade Architecture Features

### Multi-Container Orchestration

The entire application mesh is **fully containerized** and managed through **`docker-compose`**. Two services — `backend` (FastAPI + LangGraph) and `frontend` (Streamlit) — are built, networked, and started as a single declarative stack. Persistent volumes mount SQLite and FAISS data; shared `./workspaces` binds document uploads across containers.

### Microservice Isolation

**Independent layer separation** ensures the Streamlit UI never loads LangGraph, FAISS, or agent runtimes in production. The frontend issues HTTP requests to the API; the **backend worker** owns all AI orchestration under `backend/app/`. This boundary guarantees **thread-safe UI behavior** and eliminates blocking the Streamlit event loop on long-running graph invocations.

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

Containerized production code lives under **`backend/app/`** and **`frontend/`**. Root-level assets are limited to Compose configuration, shared workspace storage, and legacy local-dev references.

```
Nexus-ai-orchestrator/
├── docker-compose.yml              # Multi-container stack (backend + frontend)
├── .dockerignore
├── .streamlit/
│   └── config.toml                 # Streamlit theme / server settings
│
├── backend/                        # FastAPI + LangGraph worker image
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/                        # All core AI orchestration (production path)
│       ├── main.py                 # FastAPI entrypoint, CORS, routers
│       ├── config.py               # Pydantic settings & .env loading
│       ├── api/
│       │   └── endpoints.py        # REST API surface (/api/chat, /api/health)
│       ├── orchestrator/
│       │   └── graph.py            # LangGraph StateGraph workflow
│       ├── agents/
│       │   ├── intent_router.py    # Query classification
│       │   ├── rag_agent.py        # Document Q&A
│       │   ├── sql_agent.py        # SQL generation & validation
│       │   ├── code_agent.py       # Sandboxed code execution
│       │   ├── research_agent.py   # Web search
│       │   └── chat_agent.py       # General conversation & fallback
│       ├── state/
│       │   ├── state.py            # OrchestratorState schemas
│       │   └── normalize.py        # State normalization utilities
│       ├── storage/
│       │   └── sqlite_store.py     # SQLite persistence layer
│       ├── llm/
│       │   └── router.py           # Gemini + Hugging Face fallback routing
│       └── config/
│           └── tenant_config.py    # Multi-tenant / workspace settings
│
├── frontend/                       # Streamlit UI image (REST client only)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
│
├── workspaces/                     # Shared document & FAISS storage (bind-mounted)
│   └── preview.png                 # README dashboard screenshot (optional)
├── streamlit_app.py                # Legacy monolithic UI (local dev reference only)
├── requirements.txt                # Root deps for legacy local scripts
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
git clone https://github.com/Harsh-Sharma29/Nexus-ai-orchestrator.git
cd Nexus-ai-orchestrator
```

### 2. Configure environment variables

Create a `.env` file in the **project root** (same directory as `docker-compose.yml`). The backend loads this file automatically; it is listed in `.gitignore` and must never be committed.

```env
# ── Required ──────────────────────────────────────────────────────────
GOOGLE_API_KEY=your-google-gemini-api-key

# ── Optional — LLM fallback when Gemini is unavailable ─────────────────
HUGGINGFACEHUB_API_TOKEN=your-huggingface-token
HF_FALLBACK_MODEL=HuggingFaceH4/zephyr-7b-beta

# ── Optional — LangSmith observability ─────────────────────────────────
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=autonomous-ai-orchestrator

# ── Optional — tuning (defaults shown) ─────────────────────────────────
PRIMARY_LLM_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=gemini-embedding-001
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
docker-compose up -d               # Detached mode
docker-compose logs -f backend     # Tail backend logs
docker-compose down                # Stop and remove containers
```

On first boot, the backend waits until its health check passes before the frontend starts — preventing race conditions where the UI calls an API that is not yet ready.

---

<p align="center">
  <strong>Nexus AI Orchestrator</strong> — Enterprise graph orchestration, production container mesh, live on AWS.<br/>
  Built by <strong>Harsh Sharma</strong>
</p>
