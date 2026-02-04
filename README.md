# Autonomous Enterprise AI Orchestrator

**A graph-based AI control plane built with LangGraph, LangChain, and Gemini.**

This is an **AI orchestration platform**, not a chatbot. It routes user queries through a LangGraph workflow to specialized agents (RAG, SQL, Code, Research, Chat) with explicit state management, workspace isolation, and persistent chat history.

---

## Table of Contents

- [What This Is (and What It Is NOT)](#what-this-is-and-what-it-is-not)
- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Local Setup](#local-setup)
- [Environment Variables](#environment-variables)
- [How RAG Works](#how-rag-works)
- [Workspaces & Chat Persistence](#workspaces--chat-persistence)
- [Deployment (Render)](#deployment-render)
- [Folder Structure](#folder-structure)
- [Deployment Readiness Checklist](#deployment-readiness-checklist)
- [Potentially Unused Files](#potentially-unused-files)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## What This Is (and What It Is NOT)

### What It Is

- **Graph-first AI orchestration** using LangGraph for explicit workflow control
- **Intent-based routing** to specialized execution agents
- **Multi-tenant architecture** with workspace and chat isolation
- **Production-ready Streamlit UI** with enterprise styling
- **Persistent state** via SQLite (chat history, workspaces, documents)

### What It Is NOT

- A ChatGPT clone or simple chatbot wrapper
- A single-model prompt-response system
- A stateless API (state is explicitly managed per-session)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit UI (streamlit_app.py)            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LangGraph Orchestrator                     │
│                      (orchestrator/graph.py)                    │
├─────────────────────────────────────────────────────────────────┤
│  1. Load Persistent Context (SQLite)                            │
│  2. Intent Classification (IntentRouter)                        │
│  3. Route to Agent → Execute → Approval Gate (if needed)        │
│  4. Retry Logic → Fallback → Save Context                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────┬───────────┼───────────┬───────────┐
        ▼           ▼           ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │   RAG   │ │   SQL   │ │  Code   │ │Research │ │  Chat   │
   │  Agent  │ │  Agent  │ │  Agent  │ │  Agent  │ │  Agent  │
   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

**Flow:**
1. User query enters via Streamlit UI
2. LangGraph loads persistent context (chat history, workspace documents)
3. IntentRouter classifies query → routes to appropriate agent
4. Agent executes task, with approval gates for risky operations
5. Retry handler manages failures; fallback to chat for edge cases
6. Response saved to SQLite and displayed in UI

---

## Features

- **Graph-Based Orchestration** — LangGraph StateGraph with explicit node transitions
- **Intent Routing** — Automatic classification: RAG, SQL, Code, Research, Chat
- **RAG Agent** — Document Q&A with FAISS vector store and Hugging Face embeddings
- **SQL Agent** — Query generation with validation and risk assessment
- **Code Agent** — Sandboxed execution with approval gates for risky operations
- **Research Agent** — DuckDuckGo web search integration
- **Chat Agent** — General conversation and fallback
- **Workspace Isolation** — Per-user, per-workspace document and chat separation
- **Persistent Chat History** — SQLite-backed multi-session conversations
- **Gemini + Hugging Face Fallback** — LLM routing with quota management
- **Enterprise UI** — Professional Streamlit interface with custom styling

---

## Tech Stack

| Component         | Technology                                      |
|-------------------|-------------------------------------------------|
| **Control Plane** | LangGraph                                       |
| **LLM Framework** | LangChain                                       |
| **Primary LLM**   | Google Gemini (`langchain-google-genai`)        |
| **Fallback LLM**  | Hugging Face (`langchain-huggingface`)          |
| **Embeddings**    | `sentence-transformers` (Hugging Face)          |
| **Vector Store**  | FAISS (`faiss-cpu`)                             |
| **Web Search**    | DuckDuckGo (`duckduckgo-search`)                |
| **Database**      | SQLite                                          |
| **UI**            | Streamlit                                       |
| **PDF Processing**| PyPDF                                           |

---

## Local Setup

### Prerequisites

- Python 3.10+
- Google API key (Gemini access)
- Hugging Face API token (optional, for fallback)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "AI Orchestrator"

# Create virtual environment
python -m venv myenv
myenv\Scripts\activate  # Windows
# source myenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run streamlit_app.py
```

The UI will open at `http://localhost:8501`.

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Required
GOOGLE_API_KEY=your-google-api-key

# Optional (enables LLM fallback)
HUGGINGFACEHUB_API_TOKEN=your-huggingface-token

# Optional (LangSmith tracing)
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=autonomous-ai-orchestrator
```

| Variable                   | Required | Purpose                          |
|----------------------------|----------|----------------------------------|
| `GOOGLE_API_KEY`           | Yes      | Gemini LLM access                |
| `HUGGINGFACEHUB_API_TOKEN` | No       | Fallback LLM when Gemini fails   |
| `LANGCHAIN_API_KEY`        | No       | LangSmith tracing/observability  |

---

## How RAG Works

1. **Document Upload** — Users upload PDF/TXT files via the Streamlit sidebar
2. **Chunking** — Documents are split into chunks using LangChain text splitters
3. **Embedding** — Chunks are embedded using `sentence-transformers` (Hugging Face)
4. **Vector Storage** — Embeddings stored in FAISS index (per-workspace isolation)
5. **Retrieval** — On RAG intent, relevant chunks retrieved via similarity search
6. **Generation** — Context + query sent to Gemini for answer synthesis

**Workspace Isolation:** Each workspace maintains its own FAISS index. Documents uploaded to `Workspace A` are not searchable from `Workspace B`.

---

## Workspaces & Chat Persistence

### Workspaces

- Users can create multiple workspaces (e.g., "Project A", "Research")
- Each workspace has isolated:
  - Uploaded documents
  - RAG vector index
  - Chat sessions

### Chat Persistence

- All conversations stored in SQLite (`memory.db`)
- Chat sessions preserved across browser refreshes
- Users can switch between sessions, rename, and delete chats
- Multi-turn context maintained within sessions

**Schema Highlights:**
- `workspaces` — Workspace metadata per user
- `chat_sessions` — Session metadata (name, timestamps)
- `chat_messages` — Individual messages with role/content
- `documents` — Document metadata and vector index paths

---

## Deployment (Render)

### Render Configuration

1. **Create a new Web Service** on Render
2. **Connect your repository**
3. **Configure build settings:**

| Setting           | Value                               |
|-------------------|-------------------------------------|
| **Build Command** | `pip install -r requirements.txt`   |
| **Start Command** | `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0` |
| **Python Version**| 3.10 (via `runtime.txt`)            |

4. **Set environment variables** in Render dashboard:
   - `GOOGLE_API_KEY` (required)
   - `HUGGINGFACEHUB_API_TOKEN` (optional)

### runtime.txt

Create `runtime.txt` in project root:

```
python-3.10.12
```

### Notes

- SQLite database (`memory.db`) is ephemeral on Render free tier
- For production: migrate to PostgreSQL or use Render Disk
- FAISS indices stored in `workspaces/` directory (also ephemeral)

---

## Folder Structure

```
AI Orchestrator/
├── agents/                    # Specialized execution agents
│   ├── __init__.py
│   ├── intent_router.py       # Query classification
│   ├── rag_agent.py           # Document Q&A
│   ├── sql_agent.py           # SQL generation + validation
│   ├── code_agent.py          # Sandboxed code execution
│   ├── research_agent.py      # Web search (DuckDuckGo)
│   └── chat_agent.py          # General conversation
├── orchestrator/              # LangGraph workflow
│   ├── __init__.py
│   └── graph.py               # Main StateGraph definition
├── state/                     # State management
│   ├── __init__.py
│   ├── state.py               # OrchestratorState TypedDict
│   └── normalize.py           # State normalization utilities
├── storage/                   # Persistence layer
│   ├── __init__.py
│   └── sqlite_store.py        # SQLite operations
├── config/                    # Configuration
│   ├── __init__.py
│   └── tenant_config.py       # Multi-tenant settings
├── llm/                       # LLM routing
│   ├── __init__.py
│   └── router.py              # Gemini → HuggingFace fallback
├── workspaces/                # Per-user document storage
├── streamlit_app.py           # Main UI application
├── requirements.txt           # Python dependencies
├── memory.db                  # SQLite database (auto-created)
├── .env                       # Environment variables (not committed)
└── README.md                  # This file
```

---

## Deployment Readiness Checklist

| Item                          | Status | Notes                                      |
|-------------------------------|--------|--------------------------------------------|
| `requirements.txt` complete   | ✅     | All production dependencies listed         |
| `runtime.txt` for Python      | ⚠️     | **Create `runtime.txt` with `python-3.10.12`** |
| Streamlit start command       | ✅     | `streamlit run streamlit_app.py`           |
| Environment variables documented | ✅  | `GOOGLE_API_KEY` required                  |
| No hardcoded secrets          | ✅     | All secrets in `.env` / environment        |
| `.gitignore` configured       | ✅     | Excludes `.env`, `myenv/`, `__pycache__/`  |
| Render-compatible             | ✅     | Stateless-friendly with SQLite             |

**Action Required:**
- Create `runtime.txt` with `python-3.10.12` for explicit Python version on Render

---

## Potentially Unused Files

The following files are not imported by the main application (`streamlit_app.py` → orchestrator). They appear to be development/testing utilities:

| File                  | Reason                                         | Recommendation        |
|-----------------------|------------------------------------------------|-----------------------|
| `audit_imports.py`    | Standalone script to verify imports work       | SAFE TO DELETE        |
| `example_usage.py`    | Demo script showing API usage                  | REVIEW BEFORE DELETE  |
| `test_orchestrator.py`| Test script for orchestrator validation        | REVIEW BEFORE DELETE  |
| `verify_project.py`   | Verification script for state/approval flows   | REVIEW BEFORE DELETE  |

**Notes:**
- `example_usage.py`, `test_orchestrator.py`, and `verify_project.py` are useful for development but are not required at runtime
- If keeping for reference, consider moving to a `scripts/` or `tests/` directory
- None of these files are imported by production code paths

---

## Limitations

- **Vector Store** — In-memory FAISS; ephemeral on serverless deployments
- **Database** — SQLite; not suitable for high-concurrency production
- **Code Sandboxing** — Basic `exec()` sandbox; not fully isolated
- **Approval System** — Flag-based; no external workflow integration
- **Embeddings** — Loaded at runtime; cold start latency on first query

---

## Future Improvements

- PostgreSQL persistence for production scalability
- Pinecone/Weaviate for managed vector storage
- Docker-based code sandboxing
- Async agent execution for concurrent operations
- OAuth/SSO integration for enterprise auth
- Redis caching for frequent queries
- LangSmith integration for production observability

---

## Author

**Harsh Sharma**

Built as a portfolio project demonstrating enterprise-grade AI orchestration with LangGraph, LangChain, and modern LLM techniques.

---

*This project is designed for demonstration and educational purposes. Production deployment requires additional security hardening and infrastructure considerations.*
