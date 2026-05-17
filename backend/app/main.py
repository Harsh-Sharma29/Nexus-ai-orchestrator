"""FastAPI application entrypoint for the Nexus AI Orchestrator.

Initialises the app, configures CORS middleware, and includes the
API router. Run with:

    uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.endpoints import router as api_router
from backend.app.config import get_settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
settings = get_settings()

app = FastAPI(
    title="Nexus AI Orchestrator",
    description="Production-grade async LangGraph backend with multi-agent routing, RAG, and conversational memory.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(api_router)


# ---------------------------------------------------------------------------
# Startup / Shutdown hooks
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    logger.info("🚀  Nexus AI Orchestrator starting up …")
    logger.info("   Primary LLM : %s", settings.PRIMARY_LLM_MODEL)
    logger.info("   CORS origins : %s", origins)


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Nexus AI Orchestrator shutting down.")


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Nexus AI Orchestrator is running. Visit /docs for the API."}
