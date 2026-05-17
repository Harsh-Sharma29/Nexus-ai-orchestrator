"""FastAPI endpoint definitions for the Nexus AI Orchestrator.

Provides:
- POST /api/chat  – main conversational endpoint
- GET  /api/health – liveness probe
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.agents.graph import app_graph
from backend.app.agents.state import normalize_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["orchestrator"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming chat message from the client."""

    user_id: str = Field(default="guest", description="User identifier for session tracking")
    message: str = Field(..., min_length=1, description="The user's message / query")
    workspace_id: str = Field(default="default", description="Workspace context for RAG")
    session_id: Optional[str] = Field(default=None, description="Explicit session/thread ID. Auto-generated if omitted.")
    tenant_id: str = Field(default="default", description="Tenant identifier for multi-tenancy")
    uploaded_docs: List[str] = Field(default_factory=list, description="New document paths to index")


class ChatResponse(BaseModel):
    """Response returned to the client."""

    answer: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    model_used: Optional[str] = None
    fallback_reason: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    execution_status: str = "completed"
    session_id: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "nexus-ai-orchestrator"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness / readiness probe."""
    return HealthResponse()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main conversational endpoint.

    Accepts a user message, runs the full LangGraph orchestrator
    asynchronously, and returns the AI-generated response.
    """
    session_id = req.session_id or str(uuid.uuid4())

    # Build the input state delta
    input_state: Dict[str, Any] = {
        "tenant_id": req.tenant_id,
        "user_id": req.user_id,
        "is_guest": req.user_id == "guest",
        "session_id": session_id,
        "user_query": req.message,
        "workspace_id": req.workspace_id,
        "messages": [],
        "uploaded_docs": req.uploaded_docs,
        "metadata": {},
    }

    normalized = normalize_state(input_state)
    config = {"configurable": {"thread_id": session_id}}

    try:
        final_state = await app_graph.ainvoke(normalized, config=config)
    except Exception as exc:
        logger.exception("Graph invocation failed")
        raise HTTPException(status_code=500, detail=f"Orchestrator error: {exc}")

    final_state = normalize_state(final_state)

    return ChatResponse(
        answer=final_state.get("final_answer") or "No response generated.",
        intent=final_state.get("intent"),
        confidence=final_state.get("intent_confidence"),
        model_used=final_state.get("model_used"),
        fallback_reason=final_state.get("fallback_reason"),
        errors=final_state.get("errors", []),
        execution_status=final_state.get("execution_status", "completed"),
        session_id=session_id,
    )
