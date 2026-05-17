"""LangGraph state definitions for the AI Orchestrator."""

from __future__ import annotations

import operator
from enum import Enum
from typing import Any, Dict, List, Optional

from typing_extensions import Annotated, TypedDict


class Intent(str, Enum):
    RAG = "rag"
    SQL = "sql"
    CODE = "code"
    RESEARCH = "research"
    CHAT = "chat"
    UNKNOWN = "unknown"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_APPROVAL = "requires_approval"


class OrchestratorState(TypedDict):
    """Explicit state schema – all state is passed through the graph."""

    tenant_id: str
    user_id: str
    is_guest: bool
    session_id: str
    user_query: str
    intent: Optional[str]
    intent_confidence: Optional[float]
    messages: Annotated[List[Dict[str, Any]], operator.add]
    conversation_turn: int
    chat_history: List[Dict[str, Any]]
    memory_loaded_count: int
    uploaded_docs: List[str]
    db_connection: Optional[str]
    db_schema: Optional[Dict[str, Any]]
    workspace_id: str
    uploaded_doc_ids: List[str]
    workspace_documents: List[Dict[str, Any]]
    vector_index_path: Optional[str]
    retrieved_context: Optional[str]
    generated_sql: Optional[str]
    sql_validation_result: Optional[Dict[str, Any]]
    code_to_execute: Optional[str]
    execution_result: Optional[str]
    tool_outputs: Dict[str, Any]
    research_results: Optional[List[Dict[str, Any]]]
    errors: List[str]
    retry_count: int
    max_retries: int
    approved: bool
    approval_required: bool
    approval_reason: Optional[str]
    execution_status: str
    confidence_score: Optional[float]
    risk_level: Optional[str]
    next_node: Optional[str]
    should_continue: bool
    requires_human_input: bool
    final_answer: Optional[str]
    metadata: Dict[str, Any]
    model_used: Optional[str]
    fallback_reason: Optional[str]


def normalize_state(state: Dict[str, Any]) -> OrchestratorState:
    """Guarantee every key exists with a safe default."""
    n: Dict[str, Any] = state.copy()
    n.setdefault("tenant_id", "default")
    n.setdefault("user_id", "guest")
    n["is_guest"] = n.get("is_guest", n["user_id"] == "guest")
    n.setdefault("session_id", "")
    n.setdefault("user_query", "")
    n.setdefault("intent", Intent.UNKNOWN.value)
    n.setdefault("intent_confidence", 0.0)
    n.setdefault("messages", [])
    if not isinstance(n["messages"], list):
        n["messages"] = []
    n.setdefault("conversation_turn", 0)
    n.setdefault("chat_history", [])
    if not isinstance(n["chat_history"], list):
        n["chat_history"] = []
    n.setdefault("memory_loaded_count", 0)
    n.setdefault("uploaded_docs", [])
    if not isinstance(n["uploaded_docs"], list):
        n["uploaded_docs"] = []
    n.setdefault("db_connection", None)
    n.setdefault("db_schema", None)
    n.setdefault("workspace_id", "default")
    n.setdefault("uploaded_doc_ids", [])
    n.setdefault("workspace_documents", [])
    if not isinstance(n["workspace_documents"], list):
        n["workspace_documents"] = []
    n.setdefault("vector_index_path", None)
    n.setdefault("retrieved_context", None)
    n.setdefault("generated_sql", None)
    n.setdefault("sql_validation_result", None)
    n.setdefault("code_to_execute", None)
    n.setdefault("execution_result", None)
    n.setdefault("tool_outputs", {})
    if not isinstance(n["tool_outputs"], dict):
        n["tool_outputs"] = {}
    n.setdefault("research_results", None)
    n.setdefault("errors", [])
    if not isinstance(n["errors"], list):
        n["errors"] = []
    n.setdefault("retry_count", 0)
    n.setdefault("max_retries", 3)
    n.setdefault("approved", False)
    n.setdefault("approval_required", False)
    n.setdefault("approval_reason", None)
    n.setdefault("execution_status", "pending")
    n.setdefault("confidence_score", 0.0)
    n.setdefault("risk_level", None)
    n.setdefault("next_node", None)
    n.setdefault("should_continue", False)
    n.setdefault("requires_human_input", False)
    n.setdefault("final_answer", None)
    n.setdefault("metadata", {})
    if not isinstance(n["metadata"], dict):
        n["metadata"] = {}
    n.setdefault("model_used", None)
    n.setdefault("fallback_reason", None)
    return n  # type: ignore[return-value]


def ensure_intent(state: Dict[str, Any]) -> None:
    if "intent" not in state or state.get("intent") is None:
        state["intent"] = Intent.UNKNOWN.value
        state["intent_confidence"] = 0.0


def ensure_metadata(state: Dict[str, Any]) -> None:
    if "metadata" not in state or not isinstance(state.get("metadata"), dict):
        state["metadata"] = {}


def ensure_errors(state: Dict[str, Any]) -> None:
    if "errors" not in state or not isinstance(state.get("errors"), list):
        state["errors"] = []
