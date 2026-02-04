"""State normalization for LangGraph stability.

Ensures ALL state keys exist with safe defaults before graph execution.
Prevents KeyError failures in nodes and prompt templates.
"""

from typing import Dict, Any
from state.state import OrchestratorState, Intent


def normalize_state(state: Dict[str, Any]) -> OrchestratorState:
    """Normalize state to ensure ALL TypedDict keys exist.
    
    This function:
    - Initializes missing keys with safe defaults
    - Guarantees state["intent"] always exists (default: "unknown")
    - Guarantees state["metadata"] always exists (default: {})
    - Guarantees state["errors"] always exists (default: [])
    - Ensures all list/dict fields are initialized
    
    Args:
        state: Raw state dict (may be incomplete)
        
    Returns:
        Fully normalized OrchestratorState with all keys present
    """
    normalized: Dict[str, Any] = state.copy()  # Start with copy to preserve extras
    
    # ── Multi-Tenant Identity (Required) ──
    normalized.setdefault("tenant_id", "default")
    normalized.setdefault("user_id", "guest")
    normalized["is_guest"] = normalized.get("is_guest", normalized["user_id"] == "guest")
    normalized.setdefault("session_id", "")
    
    # ── User Input (Required) ──
    normalized.setdefault("user_query", "")
    normalized.setdefault("intent", Intent.UNKNOWN.value)
    normalized.setdefault("intent_confidence", 0.0)
    
    # ── Conversation Context ──
    normalized.setdefault("messages", [])
    if not isinstance(normalized["messages"], list):
        normalized["messages"] = []
    normalized.setdefault("conversation_turn", 0)
    normalized.setdefault("chat_history", [])
    if not isinstance(normalized["chat_history"], list):
        normalized["chat_history"] = []
    normalized.setdefault("memory_loaded_count", 0)
    
    # ── Data Context ──
    normalized.setdefault("uploaded_docs", [])
    if not isinstance(normalized["uploaded_docs"], list):
         normalized["uploaded_docs"] = []
    normalized.setdefault("db_connection", None)
    normalized.setdefault("db_schema", None)
    
    # ── Workspace ──
    normalized.setdefault("workspace_id", "default")
    normalized.setdefault("uploaded_doc_ids", [])
    normalized.setdefault("workspace_documents", [])
    if not isinstance(normalized["workspace_documents"], list):
        normalized["workspace_documents"] = []
    normalized.setdefault("vector_index_path", None)
    
    # ── Agent & Tool Outputs ──
    normalized.setdefault("retrieved_context", None)
    normalized.setdefault("generated_sql", None)
    normalized.setdefault("sql_validation_result", None)
    normalized.setdefault("code_to_execute", None)
    normalized.setdefault("execution_result", None)
    normalized.setdefault("tool_outputs", {})
    if not isinstance(normalized["tool_outputs"], dict):
        normalized["tool_outputs"] = {}
    normalized.setdefault("research_results", None)
    
    # ── Control & Safety ──
    normalized.setdefault("errors", [])
    if not isinstance(normalized["errors"], list):
        normalized["errors"] = []
    normalized.setdefault("retry_count", 0)
    normalized.setdefault("max_retries", 3)
    normalized.setdefault("approved", False)
    normalized.setdefault("approval_required", False)
    normalized.setdefault("approval_reason", None)
    normalized.setdefault("execution_status", "pending")
    normalized.setdefault("confidence_score", 0.0)
    normalized.setdefault("risk_level", None)
    
    # ── Routing & Flow Control ──
    normalized.setdefault("next_node", None)
    normalized.setdefault("should_continue", False)
    normalized.setdefault("requires_human_input", False)
    
    # ── Final Output ──
    normalized.setdefault("final_answer", None)
    normalized.setdefault("metadata", {})
    if not isinstance(normalized["metadata"], dict):
        normalized["metadata"] = {}
    
    # ── LLM Model Tracking ──
    normalized.setdefault("model_used", None)
    normalized.setdefault("fallback_reason", None)
    
    return normalized  # type: ignore


def ensure_intent(state: Dict[str, Any]) -> None:
    """Ensure intent always exists (invariant enforcement).
    
    Should be called at node entry if intent is required.
    """
    if "intent" not in state or state.get("intent") is None:
        state["intent"] = Intent.UNKNOWN.value
        state["intent_confidence"] = 0.0


def ensure_metadata(state: Dict[str, Any]) -> None:
    """Ensure metadata dict always exists (invariant enforcement)."""
    if "metadata" not in state or not isinstance(state.get("metadata"), dict):
        state["metadata"] = {}


def ensure_errors(state: Dict[str, Any]) -> None:
    """Ensure errors list always exists (invariant enforcement)."""
    if "errors" not in state or not isinstance(state.get("errors"), list):
        state["errors"] = []
