from typing import List, Dict, Optional, Any
from typing_extensions import TypedDict, Annotated
import operator
from enum import Enum


class Intent(str, Enum):
    """Supported intent types for routing."""
    RAG = "rag"
    SQL = "sql"
    CODE = "code"
    RESEARCH = "research"
    CHAT = "chat"
    UNKNOWN = "unknown"


class ExecutionStatus(str, Enum):
    """Execution status tracking."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_APPROVAL = "requires_approval"


class OrchestratorState(TypedDict):
    """Explicit state schema for the AI Orchestrator.
    
    All state is passed explicitly through the graph - no global state.
    """
    # ── Multi-Tenant Identity ──
    tenant_id: str               # Tenant identifier for isolation
    user_id: str                 # User identifier within tenant
    is_guest: bool
    session_id: str

    # ── User Input ──
    user_query: str
    intent: Optional[str]        # rag | sql | code | research | chat
    intent_confidence: Optional[float]  # Confidence score for intent classification

    # ── Conversation Context ──
    messages: Annotated[List[Dict[str, Any]], operator.add]  # Chat history (role, content, metadata)
    conversation_turn: int       # Turn number in conversation
    chat_history: List[Dict[str, Any]]  # Persisted chat history loaded from SQLite (inspectable)
    memory_loaded_count: int     # Number of messages loaded from SQLite at start of run

    # ── Data Context (Tenant-Scoped) ──
    uploaded_docs: List[str]     # Tenant-scoped document IDs/paths
    db_connection: Optional[str] # Tenant-scoped DB connection string (read-only)
    db_schema: Optional[Dict[str, Any]]  # Cached schema for SQL generation

    # ── Workspace (Persistent Documents) ──
    workspace_id: str
    uploaded_doc_ids: List[str]          # Stable doc IDs in workspace
    workspace_documents: List[Dict[str, Any]]  # Records from SQLite: {doc_id, file_path, vector_index_path, ...}
    vector_index_path: Optional[str]     # Workspace FAISS index directory on disk

    # ── Agent & Tool Outputs ──
    retrieved_context: Optional[str]  # RAG retrieved context
    generated_sql: Optional[str]      # Generated SQL query
    sql_validation_result: Optional[Dict[str, Any]]  # SQL validation details
    code_to_execute: Optional[str]     # Code awaiting execution
    execution_result: Optional[str]    # Execution output
    tool_outputs: Dict[str, Any]       # Tool execution results
    research_results: Optional[List[Dict[str, Any]]]  # Research findings

    # ── Control & Safety ──
    errors: List[str]            # Accumulated errors
    retry_count: int             # Current retry attempt
    max_retries: int             # Maximum retry attempts
    approved: bool               # Human approval flag
    approval_required: bool      # Whether approval is needed
    approval_reason: Optional[str]  # Why approval is required
    execution_status: str        # ExecutionStatus enum value
    confidence_score: Optional[float]  # Overall confidence score
    risk_level: Optional[str]    # low | medium | high

    # ── Routing & Flow Control ──
    next_node: Optional[str]     # Next node to execute
    should_continue: bool        # Whether to continue execution
    requires_human_input: bool   # Whether human input is needed

    # ── Final Output ──
    final_answer: Optional[str]
    metadata: Dict[str, Any]     # Additional metadata for response

    # ── LLM Model Tracking (Quota-Aware Routing) ──
    model_used: Optional[str]        # e.g. "gemini-2.5-flash" or "huggingface:<repo_id>"
    fallback_reason: Optional[str]   # set only when fallback is used
    
    # ── LLM Model Tracking ──
    model_used: Optional[str]     # Which LLM model was used (gemini-2.5-flash, huggingface-inference)
    fallback_reason: Optional[str]  # Reason for fallback (if any, e.g. "ResourceExhausted (429)")
