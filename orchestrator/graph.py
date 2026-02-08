"""Main LangGraph orchestrator for the AI Orchestrator.

Graph-first design with explicit state passing and intent-based routing.
"""

from typing import Dict, Any
from state.state import OrchestratorState, Intent
from state.normalize import normalize_state, ensure_metadata, ensure_errors, ensure_intent


# LAZY IMPORTS: Heavy modules imported inside class __init__ to prevent startup blocking



class AIOrchestrator:
    """Main orchestrator using LangGraph for workflow management."""
    
    def __init__(
        self,
        llm_model: str = "gemini-2.5-flash",
        enable_checkpointing: bool = True
    ):
        """Initialize the orchestrator.
        
        Args:
            llm_model: Default LLM model for agents
            enable_checkpointing: Enable state checkpointing for recovery
        """
        # LAZY IMPORTS: All heavy modules loaded only when orchestrator is created
        # This prevents Render startup timeout by deferring model loading
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        from llm.router import LLMRouter
        from agents.intent_router import IntentRouter
        from agents.rag_agent import RAGAgent
        from agents.sql_agent import SQLAgent
        from agents.code_agent import CodeAgent
        from agents.research_agent import ResearchAgent
        from agents.chat_agent import ChatAgent
        
        # Store StateGraph and END for use in _build_graph
        self._StateGraph = StateGraph
        self._END = END
        self._MemorySaver = MemorySaver
        
        # Centralized quota-aware LLM router (Gemini primary, HF fallback on 429 only)
        self.llm_router = LLMRouter(primary_model=llm_model)

        # Initialize agents (no direct Gemini calls inside agents; they use the router)
        self.intent_router = IntentRouter(llm_router=self.llm_router)
        self.rag_agent = RAGAgent(llm_router=self.llm_router)
        self.sql_agent = SQLAgent(llm_router=self.llm_router)
        self.code_agent = CodeAgent(llm_router=self.llm_router)
        self.research_agent = ResearchAgent(llm_router=self.llm_router)
        self.chat_agent = ChatAgent(llm_router=self.llm_router)
        
        # Build graph
        self.graph = self._build_graph()
        
        # Compile with checkpointing if enabled
        if enable_checkpointing:
            memory = self._MemorySaver()
            self.app = self.graph.compile(checkpointer=memory)
        else:
            self.app = self.graph.compile()
    
    def _build_graph(self):
        """Build the LangGraph workflow.
        
        Returns:
            Configured StateGraph
        """
        workflow = self._StateGraph(OrchestratorState)
        
        # Add nodes
        workflow.add_node("load_persistent_context", self._load_persistent_context_node)
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("rag_agent", self._rag_node)
        workflow.add_node("sql_agent", self._sql_node)
        workflow.add_node("code_agent", self._code_node)
        workflow.add_node("research_agent", self._research_node)
        workflow.add_node("chat_agent", self._chat_node)
        workflow.add_node("save_persistent_context", self._save_persistent_context_node)
        workflow.add_node("approval_gate", self._approval_gate_node)
        workflow.add_node("retry_handler", self._retry_handler_node)
        workflow.add_node("graceful_fallback", self._graceful_fallback_node)
        workflow.add_node("fallback_handler", self._fallback_node)
        
        # Set entry point
        workflow.set_entry_point("load_persistent_context")
        workflow.add_edge("load_persistent_context", "classify_intent")
        
        # Add conditional routing from intent classification
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_after_classification,
            {
                "rag_agent": "rag_agent",
                "sql_agent": "sql_agent",
                "code_agent": "code_agent",
                "research_agent": "research_agent",
                "chat_agent": "chat_agent",
                "graceful_fallback": "graceful_fallback",
                "fallback_handler": "fallback_handler"
            }
        )
        
        # Add conditional routing from agents (check for approval/retry)
        for agent_name in ["rag_agent", "sql_agent", "code_agent", "research_agent", "chat_agent"]:
            workflow.add_conditional_edges(
                agent_name,
                self._route_after_agent,
                {
                    "approval_gate": "approval_gate",
                    "retry_handler": "retry_handler",
                    "end": "save_persistent_context"
                }
            )
        
        # Approval gate routing
        workflow.add_conditional_edges(
            "approval_gate",
            self._route_after_approval,
            {
                "sql_agent": "sql_agent",
                "code_agent": "code_agent",
                "end": "save_persistent_context"
            }
        )
        
        # Retry handler routing
        workflow.add_conditional_edges(
            "retry_handler",
            self._route_after_retry,
            {
                "sql_agent": "sql_agent",
                "code_agent": "code_agent",
                "end": "save_persistent_context"
            }
        )
        
        # Graceful fallback always ends
        workflow.add_edge("graceful_fallback", "save_persistent_context")
        
        # Fallback handler always ends
        workflow.add_edge("fallback_handler", "save_persistent_context")
        workflow.add_edge("save_persistent_context", self._END)
        
        return workflow

    def _load_persistent_context_node(self, state: OrchestratorState) -> OrchestratorState:
        """Load persisted chat history + workspace docs into state (SQLite)."""
        # LAZY IMPORT: Storage functions loaded on first use
        from storage.sqlite_store import load_chat_messages, list_workspace_documents, upsert_document
        
        state = normalize_state(state)
        # Chat memory - SCOPED BY WORKSPACE
        ws_id = state.get("workspace_id") or "default"
        state["workspace_id"] = ws_id
        
        # Guard session_id to ensure it exists
        if not state.get("session_id"):
            import uuid
            state["session_id"] = str(uuid.uuid4())

        history = load_chat_messages(state["user_id"], ws_id, state["session_id"], limit=50)
        state["chat_history"] = history
        
        # Use persisted history as the initial working message history for agents
        if not state.get("messages"):
            state["messages"] = []
        if history:
            # Normalize to the message shape used by agents
            for m in history:
                state["messages"].append({"role": m["role"], "content": m["content"], "metadata": {"persisted": True}})
        state["memory_loaded_count"] = len(state["messages"])

        # Workspace docs (persisted)
        docs, index_path = list_workspace_documents(state["user_id"], ws_id)
        state["workspace_documents"] = docs
        state["uploaded_doc_ids"] = [d["doc_id"] for d in docs]
        state["vector_index_path"] = index_path

        # If current request includes newly uploaded doc paths, register them to workspace
        # (upload should be "once"; this ensures persistence if caller supplies new docs)
        incoming_paths = state.get("uploaded_docs", [])
        truly_new_paths = []
        
        if incoming_paths:
            import hashlib
            import os
            
            # Normalize existing paths for comparison
            existing_paths = set()
            for d in docs:
                if d.get("file_path"):
                    existing_paths.add(os.path.normpath(os.path.abspath(d["file_path"])))
            
            base_index_path = index_path or os.path.join("workspaces", state["user_id"], ws_id, "faiss_index")
            
            # Filter for truly new documents
            for p in incoming_paths:
                abs_p = os.path.normpath(os.path.abspath(p))
                if abs_p not in existing_paths:
                    truly_new_paths.append(p)
                    # Register in DB
                    doc_id = hashlib.sha256(f"{state['user_id']}|{ws_id}|{abs_p}".encode("utf-8")).hexdigest()[:32]
                    upsert_document(state["user_id"], ws_id, doc_id, abs_p, base_index_path)
            
            # Refresh workspace view after upsert if we added anything
            if truly_new_paths:
                docs, index_path = list_workspace_documents(state["user_id"], ws_id)
                state["workspace_documents"] = docs
                state["uploaded_doc_ids"] = [d["doc_id"] for d in docs]
                state["vector_index_path"] = index_path
        
        # CRITICAL: Only pass NEW documents to the agent to prevent re-indexing
        state["uploaded_docs"] = truly_new_paths

        return state

    def _save_persistent_context_node(self, state: OrchestratorState) -> OrchestratorState:
        """Persist new chat messages written during this run (SQLite)."""
        # LAZY IMPORT: Storage function loaded on first use
        from storage.sqlite_store import append_chat_messages
        
        state = normalize_state(state)
        start = state.get("memory_loaded_count", 0)
        new_msgs = state.get("messages", [])[start:]
        # Persist only role/content, keep timestamps if present
        append_chat_messages(state["user_id"], state["workspace_id"], state["session_id"], new_msgs)
        state["memory_loaded_count"] = len(state.get("messages", []))
        return state
    
    def _classify_intent_node(self, state: OrchestratorState) -> OrchestratorState:
        """Classify user intent."""
        state = normalize_state(state)
        # CRITICAL: Ensure invariants at node entry
        ensure_metadata(state)
        ensure_errors(state)
        ensure_intent(state)  # Ensures intent exists before classification
        
        # Idempotent message append: check if this exact query already exists in recent messages
        current_query = state["user_query"]
        messages = state.get("messages", [])
        
        # Only append if the last user message is different (prevents duplicate on Streamlit reruns)
        should_append = True
        if messages:
            last_msg = messages[-1]
            if last_msg.get("role") == "user" and last_msg.get("content") == current_query:
                should_append = False
        
        if should_append:
            state["conversation_turn"] = state.get("conversation_turn", 0) + 1
            state["messages"].append({
                "role": "user",
                "content": current_query,
                "metadata": {"turn": state["conversation_turn"], "message_id": f"{state['session_id']}-{state['conversation_turn']}"}
            })
        
        # Classify intent (intent router must NEVER raise)
        try:
            state = self.intent_router.classify(state)
        except Exception as e:
            # Intent classification failure must NOT crash the graph
            state["errors"].append(f"Intent classification error: {str(e)}")
            state["intent"] = Intent.UNKNOWN.value
            state["intent_confidence"] = 0.0
        
        return state
    
    def _route_after_classification(self, state: OrchestratorState) -> str:
        """Route after intent classification with tenant validation."""
        # CRITICAL: Ensure invariants before routing
        ensure_metadata(state)
        ensure_errors(state)
        ensure_intent(state)  # Guarantees intent exists
        
        from config.tenant_config import TenantConfigManager
        manager = TenantConfigManager()
        
        intent = state.get("intent", Intent.UNKNOWN.value)  # Safe default
        confidence = state.get("intent_confidence", 0.0)
        tenant_id = state.get("tenant_id", "default")
        query = state.get("user_query", "")
        
        # CRITICAL: Check for research keywords first
        query_lower = query.lower()
        research_keywords = ["weather", "temperature", "current", "today", "now", "latest", "recent", "this week", "this month"]
        if any(keyword in query_lower for keyword in research_keywords):
            # Override to research if query contains research keywords
            state["intent"] = Intent.RESEARCH.value
            state["intent_confidence"] = max(confidence, 0.75)  # Boost confidence
            metadata = state.get("metadata", {})
            metadata["routing_override"] = "keyword_based_research"
            state["metadata"] = metadata
            intent = Intent.RESEARCH.value
            confidence = state["intent_confidence"]
        
        # CRITICAL UX FIX: Low confidence or UNKNOWN → route based on context
        if not intent or intent == Intent.UNKNOWN.value or confidence < 0.4:
            # Check if we should override to RAG due to document availability
            uploaded_docs = state.get("uploaded_docs", [])
            workspace_docs = state.get("workspace_documents", [])
            has_docs = bool(uploaded_docs or workspace_docs)
            
            if has_docs:
                query_lower = query.lower()
                rag_triggers = ["explain", "summarize", "what", "where", "how", "list", "describe", "analysis", "insight"]
                is_rag_request = any(t in query_lower for t in rag_triggers)
                
                if is_rag_request and len(query) > 5:
                    state["intent"] = Intent.RAG.value
                    state["intent_confidence"] = 0.8
                    metadata = state.get("metadata", {})
                    metadata["routing_override"] = "auto_rag_due_to_docs"
                    state["metadata"] = metadata
                    return "rag_agent"
            
            if has_docs and len(query) > 10:
                 state["intent"] = Intent.RAG.value
                 state["intent_confidence"] = 0.55
                 return "rag_agent"

            # Fallback to chat handler (safe default)
            state["intent"] = Intent.CHAT.value
            state["intent_confidence"] = 0.6
            return "fallback_handler"
        
        # Route to requested agent
        if intent == Intent.RAG.value: return "rag_agent"
        if intent == Intent.SQL.value: return "sql_agent"
        if intent == Intent.CODE.value: return "code_agent"
        if intent == Intent.RESEARCH.value: return "research_agent"
        if intent == Intent.CHAT.value: return "chat_agent"
        
        return "fallback_handler"
    
    def _rag_node(self, state: OrchestratorState) -> OrchestratorState:
        state = normalize_state(state)
        return self.rag_agent.execute(state)
    
    def _sql_node(self, state: OrchestratorState) -> OrchestratorState:
        state = normalize_state(state)
        return self.sql_agent.execute(state)
    
    def _code_node(self, state: OrchestratorState) -> OrchestratorState:
        state = normalize_state(state)
        return self.code_agent.execute(state)
    
    def _research_node(self, state: OrchestratorState) -> OrchestratorState:
        state = normalize_state(state)
        return self.research_agent.execute(state)
    
    def _chat_node(self, state: OrchestratorState) -> OrchestratorState:
        state = normalize_state(state)
        return self.chat_agent.execute(state)
    
    def _approval_gate_node(self, state: OrchestratorState) -> OrchestratorState:
        state = normalize_state(state)
        if state.get("approved", False):
            state["execution_status"] = "approved"
            state["should_continue"] = True
            return state
        state["execution_status"] = "requires_approval"
        state["requires_human_input"] = True
        state["should_continue"] = False
        return state
    
    def _route_after_approval(self, state: OrchestratorState) -> str:
        if state.get("approved", False):
            intent = state.get("intent")
            if intent == "sql": return "sql_agent"
            elif intent == "code": return "code_agent"
        return "end"
    
    def _retry_handler_node(self, state: OrchestratorState) -> OrchestratorState:
        state = normalize_state(state)
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        if retry_count < max_retries:
            state["should_continue"] = True
        else:
            state["should_continue"] = False
            state["execution_status"] = "failed"
        return state
    
    def _route_after_retry(self, state: OrchestratorState) -> str:
        if state.get("should_continue", False):
            intent = state.get("intent")
            if intent == "sql": return "sql_agent"
            elif intent == "code": return "code_agent"
        return "end"
    
    def _route_after_agent(self, state: OrchestratorState) -> str:
        if state.get("approval_required", False) and not state.get("approved", False):
            return "approval_gate"
        if state.get("should_continue", False) and state.get("retry_count", 0) < state.get("max_retries", 3):
            return "retry_handler"
        return "end"
    
    def _graceful_fallback_node(self, state: OrchestratorState) -> OrchestratorState:
        state = normalize_state(state)
        # Fallback for blocked agents (tier limits, etc)
        meta = state.get("metadata", {})
        query = state.get("user_query", "")
        blocked_reason = meta.get("blocked_reason", "Agent unavailable")
        
        response = f"I understand you're asking about: '{query}'\n\n"
        response += f"However, execution was blocked: {blocked_reason}.\n"
        response += "I can help with general questions or document queries instead."
        
        state["final_answer"] = response
        state["execution_status"] = "completed"
        state["messages"].append({
            "role": "assistant",
            "content": response,
            "metadata": {"agent": "graceful_fallback"}
        })
        return state
    
    def _fallback_node(self, state: OrchestratorState) -> OrchestratorState:
        """Fallback - route to chat agent for direct answer.
        
        Renamed from _fallback_handler to match graph registration.
        """
        state = normalize_state(state)
        try:
            return self.chat_agent.execute(state)
        except Exception as e:
            state["errors"].append(f"Fallback error: {str(e)}")
            state["final_answer"] = "I apologize, but I encountered an error. Please try again."
            state["execution_status"] = "failed"
            return state

    def invoke(
        self,
        query: str,
        tenant_id: str = "default",
        user_id: str = "guest",
        session_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Invoke the orchestrator with a query."""
        import uuid
        
        # Initialize state delta
        input_state: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "is_guest": user_id == "guest",
            "session_id": session_id or str(uuid.uuid4()),
            "user_query": query,
            # Defaults will be handled by normalize_state, but providing key fields
            "messages": [], 
            "uploaded_docs": [],
            "metadata": {}
        }
        input_state.update(kwargs)
        
        # Normalize state
        normalized_state = normalize_state(input_state)
        
        # Invoke graph
        config = {"configurable": {"thread_id": normalized_state["session_id"]}}
        final_state = self.app.invoke(normalized_state, config)
        
        return normalize_state(final_state)
