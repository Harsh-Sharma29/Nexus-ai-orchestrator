"""Async LangGraph orchestrator – nodes, routing, and graph compilation.

Every node is ``async def`` and LLM calls use ``ainvoke``.
The compiled graph is exposed as ``app_graph`` for use by the API layer.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from typing import Any, Dict

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.app.agents.state import (
    Intent,
    OrchestratorState,
    ensure_errors,
    ensure_intent,
    ensure_metadata,
    normalize_state,
)
from backend.app.services.llm_router import AsyncLLMRouter
from backend.app.services.rag_service import RAGService
from backend.app.services import storage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared singletons (created once at import time)
# ---------------------------------------------------------------------------
llm_router = AsyncLLMRouter()
rag_service = RAGService()

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intent classification system for an Enterprise AI Orchestrator.

Classify the user query into one of these intents:
- "rag": Questions about documents, knowledge base queries, document search
- "sql": Database queries, data analysis requests, SQL generation needs
- "code": Code execution requests, data processing scripts, computational tasks
- "research": Web research, current information lookup, external data gathering
- "chat": General conversation, clarifications, follow-up questions

Context: {context_info}

Respond ONLY with valid JSON:
{{"intent": "<intent>", "confidence": <0-1>, "reasoning": "<short explanation>"}}"""),
    ("human", "Query: {query}\nConversation history:\n{history}"),
])

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.

Answer questions directly and helpfully.

CRITICAL RULES:
- NEVER explain what RAG/document search is - it's automatic
- NEVER suggest using research tools - they're automatic
- NEVER say "I don't have access" if tools exist
- Focus on answering questions with your knowledge
- Be concise and accurate

If unsure, provide your best answer based on general knowledge."""),
    ("human", """Conversation history:
{history}

User: {query} """),
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based on the provided documents.

Rules:
1. Check if the context contains the exact answer.
   - If YES: Answer efficiently and cite the filename (e.g., [Source: file.pdf]).
2. If the context contains *related* but not exact info:
   - Say: "The document contains related information, but not that exact detail."
   - Then provide the related info or a summary of the context.
3. If the context is relevant to the topic but completely missing the specific detail:
   - Say: "I found the document, but it doesn't seem to contain that specific detail. Here is a summary of what it does cover:"
   - Then summarize the provided context.
4. DO NOT use general knowledge to hallucinate details not in the text.
5. DO NOT say "I cannot answer". Always provide at least a summary of what you DO see."""),
    ("human", """Context from documents:
{context}

Question: {question}

Conversation history:
{history}

Answer:"""),
])

RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant. Answer questions using available information.

Rules:
- If search results are valid: Answer and cite sources.
- If search results are empty or irrelevant:
  1. Say: "Live data unavailable right now."
  2. Then provide your best answer based on general knowledge.
  3. Clearly label it as general knowledge, not current.
- NEVER refuse to answer.
- NEVER explain tool errors/limitations to the user."""),
    ("human", """Query: {query}

Search Results:
{search_results}

Conversation context:
{history}

Provide a comprehensive answer:"""),
])


# ===================================================================
# Graph Nodes (all async)
# ===================================================================

async def load_persistent_context_node(state: OrchestratorState) -> OrchestratorState:
    """Load persisted chat history + workspace docs from SQLite."""
    state = normalize_state(state)
    ws_id = state.get("workspace_id") or "default"
    state["workspace_id"] = ws_id

    if not state.get("session_id"):
        state["session_id"] = str(uuid.uuid4())

    history = await storage.load_chat_messages(state["user_id"], ws_id, state["session_id"], limit=50)
    state["chat_history"] = history

    if not state.get("messages"):
        state["messages"] = []
    if history:
        for m in history:
            state["messages"].append({"role": m["role"], "content": m["content"], "metadata": {"persisted": True}})
    state["memory_loaded_count"] = len(state["messages"])

    docs, index_path = await storage.list_workspace_documents(state["user_id"], ws_id)
    state["workspace_documents"] = docs
    state["uploaded_doc_ids"] = [d["doc_id"] for d in docs]
    state["vector_index_path"] = index_path

    incoming_paths = state.get("uploaded_docs", [])
    truly_new_paths = []

    if incoming_paths:
        existing_paths = set()
        for d in docs:
            if d.get("file_path"):
                existing_paths.add(os.path.normpath(os.path.abspath(d["file_path"])))

        base_index_path = index_path or os.path.join("workspaces", state["user_id"], ws_id, "faiss_index")

        for p in incoming_paths:
            abs_p = os.path.normpath(os.path.abspath(p))
            if abs_p not in existing_paths:
                truly_new_paths.append(p)
                doc_id = hashlib.sha256(f"{state['user_id']}|{ws_id}|{abs_p}".encode()).hexdigest()[:32]
                await storage.upsert_document(state["user_id"], ws_id, doc_id, abs_p, base_index_path)

        if truly_new_paths:
            docs, index_path = await storage.list_workspace_documents(state["user_id"], ws_id)
            state["workspace_documents"] = docs
            state["uploaded_doc_ids"] = [d["doc_id"] for d in docs]
            state["vector_index_path"] = index_path

    state["uploaded_docs"] = truly_new_paths
    return state


async def save_persistent_context_node(state: OrchestratorState) -> OrchestratorState:
    """Persist new chat messages written during this run."""
    state = normalize_state(state)
    start = state.get("memory_loaded_count", 0)
    new_msgs = state.get("messages", [])[start:]
    await storage.append_chat_messages(state["user_id"], state["workspace_id"], state["session_id"], new_msgs)
    state["memory_loaded_count"] = len(state.get("messages", []))
    return state


async def classify_intent_node(state: OrchestratorState) -> OrchestratorState:
    """Classify user intent via LLM."""
    state = normalize_state(state)
    ensure_metadata(state)
    ensure_errors(state)
    ensure_intent(state)

    current_query = state["user_query"]
    messages = state.get("messages", [])

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
            "metadata": {"turn": state["conversation_turn"], "message_id": f"{state['session_id']}-{state['conversation_turn']}"},
        })

    # Build history context
    history_context = "No previous conversation"
    try:
        if messages:
            recent = messages[-5:]
            history_context = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')[:200]}" for msg in recent
            ) or "No previous conversation"
    except Exception:
        history_context = "No previous conversation"

    # Build context info
    uploaded_docs = state.get("uploaded_docs", [])
    workspace_docs = state.get("workspace_documents", [])
    has_docs = bool(uploaded_docs or workspace_docs)
    parts = []
    if has_docs:
        parts.append(f"Documents available: {len(uploaded_docs) + len(workspace_docs)} files uploaded")
    else:
        parts.append("No documents uploaded")
    parts.append("Research tool: DuckDuckGo search available")
    context_info = " | ".join(parts)

    query = state.get("user_query", "")
    if not query:
        state["intent"] = Intent.UNKNOWN.value
        state["intent_confidence"] = 0.0
        state["metadata"]["intent_reasoning"] = "Empty query"
        return state

    try:
        prompt_msgs = INTENT_PROMPT.format_messages(query=query, history=history_context, context_info=context_info)
    except Exception as e:
        state["errors"].append(f"Prompt formatting error: {e}")
        state["intent"] = Intent.UNKNOWN.value
        state["intent_confidence"] = 0.0
        return state

    try:
        response = await llm_router.ainvoke(prompt_msgs, state=state, temperature=0.1)
    except Exception as e:
        state["errors"].append(f"LLM invocation error: {e}")
        state["intent"] = Intent.UNKNOWN.value
        state["intent_confidence"] = 0.0
        return state

    text = getattr(response, "content", None)
    if not isinstance(text, str):
        state["intent"] = Intent.UNKNOWN.value
        state["intent_confidence"] = 0.0
        return state

    try:
        parsed = JsonOutputParser().parse(text)
    except Exception:
        state["intent"] = Intent.UNKNOWN.value
        state["intent_confidence"] = 0.0
        return state

    intent = parsed.get("intent") or Intent.UNKNOWN.value
    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (ValueError, TypeError):
        confidence = 0.5

    valid_intents = {i.value for i in Intent}
    if intent not in valid_intents:
        intent = Intent.UNKNOWN.value
        confidence = 0.0

    state["intent"] = intent
    state["intent_confidence"] = confidence
    state["metadata"]["intent_reasoning"] = str(parsed.get("reasoning", ""))
    return state


def route_after_classification(state: OrchestratorState) -> str:
    """Route to the appropriate agent node."""
    ensure_metadata(state)
    ensure_errors(state)
    ensure_intent(state)

    intent = state.get("intent", Intent.UNKNOWN.value)
    confidence = state.get("intent_confidence", 0.0)
    query = state.get("user_query", "")

    # Research keyword override
    query_lower = query.lower()
    research_kws = ["weather", "temperature", "current", "today", "now", "latest", "recent", "this week", "this month"]
    if any(kw in query_lower for kw in research_kws):
        state["intent"] = Intent.RESEARCH.value
        state["intent_confidence"] = max(confidence, 0.75)
        state.setdefault("metadata", {})["routing_override"] = "keyword_based_research"
        return "research_agent"

    # Low confidence / unknown → context-based routing
    if not intent or intent == Intent.UNKNOWN.value or confidence < 0.4:
        uploaded_docs = state.get("uploaded_docs", [])
        workspace_docs = state.get("workspace_documents", [])
        has_docs = bool(uploaded_docs or workspace_docs)

        if has_docs:
            rag_triggers = ["explain", "summarize", "what", "where", "how", "list", "describe", "analysis", "insight"]
            if any(t in query_lower for t in rag_triggers) and len(query) > 5:
                state["intent"] = Intent.RAG.value
                state["intent_confidence"] = 0.8
                return "rag_agent"
            if len(query) > 10:
                state["intent"] = Intent.RAG.value
                state["intent_confidence"] = 0.55
                return "rag_agent"

        state["intent"] = Intent.CHAT.value
        state["intent_confidence"] = 0.6
        return "fallback_handler"

    routing = {
        Intent.RAG.value: "rag_agent",
        Intent.SQL.value: "sql_agent",
        Intent.CODE.value: "code_agent",
        Intent.RESEARCH.value: "research_agent",
        Intent.CHAT.value: "chat_agent",
    }
    return routing.get(intent, "fallback_handler")


async def rag_node(state: OrchestratorState) -> OrchestratorState:
    """RAG agent node."""
    state = normalize_state(state)
    try:
        # Load new documents if any
        await rag_service.load_documents(
            doc_paths=state.get("uploaded_docs", []),
            tenant_id=state.get("tenant_id", "default"),
            user_id=state.get("user_id", "guest"),
            workspace_id=state.get("workspace_id", "default"),
            vector_index_path=state.get("vector_index_path"),
            errors=state["errors"],
        )

        # Retrieve context
        context = await rag_service.search(
            query=state.get("user_query", ""),
            tenant_id=state.get("tenant_id", "default"),
            user_id=state.get("user_id", "guest"),
            workspace_id=state.get("workspace_id", "default"),
            vector_index_path=state.get("vector_index_path"),
            errors=state["errors"],
        )
        state["retrieved_context"] = context

        if not context or len(context.strip()) < 10:
            state["final_answer"] = "The uploaded documents do not contain this information."
            state["execution_status"] = "completed"
            return state

        history = ""
        if state.get("messages"):
            recent = (state.get("messages") or [])[-3:]
            history = "\n".join(f"{m.get('role','unknown')}: {str(m.get('content',''))[:150]}" for m in recent)

        msgs = RAG_PROMPT.format_messages(
            context=context, question=state.get("user_query", ""), history=history or "No previous conversation",
        )
        resp = await llm_router.ainvoke(msgs, state=state, temperature=0.0)
        answer = getattr(resp, "content", str(resp))

        if state.get("fallback_reason") and not state.get("metadata", {}).get("fallback_notified"):
            state["metadata"]["fallback_notified"] = True
            answer = f"(Note: Gemini quota was reached; using a fallback model.)\n\n{answer}"

        state["final_answer"] = answer
        state["execution_status"] = "completed"
        state["confidence_score"] = state.get("intent_confidence", 0.8)
        state["messages"].append({"role": "assistant", "content": answer, "metadata": {"agent": "rag"}})
    except Exception as e:
        state = normalize_state(state)
        state["errors"].append(f"RAG execution error: {e}")
        state["execution_status"] = "failed"
        state["final_answer"] = "I encountered an error while processing your document query. Please try again."
    return state


async def chat_node(state: OrchestratorState) -> OrchestratorState:
    """Chat agent node."""
    state = normalize_state(state)
    try:
        history = ""
        if state.get("messages"):
            recent = (state.get("messages") or [])[-5:]
            history = "\n".join(f"{m.get('role','unknown')}: {str(m.get('content',''))[:200]}" for m in recent)

        msgs = CHAT_PROMPT.format_messages(history=history or "No previous conversation", query=state.get("user_query", ""))
        resp = await llm_router.ainvoke(msgs, state=state, temperature=0.7)
        answer = getattr(resp, "content", str(resp))

        state["final_answer"] = answer
        state["execution_status"] = "completed"
        state["confidence_score"] = 0.8
        state["messages"].append({"role": "assistant", "content": answer, "metadata": {"agent": "chat"}})
    except Exception as e:
        state = normalize_state(state)
        state["errors"].append(f"Chat execution error: {e}")
        state["execution_status"] = "failed"
        state["final_answer"] = "I encountered an error. Please try again."
    return state


async def research_node(state: OrchestratorState) -> OrchestratorState:
    """Research agent node – web search + LLM synthesis."""
    import asyncio
    state = normalize_state(state)
    try:
        query = state.get("user_query", "")

        # Best-effort web search
        search_results = []
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            tool = DuckDuckGoSearchRun()
            raw = await asyncio.to_thread(tool.run, query)
            if isinstance(raw, str):
                for i, part in enumerate(raw.split("\n\n")[:5]):
                    if part.strip():
                        search_results.append({"title": f"Result {i+1}", "snippet": part[:500], "url": ""})
        except Exception:
            pass

        state["research_results"] = search_results

        history = ""
        if state.get("messages"):
            recent = (state.get("messages") or [])[-3:]
            history = "\n".join(f"{m.get('role','unknown')}: {str(m.get('content',''))[:150]}" for m in recent)

        search_text = "\n\n".join(f"[{i+1}] {r.get('title','')}\n{r.get('snippet','')}" for i, r in enumerate(search_results))
        msgs = RESEARCH_PROMPT.format_messages(query=query, search_results=search_text or "No results found.", history=history or "No previous conversation")
        resp = await llm_router.ainvoke(msgs, state=state, temperature=0.3)
        answer = getattr(resp, "content", str(resp))

        if state.get("fallback_reason") and not state.get("metadata", {}).get("fallback_notified"):
            state["metadata"]["fallback_notified"] = True
            answer = f"(Note: Gemini quota was reached; using a fallback model.)\n\n{answer}"

        state["final_answer"] = answer
        state["execution_status"] = "completed"
        state["confidence_score"] = 0.75
        state["messages"].append({"role": "assistant", "content": answer, "metadata": {"agent": "research", "sources_count": len(search_results)}})
    except Exception as e:
        state = normalize_state(state)
        state["errors"].append(f"Research execution error: {e}")
        state["execution_status"] = "failed"
        state["final_answer"] = "I encountered an error while performing research. Please try again."
    return state


async def sql_node(state: OrchestratorState) -> OrchestratorState:
    """SQL agent node (placeholder – generates SQL via LLM)."""
    state = normalize_state(state)
    try:
        state["final_answer"] = "SQL agent functionality is available. Please configure a database connection."
        state["execution_status"] = "completed"
    except Exception as e:
        state["errors"].append(f"SQL error: {e}")
        state["execution_status"] = "failed"
        state["final_answer"] = "SQL agent error."
    return state


async def code_node(state: OrchestratorState) -> OrchestratorState:
    """Code agent node (placeholder – generates & executes code via LLM)."""
    state = normalize_state(state)
    try:
        state["final_answer"] = "Code execution agent is available. Provide a code task to proceed."
        state["execution_status"] = "completed"
    except Exception as e:
        state["errors"].append(f"Code error: {e}")
        state["execution_status"] = "failed"
        state["final_answer"] = "Code agent error."
    return state


async def approval_gate_node(state: OrchestratorState) -> OrchestratorState:
    state = normalize_state(state)
    if state.get("approved", False):
        state["execution_status"] = "approved"
        state["should_continue"] = True
    else:
        state["execution_status"] = "requires_approval"
        state["requires_human_input"] = True
        state["should_continue"] = False
    return state


async def retry_handler_node(state: OrchestratorState) -> OrchestratorState:
    state = normalize_state(state)
    if state.get("retry_count", 0) < state.get("max_retries", 3):
        state["should_continue"] = True
    else:
        state["should_continue"] = False
        state["execution_status"] = "failed"
    return state


async def graceful_fallback_node(state: OrchestratorState) -> OrchestratorState:
    state = normalize_state(state)
    query = state.get("user_query", "")
    blocked_reason = state.get("metadata", {}).get("blocked_reason", "Agent unavailable")
    response = f"I understand you're asking about: '{query}'\n\nHowever, execution was blocked: {blocked_reason}.\nI can help with general questions or document queries instead."
    state["final_answer"] = response
    state["execution_status"] = "completed"
    state["messages"].append({"role": "assistant", "content": response, "metadata": {"agent": "graceful_fallback"}})
    return state


async def fallback_node(state: OrchestratorState) -> OrchestratorState:
    """Fallback – route to chat for a direct answer."""
    state = normalize_state(state)
    try:
        return await chat_node(state)
    except Exception as e:
        state["errors"].append(f"Fallback error: {e}")
        state["final_answer"] = "I apologize, but I encountered an error. Please try again."
        state["execution_status"] = "failed"
        return state


# ── Routing helpers ──

def route_after_agent(state: OrchestratorState) -> str:
    if state.get("approval_required", False) and not state.get("approved", False):
        return "approval_gate"
    if state.get("should_continue", False) and state.get("retry_count", 0) < state.get("max_retries", 3):
        return "retry_handler"
    return "end"


def route_after_approval(state: OrchestratorState) -> str:
    if state.get("approved", False):
        intent = state.get("intent")
        if intent == "sql":
            return "sql_agent"
        elif intent == "code":
            return "code_agent"
    return "end"


def route_after_retry(state: OrchestratorState) -> str:
    if state.get("should_continue", False):
        intent = state.get("intent")
        if intent == "sql":
            return "sql_agent"
        elif intent == "code":
            return "code_agent"
    return "end"


# ===================================================================
# Build & compile the graph
# ===================================================================

def build_graph() -> Any:
    """Construct and compile the LangGraph workflow."""
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("load_persistent_context", load_persistent_context_node)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("rag_agent", rag_node)
    workflow.add_node("sql_agent", sql_node)
    workflow.add_node("code_agent", code_node)
    workflow.add_node("research_agent", research_node)
    workflow.add_node("chat_agent", chat_node)
    workflow.add_node("save_persistent_context", save_persistent_context_node)
    workflow.add_node("approval_gate", approval_gate_node)
    workflow.add_node("retry_handler", retry_handler_node)
    workflow.add_node("graceful_fallback", graceful_fallback_node)
    workflow.add_node("fallback_handler", fallback_node)

    workflow.set_entry_point("load_persistent_context")
    workflow.add_edge("load_persistent_context", "classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "rag_agent": "rag_agent",
            "sql_agent": "sql_agent",
            "code_agent": "code_agent",
            "research_agent": "research_agent",
            "chat_agent": "chat_agent",
            "graceful_fallback": "graceful_fallback",
            "fallback_handler": "fallback_handler",
        },
    )

    for agent in ["rag_agent", "sql_agent", "code_agent", "research_agent", "chat_agent"]:
        workflow.add_conditional_edges(
            agent,
            route_after_agent,
            {"approval_gate": "approval_gate", "retry_handler": "retry_handler", "end": "save_persistent_context"},
        )

    workflow.add_conditional_edges(
        "approval_gate",
        route_after_approval,
        {"sql_agent": "sql_agent", "code_agent": "code_agent", "end": "save_persistent_context"},
    )
    workflow.add_conditional_edges(
        "retry_handler",
        route_after_retry,
        {"sql_agent": "sql_agent", "code_agent": "code_agent", "end": "save_persistent_context"},
    )

    workflow.add_edge("graceful_fallback", "save_persistent_context")
    workflow.add_edge("fallback_handler", "save_persistent_context")
    workflow.add_edge("save_persistent_context", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Module-level compiled graph – import this from the API layer
app_graph = build_graph()
