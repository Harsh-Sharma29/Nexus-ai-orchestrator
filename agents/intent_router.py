"""Intent-based routing for query classification.

Routes user queries to appropriate agents: RAG, SQL, Code, Research, or Chat.
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from state.state import OrchestratorState, Intent
from state.normalize import ensure_metadata, ensure_errors, ensure_intent
from llm.router import LLMRouter


class IntentRouter:
    """Classifies user queries into specific intents for routing."""

    def __init__(self, llm_router: LLMRouter, temperature: float = 0.1):
        """Initialize the intent router.

        Args:
            llm_router: Centralized LLM router (quota-aware)
            temperature: Temperature for classification (low for consistency)
        """
        self._temperature = temperature
        self.llm_router = llm_router
        self.parser = JsonOutputParser()

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an intent classification system for an Enterprise AI Orchestrator.

Classify the user query into one of these intents:
- "rag": Questions about documents, knowledge base queries, document search
- "sql": Database queries, data analysis requests, SQL generation needs
- "code": Code execution requests, data processing scripts, computational tasks
- "research": Web research, current information lookup, external data gathering
- "chat": General conversation, clarifications, follow-up questions

Context: {context_info}

Respond ONLY with valid JSON:
{"intent": "<intent>", "confidence": <0-1>, "reasoning": "<short explanation>"}"""
            ),
            ("human", "Query: {query}\nConversation history:\n{history}")
        ])

    def classify(self, state: OrchestratorState) -> OrchestratorState:
        """Classify the user query and update state.
        
        NEVER raises - always sets intent (default: UNKNOWN if classification fails).
        """

        # CRITICAL: Ensure invariants before classification
        ensure_metadata(state)
        ensure_errors(state)
        ensure_intent(state)  # Ensures intent exists (safety default)

        # Build conversation history context (safe defaults)
        history_context = "No previous conversation"
        try:
            messages = state.get("messages", [])
            if messages:
                recent_messages = messages[-5:]
                history_context = "\n".join(
                    f"{msg.get('role', 'user')}: {msg.get('content', '')[:200]}"
                    for msg in recent_messages
                ) or "No previous conversation"
        except Exception:
            history_context = "No previousconversation"
        
        # CRITICAL: Build context info for better classification
        context_info_parts = []
        
        # Check if documents exist
        uploaded_docs = state.get("uploaded_docs", [])
        workspace_docs = state.get("workspace_documents", [])
        has_docs = bool(uploaded_docs or workspace_docs)
        
        if has_docs:
            doc_count = len(uploaded_docs) + len(workspace_docs)
            context_info_parts.append(f"Documents available: {doc_count} files uploaded")
        else:
            context_info_parts.append("No documents uploaded")
        
        # Check if research tool is available
        context_info_parts.append("Research tool: DuckDuckGo search available")
        
        context_info = " | ".join(context_info_parts)

        # Safe query extraction
        query = state.get("user_query", "")
        if not query:
            # Empty query → unknown intent
            state["intent"] = Intent.UNKNOWN.value
            state["intent_confidence"] = 0.0
            metadata = state.get("metadata", {})
            metadata["intent_reasoning"] = "Empty query"
            state["metadata"] = metadata
            return state

        # Prompt-safe formatting (all variables guaranteed)
        try:
            messages = self.prompt.format_messages(
                query=query,
                history=history_context,
                context_info=context_info,
            )
        except Exception as e:
            # Prompt formatting failure → unknown intent
            state["errors"].append(f"Prompt formatting error: {str(e)}")
            state["intent"] = Intent.UNKNOWN.value
            state["intent_confidence"] = 0.0
            metadata = state.get("metadata", {})
            metadata["intent_reasoning"] = f"Prompt formatting failed: {str(e)}"
            state["metadata"] = metadata
            return state

        # Invoke via centralized router (Gemini → HF fallback)
        try:
            response = self.llm_router.invoke(
                messages,
                state=state,
                temperature=self._temperature,
            )
        except Exception as e:
            # LLM invocation failure → unknown intent
            state["errors"].append(f"LLM invocation error: {str(e)}")
            state["intent"] = Intent.UNKNOWN.value
            state["intent_confidence"] = 0.0
            metadata = state.get("metadata", {})
            metadata["intent_reasoning"] = f"LLM invocation failed: {str(e)}"
            state["metadata"] = metadata
            return state

        # Safely extract text
        text = getattr(response, "content", None)
        if not isinstance(text, str):
            # Hard fallback — never crash intent routing
            state["intent"] = Intent.UNKNOWN.value
            state["intent_confidence"] = 0.0
            metadata = state.get("metadata", {})
            metadata["intent_reasoning"] = "Failed to parse intent response"
            state["metadata"] = metadata
            return state

        # Parse JSON safely
        try:
            parsed = self.parser.parse(text)
        except Exception as e:
            state["intent"] = Intent.UNKNOWN.value
            state["intent_confidence"] = 0.0
            metadata = state.get("metadata", {})
            metadata["intent_reasoning"] = f"Invalid JSON from intent classifier: {str(e)}"
            state["metadata"] = metadata
            return state

        # Extract and validate intent (safe defaults)
        intent = parsed.get("intent") or Intent.UNKNOWN.value
        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (ValueError, TypeError):
            confidence = 0.5
        reasoning = str(parsed.get("reasoning", ""))

        # Validate intent against enum
        valid_intents = {i.value for i in Intent}
        if intent not in valid_intents:
            intent = Intent.UNKNOWN.value
            confidence = 0.0

        # Update state (all keys guaranteed to exist)
        state["intent"] = intent
        state["intent_confidence"] = confidence
        metadata = state.get("metadata", {})
        metadata["intent_reasoning"] = reasoning
        state["metadata"] = metadata

        return state

    def route(self, state: OrchestratorState) -> str:
        """Determine next node based on classified intent.
        
        CRITICAL: UNKNOWN intent defaults to CHAT (safe, user-friendly fallback).
        Fallback handler is ONLY for truly exceptional cases.
        """

        routing_map = {
            Intent.RAG.value: "rag_agent",
            Intent.SQL.value: "sql_agent",
            Intent.CODE.value: "code_agent",
            Intent.RESEARCH.value: "research_agent",
            Intent.CHAT.value: "chat_agent",
            Intent.UNKNOWN.value: "chat_agent",  # CRITICAL: Default to chat, NOT fallback
        }

        return routing_map.get(state.get("intent"), "chat_agent")  # Safe default: chat
