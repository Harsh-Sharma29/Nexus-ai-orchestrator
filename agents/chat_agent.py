"""Chat Agent for general conversation and clarifications.

Handles conversational queries that don't require specialized agents.
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state.state import OrchestratorState
from state.normalize import ensure_metadata, ensure_errors, normalize_state
from llm.router import LLMRouter


class ChatAgent:
    """Chat agent for general conversation."""
    
    def __init__(
        self,
        llm_router: LLMRouter,
        temperature: float = 0.7
    ):
        """Initialize chat agent.
        
        Args:
            llm_router: LLM router for chat
            temperature: Temperature for generation
        """
        self._temperature = temperature
        self.llm_router = llm_router
        self.parser = StrOutputParser()
        
        self.chat_prompt = ChatPromptTemplate.from_messages([
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

User: {query} """)
        ])

    def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute chat agent to answer using general knowledge.
        
        Args:
            state: Current orchestrator state
            
        Returns:
            Updated state with answer
        """
        state = normalize_state(state)  # type: ignore[arg-type]
        
        try:
            # Build conversation history
            history = ""
            if state.get("messages"):
                recent = (state.get("messages") or [])[-5:]
                history = "\n".join([
                    f"{msg.get('role', 'unknown')}: {str(msg.get('content', ''))[:200]}"
                    for msg in recent
                ])
            
            # Get query from state
            query = state.get("user_query") or ""
            
            # Invoke via LLM router
            msgs = self.chat_prompt.format_messages(
                history=history or "No previous conversation",
                query=query
            )
            resp = self.llm_router.invoke(msgs, state=state, temperature=self._temperature)
            answer = getattr(resp, "content", str(resp))
            
            state["final_answer"] = answer
            state["execution_status"] = "completed"
            state["confidence_score"] = 0.8
            
            # Add to conversation history
            state["messages"].append({
                "role": "assistant",
                "content": answer,
                "metadata": {"agent": "chat"}
            })
            
        except Exception as e:
            state = normalize_state(state)  # type: ignore[arg-type]
            state["errors"].append(f"Chat execution error: {str(e)}")
            state["execution_status"] = "failed"
            state["final_answer"] = "I encountered an error. Please try again."
        
        return state