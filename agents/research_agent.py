"""Research Agent for web-based queries.

Performs web research using search APIs and summarizes findings.
"""

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from state.state import OrchestratorState
from state.normalize import normalize_state
from llm.router import LLMRouter


class ResearchAgent:
    """Research agent for web-based information gathering."""
    
    def __init__(
        self,
        llm_router: LLMRouter,
        temperature: float = 0.3,
        max_results: int = 5
    ):
        """Initialize research agent.
        
        Args:
            llm_model: LLM model for summarization
            temperature: Temperature for generation
            max_results: Maximum search results to process
        """
        self._temperature = temperature
        self.llm_router = llm_router
        self.parser = StrOutputParser()
        self.max_results = max_results
        
        # Initialize search tool
        try:
            self.search_tool = DuckDuckGoSearchRun()
        except Exception:
            # Fallback if DuckDuckGo not available
            self.search_tool = None
        
        self.research_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant. Answer questions using available information.

Rules:
- If search results are valid: Answer and cite sources.
- If search results are empty or irrelevant:
  1. Say: "Live data unavailable right now."
  2. Then provide your best answer based on general knowledge (historical data/definitions).
  3. clearly label it as general knowledge, not current.
- NEVER refuse to answer.
- NEVER explain tool errors/limitations to the user."""),
            ("human", """Query: {query}

Search Results:
{search_results}

Conversation context:
{history}

Provide a comprehensive answer:""")
        ])

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search.
        
        Args:
            query: Search query
            
        Returns:
            List of search results (empty if search unavailable/fails)
        """
        if not self.search_tool:
            return []
        
        try:
            results = self.search_tool.run(query)
            
            # Parse results (DuckDuckGo returns formatted string)
            # In production, use structured search APIs for better parsing
            results_list = []
            
            # Simple parsing (DuckDuckGo format may vary)
            if isinstance(results, str):
                # Split by results if possible
                parts = results.split('\n\n')[:self.max_results]
                for i, part in enumerate(parts):
                    if part.strip():
                        results_list.append({
                            "title": f"Result {i+1}",
                            "snippet": part[:500],  # Limit snippet length
                            "url": ""
                        })
            else:
                results_list = [{"title": "Search Result", "snippet": str(results), "url": ""}]
            
            return results_list[:self.max_results]
        except Exception:
            # Silently fail - LLM will answer using general knowledge
            return []
    
    def synthesize(self, state: OrchestratorState, query: str, results: List[Dict[str, Any]], history: str) -> str:
        """Synthesize research results into answer.
        
        Args:
            query: Original query
            results: Search results
            history: Conversation history
            
        Returns:
            Synthesized answer
        """
        # Format search results
        search_results_text = "\n\n".join([
            f"[{i+1}] {r.get('title', 'Untitled')}\n{r.get('snippet', '')}"
            for i, r in enumerate(results)
        ])
        
        msgs = self.research_prompt.format_messages(
            query=query,
            search_results=search_results_text,
            history=history or "No previous conversation",
        )
        resp = self.llm_router.invoke(msgs, state=state, temperature=self._temperature)
        answer_text = getattr(resp, "content", str(resp))
        return answer_text
    
    def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute research agent workflow.
        
        Args:
            state: Orchestrator state
            
        Returns:
            Updated state with research results
        """
        # CRITICAL: enforce invariants at node entry (prevents KeyError-class failures)
        state = normalize_state(state)  # type: ignore[arg-type]

        try:
            query = state.get("user_query", "")
            
            # Perform search (best-effort)
            search_results = self.search(query)
            state["research_results"] = search_results
            
            # Build conversation history
            history = ""
            if state.get("messages"):
                recent = (state.get("messages") or [])[-3:]
                history = "\n".join([
                    f"{msg.get('role', 'unknown')}: {str(msg.get('content', ''))[:150]}"
                    for msg in recent
                ])
            
            # Synthesize answer (handles empty results gracefully)
            answer = self.synthesize(state, query, search_results, history)
            # If router set fallback info on state, inform user once.
            if state.get("fallback_reason") and not state.get("metadata", {}).get("fallback_notified"):
                state["metadata"]["fallback_notified"] = True
                answer = f"(Note: Gemini quota was reached; using a fallback model for this response.)\n\n{answer}"
            state["final_answer"] = answer
            state["execution_status"] = "completed"
            state["confidence_score"] = 0.75  # Research has inherent uncertainty
            
            # Add to conversation history
            state["messages"].append({
                "role": "assistant",
                "content": answer,
                "metadata": {
                    "agent": "research",
                    "sources_count": len(search_results)
                }
            })
            
        except Exception as e:
            # Never allow exceptions to escape node execution
            state = normalize_state(state)  # type: ignore[arg-type]
            state["errors"].append(f"Research execution error: {str(e)}")
            state["execution_status"] = "failed"
            state["final_answer"] = "I encountered an error while performing research. Please try again."
        
        return state

