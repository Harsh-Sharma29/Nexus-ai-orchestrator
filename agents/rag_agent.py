"""RAG (Retrieval-Augmented Generation) Agent.

Handles document-based queries using vector embeddings and retrieval.
"""

from typing import Dict, Any, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import os
from state.state import OrchestratorState
from state.normalize import normalize_state
from llm.router import LLMRouter

class RAGAgent:
    """RAG agent for document-based question answering."""
    
    def __init__(
        self,
        llm_router: LLMRouter,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Initialize RAG agent.
  
        Args:
            llm_model: LLM model for generation
            embedding_model: Embedding model for vectorization
            chunk_size: Document chunk size
            chunk_overlap: Overlap between chunks
        """
        self.llm_router = llm_router
        self._embedding_model_name = embedding_model
        self._embeddings = None  # LAZY: Loaded on first use
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # Tenant-scoped vector stores (in-memory for demo, use persistent storage in production)
        self.vector_stores: Dict[str, FAISS] = {}
        
        self.qa_prompt = ChatPromptTemplate.from_messages([
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
5. DO NOT say "I cannot answer". Always provide at least a summary of what you DO see.
"""),
            ("human", """Context from documents:
{context}

Question: {question}

Conversation history:
{history}

Answer:""")
        ])

    @property
    def embeddings(self):
        """Lazily load embeddings on first use (prevents startup blocking)."""
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(model_name=self._embedding_model_name)
        return self._embeddings

    def _get_workspace_store(self, state: OrchestratorState) -> FAISS:
        """Get or create vector store for workspace (persisted on disk).
        
        Behavior:
        - If `state["vector_index_path"]` exists on disk, load it.
        - Else, create a new store and keep it in-memory until saved.
        """
        # Defensive: this helper may be called outside graph context
        state = normalize_state(state)  # type: ignore[arg-type]
        workspace_id = state.get("workspace_id") or "default"
        tenant_id = state.get("tenant_id", "default")
        user_id = state.get("user_id", "guest")
        cache_key = f"{tenant_id}::{user_id}::{workspace_id}"

        if cache_key in self.vector_stores:
            return self.vector_stores[cache_key]

        index_path = state.get("vector_index_path")
        if index_path and os.path.isdir(index_path):
            try:
                store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
                self.vector_stores[cache_key] = store
                return store
            except Exception as e:
                state["errors"].append(f"Failed to load FAISS index at {index_path}: {str(e)}")

        # Create empty store
        store = FAISS.from_texts(["No documents loaded yet."], self.embeddings)
        self.vector_stores[cache_key] = store
        return store
    
    def load_documents(self, state: OrchestratorState) -> None:
        """Load and index documents for the tenant.
        
        Args:
            state: Orchestrator state with uploaded_docs
        """
        state = normalize_state(state)  # type: ignore[arg-type]
        doc_paths = state.get("uploaded_docs", [])
        
        if not doc_paths:
            return
        
        all_documents = []
        
        for doc_path in doc_paths:
            if not os.path.exists(doc_path):
                state["errors"].append(f"Document not found: {doc_path}")
                continue
            
            try:
                # Load based on file extension
                if doc_path.endswith(".pdf"):
                    loader = PyPDFLoader(doc_path)
                elif doc_path.endswith((".txt", ".md")):
                    loader = TextLoader(doc_path)
                else:
                    state["errors"].append(f"Unsupported file type: {doc_path}")
                    continue
                
                docs = loader.load()
                all_documents.extend(docs)
            except Exception as e:
                state["errors"].append(f"Error loading {doc_path}: {str(e)}")
        
        if not all_documents:
            return
        
        # Split documents
        splits = self.text_splitter.split_documents(all_documents)
        
        # Add tenant/workspace/filename metadata to each split
        for split in splits:
            split.metadata["tenant_id"] = state.get("tenant_id", "default")
            split.metadata["workspace_id"] = state.get("workspace_id") or "default"
            # Ensure filename is present for citations
            if "source" in split.metadata:
                split.metadata["filename"] = os.path.basename(split.metadata["source"])
            else:
                split.metadata["filename"] = "unknown_document"
        
        # Create or update workspace store and persist if index path is configured
        store = self._get_workspace_store(state)
        store.add_documents(splits)

        index_path = state.get("vector_index_path")
        if index_path:
            try:
                os.makedirs(index_path, exist_ok=True)
                store.save_local(index_path)
            except Exception as e:
                state["errors"].append(f"Failed to save FAISS index to {index_path}: {str(e)}")
    
    def retrieve(self, state: OrchestratorState, k: int = 5) -> str:
        """Retrieve relevant context for query.
        
        Args:
            state: Orchestrator state
            k: Number of chunks to retrieve
            
        Returns:
            Retrieved context string
        """
        state = normalize_state(state)  # type: ignore[arg-type]
        query = state.get("user_query", "")
        
        store = self._get_workspace_store(state)
        
        # Retrieve relevant documents
        docs = store.similarity_search(query, k=k)
        
        # Combine context
        context = "\n\n".join([
            f"[Document {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])
        
        state["retrieved_context"] = context
        return context
    
    def generate_answer(self, state: OrchestratorState) -> str:
        """Generate answer using RAG.
        
        Args:
            state: Orchestrator state
            
        Returns:
            Generated answer
        """
        state = normalize_state(state)  # type: ignore[arg-type]
        # Ensure documents are loaded
        self.load_documents(state)
        
        # Retrieve context
        try:
            context = self.retrieve(state)
        except Exception as e:
            state["errors"].append(f"Retrieval error: {str(e)}")
            return "I encountered an error searching your documents."

        # HARDENING: If no context retrieved, do not attempt generation
        if not context or "No relevant documents found" in context or len(context.strip()) < 10:
             return "The uploaded documents do not contain this information."
        
        # Build conversation history
        history = ""
        if state.get("messages"):
            recent = (state.get("messages") or [])[-3:]
            history = "\n".join([
                f"{msg.get('role', 'unknown')}: {str(msg.get('content', ''))[:150]}"
                for msg in recent
            ])
        
        # PROMPT SAFETY: Ensure no NoneTypes passed to format_messages
        msgs = self.qa_prompt.format_messages(
            context=context or "No context available.",
            question=state.get("user_query") or "No query provided.",
            history=history or "No previous conversation",
        )
        resp = self.llm_router.invoke(msgs, state=state, temperature=0.0)
        answer_text = getattr(resp, "content", str(resp))

        if state.get("fallback_reason") and not state.get("metadata", {}).get("fallback_notified"):
            state["metadata"]["fallback_notified"] = True
            answer_text = f"(Note: Gemini quota was reached; using a fallback model for this response.)\n\n{answer_text}"

        return answer_text
    
    def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute RAG agent workflow.
        
        Args:
            state: Orchestrator state
            
        Returns:
            Updated state with answer
        """
        # CRITICAL: enforce invariants at node entry (prevents KeyError-class failures)
        state = normalize_state(state)  # type: ignore[arg-type]

        try:
            answer = self.generate_answer(state)
            state["final_answer"] = answer
            state["execution_status"] = "completed"
            state["confidence_score"] = state.get("intent_confidence", 0.8)
            
            # Add to conversation history
            state["messages"].append({
                "role": "assistant",
                "content": answer,
                "metadata": {"agent": "rag", "context_used": bool(state.get("retrieved_context"))}
            })
        except Exception as e:
            state = normalize_state(state)  # type: ignore[arg-type]
            state["errors"].append(f"RAG execution error: {str(e)}")
            state["execution_status"] = "failed"
            state["final_answer"] = "I encountered an error while processing your document query. Please try again."
        
        return state