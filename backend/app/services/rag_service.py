"""Asynchronous RAG (Retrieval-Augmented Generation) service.

Encapsulates FAISS vector-store loading, document indexing, and
similarity search.  All public methods are ``async def`` so they
integrate cleanly with the async LangGraph nodes.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.config import get_settings

logger = logging.getLogger(__name__)


class RAGService:
    """Async-ready FAISS-backed retrieval service.

    The service lazily initialises embeddings and maintains an in-memory
    cache of workspace-scoped vector stores.
    """

    def __init__(
        self,
        embedding_model: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()
        self._embedding_model_name = embedding_model or settings.EMBEDDING_MODEL
        self._embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.FAISS_CHUNK_SIZE,
            chunk_overlap=chunk_overlap or settings.FAISS_CHUNK_OVERLAP,
            length_function=len,
        )
        # Tenant-scoped vector stores cached in-memory
        self.vector_stores: Dict[str, FAISS] = {}

    # ------------------------------------------------------------------
    # Embeddings (lazy)
    # ------------------------------------------------------------------
    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Lazily initialise the Google embedding model."""
        if self._embeddings is None:
            settings = get_settings()
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self._embedding_model_name,
                google_api_key=settings.GOOGLE_API_KEY,
            )
        return self._embeddings

    # ------------------------------------------------------------------
    # Workspace vector store helpers
    # ------------------------------------------------------------------
    def _cache_key(self, tenant_id: str, user_id: str, workspace_id: str) -> str:
        return f"{tenant_id}::{user_id}::{workspace_id}"

    async def get_or_load_store(
        self,
        tenant_id: str = "default",
        user_id: str = "guest",
        workspace_id: str = "default",
        vector_index_path: Optional[str] = None,
        errors: Optional[List[str]] = None,
    ) -> FAISS:
        """Return an existing store from cache or load from disk."""
        key = self._cache_key(tenant_id, user_id, workspace_id)

        if key in self.vector_stores:
            return self.vector_stores[key]

        # Attempt to load persisted FAISS index from disk
        if vector_index_path and os.path.isdir(vector_index_path):
            try:
                store = await asyncio.to_thread(
                    FAISS.load_local,
                    vector_index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                self.vector_stores[key] = store
                return store
            except Exception as exc:
                msg = f"Failed to load FAISS index at {vector_index_path}: {exc}"
                logger.warning(msg)
                if errors is not None:
                    errors.append(msg)

        # Seed with a placeholder so downstream code never sees an empty store
        store = await asyncio.to_thread(
            FAISS.from_texts, ["No documents loaded yet."], self.embeddings
        )
        self.vector_stores[key] = store
        return store

    # ------------------------------------------------------------------
    # Document indexing
    # ------------------------------------------------------------------
    async def load_documents(
        self,
        doc_paths: List[str],
        tenant_id: str = "default",
        user_id: str = "guest",
        workspace_id: str = "default",
        vector_index_path: Optional[str] = None,
        errors: Optional[List[str]] = None,
    ) -> None:
        """Load, split, embed and index documents into the workspace store."""
        if not doc_paths:
            return

        all_documents: List[Document] = []

        for doc_path in doc_paths:
            if not os.path.exists(doc_path):
                if errors is not None:
                    errors.append(f"Document not found: {doc_path}")
                continue
            try:
                if doc_path.endswith(".pdf"):
                    loader = PyPDFLoader(doc_path)
                elif doc_path.endswith((".txt", ".md")):
                    loader = TextLoader(doc_path)
                else:
                    if errors is not None:
                        errors.append(f"Unsupported file type: {doc_path}")
                    continue

                docs = await asyncio.to_thread(loader.load)
                all_documents.extend(docs)
            except Exception as exc:
                if errors is not None:
                    errors.append(f"Error loading {doc_path}: {exc}")

        if not all_documents:
            return

        splits = self.text_splitter.split_documents(all_documents)

        # Inject metadata for citation support
        for split in splits:
            split.metadata["tenant_id"] = tenant_id
            split.metadata["workspace_id"] = workspace_id
            if "source" in split.metadata:
                split.metadata["filename"] = os.path.basename(split.metadata["source"])
            else:
                split.metadata["filename"] = "unknown_document"

        store = await self.get_or_load_store(
            tenant_id, user_id, workspace_id, vector_index_path, errors
        )
        await asyncio.to_thread(store.add_documents, splits)

        # Persist to disk
        if vector_index_path:
            try:
                os.makedirs(vector_index_path, exist_ok=True)
                await asyncio.to_thread(store.save_local, vector_index_path)
            except Exception as exc:
                if errors is not None:
                    errors.append(f"Failed to save FAISS index to {vector_index_path}: {exc}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    async def search(
        self,
        query: str,
        tenant_id: str = "default",
        user_id: str = "guest",
        workspace_id: str = "default",
        vector_index_path: Optional[str] = None,
        k: int = 5,
        errors: Optional[List[str]] = None,
    ) -> str:
        """Perform async similarity search and return combined context string."""
        store = await self.get_or_load_store(
            tenant_id, user_id, workspace_id, vector_index_path, errors
        )
        docs = await asyncio.to_thread(store.similarity_search, query, k=k)

        context = "\n\n".join(
            f"[Document {i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)
        )
        return context
