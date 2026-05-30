"""Google Gemini embedding model normalization and resilient client factory."""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

# Deprecated on v1beta — map to current stable text embedding model
_LEGACY_EMBEDDING_ALIASES = {
    "embedding-001": "gemini-embedding-001",
    "text-embedding-004": "gemini-embedding-001",
    "text-embedding-005": "gemini-embedding-001",
    "models/embedding-001": "gemini-embedding-001",
    "models/text-embedding-004": "gemini-embedding-001",
    "models/text-embedding-005": "gemini-embedding-001",
}

# Self-healing fallbacks (LangChain adds ``models/`` when absent — pass bare IDs here)
_EMBEDDING_FALLBACK_CHAIN: List[str] = [
    "gemini-embedding-001",
    "text-embedding-004",
    "embedding-001",
]

_NOT_FOUND_MARKERS = ("404", "not found", "NOT_FOUND", "is not supported", "embedContent")


def _strip_wrapping_quotes(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _collapse_models_prefix(model: str) -> str:
    """Fix ``models/models/...`` double-prefix from env + SDK composition."""
    collapsed = _strip_wrapping_quotes(model)
    while re.match(r"^models/models/", collapsed, re.IGNORECASE):
        collapsed = collapsed[len("models/") :]
    return collapsed


def normalize_embedding_model_name(raw: str) -> str:
    """Return a canonical model string safe for GoogleGenerativeAIEmbeddings.

    LangChain's validator adds ``models/`` only when the name does not already
    start with ``models/``. We collapse duplicate prefixes and remap deprecated IDs.
    """
    collapsed = _collapse_models_prefix(raw or "")
    if not collapsed:
        collapsed = "gemini-embedding-001"

    lowered = collapsed.lower()
    if lowered in _LEGACY_EMBEDDING_ALIASES:
        collapsed = _LEGACY_EMBEDDING_ALIASES[lowered]

    # Prefer bare model id — LangChain applies exactly one ``models/`` prefix
    if collapsed.startswith("models/"):
        return collapsed[len("models/") :]

    return collapsed


def build_embedding_model_candidates(raw: str) -> List[str]:
    """Ordered unique candidates to try (configured name first, then fallbacks)."""
    primary = normalize_embedding_model_name(raw)
    ordered: List[str] = [primary, *_EMBEDDING_FALLBACK_CHAIN]

    seen: set[str] = set()
    unique: List[str] = []
    for name in ordered:
        normalized = normalize_embedding_model_name(name)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _is_not_found_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker.lower() in message for marker in _NOT_FOUND_MARKERS)


def _probe_embedding_client(client: GoogleGenerativeAIEmbeddings) -> None:
    """Raise if the configured model cannot embed (404 / unsupported)."""
    client.embed_query("nexus embedding healthcheck")


def create_google_embeddings(
    configured_model: str,
    google_api_key: str,
    *,
    task_type: Optional[str] = "retrieval_document",
) -> GoogleGenerativeAIEmbeddings:
    """Create GoogleGenerativeAIEmbeddings with prefix sanitization and 404 fallbacks."""
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is required for embedding generation")

    candidates = build_embedding_model_candidates(configured_model)
    last_error: Optional[Exception] = None

    for model_id in candidates:
        kwargs = {
            "model": model_id,
            "google_api_key": google_api_key,
        }
        if task_type:
            kwargs["task_type"] = task_type

        try:
            client = GoogleGenerativeAIEmbeddings(**kwargs)
            resolved = getattr(client, "model", model_id)
            _probe_embedding_client(client)
            logger.info(
                "Initialized Google embeddings (requested=%r, resolved=%r)",
                configured_model,
                resolved,
            )
            return client
        except Exception as exc:
            last_error = exc
            if _is_not_found_error(exc):
                logger.warning(
                    "Embedding model %r not available (%s); trying fallback",
                    model_id,
                    exc,
                )
                continue
            raise

    raise RuntimeError(
        f"All embedding model candidates failed. Last error: {last_error}"
    ) from last_error
