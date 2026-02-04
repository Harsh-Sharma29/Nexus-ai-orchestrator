"""Quota-aware, enterprise-safe LLM router.

Requirements implemented:
- Centralized router (agents do not call Gemini directly)
- Primary: Gemini 2.5 Flash (ChatGoogleGenerativeAI)
- Fallback: Hugging Face Inference (HuggingFaceEndpoint via langchain-huggingface)
- Fallback triggers ONLY on Google ResourceExhausted (429)
- Records state["model_used"] and state["fallback_reason"]
- Does not silently degrade: provides a short, user-facing note hook via state metadata
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Sequence

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

try:
    # google-api-core is a transitive dep of google generative stack used by langchain-google-genai
    from google.api_core import exceptions as google_exceptions  # type: ignore
except Exception:  # pragma: no cover
    google_exceptions = None  # type: ignore

logger = logging.getLogger(__name__)


class LLMRouter:
    """Centralized LLM router with quota-aware fallback."""

    def __init__(
        self,
        primary_model: str = "gemini-2.5-flash",
        fallback_repo_id: str | None = None,
        primary_temperature: float = 0.7,
        fallback_temperature: float = 0.7,
        enable_fallback: bool = True,
    ) -> None:
        self.primary_model = primary_model
        self.primary_temperature = primary_temperature

        # Allow env override for enterprise ops
        self.fallback_repo_id = fallback_repo_id or os.getenv(
            "HF_FALLBACK_MODEL", "HuggingFaceH4/zephyr-7b-beta"
        )
        self.fallback_temperature = fallback_temperature
        self.enable_fallback = enable_fallback

        self._primary: Optional[ChatGoogleGenerativeAI] = None
        self._fallback: Optional[ChatHuggingFace] = None

    def _get_primary(self) -> ChatGoogleGenerativeAI:
        if self._primary is None:
            self._primary = ChatGoogleGenerativeAI(
                model=self.primary_model,
                temperature=self.primary_temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
        return self._primary

    def _get_fallback(self) -> ChatHuggingFace:
        if self._fallback is None:
            endpoint = HuggingFaceEndpoint(
                repo_id=self.fallback_repo_id,
                temperature=self.fallback_temperature,
                huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
            )
            self._fallback = ChatHuggingFace(llm=endpoint)
        return self._fallback

    def _is_quota_exhausted_429(self, err: Exception) -> bool:
        # Strong signal: ResourceExhausted from google.api_core
        if google_exceptions is not None and isinstance(err, getattr(google_exceptions, "ResourceExhausted", ())):
            return True

        # Defensive: some wrappers stringify status/enum
        s = str(err).lower()
        if "resourceexhausted" in s and "429" in s:
            return True
        if "resource exhausted" in s and "429" in s:
            return True
        if "429" in s and "quota" in s:
            return True
        return False

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke primary model, falling back ONLY on quota (429 ResourceExhausted)."""
        try:
            resp = self._get_primary().invoke(list(messages), **kwargs)
            if state is not None:
                state["model_used"] = self.primary_model
                state["fallback_reason"] = None
            logger.info("LLMRouter used primary=%s", self.primary_model)
            return resp
        except Exception as e:
            if not (self.enable_fallback and self._is_quota_exhausted_429(e)):
                # Non-quota errors must propagate (enterprise-safe)
                raise

            logger.warning("LLMRouter quota hit on primary=%s; falling back to hf=%s", self.primary_model, self.fallback_repo_id)
            resp = self._get_fallback().invoke(list(messages), **kwargs)
            if state is not None:
                state["model_used"] = f"huggingface:{self.fallback_repo_id}"
                state["fallback_reason"] = "ResourceExhausted (429) - Gemini quota limit reached"
                md = state.get("metadata") or {}
                md["fallback_used"] = True
                state["metadata"] = md
            return resp