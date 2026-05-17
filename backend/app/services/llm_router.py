"""Async quota-aware LLM router.

Mirrors the original synchronous LLMRouter but exposes an ``ainvoke``
method for use inside async LangGraph nodes.

- Primary:  Gemini 2.5 Flash  (ChatGoogleGenerativeAI)
- Fallback: Hugging Face Inference  (triggered ONLY on 429 ResourceExhausted)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from backend.app.config import get_settings

try:
    from google.api_core import exceptions as google_exceptions  # type: ignore
except Exception:  # pragma: no cover
    google_exceptions = None  # type: ignore

logger = logging.getLogger(__name__)


class AsyncLLMRouter:
    """Centralized async LLM router with quota-aware fallback."""

    def __init__(
        self,
        primary_model: str | None = None,
        fallback_repo_id: str | None = None,
        primary_temperature: float = 0.7,
        fallback_temperature: float = 0.7,
        enable_fallback: bool = True,
    ) -> None:
        settings = get_settings()
        self.primary_model = primary_model or settings.PRIMARY_LLM_MODEL
        self.primary_temperature = primary_temperature

        self.fallback_repo_id = fallback_repo_id or settings.HF_FALLBACK_MODEL
        self.fallback_temperature = fallback_temperature
        self.enable_fallback = enable_fallback

        self._primary: Optional[ChatGoogleGenerativeAI] = None
        self._fallback: Optional[ChatHuggingFace] = None

    # ---- lazy model construction ------------------------------------------------

    def _get_primary(self) -> ChatGoogleGenerativeAI:
        if self._primary is None:
            settings = get_settings()
            self._primary = ChatGoogleGenerativeAI(
                model=self.primary_model,
                temperature=self.primary_temperature,
                google_api_key=settings.GOOGLE_API_KEY,
            )
        return self._primary

    def _get_fallback(self) -> ChatHuggingFace:
        if self._fallback is None:
            settings = get_settings()
            token = settings.HUGGINGFACEHUB_API_TOKEN
            endpoint = HuggingFaceEndpoint(
                repo_id=self.fallback_repo_id,
                temperature=self.fallback_temperature,
                huggingfacehub_api_token=token,
            )
            self._fallback = ChatHuggingFace(llm=endpoint)
        return self._fallback

    # ---- 429 detection ----------------------------------------------------------

    @staticmethod
    def _is_quota_exhausted_429(err: Exception) -> bool:
        if google_exceptions is not None and isinstance(
            err, getattr(google_exceptions, "ResourceExhausted", ())
        ):
            return True
        s = str(err).lower()
        if "resourceexhausted" in s and "429" in s:
            return True
        if "resource exhausted" in s and "429" in s:
            return True
        if "429" in s and "quota" in s:
            return True
        return False

    # ---- synchronous invoke (kept for compatibility) ----------------------------

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronous invoke – mirrors the legacy LLMRouter interface."""
        try:
            resp = self._get_primary().invoke(list(messages), **kwargs)
            if state is not None:
                state["model_used"] = self.primary_model
                state["fallback_reason"] = None
            return resp
        except Exception as e:
            if not (self.enable_fallback and self._is_quota_exhausted_429(e)):
                raise
            logger.warning("Quota hit on primary=%s; falling back to hf=%s", self.primary_model, self.fallback_repo_id)
            resp = self._get_fallback().invoke(list(messages), **kwargs)
            if state is not None:
                state["model_used"] = f"huggingface:{self.fallback_repo_id}"
                state["fallback_reason"] = "ResourceExhausted (429) - Gemini quota limit reached"
                md = state.get("metadata") or {}
                md["fallback_used"] = True
                state["metadata"] = md
            return resp

    # ---- async invoke -----------------------------------------------------------

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Async invoke – uses the underlying LangChain ``ainvoke``."""
        try:
            resp = await self._get_primary().ainvoke(list(messages), **kwargs)
            if state is not None:
                state["model_used"] = self.primary_model
                state["fallback_reason"] = None
            logger.info("AsyncLLMRouter used primary=%s", self.primary_model)
            return resp
        except Exception as e:
            if not (self.enable_fallback and self._is_quota_exhausted_429(e)):
                raise
            logger.warning(
                "AsyncLLMRouter quota hit on primary=%s; falling back to hf=%s",
                self.primary_model,
                self.fallback_repo_id,
            )
            resp = await self._get_fallback().ainvoke(list(messages), **kwargs)
            if state is not None:
                state["model_used"] = f"huggingface:{self.fallback_repo_id}"
                state["fallback_reason"] = "ResourceExhausted (429) - Gemini quota limit reached"
                md = state.get("metadata") or {}
                md["fallback_used"] = True
                state["metadata"] = md
            return resp
