"""Centralized LLM routing (quota-aware fallback)."""

from .router import LLMRouter

__all__ = ["LLMRouter"]

"""LLM router with quota-aware fallback.

Centralized LLM router that provides quota-aware fallback from Gemini to Hugging Face Inference.
"""

from llm.router import LLMRouter

__all__ = ["LLMRouter"]
