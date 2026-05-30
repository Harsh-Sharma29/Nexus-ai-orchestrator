"""Robust parsing for LLM intent-classification responses."""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from langchain_core.output_parsers import JsonOutputParser

# Self-healing fallback when JSON extraction fails (document queries → RAG path)
INTENT_PARSE_FALLBACK: Dict[str, Any] = {
    "intent": "rag",
    "confidence": 0.9,
    "reasoning": "Fallback routing after intent parse failure",
    "model_used": "gemini-pro",
    "status": "completed",
    "errors": [],
}

_INTENT_ALIASES = {
    "document_search": "rag",
    "doc_search": "rag",
    "documents": "rag",
    "knowledge_base": "rag",
}


def escape_prompt_template_value(value: str) -> str:
    """Escape braces so user/history text cannot break ChatPromptTemplate.format."""
    return value.replace("{", "{{").replace("}", "}}")


def strip_llm_json_markdown(text: str) -> str:
    """Remove markdown fences and surrounding noise from LLM JSON output."""
    raw = text.strip()
    raw = raw.replace("```json", "").replace("```JSON", "").replace("```", "")
    return raw.strip()


def _normalize_intent_value(intent: Any) -> str:
    if not isinstance(intent, str):
        return "unknown"
    normalized = intent.strip().strip('"').strip("'")
    return _INTENT_ALIASES.get(normalized.lower(), normalized)


def parse_intent_llm_response(text: str) -> Dict[str, Any]:
    """Parse intent JSON from raw LLM text with markdown stripping and fallback."""
    raw_response = strip_llm_json_markdown(text)

    try:
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict):
            parsed["intent"] = _normalize_intent_value(parsed.get("intent"))
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        parsed = JsonOutputParser().parse(text)
        if isinstance(parsed, dict):
            parsed["intent"] = _normalize_intent_value(parsed.get("intent"))
            return parsed
    except Exception:
        pass

    # Last resort: extract first JSON object substring
    match = re.search(r"\{[^{}]*\"intent\"[^{}]*\}", raw_response, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                parsed["intent"] = _normalize_intent_value(parsed.get("intent"))
                return parsed
        except json.JSONDecodeError:
            pass

    fallback = dict(INTENT_PARSE_FALLBACK)
    fallback["intent"] = _normalize_intent_value(fallback["intent"])
    return fallback
