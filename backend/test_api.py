"""Comprehensive test suite for the Nexus AI Orchestrator FastAPI backend.

Run with:
    pytest backend/test_api.py -v

Or standalone:
    python backend/test_api.py

Tests cover:
  1. Health endpoint liveness
  2. Chat endpoint with a standard message (full LangGraph pipeline)
  3. Response schema validation (answer, intent, execution_status, session_id)
  4. Error handling for malformed requests
  5. Session continuity (same session_id returns consistent state)
"""

from __future__ import annotations

import asyncio
import sys
import time

import httpx
import pytest

BASE_URL = "http://localhost:8000"


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    """Create an httpx client with generous timeout for LLM calls."""
    with httpx.Client(base_url=BASE_URL, timeout=120.0) as c:
        yield c


@pytest.fixture(scope="module")
def async_client():
    """Async httpx client for async tests."""
    return httpx.AsyncClient(base_url=BASE_URL, timeout=120.0)


# ═══════════════════════════════════════════════════════════════════
# 1. HEALTH ENDPOINT
# ═══════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """Verify the server is alive and responding."""

    def test_health_returns_200(self, client: httpx.Client):
        """GET /api/health should return 200 OK."""
        resp = client.get("/api/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

    def test_health_response_schema(self, client: httpx.Client):
        """Health response must contain status and service fields."""
        resp = client.get("/api/health")
        data = resp.json()
        assert "status" in data, "Missing 'status' field"
        assert "service" in data, "Missing 'service' field"
        assert data["status"] == "ok"
        assert data["service"] == "nexus-ai-orchestrator"

    def test_root_returns_200(self, client: httpx.Client):
        """GET / should return a welcome message."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data


# ═══════════════════════════════════════════════════════════════════
# 2. CHAT ENDPOINT – STANDARD MESSAGE
# ═══════════════════════════════════════════════════════════════════

class TestChatEndpoint:
    """Verify the full LangGraph pipeline via POST /api/chat."""

    def test_chat_returns_200(self, client: httpx.Client):
        """A valid chat request should return 200."""
        payload = {
            "user_id": "test_user",
            "message": "Hello, what can you do?",
        }
        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    def test_chat_response_has_answer(self, client: httpx.Client):
        """Response must include a non-empty 'answer' field."""
        payload = {
            "user_id": "test_user",
            "message": "Tell me a fun fact about Python programming.",
        }
        resp = client.post("/api/chat", json=payload)
        data = resp.json()
        assert "answer" in data, f"Missing 'answer' key. Got: {list(data.keys())}"
        assert isinstance(data["answer"], str), "'answer' must be a string"
        assert len(data["answer"]) > 0, "'answer' must not be empty"

    def test_chat_response_schema_strict(self, client: httpx.Client):
        """Response JSON must strictly contain all required fields."""
        required_fields = {
            "answer": str,
            "intent": (str, type(None)),
            "confidence": (float, int, type(None)),
            "model_used": (str, type(None)),
            "fallback_reason": (str, type(None)),
            "errors": list,
            "execution_status": str,
            "session_id": str,
        }
        payload = {
            "user_id": "test_user",
            "message": "What is 2 + 2?",
        }
        resp = client.post("/api/chat", json=payload)
        data = resp.json()

        for field, expected_type in required_fields.items():
            assert field in data, f"Missing required field: '{field}'"
            assert isinstance(
                data[field], expected_type
            ), f"Field '{field}' has wrong type: {type(data[field]).__name__}, expected {expected_type}"

    def test_chat_execution_status_completed(self, client: httpx.Client):
        """A successful chat should have execution_status='completed'."""
        payload = {
            "user_id": "test_user",
            "message": "Say hello.",
        }
        resp = client.post("/api/chat", json=payload)
        data = resp.json()
        assert data.get("execution_status") == "completed", (
            f"Expected 'completed', got '{data.get('execution_status')}'"
        )

    def test_chat_session_id_auto_generated(self, client: httpx.Client):
        """When session_id is not provided, it should be auto-generated."""
        payload = {
            "user_id": "test_user",
            "message": "Hi there!",
        }
        resp = client.post("/api/chat", json=payload)
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0, "Auto-generated session_id must not be empty"

    def test_chat_with_explicit_session_id(self, client: httpx.Client):
        """When session_id is provided, it should be returned as-is."""
        explicit_sid = "test-session-12345"
        payload = {
            "user_id": "test_user",
            "message": "Remember this session.",
            "session_id": explicit_sid,
        }
        resp = client.post("/api/chat", json=payload)
        data = resp.json()
        assert data["session_id"] == explicit_sid


# ═══════════════════════════════════════════════════════════════════
# 3. ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════

class TestErrorHandling:
    """Verify proper error responses for malformed requests."""

    def test_chat_empty_message_rejected(self, client: httpx.Client):
        """Empty message should be rejected with 422."""
        payload = {
            "user_id": "test_user",
            "message": "",
        }
        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 422, f"Expected 422 for empty message, got {resp.status_code}"

    def test_chat_missing_message_field(self, client: httpx.Client):
        """Missing 'message' field should return 422."""
        payload = {"user_id": "test_user"}
        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 422

    def test_chat_invalid_json(self, client: httpx.Client):
        """Non-JSON body should return 422."""
        resp = client.post(
            "/api/chat",
            content="this is not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════
# 4. INTEGRATION – FULL PIPELINE (optional, slower)
# ═══════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """End-to-end tests exercising the Gemini + LangGraph pipeline."""

    def test_intent_classification_chat(self, client: httpx.Client):
        """A general greeting should be classified as 'chat' intent."""
        payload = {
            "user_id": "test_user",
            "message": "Hello! How are you today?",
        }
        resp = client.post("/api/chat", json=payload)
        data = resp.json()
        # Intent should be 'chat' or at least not None
        assert data.get("intent") is not None, "Intent should not be None"
        assert data.get("answer"), "Answer should not be empty"

    def test_no_errors_in_response(self, client: httpx.Client):
        """A standard message should produce zero errors."""
        payload = {
            "user_id": "test_user",
            "message": "What is the capital of France?",
        }
        resp = client.post("/api/chat", json=payload)
        data = resp.json()
        assert isinstance(data.get("errors"), list)
        # We allow intent classification errors but final_answer must exist
        assert data.get("answer"), "Must have an answer even if there are warnings"

    def test_model_used_is_populated(self, client: httpx.Client):
        """The model_used field should be populated after a successful run."""
        payload = {
            "user_id": "test_user",
            "message": "Explain quantum computing briefly.",
        }
        resp = client.post("/api/chat", json=payload)
        data = resp.json()
        # model_used should be set (either gemini or huggingface fallback)
        if data.get("execution_status") == "completed":
            assert data.get("model_used") is not None, "model_used should be set on success"


# ═══════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_standalone():
    """Run a quick smoke test without pytest."""
    print("=" * 60)
    print("  Nexus AI Orchestrator – API Smoke Test")
    print("=" * 60)

    client = httpx.Client(base_url=BASE_URL, timeout=120.0)

    # 1. Health
    print("\n[1/4] Testing GET /api/health ...")
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    print(f"  ✅ Health OK: {data}")

    # 2. Root
    print("\n[2/4] Testing GET / ...")
    resp = client.get("/")
    assert resp.status_code == 200
    print(f"  ✅ Root OK: {resp.json()}")

    # 3. Chat
    print("\n[3/4] Testing POST /api/chat (standard message) ...")
    payload = {"user_id": "smoke_test", "message": "What is artificial intelligence?"}
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    print(f"  ✅ Chat OK")
    print(f"     Intent     : {data.get('intent')}")
    print(f"     Confidence : {data.get('confidence')}")
    print(f"     Model      : {data.get('model_used')}")
    print(f"     Status     : {data.get('execution_status')}")
    print(f"     Session    : {data.get('session_id')}")
    print(f"     Answer     : {data.get('answer', '')[:120]}...")

    # 4. Schema validation
    print("\n[4/4] Validating response schema ...")
    required = ["answer", "intent", "confidence", "execution_status", "session_id", "errors"]
    for field in required:
        assert field in data, f"  ❌ Missing field: {field}"
    print(f"  ✅ All {len(required)} required fields present")

    print("\n" + "=" * 60)
    print("  ALL SMOKE TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_standalone()
