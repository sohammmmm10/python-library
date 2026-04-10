import json
import urllib.error
from typing import Any

import pytest

from ai_bridge_kit.models import ChatMessage, ChatRequest, EmbedRequest
from ai_bridge_kit.providers.ollama_provider import OllamaProvider


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_ollama_chat_and_embed_with_api_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        body = json.loads(req.data.decode("utf-8"))
        calls.append((req.full_url, body))
        if req.full_url.endswith("/api/chat"):
            return _FakeResponse(
                {"message": {"content": "hello from ollama"}, "prompt_eval_count": 10, "eval_count": 5}
            )
        if req.full_url.endswith("/api/embed"):
            return _FakeResponse({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
        raise AssertionError(f"Unexpected URL: {req.full_url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    provider = OllamaProvider(base_url="http://mock-ollama:11434", default_chat_model="llama3")

    chat = provider.chat(ChatRequest(messages=[ChatMessage(role="user", content="hi")]))
    assert chat.content == "hello from ollama"
    assert chat.tokens_input == 10
    assert chat.tokens_output == 5

    emb = provider.embed(EmbedRequest(input=["a", "b"]))
    assert len(emb.vectors) == 2
    assert emb.vectors[0] == [0.1, 0.2]

    assert any(url.endswith("/api/chat") for url, _ in calls)
    assert any(url.endswith("/api/embed") for url, _ in calls)


def test_ollama_embed_falls_back_to_legacy_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        body = json.loads(req.data.decode("utf-8"))
        calls.append(req.full_url)
        if req.full_url.endswith("/api/embed"):
            raise urllib.error.URLError("endpoint not available")
        if req.full_url.endswith("/api/embeddings"):
            prompt = body["prompt"]
            return _FakeResponse({"embedding": [float(len(prompt)), 1.0]})
        raise AssertionError(f"Unexpected URL: {req.full_url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    provider = OllamaProvider(base_url="http://mock-ollama:11434")
    emb = provider.embed(EmbedRequest(input=["x", "yz"]))

    assert emb.vectors == [[1.0, 1.0], [2.0, 1.0]]
    assert calls.count("http://mock-ollama:11434/api/embed") == 1
    assert calls.count("http://mock-ollama:11434/api/embeddings") == 2
