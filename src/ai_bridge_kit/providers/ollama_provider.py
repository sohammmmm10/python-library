from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from ..errors import ProviderExecutionError
from ..models import AIResponse, ChatRequest, EmbedRequest, EmbeddingResponse
from .base import AIProvider, ProviderCapabilities


class OllamaProvider(AIProvider):
    capabilities = ProviderCapabilities(chat=True, embeddings=True, functions=False)

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        default_chat_model: str = "llama3.2",
        default_embedding_model: str = "nomic-embed-text",
        request_timeout: float = 60.0,
        name: str = "ollama",
    ) -> None:
        super().__init__(name)
        self.base_url = base_url.rstrip("/")
        self.default_chat_model = default_chat_model
        self.default_embedding_model = default_embedding_model
        self.request_timeout = request_timeout

    def chat(self, request: ChatRequest) -> AIResponse:
        model = request.model or self.default_chat_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.as_provider_dict() for m in request.messages],
            "stream": False,
        }
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if options:
            payload["options"] = options

        response = self._post_json("/api/chat", payload)
        message = response.get("message") or {}
        content = message.get("content") or ""
        return AIResponse(
            content=str(content),
            provider=self.name,
            model=model,
            tokens_input=response.get("prompt_eval_count"),
            tokens_output=response.get("eval_count"),
            raw=response,
        )

    def embed(self, request: EmbedRequest) -> EmbeddingResponse:
        model = request.model or self.default_embedding_model
        texts = list(request.input)

        # Newer Ollama API endpoint.
        try:
            response = self._post_json("/api/embed", {"model": model, "input": texts})
            vectors = response.get("embeddings")
            if isinstance(vectors, list):
                return EmbeddingResponse(vectors=vectors, provider=self.name, model=model, raw=response)
        except ProviderExecutionError:
            pass

        # Backward-compatible fallback endpoint.
        vectors: list[list[float]] = []
        for text in texts:
            response = self._post_json("/api/embeddings", {"model": model, "prompt": text})
            vector = response.get("embedding")
            if not isinstance(vector, list):
                raise ProviderExecutionError(
                    "Ollama embedding response did not include 'embedding' list."
                )
            vectors.append([float(x) for x in vector])

        return EmbeddingResponse(vectors=vectors, provider=self.name, model=model, raw=vectors)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                body = resp.read().decode("utf-8")
                parsed = json.loads(body)
                if not isinstance(parsed, dict):
                    raise ProviderExecutionError(
                        f"Ollama returned unexpected response type: {type(parsed)!r}"
                    )
                return parsed
        except urllib.error.URLError as exc:
            raise ProviderExecutionError(
                f"Ollama request failed for {url}: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise ProviderExecutionError(
                f"Ollama request timed out for {url}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ProviderExecutionError(
                f"Ollama returned invalid JSON for {url}: {exc}"
            ) from exc
