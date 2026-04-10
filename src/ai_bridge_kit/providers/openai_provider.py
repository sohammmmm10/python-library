from __future__ import annotations

from typing import Any

from ..errors import CapabilityNotSupportedError, ProviderExecutionError
from ..models import AIResponse, ChatRequest, EmbedRequest, EmbeddingResponse
from .base import AIProvider, ProviderCapabilities

try:  # pragma: no cover - depends on optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - depends on optional dependency
    OpenAI = None  # type: ignore[assignment]


def _maybe_dump_raw(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return obj


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class OpenAIProvider(AIProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        default_chat_model: str = "gpt-4o-mini",
        default_embedding_model: str = "text-embedding-3-small",
        supports_embeddings: bool = True,
        name: str = "openai",
    ) -> None:
        super().__init__(name)
        if OpenAI is None:
            raise ImportError(
                "OpenAI SDK is not installed. Install with: pip install 'ai-bridge-kit[openai]'"
            )

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization
        if project:
            kwargs["project"] = project

        self._client = OpenAI(**kwargs)
        self.default_chat_model = default_chat_model
        self.default_embedding_model = default_embedding_model
        self.supports_embeddings = supports_embeddings

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            embeddings=self.supports_embeddings,
            functions=False,
        )

    def chat(self, request: ChatRequest) -> AIResponse:
        model = request.model or self.default_chat_model
        messages = [m.as_provider_dict() for m in request.messages]

        try:
            if hasattr(self._client, "responses"):
                input_payload = [
                    {
                        "role": m["role"],
                        "content": [{"type": "input_text", "text": m["content"]}],
                    }
                    for m in messages
                ]
                kwargs: dict[str, Any] = {"model": model, "input": input_payload}
                if request.temperature is not None:
                    kwargs["temperature"] = request.temperature
                if request.max_output_tokens is not None:
                    kwargs["max_output_tokens"] = request.max_output_tokens

                response = self._client.responses.create(**kwargs)
                text = getattr(response, "output_text", None) or self._extract_output_text(
                    response
                )
                usage = getattr(response, "usage", None)
                return AIResponse(
                    content=text or "",
                    provider=self.name,
                    model=model,
                    tokens_input=getattr(usage, "input_tokens", None),
                    tokens_output=getattr(usage, "output_tokens", None),
                    raw=_maybe_dump_raw(response),
                )

            kwargs = {
                "model": model,
                "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            }
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.max_output_tokens is not None:
                kwargs["max_tokens"] = request.max_output_tokens

            completion = self._client.chat.completions.create(**kwargs)
            content = ""
            if completion.choices and completion.choices[0].message:
                content = completion.choices[0].message.content or ""
            usage = getattr(completion, "usage", None)
            return AIResponse(
                content=content,
                provider=self.name,
                model=model,
                tokens_input=getattr(usage, "prompt_tokens", None),
                tokens_output=getattr(usage, "completion_tokens", None),
                raw=_maybe_dump_raw(completion),
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise ProviderExecutionError(f"OpenAI chat failed: {exc}") from exc

    def embed(self, request: EmbedRequest) -> EmbeddingResponse:
        if not self.supports_embeddings:
            raise CapabilityNotSupportedError(
                f"Provider '{self.name}' does not support embeddings."
            )
        model = request.model or self.default_embedding_model
        if not model:
            raise CapabilityNotSupportedError(
                f"Provider '{self.name}' does not have a default embedding model configured."
            )
        try:
            response = self._client.embeddings.create(model=model, input=list(request.input))
            vectors = [list(item.embedding) for item in response.data]
            return EmbeddingResponse(
                vectors=vectors,
                provider=self.name,
                model=model,
                raw=_maybe_dump_raw(response),
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise ProviderExecutionError(f"OpenAI embeddings failed: {exc}") from exc

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        output = getattr(response, "output", None)
        if not output:
            return ""

        chunks: list[str] = []
        for item in output:
            content_items = _obj_get(item, "content", [])
            for content_item in content_items:
                text = _obj_get(content_item, "text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()
