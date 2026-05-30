from __future__ import annotations

from typing import Any

from ..errors import ProviderExecutionError
from ..models import AIResponse, ChatRequest
from .base import AIProvider, ProviderCapabilities

try:  # pragma: no cover - depends on optional dependency
    from anthropic import Anthropic
except Exception:  # pragma: no cover - depends on optional dependency
    Anthropic = None  # type: ignore[assignment]


def _maybe_dump_raw(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return obj


class AnthropicProvider(AIProvider):
    capabilities = ProviderCapabilities(chat=True, embeddings=False, functions=False)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_chat_model: str = "claude-3-5-sonnet-latest",
        max_output_tokens: int = 1024,
        name: str = "anthropic",
    ) -> None:
        super().__init__(name)
        if Anthropic is None:
            raise ImportError(
                "Anthropic SDK is not installed. Install with: pip install 'ai-bridge-kit[anthropic]'"
            )

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = Anthropic(**kwargs)
        self.default_chat_model = default_chat_model
        self.max_output_tokens = max_output_tokens

    def chat(self, request: ChatRequest) -> AIResponse:
        model = request.model or self.default_chat_model

        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "system":
                system_parts.append(msg.content)
                continue

            role = "assistant" if msg.role == "assistant" else "user"
            anthropic_messages.append(
                {"role": role, "content": [{"type": "text", "text": msg.content}]}
            )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": request.max_output_tokens or self.max_output_tokens,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        try:
            response = self._client.messages.create(**kwargs)
            text_parts: list[str] = []
            for item in getattr(response, "content", []):
                if getattr(item, "type", None) == "text":
                    value = getattr(item, "text", "")
                    if isinstance(value, str):
                        text_parts.append(value)
            usage = getattr(response, "usage", None)
            return AIResponse(
                content="\n".join(text_parts).strip(),
                provider=self.name,
                model=model,
                tokens_input=getattr(usage, "input_tokens", None),
                tokens_output=getattr(usage, "output_tokens", None),
                raw=_maybe_dump_raw(response),
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise ProviderExecutionError(f"Anthropic chat failed: {exc}") from exc
