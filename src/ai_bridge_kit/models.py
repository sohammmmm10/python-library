from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Sequence

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str
    name: str | None = None
    metadata: Mapping[str, Any] | None = None

    def as_provider_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class ChatRequest:
    messages: Sequence[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class EmbedRequest:
    input: Sequence[str]
    model: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class FunctionCallRequest:
    function_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class AIResponse:
    content: str
    provider: str
    model: str | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw: Any = None


@dataclass(frozen=True)
class EmbeddingResponse:
    vectors: list[list[float]]
    provider: str
    model: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw: Any = None


@dataclass(frozen=True)
class FunctionCallResponse:
    function_name: str
    provider: str
    result: Any
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw: Any = None
