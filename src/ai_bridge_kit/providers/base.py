from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..errors import CapabilityNotSupportedError, ValidationError
from ..models import (
    AIResponse,
    ChatRequest,
    EmbedRequest,
    EmbeddingResponse,
    FunctionCallRequest,
    FunctionCallResponse,
)


@dataclass(frozen=True)
class ProviderCapabilities:
    chat: bool = False
    embeddings: bool = False
    functions: bool = False


class AIProvider:
    capabilities = ProviderCapabilities()

    def __init__(self, name: str) -> None:
        candidate = name.strip()
        if not candidate:
            raise ValidationError("Provider name cannot be empty.")
        self.name = candidate

    def chat(self, request: ChatRequest) -> AIResponse:
        raise CapabilityNotSupportedError(
            f"Provider '{self.name}' does not support chat."
        )

    async def achat(self, request: ChatRequest) -> AIResponse:
        return await asyncio.to_thread(self.chat, request)

    def embed(self, request: EmbedRequest) -> EmbeddingResponse:
        raise CapabilityNotSupportedError(
            f"Provider '{self.name}' does not support embeddings."
        )

    async def aembed(self, request: EmbedRequest) -> EmbeddingResponse:
        return await asyncio.to_thread(self.embed, request)

    def call_function(self, request: FunctionCallRequest) -> FunctionCallResponse:
        raise CapabilityNotSupportedError(
            f"Provider '{self.name}' does not support function calls."
        )

    async def acall_function(
        self, request: FunctionCallRequest
    ) -> FunctionCallResponse:
        return await asyncio.to_thread(self.call_function, request)
