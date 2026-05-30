from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ..errors import CapabilityNotSupportedError, ProviderExecutionError, ValidationError
from ..models import (
    AIResponse,
    ChatRequest,
    EmbedRequest,
    EmbeddingResponse,
    FunctionCallRequest,
    FunctionCallResponse,
)
from .base import AIProvider, ProviderCapabilities

FunctionType = Callable[[dict[str, Any]], Any] | Callable[[dict[str, Any]], Awaitable[Any]]


class LocalFunctionProvider(AIProvider):
    def __init__(
        self,
        *,
        name: str = "local",
        functions: Mapping[str, FunctionType] | None = None,
        chat_function: str | None = None,
        embed_function: str | None = None,
    ) -> None:
        super().__init__(name)
        self._functions: dict[str, FunctionType] = dict(functions or {})
        self._chat_function = chat_function
        self._embed_function = embed_function

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=self._chat_function in self._functions if self._chat_function else False,
            embeddings=self._embed_function in self._functions
            if self._embed_function
            else False,
            functions=True,
        )

    def register(self, function_name: str, fn: FunctionType) -> None:
        candidate = function_name.strip()
        if not candidate:
            raise ValidationError("Function name cannot be empty.")
        self._functions[candidate] = fn

    def set_chat_function(self, function_name: str | None) -> None:
        if function_name is None:
            self._chat_function = None
            return
        if function_name not in self._functions:
            raise ValidationError(
                f"Chat function '{function_name}' is not registered on provider '{self.name}'."
            )
        self._chat_function = function_name

    def set_embed_function(self, function_name: str | None) -> None:
        if function_name is None:
            self._embed_function = None
            return
        if function_name not in self._functions:
            raise ValidationError(
                f"Embedding function '{function_name}' is not registered on provider '{self.name}'."
            )
        self._embed_function = function_name

    def list_functions(self) -> list[str]:
        return sorted(self._functions)

    def _get_function(self, function_name: str) -> FunctionType:
        if function_name not in self._functions:
            raise ValidationError(
                f"Function '{function_name}' is not registered on provider '{self.name}'."
            )
        return self._functions[function_name]

    def _invoke_sync(self, fn: FunctionType, payload: dict[str, Any]) -> Any:
        try:
            result = fn(payload)
            if inspect.isawaitable(result):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    return asyncio.run(result)
                raise ProviderExecutionError(
                    "Cannot run async local function in sync context while an event loop is running. "
                    "Use the async client methods instead."
                )
            return result
        except ProviderExecutionError:
            raise
        except Exception as exc:  # pragma: no cover - passthrough behavior
            raise ProviderExecutionError(
                f"Local function provider '{self.name}' failed: {exc}"
            ) from exc

    async def _invoke_async(self, fn: FunctionType, payload: dict[str, Any]) -> Any:
        try:
            result = fn(payload)
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception as exc:  # pragma: no cover - passthrough behavior
            raise ProviderExecutionError(
                f"Local function provider '{self.name}' failed: {exc}"
            ) from exc

    def chat(self, request: ChatRequest) -> AIResponse:
        if not self._chat_function:
            raise CapabilityNotSupportedError(
                f"Provider '{self.name}' does not have a configured chat function."
            )

        fn = self._get_function(self._chat_function)
        payload = {
            "messages": [m.as_provider_dict() for m in request.messages],
            "model": request.model,
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
            "metadata": dict(request.metadata or {}),
        }
        result = self._invoke_sync(fn, payload)
        if isinstance(result, AIResponse):
            return result
        return AIResponse(content=str(result), provider=self.name, model=request.model, raw=result)

    async def achat(self, request: ChatRequest) -> AIResponse:
        if not self._chat_function:
            raise CapabilityNotSupportedError(
                f"Provider '{self.name}' does not have a configured chat function."
            )
        fn = self._get_function(self._chat_function)
        payload = {
            "messages": [m.as_provider_dict() for m in request.messages],
            "model": request.model,
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
            "metadata": dict(request.metadata or {}),
        }
        result = await self._invoke_async(fn, payload)
        if isinstance(result, AIResponse):
            return result
        return AIResponse(content=str(result), provider=self.name, model=request.model, raw=result)

    def embed(self, request: EmbedRequest) -> EmbeddingResponse:
        if not self._embed_function:
            raise CapabilityNotSupportedError(
                f"Provider '{self.name}' does not have a configured embedding function."
            )

        fn = self._get_function(self._embed_function)
        payload = {
            "input": list(request.input),
            "model": request.model,
            "metadata": dict(request.metadata or {}),
        }
        result = self._invoke_sync(fn, payload)
        if isinstance(result, EmbeddingResponse):
            return result
        vectors = self._coerce_vectors(result)
        return EmbeddingResponse(vectors=vectors, provider=self.name, model=request.model, raw=result)

    async def aembed(self, request: EmbedRequest) -> EmbeddingResponse:
        if not self._embed_function:
            raise CapabilityNotSupportedError(
                f"Provider '{self.name}' does not have a configured embedding function."
            )
        fn = self._get_function(self._embed_function)
        payload = {
            "input": list(request.input),
            "model": request.model,
            "metadata": dict(request.metadata or {}),
        }
        result = await self._invoke_async(fn, payload)
        if isinstance(result, EmbeddingResponse):
            return result
        vectors = self._coerce_vectors(result)
        return EmbeddingResponse(vectors=vectors, provider=self.name, model=request.model, raw=result)

    def call_function(self, request: FunctionCallRequest) -> FunctionCallResponse:
        fn = self._get_function(request.function_name)
        payload = dict(request.arguments)
        result = self._invoke_sync(fn, payload)
        if isinstance(result, FunctionCallResponse):
            return result
        return FunctionCallResponse(
            function_name=request.function_name,
            provider=self.name,
            result=result,
            raw=result,
        )

    async def acall_function(self, request: FunctionCallRequest) -> FunctionCallResponse:
        fn = self._get_function(request.function_name)
        payload = dict(request.arguments)
        result = await self._invoke_async(fn, payload)
        if isinstance(result, FunctionCallResponse):
            return result
        return FunctionCallResponse(
            function_name=request.function_name,
            provider=self.name,
            result=result,
            raw=result,
        )

    @staticmethod
    def _coerce_vectors(value: Any) -> list[list[float]]:
        if isinstance(value, list) and (not value):
            return []
        if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
            return [[float(item) for item in value]]
        if (
            isinstance(value, list)
            and all(isinstance(item, list) for item in value)
            and all(all(isinstance(x, (int, float)) for x in item) for item in value)
        ):
            return [[float(x) for x in item] for item in value]
        raise ProviderExecutionError(
            "Embedding function must return list[float] or list[list[float]]."
        )
