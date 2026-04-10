from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, TypeVar

from .config import AIBridgeSettings
from .errors import (
    AIBridgeError,
    CapabilityNotSupportedError,
    ProviderExecutionError,
    ProviderNotFoundError,
    ValidationError,
)
from .models import (
    AIResponse,
    ChatMessage,
    ChatRequest,
    EmbedRequest,
    EmbeddingResponse,
    FunctionCallRequest,
    FunctionCallResponse,
)
from .providers.base import AIProvider
from .providers.local_function_provider import LocalFunctionProvider
from .retry import arun_with_retry, run_with_retry

T = TypeVar("T")


def _default_local_chat(payload: dict[str, Any]) -> str:
    messages = payload.get("messages", [])
    for message in reversed(messages):
        if message.get("role") == "user":
            return f"[local-ai] {message.get('content', '')}"
    return "[local-ai] No user message provided."


def _vector_for_text(text: str, *, dims: int = 16) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [round(digest[idx] / 255.0, 6) for idx in range(dims)]


def _default_local_embed(payload: dict[str, Any]) -> list[list[float]]:
    values = payload.get("input", [])
    return [_vector_for_text(str(item)) for item in values]


class AIClient:
    def __init__(
        self,
        *,
        settings: AIBridgeSettings | None = None,
        providers: Iterable[AIProvider] | None = None,
    ) -> None:
        self.settings = settings or AIBridgeSettings()
        self._providers: dict[str, AIProvider] = {}

        self._register_builtin_local_provider()

        for provider in providers or []:
            self.register_provider(provider)

        if self.settings.default_provider not in self._providers and self._providers:
            self.settings.default_provider = next(iter(self._providers))

    @classmethod
    def from_env(cls) -> "AIClient":
        return cls(settings=AIBridgeSettings.from_env())

    def register_provider(self, provider: AIProvider, *, set_default: bool = False) -> None:
        self._providers[provider.name] = provider
        if set_default:
            self.settings.default_provider = provider.name

    def available_providers(self) -> list[str]:
        return sorted(self._providers)

    def get_provider(self, provider_name: str | None = None) -> AIProvider:
        resolved = provider_name or self.settings.default_provider
        if not resolved:
            raise ProviderNotFoundError("No provider requested and no default provider configured.")
        provider = self._providers.get(resolved)
        if provider is None:
            raise ProviderNotFoundError(
                f"Provider '{resolved}' is not registered. "
                f"Available providers: {', '.join(self.available_providers()) or 'none'}"
            )
        return provider

    def chat(
        self,
        messages: str | Sequence[ChatMessage],
        *,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AIResponse:
        prepared = self._normalize_messages(messages)
        request = ChatRequest(
            messages=prepared,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            metadata=metadata,
        )
        selected_provider = self.get_provider(provider)
        return self._run(lambda: selected_provider.chat(request))

    async def achat(
        self,
        messages: str | Sequence[ChatMessage],
        *,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AIResponse:
        prepared = self._normalize_messages(messages)
        request = ChatRequest(
            messages=prepared,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            metadata=metadata,
        )
        selected_provider = self.get_provider(provider)
        return await self._arun(lambda: selected_provider.achat(request))

    def embed(
        self,
        input_text: str | Sequence[str],
        *,
        provider: str | None = None,
        model: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EmbeddingResponse:
        payload = [input_text] if isinstance(input_text, str) else list(input_text)
        if not payload:
            raise ValidationError("Embedding input cannot be empty.")

        request = EmbedRequest(input=payload, model=model, metadata=metadata)
        selected_provider = self.get_provider(provider)
        return self._run(lambda: selected_provider.embed(request))

    async def aembed(
        self,
        input_text: str | Sequence[str],
        *,
        provider: str | None = None,
        model: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EmbeddingResponse:
        payload = [input_text] if isinstance(input_text, str) else list(input_text)
        if not payload:
            raise ValidationError("Embedding input cannot be empty.")

        request = EmbedRequest(input=payload, model=model, metadata=metadata)
        selected_provider = self.get_provider(provider)
        return await self._arun(lambda: selected_provider.aembed(request))

    def call_function(
        self,
        function_name: str,
        *,
        arguments: Mapping[str, Any] | None = None,
        provider: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FunctionCallResponse:
        candidate = function_name.strip()
        if not candidate:
            raise ValidationError("Function name cannot be empty.")

        request = FunctionCallRequest(
            function_name=candidate,
            arguments=dict(arguments or {}),
            metadata=metadata,
        )
        selected_provider = self.get_provider(provider)
        return self._run(lambda: selected_provider.call_function(request))

    async def acall_function(
        self,
        function_name: str,
        *,
        arguments: Mapping[str, Any] | None = None,
        provider: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FunctionCallResponse:
        candidate = function_name.strip()
        if not candidate:
            raise ValidationError("Function name cannot be empty.")

        request = FunctionCallRequest(
            function_name=candidate,
            arguments=dict(arguments or {}),
            metadata=metadata,
        )
        selected_provider = self.get_provider(provider)
        return await self._arun(lambda: selected_provider.acall_function(request))

    def _run(self, operation: Callable[[], T]) -> T:
        def wrapped() -> T:
            try:
                return self._run_with_timeout(operation)
            except AIBridgeError:
                raise
            except Exception as exc:
                raise ProviderExecutionError(str(exc)) from exc

        return run_with_retry(
            wrapped,
            retries=self.settings.retries,
            backoff_seconds=self.settings.backoff_seconds,
            max_backoff_seconds=self.settings.max_backoff_seconds,
            jitter=self.settings.use_jitter,
            retry_on=(Exception,),
            should_retry=self._should_retry,
        )

    async def _arun(self, operation: Callable[[], Awaitable[T]]) -> T:
        async def wrapped() -> T:
            try:
                return await self._arun_with_timeout(operation)
            except AIBridgeError:
                raise
            except Exception as exc:
                raise ProviderExecutionError(str(exc)) from exc

        return await arun_with_retry(
            wrapped,
            retries=self.settings.retries,
            backoff_seconds=self.settings.backoff_seconds,
            max_backoff_seconds=self.settings.max_backoff_seconds,
            jitter=self.settings.use_jitter,
            retry_on=(Exception,),
            should_retry=self._should_retry,
        )

    def _run_with_timeout(self, operation: Callable[[], T]) -> T:
        timeout = self.settings.request_timeout
        if timeout is None:
            return operation()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(operation)
            try:
                return future.result(timeout=timeout)
            except FutureTimeoutError as exc:
                raise ProviderExecutionError(
                    f"Operation timed out after {timeout} seconds."
                ) from exc

    async def _arun_with_timeout(self, operation: Callable[[], Awaitable[T]]) -> T:
        timeout = self.settings.request_timeout
        if timeout is None:
            return await operation()
        return await asyncio.wait_for(operation(), timeout=timeout)

    def _register_builtin_local_provider(self) -> None:
        if "local" in self._providers:
            return
        local = LocalFunctionProvider(name="local")
        local.register("chat", _default_local_chat)
        local.register("embed", _default_local_embed)
        local.register("echo", lambda payload: payload)
        local.set_chat_function("chat")
        local.set_embed_function("embed")
        self.register_provider(local, set_default=self.settings.default_provider == "local")

    @staticmethod
    def _normalize_messages(messages: str | Sequence[ChatMessage]) -> list[ChatMessage]:
        if isinstance(messages, str):
            text = messages.strip()
            if not text:
                raise ValidationError("Chat message text cannot be empty.")
            return [ChatMessage(role="user", content=text)]

        normalized = list(messages)
        if not normalized:
            raise ValidationError("Chat messages cannot be empty.")
        return normalized

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        return not isinstance(
            exc,
            (ValidationError, ProviderNotFoundError, CapabilityNotSupportedError),
        )
