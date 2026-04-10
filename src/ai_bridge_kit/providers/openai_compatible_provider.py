from __future__ import annotations

from .openai_provider import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_chat_model: str = "gpt-4o-mini",
        default_embedding_model: str = "text-embedding-3-small",
        supports_embeddings: bool = True,
        name: str = "openai-compatible",
    ) -> None:
        if not base_url:
            raise ValueError("OpenAICompatibleProvider requires a base_url.")
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_chat_model=default_chat_model,
            default_embedding_model=default_embedding_model,
            supports_embeddings=supports_embeddings,
            name=name,
        )

    @classmethod
    def for_openrouter(
        cls,
        *,
        api_key: str,
        default_chat_model: str = "openai/gpt-4o-mini",
        default_embedding_model: str = "openai/text-embedding-3-small",
    ) -> "OpenAICompatibleProvider":
        return cls(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_chat_model=default_chat_model,
            default_embedding_model=default_embedding_model,
            name="openrouter",
        )

    @classmethod
    def for_groq(
        cls, *, api_key: str, default_chat_model: str = "llama-3.1-8b-instant"
    ) -> "OpenAICompatibleProvider":
        return cls(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            default_chat_model=default_chat_model,
            default_embedding_model="",
            supports_embeddings=False,
            name="groq",
        )

    @classmethod
    def for_together(
        cls, *, api_key: str, default_chat_model: str = "meta-llama/Llama-3.1-8B-Instruct-Turbo"
    ) -> "OpenAICompatibleProvider":
        return cls(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            default_chat_model=default_chat_model,
            default_embedding_model="",
            supports_embeddings=False,
            name="together",
        )
