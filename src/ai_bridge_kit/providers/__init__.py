from .base import AIProvider, ProviderCapabilities
from .anthropic_provider import AnthropicProvider
from .local_function_provider import LocalFunctionProvider
from .ollama_provider import OllamaProvider

try:
    from .openai_compatible_provider import OpenAICompatibleProvider
    from .openai_provider import OpenAIProvider
except Exception:  # pragma: no cover - optional dependency
    OpenAICompatibleProvider = None  # type: ignore[assignment]
    OpenAIProvider = None  # type: ignore[assignment]

__all__ = [
    "AIProvider",
    "ProviderCapabilities",
    "AnthropicProvider",
    "LocalFunctionProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
]
