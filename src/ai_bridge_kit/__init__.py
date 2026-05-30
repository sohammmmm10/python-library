from .client import AIClient
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
from .providers import (
    AIProvider,
    AnthropicProvider,
    LocalFunctionProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    ProviderCapabilities,
)

__version__ = "0.1.0"

__all__ = [
    "AIClient",
    "AIBridgeSettings",
    "AIBridgeError",
    "ProviderNotFoundError",
    "ProviderExecutionError",
    "ValidationError",
    "CapabilityNotSupportedError",
    "ChatMessage",
    "ChatRequest",
    "EmbedRequest",
    "FunctionCallRequest",
    "AIResponse",
    "EmbeddingResponse",
    "FunctionCallResponse",
    "AIProvider",
    "ProviderCapabilities",
    "AnthropicProvider",
    "LocalFunctionProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
]
