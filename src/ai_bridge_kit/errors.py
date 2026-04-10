class AIBridgeError(Exception):
    """Base exception for ai-bridge-kit."""


class ProviderNotFoundError(AIBridgeError):
    """Raised when a requested provider is not registered."""


class ProviderExecutionError(AIBridgeError):
    """Raised when provider execution fails."""


class ValidationError(AIBridgeError):
    """Raised for invalid user input."""


class CapabilityNotSupportedError(AIBridgeError):
    """Raised when a provider capability is called but not supported."""
