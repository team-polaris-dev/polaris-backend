"""Application-level errors for agent session orchestration."""


class AgentError(Exception):
    """Base class for agent orchestration errors."""


class UnsupportedProviderError(AgentError):
    """Raised when the requested provider is not registered."""


class SessionNotFoundError(AgentError):
    """Raised when a session id is unknown."""


class ProviderRuntimeError(AgentError):
    """Raised when a provider SDK fails or is unavailable."""


class InvalidProviderOptionError(AgentError):
    """Raised when an option is unsupported for the selected provider."""
