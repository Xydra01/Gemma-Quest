"""Abstract base class and domain exceptions for LLM inference providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class InferenceError(Exception):
    """Base exception for inference client failures."""
    pass


class InferenceConnectionError(InferenceError):
    """Raised when the inference service or daemon is unreachable."""
    pass


class InferenceTimeoutError(InferenceError):
    """Raised when model inference exceeds the configured timeout."""
    pass


class InferenceAuthError(InferenceError):
    """Raised when authentication credentials (API key) are missing or invalid."""
    pass


class InferenceRateLimitError(InferenceError):
    """Raised when the inference provider returns HTTP 429 / rate limit exceeded."""
    pass


class InferenceProvider(ABC):
    """Abstract base class for LLM inference backends.

    Enables the game engine to communicate interchangeably with local models (Ollama)
    or cloud APIs without coupling core narrative logic to a specific vendor.
    """

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        options: Optional[dict] = None,
    ) -> str:
        """Sends chat messages to the model and returns the generated text response.

        Args:
            messages: List of message dictionaries with 'role' ('system', 'user', 'assistant')
                      and 'content'.
            options: Optional sampling overrides (e.g. temperature, max_tokens).

        Returns:
            The raw text completion from the language model.

        Raises:
            InferenceConnectionError: If server connection fails.
            InferenceTimeoutError: If request times out.
            InferenceAuthError: If authentication fails.
            InferenceRateLimitError: If rate limit is exceeded.
            InferenceError: On generic inference failures.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if the inference provider is reachable and ready to serve requests.

        Returns:
            True if the service is online and usable, False otherwise.
        """
        pass

    @abstractmethod
    def get_status(self) -> str:
        """Returns a concise label for display in the TUI status bar.

        Examples:
            '[LOCAL: gemma4:2b-q4_0]'
            '[CLOUD: vertex_ai]'
            '[HYBRID: Cloud Active]'
        """
        pass
