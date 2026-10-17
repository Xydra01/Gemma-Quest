"""Hybrid inference wrapper with smart fallback and circuit breaker logic."""

import logging
from typing import Dict, List, Optional
from .base import (
    InferenceAuthError,
    InferenceConnectionError,
    InferenceError,
    InferenceProvider,
    InferenceRateLimitError,
    InferenceTimeoutError,
)

logger = logging.getLogger(__name__)


class HybridWrapper(InferenceProvider):
    """Smart hybrid inference provider routing requests between Cloud and Local.

    Prioritizes high-quality Cloud inference while automatically falling back
    to Local execution during network timeouts, rate limits, or connectivity issues.
    Includes a Circuit Breaker that locks to Local mode after consecutive failures.
    """

    def __init__(
        self,
        local: InferenceProvider,
        cloud: Optional[InferenceProvider] = None,
        fallback_timeout: float = 5.0,
        circuit_breaker_threshold: int = 3,
    ):
        self.local = local
        self.cloud = cloud
        self.fallback_timeout = fallback_timeout
        self.circuit_breaker_threshold = circuit_breaker_threshold

        self.consecutive_failures: int = 0
        self.circuit_open: bool = False
        self.last_used_backend: str = "Cloud" if (self.cloud and self.cloud.is_available()) else "Local"

    def is_available(self) -> bool:
        """Returns True if either the local engine or the cloud engine is available."""
        return self.local.is_available() or (bool(self.cloud) and self.cloud.is_available())

    def chat(
        self,
        messages: List[Dict[str, str]],
        options: Optional[dict] = None,
    ) -> str:
        """Sends chat messages, trying Cloud first and falling back to Local."""
        options = dict(options or {})

        # If Circuit Breaker is open or Cloud is not configured/available, route directly to Local
        if self.circuit_open or not self.cloud or not self.cloud.is_available():
            if self.circuit_open:
                self.last_used_backend = "Local (Circuit Open)"
            else:
                self.last_used_backend = "Local"
            return self.local.chat(messages, options)

        # Attempt Cloud Inference with hybrid fallback timeout
        cloud_options = dict(options)
        # Apply the shorter fallback timeout to prevent hanging the player's TUI
        cloud_options["timeout_seconds"] = self.fallback_timeout

        try:
            response = self.cloud.chat(messages, cloud_options)
            # Cloud succeeded: reset failure counter
            self.consecutive_failures = 0
            self.last_used_backend = "Cloud"
            return response

        except (
            InferenceTimeoutError,
            InferenceConnectionError,
            InferenceRateLimitError,
            InferenceAuthError,
            InferenceError,
        ) as e:
            self.consecutive_failures += 1
            logger.warning(
                "Cloud inference failed (%s/%s): %s. Falling back to local model.",
                self.consecutive_failures,
                self.circuit_breaker_threshold,
                e,
            )

            if self.consecutive_failures >= self.circuit_breaker_threshold:
                self.circuit_open = True
                self.last_used_backend = "Local (Circuit Open)"
                logger.warning(
                    "Circuit breaker tripped after %s failures. Locking to Local mode.",
                    self.consecutive_failures,
                )
            else:
                self.last_used_backend = "Local (Fallback)"

            # Fallback to local provider
            return self.local.chat(messages, options)

    def reset_circuit_breaker(self) -> None:
        """Manually resets the circuit breaker to re-enable Cloud attempts."""
        self.consecutive_failures = 0
        self.circuit_open = False
        self.last_used_backend = "Cloud" if (self.cloud and self.cloud.is_available()) else "Local"

    def get_status(self) -> str:
        """Returns dynamic TUI status reflecting the current routing state."""
        if self.circuit_open:
            return "[HYBRID: Circuit Open (Local)]"
        if self.last_used_backend == "Local (Fallback)":
            return "[HYBRID -> LOCAL (Fallback)]"
        if self.cloud and self.cloud.is_available():
            return "[HYBRID: Cloud Active]"
        return self.local.get_status()
