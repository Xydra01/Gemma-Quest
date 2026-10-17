"""Local inference provider communicating with an Ollama daemon."""

import logging
from typing import Dict, List, Optional
import httpx
import ollama

from .base import (
    InferenceConnectionError,
    InferenceError,
    InferenceProvider,
    InferenceTimeoutError,
)

logger = logging.getLogger(__name__)


class LocalProvider(InferenceProvider):
    """Local inference client communicating with Ollama running Gemma 4."""

    def __init__(
        self,
        model_name: str = "gemma4:2b-q4_0",
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 30,
    ):
        self.model_name = model_name
        self.provider_name = "ollama"
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client = ollama.Client(host=self.base_url, timeout=self.timeout_seconds)

    def is_available(self) -> bool:
        """Pings Ollama server with a quick timeout check."""
        try:
            # Short timeout probe to verify server availability without hanging
            probe_client = ollama.Client(host=self.base_url, timeout=2.0)
            probe_client.list()
            return True
        except Exception:
            return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        options: Optional[dict] = None,
    ) -> str:
        """Sends chat messages to the local Ollama instance."""
        options = options or {}

        # Map generic sampling options to Ollama options schema
        ollama_options = {}
        if "temperature" in options:
            ollama_options["temperature"] = float(options["temperature"])
        if "top_p" in options:
            ollama_options["top_p"] = float(options["top_p"])
        if "top_k" in options:
            ollama_options["top_k"] = int(options["top_k"])
        if "max_tokens" in options:
            ollama_options["num_predict"] = int(options["max_tokens"])

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options=ollama_options,
            )
            if hasattr(response, "message"):
                msg = response.message
                content = getattr(msg, "content", "") or ""
                if not content and hasattr(msg, "thinking") and msg.thinking:
                    content = msg.thinking
                return content
            elif isinstance(response, dict):
                msg_dict = response.get("message", {})
                return msg_dict.get("content", "") or msg_dict.get("thinking", "")
            return str(response)

        except (httpx.ConnectError, ConnectionError, ollama.RequestError) as e:
            raise InferenceConnectionError(
                f"Failed to connect to local Ollama server at {self.base_url}: {e}"
            ) from e

        except (httpx.TimeoutException, TimeoutError) as e:
            raise InferenceTimeoutError(
                f"Local model generation exceeded {self.timeout_seconds}s timeout: {e}"
            ) from e

        except ollama.ResponseError as e:
            raise InferenceError(
                f"Ollama server returned error: {e}"
            ) from e

        except Exception as e:
            raise InferenceError(f"Unexpected local LLM failure: {e}") from e

    def get_status(self) -> str:
        """Returns TUI status string."""
        return f"[LOCAL: {self.model_name}]"
