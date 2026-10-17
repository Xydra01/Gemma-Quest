"""Cloud inference provider communicating with remote APIs via HTTPX."""

import logging
from typing import Dict, List, Optional
import httpx

from .base import (
    InferenceAuthError,
    InferenceConnectionError,
    InferenceError,
    InferenceProvider,
    InferenceRateLimitError,
    InferenceTimeoutError,
)

logger = logging.getLogger(__name__)


class CloudProvider(InferenceProvider):
    """Remote inference client communicating with Cloud APIs (Vertex AI, Together AI, etc.)."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model_id: str = "gemma-4-instruct",
        provider_name: str = "vertex_ai",
        timeout_seconds: int = 30,
    ):
        self.api_key = api_key.strip() if api_key else ""
        self.endpoint = endpoint.strip() if endpoint else ""
        self.model_id = model_id
        self.provider_name = provider_name
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        """Verifies that an API key and valid endpoint are configured."""
        return bool(self.api_key) and bool(self.endpoint)

    def chat(
        self,
        messages: List[Dict[str, str]],
        options: Optional[dict] = None,
    ) -> str:
        """Sends chat messages to the remote cloud API endpoint."""
        if not self.api_key:
            raise InferenceAuthError("Cloud inference failed: Missing API key.")
        if not self.endpoint:
            raise InferenceConnectionError("Cloud inference failed: Missing endpoint URL.")

        options = options or {}

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": float(options.get("temperature", 0.8)),
            "top_p": float(options.get("top_p", 0.9)),
            "max_tokens": int(options.get("max_tokens", 1024)),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=float(self.timeout_seconds)) as client:
                response = client.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                )

                if response.status_code == 200:
                    data = response.json()
                    # Standard OpenAI / REST schema
                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            return choice["message"]["content"]
                        if "text" in choice:
                            return choice["text"]

                    # Generic/Vertex content schema
                    if "content" in data:
                        return data["content"]

                    raise InferenceError(f"Unexpected response schema from cloud API: {data}")

                elif response.status_code in (401, 403):
                    raise InferenceAuthError(
                        f"Authentication failed ({response.status_code}): {response.text}"
                    )
                elif response.status_code == 429:
                    raise InferenceRateLimitError(
                        f"Rate limit exceeded (429): {response.text}"
                    )
                elif response.status_code >= 500:
                    raise InferenceConnectionError(
                        f"Cloud API server error ({response.status_code}): {response.text}"
                    )
                else:
                    raise InferenceError(
                        f"Cloud API request failed with status {response.status_code}: {response.text}"
                    )

        except (httpx.ConnectError, httpx.NetworkError) as e:
            raise InferenceConnectionError(f"Network error connecting to cloud API: {e}") from e

        except httpx.TimeoutException as e:
            raise InferenceTimeoutError(
                f"Cloud API request timed out after {self.timeout_seconds}s: {e}"
            ) from e

        except (InferenceError, ValueError):
            raise
        except Exception as e:
            raise InferenceError(f"Unexpected cloud inference failure: {e}") from e

    def get_status(self) -> str:
        """Returns TUI status string."""
        return f"[CLOUD: {self.provider_name}]"
