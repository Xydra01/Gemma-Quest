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

# Default base endpoint for Google AI Studio / Gemini API
GOOGLE_AI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class CloudProvider(InferenceProvider):
    """Remote inference client communicating with Cloud APIs (Google AI Studio, Vertex AI, Together AI).

    Automatically supports:
    1. Google AI Studio native generateContent API (default when using Google API keys).
    2. OpenAI-compatible chat/completions REST endpoints when custom endpoints are provided.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str = "",
        model_id: str = "gemma-4-26b-a4b-it",
        provider_name: str = "vertex_ai",
        timeout_seconds: int = 30,
    ):
        self.api_key = api_key.strip() if api_key else ""
        self.endpoint = endpoint.strip() if endpoint else ""
        self.model_id = model_id
        self.provider_name = provider_name
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        """Verifies that an API key is configured."""
        return bool(self.api_key)

    def _is_google_ai(self) -> bool:
        """Checks if the request should use the Google AI Studio protocol."""
        if not self.endpoint:
            return True
        return "generativelanguage.googleapis.com" in self.endpoint

    def chat(
        self,
        messages: List[Dict[str, str]],
        options: Optional[dict] = None,
    ) -> str:
        """Sends chat messages to the remote cloud API endpoint."""
        if not self.api_key:
            raise InferenceAuthError("Cloud inference failed: Missing API key.")

        options = options or {}
        temperature = float(options.get("temperature", 0.8))
        max_tokens = int(options.get("max_tokens", 1024))
        timeout = float(options.get("timeout_seconds", self.timeout_seconds))

        if self._is_google_ai():
            return self._chat_google_ai(messages, temperature, max_tokens, timeout)
        else:
            return self._chat_openai_rest(messages, options, timeout)

    def _chat_google_ai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        """Executes completion via Google AI Studio / Gemini API."""
        url = self.endpoint or f"{GOOGLE_AI_BASE_URL}/{self.model_id}:generateContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Separate system instruction from conversational contents
        system_instructions = []
        contents = []

        for m in messages:
            role = m.get("role", "user").lower()
            text = m.get("content", "")

            if role == "system":
                system_instructions.append({"text": text})
            else:
                api_role = "model" if role == "assistant" else "user"
                contents.append({"role": api_role, "parts": [{"text": text}]})

        # Ensure at least one content part exists
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Continue the story."}]}]

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        if system_instructions:
            payload["systemInstruction"] = {"parts": system_instructions}

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                return self._handle_response(response, is_google=True)
        except (httpx.ConnectError, httpx.NetworkError) as e:
            raise InferenceConnectionError(f"Network error connecting to Google AI API: {e}") from e
        except httpx.TimeoutException as e:
            raise InferenceTimeoutError(f"Google AI API request timed out after {timeout}s: {e}") from e

    def _chat_openai_rest(
        self,
        messages: List[Dict[str, str]],
        options: dict,
        timeout: float,
    ) -> str:
        """Executes completion via standard OpenAI-compatible REST endpoint."""
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
            with httpx.Client(timeout=timeout) as client:
                response = client.post(self.endpoint, json=payload, headers=headers)
                return self._handle_response(response, is_google=False)
        except (httpx.ConnectError, httpx.NetworkError) as e:
            raise InferenceConnectionError(f"Network error connecting to cloud API: {e}") from e
        except httpx.TimeoutException as e:
            raise InferenceTimeoutError(f"Cloud API request timed out after {timeout}s: {e}") from e

    def _handle_response(self, response: httpx.Response, is_google: bool) -> str:
        """Parses response and maps HTTP errors to domain exceptions."""
        if response.status_code == 200:
            data = response.json()

            if is_google:
                try:
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"]
                except Exception as e:
                    raise InferenceError(f"Failed to parse Google AI response: {e}") from e

            # Standard OpenAI / REST schema fallback
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
                if "text" in choice:
                    return choice["text"]

            if "content" in data:
                return data["content"]

            raise InferenceError(f"Unexpected response schema from cloud API: {data}")

        elif response.status_code in (401, 403):
            raise InferenceAuthError(f"Authentication failed ({response.status_code}): {response.text}")
        elif response.status_code == 429:
            raise InferenceRateLimitError(f"Rate limit exceeded (429): {response.text}")
        elif response.status_code >= 500:
            raise InferenceConnectionError(f"Cloud API server error ({response.status_code}): {response.text}")
        else:
            raise InferenceError(f"Cloud API request failed with status {response.status_code}: {response.text}")

    def get_status(self) -> str:
        """Returns TUI status string."""
        return f"[CLOUD: {self.provider_name}]"
