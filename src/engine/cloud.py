"""Cloud inference provider communicating with remote APIs via HTTPX."""

import logging
import time
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
        api_key: str = "",
        endpoint: str = "",
        model_id: str = "gemini-2.5-flash",
        provider_name: str = "vertex_ai",
        timeout_seconds: int = 30,
        use_vertexai: bool = False,
        project_id: Optional[str] = None,
        location: str = "us-central1",
    ):
        self.api_key = api_key.strip() if api_key else ""
        self.endpoint = endpoint.strip() if endpoint else ""
        self.model_id = model_id
        self.provider_name = provider_name
        self.timeout_seconds = timeout_seconds
        self.use_vertexai = use_vertexai
        self.project_id = project_id
        self.location = location
        self._genai_client = None

    def is_available(self) -> bool:
        """Verifies that an API key or Vertex AI project is configured."""
        if self.use_vertexai and self.project_id:
            return True
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
        if not self.use_vertexai and not self.api_key:
            raise InferenceAuthError("Cloud inference failed: Missing API key.")

        options = options or {}
        temperature = float(options.get("temperature", 0.8))
        max_tokens = int(options.get("max_tokens", 1024))
        timeout = float(options.get("timeout_seconds", self.timeout_seconds))

        if self.use_vertexai:
            result = self._chat_vertex_ai(messages, temperature, max_tokens, timeout)
        elif self._is_google_ai():
            result = self._chat_google_ai(messages, temperature, max_tokens, timeout)
        else:
            result = self._chat_openai_rest(messages, options, timeout)

        if not result or not result.strip():
            raise InferenceError("Cloud provider returned empty or whitespace response.")

        return result

    def _chat_vertex_ai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        """Executes completion via Google GenAI Vertex AI client using Application Default Credentials."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise InferenceConnectionError("google-genai package is not installed.") from e

        try:
            if not self._genai_client:
                self._genai_client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location,
                )

            system_instruction = None
            contents = []

            for m in messages:
                role = m.get("role", "user").lower()
                text = m.get("content", "")

                if role == "system":
                    system_instruction = text
                else:
                    api_role = "model" if role == "assistant" else "user"
                    contents.append(
                        types.Content(
                            role=api_role,
                            parts=[types.Part.from_text(text=text)],
                        )
                    )

            if not contents:
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text="Continue the story.")],
                    )
                ]

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max(max_tokens, 2048),
                system_instruction=system_instruction,
            )

            # Map model if using legacy or placeholder model ID
            model = self.model_id
            if model in ("gemma-4-26b-a4b-it", "default", ""):
                model = "gemini-2.5-flash"

            response = self._genai_client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            if not response or not response.text:
                raise InferenceError("Vertex AI returned empty response.")
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if any(k in err_str for k in ("401", "403", "PermissionDenied", "Unauthenticated")):
                raise InferenceAuthError(f"Vertex AI authentication failed: {e}") from e
            if any(k in err_str for k in ("429", "ResourceExhausted", "RESOURCE_EXHAUSTED")):
                raise InferenceRateLimitError(f"Vertex AI quota / rate limit exceeded: {e}") from e
            if "timeout" in err_str.lower() or "deadline" in err_str.lower():
                raise InferenceTimeoutError(f"Vertex AI request timed out: {e}") from e
            raise InferenceError(f"Vertex AI request failed: {e}") from e

    def _chat_google_ai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        """Executes completion via Google AI Studio / Gemini API with retry on server errors."""
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

        # Ensure sufficient output token headroom for reasoning models (e.g. Gemma 4)
        effective_max_tokens = max(max_tokens, 2048)

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": effective_max_tokens,
            },
        }

        if system_instructions:
            payload["systemInstruction"] = {"parts": system_instructions}

        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                    if response.status_code in (500, 503) and attempt < max_attempts:
                        logger.warning(
                            "Google AI API server error (%d). Retrying (attempt %d/%d)...",
                            response.status_code,
                            attempt,
                            max_attempts,
                        )
                        time.sleep(1.0)
                        continue
                    return self._handle_response(response, is_google=True)
            except (httpx.ConnectError, httpx.NetworkError) as e:
                if attempt < max_attempts:
                    logger.warning("Network error on Google AI API attempt %d: %s. Retrying...", attempt, e)
                    time.sleep(1.0)
                    continue
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
                prompt_feedback = data.get("promptFeedback", {})
                if prompt_feedback.get("blockReason"):
                    raise InferenceError(
                        f"Google AI API blocked prompt: {prompt_feedback.get('blockReason')}"
                    )

                candidates = data.get("candidates", [])
                if not candidates:
                    raise InferenceError("Google AI API returned no candidates.")

                candidate = candidates[0]
                finish_reason = candidate.get("finishReason", "")
                if finish_reason in ("SAFETY", "RECITATION", "BLOCKLIST"):
                    raise InferenceError(
                        f"Google AI API content generation blocked by safety filters (finishReason: {finish_reason})."
                    )

                if "content" in candidate:
                    parts = candidate["content"].get("parts", [])
                    # Separate non-thought completion parts from internal reasoning/thought parts
                    non_thought_parts = [
                        p["text"]
                        for p in parts
                        if not p.get("thought", False) and "text" in p
                    ]
                    if non_thought_parts:
                        full_text = "".join(non_thought_parts).strip()
                        if full_text:
                            return full_text

                    # Fallback: if only thought parts were emitted
                    thought_parts = [p["text"] for p in parts if "text" in p]
                    if thought_parts:
                        thought_text = "".join(thought_parts).strip()
                        if finish_reason == "MAX_TOKENS":
                            logger.warning(
                                "Google AI model reached maxOutputTokens during internal reasoning."
                            )
                        if thought_text:
                            return thought_text

                raise InferenceError(f"Google AI candidate contained no usable text parts: {candidate}")

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
        if self.use_vertexai:
            model_display = self.model_id if self.model_id not in ("gemma-4-26b-a4b-it", "default", "") else "gemini-2.5-flash"
            return f"[CLOUD: Vertex AI ({model_display})]"
        return f"[CLOUD: {self.provider_name}]"
