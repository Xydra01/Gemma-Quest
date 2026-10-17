"""Fuzzy JSON extractor, Pydantic parser, and retry recovery engine."""

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import ValidationError

from src.config import ResilienceConfig, load_config
from src.engine.base import InferenceProvider
from src.engine.prompt_builder import PromptBuilder
from src.state.models import GameResponse, StoryContext

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_CHOICES = ["[1] Look around", "[2] Wait"]


@dataclass
class ParseResult:
    """Encapsulates the outcome of a JSON extraction and parsing attempt."""

    success: bool
    response: Optional[GameResponse] = None
    error: Optional[str] = None
    raw_json: Optional[str] = None
    raw_text: str = ""
    attempts: int = 1
    is_fallback: bool = False


def extract_json_string(raw_text: str) -> Optional[str]:
    """Extract a candidate JSON string from raw LLM output.

    Handles:
    - Markdown code fences (```json ... ``` or ``` ... ```)
    - Leading conversational preamble and trailing commentary
    - Outermost curly brace boundary extraction
    - Trailing comma removal before object/array closures

    Args:
        raw_text: Raw response string from the model.

    Returns:
        Cleaned candidate JSON string, or None if no valid braces found.
    """
    if not raw_text:
        return None

    text = raw_text.strip()

    # 1. Attempt to extract from markdown code fences if present
    fence_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    fence_match = re.search(fence_pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
    else:
        # 2. Outermost curly brace search
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return None
        candidate = text[start_idx : end_idx + 1].strip()

    # 3. Repair common JSON syntax quirks (trailing commas before } or ])
    candidate = re.sub(r",\s*\}", "}", candidate)
    candidate = re.sub(r",\s*\]", "]", candidate)

    return candidate


def create_fallback_response(
    raw_text: str, local_context: StoryContext, error_reason: str = ""
) -> GameResponse:
    """Construct a safe, non-crashing GameResponse when parsing fails completely.

    Args:
        raw_text: The raw text that could not be parsed.
        local_context: The current authoritative StoryContext.
        error_reason: Diagnostic reason for failure.

    Returns:
        Guaranteed valid GameResponse preserving existing world state.
    """
    logger.warning("Using fallback GameResponse due to parse failure: %s", error_reason)

    # Clean raw text for narration excerpt
    clean_narration = raw_text.strip()
    # Strip any markdown fences
    clean_narration = re.sub(r"```(?:json)?", "", clean_narration).replace("```", "").strip()

    # If completely empty or gibberish, provide a standard descriptive narrative
    if not clean_narration or len(clean_narration) < 10:
        clean_narration = (
            f"You stand in {local_context.current_location}, carefully assessing your options. "
            f"The environment is quiet."
        )

    # Maintain existing story context
    fallback_context = local_context.model_copy(deep=True)

    return GameResponse(
        narration=clean_narration,
        choices=list(DEFAULT_FALLBACK_CHOICES),
        updated_story_context=fallback_context,
    )


class ResponseParser:
    """Robust JSON response parser with automatic correction retries and fallback."""

    def __init__(self, config: Optional[ResilienceConfig] = None):
        """Initialize parser with resilience configuration."""
        if config is None:
            try:
                game_config = load_config()
                self.config = game_config.resilience
            except Exception:
                self.config = ResilienceConfig()
        else:
            self.config = config

    def parse_raw(self, raw_text: str) -> ParseResult:
        """Attempt to extract and parse raw model text into a GameResponse.

        Args:
            raw_text: The string output received from the LLM.

        Returns:
            ParseResult detailing success or validation error details.
        """
        candidate = extract_json_string(raw_text)
        if not candidate:
            return ParseResult(
                success=False,
                error="No JSON structure found between '{' and '}'.",
                raw_text=raw_text,
            )

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return ParseResult(
                success=False,
                error=f"JSONDecodeError: {exc.msg} at line {exc.lineno} col {exc.colno}",
                raw_json=candidate,
                raw_text=raw_text,
            )

        if not isinstance(data, dict):
            return ParseResult(
                success=False,
                error="Extracted JSON root is not an object/dictionary.",
                raw_json=candidate,
                raw_text=raw_text,
            )

        try:
            game_response = GameResponse(**data)
            return ParseResult(
                success=True,
                response=game_response,
                raw_json=candidate,
                raw_text=raw_text,
            )
        except ValidationError as val_err:
            return ParseResult(
                success=False,
                error=f"Pydantic ValidationError: {val_err}",
                raw_json=candidate,
                raw_text=raw_text,
            )

    def parse_with_retry(
        self,
        raw_text: str,
        local_context: StoryContext,
        provider: Optional[InferenceProvider] = None,
        max_retries: Optional[int] = None,
    ) -> GameResponse:
        """Parse raw text, triggering automatic correction retry if JSON is invalid.

        If parsing fails and retries are exhausted, gracefully degrades to a
        fallback GameResponse maintaining existing game state.

        Args:
            raw_text: Raw output received from the model.
            local_context: Authoritative local StoryContext.
            provider: Active InferenceProvider for executing correction calls.
            max_retries: Optional override for max retry attempts (default from config).

        Returns:
            A validated GameResponse (either successfully parsed, repaired, or fallback).
        """
        retries_limit = (
            max_retries
            if max_retries is not None
            else self.config.max_parse_retries
        )

        current_raw = raw_text
        attempts = 1

        # Attempt initial parse
        result = self.parse_raw(current_raw)
        if result.success and result.response:
            return result.response

        logger.warning(
            "Initial JSON parsing failed: %s. Initiating retry loop (max=%d).",
            result.error,
            retries_limit,
        )

        # Retry loop if provider is available
        if provider is not None:
            while attempts <= retries_limit:
                logger.info("Dispatching JSON repair attempt %d/%d...", attempts, retries_limit)
                repair_messages = PromptBuilder.build_json_repair_messages(
                    invalid_output=current_raw,
                    error_details=result.error or "Malformed JSON syntax",
                )

                try:
                    repaired_raw = provider.chat(
                        repair_messages,
                        options={"temperature": 0.1, "max_tokens": 1024},
                    )
                    attempts += 1
                    current_raw = repaired_raw
                    repair_result = self.parse_raw(current_raw)

                    if repair_result.success and repair_result.response:
                        logger.info("JSON successfully repaired on attempt %d.", attempts - 1)
                        return repair_result.response

                    result = repair_result
                except Exception as exc:
                    logger.warning("Repair call attempt %d failed: %s", attempts, exc)
                    attempts += 1

        # Graceful degradation
        logger.error(
            "All JSON parse/repair attempts exhausted (%d). Falling back to safe response.",
            attempts,
        )
        return create_fallback_response(
            raw_text=raw_text,
            local_context=local_context,
            error_reason=result.error or "Parsing failed after retries",
        )
