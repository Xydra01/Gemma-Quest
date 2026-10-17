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


def repair_syntax_quirks(candidate: str) -> str:
    """Repair common LLM JSON syntax quirks (trailing commas before closures)."""
    candidate = re.sub(r",\s*\}", "}", candidate)
    candidate = re.sub(r",\s*\]", "]", candidate)
    return candidate


def repair_truncated_json(text: str) -> Optional[str]:
    """Attempt to balance unclosed quotes, brackets, and braces in truncated JSON."""
    start = text.find("{")
    if start == -1 or '"narration"' not in text[start:]:
        return None
    sub = text[start:].strip()

    candidates_to_try = [sub]
    last_comma = sub.rfind(",")
    if last_comma != -1:
        candidates_to_try.append(sub[:last_comma])

    for cand in candidates_to_try:
        quotes = 0
        escape = False
        for c in cand:
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                quotes += 1
        if quotes % 2 != 0:
            cand += '"'

        open_brackets = cand.count("[") - cand.count("]")
        if open_brackets > 0:
            cand += "]" * open_brackets
        open_braces = cand.count("{") - cand.count("}")
        if open_braces > 0:
            cand += "}" * open_braces

        repaired = repair_syntax_quirks(cand)
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict) and "narration" in parsed:
                return repaired
        except Exception:
            continue

    return None


def extract_json_string(raw_text: str) -> Optional[str]:
    """Extract a candidate JSON string from raw LLM output.

    Handles:
    - Markdown code fences (```json ... ``` or ``` ... ```)
    - Leading conversational preamble and trailing commentary
    - Balanced curly brace depth scanning across conversational scratchpads
    - Trailing comma removal before object/array closures
    - Incomplete/truncated JSON recovery

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
    fence_matches = list(re.finditer(fence_pattern, text, flags=re.DOTALL | re.IGNORECASE))
    for match in fence_matches:
        candidate = repair_syntax_quirks(match.group(1).strip())
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return candidate
        except Exception:
            pass

    # 2. Balanced brace depth scan for JSON candidate objects
    narr_candidate = None
    fallback_candidate = None

    for match in re.finditer(r"\{", text):
        start = match.start()
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = repair_syntax_quirks(text[start : i + 1].strip())
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                if "narration" in parsed:
                                    return candidate
                                elif "choices" in parsed or "updated_story_context" in parsed:
                                    if not fallback_candidate:
                                        fallback_candidate = candidate
                        except Exception:
                            pass
                        break

    if fallback_candidate:
        return fallback_candidate

    # 3. Fallback: Outermost curly brace search
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = repair_syntax_quirks(text[start_idx : end_idx + 1].strip())
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return candidate
        except Exception:
            pass

    # 4. Repair truncated JSON if cut off mid-response
    repaired = repair_truncated_json(text)
    if repaired:
        return repaired

    return None


def create_fallback_response(
    raw_text: str, local_context: StoryContext, error_reason: str = ""
) -> GameResponse:
    """Construct a safe, clean, non-crashing GameResponse when parsing fails completely.

    Guarantees that model thinking traces, scratchpad analysis, and raw metadata
    are never leaked into the player's narration box.

    Args:
        raw_text: The raw text that could not be parsed.
        local_context: The current authoritative StoryContext.
        error_reason: Diagnostic reason for failure.

    Returns:
        Guaranteed valid GameResponse with clean narration and preserved state.
    """
    logger.warning("Using fallback GameResponse due to parse failure: %s", error_reason)

    fallback_context = local_context.model_copy(deep=True)
    extracted_narration: Optional[str] = None
    extracted_choices: List[str] = []

    # 1. Fuzzy regex extraction of drafted narration
    m_narr = re.search(r'\"narration\"\s*:\s*\"([^\"]{15,})\"', raw_text)
    if m_narr:
        extracted_narration = m_narr.group(1).strip()
    else:
        m_narr2 = re.search(
            r'(?:Refining\s+)?Narration:?\s*["\']([^"\'\n]{20,})["\']',
            raw_text,
            re.IGNORECASE,
        )
        if m_narr2:
            extracted_narration = m_narr2.group(1).strip()

    # 2. Fuzzy regex extraction of drafted choices
    m_choices = re.search(r'\"choices\"\s*:\s*\[(.*?)\]', raw_text, re.DOTALL)
    if m_choices:
        for c in m_choices.group(1).split(","):
            clean_c = c.strip().strip("\"'").strip()
            if clean_c and len(clean_c) > 2 and clean_c not in extracted_choices:
                extracted_choices.append(clean_c)

    if not extracted_choices:
        m_choices2 = re.search(r'(?:Choices|Choice 1):?\s*([^\n]+)', raw_text, re.IGNORECASE)
        if m_choices2:
            raw_c = m_choices2.group(1)
            for c in re.split(r'\",\s*\"|\',\s*\'|,\s*', raw_c):
                clean_c = c.strip().strip("\"'").strip()
                if clean_c and len(clean_c) > 2 and clean_c not in extracted_choices:
                    extracted_choices.append(clean_c)

    # 3. Fuzzy regex extraction of updated location
    m_loc = re.search(r'\"current_location\"\s*:\s*\"([^\"]+)\"', raw_text)
    if not m_loc:
        m_loc = re.search(r'`current_location`\s*:\s*["\']?([^"\'\n]+)["\']?', raw_text)
    if m_loc:
        loc_candidate = m_loc.group(1).strip()
        if len(loc_candidate) > 2 and not any(k in loc_candidate.lower() for k in ["string", "{", "}", "location"]):
            fallback_context.current_location = loc_candidate

    # 4. Fuzzy regex extraction of story summary
    m_sum = re.search(r'\"story_summary\"\s*:\s*\"([^\"]{20,})\"', raw_text)
    if not m_sum:
        m_sum = re.search(r'(?:Summary|`story_summary`):\s*["\']?([^"\'\n]{20,})["\']?', raw_text)
    if m_sum:
        sum_candidate = m_sum.group(1).strip()
        is_contaminated = (
            sum_candidate.startswith("*")
            or sum_candidate.startswith("-")
            or any(k in sum_candidate for k in ["Role: Editor", "Task: Summarize", "Constraint", "Draft 1:"])
        )
        if len(sum_candidate) > 15 and not is_contaminated:
            fallback_context.story_summary = sum_candidate

    # 5. Clean narration sanitization: filter out thinking/meta lines
    if not extracted_narration:
        thinking_filter_patterns = [
            r"^\s*\*\s*",
            r"^\s*-\s*",
            r"Game Title:", r"Game Name:", r"Player Character:", r"World Setting:",
            r"Initial Quest:", r"Goal:", r"Constraint", r"Style:", r"Location:",
            r"Atmosphere:", r"Choice \d+:", r"Choices:", r"Updated Story Context:",
            r"Coherence:", r"Self-Correction", r"Drafting", r"Double check",
            r"Final check", r"Valid JSON", r"`narration`", r"`choices`",
            r"`updated_story_context`", r"Everything looks correct",
            r"Return ONLY", r"\{", r"\}",
        ]
        candidate_lines = []
        for line in raw_text.splitlines():
            line_str = line.strip()
            if not line_str or len(line_str) < 5:
                continue
            if any(re.search(pat, line_str, re.IGNORECASE) for pat in thinking_filter_patterns):
                continue
            candidate_lines.append(line_str)

        if candidate_lines:
            extracted_narration = " ".join(candidate_lines).strip()

    # 6. Safety check: if raw_text did not contain thinking metadata, allow original clean text
    if not extracted_narration:
        clean_cand = re.sub(r"```(?:json)?", "", raw_text).replace("```", "").strip()
        if clean_cand and not clean_cand.startswith("*") and "Game Title" not in clean_cand:
            extracted_narration = clean_cand

    # 7. Final safety check: if still empty or tainted, use contextual fallback
    if not extracted_narration or len(extracted_narration) < 15 or extracted_narration.startswith("*"):
        extracted_narration = (
            f"You stand in {fallback_context.current_location}, taking in your immediate surroundings. "
            f"The environment is quiet as you consider your next move."
        )

    # 8. Final choices validation
    final_choices = extracted_choices[:4] if (len(extracted_choices) >= 2) else list(DEFAULT_FALLBACK_CHOICES)

    return GameResponse(
        narration=extracted_narration,
        choices=final_choices,
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
