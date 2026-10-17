"""Unit and verification tests for ResponseParser and NarrativeValidator."""

import json
from unittest.mock import MagicMock
import pytest

from src.config import ResilienceConfig
from src.engine import (
    InferenceProvider,
    NarrativeValidator,
    ParseResult,
    ResponseParser,
    create_fallback_response,
    extract_json_string,
)
from src.state.models import CharacterStatus, GameResponse, StoryContext


# --- Fuzzy JSON Extractor Tests ---

def test_extract_json_string_clean():
    """Verify clean JSON string extraction without fences."""
    raw = '{"narration": "A shadowy path lies ahead.", "choices": ["[1] Go forward", "[2] Turn back"], "updated_story_context": {"current_location": "Dark Forest", "active_characters": {}, "key_items_held": [], "story_summary": "You entered the dark forest."}}'
    extracted = extract_json_string(raw)
    assert extracted == raw


def test_extract_json_string_markdown_fences():
    """Verify extraction of JSON wrapped in markdown code fences."""
    content = '{"narration": "Text", "choices": ["A", "B"], "updated_story_context": {"current_location": "Cave", "active_characters": {}, "key_items_held": [], "story_summary": "Inside cave."}}'
    
    # ```json fence
    fenced_json = f"```json\n{content}\n```"
    assert extract_json_string(fenced_json) == content

    # ``` fence without json label
    fenced_bare = f"```\n{content}\n```"
    assert extract_json_string(fenced_bare) == content


def test_extract_json_string_with_chatter():
    """Verify extraction with leading preamble and trailing commentary."""
    content = '{"narration": "Text", "choices": ["A", "B"], "updated_story_context": {"current_location": "Cave", "active_characters": {}, "key_items_held": [], "story_summary": "Inside cave."}}'
    noisy = f"Certainly! Here is the adventure update you requested:\n\n{content}\n\nI hope this aids your journey!"
    assert extract_json_string(noisy) == content


def test_extract_json_string_trailing_commas():
    """Verify repair of trailing commas before object and array closures."""
    buggy = '{"narration": "Text", "choices": ["A", "B",], "updated_story_context": {"current_location": "Cave", "active_characters": {}, "key_items_held": [], "story_summary": "Inside cave.",},}'
    fixed = extract_json_string(buggy)
    assert fixed is not None
    # Should successfully parse with standard json.loads after cleanup
    parsed = json.loads(fixed)
    assert parsed["choices"] == ["A", "B"]
    assert parsed["updated_story_context"]["current_location"] == "Cave"


def test_extract_json_string_missing_braces():
    """Verify None return when no braces exist."""
    assert extract_json_string("Just plain text with no json.") is None
    assert extract_json_string("") is None


# --- ResponseParser Tests ---

def test_response_parser_parse_raw_success():
    """Verify parsing valid GameResponse payload."""
    parser = ResponseParser()
    raw = json.dumps({
        "narration": "You stand before the gates of the citadel.",
        "choices": ["[1] Knock on the gate", "[2] Search for a side entrance"],
        "updated_story_context": {
            "current_location": "Citadel Gate",
            "world_tone": "Dark Fantasy",
            "active_characters": {"Gatekeeper": "Neutral"},
            "key_items_held": ["Brass Seal"],
            "story_summary": "You reached the Citadel Gates carrying the Brass Seal.",
            "turn_count": 3,
        },
    })

    result = parser.parse_raw(raw)
    assert result.success is True
    assert result.response is not None
    assert result.response.narration == "You stand before the gates of the citadel."
    assert len(result.response.choices) == 2
    assert result.response.updated_story_context.current_location == "Citadel Gate"


def test_response_parser_parse_raw_validation_error():
    """Verify validation failure when required schema fields are missing or invalid."""
    parser = ResponseParser()

    # Missing narration
    invalid_json = json.dumps({
        "choices": ["[1] Choice 1", "[2] Choice 2"],
        "updated_story_context": {
            "current_location": "Here",
            "active_characters": {},
            "key_items_held": [],
            "story_summary": "Summary here.",
        },
    })
    res = parser.parse_raw(invalid_json)
    assert res.success is False
    assert "ValidationError" in res.error

    # Choices with only 1 item (< 2)
    invalid_choices = json.dumps({
        "narration": "Something happened.",
        "choices": ["[1] Only choice"],
        "updated_story_context": {
            "current_location": "Here",
            "active_characters": {},
            "key_items_held": [],
            "story_summary": "Summary here.",
        },
    })
    res2 = parser.parse_raw(invalid_choices)
    assert res2.success is False
    assert "ValidationError" in res2.error


def test_response_parser_fallback_response():
    """Verify fallback response construction retains local state and safe choices."""
    local_ctx = StoryContext(
        current_location="Ancient Library",
        world_tone="Mystery",
        active_characters={"Librarian": CharacterStatus.ALLY},
        key_items_held=["Tome of Spells"],
        story_summary="You are researching ancient forgotten magic.",
        turn_count=4,
    )

    fallback = create_fallback_response(
        raw_text="Random gibberish that broke the parser",
        local_context=local_ctx,
        error_reason="Bad JSON",
    )

    assert isinstance(fallback, GameResponse)
    assert "Random gibberish that broke the parser" in fallback.narration
    assert len(fallback.choices) == 2
    assert fallback.updated_story_context.current_location == "Ancient Library"
    assert fallback.updated_story_context.key_items_held == ["Tome of Spells"]


def test_response_parser_retry_recovery():
    """Verify automatic recovery when first parse fails but retry succeeds."""
    config = ResilienceConfig(max_parse_retries=1)
    parser = ResponseParser(config=config)
    local_ctx = StoryContext()

    mock_provider = MagicMock(spec=InferenceProvider)
    # Valid response provided on repair call
    valid_repaired = json.dumps({
        "narration": "The lock clicks open.",
        "choices": ["[1] Enter", "[2] Wait"],
        "updated_story_context": {
            "current_location": "Chamber",
            "active_characters": {},
            "key_items_held": [],
            "story_summary": "You unlocked the chamber door.",
        },
    })
    mock_provider.chat.return_value = valid_repaired

    # Initial input is broken JSON
    broken_initial = "{broken json, missing end"
    response = parser.parse_with_retry(
        raw_text=broken_initial,
        local_context=local_ctx,
        provider=mock_provider,
    )

    assert isinstance(response, GameResponse)
    assert response.narration == "The lock clicks open."
    assert response.updated_story_context.current_location == "Chamber"
    mock_provider.chat.assert_called_once()


def test_response_parser_retry_exhausted_fallback():
    """Verify graceful fallback when all retry attempts fail."""
    config = ResilienceConfig(max_parse_retries=1)
    parser = ResponseParser(config=config)
    local_ctx = StoryContext(current_location="Dungeon Cell")

    mock_provider = MagicMock(spec=InferenceProvider)
    # Mock provider also returns broken text
    mock_provider.chat.return_value = "Still bad text."

    response = parser.parse_with_retry(
        raw_text="Broken initial",
        local_context=local_ctx,
        provider=mock_provider,
    )

    assert isinstance(response, GameResponse)
    assert response.updated_story_context.current_location == "Dungeon Cell"
    assert response.choices == ["[1] Look around", "[2] Wait"]


# --- NarrativeValidator (The Guardian) Tests ---

def test_narrative_validator_dead_mans_switch_filters_choice():
    """Verify choice referencing dead character is filtered out when extra choices exist."""
    context = StoryContext(
        active_characters={
            "Goblin King": CharacterStatus.DEAD,
            "Elara": CharacterStatus.ALLY,
        }
    )

    candidate = GameResponse(
        narration="The throne room is quiet.",
        choices=[
            "[1] Speak to Goblin King",
            "[2] Ask Elara for advice",
            "[3] Inspect the golden throne",
        ],
        updated_story_context=StoryContext(current_location="Throne Room"),
    )

    validated = NarrativeValidator.validate_dead_characters(candidate, context)
    assert len(validated.choices) == 2
    assert "[1] Speak to Goblin King" not in validated.choices
    assert "[2] Ask Elara for advice" in validated.choices
    assert "[3] Inspect the golden throne" in validated.choices


def test_narrative_validator_dead_mans_switch_sanitizes_choice():
    """Verify choice referencing dead character is sanitized if filtering violates min choice count."""
    context = StoryContext(
        active_characters={"Goblin King": CharacterStatus.DEAD}
    )

    candidate = GameResponse(
        narration="The throne room is quiet.",
        choices=[
            "[1] Speak to Goblin King",
            "[2] Examine the room",
        ],
        updated_story_context=StoryContext(current_location="Throne Room"),
    )

    validated = NarrativeValidator.validate_dead_characters(candidate, context)
    assert len(validated.choices) == 2
    # Should be sanitized to inspect remains
    assert any("remains of Goblin King" in c for c in validated.choices)
    assert "[2] Examine the room" in validated.choices


def test_narrative_validator_dead_mans_switch_blocks_resurrection():
    """Verify model attempt to revive dead NPC in updated context is blocked."""
    context = StoryContext(
        active_characters={"Goblin King": CharacterStatus.DEAD}
    )

    candidate = GameResponse(
        narration="Goblin King coughs and stands back up.",
        choices=["[1] Fight", "[2] Flee"],
        updated_story_context=StoryContext(
            active_characters={"Goblin King": CharacterStatus.ALIVE}
        ),
    )

    validated = NarrativeValidator.validate_dead_characters(candidate, context)
    assert validated.updated_story_context.active_characters["Goblin King"] == CharacterStatus.DEAD


def test_narrative_validator_summary_and_location_fallback():
    """Verify fallback to authoritative context on empty summary or location."""
    context = StoryContext(
        current_location="Deep Crypt",
        story_summary="You entered the crypt looking for the sacred amulet.",
    )

    # Empty summary (< 10 chars) and empty location
    candidate = GameResponse(
        narration="Dust settles around you.",
        choices=["[1] Proceed", "[2] Halt"],
        updated_story_context=StoryContext(
            current_location="   ",
            story_summary="Too short",
        ),
    )

    validated = NarrativeValidator.validate_response(candidate, context)
    assert validated.updated_story_context.current_location == "Deep Crypt"
    assert validated.updated_story_context.story_summary == "You entered the crypt looking for the sacred amulet."
