"""Unit and verification tests for PromptBuilder and SummaryRefiner."""

from unittest.mock import MagicMock
import pytest

from src.config import ContextConfig
from src.engine import (
    InferenceConnectionError,
    InferenceProvider,
    PromptBuilder,
    SummaryRefiner,
)
from src.state.models import CharacterStatus, PlayerSetup, StoryContext


# --- PromptBuilder Tests ---

def test_prompt_builder_character_formatting():
    """Verify formatting of active characters map to prompt string."""
    empty_chars = {}
    assert PromptBuilder.format_characters(empty_chars) == "None"

    chars = {
        "Goblin Scout": CharacterStatus.DEAD,
        "Elara": CharacterStatus.ALLY,
        "Prison Guard": CharacterStatus.HOSTILE,
    }
    formatted = PromptBuilder.format_characters(chars)
    assert "- Goblin Scout: Dead" in formatted
    assert "- Elara: Ally" in formatted
    assert "- Prison Guard: Hostile" in formatted


def test_prompt_builder_items_formatting():
    """Verify formatting of key items held to prompt string."""
    assert PromptBuilder.format_items([]) == "None"
    assert PromptBuilder.format_items(["Rusty Key"]) == "Rusty Key"
    assert PromptBuilder.format_items(["Rusty Key", "Health Potion", "Old Map"]) == (
        "Rusty Key, Health Potion, Old Map"
    )


def test_prompt_builder_system_prompt():
    """Verify that system prompt contains persona, rules, and schema."""
    system_prompt = PromptBuilder.build_system_prompt()
    assert "Dungeon Master" in system_prompt
    assert "Gemma Dungeon" in system_prompt
    assert "RULES:" in system_prompt
    assert "1. COHERENCE:" in system_prompt
    assert "2. STYLE:" in system_prompt
    assert "3. CHOICES:" in system_prompt
    assert "4. STATE MANAGEMENT:" in system_prompt
    assert "5. OUTPUT FORMAT:" in system_prompt
    assert "[OUTPUT FORMAT]" in system_prompt
    assert '"narration":' in system_prompt
    assert '"choices":' in system_prompt
    assert '"updated_story_context":' in system_prompt


def test_prompt_builder_user_context_block():
    """Verify interpolation of dynamic story state into user context block."""
    context = StoryContext(
        current_location="Abandoned Crypt",
        world_tone="Gothic Horror",
        active_characters={"Skeleton King": CharacterStatus.HOSTILE},
        key_items_held=["Bone Dagger"],
        story_summary="You descended the staircase into the crypt.",
        turn_count=2,
    )

    block = PromptBuilder.build_user_context_block(context, "Light the torch")

    assert "[STORY CONTEXT]" in block
    assert "Location: Abandoned Crypt" in block
    assert "Tone: Gothic Horror" in block
    assert "Active Characters:\n- Skeleton King: Hostile" in block
    assert "Key Items Held: Bone Dagger" in block
    assert "Recent Story Summary: You descended the staircase into the crypt." in block
    assert "[PLAYER ACTION]\nLight the torch" in block


def test_prompt_builder_chat_messages():
    """Verify chat messages structure with separate system and user roles."""
    context = StoryContext()
    messages = PromptBuilder.build_chat_messages(context, "[1] Look around")

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Dungeon Master" in messages[0]["content"]
    assert "[STORY CONTEXT]" in messages[1]["content"]
    assert "[PLAYER ACTION]\n[1] Look around" in messages[1]["content"]


def test_prompt_builder_monolithic_prompt():
    """Verify monolithic single string prompt construction."""
    context = StoryContext()
    monolithic = PromptBuilder.build_monolithic_prompt(context, "[2] Wait")

    assert "You are the Dungeon Master" in monolithic
    assert "[STORY CONTEXT]" in monolithic
    assert "[PLAYER ACTION]\n[2] Wait" in monolithic
    assert "[OUTPUT FORMAT]" in monolithic


def test_prompt_builder_opening_messages():
    """Verify generation of initial opening scene prompts from PlayerSetup."""
    setup = PlayerSetup(
        player_name="Kaelen",
        setting_type="cyberpunk",
        custom_setting_details="Neon rain soaked alleyways of Sector 7",
        initial_goal="Infiltrate the Arasaka sub-vault",
    )

    messages = PromptBuilder.build_opening_messages(setup)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    user_content = messages[1]["content"]
    assert "[NEW GAME INITIALIZATION]" in user_content
    assert "Player Character: Kaelen" in user_content
    assert "World Setting: cyberpunk (Neon rain soaked alleyways of Sector 7)" in user_content
    assert "Initial Quest/Goal: Infiltrate the Arasaka sub-vault" in user_content


def test_prompt_builder_repair_and_correction_messages():
    """Verify JSON repair and dead character correction prompt generation."""
    repair_msgs = PromptBuilder.build_json_repair_messages(
        invalid_output="{bad json...", error_details="Unterminated string"
    )
    assert len(repair_msgs) == 2
    assert "Validation Error: Unterminated string" in repair_msgs[1]["content"]
    assert "{bad json..." in repair_msgs[1]["content"]

    dead_msgs = PromptBuilder.build_dead_character_correction_messages(
        dead_character="Lord Malakor", original_narration="Lord Malakor walks up to you."
    )
    assert len(dead_msgs) == 2
    assert "Character 'Lord Malakor' is DEAD" in dead_msgs[1]["content"]
    assert "Lord Malakor walks up to you." in dead_msgs[1]["content"]


# --- SummaryRefiner Tests ---

def test_summary_refiner_should_refine_heuristics():
    """Verify heuristic checks for length, interval, and major events."""
    config = ContextConfig(
        summary_refinement_char_threshold=200,
        summary_refinement_turn_interval=5,
    )
    refiner = SummaryRefiner(config=config)

    # 1. Short summary on turn 1 -> False
    short_context = StoryContext(story_summary="Short summary.", turn_count=1)
    assert refiner.should_refine(short_context) is False

    # 2. Length > 200 chars -> True
    long_summary = "A" * 201
    long_context = StoryContext(story_summary=long_summary, turn_count=1)
    assert refiner.should_refine(long_context) is True

    # 3. Candidate summary override > 200 chars -> True
    assert refiner.should_refine(short_context, candidate_summary="B" * 205) is True

    # 4. Turn count multiple of 5 -> True
    turn_5_context = StoryContext(story_summary="Normal length", turn_count=5)
    assert refiner.should_refine(turn_5_context) is True

    turn_10_context = StoryContext(story_summary="Normal length", turn_count=10)
    assert refiner.should_refine(turn_10_context) is True

    turn_3_context = StoryContext(story_summary="Normal length", turn_count=3)
    assert refiner.should_refine(turn_3_context) is False

    # Turn 0 should not trigger interval refinement
    turn_0_context = StoryContext(story_summary="Normal length", turn_count=0)
    assert refiner.should_refine(turn_0_context) is False

    # 5. Major event flag -> True regardless of length or turn count
    assert refiner.should_refine(short_context, is_major_event=True) is True


def test_summary_refiner_clean_output():
    """Verify output cleaning removes code fences, [SUMMARY] labels, and quotes."""
    raw_with_fence = "```\nYou escaped the cell and made it to the forest.\n```"
    assert SummaryRefiner._clean_summary_output(raw_with_fence) == (
        "You escaped the cell and made it to the forest."
    )

    raw_with_prefix = "[SUMMARY] You breached the inner ward gates."
    assert SummaryRefiner._clean_summary_output(raw_with_prefix) == (
        "You breached the inner ward gates."
    )

    raw_with_quotes = '"The dragon retreated to the mountain peak."'
    assert SummaryRefiner._clean_summary_output(raw_with_quotes) == (
        "The dragon retreated to the mountain peak."
    )


def test_summary_refiner_execution_success():
    """Verify successful summary refinement using a mock InferenceProvider."""
    refiner = SummaryRefiner()
    mock_provider = MagicMock(spec=InferenceProvider)
    mock_provider.chat.return_value = (
        "You escaped the dungeon. You entered the dark woods. A howling sound echoed nearby."
    )

    result = refiner.refine_summary(
        provider=mock_provider,
        player_input="Sneak out the door",
        last_narration="You tiptoe past the sleeping guard and reach the woods.",
        current_summary="You were trapped in a cell.",
    )

    assert result == (
        "You escaped the dungeon. You entered the dark woods. A howling sound echoed nearby."
    )
    mock_provider.chat.assert_called_once()
    call_args = mock_provider.chat.call_args
    options = call_args[1]["options"]
    assert options["temperature"] == 0.2
    assert options["max_tokens"] == 512


def test_summary_refiner_execution_fallback():
    """Verify fallback to current summary on error or invalid output."""
    refiner = SummaryRefiner()
    mock_provider = MagicMock(spec=InferenceProvider)

    # 1. Provider exception fallback
    mock_provider.chat.side_effect = InferenceConnectionError("Server unreachable")
    current_summary = "Existing summary that must be preserved."
    result = refiner.refine_summary(
        provider=mock_provider,
        player_input="Fight",
        last_narration="Battle takes place.",
        current_summary=current_summary,
    )
    assert result == current_summary

    # 2. Too short / empty output fallback (< 15 chars)
    mock_provider.chat.side_effect = None
    mock_provider.chat.return_value = "Short."
    result2 = refiner.refine_summary(
        provider=mock_provider,
        player_input="Fight",
        last_narration="Battle takes place.",
        current_summary=current_summary,
    )
    assert result2 == current_summary
