"""Unit and integration tests for GameEngine async game loop and fallback handling."""

import asyncio
import json
import pytest
import pytest_asyncio
from rich.console import Console

from src.config import AppConfig, GameConfig
from src.engine.base import InferenceError, InferenceProvider
from src.engine.loop import GameEngine, create_turn_fallback_response
from src.state.models import CharacterStatus, GameResponse, StoryContext
from src.ui.input_engine import ActionPayload, InputEngine, InputType
from src.ui.renderer import RetroTUI


class MockProvider(InferenceProvider):
    """Mock inference provider for testing game loop turns."""

    def __init__(self, response_text: str = "", delay: float = 0.0, should_fail: bool = False):
        self.response_text = response_text
        self.delay = delay
        self.should_fail = should_fail
        self.calls = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-gemma"

    def chat(self, messages, options=None) -> str:
        self.calls.append(messages)
        if self.delay > 0:
            import time
            time.sleep(self.delay)
        if self.should_fail:
            raise InferenceError("Mock provider connection failed")
        return self.response_text

    def is_available(self) -> bool:
        return not self.should_fail

    def get_status(self) -> str:
        return "[MOCK: Ready]"


@pytest.fixture
def test_context() -> StoryContext:
    return StoryContext(
        turn_count=0,
        player_name="Lyra",
        current_location="Forgotten Crypt",
        world_tone="Dark Fantasy",
        key_items_held=["Rusty Dagger"],
        active_characters={"Skeleton Guard": CharacterStatus.HOSTILE, "Lyra": CharacterStatus.ALIVE},
        story_summary="Lyra entered the Forgotten Crypt in search of the Sunken Amulet.",
    )


@pytest.fixture
def valid_llm_json() -> str:
    return json.dumps({
        "narration": "You slip past the skeleton guard and discover a silver chest resting upon a pedestal.",
        "choices": [
            "Open the silver chest",
            "Check the pedestal for pressure plates",
            "Sneak around behind the guard",
        ],
        "updated_story_context": {
            "current_location": "Inner Vault",
            "active_characters": {"Skeleton Guard": "Hostile", "Lyra": "Alive"},
            "key_items_held": ["Rusty Dagger", "Silver Key"],
            "story_summary": "Lyra bypassed the guard and discovered the Inner Vault silver chest.",
        },
    })


@pytest.fixture
def mock_tui() -> RetroTUI:
    console = Console(record=True, width=100)
    return RetroTUI(console=console)


class TestTurnFallback:
    """Test algorithmic fallback turn generation."""

    def test_create_turn_fallback_response(self, test_context):
        fallback = create_turn_fallback_response(test_context, "Inspect the locked gate")
        assert "Inspect the locked gate" in fallback.narration or "inspect the locked gate" in fallback.narration
        assert len(fallback.choices) >= 2
        assert fallback.updated_story_context.current_location == "Forgotten Crypt"
        assert "Rusty Dagger" in fallback.updated_story_context.key_items_held
        assert "Action taken: Inspect the locked gate" in fallback.updated_story_context.story_summary


class TestGameEngineTurns:
    """Test GameEngine turn execution, parsing, validation, and state sync."""

    @pytest.mark.asyncio
    async def test_execute_turn_success(self, test_context, valid_llm_json, mock_tui):
        provider = MockProvider(response_text=valid_llm_json)
        engine = GameEngine(
            config=AppConfig(),
            provider=provider,
            tui=mock_tui,
        )
        engine.context = test_context

        response = await engine.execute_turn("Sneak past the guard")
        assert response is not None
        assert "silver chest" in response.narration
        assert len(response.choices) == 3

        # State should be synchronized and turn incremented
        assert engine.context.turn_count == 1
        assert engine.context.current_location == "Inner Vault"
        assert "Silver Key" in engine.context.key_items_held
        assert len(provider.calls) == 1

    @pytest.mark.asyncio
    async def test_execute_turn_timeout_with_fallback(self, test_context, mock_tui):
        # Provider with delay > timeout
        provider = MockProvider(response_text="{}", delay=0.1)
        engine = GameEngine(
            config=AppConfig(),
            provider=provider,
            tui=mock_tui,
            timeout_seconds=0.01,  # Short timeout to trigger timeout error
        )
        engine.context = test_context

        # Simulate user choosing 'f' (fallback) in recovery prompt
        recovery_inputs = iter(["f"])
        engine.input_engine._input_func = lambda _: next(recovery_inputs)

        response = await engine.execute_turn("Charge into the darkness")
        assert response is not None
        assert "charge into the darkness" in response.narration.lower()
        assert engine.context.turn_count == 1

    @pytest.mark.asyncio
    async def test_execute_turn_provider_error_with_quit(self, test_context, mock_tui):
        provider = MockProvider(should_fail=True)
        engine = GameEngine(
            config=AppConfig(),
            provider=provider,
            tui=mock_tui,
        )
        engine.context = test_context

        # Simulate user choosing 'q' (quit) in recovery prompt
        recovery_inputs = iter(["q"])
        engine.input_engine._input_func = lambda _: next(recovery_inputs)

        response = await engine.execute_turn("Examine the runes")
        assert response is None


class TestGameEngineLoop:
    """Test GameEngine interactive loop handling commands and game over."""

    @pytest.mark.asyncio
    async def test_run_quit_command(self, test_context, mock_tui):
        provider = MockProvider(response_text="{}")
        engine = GameEngine(
            config=AppConfig(),
            provider=provider,
            tui=mock_tui,
        )

        initial_response = GameResponse(
            narration="You stand before the ancient gate.",
            choices=["Open gate", "Turn back"],
            updated_story_context=StoryContext(
                current_location="Gate",
                story_summary="Standing at the gate.",
            ),
        )

        # Simulate user typing /quit
        inputs = iter(["/quit"])
        engine.input_engine._input_func = lambda _: next(inputs)

        exit_code = await engine.run(test_context, initial_response)
        assert exit_code == "quit"

    @pytest.mark.asyncio
    async def test_run_help_and_status_commands(self, test_context, mock_tui):
        provider = MockProvider(response_text="{}")
        engine = GameEngine(
            config=AppConfig(),
            provider=provider,
            tui=mock_tui,
        )

        initial_response = GameResponse(
            narration="The gate stands open.",
            choices=["Enter gate", "Wait"],
            updated_story_context=StoryContext(
                current_location="Gate",
                story_summary="The gate is open.",
            ),
        )

        # User types /help, presses Enter, then /status, presses Enter, then /quit
        inputs = iter(["/help", "", "/status", "", "/quit"])
        engine.input_engine._input_func = lambda _: next(inputs)

        exit_code = await engine.run(test_context, initial_response)
        assert exit_code == "quit"

        # Verify panels were rendered to console output
        output = mock_tui.console.export_text()
        assert "COMMANDS & CONTROLS" in output
        assert "DETAILED STORY CONTEXT" in output

    @pytest.mark.asyncio
    async def test_run_game_over(self, test_context, mock_tui):
        # Mark Hero as DEAD
        test_context.active_characters["Lyra"] = CharacterStatus.DEAD

        provider = MockProvider(response_text="{}")
        engine = GameEngine(
            config=AppConfig(),
            provider=provider,
            tui=mock_tui,
            player_name="Lyra",
        )

        initial_response = GameResponse(
            narration="Darkness surrounds you.",
            choices=["Embrace the darkness", "Call out into the void"],
            updated_story_context=StoryContext(
                current_location="Chasm",
                story_summary="Fell into the chasm.",
            ),
        )

        # Player chooses 'q' on game over screen
        inputs = iter(["q"])
        engine.input_engine._input_func = lambda _: next(inputs)

        exit_code = await engine.run(test_context, initial_response)
        assert exit_code == "game_over"

        output = mock_tui.console.export_text()
        assert "GAME OVER" in output or "Your quest came to an end" in output

    @pytest.mark.asyncio
    async def test_run_restart_command(self, test_context, mock_tui):
        provider = MockProvider(response_text="{}")
        engine = GameEngine(
            config=AppConfig(),
            provider=provider,
            tui=mock_tui,
        )

        initial_response = GameResponse(
            narration="You stand at a crossroads.",
            choices=["Left", "Right"],
            updated_story_context=StoryContext(
                current_location="Crossroads",
                story_summary="At the crossroads.",
            ),
        )

        # Simulate user typing /restart
        inputs = iter(["/restart"])
        engine.input_engine._input_func = lambda _: next(inputs)

        exit_code = await engine.run(test_context, initial_response)
        assert exit_code == "restart"
