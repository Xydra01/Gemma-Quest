"""Unit tests for InputEngine, sanitization, choice validation, and slash commands."""

import pytest
import pytest_asyncio

from src.ui.input_engine import (
    ActionPayload,
    InputEngine,
    InputType,
    sanitize_player_input,
)


class TestInputSanitization:
    """Test string sanitization against injection, HTML, and long text."""

    def test_sanitize_empty_and_none(self):
        assert sanitize_player_input(None) == ""
        assert sanitize_player_input("") == ""
        assert sanitize_player_input("   ") == ""

    def test_sanitize_html_tags(self):
        raw = "<b>Inspect</b> the <div class='altar'>ancient altar</div>"
        clean = sanitize_player_input(raw)
        assert clean == "Inspect the ancient altar"

    def test_sanitize_script_and_style_tags(self):
        raw = "<script>alert('pwned')</script>Cast lightning bolt<style>body{color:red}</style>"
        clean = sanitize_player_input(raw)
        assert clean == "Cast lightning bolt"
        assert "alert" not in clean
        assert "pwned" not in clean
        assert "style" not in clean

    def test_sanitize_collapses_whitespace(self):
        raw = "Drink \n\t  from   the\r\n fountain  "
        clean = sanitize_player_input(raw)
        assert clean == "Drink from the fountain"

    def test_sanitize_truncates_long_input(self):
        raw = "A" * 300
        clean = sanitize_player_input(raw, max_length=150)
        assert len(clean) == 150


class TestInputParsing:
    """Test parsing logic against choices, commands, and edge cases."""

    @pytest.fixture
    def sample_choices(self):
        return [
            "[1] Examine the stone gargoyle",
            "Open the heavy oak door",
            "3. Retreat back to the entrance",
        ]

    def test_parse_empty_input_defaults_to_wait(self, sample_choices):
        engine = InputEngine()
        payload = engine.parse_raw_input("", sample_choices)
        assert payload.input_type == InputType.WAIT
        assert "[WAIT]" in payload.action_text

    def test_parse_valid_numeric_choices(self, sample_choices):
        engine = InputEngine()

        # Choice 1
        p1 = engine.parse_raw_input("1", sample_choices)
        assert p1.input_type == InputType.CHOICE
        assert p1.action_text == "Examine the stone gargoyle"

        # Choice 2
        p2 = engine.parse_raw_input("2", sample_choices)
        assert p2.input_type == InputType.CHOICE
        assert p2.action_text == "Open the heavy oak door"

        # Choice 3
        p3 = engine.parse_raw_input("3", sample_choices)
        assert p3.input_type == InputType.CHOICE
        assert p3.action_text == "Retreat back to the entrance"

    def test_parse_out_of_bounds_numeric_choice(self, sample_choices):
        engine = InputEngine()

        p_neg = engine.parse_raw_input("4", sample_choices)
        assert p_neg.input_type == InputType.INVALID
        assert "Invalid selection [4]" in p_neg.error_message

        p_zero = engine.parse_raw_input("0", sample_choices)
        assert p_zero.input_type == InputType.INVALID
        assert "custom actions" in p_zero.error_message

    def test_parse_custom_narrative_action(self, sample_choices):
        engine = InputEngine()
        payload = engine.parse_raw_input("Throw a torch into the dark pit", sample_choices)
        assert payload.input_type == InputType.CUSTOM
        assert payload.action_text == "Throw a torch into the dark pit"
        assert payload.error_message is None

    def test_parse_commands(self, sample_choices):
        engine = InputEngine()

        # Help
        assert engine.parse_raw_input("/help", sample_choices).command_name == "help"
        assert engine.parse_raw_input("/h", sample_choices).command_name == "help"

        # Status
        assert engine.parse_raw_input("/status", sample_choices).command_name == "status"
        assert engine.parse_raw_input("/inv", sample_choices).command_name == "status"

        # Quit
        assert engine.parse_raw_input("/quit", sample_choices).command_name == "quit"
        assert engine.parse_raw_input("/exit", sample_choices).command_name == "quit"

        # Restart
        assert engine.parse_raw_input("/restart", sample_choices).command_name == "restart"

        # Unknown command
        p_unknown = engine.parse_raw_input("/teleport", sample_choices)
        assert p_unknown.input_type == InputType.INVALID
        assert "Unknown command" in p_unknown.error_message


class TestInputEngineInteractive:
    """Test interactive sync and async input capture loops."""

    def test_get_action_reprompts_on_invalid_until_valid(self):
        inputs = iter(["99", "0", "2"])
        engine = InputEngine(input_func=lambda _: next(inputs))

        choices = ["Look left", "Look right"]
        payload = engine.get_action(choices)

        assert payload.input_type == InputType.CHOICE
        assert payload.action_text == "Look right"

    @pytest.mark.asyncio
    async def test_get_action_async(self):
        inputs = iter(["Search the mossy wall"])
        engine = InputEngine(input_func=lambda _: next(inputs))

        choices = ["Wait", "Rest"]
        payload = await engine.get_action_async(choices)

        assert payload.input_type == InputType.CUSTOM
        assert payload.action_text == "Search the mossy wall"

    def test_get_action_handles_keyboard_interrupt(self):
        def raise_interrupt(_):
            raise KeyboardInterrupt()

        engine = InputEngine(input_func=raise_interrupt)
        payload = engine.get_action(["Action 1"])
        assert payload.input_type == InputType.COMMAND
        assert payload.command_name == "quit"
