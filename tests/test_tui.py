"""Unit and verification tests for Retro DOS TUI theme, layout, and rendering."""

import os
from unittest.mock import patch
import pytest
from rich import box
from rich.console import Console

from src.state.models import CharacterStatus, StoryContext
from src.ui import (
    RetroTUI,
    TerminalCapabilities,
    detect_terminal_capabilities,
    get_box_style,
    get_retro_theme,
    render_choices_panel,
    render_game_over_screen,
    render_header,
    render_narrative_panel,
    render_status_panel,
    render_title_splash,
)


# --- Theme & Capability Tests ---

def test_detect_terminal_capabilities():
    """Verify terminal capability detection with default and mocked environments."""
    caps = detect_terminal_capabilities()
    assert isinstance(caps, TerminalCapabilities)
    assert caps.terminal_width > 0
    assert caps.terminal_height > 0

    # Test NO_COLOR disables color
    with patch.dict(os.environ, {"NO_COLOR": "1"}):
        caps_no_color = detect_terminal_capabilities()
        assert caps_no_color.supports_color is False

    # Test FORCE_ASCII disables unicode
    with patch.dict(os.environ, {"FORCE_ASCII": "1"}):
        caps_ascii = detect_terminal_capabilities()
        assert caps_ascii.supports_unicode is False


def test_get_retro_theme():
    """Verify retro theme definition contains required semantic style tags."""
    theme = get_retro_theme(monochrome=False)
    required_tags = [
        "tui.primary",
        "tui.header",
        "tui.location",
        "tui.choice",
        "tui.choice.num",
        "tui.error",
        "tui.dim",
        "tui.border",
        "tui.status.alive",
        "tui.status.dead",
        "tui.status.hostile",
        "tui.status.ally",
    ]
    for tag in required_tags:
        assert tag in theme.styles

    # Verify monochrome theme builds valid styles
    mono_theme = get_retro_theme(monochrome=True)
    for tag in required_tags:
        assert tag in mono_theme.styles


def test_get_box_style():
    """Verify box style selection falls back to ASCII when unicode is unsupported."""
    unicode_caps = TerminalCapabilities(
        supports_unicode=True, supports_color=True, terminal_width=80, terminal_height=25
    )
    assert get_box_style(unicode_caps) == box.HEAVY

    ascii_caps = TerminalCapabilities(
        supports_unicode=False, supports_color=True, terminal_width=80, terminal_height=25
    )
    assert get_box_style(ascii_caps) == box.ASCII


# --- Layout & Panel Tests ---

def test_render_header():
    """Verify header panel contains title, turn number, and inference provider badge."""
    panel = render_header(
        title="GEMMA QUEST",
        turn_count=12,
        provider_status="[LOCAL: gemma4:e2b]",
        box_style=box.HEAVY,
    )

    console = Console(record=True, width=100)
    console.print(panel)
    output = console.export_text()

    assert "GEMMA QUEST" in output
    assert "Turn:" in output
    assert "12" in output
    assert "[LOCAL: gemma4:e2b]" in output


def test_render_status_panel():
    """Verify status panel contains location, tone, characters, items, and summary."""
    context = StoryContext(
        current_location="Iron Fortress",
        world_tone="Grimdark",
        active_characters={
            "Captain Vane": CharacterStatus.HOSTILE,
            "Old Jorah": CharacterStatus.DEAD,
        },
        key_items_held=["Rusty Halberd", "Iron Key"],
        story_summary="You entered the courtyard under heavy rain.",
        turn_count=4,
    )

    panel = render_status_panel(context, box_style=box.HEAVY)
    console = Console(record=True, width=100)
    console.print(panel)
    output = console.export_text()

    assert "Iron Fortress" in output
    assert "Grimdark" in output
    assert "Captain Vane" in output
    assert "Old Jorah" in output
    assert "Rusty Halberd" in output
    assert "Iron Key" in output
    assert "courtyard under heavy rain" in output


def test_render_status_panel_empty_lists():
    """Verify status panel renders None for empty characters and items."""
    context = StoryContext(
        current_location="Empty Cell",
        active_characters={},
        key_items_held=[],
        story_summary="You wake up alone.",
    )

    panel = render_status_panel(context, box_style=box.HEAVY)
    console = Console(record=True, width=100)
    console.print(panel)
    output = console.export_text()

    assert "Empty Cell" in output
    assert "None" in output


def test_render_narrative_panel():
    """Verify narrative panel renders atmospheric story text with line wrapping."""
    narration = "The heavy oak door groans on ancient hinges as a freezing gust of wind sweeps across the stone floor."
    panel = render_narrative_panel(narration, box_style=box.HEAVY)

    console = Console(record=True, width=100)
    console.print(panel)
    output = console.export_text()

    assert "NARRATION" in output
    assert "heavy oak door groans" in output


def test_render_choices_panel():
    """Verify choices panel numbers items and includes free-form hint."""
    choices = [
        "Pick the rusted lock",
        "Examine the torch sconce",
        "Wait quietly in the shadows",
    ]
    panel = render_choices_panel(choices, box_style=box.HEAVY)

    console = Console(record=True, width=100)
    console.print(panel)
    output = console.export_text()

    assert "[1] Pick the rusted lock" in output
    assert "[2] Examine the torch sconce" in output
    assert "[3] Wait quietly in the shadows" in output
    assert "[0] Free-form" in output


def test_render_game_over_screen():
    """Verify game over screen contains banner, location, and quit prompt."""
    context = StoryContext(
        current_location="The Dragon Lair",
        story_summary="You fell beneath the fiery breath of the dragon.",
        turn_count=15,
    )
    panel = render_game_over_screen(context, box_style=box.HEAVY)

    console = Console(record=True, width=100)
    console.print(panel)
    output = console.export_text()

    assert "The Dragon Lair" in output
    assert "Turns Survived: 15" in output
    assert "fiery breath of the dragon" in output
    assert "Press Q to quit" in output


def test_render_title_splash():
    """Verify opening title splash contains game title art and retro subtitle."""
    panel = render_title_splash(box_style=box.HEAVY)

    console = Console(record=True, width=100)
    console.print(panel)
    output = console.export_text()

    assert "D U N G E O N" in output
    assert "Retro 1980s Text Adventure" in output


# --- RetroTUI Controller Tests ---

def test_retrotui_game_frame_rendering():
    """Verify RetroTUI coordinates and renders a complete 3-zone frame."""
    console = Console(record=True, width=100)
    tui = RetroTUI(console=console)

    context = StoryContext(current_location="Forest Clearing", turn_count=2)
    narration = "The wind rustles through the birch branches."
    choices = ["[1] Walk north", "[2] Rest by campfire"]

    tui.render_game_frame(
        context=context,
        narration=narration,
        choices=choices,
        provider_status="[LOCAL: gemma4:e2b]",
        soft_clear=True,
    )

    output = console.export_text()
    assert "Forest Clearing" in output
    assert "wind rustles" in output
    assert "[1] Walk north" in output
    assert "[2] Rest by campfire" in output


def test_retrotui_input_handling():
    """Verify numeric selection, out-of-range error, and free-form input."""
    console = Console(record=True, width=100)
    tui = RetroTUI(console=console)
    choices = ["Open chest", "Flee room"]

    # 1. Valid numeric selection '1'
    result_1 = tui.get_user_input(choices, input_func=lambda: "1")
    assert result_1 == "Open chest"

    # 2. Valid numeric selection '2'
    result_2 = tui.get_user_input(choices, input_func=lambda: "2")
    assert result_2 == "Flee room"

    # 3. Out-of-range number '9'
    result_invalid = tui.get_user_input(choices, input_func=lambda: "9")
    assert result_invalid == "INVALID_INPUT"

    # 4. Free-form text input
    result_free = tui.get_user_input(choices, input_func=lambda: "cast light spell")
    assert result_free == "cast light spell"

    # 5. Empty input defaults to WAIT
    result_empty = tui.get_user_input(choices, input_func=lambda: "")
    assert result_empty == "WAIT"
