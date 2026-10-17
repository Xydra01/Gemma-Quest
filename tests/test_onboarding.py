"""Unit and verification tests for Player Onboarding Flow and Opening Scene Generator."""

import json
from unittest.mock import MagicMock
import pytest
from rich.console import Console

from src.config import SettingTemplate
from src.engine import (
    InferenceConnectionError,
    InferenceProvider,
    OpeningSceneGenerator,
    create_initial_context,
    create_starter_fallback_response,
)
from src.state.models import GameResponse, PlayerSetup, StoryContext
from src.ui import (
    OnboardingFlow,
    map_setting_choice,
    render_settings_menu,
    sanitize_player_name,
)


# --- Sanitization & Mapping Tests ---

def test_sanitize_player_name():
    """Verify name sanitization, tag stripping, length limiting, and Hero fallback."""
    assert sanitize_player_name("Arthur") == "Arthur"
    assert sanitize_player_name("   Galahad   ") == "Galahad"
    assert sanitize_player_name("<script>evil()</script>Sir Robin") == "Sir Robin"
    assert sanitize_player_name("<b>Lancelot</b>") == "Lancelot"
    assert sanitize_player_name("") == "Hero"
    assert sanitize_player_name("   ") == "Hero"
    assert sanitize_player_name("<<<>>>") == "Hero"

    long_name = "Alexander The Great Conqueror of Known Realms"
    assert len(sanitize_player_name(long_name)) <= 30


def test_map_setting_choice():
    """Verify numeric and text mapping to setting archetypes."""
    assert map_setting_choice("1") == "fantasy"
    assert map_setting_choice("fantasy") == "fantasy"
    assert map_setting_choice("Dark Fantasy") == "fantasy"

    assert map_setting_choice("2") == "scifi"
    assert map_setting_choice("sci-fi") == "scifi"
    assert map_setting_choice("space") == "scifi"

    assert map_setting_choice("3") == "cyberpunk"
    assert map_setting_choice("neon") == "cyberpunk"

    assert map_setting_choice("4") == "custom"
    assert map_setting_choice("custom") == "custom"

    # Default fallback
    assert map_setting_choice("unknown_setting") == "fantasy"


# --- OnboardingFlow Tests ---

def test_onboarding_flow_defaults():
    """Verify OnboardingFlow interactive steps with default fallbacks."""
    inputs = iter([
        "",    # Name: empty -> 'Hero'
        "1",   # Setting: 1 -> 'fantasy'
        "",    # Goal: empty -> template default goal
    ])

    flow = OnboardingFlow(console=Console(record=True))
    setup = flow.run(input_func=lambda: next(inputs))

    assert isinstance(setup, PlayerSetup)
    assert setup.player_name == "Hero"
    assert setup.setting_type == "fantasy"
    assert setup.custom_setting_details is None
    assert "dungeon" in setup.initial_goal.lower()


def test_onboarding_flow_custom_setting():
    """Verify OnboardingFlow captures custom setting details and player goal."""
    inputs = iter([
        "Kaelen",                                       # Name
        "4",                                            # Custom setting
        "Post-apocalyptic wasteland with acid geysers", # Custom details
        "Find clean water source",                      # Custom goal
    ])

    flow = OnboardingFlow(console=Console(record=True))
    setup = flow.run(input_func=lambda: next(inputs))

    assert setup.player_name == "Kaelen"
    assert setup.setting_type == "custom"
    assert setup.custom_setting_details == "Post-apocalyptic wasteland with acid geysers"
    assert setup.initial_goal == "Find clean water source"


def test_render_settings_menu():
    """Verify render_settings_menu displays archetype descriptions."""
    templates = {
        "fantasy": SettingTemplate(
            name="Dark Fantasy",
            description="Ancient castle ruins.",
            default_goal="Find escape.",
            starter_location="Dungeon Gate",
        ),
        "scifi": SettingTemplate(
            name="Sci-Fi Void",
            description="Derelict freighter.",
            default_goal="Fix engine.",
            starter_location="Cockpit",
        ),
        "cyberpunk": SettingTemplate(
            name="Cyberpunk",
            description="High-tech low-life neon city.",
            default_goal="Hack node.",
            starter_location="Cyber Bar",
        ),
    }

    panel = render_settings_menu(templates)
    console = Console(record=True, width=100)
    console.print(panel)
    output = console.export_text()

    assert "AVAILABLE WORLD SETTINGS" in output
    assert "Fantasy" in output
    assert "Sci-Fi" in output
    assert "Cyberpunk" in output
    assert "Custom" in output


# --- Context Seeding & Fallback Tests ---

def test_create_initial_context_archetypes():
    """Verify initial StoryContext seeding for fantasy, scifi, and cyberpunk."""
    # 1. Fantasy
    f_setup = PlayerSetup(player_name="Thorin", setting_type="fantasy", initial_goal="Recover stolen ring")
    f_ctx = create_initial_context(f_setup)
    assert f_ctx.current_location == "The Rusty Gate"
    assert f_ctx.world_tone == "Dark Fantasy"
    assert f_ctx.turn_count == 0
    assert "Thorin" in f_ctx.story_summary
    assert "Recover stolen ring" in f_ctx.story_summary

    # 2. Sci-Fi
    s_setup = PlayerSetup(player_name="Ripley", setting_type="scifi", initial_goal="Contain the entity")
    s_ctx = create_initial_context(s_setup)
    assert s_ctx.current_location == "Airlock Bay 4"
    assert s_ctx.world_tone == "Sci-Fi Void"
    assert "Ripley" in s_ctx.story_summary

    # 3. Cyberpunk
    c_setup = PlayerSetup(player_name="Zero", setting_type="cyberpunk", initial_goal="Jack into the mainframe")
    c_ctx = create_initial_context(c_setup)
    assert s_ctx.turn_count == 0
    assert c_ctx.current_location == "Neon Alley Underpass"
    assert c_ctx.world_tone == "Cyberpunk Metropolis"


def test_create_starter_fallback_response():
    """Verify deterministic starter scenes provide 3 choices and turn 1 context."""
    setup = PlayerSetup(player_name="Aria", setting_type="fantasy", initial_goal="Escape the catacombs")
    ctx = create_initial_context(setup)

    starter = create_starter_fallback_response(setup, ctx)
    assert isinstance(starter, GameResponse)
    assert "Aria" in starter.narration
    assert "Escape the catacombs" in starter.narration
    assert len(starter.choices) == 3
    assert starter.updated_story_context.turn_count == 1
    assert starter.updated_story_context.current_location == ctx.current_location


# --- OpeningSceneGenerator Tests ---

def test_opening_scene_generator_with_mock_provider():
    """Verify OpeningSceneGenerator dispatches opening messages and parses response."""
    generator = OpeningSceneGenerator()
    setup = PlayerSetup(player_name="Kael", setting_type="scifi", initial_goal="Repair nav-computer")

    mock_provider = MagicMock(spec=InferenceProvider)
    mock_provider.is_available.return_value = True

    response_json = json.dumps({
        "narration": "Klaxons blare across the bridge. Kael, you stand before the smoking navigation console.",
        "choices": [
            "[1] Bypass the burned-out fuse",
            "[2] Check the ship's diagnostic terminal",
            "[3] Radio engineering for assistance",
        ],
        "updated_story_context": {
            "current_location": "Bridge",
            "world_tone": "Sci-Fi Void",
            "active_characters": {"Ship AI": "Neutral"},
            "key_items_held": [],
            "story_summary": "Kael arrived on the bridge to repair the nav-computer.",
            "turn_count": 1,
        },
    })
    mock_provider.chat.return_value = response_json

    seed_ctx, opening_resp = generator.generate(setup, provider=mock_provider)

    assert seed_ctx.turn_count == 0
    assert opening_resp.updated_story_context.turn_count == 1
    assert "Klaxons blare" in opening_resp.narration
    assert len(opening_resp.choices) == 3
    assert opening_resp.updated_story_context.current_location == "Bridge"
    mock_provider.chat.assert_called_once()


def test_opening_scene_generator_offline_fallback():
    """Verify graceful fallback to template opening when provider fails or is offline."""
    generator = OpeningSceneGenerator()
    setup = PlayerSetup(player_name="Hero", setting_type="cyberpunk", initial_goal="Evade the strike team")

    mock_provider = MagicMock(spec=InferenceProvider)
    mock_provider.is_available.return_value = True
    mock_provider.chat.side_effect = InferenceConnectionError("Server unreachable")

    seed_ctx, opening_resp = generator.generate(setup, provider=mock_provider)

    assert seed_ctx.turn_count == 0
    assert opening_resp.updated_story_context.turn_count == 1
    assert "Acid rain hisses" in opening_resp.narration
    assert len(opening_resp.choices) == 3
