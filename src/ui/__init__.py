"""TUI layer: Theme, layout, panels, and terminal rendering."""

from .layout import (
    render_choices_panel,
    render_game_over_screen,
    render_header,
    render_narrative_panel,
    render_settings_menu,
    render_status_panel,
    render_title_splash,
)
from .onboarding import (
    OnboardingFlow,
    map_setting_choice,
    sanitize_player_name,
)
from .renderer import RetroTUI
from .theme import (
    TerminalCapabilities,
    detect_terminal_capabilities,
    get_box_style,
    get_retro_theme,
)

__all__ = [
    "RetroTUI",
    "OnboardingFlow",
    "sanitize_player_name",
    "map_setting_choice",
    "render_header",
    "render_status_panel",
    "render_narrative_panel",
    "render_choices_panel",
    "render_settings_menu",
    "render_game_over_screen",
    "render_title_splash",
    "detect_terminal_capabilities",
    "get_retro_theme",
    "get_box_style",
    "TerminalCapabilities",
]
