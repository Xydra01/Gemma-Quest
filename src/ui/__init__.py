"""TUI layer: Theme, layout, panels, and terminal rendering."""

from .layout import (
    render_choices_panel,
    render_game_over_screen,
    render_header,
    render_narrative_panel,
    render_status_panel,
    render_title_splash,
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
    "render_header",
    "render_status_panel",
    "render_narrative_panel",
    "render_choices_panel",
    "render_game_over_screen",
    "render_title_splash",
    "detect_terminal_capabilities",
    "get_retro_theme",
    "get_box_style",
    "TerminalCapabilities",
]
