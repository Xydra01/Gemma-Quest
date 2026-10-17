"""Retro DOS terminal color theme, box styles, and terminal capability detection."""

from dataclasses import dataclass
import os
import shutil
import sys
from typing import Optional
from rich import box
from rich.box import Box
from rich.theme import Theme

# Retro DOS Color Palette Constants
COLOR_PRIMARY = "#00FF00"      # Bright green (narration, primary text)
COLOR_BACKGROUND = "#000000"   # Black background
COLOR_HEADER = "#FFFF00"       # Bold yellow (titles, headers, location names)
COLOR_CHOICE = "#FFFFFF"       # White (choice text, user input)
COLOR_CHOICE_NUM = "#80FF80"   # Light green (choice numbers)
COLOR_ERROR = "#FF0000"        # Bold red (errors, game over, dead status)
COLOR_DIM = "#555555"          # Dim grey (system messages, borders, hints)
COLOR_ALLY = "#00FFFF"         # Cyan (allies)
COLOR_HOSTILE = "#FFA500"      # Orange/amber (hostile NPCs)


@dataclass
class TerminalCapabilities:
    """Detected terminal capabilities for formatting and box rendering."""

    supports_unicode: bool
    supports_color: bool
    terminal_width: int
    terminal_height: int


def detect_terminal_capabilities() -> TerminalCapabilities:
    """Detect whether the current terminal supports Unicode box drawing and ANSI colors.

    Per Risk_Mitigation.md (R6), checks stdout encoding, terminal type,
    and NO_COLOR environment settings.

    Returns:
        TerminalCapabilities instance with detected features.
    """
    encoding = getattr(sys.stdout, "encoding", "") or ""
    supports_unicode = "utf" in encoding.lower()

    # Check for forced color disabling or dumb terminal
    term_env = os.environ.get("TERM", "").lower()
    no_color = "NO_COLOR" in os.environ or term_env == "dumb"
    supports_color = not no_color

    # If terminal is dumb or explicitly ascii, disable unicode
    if term_env in ("dumb", "emacs", "vanilla") or os.environ.get("FORCE_ASCII") == "1":
        supports_unicode = False

    size = shutil.get_terminal_size(fallback=(80, 25))

    return TerminalCapabilities(
        supports_unicode=supports_unicode,
        supports_color=supports_color,
        terminal_width=size.columns,
        terminal_height=size.lines,
    )


def get_retro_theme(monochrome: bool = False) -> Theme:
    """Build a Rich Theme configured with the retro DOS aesthetic.

    Args:
        monochrome: If True, falls back to monochrome bold/dim styles.

    Returns:
        Rich Theme instance with semantic TUI styles.
    """
    if monochrome:
        return Theme({
            "tui.primary": "default",
            "tui.header": "bold",
            "tui.location": "bold",
            "tui.choice": "default",
            "tui.choice.num": "bold",
            "tui.error": "bold reverse",
            "tui.dim": "dim",
            "tui.border": "default",
            "tui.border.header": "bold",
            "tui.border.choice": "bold",
            "tui.status.alive": "default",
            "tui.status.dead": "dim strike",
            "tui.status.hostile": "bold",
            "tui.status.ally": "bold underline",
        })

    return Theme({
        "tui.primary": f"bold {COLOR_PRIMARY}",
        "tui.header": f"bold {COLOR_HEADER}",
        "tui.location": f"bold {COLOR_HEADER}",
        "tui.choice": f"{COLOR_CHOICE}",
        "tui.choice.num": f"bold {COLOR_CHOICE_NUM}",
        "tui.error": f"bold {COLOR_ERROR}",
        "tui.dim": f"{COLOR_DIM}",
        "tui.border": f"{COLOR_PRIMARY}",
        "tui.border.header": f"{COLOR_HEADER}",
        "tui.border.choice": f"{COLOR_CHOICE_NUM}",
        "tui.status.alive": f"bold {COLOR_PRIMARY}",
        "tui.status.dead": f"bold {COLOR_ERROR}",
        "tui.status.hostile": f"bold {COLOR_HOSTILE}",
        "tui.status.ally": f"bold {COLOR_ALLY}",
    })


def get_box_style(capabilities: Optional[TerminalCapabilities] = None) -> Box:
    """Select the appropriate box-drawing character set based on terminal capability.

    Returns Unicode HEAVY (╔, ═, ║, ╚, ╝) for modern UTF-8 terminals,
    or ASCII (+, -, |) as fallback.

    Args:
        capabilities: Optional detected capabilities. If None, detects automatically.

    Returns:
        Rich Box character set.
    """
    caps = capabilities if capabilities is not None else detect_terminal_capabilities()
    if caps.supports_unicode:
        return box.HEAVY
    return box.ASCII
