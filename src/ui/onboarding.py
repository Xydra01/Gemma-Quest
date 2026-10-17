"""Interactive player onboarding flow and setting selection for Gemma-Quest."""

import html
import re
import time
from typing import Callable, Dict, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import SettingTemplate, load_setting_templates
from src.state.models import PlayerSetup
from src.ui.theme import get_box_style

DEFAULT_PLAYER_NAME = "Hero"
MAX_NAME_LENGTH = 30

BOOT_SEQUENCE_LINES = [
    "PHOENIX ROM BIOS PLUS Version 1.10 A04",
    "Copyright (C) 1985-1988 Phoenix Technologies Ltd.",
    "640KB Base Memory OK, 384KB Extended OK",
    "Starting MS-DOS 4.01...",
    "C:\\GEMMA> RUN DUNGEON.EXE",
    "INITIALIZING RETRO ADVENTURE ENGINE... [OK]",
]


def sanitize_player_name(raw_name: str) -> str:
    """Sanitize player input name, stripping tags, special characters, and enforcing length.

    Args:
        raw_name: Raw input string from the player.

    Returns:
        Sanitized character name, falling back to 'Hero' if blank.
    """
    if not raw_name:
        return DEFAULT_PLAYER_NAME

    # Strip script and style blocks including contents
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw_name, flags=re.DOTALL | re.IGNORECASE)
    # Strip remaining HTML tags
    cleaned = re.sub(r"<[^>]*>", "", cleaned)
    # Unescape HTML entities
    cleaned = html.unescape(cleaned)
    # Allow alphanumeric, spaces, hyphens, and apostrophes
    cleaned = re.sub(r"[^\w\s\'-]", "", cleaned).strip()

    if not cleaned:
        return DEFAULT_PLAYER_NAME

    return cleaned[:MAX_NAME_LENGTH]


def map_setting_choice(choice: str) -> str:
    """Map user selection input to one of the supported setting archetypes.

    Args:
        choice: Raw player selection string (e.g. '1', 'fantasy', '4').

    Returns:
        One of 'fantasy', 'scifi', 'cyberpunk', or 'custom'.
    """
    cleaned = choice.strip().lower()
    if cleaned in ("1", "fantasy", "dark fantasy"):
        return "fantasy"
    elif cleaned in ("2", "scifi", "sci-fi", "space"):
        return "scifi"
    elif cleaned in ("3", "cyberpunk", "cyber", "neon"):
        return "cyberpunk"
    elif cleaned in ("4", "custom"):
        return "custom"

    # Default to fantasy if unrecognized
    return "fantasy"


class OnboardingFlow:
    """Coordinates the 3-step text-based player onboarding sequence in the TUI."""

    def __init__(
        self,
        console: Optional[Console] = None,
        templates: Optional[Dict[str, SettingTemplate]] = None,
    ):
        """Initialize onboarding flow with console and setting templates."""
        self.console = console if console is not None else Console()
        self.templates = templates if templates is not None else load_setting_templates()
        self.box_style = get_box_style()

    def render_boot_sequence(self, sleep_delay: float = 0.0) -> None:
        """Render atmospheric retro DOS boot sequence lines.

        Args:
            sleep_delay: Optional delay in seconds between lines for animation effect.
        """
        for line in BOOT_SEQUENCE_LINES:
            self.console.print(f"[dim grey]{line}[/dim grey]")
            if sleep_delay > 0:
                time.sleep(sleep_delay)
        self.console.print()

    def render_settings_menu(self) -> None:
        """Render formatted menu of available world setting archetypes."""
        table = Table(
            show_header=False,
            box=self.box_style,
            expand=True,
            border_style="bright_green",
            title="[bold yellow]AVAILABLE WORLD SETTINGS[/bold yellow]",
        )
        table.add_column("Key", style="bold #80FF80", width=6)
        table.add_column("Setting", style="bold yellow", width=22)
        table.add_column("Description", style="bright_green")

        # 1. Fantasy
        f_desc = self.templates.get("fantasy", SettingTemplate(
            name="Dark Fantasy",
            description="A medieval kingdom where ancient magic is fading.",
            default_goal="Escape the dungeon.",
            starter_location="The Rusty Gate",
        )).description
        table.add_row("[1]", "Fantasy", f_desc)

        # 2. Sci-Fi
        s_desc = self.templates.get("scifi", SettingTemplate(
            name="Sci-Fi Void",
            description="A drifting space station with failing life support.",
            default_goal="Restore auxiliary power.",
            starter_location="Airlock Bay 4",
        )).description
        table.add_row("[2]", "Sci-Fi", s_desc)

        # 3. Cyberpunk
        c_desc = self.templates.get("cyberpunk", SettingTemplate(
            name="Cyberpunk",
            description="A neon-drenched metropolis ruled by megacorporations.",
            default_goal="Deliver the encrypted data shard.",
            starter_location="Neon Alley",
        )).description
        table.add_row("[3]", "Cyberpunk", c_desc)

        # 4. Custom
        table.add_row("[4]", "Custom", "Create your own unique setting, tone, and opening premise.")

        self.console.print(table)

    def run(self, input_func: Callable[[], str] = input) -> PlayerSetup:
        """Execute the interactive 3-step onboarding sequence.

        Args:
            input_func: Callable providing player input strings (defaults to built-in input).

        Returns:
            Validated PlayerSetup object.
        """
        # Step 1: Character Name
        self.console.print("[bold yellow]════════ CHARACTER INITIALIZATION ════════[/bold yellow]\n")
        self.console.print("[bold #80FF80]> ENTER YOUR CHARACTER NAME (default: Hero):[/bold #80FF80] ", end="")
        raw_name = input_func().strip()
        player_name = sanitize_player_name(raw_name)
        self.console.print(f"[dim grey]Character Name registered: {player_name}[/dim grey]\n")

        # Step 2: Setting Selection
        self.render_settings_menu()
        self.console.print(
            "\n[bold #80FF80]> SELECT SETTING [1] Fantasy [2] Sci-Fi [3] Cyberpunk [4] Custom:[/bold #80FF80] ",
            end="",
        )
        raw_setting = input_func().strip()
        setting_type = map_setting_choice(raw_setting)

        custom_details: Optional[str] = None
        if setting_type == "custom":
            self.console.print("[bold #80FF80]> DESCRIBE YOUR CUSTOM SETTING:[/bold #80FF80] ", end="")
            custom_details = input_func().strip() or "A mysterious and unexplored realm."
            self.console.print(f"[dim grey]Custom setting saved: {custom_details}[/dim grey]\n")
        else:
            self.console.print(f"[dim grey]Setting confirmed: {setting_type.upper()}[/dim grey]\n")

        # Step 3: Initial Goal
        default_goal = "Survive and discover the secrets of the realm."
        if setting_type in self.templates:
            default_goal = self.templates[setting_type].default_goal

        self.console.print(
            f"[bold #80FF80]> WHAT IS YOUR FIRST GOAL? (Press Enter for default: '{default_goal}'):[/bold #80FF80] ",
            end="",
        )
        raw_goal = input_func().strip()
        initial_goal = raw_goal if raw_goal else default_goal
        self.console.print(f"[dim grey]Initial Quest Goal: {initial_goal}[/dim grey]\n")

        return PlayerSetup(
            player_name=player_name,
            setting_type=setting_type,
            custom_setting_details=custom_details,
            initial_goal=initial_goal,
        )
