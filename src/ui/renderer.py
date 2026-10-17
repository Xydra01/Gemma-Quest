"""RetroTUI controller coordinating theme, console rendering, and input mechanics."""

import logging
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel

from src.state.models import StoryContext
from src.ui.layout import (
    render_choices_panel,
    render_game_over_screen,
    render_header,
    render_narrative_panel,
    render_status_panel,
    render_title_splash,
)
from src.ui.theme import (
    TerminalCapabilities,
    detect_terminal_capabilities,
    get_box_style,
    get_retro_theme,
)

logger = logging.getLogger(__name__)


class RetroTUI:
    """Terminal User Interface controller for Gemma-Quest."""

    def __init__(
        self,
        console: Optional[Console] = None,
        monochrome: bool = False,
        force_ascii: bool = False,
    ):
        """Initialize the retro TUI with terminal capabilities and theme.

        Args:
            console: Optional pre-configured Rich Console instance.
            monochrome: If True, uses high-contrast monochrome styles.
            force_ascii: If True, forces ASCII box drawing (+, -, |).
        """
        self.capabilities: TerminalCapabilities = detect_terminal_capabilities()
        if force_ascii:
            self.capabilities.supports_unicode = False

        self.box_style = get_box_style(self.capabilities)
        self.theme = get_retro_theme(monochrome=monochrome)

        if console is None:
            self.console = Console(theme=self.theme, force_terminal=True)
        else:
            self.console = console

    def clear_screen(self, soft: bool = True) -> None:
        """Clear the terminal or perform a soft scroll clear to avoid flickering.

        Args:
            soft: If True, prints vertical spacing instead of hard terminal clear.
        """
        if soft:
            self.console.print("\n" * 3)
        else:
            self.console.clear()

    def print_panel(self, panel: Panel) -> None:
        """Print a single panel to the console."""
        self.console.print(panel)

    def render_game_frame(
        self,
        context: StoryContext,
        narration: str,
        choices: List[str],
        provider_status: str = "[LOCAL: Ollama]",
        title: str = "GEMMA DUNGEON v1.0",
        soft_clear: bool = False,
    ) -> None:
        """Render a complete coordinated 3-zone game frame.

        Zones rendered:
        1. Header bar with game title, turn count, and provider status badge.
        2. Status panel with location, tone, active characters, and key items.
        3. Narrative panel with story description.
        4. Choices panel with numbered actions and free-form indicator.

        Args:
            context: Current authoritative StoryContext.
            narration: Story text from turn generation.
            choices: List of 2 to 4 actionable choices.
            provider_status: Active inference provider status badge.
            title: Title text for the header.
            soft_clear: If True, adds vertical spacing before rendering.
        """
        if soft_clear:
            self.clear_screen(soft=True)

        header = render_header(
            title=title,
            turn_count=context.turn_count,
            provider_status=provider_status,
            box_style=self.box_style,
        )
        status = render_status_panel(context, box_style=self.box_style)
        narrative = render_narrative_panel(narration, box_style=self.box_style)
        actions = render_choices_panel(choices, box_style=self.box_style)

        self.console.print(header)
        self.console.print(status)
        self.console.print(narrative)
        self.console.print(actions)

    def render_title(self) -> None:
        """Render the retro title splash screen."""
        self.console.print(render_title_splash(box_style=self.box_style))

    def render_game_over(self, context: StoryContext) -> None:
        """Render the dramatic Game Over conclusion screen."""
        self.console.print(render_game_over_screen(context, box_style=self.box_style))

    def get_user_input(
        self,
        choices: List[str],
        prompt_text: str = "> Select action [1-{max}] or type command: ",
        input_func=input,
    ) -> str:
        """Capture player choice or free-form input with validation.

        Args:
            choices: List of available action strings.
            prompt_text: Prompt format string.
            input_func: Callable for input capture (defaults to built-in input).

        Returns:
            The selected choice string or raw free-form text entered by the player.
        """
        formatted_prompt = prompt_text.replace("{max}", str(len(choices)))
        self.console.print(f"\n[tui.choice.num]{formatted_prompt}[/tui.choice.num]", end="")

        try:
            raw_input = input_func().strip()
        except (EOFError, KeyboardInterrupt):
            return "QUIT"

        if not raw_input:
            return "WAIT"

        # Check for numeric selection
        try:
            choice_num = int(raw_input)
            if 1 <= choice_num <= len(choices):
                return choices[choice_num - 1]
            elif choice_num == 0:
                self.console.print("[tui.dim]> Enter free-form action:[/tui.dim] ", end="")
                return input_func().strip() or "WAIT"
            else:
                self.console.print(
                    f"[tui.error]Invalid selection ({choice_num}). Choose between 1 and {len(choices)}.[/tui.error]"
                )
                return "INVALID_INPUT"
        except ValueError:
            # Player entered free-form text
            return raw_input

    def run_onboarding(
        self,
        input_func=input,
        show_boot: bool = False,
    ):
        """Execute the interactive player onboarding sequence.

        Args:
            input_func: Callable providing player inputs.
            show_boot: If True, renders atmospheric DOS boot lines.

        Returns:
            Validated PlayerSetup object.
        """
        from src.ui.onboarding import OnboardingFlow

        flow = OnboardingFlow(console=self.console)
        if show_boot:
            flow.render_boot_sequence()

        return flow.run(input_func=input_func)
