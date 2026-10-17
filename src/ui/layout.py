"""Layout renderers, panel builders, and retro screen components."""

from typing import Any, List, Optional
from rich import box
from rich.box import Box
from rich.panel import Panel
from rich.table import Table

from src.state.models import CharacterStatus, StoryContext

TITLE_ART = r"""
 ██████╗ ███████╗███╗   ███╗███╗   ███╗ █████╗ 
██╔════╝ ██╔════╝████╗ ████║████╗ ████║██╔══██╗
██║  ███╗█████╗  ██╔████╔██║██╔████╔██║███████║
██║   ██║██╔══╝  ██║╚██╔╝██║██║╚██╔╝██║██╔══██║
╚██████╔╝███████╗██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║
 ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝
          -- D U N G E O N --
"""

GAME_OVER_ART = r"""
  ██████╗  █████╗ ███╗   ███╗███████╗     ██████╗ ██╗   ██╗███████╗██████╗ 
 ██╔════╝ ██╔══██╗████╗ ████║██╔════╝    ██╔═══██╗██║   ██║██╔════╝██╔══██╗
 ██║  ███╗███████║██╔████╔██║█████╗      ██║   ██║██║   ██║█████╗  ██████╔╝
 ██║   ██║██╔══██║██║╚██╔╝██║██╔══╝      ██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗
 ╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗    ╚██████╔╝ ╚████╔╝ ███████╗██║  ██║
  ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝     ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝
"""


def render_header(
    title: str = "GEMMA DUNGEON v1.0",
    turn_count: int = 0,
    provider_status: str = "[LOCAL: Ollama]",
    box_style: Box = box.HEAVY,
) -> Panel:
    """Render the top header bar with title, turn counter, and inference status badge.

    Args:
        title: Game title string.
        turn_count: Current turn number.
        provider_status: Status badge from InferenceProvider (e.g. [LOCAL: gemma4:e2b]).
        box_style: Rich Box character set.

    Returns:
        Formatted Header Panel.
    """
    header_table = Table.grid(expand=True)
    header_table.add_column(justify="left", ratio=3)
    header_table.add_column(justify="center", ratio=2)
    header_table.add_column(justify="right", ratio=3)

    title_part = f"[bold yellow]{title}[/bold yellow]"
    turn_part = f"[bold yellow]Turn:[/] [bright_green]{turn_count}[/bright_green]"
    provider_part = f"[dim grey]{provider_status}[/dim grey]"

    header_table.add_row(title_part, turn_part, provider_part)

    return Panel(
        header_table,
        box=box_style,
        border_style="bold yellow",
        padding=(0, 1),
    )


def render_status_panel(
    context: StoryContext,
    box_style: Box = box.HEAVY,
) -> Panel:
    """Render the status monitor panel displaying location, characters, and items.

    Args:
        context: Current authoritative StoryContext.
        box_style: Rich Box character set.

    Returns:
        Formatted Status Panel.
    """
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="bold yellow", width=14)
    table.add_column(style="bright_green")

    # 1. Location & Tone
    table.add_row("[LOCATION]", f"[bold yellow]{context.current_location}[/bold yellow]")
    if context.world_tone:
        table.add_row("[WORLD TONE]", f"[dim grey]{context.world_tone}[/dim grey]")

    # 2. Active Characters with status styling
    if context.active_characters:
        char_entries = []
        for name, status in context.active_characters.items():
            st_val = status.value if hasattr(status, "value") else str(status)
            if status == CharacterStatus.DEAD:
                tag = "bold red"
            elif status == CharacterStatus.HOSTILE:
                tag = "bold yellow"
            elif status == CharacterStatus.ALLY:
                tag = "bold cyan"
            else:
                tag = "bright_green"
            char_entries.append(f"[{tag}]{name} ({st_val})[/{tag}]")
        chars_text = ", ".join(char_entries)
    else:
        chars_text = "[dim grey]None[/dim grey]"
    table.add_row("[CHARACTERS]", chars_text)

    # 3. Key Items Held
    if context.key_items_held:
        items_text = ", ".join(f"[white]{item}[/white]" for item in context.key_items_held)
    else:
        items_text = "[dim grey]None[/dim grey]"
    table.add_row("[KEY ITEMS]", items_text)

    # Subtitle with recent rolling summary
    clean_summary = " ".join(context.story_summary.split()) if context.story_summary else "Beginning adventure..."
    summary_subtitle = f"[dim grey]Memory: {clean_summary}[/dim grey]"

    return Panel(
        table,
        box=box_style,
        title="[bold yellow]GAME STATUS[/bold yellow]",
        subtitle=summary_subtitle,
        border_style="bright_green",
        padding=(0, 1),
    )


def render_scene_panel(
    visual_art: str,
    title: str = "SCENE ILLUSTRATION",
    box_style: Box = box.HEAVY,
) -> Panel:
    """Render the retro pixel art / ASCII scene illustration panel.

    Args:
        visual_art: Pre-formatted ANSI TrueColor half-blocks or ASCII art string.
        title: Title banner for the panel.
        box_style: Rich Box character set.

    Returns:
        Formatted Scene Illustration Panel.
    """
    from rich.text import Text

    clean_art = visual_art.strip("\r\n") if visual_art else ""
    rich_content = Text.from_ansi(clean_art, no_wrap=True)

    return Panel(
        rich_content,
        box=box_style,
        title=f"[bold yellow]{title}[/bold yellow]",
        border_style="yellow",
        padding=(0, 1),
    )


def render_narrative_panel(
    narration: str,
    box_style: Box = box.HEAVY,
) -> Panel:
    """Render the main narrative descriptive text panel.

    Args:
        narration: Story description text from model or narrator.
        box_style: Rich Box character set.

    Returns:
        Formatted Narrative Panel.
    """
    clean_text = narration.strip() if narration else "[dim grey]The dungeon is silent.[/dim grey]"

    return Panel(
        f"[bright_green]{clean_text}[/bright_green]",
        box=box_style,
        title="[bold yellow]NARRATION[/bold yellow]",
        border_style="bright_green",
        padding=(1, 2),
    )


def render_choices_panel(
    choices: List[str],
    box_style: Box = box.HEAVY,
) -> Panel:
    """Render the numbered interactive choices panel.

    Args:
        choices: List of 2 to 4 action strings.
        box_style: Rich Box character set.

    Returns:
        Formatted Choices Panel.
    """
    lines = []
    for i, choice in enumerate(choices, 1):
        clean_choice = choice.strip()
        if clean_choice.startswith(f"[{i}]"):
            clean_choice = clean_choice[len(f"[{i}]"):].strip()
        elif clean_choice.startswith(f"{i}."):
            clean_choice = clean_choice[len(f"{i}."):].strip()

        lines.append(
            f"[bold #80FF80][{i}][/bold #80FF80] [white]{clean_choice}[/white]"
        )

    lines.append("")
    lines.append("[dim grey][0] Free-form: type any custom action...[/dim grey]")

    return Panel(
        "\n".join(lines),
        box=box_style,
        title="[bold yellow]ACTIONS[/bold yellow]",
        border_style="bold #80FF80",
        padding=(0, 2),
    )


def render_game_over_screen(
    context: StoryContext,
    box_style: Box = box.HEAVY,
) -> Panel:
    """Render the Game Over dramatic death or conclusion screen.

    Args:
        context: Final StoryContext.
        box_style: Rich Box character set.

    Returns:
        Formatted Game Over Panel.
    """
    content = [
        f"[bold red]{GAME_OVER_ART.strip()}[/bold red]",
        "",
        f"[bold yellow]Your quest came to an end in {context.current_location}.[/bold yellow]",
        f"[dim grey]Turns Survived: {context.turn_count}[/dim grey]",
        "",
        f"[bright_green]Final Tale: {context.story_summary}[/bright_green]",
        "",
        "[dim grey]Press [bold]Q[/bold] to quit or [bold]R[/bold] to restart a new adventure.[/dim grey]",
    ]

    return Panel(
        "\n".join(content),
        box=box_style,
        border_style="bold red",
        padding=(1, 2),
    )


def render_title_splash(
    box_style: Box = box.HEAVY,
) -> Panel:
    """Render the opening title splash screen.

    Args:
        box_style: Rich Box character set.

    Returns:
        Formatted Splash Title Panel.
    """
    content = [
        f"[bright_green]{TITLE_ART.strip()}[/bright_green]",
        "",
        "[bold yellow]A Retro 1980s Text Adventure powered by Gemma 4[/bold yellow]",
        "[dim grey]Strict Coherence • Context-First Memory • Terminal Aesthetics[/dim grey]",
    ]

    return Panel(
        "\n".join(content),
        box=box_style,
        border_style="bold yellow",
        padding=(1, 2),
    )


def render_settings_menu(
    templates: dict,
    box_style: Box = box.HEAVY,
) -> Panel:
    """Render formatted menu of available world setting archetypes.

    Args:
        templates: Dictionary mapping setting keys to SettingTemplate objects.
        box_style: Rich Box character set.

    Returns:
        Formatted Table panel showing options [1] to [4].
    """
    table = Table(
        show_header=False,
        box=box.SIMPLE,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Key", style="bold #80FF80", width=6)
    table.add_column("Setting", style="bold yellow", width=22)
    table.add_column("Description", style="bright_green")

    f_desc = getattr(templates.get("fantasy"), "description", "A medieval kingdom where magic is fading.")
    s_desc = getattr(templates.get("scifi"), "description", "A drifting space station with failing life support.")
    c_desc = getattr(templates.get("cyberpunk"), "description", "A neon-drenched metropolis ruled by megacorps.")

    table.add_row("[1]", "Fantasy", f_desc)
    table.add_row("[2]", "Sci-Fi", s_desc)
    table.add_row("[3]", "Cyberpunk", c_desc)
    table.add_row("[4]", "Custom", "Create your own unique setting, tone, and opening premise.")

    return Panel(
        table,
        box=box_style,
        title="[bold yellow]AVAILABLE WORLD SETTINGS[/bold yellow]",
        border_style="bright_green",
        padding=(0, 1),
    )


def render_help_panel(
    box_style: Box = box.HEAVY,
) -> Panel:
    """Render in-game help menu and player command reference.

    Args:
        box_style: Rich Box character set.

    Returns:
        Formatted Help Panel.
    """
    table = Table(
        show_header=False,
        box=box.SIMPLE,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Input", style="bold #80FF80", width=18)
    table.add_column("Action", style="bright_green")

    table.add_row("[1] - [4]", "Select one of the numbered narrative choices.")
    table.add_row("Custom Text", "Type any custom free-form action (e.g. 'search the chest').")
    table.add_row("<Enter>", "Empty input defaults to [WAIT] (observe surroundings).")
    table.add_row("/status, /inv", "View detailed player context, full inventory, and memory.")
    table.add_row("[?], /debug", "Open live system diagnostics, token monitor, and state inspector.")
    table.add_row("/help", "Display this commands reference menu.")
    table.add_row("/restart", "Restart the adventure and character setup from scratch.")
    table.add_row("/quit, /exit", "Exit Gemma-Quest safely.")

    return Panel(
        table,
        box=box_style,
        title="[bold yellow]COMMANDS & CONTROLS[/bold yellow]",
        border_style="bold yellow",
        padding=(0, 1),
    )


def render_detailed_status(
    context: StoryContext,
    box_style: Box = box.HEAVY,
) -> Panel:
    """Render full detailed story context inspection panel.

    Args:
        context: Current authoritative StoryContext.
        box_style: Rich Box character set.

    Returns:
        Formatted Detailed Status Panel.
    """
    lines = [
        f"[bold yellow]Current Turn:[/] [bright_green]{context.turn_count}[/bright_green]",
        f"[bold yellow]Location:[/]     [white]{context.current_location}[/white]",
    ]
    if context.world_tone:
        lines.append(f"[bold yellow]World Tone:[/]   [dim grey]{context.world_tone}[/dim grey]")

    # Inventory
    lines.append("")
    lines.append("[bold yellow]Inventory:[/] ")
    if context.key_items_held:
        for item in context.key_items_held:
            lines.append(f"  • [white]{item}[/white]")
    else:
        lines.append("  [dim grey](Empty)[/dim grey]")

    # Active Characters
    lines.append("")
    lines.append("[bold yellow]Characters Met:[/] ")
    if context.active_characters:
        for name, status in context.active_characters.items():
            st_val = status.value if hasattr(status, "value") else str(status)
            color = "bright_green"
            if str(st_val).upper() == "DEAD":
                color = "bold red"
            elif str(st_val).upper() == "HOSTILE":
                color = "bold yellow"
            elif str(st_val).upper() == "ALLY":
                color = "bold cyan"
            lines.append(f"  • [white]{name}[/white] - [{color}]{st_val}[/{color}]")
    else:
        lines.append("  [dim grey](None encountered yet)[/dim grey]")

    # Memory Summary
    lines.append("")
    lines.append(f"[bold yellow]Recent Story Memory:[/] \n[dim grey]{context.story_summary}[/dim grey]")

    return Panel(
        "\n".join(lines),
        box=box_style,
        title="[bold yellow]DETAILED STORY CONTEXT[/bold yellow]",
        border_style="bright_green",
        padding=(1, 2),
    )


def render_error_panel(
    message: str,
    options: Optional[List[str]] = None,
    box_style: Box = box.HEAVY,
) -> Panel:
    """Render retro error or connection alert modal.

    Args:
        message: Error explanation message.
        options: List of available recovery actions (e.g. ['[R]etry', '[F]allback', '[Q]uit']).
        box_style: Rich Box character set.

    Returns:
        Formatted Alert Panel.
    """
    lines = [
        f"[bold red]! SYSTEM ALERT ![/bold red]",
        f"[white]{message}[/white]",
    ]
    if options:
        lines.append("")
        lines.append("[bold yellow]Options:[/] " + "  ".join(f"[bold]{opt}[/bold]" for opt in options))

    return Panel(
        "\n".join(lines),
        box=box_style,
        border_style="bold red",
        padding=(1, 2),
    )


def render_debug_overlay(
    context: StoryContext,
    token_monitor: Optional[Any] = None,
    box_style: Box = box.HEAVY,
) -> Panel:
    """Render comprehensive live diagnostics, StoryContext inspection, and token monitor overlay.

    Args:
        context: Authoritative current StoryContext.
        token_monitor: Optional TokenMonitor instance providing usage and latency statistics.
        box_style: Rich Box character set.

    Returns:
        Formatted Diagnostics Overlay Panel.
    """
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="bold yellow", width=18)
    table.add_column(style="bright_green")

    # 1. State Inspection
    table.add_row("[LOCATION]", f"[bold yellow]{context.current_location}[/bold yellow]")
    table.add_row("[WORLD TONE]", f"[white]{context.world_tone or 'Default'}[/white]")
    table.add_row("[TURN COUNT]", f"[bright_green]{context.turn_count}[/bright_green]")

    # Items
    items_str = ", ".join(context.key_items_held) if context.key_items_held else "[dim grey]None[/dim grey]"
    table.add_row("[KEY ITEMS]", f"[white]{items_str}[/white]")

    # Characters
    if context.active_characters:
        chars_list = []
        for name, status in context.active_characters.items():
            st_val = status.value if hasattr(status, "value") else str(status)
            color = "bright_green"
            if str(st_val).upper() == "DEAD":
                color = "bold red"
            elif str(st_val).upper() == "HOSTILE":
                color = "bold yellow"
            elif str(st_val).upper() == "ALLY":
                color = "bold cyan"
            chars_list.append(f"{name}: [{color}]{st_val}[/{color}]")
        chars_str = ", ".join(chars_list)
    else:
        chars_str = "[dim grey]None[/dim grey]"
    table.add_row("[CHARACTERS]", chars_str)

    # Rolling Summary
    table.add_row("[MEMORY SUMMARY]", f"[dim grey]{context.story_summary}[/dim grey]")

    # 2. Token Monitor & Metrics
    if token_monitor is not None:
        table.add_row("", "")
        table.add_row("[bold cyan]-- MONITOR --[/bold cyan]", "[bold cyan]----------------------------------------[/bold cyan]")

        last_pt = getattr(token_monitor, "last_prompt_tokens", 0)
        limit = getattr(token_monitor, "context_limit", 4096)
        pct = getattr(token_monitor, "context_utilization_percent", 0.0)
        status_color = "bright_green" if pct < 80.0 else "bold red"

        table.add_row(
            "[CONTEXT USAGE]",
            f"[{status_color}]{last_pt}[/{status_color}] / [white]{limit} tokens[/white] ([{status_color}]{pct:.1f}%[/{status_color}])",
        )

        last_ct = getattr(token_monitor, "last_completion_tokens", 0)
        last_lat = getattr(token_monitor, "last_latency", 0.0)
        table.add_row(
            "[LAST INFERENCE]",
            f"[white]{last_ct} completion tokens[/white] in [bright_green]{last_lat:.2f}s[/bright_green]",
        )

        cum_tot = getattr(token_monitor, "cumulative_total_tokens", 0)
        avg_lat = getattr(token_monitor, "average_latency", 0.0)
        table.add_row(
            "[CUMULATIVE]",
            f"[white]{cum_tot} total tokens[/white] | Avg Latency: [bright_green]{avg_lat:.2f}s/turn[/bright_green]",
        )

    return Panel(
        table,
        box=box_style,
        title="[bold yellow]SYSTEM DIAGNOSTICS & LIVE STATE INSPECTOR[/bold yellow]",
        subtitle="[dim grey]Session log active at debug_log.txt • Press Enter to resume[/dim grey]",
        border_style="bold cyan",
        padding=(1, 2),
    )



