# Technical Design Document: TUI Design, Aesthetics & Implementation

## 1. Overview
This document defines the visual language, layout structure, and interactive mechanics of the Terminal User Interface (TUI). The goal is to evoke the feel of a late-80s/early-90s DOS-based text adventure while leveraging modern terminal capabilities (color, Unicode) for readability. We will use the **Rich** library for rendering, as it provides robust support for complex layouts, colors, and styles that are difficult to achieve with raw ANSI codes.

## 2. Visual Style & Aesthetic Guidelines

### 2.1 Color Palette
We adhere to a strict "Retro Terminal" palette to maintain immersion and visual consistency.

| Element | Color Code (Hex) | Rich Style String | Usage |
| :--- | :--- | :--- | :--- |
| **Background** | `#000000` | `black` | Terminal background (default). |
| **Primary Text** | `#00FF00` | `bright_green` | Narration, standard UI text. |
| **Headers/Title** | `#FFFF00` | `bold yellow` | Main Title, Location Name. |
| **Choices/Interactive**| `#FFFFFF` or `#80FF80` | `white` / `light_green` | Numbered options, user input echo. |
| **Errors/Negative** | `#FF0000` | `bold red` | Errors, death events, failed parses. |
| **Dim/Secondary** | `#555555` | `dim grey` | System messages, timestamps, debug info. |

### 2.2 Typography & Borders
*   **Font:** The application assumes the user's terminal is set to a monospace font (Consolas, Courier New, VT323). We do not control the font, but we design for 80x25 character grids as the baseline.
*   **Borders:** Use Unicode Box-Drawing characters (`╔`, `╗`, `═`, `║`, `╚`, `╝`) rather than ASCII (`+`, `-`, `|`). These render more cleanly in modern terminals and provide a sharper "DOS" look.

## 3. Layout Architecture

The screen is divided into three horizontal zones:
1.  **Header Bar:** Fixed at the top. Contains the Game Title and current Turn Count.
2.  **Status Panel (Left/Top):** A compact summary of `Location`, `Active Characters`, and `Key Items`. This is static until a state update occurs.
3.  **Narrative & Choices (Center):** The main content area. Displays the LLM-generated narration and the list of choices.

### 3.1 Layout Diagram (ASCII Representation)

```text
╔══════════════════════════════════════════════════╗
║  GEMMA DUNGEON v1.0  │  Turn: 5                ║
╠══════════════════════════════════════════════════╣
║  [LOCATION] The Rusty Gate                       ║
║  [CHARACTERS] Goblin King (Hostile)              ║
║  [ITEMS] Map to Castle                           ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  You stand before a massive iron door.           ║
║  The lock is rusted, but the hinges creak with   ║
║  every gust of wind.                             ║
║                                                  ║
║  ┌──────────────────────────────┐                ║
║  │ [1] Pick the lock            │                ║
║  │ [2] Search for another way   │                ║
║  │ [3] Type your own action...  │                ║
║  └──────────────────────────────┘                ║
║                                                  ║
╚══════════════════════════════════════════════════╝
```

## 4. Interactive Elements & Input Handling

### 4.1 Choice Selection
*   **Mechanism:** The user types a number corresponding to the choice.
*   **Validation:** If the input is not a valid number (within range), display a subtle error: `[ERROR] Invalid choice.` and re-prompt.
*   **Free Text:** Option `[0]` or simply typing any non-numeric text triggers "Free Form" mode. The entire typed string is passed to the LLM as `player_input`.

### 4.2 Screen Clearing & Transitions
*   **Major Scene Changes:** When `current_location` changes significantly (e.g., entering a new room), trigger a soft screen clear or a "fade" effect (printing `\n` multiple times) before rendering the new scene.
*   **Death/End Game:** On death, flush the screen to black, print `[GAME OVER]` in red, and allow `q` to quit.

## 5. Implementation Details (Python + Rich)

### 5.1 Dependencies
*   `rich`: For console output, panels, tables, and styling.
*   `ollama-python`: For LLM interaction.
*   `pydantic`: For state models.

### 5.2 Core Class Structure

```python
import ollama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.box import HEAVY # Unicode box-drawing
from rich.group import Group
import json
from typing import List, Dict
from state.models import StoryContext

class RetroGameUI:
    def __init__(self):
        self.console = Console()
        self.theme = {
            "primary": "bright_green",
            "header": "bold yellow",
            "error": "bold red",
            "choice": "white"
        }

    def render_header(self, turn_count: int) -> str:
        return f"[{self.theme['header']}]GEMMA DUNGEON v1.0[/] │ Turn: {turn_count}"

    def render_status_panel(self, context: StoryContext) -> Panel:
        # Create a table for status info
        table = Table(show_header=False, box=HEAVY, title="[Location]", expand=True)
        table.add_column("Status", style="bold yellow")
        table.add_column("Value", style="bright_green")
        
        # Populate rows
        table.add_row("[Location]", context.current_location)
        
        chars = [f"{k} ({v.value})" for k, v in context.active_characters.items()]
        table.add_row("[Characters]", ", ".join(chars) if chars else "None")
        
        items = context.key_items_held
        table.add_row("[Items]", ", ".join(items) if items else "None")
        
        # Wrap in a panel for the main layout
        return Panel(
            Group(table),
            title="[bold]GEMMA DUNGEON[/bold]",
            border_style="bright_green",
            subtitle=f"[italic]Summary: {context.story_summary}[/italic]"
        )

    def render_narrative_and_choices(self, narration: str, choices: List[str]) -> Panel:
        # Render narrative with line wrapping
        narrative_panel = Panel(
            narration,
            title="[bold]Narration[/bold]",
            border_style="bright_green",
            height=5 # Fixed height to prevent layout jumps
        )
        
        # Render choices
        choice_lines = []
        for i, choice in enumerate(choices, 1):
            choice_lines.append(f"[{self.theme['choice']}][{i}]{self.theme['choice']}: {choice}")
        choice_panel = Panel(
            "\n".join(choice_lines),
            title="[bold]Actions[/bold]",
            border_style="yellow",
            height=3
        )
        
        return Group(narrative_panel, choice_panel)

    def get_user_input(self, choices: List[str]) -> str:
        # Display input prompt
        self.console.print("\n[white]> Enter choice number or type action:[/white] ", end="")
        user_input = input().strip()
        
        if not user_input:
            return "WAIT"

        try:
            idx = int(user_input)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
            else:
                self.console.print("[red]Invalid choice number.[/red]")
                return "RETRY_INPUT"
        except ValueError:
            # Free form input
            return user_input

    def run(self, context: StoryContext) -> None:
        while True:
            turn_count = context.turn_count

            # 1. Clear Screen (Optional for smooth flow, use scroll instead if preferred)
            # self.console.clear() 
            self.console.print("\n" * 5) # Soft clear

            # 2. Render Status
            status_panel = self.render_status_panel(context)
            self.console.print(status_panel)

            # 3. Get Previous Narration (Stored in context or local var)
            # For this example, we assume 'last_narration' is available

            # 4. Get Input
            choices = ["Pick the lock", "Search for another way"] # Placeholder
            user_input = self.get_user_input(choices)
            
            if user_input == "RETRY_INPUT":
                continue

            # 5. Call LLM (Simulated here)
            response_data = call_llm(context, user_input)

            # 6. Update Context & Render New State
            new_context = update_state_from_response(response_data, context)
            self.render_narrative_and_choices(response_data['narration'], response_data['choices'])
            
            context = new_context

            if context.is_game_over:
                break
```

## 6. Performance & UX Optimizations

1.  **Non-Blocking Input:** In a production environment, use `rich.live` to keep the UI responsive while waiting for LLM tokens. If using Ollama streaming, yield tokens and update the "Narration" panel in real-time (typewriter effect).
2.  **Scrolling vs. Clearing:** Instead of clearing the screen every turn (which causes flickering), append new narration to a scrollable log. Keep the "Status Panel" fixed at the top using terminal cursor positioning or `rich`'s layout capabilities.
3.  **Error Handling in UI:** If the LLM times out, display a spinning ASCII loader:
    ```text
    ⏳ Thinking... [Spinner]
    ```
4.  **Accessibility:** Ensure color contrast is high. Provide a "Monochrome Mode" flag in `config.yaml` that strips all colors for users with visual impairments or older terminals.

## 7. Edge Cases & UI Failures

| Scenario | UI Behavior |
| :--- | :--- |
| **LLM Returns No Choices** | If `choices` is empty, default to: `["Look around", "Wait"]`. Display a dim warning: `[WARN] AI could not generate choices.` |
| **Narration Too Long** | Truncate narration at 200 characters in the UI if it exceeds panel height. Add a `[...]` indicator. Full text available in debug log. |
| **Terminal Resize** | Rich handles reflowing tables automatically. Test on narrow terminals (80 cols) to ensure borders don't break. |
| **User Presses Ctrl+C** | Gracefully catch `KeyboardInterrupt`. Print "Saving game..." (if enabled) and exit with code 0. |
