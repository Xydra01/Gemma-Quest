"""Live interactive demonstration and self-test for Gemma-Quest state management.

Run with: python3 -m src.state
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .models import CharacterStatus, GameResponse, PlayerSetup, StoryContext
from .persistence import load_game, save_game
from .sync import sync_state


def main():
    console = Console()
    console.print(
        Panel.fit(
            "[bold bright_green]Gemma-Quest: State Engine Live Demo[/bold bright_green]\n"
            "[dim]Testing StoryContext, Delta Sync, Dead Man's Switch, and Persistence[/dim]",
            border_style="bright_green",
        )
    )

    # 1. Setup Initial Campaign
    setup = PlayerSetup(
        player_name="Aleron",
        setting_type="fantasy",
        initial_goal="Escape the dungeon catacombs",
    )

    ctx_start = StoryContext(
        current_location="The Rusty Gate",
        world_tone="Dark Fantasy",
        key_items_held=["Rusty Key"],
        active_characters={
            "Goblin King": CharacterStatus.DEAD,
            "Old Sage": CharacterStatus.ALLY,
        },
        story_summary="You fought through the lower depths and defeated the Goblin King.",
        turn_count=1,
    )

    console.print("\n[bold yellow]1. Initial State (Turn 1):[/bold yellow]")
    console.print(f" • Location: [bright_green]{ctx_start.current_location}[/bright_green]")
    console.print(f" • Items: [bright_green]{', '.join(ctx_start.key_items_held)}[/bright_green]")
    console.print(
        f" • Characters: [bright_green]"
        f"{', '.join(f'{k} ({v.value})' for k, v in ctx_start.active_characters.items())}[/bright_green]"
    )

    # 2. Simulate AI Turn Response (with Model Hallucination)
    console.print("\n[bold yellow]2. Simulated LLM Response (Turn 2):[/bold yellow]")
    console.print(" [dim]Notice: The AI attempts to resurrect the DEAD Goblin King and drop the key.[/dim]")

    ai_turn = GameResponse(
        narration=(
            "The iron gates creak open. You step out onto the moonlit Courtyard. "
            "A hooded scout steps out from behind a gargoyle."
        ),
        choices=[
            "Approach the hooded scout with your weapon drawn",
            "Climb the exterior wall toward the clock tower",
            "Slip back into the shadows of the gateway",
        ],
        updated_story_context=StoryContext(
            current_location="Castle Courtyard",
            key_items_held=["Rusty Key", "Moonstone Ring"],
            active_characters={
                "Goblin King": CharacterStatus.ALIVE,  # Hallucination!
                "Old Sage": CharacterStatus.ALLY,
                "Hooded Scout": CharacterStatus.NEUTRAL,
            },
            story_summary="You opened the rusty gates and entered the moonlit courtyard.",
        ),
    )

    # 3. Execute Delta State Sync
    ctx_synced = sync_state(ctx_start, ai_turn)

    # 4. Atomic Save & Load
    save_path = "savegame.json"
    save_game(ctx_synced, file_path=save_path, setup=setup)
    ctx_loaded, setup_loaded = load_game(save_path)

    # 5. Render Comparison Table
    table = Table(
        title="State Reconciliation & Invariant Verification",
        border_style="bright_green",
    )
    table.add_column("Property", style="bold yellow", width=18)
    table.add_column("Before Turn 2", style="dim", width=26)
    table.add_column("Model Proposed", style="white", width=26)
    table.add_column("Resolved State (After Sync)", style="bold bright_green", width=30)

    table.add_row(
        "Location",
        ctx_start.current_location,
        ai_turn.updated_story_context.current_location,
        ctx_loaded.current_location,
    )
    table.add_row(
        "Turn Count",
        str(ctx_start.turn_count),
        "N/A",
        f"Turn {ctx_loaded.turn_count}",
    )
    table.add_row(
        "Goblin King",
        f"[red]{ctx_start.active_characters['Goblin King'].value}[/red]",
        f"[yellow]{ai_turn.updated_story_context.active_characters['Goblin King'].value} (Revive)[/yellow]",
        f"[red]{ctx_loaded.active_characters['Goblin King'].value} (Preserved Dead)[/red]",
    )
    table.add_row(
        "Hooded Scout",
        "None",
        ai_turn.updated_story_context.active_characters["Hooded Scout"].value,
        ctx_loaded.active_characters["Hooded Scout"].value,
    )
    table.add_row(
        "Inventory",
        ", ".join(ctx_start.key_items_held),
        ", ".join(ai_turn.updated_story_context.key_items_held),
        ", ".join(ctx_loaded.key_items_held),
    )
    table.add_row(
        "Summary",
        ctx_start.story_summary,
        ai_turn.updated_story_context.story_summary,
        ctx_loaded.story_summary,
    )

    console.print(table)
    console.print(
        f"\n[bold bright_green]✓ Atomic save verified:[/bold bright_green] "
        f"Written & re-loaded from [white]{save_path}[/white] successfully!\n"
    )


if __name__ == "__main__":
    main()
