"""Interactive demonstration and diagnostics for the Retro DOS TUI."""

from src.state.models import CharacterStatus, StoryContext
from src.ui import RetroTUI

def main():
    """Run interactive demonstration of RetroTUI layout and components."""
    tui = RetroTUI()
    tui.console.print("[tui.header]════════ GEMMA-QUEST RETRO TUI DIAGNOSTICS ════════[/tui.header]\n")

    # 1. Title Splash
    tui.render_title()
    tui.console.print("\n")

    # 2. Complete 3-Zone Game Frame
    context = StoryContext(
        current_location="The Sunken Catacombs",
        world_tone="Gothic Horror",
        active_characters={
            "Skeleton Warlord": CharacterStatus.HOSTILE,
            "Lady Vanya": CharacterStatus.ALLY,
            "Fallen Paladin": CharacterStatus.DEAD,
        },
        key_items_held=["Sunstone Amulet", "Rusted Key"],
        story_summary="You broke through the subterranean gates and entered the waterlogged catacombs.",
        turn_count=7,
    )

    narration = (
        "Green phosphorescence illuminates the dripping stone pillars of the catacombs. "
        "Before you stands the Skeleton Warlord, blade raised, while Lady Vanya prepares an incantation. "
        "The skeletal remains of the Fallen Paladin lie motionless near the altar."
    )

    choices = [
        "Brandish the Sunstone Amulet against the Warlord",
        "Signal Lady Vanya to unleash her spell",
        "Search the Fallen Paladin's armor for supplies",
    ]

    tui.render_game_frame(
        context=context,
        narration=narration,
        choices=choices,
        provider_status="[LOCAL: gemma4:e2b]",
    )

    # 3. Game Over Screen Demo
    tui.console.print("\n")
    tui.render_game_over(context)


if __name__ == "__main__":
    main()
