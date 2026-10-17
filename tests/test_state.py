"""Unit tests and verification suite for narrative state, delta sync, and persistence."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.state import (
    CharacterStatus,
    GameResponse,
    PlayerSetup,
    SaveFileCorruptedError,
    SaveFileNotFoundError,
    StoryContext,
    load_game,
    save_game,
    sync_state,
)


def test_story_context_defaults():
    """Verify default values and types of StoryContext."""
    ctx = StoryContext()
    assert ctx.current_location == "The Rusty Gate"
    assert ctx.world_tone == "Dark Fantasy"
    assert ctx.active_characters == {}
    assert ctx.key_items_held == []
    assert "dark cell" in ctx.story_summary
    assert ctx.turn_count == 0


def test_game_response_choice_constraints():
    """Verify that GameResponse strictly enforces 2 to 4 choices."""
    valid_ctx = StoryContext()

    # Valid: 2 choices
    resp_2 = GameResponse(
        narration="You see a door.",
        choices=["Open it", "Turn back"],
        updated_story_context=valid_ctx,
    )
    assert len(resp_2.choices) == 2

    # Valid: 4 choices
    resp_4 = GameResponse(
        narration="You see a door.",
        choices=["1", "2", "3", "4"],
        updated_story_context=valid_ctx,
    )
    assert len(resp_4.choices) == 4

    # Invalid: 1 choice
    with pytest.raises(ValidationError):
        GameResponse(
            narration="You see a door.",
            choices=["Only one option"],
            updated_story_context=valid_ctx,
        )

    # Invalid: 5 choices
    with pytest.raises(ValidationError):
        GameResponse(
            narration="You see a door.",
            choices=["1", "2", "3", "4", "5"],
            updated_story_context=valid_ctx,
        )


def test_sync_state_basic_progression():
    """Verify normal state progression through sync_state."""
    local_ctx = StoryContext(
        current_location="Dungeon Cell",
        key_items_held=["Torch"],
        active_characters={"Rats": CharacterStatus.HOSTILE},
        story_summary="You woke up trapped in the dungeon cell.",
        turn_count=1,
    )

    ai_resp = GameResponse(
        narration="You slip past the rats and emerge into the armory.",
        choices=["Take sword", "Inspect rack"],
        updated_story_context=StoryContext(
            current_location="The Armory",
            key_items_held=["Torch", "Iron Key"],
            active_characters={"Rats": CharacterStatus.NEUTRAL},
            story_summary="You slipped past the rats into the armory.",
        ),
    )

    new_ctx = sync_state(local_ctx, ai_resp)

    assert new_ctx.current_location == "The Armory"
    assert new_ctx.key_items_held == ["Torch", "Iron Key"]
    assert new_ctx.active_characters["Rats"] == CharacterStatus.NEUTRAL
    assert new_ctx.story_summary == "You slipped past the rats into the armory."
    assert new_ctx.turn_count == 2


def test_sync_state_location_fallback():
    """Verify that empty location in AI response retains local location."""
    local_ctx = StoryContext(current_location="Throne Room", turn_count=5)

    ai_resp = GameResponse(
        narration="You wait patiently.",
        choices=["Look around", "Call out"],
        updated_story_context=StoryContext(
            current_location="   ",  # Blank/whitespace
            story_summary="You decided to wait in the throne room.",
        ),
    )

    new_ctx = sync_state(local_ctx, ai_resp)
    assert new_ctx.current_location == "Throne Room"
    assert new_ctx.turn_count == 6


def test_sync_state_dead_mans_switch():
    """Verify that DEAD characters cannot be resurrected by the model."""
    local_ctx = StoryContext(
        active_characters={
            "Goblin King": CharacterStatus.DEAD,
            "Forest Guide": CharacterStatus.ALLY,
        },
        turn_count=3,
    )

    # Model hallucination: attempts to make Goblin King ALIVE
    ai_resp = GameResponse(
        narration="The Goblin King speaks from the throne...",
        choices=["Attack him", "Flee"],
        updated_story_context=StoryContext(
            active_characters={
                "Goblin King": CharacterStatus.ALIVE,
                "Forest Guide": CharacterStatus.HOSTILE,
                "Cave Troll": CharacterStatus.HOSTILE,
            },
            story_summary="The battle rages on in the caverns.",
        ),
    )

    new_ctx = sync_state(local_ctx, ai_resp)

    # Dead Man's Switch: Goblin King remains DEAD
    assert new_ctx.active_characters["Goblin King"] == CharacterStatus.DEAD
    # Normal status updates to living characters are allowed
    assert new_ctx.active_characters["Forest Guide"] == CharacterStatus.HOSTILE
    # New characters are added
    assert new_ctx.active_characters["Cave Troll"] == CharacterStatus.HOSTILE


def test_sync_state_item_sanitization_and_protection():
    """Verify item length filtering, deduplication, and protected items retention."""
    local_ctx = StoryContext(
        key_items_held=["Rusty Dagger", "Royal Crest"],
        turn_count=2,
    )

    # Model proposes a very long hallucinated string and an empty string, omitting Royal Crest
    long_hallucination = "A" * 60
    ai_resp = GameResponse(
        narration="You find some strange herbs.",
        choices=["Eat herbs", "Store them"],
        updated_story_context=StoryContext(
            key_items_held=[
                "Rusty Dagger",
                "  Healing Herbs  ",
                "rusty dagger",  # Duplicate check
                long_hallucination,
                "   ",
            ],
            story_summary="You picked up healing herbs from the mossy floor.",
        ),
    )

    # Protect Royal Crest from being dropped
    new_ctx = sync_state(local_ctx, ai_resp, protected_items={"Royal Crest"})

    assert "Rusty Dagger" in new_ctx.key_items_held
    assert "Healing Herbs" in new_ctx.key_items_held
    assert "Royal Crest" in new_ctx.key_items_held
    assert long_hallucination not in new_ctx.key_items_held
    assert len(new_ctx.key_items_held) == 3


def test_sync_state_summary_fallback():
    """Verify that empty/short summary retains previous summary."""
    local_ctx = StoryContext(
        story_summary="You escaped the prison and reached the courtyard.",
        turn_count=4,
    )

    ai_resp = GameResponse(
        narration="The wind howls.",
        choices=["Seek shelter", "Keep going"],
        updated_story_context=StoryContext(
            story_summary="   ",  # Empty
        ),
    )

    new_ctx = sync_state(local_ctx, ai_resp)
    assert new_ctx.story_summary == "You escaped the prison and reached the courtyard."


def test_save_and_load_roundtrip(tmp_path: Path):
    """Verify atomic saving and loading of game state."""
    save_file = tmp_path / "test_save.json"

    setup = PlayerSetup(
        player_name="Aleron",
        setting_type="fantasy",
        initial_goal="Defeat the Shadow Drake",
    )

    ctx = StoryContext(
        current_location="Castle Gates",
        world_tone="Grimdark",
        active_characters={"Gatekeeper": CharacterStatus.HOSTILE},
        key_items_held=["Siege Map"],
        story_summary="Aleron arrived at the castle gates.",
        turn_count=7,
    )

    save_game(context=ctx, file_path=save_file, setup=setup)
    assert save_file.is_file()

    loaded_ctx, loaded_setup = load_game(save_file)

    assert loaded_ctx == ctx
    assert loaded_setup is not None
    assert loaded_setup.player_name == "Aleron"
    assert loaded_setup.setting_type == "fantasy"
    assert loaded_setup.initial_goal == "Defeat the Shadow Drake"


def test_load_bare_story_context(tmp_path: Path):
    """Verify that load_game can deserialize bare StoryContext JSON."""
    save_file = tmp_path / "bare_save.json"

    ctx = StoryContext(
        current_location="Forgotten Tomb",
        turn_count=12,
    )

    # Save as bare StoryContext JSON (without envelope)
    with open(save_file, "w", encoding="utf-8") as f:
        f.write(ctx.model_dump_json(indent=2))

    loaded_ctx, loaded_setup = load_game(save_file)
    assert loaded_ctx == ctx
    assert loaded_setup is None


def test_load_errors(tmp_path: Path):
    """Verify appropriate exceptions for missing or corrupted save files."""
    # File not found
    with pytest.raises(SaveFileNotFoundError):
        load_game(tmp_path / "non_existent.json")

    # Invalid JSON
    corrupted_json = tmp_path / "bad.json"
    with open(corrupted_json, "w", encoding="utf-8") as f:
        f.write("{invalid json content")

    with pytest.raises(SaveFileCorruptedError):
        load_game(corrupted_json)

    # Valid JSON but schema violation
    bad_schema = tmp_path / "bad_schema.json"
    with open(bad_schema, "w", encoding="utf-8") as f:
        f.write('{"turn_count": "not-an-integer", "current_location": 12345}')

    with pytest.raises(SaveFileCorruptedError):
        load_game(bad_schema)
