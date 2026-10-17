"""Persistence layer: Atomic JSON save and load mechanics for Gemma-Quest."""

import json
import os
from pathlib import Path
from typing import Optional, Tuple, Union
from pydantic import ValidationError

from .models import GameStateEnvelope, PlayerSetup, StoryContext


class StatePersistenceError(Exception):
    """Base exception for game state persistence errors."""
    pass


class SaveFileNotFoundError(StatePersistenceError):
    """Raised when the specified save game file does not exist."""
    pass


class SaveFileCorruptedError(StatePersistenceError):
    """Raised when save file data cannot be parsed or validated."""
    pass


def save_game(
    context: StoryContext,
    file_path: Union[str, Path] = "savegame.json",
    setup: Optional[PlayerSetup] = None,
) -> None:
    """Atomically saves the game state to a JSON file.

    Uses a temporary file and atomic file replacement to prevent save game corruption
    in the event of process interruptions or power loss.

    Args:
        context: Current active StoryContext.
        file_path: Destination path for the save file. Defaults to 'savegame.json'.
        setup: Optional PlayerSetup profile associated with this campaign.

    Raises:
        StatePersistenceError: If file writing or renaming fails.
    """
    dest_path = Path(file_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    envelope = GameStateEnvelope(
        player_setup=setup,
        story_context=context,
    )

    data_json = envelope.model_dump_json(indent=2)
    temp_path = dest_path.with_suffix(f"{dest_path.suffix}.tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(data_json)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, dest_path)
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise StatePersistenceError(f"Failed to atomically save game state: {e}") from e


def load_game(
    file_path: Union[str, Path] = "savegame.json",
) -> Tuple[StoryContext, Optional[PlayerSetup]]:
    """Loads and validates game state from a JSON save file.

    Supports both full GameStateEnvelope structures and bare StoryContext exports.

    Args:
        file_path: Path to the save file. Defaults to 'savegame.json'.

    Returns:
        A tuple of (StoryContext, Optional[PlayerSetup]).

    Raises:
        SaveFileNotFoundError: If the save file does not exist.
        SaveFileCorruptedError: If JSON parsing or Pydantic validation fails.
    """
    path = Path(file_path)
    if not path.is_file():
        raise SaveFileNotFoundError(f"Save file not found at: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        raise SaveFileCorruptedError(f"Save file contains invalid JSON: {e}") from e
    except Exception as e:
        raise SaveFileCorruptedError(f"Could not read save file: {e}") from e

    if not isinstance(raw_data, dict):
        raise SaveFileCorruptedError("Save file root must be a JSON object.")

    # Try loading as GameStateEnvelope first
    try:
        envelope = GameStateEnvelope.model_validate(raw_data)
        return envelope.story_context, envelope.player_setup
    except ValidationError:
        pass

    # Fallback: Try loading directly as StoryContext (bare export)
    try:
        context = StoryContext.model_validate(raw_data)
        return context, None
    except ValidationError as e:
        raise SaveFileCorruptedError(f"Save file does not match expected schema: {e}") from e
