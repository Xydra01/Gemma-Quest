"""State management: Data models, delta sync, and persistence."""

from .models import (
    CharacterStatus,
    GameResponse,
    GameStateEnvelope,
    PlayerSetup,
    StoryContext,
)
from .persistence import (
    SaveFileCorruptedError,
    SaveFileNotFoundError,
    StatePersistenceError,
    load_game,
    save_game,
)
from .sync import (
    MAX_ITEM_NAME_LENGTH,
    MIN_SUMMARY_LENGTH,
    sync_state,
)

__all__ = [
    "CharacterStatus",
    "StoryContext",
    "PlayerSetup",
    "GameResponse",
    "GameStateEnvelope",
    "sync_state",
    "save_game",
    "load_game",
    "StatePersistenceError",
    "SaveFileNotFoundError",
    "SaveFileCorruptedError",
    "MAX_ITEM_NAME_LENGTH",
    "MIN_SUMMARY_LENGTH",
]
