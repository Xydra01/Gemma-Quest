"""Core narrative state data models and enums for Gemma-Quest."""

from enum import Enum
import time
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class CharacterStatus(str, Enum):
    """Standardized states for NPCs to enforce narrative consistency."""

    ALIVE = "Alive"
    DEAD = "Dead"
    HOSTILE = "Hostile"
    ALLY = "Ally"
    NEUTRAL = "Neutral"
    UNKNOWN = "Unknown"


class StoryContext(BaseModel):
    """The core narrative state of the game world.

    Serialized and passed to Gemma 4 every turn to maintain immersion and consistency.
    """

    current_location: str = Field(
        default="The Rusty Gate",
        description="The specific physical location (noun phrase).",
    )
    world_tone: str = Field(
        default="Dark Fantasy",
        description="High-level genre or tone modifier steering narrative voice.",
    )
    active_characters: Dict[str, CharacterStatus] = Field(
        default_factory=dict,
        description="Map of NPC Name to CharacterStatus.",
    )
    key_items_held: List[str] = Field(
        default_factory=list,
        description="Plot-relevant inventory items held by the player.",
    )
    story_summary: str = Field(
        default="You are waking up in a dark cell.",
        description="3-4 sentence rolling memory summary preventing context bloat.",
    )
    turn_count: int = Field(
        default=0,
        ge=0,
        description="Current game turn count.",
    )


class PlayerSetup(BaseModel):
    """Player onboarding choices captured during initialization."""

    player_name: str = Field(
        default="Hero",
        description="Sanitized player character name.",
    )
    setting_type: Literal["fantasy", "scifi", "cyberpunk", "custom"] = Field(
        default="fantasy",
        description="Selected world setting archetype.",
    )
    custom_setting_details: Optional[str] = Field(
        default=None,
        description="Custom setting text if setting_type is custom.",
    )
    initial_goal: str = Field(
        default="",
        description="The player's primary motivation or initial goal.",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Creation timestamp of player setup.",
    )


class GameResponse(BaseModel):
    """Structured output expected from the LLM turn generation."""

    narration: str = Field(
        description="Descriptive story text shown to the player.",
    )
    choices: List[str] = Field(
        min_length=2,
        max_length=4,
        description="List of 2 to 4 distinct narrative choices available to the player.",
    )
    updated_story_context: StoryContext = Field(
        description="Proposed new state of the world following this turn's actions.",
    )


class GameStateEnvelope(BaseModel):
    """Persistence container packaging StoryContext with campaign metadata."""

    version: str = Field(
        default="1.0.0",
        description="Save file schema version.",
    )
    saved_at: float = Field(
        default_factory=time.time,
        description="Unix timestamp when state was saved.",
    )
    player_setup: Optional[PlayerSetup] = Field(
        default=None,
        description="Player onboarding profile if available.",
    )
    story_context: StoryContext = Field(
        description="Current active StoryContext.",
    )
