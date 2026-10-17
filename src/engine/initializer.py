"""Context seeder and opening scene generator for Gemma-Quest."""

import logging
from typing import Dict, Optional, Tuple

from src.config import GameConfig, SettingTemplate, load_config, load_setting_templates
from src.engine.base import InferenceProvider
from src.engine.parser import ResponseParser
from src.engine.prompt_builder import PromptBuilder
from src.engine.validator import NarrativeValidator
from src.state.models import CharacterStatus, GameResponse, PlayerSetup, StoryContext

logger = logging.getLogger(__name__)


def create_initial_context(
    setup: PlayerSetup,
    templates: Optional[Dict[str, SettingTemplate]] = None,
) -> StoryContext:
    """Construct the initial seed StoryContext based on onboarding choices.

    Args:
        setup: Validated PlayerSetup object.
        templates: Optional map of setting archetype definitions.

    Returns:
        Initial StoryContext seeded with location, tone, and memory anchor.
    """
    all_templates = templates if templates is not None else load_setting_templates()
    template = all_templates.get(setup.setting_type)

    if template:
        starter_location = template.starter_location
        world_tone = template.name
    else:
        starter_location = "The Uncharted Threshold"
        world_tone = setup.custom_setting_details or "Custom Adventure"

    # Initial story summary anchors player identity, setting, and goal per Player_Setup_Init.md
    seed_summary = (
        f"{setup.player_name} has arrived at {starter_location}. "
        f"Goal: {setup.initial_goal}"
    )

    return StoryContext(
        current_location=starter_location,
        world_tone=world_tone,
        active_characters={},
        key_items_held=[],
        story_summary=seed_summary,
        turn_count=0,
    )


def create_starter_fallback_response(
    setup: PlayerSetup,
    context: StoryContext,
) -> GameResponse:
    """Generate a deterministic, atmospheric opening scene from templates if model is offline.

    Args:
        setup: PlayerSetup choices.
        context: Seed StoryContext.

    Returns:
        A complete, valid GameResponse providing immediate gameplay.
    """
    stype = setup.setting_type

    if stype == "fantasy":
        narration = (
            f"Cold drops of condensation fall from the arched stone ceiling. "
            f"{setup.player_name}, you awaken on the stone floor of {context.current_location}. "
            f"Ancient shadows lengthen around you as you recall your quest: {setup.initial_goal}."
        )
        choices = [
            "[1] Search the surroundings for a weapon or tool",
            "[2] Approach the iron portcullis ahead",
            "[3] Wait quietly and listen for footsteps",
        ]
    elif stype == "scifi":
        narration = (
            f"Red emergency strobe lights pulse silently across {context.current_location}. "
            f"{setup.player_name}, warning sirens hum through the bulkhead. "
            f"Your mission is critical: {setup.initial_goal}."
        )
        choices = [
            "[1] Access the nearby terminal console",
            "[2] Inspect the emergency supply locker",
            "[3] Check the manual airlock seals",
        ]
    elif stype == "cyberpunk":
        narration = (
            f"Acid rain hisses against the rusted metal grating above {context.current_location}. "
            f"Flickering holographic billboards illuminate the alleyway. "
            f"{setup.player_name}, your directive is clear: {setup.initial_goal}."
        )
        choices = [
            "[1] Check your cyberdeck connection",
            "[2] Step out into the crowded neon street",
            "[3] Seek cover beneath the overhang",
        ]
    else:
        narration = (
            f"You take in your strange surroundings in {context.current_location}. "
            f"{setup.player_name}, your journey begins here: {setup.initial_goal}."
        )
        choices = [
            "[1] Explore the immediate area",
            "[2] Check your inventory",
            "[3] Look for someone to speak with",
        ]

    # Create copy of initial context for turn 1
    updated_context = context.model_copy(deep=True)
    updated_context.turn_count = 1

    return GameResponse(
        narration=narration,
        choices=choices,
        updated_story_context=updated_context,
    )


class OpeningSceneGenerator:
    """Coordinates context seeding and opening scene generation for new campaigns."""

    def __init__(
        self,
        templates: Optional[Dict[str, SettingTemplate]] = None,
        config: Optional[GameConfig] = None,
    ):
        """Initialize the generator with templates and engine config."""
        self.templates = templates if templates is not None else load_setting_templates()
        self.config = config if config is not None else load_config()
        self.parser = ResponseParser(config=self.config.resilience)

    def generate(
        self,
        setup: PlayerSetup,
        provider: Optional[InferenceProvider] = None,
    ) -> Tuple[StoryContext, GameResponse]:
        """Seed context and generate the opening scene and initial choices.

        Dispatches the START opening messages to the provider, parses the response,
        and enforces consistency with NarrativeValidator. Falls back cleanly to a
        setting-specific starter scene if provider is unavailable or fails.

        Args:
            setup: PlayerSetup object from onboarding.
            provider: Active InferenceProvider (Local, Cloud, or Hybrid).

        Returns:
            Tuple of (initial_seed_context, validated_opening_response).
        """
        initial_context = create_initial_context(setup, self.templates)

        if provider is None or not provider.is_available():
            logger.info("Provider unavailable; using template starter opening scene.")
            return initial_context, create_starter_fallback_response(setup, initial_context)

        messages = PromptBuilder.build_opening_messages(setup)

        try:
            raw_output = provider.chat(
                messages,
                options={
                    "temperature": self.config.llm.temperature,
                    "top_p": self.config.llm.top_p,
                    "max_tokens": self.config.llm.max_tokens,
                },
            )

            game_response = self.parser.parse_with_retry(
                raw_text=raw_output,
                local_context=initial_context,
                provider=provider,
            )

            validated_response = NarrativeValidator.validate_response(
                response=game_response,
                context=initial_context,
            )

            # Ensure opening scene advances turn count to 1
            if validated_response.updated_story_context.turn_count == 0:
                validated_response.updated_story_context.turn_count = 1

            return initial_context, validated_response

        except Exception as exc:
            logger.warning("Opening scene generation failed (%s); using starter fallback.", exc)
            return initial_context, create_starter_fallback_response(setup, initial_context)
