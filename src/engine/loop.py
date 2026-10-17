"""Main async game loop and turn orchestration engine for Gemma-Quest.

Connects the TUI renderer, inference engine, state synchronization,
non-blocking asyncio task management, input processing, and timeout resilience.
"""

import asyncio
import logging
from typing import Callable, List, Optional, Tuple
from rich import box

from src.config import GameConfig, load_config
from src.engine.base import InferenceError, InferenceProvider
from src.engine.parser import ResponseParser
from src.engine.prompt_builder import PromptBuilder
from src.engine.summary_refiner import SummaryRefiner
from src.engine.validator import NarrativeValidator
from src.state.models import CharacterStatus, GameResponse, StoryContext
from src.state.sync import sync_state
from src.ui.input_engine import ActionPayload, InputEngine, InputType
from src.ui.layout import (
    render_detailed_status,
    render_error_panel,
    render_help_panel,
)
from src.ui.renderer import RetroTUI

logger = logging.getLogger(__name__)

SPINNER_FLAVOR_TEXTS = [
    "⏳ The dungeon whispers in the dark...",
    "⏳ Consulting the ancient oracle...",
    "⏳ Shadows shift across the stone walls...",
    "⏳ Gemma weaves the tapestry of fate...",
    "⏳ Deciphering the runes of destiny...",
    "⏳ The torches flicker as unknown footsteps approach...",
]


def create_turn_fallback_response(context: StoryContext, action: str) -> GameResponse:
    """Generate a coherent algorithmic fallback turn if the AI engine fails or times out.

    Ensures gameplay is never abruptly interrupted by connection loss or model exhaustion.

    Args:
        context: Current authoritative StoryContext.
        action: The player's attempted action.

    Returns:
        Structured GameResponse continuing the adventure gracefully.
    """
    clean_action = action.strip().rstrip(".")
    narration = (
        f"You attempt to {clean_action.lower()}. "
        f"The cold stone of {context.current_location} echoes with your footsteps. "
        f"Though the path ahead remains treacherous, your resolve remains unbroken."
    )

    choices = [
        "Search the immediate area for hidden threats or clues",
        "Advance cautiously into the next chamber",
        "Check your inventory and take a defensive stance",
        "Listen closely to the darkness ahead",
    ]

    action_summary = f"{context.story_summary} Action taken: {clean_action}."
    if len(action_summary) > 400:
        action_summary = action_summary[-350:]

    return GameResponse(
        narration=narration,
        choices=choices,
        updated_story_context=StoryContext(
            current_location=context.current_location,
            world_tone=context.world_tone,
            active_characters=context.active_characters,
            key_items_held=context.key_items_held,
            story_summary=action_summary,
            turn_count=context.turn_count,
        ),
    )


class GameEngine:
    """Async Game Loop controller managing turn execution, timeouts, and state updates."""

    def __init__(
        self,
        config: Optional[GameConfig] = None,
        provider: Optional[InferenceProvider] = None,
        tui: Optional[RetroTUI] = None,
        input_engine: Optional[InputEngine] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        parser: Optional[ResponseParser] = None,
        validator: Optional[NarrativeValidator] = None,
        summary_refiner: Optional[SummaryRefiner] = None,
        timeout_seconds: float = 30.0,
        player_name: str = "Hero",
    ):
        """Initialize the game engine with required pipeline components.

        Args:
            config: Full game configuration.
            provider: Active InferenceProvider (Local Ollama, Cloud, or Hybrid).
            tui: RetroTUI renderer.
            input_engine: Input capture and validator.
            prompt_builder: Context-first prompt builder.
            parser: JSON response extractor and retry coordinator.
            validator: Narrative consistency and guardian validator.
            summary_refiner: Optional 2nd-pass summary condenser.
            timeout_seconds: Timeout threshold for model inference in seconds.
            player_name: Character name for status and game-over checks.
        """
        self.config = config or load_config()
        self.provider = provider
        self.tui = tui or RetroTUI()
        self.input_engine = input_engine or InputEngine()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.parser = parser or ResponseParser()
        self.validator = validator or NarrativeValidator()
        context_cfg = getattr(self.config, "context", None)
        self.summary_refiner = summary_refiner or SummaryRefiner(config=context_cfg)
        self.timeout_seconds = timeout_seconds
        self.player_name = player_name

        self.context: Optional[StoryContext] = None
        self.current_response: Optional[GameResponse] = None
        self.is_running: bool = False

    def _get_spinner_text(self, turn: int) -> str:
        """Select a flavor string for the inference loading spinner."""
        idx = turn % len(SPINNER_FLAVOR_TEXTS)
        return SPINNER_FLAVOR_TEXTS[idx]

    async def execute_inference(
        self,
        messages: List[dict],
        turn: int,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Execute non-blocking model inference with a status spinner and timeout.

        Args:
            messages: List of chat message dictionaries.
            turn: Current turn number for spinner flavor selection.

        Returns:
            Tuple of (raw_output, error_reason). One will be None.
        """
        spinner_flavor = self._get_spinner_text(turn)
        spinner_type = "dots" if self.tui.capabilities.supports_unicode else "line"

        raw_output: Optional[str] = None
        error_reason: Optional[str] = None

        with self.tui.console.status(
            f"[bold yellow]{spinner_flavor}[/bold yellow]",
            spinner=spinner_type,
            spinner_style="bold #80FF80",
        ):
            try:
                raw_output = await asyncio.wait_for(
                    asyncio.to_thread(self.provider.chat, messages),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_reason = f"Model inference timed out (> {self.timeout_seconds:.0f}s)"
                logger.warning(error_reason)
            except InferenceError as exc:
                error_reason = f"AI Provider error: {str(exc)}"
                logger.error(error_reason)
            except Exception as exc:
                error_reason = f"Unexpected inference exception: {str(exc)}"
                logger.exception(error_reason)

        return raw_output, error_reason

    def handle_inference_recovery(
        self,
        error_reason: str,
        current_action: str,
    ) -> Tuple[str, Optional[GameResponse]]:
        """Render error modal and capture player's chosen recovery action.

        Args:
            error_reason: Explanation of the failure.
            current_action: The action player attempted.

        Returns:
            Tuple of (decision, optional_fallback_response) where decision is 'retry', 'fallback', or 'quit'.
        """
        self.tui.console.print()
        error_panel = render_error_panel(
            message=error_reason,
            options=["[R]etry", "[F]allback Scene", "[Q]uit"],
            box_style=self.tui.box_style,
        )
        self.tui.print_panel(error_panel)

        while True:
            try:
                choice = self.input_engine._input_func("\nSelect recovery option [R/F/Q] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return "quit", None

            if choice in ("r", "retry"):
                return "retry", None
            elif choice in ("f", "fallback"):
                fallback = create_turn_fallback_response(self.context, current_action)
                return "fallback", fallback
            elif choice in ("q", "quit", "exit"):
                return "quit", None
            else:
                self.tui.console.print("[bold red]Invalid option. Enter 'R' to retry, 'F' for fallback, or 'Q' to quit.[/bold red]")

    async def execute_turn(self, player_action: str) -> Optional[GameResponse]:
        """Execute a complete single turn pipeline.

        Builds prompt -> Non-blocking inference -> Parse JSON -> Validate -> Sync State -> Refine Summary.

        Args:
            player_action: Action string selected or typed by the player.

        Returns:
            Validated GameResponse, or None if the player elected to quit during recovery.
        """
        assert self.context is not None, "StoryContext must be initialized before executing turns."

        # 1. Build messages
        messages = self.prompt_builder.build_chat_messages(self.context, player_action)

        # 2. Inference Loop with retry & fallback handling
        while True:
            raw_output, error_reason = await self.execute_inference(messages, self.context.turn_count)

            if error_reason is None and raw_output:
                # Successfully received model output, process response
                try:
                    parsed_response = self.parser.parse_with_retry(
                        raw_text=raw_output,
                        local_context=self.context,
                        provider=self.provider,
                    )
                    break
                except Exception as exc:
                    logger.warning(f"Response parsing failed: {exc}")
                    error_reason = f"Malformed model response: {exc}"

            # If we reached here, inference or parsing failed
            decision, fallback_resp = self.handle_inference_recovery(error_reason or "Unknown error", player_action)
            if decision == "retry":
                continue
            elif decision == "fallback":
                parsed_response = fallback_resp
                break
            else:
                # Player chose quit
                return None

        # 3. Guardian Narrative Validation
        validated_response = self.validator.validate_response(parsed_response, self.context)

        # 4. Sync Narrative State
        self.context = sync_state(self.context, validated_response)

        # 5. Optional Summary Refinement (2nd-pass)
        if self.summary_refiner.should_refine(self.context):
            try:
                refined = await asyncio.to_thread(
                    self.summary_refiner.refine_summary,
                    self.provider,
                    player_action,
                    validated_response.narration,
                    self.context.story_summary,
                )
                if refined and refined != self.context.story_summary:
                    self.context.story_summary = refined
            except Exception as exc:
                logger.debug(f"Summary refinement non-fatal error: {exc}")

        self.current_response = validated_response
        return validated_response

    async def run(
        self,
        initial_context: StoryContext,
        initial_response: GameResponse,
    ) -> str:
        """Run the main interactive async game loop until exit or game over.

        Args:
            initial_context: Turn 0 / Seeded StoryContext.
            initial_response: Turn 1 opening scene GameResponse.

        Returns:
            Exit code string ('quit', 'restart', or 'game_over').
        """
        self.context = initial_context
        self.current_response = initial_response
        self.is_running = True

        while self.is_running:
            # Check for Game Over condition (player death)
            hero_status = self.context.active_characters.get(self.player_name)
            if hero_status == CharacterStatus.DEAD:
                self.tui.render_game_over(self.context)
                try:
                    choice = self.input_engine._input_func("\n[R]estart or [Q]uit? > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    return "quit"

                if choice.startswith("r"):
                    return "restart"
                return "game_over"

            # Render current 3-zone game frame
            provider_badge = (
                self.provider.get_status()
                if hasattr(self.provider, "get_status")
                else "[AI: Ready]"
            )
            self.tui.render_game_frame(
                context=self.context,
                narration=self.current_response.narration,
                choices=self.current_response.choices,
                provider_status=provider_badge,
            )

            # Capture player action or command
            payload: ActionPayload = await self.input_engine.get_action_async(
                current_choices=self.current_response.choices,
                error_callback=lambda msg: self.tui.console.print(f"[bold red]{msg}[/bold red]"),
            )

            # Handle Commands
            if payload.input_type == InputType.COMMAND:
                if payload.command_name == "help":
                    self.tui.print_panel(render_help_panel(box_style=self.tui.box_style))
                    self.input_engine._input_func("\nPress Enter to return to the game...")
                    continue
                elif payload.command_name == "status":
                    self.tui.print_panel(render_detailed_status(self.context, box_style=self.tui.box_style))
                    self.input_engine._input_func("\nPress Enter to return to the game...")
                    continue
                elif payload.command_name == "restart":
                    return "restart"
                elif payload.command_name == "quit":
                    return "quit"

            # Execute Turn Action
            result = await self.execute_turn(payload.action_text)
            if result is None:
                # Player quit during error recovery
                return "quit"

        return "quit"
