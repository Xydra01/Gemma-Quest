"""Main async game loop and turn orchestration engine for Gemma-Quest.

Connects the TUI renderer, inference engine, state synchronization,
non-blocking asyncio task management, input processing, and timeout resilience.
"""

import asyncio
import logging
import time
from typing import Callable, List, Optional, Tuple
from rich import box

from src.config import GameConfig, load_config
from src.engine.base import InferenceError, InferenceProvider
from src.engine.diagnostics import SessionLogger
from src.engine.parser import ResponseParser
from src.engine.prompt_builder import PromptBuilder
from src.engine.summary_refiner import SummaryRefiner
from src.engine.token_monitor import (
    ContextGuard,
    TokenMonitor,
    estimate_messages_tokens,
    estimate_tokens,
)
from src.engine.validator import NarrativeValidator
from src.engine.visualizer import SceneVisualizer
from src.state.models import CharacterStatus, GameResponse, StoryContext
from src.state.sync import sync_state
from src.ui.input_engine import ActionPayload, InputEngine, InputType
from src.ui.layout import (
    render_debug_overlay,
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
        session_logger: Optional[SessionLogger] = None,
        token_monitor: Optional[TokenMonitor] = None,
        context_guard: Optional[ContextGuard] = None,
        visualizer: Optional[SceneVisualizer] = None,
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
            session_logger: Optional SessionLogger for recording debug_log.txt.
            token_monitor: Optional TokenMonitor for tracking token metrics.
            context_guard: Optional ContextGuard for memory and NPC protection.
            visualizer: Optional SceneVisualizer for ANSI/ASCII scene illustration.
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

        self.session_logger = session_logger or SessionLogger()
        self.token_monitor = token_monitor or TokenMonitor(
            context_limit=4096
        )
        self.context_guard = context_guard or ContextGuard(
            config=context_cfg,
            context_limit=4096,
        )
        self.visualizer = visualizer or SceneVisualizer(self.config)

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
                if not raw_output or not raw_output.strip():
                    error_reason = "Model returned empty or whitespace response."
                    logger.warning(error_reason)
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

        Prunes stale NPCs -> Estimates tokens & guards context -> Dispatches prompt ->
        Records latency & tokens -> Parses JSON -> Validates -> Syncs state -> Logs deltas.

        Args:
            player_action: Action string selected or typed by the player.

        Returns:
            Validated GameResponse, or None if the player elected to quit during recovery.
        """
        assert self.context is not None, "StoryContext must be initialized before executing turns."
        turn_num = self.context.turn_count + 1
        self.session_logger.log_turn_start(turn_num, player_action)

        # 1. Prune characters (>10 turns dead or inactive)
        retained_chars, pruned_names = self.context_guard.prune_characters(
            self.context.active_characters,
            current_turn=self.context.turn_count,
            player_name=self.player_name,
        )
        if pruned_names:
            self.session_logger.log_guard_event(
                "CHARACTER_PRUNING",
                f"Pruned inactive/dead NPCs: {', '.join(pruned_names)}",
            )
            self.context.active_characters = retained_chars

        # 2. Build messages and estimate token usage
        messages = self.prompt_builder.build_chat_messages(self.context, player_action)
        est_prompt_tokens = estimate_messages_tokens(messages)

        # 3. Context Guard overflow check (> 80% of 4096 = 3276 tokens)
        if self.context_guard.is_approaching_limit(est_prompt_tokens):
            compressed = self.context_guard.compress_summary(self.context.story_summary)
            self.session_logger.log_guard_event(
                "CONTEXT_WINDOW_WARNING",
                f"Prompt estimated at {est_prompt_tokens} tokens (> {self.context_guard.warning_token_threshold}). Compressing story summary.",
            )
            self.context.story_summary = compressed
            messages = self.prompt_builder.build_chat_messages(self.context, player_action)
            est_prompt_tokens = estimate_messages_tokens(messages)

        self.session_logger.log_prompt(turn_num, messages, est_prompt_tokens)

        # 4. Inference Loop with retry & fallback handling
        t0 = time.time()
        while True:
            raw_output, error_reason = await self.execute_inference(messages, self.context.turn_count)
            latency = time.time() - t0

            if error_reason is None and raw_output and raw_output.strip():
                completion_tokens = estimate_tokens(raw_output)
                self.token_monitor.record_turn(est_prompt_tokens, completion_tokens, latency)
                self.session_logger.log_response(
                    turn=turn_num,
                    raw_response=raw_output,
                    latency_seconds=latency,
                    estimated_tokens=completion_tokens,
                )
                try:
                    parsed_response = self.parser.parse_with_retry(
                        raw_text=raw_output,
                        local_context=self.context,
                        provider=self.provider,
                    )
                    self.session_logger.log_parse_result(
                        turn=turn_num,
                        success=True,
                    )
                    break
                except Exception as exc:
                    logger.warning(f"Response parsing failed: {exc}")
                    error_reason = f"Malformed model response: {exc}"
                    self.session_logger.log_parse_result(
                        turn=turn_num,
                        success=False,
                        error=str(exc),
                    )

            # If we reached here, inference or parsing failed
            display_error = error_reason or "Inference failed to produce a valid response."
            decision, fallback_resp = self.handle_inference_recovery(display_error, player_action)
            if decision == "retry":
                t0 = time.time()
                continue
            elif decision == "fallback":
                parsed_response = fallback_resp
                completion_tokens = estimate_tokens(fallback_resp.narration)
                self.token_monitor.record_turn(est_prompt_tokens, completion_tokens, latency)
                self.session_logger.log_guard_event("FALLBACK_USED", f"Used turn fallback response for action '{player_action}'.")
                break
            else:
                # Player chose quit
                return None

        # 5. Guardian Narrative Validation
        validated_response = self.validator.validate_response(parsed_response, self.context)

        # 6. Sync Narrative State & log delta
        old_context = self.context.model_copy(deep=True)
        self.context = sync_state(self.context, validated_response)
        self.session_logger.log_state_transition(self.context.turn_count, old_context, self.context)

        # 7. Optional Summary Refinement (2nd-pass)
        if self.summary_refiner.should_refine(self.context):
            try:
                refined = await asyncio.to_thread(
                    self.summary_refiner.refine_summary,
                    self.provider,
                    player_action,
                    validated_response.narration,
                    old_context.story_summary,
                )
                if (
                    refined
                    and refined != self.context.story_summary
                    and not refined.startswith("*")
                    and not refined.startswith("-")
                    and len(refined) >= 15
                ):
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

        model_name = getattr(self.provider, "model_name", "unknown")
        provider_name = getattr(self.provider, "provider_name", "unknown")
        self.session_logger.log_session_start(model_name, provider_name)

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

            # Generate or retrieve cached scene illustration
            scene_art = None
            vis_cfg = getattr(self.config, "visualizer", None)
            is_enabled = getattr(vis_cfg, "enabled", True) if vis_cfg is not None else True
            if self.visualizer and is_enabled:
                try:
                    scene_art = await asyncio.to_thread(
                        self.visualizer.generate_scene_art,
                        narration=self.current_response.narration,
                        location=self.context.current_location,
                        world_tone=self.context.world_tone,
                    )
                except Exception as exc:
                    logger.debug(f"Visualizer error (non-fatal): {exc}")
                    scene_art = None

            # Render current 4-zone game frame
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
                visual_art=scene_art,
            )

            # Capture player action or command
            payload: ActionPayload = await self.input_engine.get_action_async(
                current_choices=self.current_response.choices,
                error_callback=lambda msg: self.tui.console.print(f"[bold red]{msg}[/bold red]"),
            )

            # Handle Commands
            if payload.input_type == InputType.COMMAND:
                if payload.command_name == "debug":
                    self.tui.print_panel(
                        render_debug_overlay(
                            context=self.context,
                            token_monitor=self.token_monitor,
                            box_style=self.tui.box_style,
                        )
                    )
                    self.input_engine._input_func("\nPress Enter to return to the game...")
                    continue
                elif payload.command_name == "help":
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
