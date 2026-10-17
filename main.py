#!/usr/bin/env python3
"""Gemma-Quest: Retro 1980s Text Adventure powered by Gemma 4.

Main entry point orchestrating CLI flags, environment setup, TUI initialization,
player onboarding, and the async game loop.
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

from src.config import GameConfig, load_config
from src.engine.base import InferenceProvider
from src.engine.factory import LLMFactory
from src.engine.initializer import OpeningSceneGenerator
from src.engine.loop import GameEngine
from src.state.models import PlayerSetup
from src.ui.input_engine import InputEngine
from src.ui.renderer import RetroTUI

logger = logging.getLogger("gemma_quest")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for Gemma-Quest."""
    parser = argparse.ArgumentParser(
        description="Gemma-Quest: A Retro 1980s Text Adventure powered by Gemma 4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "local", "cloud", "mock"],
        default="auto",
        help="Inference backend provider",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override (e.g. gemma4:e2b, gemini-2.5-flash)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Inference timeout threshold in seconds",
    )
    parser.add_argument(
        "--boot",
        action="store_true",
        default=False,
        help="Show retro 1980s DOS BIOS memory check on startup",
    )
    parser.add_argument(
        "--monochrome",
        action="store_true",
        default=False,
        help="Force high-contrast monochrome terminal mode",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        default=False,
        help="Force plain ASCII borders (+, -, |) instead of Unicode box characters",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose debug logging to gemma_quest.log",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    """Configure file-based and console logging."""
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("gemma_quest.log", mode="a", encoding="utf-8"),
        ],
    )


def preflight_provider_check(
    provider_type: str,
    config: GameConfig,
    tui: RetroTUI,
) -> Optional[InferenceProvider]:
    """Verify backend connectivity and return initialized InferenceProvider.

    Displays user-friendly retro diagnostic guidance if Ollama or Cloud API is offline.
    """
    try:
        mode = None if provider_type == "auto" else provider_type
        provider = LLMFactory.get_provider(config=config, mode_override=mode)
        return provider
    except Exception as exc:
        tui.console.print(f"\n[bold red]⚠️  Inference Provider Warning:[/] [white]{exc}[/white]")
        if provider_type in ("local", "auto"):
            tui.console.print(
                "[bold yellow]Hint:[/] If using local Ollama, ensure it is running:\n"
                "  $ [bright_green]ollama serve[/bright_green]\n"
                "  $ [bright_green]ollama run gemma4:e2b[/bright_green]\n"
            )
        tui.console.print("[dim grey]Falling back to offline fallback generation mode.[/dim grey]\n")
        return None


async def run_adventure(args: argparse.Namespace) -> None:
    """Coordinate the player adventure lifecycle (onboarding, opening scene, and game loop)."""
    configure_logging(args.verbose)

    # 1. Load Configuration
    config = load_config()
    if args.model:
        if args.provider == "cloud":
            config.llm.cloud_model_id = args.model
        else:
            config.llm.local_model = args.model

    # 2. Initialize TUI & Input Engine
    tui = RetroTUI(
        monochrome=args.monochrome,
        force_ascii=args.ascii,
    )
    input_engine = InputEngine()

    # 3. Preflight Inference Provider Check
    provider = preflight_provider_check(args.provider, config, tui)

    # 4. Display Title Splash
    tui.clear_screen(soft=False)
    tui.render_title()

    while True:
        # 5. Run Player Onboarding Flow
        try:
            player_setup: PlayerSetup = tui.run_onboarding(
                show_boot=args.boot,
            )
        except (KeyboardInterrupt, EOFError):
            break

        # 6. Generate Opening Scene
        with tui.console.status(
            "[bold yellow]⏳ Awakening the world of Gemma-Quest...[/bold yellow]",
            spinner="dots" if tui.capabilities.supports_unicode else "line",
            spinner_style="bold #80FF80",
        ):
            scene_gen = OpeningSceneGenerator()
            context, opening_response = await asyncio.to_thread(
                scene_gen.generate,
                player_setup,
                provider=provider,
            )

        # 7. Start Async Game Loop
        engine = GameEngine(
            config=config,
            provider=provider or LLMFactory.get_provider(config, mode_override="local"),
            tui=tui,
            input_engine=input_engine,
            timeout_seconds=args.timeout,
            player_name=player_setup.player_name,
        )

        exit_code = await engine.run(context, opening_response)

        if exit_code == "restart":
            tui.clear_screen(soft=True)
            tui.console.print("\n[bold yellow]═══ RESTARTING NEW ADVENTURE ═══[/bold yellow]\n")
            continue
        else:
            break

    # Graceful exit banner
    tui.console.print(
        "\n[bold yellow]Farewell, brave adventurer. Your legend sleeps in the chronicles of Gemma...[/bold yellow]\n"
    )


def main() -> None:
    """Application CLI entry point."""
    args = parse_args()
    try:
        asyncio.run(run_adventure(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\n[Session ended by adventurer. Farewell!]")
        sys.exit(0)


if __name__ == "__main__":
    main()
