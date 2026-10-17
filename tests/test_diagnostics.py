"""Unit and integration tests for SessionLogger, TokenMonitor, ContextGuard, and Debug Overlay."""

from pathlib import Path
import pytest
from rich.console import Console

from src.config import ContextConfig
from src.engine.diagnostics import SessionLogger
from src.engine.loop import GameEngine
from src.engine.token_monitor import (
    ContextGuard,
    TokenMonitor,
    estimate_messages_tokens,
    estimate_tokens,
)
from src.state.models import CharacterStatus, GameResponse, StoryContext
from src.ui.input_engine import InputEngine, InputType
from src.ui.layout import render_debug_overlay
from src.ui.renderer import RetroTUI
from tests.test_game_loop import MockProvider


class TestSessionLogger:
    """Test diagnostic session logging to file."""

    def test_session_logger_creation_and_header(self, tmp_path: Path):
        log_file = tmp_path / "test_debug_log.txt"
        logger = SessionLogger(log_path=log_file)

        logger.log_session_start(model_name="gemma4:e2b", provider_name="[LOCAL: gemma4:e2b]")

        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "GEMMA-QUEST SESSION START" in content
        assert "gemma4:e2b" in content
        assert "[LOCAL: gemma4:e2b]" in content

    def test_log_turn_lifecycle(self, tmp_path: Path):
        log_file = tmp_path / "test_debug_log.txt"
        logger = SessionLogger(log_path=log_file)

        # 1. Turn Start
        logger.log_turn_start(turn=1, player_action="Examine altar")

        # 2. Prompt
        messages = [
            {"role": "system", "content": "You are DM."},
            {"role": "user", "content": "Examine altar"},
        ]
        logger.log_prompt(turn=1, messages=messages, estimated_tokens=150)

        # 3. Response
        logger.log_response(
            turn=1,
            raw_response='{"narration": "Altar shines."}',
            latency_seconds=12.45,
            estimated_tokens=45,
        )

        # 4. Parse Result
        logger.log_parse_result(turn=1, success=True, retries=0)

        # 5. State Transition
        old_ctx = StoryContext(
            current_location="Chamber",
            active_characters={},
            key_items_held=[],
            story_summary="Old summary.",
        )
        new_ctx = StoryContext(
            current_location="Altar Room",
            active_characters={"Priest": CharacterStatus.ALLY},
            key_items_held=["Sacred Relic"],
            story_summary="Found relic.",
        )
        logger.log_state_transition(turn=1, old_context=old_ctx, new_context=new_ctx)

        content = log_file.read_text(encoding="utf-8")
        assert "TURN 1 - PLAYER ACTION: Examine altar" in content
        assert "PROMPT DISPATCH" in content
        assert "Latency: 12.45s" in content
        assert "PARSE RESULT (Turn: 1): SUCCESS" in content
        assert "Location: 'Chamber' -> 'Altar Room'" in content
        assert "Items Added: Sacred Relic" in content
        assert "Character Met: Priest (Ally)" in content

    def test_log_guard_event(self, tmp_path: Path):
        log_file = tmp_path / "test_debug_log.txt"
        logger = SessionLogger(log_path=log_file)

        logger.log_guard_event(
            event_type="CONTEXT_OVERFLOW_WARNING",
            details="Estimated tokens 3400 exceeded 80% threshold.",
        )

        content = log_file.read_text(encoding="utf-8")
        assert "[CONTEXT GUARD ALERT] CONTEXT_OVERFLOW_WARNING" in content
        assert "Estimated tokens 3400" in content


class TestTokenMonitor:
    """Test token estimation heuristics and cumulative metric recording."""

    def test_estimate_tokens(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("   ") == 0

        # Short phrase (~5 tokens)
        short_est = estimate_tokens("Inspect the ancient brass lantern.")
        assert 5 <= short_est <= 15

        # JSON markup with structural symbols
        json_sample = '{"narration": "You hear footsteps in the hall.", "choices": ["Flee", "Fight"]}'
        json_est = estimate_tokens(json_sample)
        assert 15 <= json_est <= 35

    def test_estimate_messages_tokens(self):
        messages = [
            {"role": "system", "content": "You are the retro dungeon master."},
            {"role": "user", "content": "I draw my broadsword and step forward."},
        ]
        total_est = estimate_messages_tokens(messages)
        assert total_est > 15

    def test_token_monitor_metrics(self):
        monitor = TokenMonitor(context_limit=4000)

        # Record Turn 1
        monitor.record_turn(prompt_tokens=400, completion_tokens=100, latency_seconds=10.0)
        assert monitor.last_prompt_tokens == 400
        assert monitor.last_completion_tokens == 100
        assert monitor.cumulative_total_tokens == 500
        assert monitor.context_utilization_ratio == 0.1
        assert monitor.context_utilization_percent == 10.0
        assert monitor.average_latency == 10.0

        # Record Turn 2
        monitor.record_turn(prompt_tokens=800, completion_tokens=200, latency_seconds=20.0)
        assert monitor.last_prompt_tokens == 800
        assert monitor.cumulative_prompt_tokens == 1200
        assert monitor.cumulative_completion_tokens == 300
        assert monitor.cumulative_total_tokens == 1500
        assert monitor.average_latency == 15.0


class TestContextGuard:
    """Test context limit checking, summary compression, and character pruning."""

    def test_warning_token_threshold(self):
        guard = ContextGuard(context_limit=4096)
        # Default ratio is 0.8
        assert guard.warning_token_threshold == int(4096 * 0.8)

    def test_is_approaching_limit(self):
        guard = ContextGuard(context_limit=4000)  # threshold 3200
        assert not guard.is_approaching_limit(3000)
        assert guard.is_approaching_limit(3200)
        assert guard.is_approaching_limit(3500)

    def test_compress_summary(self):
        guard = ContextGuard()
        long_summary = (
            "Sentence one anchors the quest goal. "
            "Sentence two describes the old gate. "
            "Sentence three describes fighting a ghoul. "
            "Sentence four describes finding a key. "
            "Sentence five describes opening the final door."
        )
        compressed = guard.compress_summary(long_summary, max_sentences=3)
        assert "Sentence one anchors the quest goal." in compressed
        assert "Sentence five describes opening the final door." in compressed
        # Middle sentence two should be compressed away
        assert "Sentence two describes" not in compressed

    def test_prune_characters_dead_threshold(self):
        cfg = ContextConfig(npc_prune_turn_threshold=10)
        guard = ContextGuard(config=cfg)

        characters = {
            "Hero": CharacterStatus.ALIVE,
            "Goblin Guard": CharacterStatus.DEAD,
            "Old Hermit": CharacterStatus.ALLY,
        }

        # Turn 1: Goblin dies
        guard.update_character_tracking(characters, current_turn=1)

        # Turn 5: Goblin dead for 4 turns (below 10 threshold) -> not pruned
        retained, pruned = guard.prune_characters(characters, current_turn=5, player_name="Hero")
        assert "Goblin Guard" in retained
        assert len(pruned) == 0

        # Turn 12: Goblin dead for 11 turns (exceeds 10) -> pruned
        retained, pruned = guard.prune_characters(characters, current_turn=12, player_name="Hero")
        assert "Goblin Guard" not in retained
        assert "Goblin Guard" in pruned
        assert "Hero" in retained
        assert "Old Hermit" in retained

    def test_prune_characters_capacity_cap(self):
        cfg = ContextConfig(max_active_npcs=3)
        guard = ContextGuard(config=cfg)

        # 5 characters + Player
        characters = {
            "Hero": CharacterStatus.ALIVE,
            "NPC 1": CharacterStatus.ALLY,
            "NPC 2": CharacterStatus.ALLY,
            "NPC 3": CharacterStatus.ALLY,
            "NPC 4": CharacterStatus.ALLY,
        }
        # Simulate last active turns
        guard.character_last_active_turn = {
            "NPC 1": 1,
            "NPC 2": 2,
            "NPC 3": 5,
            "NPC 4": 6,
        }

        retained, pruned = guard.prune_characters(characters, current_turn=6, player_name="Hero")
        # Retained must be capped to max_active_npcs (3) + Hero
        non_player_retained = [n for n in retained if n != "Hero"]
        assert len(non_player_retained) == 3
        # Oldest active (NPC 1) should be pruned
        assert "NPC 1" in pruned


class TestDebugCommandsAndOverlay:
    """Test debug command parsing and overlay panel rendering."""

    def test_debug_command_recognition(self):
        engine = InputEngine()
        choices = ["Look", "Walk"]

        for cmd in ("?", "[?]", "debug", "/debug", "/inspect", "DEBUG"):
            payload = engine.parse_raw_input(cmd, choices)
            assert payload.input_type == InputType.COMMAND
            assert payload.command_name == "debug"

    def test_render_debug_overlay(self):
        context = StoryContext(
            current_location="Crystal Cave",
            world_tone="Mystic Fantasy",
            active_characters={"Crystal Golem": CharacterStatus.HOSTILE},
            key_items_held=["Glowing Shard"],
            story_summary="Exploring the crystal cavern.",
            turn_count=4,
        )
        monitor = TokenMonitor(context_limit=4096)
        monitor.record_turn(prompt_tokens=520, completion_tokens=140, latency_seconds=14.2)

        console = Console(record=True, width=100)
        panel = render_debug_overlay(context, monitor)
        console.print(panel)

        output = console.export_text()
        assert "SYSTEM DIAGNOSTICS & LIVE STATE INSPECTOR" in output
        assert "Crystal Cave" in output
        assert "Crystal Golem" in output
        assert "Glowing Shard" in output
        assert "520 / 4096 tokens" in output
        assert "14.20s" in output

    @pytest.mark.asyncio
    async def test_game_engine_debug_command_flow(self, tmp_path: Path):
        log_file = tmp_path / "debug_flow.txt"
        logger = SessionLogger(log_path=log_file)
        tui = RetroTUI(console=Console(record=True, width=100))
        provider = MockProvider(response_text="{}")

        engine = GameEngine(
            provider=provider,
            tui=tui,
            session_logger=logger,
        )

        initial_context = StoryContext(
            current_location="Dungeon Entrance",
            story_summary="Ready to explore.",
        )
        initial_response = GameResponse(
            narration="You stand before the arch.",
            choices=["Step inside", "Wait outside"],
            updated_story_context=initial_context,
        )

        # Simulate user entering "[?]", pressing Enter, then entering "/quit"
        inputs = iter(["[?]", "", "/quit"])
        engine.input_engine._input_func = lambda _: next(inputs)

        exit_code = await engine.run(initial_context, initial_response)
        assert exit_code == "quit"

        output = tui.console.export_text()
        assert "SYSTEM DIAGNOSTICS & LIVE STATE INSPECTOR" in output
        assert log_file.exists()
