"""Developer session logger and diagnostic recording engine for Gemma-Quest.

Records timestamped raw prompts, LLM responses, inference latencies, parsing results,
validation warnings, retry events, context guard triggers, and state transitions to debug_log.txt.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.state.models import StoryContext

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("debug_log.txt")


class SessionLogger:
    """Manages structured diagnostic session logging to a dedicated file."""

    def __init__(self, log_path: Optional[Path | str] = None, auto_flush: bool = True):
        """Initialize session logger.

        Args:
            log_path: Path to diagnostic log file (defaults to debug_log.txt).
            auto_flush: If True, flushes output immediately after each write.
        """
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG_PATH
        self.auto_flush = auto_flush
        self._ensure_parent_dir()

    def _ensure_parent_dir(self) -> None:
        """Ensure the parent directory of the log file exists."""
        if self.log_path.parent and not self.log_path.parent.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, text: str) -> None:
        """Append raw text to the log file."""
        try:
            with open(self.log_path, mode="a", encoding="utf-8") as f:
                f.write(text)
                if self.auto_flush:
                    f.flush()
        except Exception as exc:
            logger.warning(f"Failed to write to session log {self.log_path}: {exc}")

    def log_session_start(self, model_name: str, provider_name: str) -> None:
        """Record diagnostic session header with timestamp and environment details."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        divider = "=" * 80
        header = (
            f"\n{divider}\n"
            f"[{timestamp}] GEMMA-QUEST SESSION START\n"
            f"Model: {model_name} | Provider: {provider_name}\n"
            f"{divider}\n\n"
        )
        self._append(header)

    def log_turn_start(self, turn: int, player_action: str) -> None:
        """Record the beginning of a turn and the player's attempted action."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"----------------------------------------\n"
            f"[{timestamp}] TURN {turn} - PLAYER ACTION: {player_action}\n"
            f"----------------------------------------\n"
        )
        self._append(entry)

    def log_prompt(
        self,
        turn: int,
        messages: List[Dict[str, str]],
        estimated_tokens: int,
    ) -> None:
        """Record the complete prompt payload dispatched to the LLM."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"[{timestamp}] PROMPT DISPATCH (Turn: {turn} | Est. Tokens: {estimated_tokens})",
        ]
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "").strip()
            lines.append(f"--- [{role}] ---")
            lines.append(content)
        lines.append("\n")
        self._append("\n".join(lines) + "\n")

    def log_response(
        self,
        turn: int,
        raw_response: str,
        latency_seconds: float,
        estimated_tokens: Optional[int] = None,
    ) -> None:
        """Record the raw completion received from the LLM and inference latency."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        token_info = f" | Est. Tokens: {estimated_tokens}" if estimated_tokens is not None else ""
        entry = (
            f"[{timestamp}] RESPONSE RECEIVED (Turn: {turn} | Latency: {latency_seconds:.2f}s{token_info})\n"
            f"--- [RAW LLM OUTPUT] ---\n"
            f"{raw_response.strip()}\n\n"
        )
        self._append(entry)

    def log_parse_result(
        self,
        turn: int,
        success: bool,
        retries: int = 0,
        error: Optional[str] = None,
        warnings: Optional[List[str]] = None,
    ) -> None:
        """Record parsing and validation outcomes."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_str = "SUCCESS" if success else "FAILED"
        lines = [
            f"[{timestamp}] PARSE RESULT (Turn: {turn}): {status_str} (Retries: {retries})",
        ]
        if error:
            lines.append(f"  Error: {error}")
        if warnings:
            for w in warnings:
                lines.append(f"  Warning: {w}")
        self._append("\n".join(lines) + "\n\n")

    def log_state_transition(
        self,
        turn: int,
        old_context: StoryContext,
        new_context: StoryContext,
    ) -> None:
        """Record state deltas between pre-turn and post-turn contexts."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        deltas = []

        # Location delta
        if old_context.current_location != new_context.current_location:
            deltas.append(f"  Location: '{old_context.current_location}' -> '{new_context.current_location}'")

        # Items delta
        old_items = set(old_context.key_items_held)
        new_items = set(new_context.key_items_held)
        added_items = new_items - old_items
        removed_items = old_items - new_items
        if added_items:
            deltas.append(f"  Items Added: {', '.join(added_items)}")
        if removed_items:
            deltas.append(f"  Items Removed: {', '.join(removed_items)}")

        # Characters delta
        old_chars = old_context.active_characters
        new_chars = new_context.active_characters
        for name, status in new_chars.items():
            st_val = status.value if hasattr(status, "value") else str(status)
            if name not in old_chars:
                deltas.append(f"  Character Met: {name} ({st_val})")
            elif old_chars[name] != status:
                old_val = old_chars[name].value if hasattr(old_chars[name], "value") else str(old_chars[name])
                deltas.append(f"  Character Status Shift: {name} ({old_val} -> {st_val})")

        # Pruned characters
        pruned_chars = set(old_chars.keys()) - set(new_chars.keys())
        if pruned_chars:
            deltas.append(f"  Characters Pruned: {', '.join(pruned_chars)}")

        # Summary delta
        if old_context.story_summary != new_context.story_summary:
            deltas.append(f"  Updated Memory: {new_context.story_summary}")

        if not deltas:
            delta_text = "  (No state invariants changed)\n"
        else:
            delta_text = "\n".join(deltas) + "\n"

        entry = (
            f"[{timestamp}] STATE DELTA (Turn: {turn}):\n"
            f"{delta_text}\n"
        )
        self._append(entry)

    def log_guard_event(self, event_type: str, details: str) -> None:
        """Record context guard events such as token overflow, compression, or character pruning."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"[{timestamp}] [CONTEXT GUARD ALERT] {event_type}\n"
            f"  Details: {details}\n\n"
        )
        self._append(entry)
