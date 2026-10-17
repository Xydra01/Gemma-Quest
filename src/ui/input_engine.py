"""Interactive input engine for Gemma-Quest retro TUI.

Provides numbered choice validation (1-4), out-of-range re-prompting,
free-form custom actions, input sanitization against prompt injection,
empty input [WAIT] handling, and slash command parsing.
"""

import asyncio
from enum import Enum
import logging
import re
from typing import Callable, List, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)


class InputType(str, Enum):
    """Categorized type of player input."""
    CHOICE = "choice"          # Numbered selection mapped to choice text
    CUSTOM = "custom"          # Free-form narrative action
    WAIT = "wait"              # Empty input defaulting to observation
    COMMAND = "command"        # Slash command (/help, /status, /quit, /restart)
    INVALID = "invalid"        # Out-of-range number or unrecognized command


class ActionPayload(NamedTuple):
    """Structured payload returned by InputEngine."""
    input_type: InputType
    action_text: str
    command_name: Optional[str] = None
    error_message: Optional[str] = None


def sanitize_player_input(raw_input: Optional[str], max_length: int = 200) -> str:
    """Sanitize player input string against prompt injection and terminal artifacts.

    Strips script/style blocks along with their contents, strips HTML tags,
    collapses whitespace, and enforces a maximum character length.

    Args:
        raw_input: Raw string received from terminal/user.
        max_length: Maximum allowed length.

    Returns:
        Clean, sanitized string.
    """
    if not raw_input:
        return ""

    # 1. Strip script and style blocks and their internal content
    cleaned = re.sub(
        r"<(script|style)[^>]*>.*?</\1>", "", raw_input, flags=re.IGNORECASE | re.DOTALL
    )

    # 2. Strip any other HTML / XML tags
    cleaned = re.sub(r"<[^>]+>", "", cleaned)

    # 3. Collapse whitespace and line breaks
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()

    # 4. Truncate to maximum allowed length
    return cleaned[:max_length]


class InputEngine:
    """Coordinates player input capture, validation, mapping, and sanitization."""

    DEFAULT_WAIT_ACTION = "[WAIT] You pause and observe your surroundings closely."

    def __init__(
        self,
        input_func: Optional[Callable[[str], str]] = None,
        max_input_length: int = 200,
    ):
        """Initialize the input engine.

        Args:
            input_func: Optional synchronous callable for input capture (defaults to builtin input).
            max_input_length: Maximum length for free-form actions.
        """
        self._input_func = input_func or input
        self.max_input_length = max_input_length

    def parse_raw_input(
        self,
        raw_input: Optional[str],
        current_choices: List[str],
    ) -> ActionPayload:
        """Parse, categorize, and validate raw text against current choices.

        Args:
            raw_input: Text input by the player.
            current_choices: List of currently available choices (typically 2-4).

        Returns:
            ActionPayload containing the categorized input and resolved text.
        """
        sanitized = sanitize_player_input(raw_input, max_length=self.max_input_length)

        # 1. Empty input defaults to WAIT
        if not sanitized:
            return ActionPayload(
                input_type=InputType.WAIT,
                action_text=self.DEFAULT_WAIT_ACTION,
            )

        # 2. Check for debug / inspection commands
        lower_input = sanitized.lower()
        if lower_input in ("?", "[?]", "debug", "/debug", "/inspect"):
            return ActionPayload(
                input_type=InputType.COMMAND,
                action_text=sanitized,
                command_name="debug",
            )

        # 3. Check for slash commands
        if sanitized.startswith("/"):
            cmd_lower = sanitized.lower()
            if cmd_lower in ("/help", "/h", "/?"):
                return ActionPayload(
                    input_type=InputType.COMMAND,
                    action_text=sanitized,
                    command_name="help",
                )
            elif cmd_lower in ("/status", "/inv", "/inventory", "/stats"):
                return ActionPayload(
                    input_type=InputType.COMMAND,
                    action_text=sanitized,
                    command_name="status",
                )
            elif cmd_lower in ("/quit", "/exit", "/q"):
                return ActionPayload(
                    input_type=InputType.COMMAND,
                    action_text=sanitized,
                    command_name="quit",
                )
            elif cmd_lower in ("/restart", "/new", "/reset"):
                return ActionPayload(
                    input_type=InputType.COMMAND,
                    action_text=sanitized,
                    command_name="restart",
                )
            else:
                return ActionPayload(
                    input_type=InputType.INVALID,
                    action_text=sanitized,
                    error_message=f"Unknown command '{sanitized}'. Type /help for available commands.",
                )

        # 3. Check for numeric selection
        if sanitized.isdigit():
            idx = int(sanitized)
            if 1 <= idx <= len(current_choices):
                chosen_choice = current_choices[idx - 1].strip()
                # Clean leading bracketed numbering like "[1] Inspect the door" if present
                clean_text = re.sub(r"^\[\d+\]\s*", "", chosen_choice)
                clean_text = re.sub(r"^\d+\.\s*", "", clean_text)
                return ActionPayload(
                    input_type=InputType.CHOICE,
                    action_text=clean_text,
                )
            elif idx == 0:
                # 0 is designated as free-form prompt
                return ActionPayload(
                    input_type=InputType.INVALID,
                    action_text=sanitized,
                    error_message="Option [0] is for custom actions: simply type your action directly into the prompt.",
                )
            else:
                return ActionPayload(
                    input_type=InputType.INVALID,
                    action_text=sanitized,
                    error_message=f"Invalid selection [{idx}]. Please enter a number between 1 and {len(current_choices)}.",
                )

        # 4. Otherwise, treated as valid custom narrative action
        return ActionPayload(
            input_type=InputType.CUSTOM,
            action_text=sanitized,
        )

    def get_action(
        self,
        current_choices: List[str],
        prompt: str = "\n> Select [1-{max}] or type custom action: ",
        error_callback: Optional[Callable[[str], None]] = None,
    ) -> ActionPayload:
        """Synchronously prompt the user until a valid action or command is received.

        Args:
            current_choices: List of currently available choices.
            prompt: Text to display at the input prompt.
            error_callback: Optional callable to display validation errors.

        Returns:
            ActionPayload containing valid action or command.
        """
        formatted_prompt = prompt.replace("{max}", str(len(current_choices)))

        while True:
            try:
                raw = self._input_func(formatted_prompt)
            except (EOFError, KeyboardInterrupt):
                return ActionPayload(
                    input_type=InputType.COMMAND,
                    action_text="/quit",
                    command_name="quit",
                )

            payload = self.parse_raw_input(raw, current_choices)
            if payload.input_type != InputType.INVALID:
                return payload

            if error_callback and payload.error_message:
                error_callback(payload.error_message)
            elif payload.error_message:
                print(payload.error_message)

    async def get_action_async(
        self,
        current_choices: List[str],
        prompt: str = "\n> Select [1-{max}] or type custom action: ",
        error_callback: Optional[Callable[[str], None]] = None,
    ) -> ActionPayload:
        """Asynchronously capture player input without blocking the asyncio event loop.

        Args:
            current_choices: List of currently available choices.
            prompt: Text to display at the input prompt.
            error_callback: Optional callable to display validation errors.

        Returns:
            ActionPayload containing valid action or command.
        """
        return await asyncio.to_thread(
            self.get_action,
            current_choices,
            prompt,
            error_callback,
        )
