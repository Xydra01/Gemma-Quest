"""Token estimation, context window protection, and NPC character pruning for Gemma-Quest.

Guards the 4096-token context window against overflow, triggers aggressive memory
compression when exceeding 80% capacity, and prunes stale/dead NPCs.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from src.config import ContextConfig
from src.state.models import CharacterStatus, StoryContext

logger = logging.getLogger(__name__)

# Default Gemma 4 context window specification
DEFAULT_CONTEXT_WINDOW_LIMIT = 4096
# Typical characters-per-token ratio for English text + JSON formatting
CHARS_PER_TOKEN_RATIO = 3.8
# Protocol framing overhead per chat message
MESSAGE_FRAMING_OVERHEAD_TOKENS = 4


def estimate_tokens(text: Optional[str]) -> int:
    """Estimate token count for a text string using calibrated heuristic.

    Calibrated against Gemma/Llama tokenizers for English prose, Markdown, and JSON.

    Args:
        text: Input string to measure.

    Returns:
        Estimated number of tokens (minimum 0).
    """
    if not text:
        return 0

    stripped = text.strip()
    if not stripped:
        return 0

    # Punctuation/structural tokens in JSON markup (brackets, quotes, braces)
    structure_count = len(re.findall(r"[{}\[\]\":,]", stripped))
    base_token_est = int(len(stripped) / CHARS_PER_TOKEN_RATIO)

    # Return blended estimate with a small structural weight
    return max(1, base_token_est + int(structure_count * 0.2))


def estimate_messages_tokens(messages: List[Dict[str, str]]) -> int:
    """Estimate total tokens across a list of chat message dictionaries.

    Includes role framing token overhead per turn.

    Args:
        messages: List of message dictionaries with 'role' and 'content'.

    Returns:
        Total estimated token count.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "")
        total += estimate_tokens(content) + estimate_tokens(role) + MESSAGE_FRAMING_OVERHEAD_TOKENS
    return total


class TokenMonitor:
    """Tracks token consumption, context window utilization, and latency metrics."""

    def __init__(self, context_limit: int = DEFAULT_CONTEXT_WINDOW_LIMIT):
        """Initialize token monitor.

        Args:
            context_limit: Maximum allowed context window tokens (default 4096).
        """
        self.context_limit = context_limit
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_latency: float = 0.0

        self.cumulative_prompt_tokens: int = 0
        self.cumulative_completion_tokens: int = 0
        self.cumulative_total_tokens: int = 0
        self.latencies: List[float] = []

    def record_turn(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float,
    ) -> None:
        """Record usage metrics for an individual turn.

        Args:
            prompt_tokens: Tokens in dispatched prompt.
            completion_tokens: Tokens in received completion.
            latency_seconds: Total duration of model inference.
        """
        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens
        self.last_latency = latency_seconds

        self.cumulative_prompt_tokens += prompt_tokens
        self.cumulative_completion_tokens += completion_tokens
        self.cumulative_total_tokens += (prompt_tokens + completion_tokens)
        self.latencies.append(latency_seconds)

    @property
    def context_utilization_ratio(self) -> float:
        """Ratio of context window consumed by the last prompt (0.0 to 1.0+)."""
        if self.context_limit <= 0:
            return 0.0
        return self.last_prompt_tokens / self.context_limit

    @property
    def context_utilization_percent(self) -> float:
        """Percentage of context window consumed by the last prompt."""
        return self.context_utilization_ratio * 100.0

    @property
    def average_latency(self) -> float:
        """Average latency in seconds across all recorded turns."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)


class ContextGuard:
    """Enforces context window limits, triggers memory compression, and prunes stale NPCs."""

    def __init__(
        self,
        config: Optional[ContextConfig] = None,
        context_limit: int = DEFAULT_CONTEXT_WINDOW_LIMIT,
    ):
        """Initialize ContextGuard with context management configuration.

        Args:
            config: Context configuration options.
            context_limit: Total context window size in tokens (default 4096).
        """
        self.config = config or ContextConfig()
        self.context_limit = context_limit
        # Turn threshold tracking: npc_name -> last_active_turn
        self.character_last_active_turn: Dict[str, int] = {}
        # Death turn tracking: npc_name -> death_turn
        self.character_death_turn: Dict[str, int] = {}

    @property
    def warning_token_threshold(self) -> int:
        """Token count threshold that triggers compression (e.g. 80% of 4096 = 3276 tokens)."""
        return int(self.context_limit * self.config.context_window_warning_ratio)

    def is_approaching_limit(self, estimated_prompt_tokens: int) -> bool:
        """Check if prompt token length exceeds warning ratio threshold."""
        return estimated_prompt_tokens >= self.warning_token_threshold

    def update_character_tracking(
        self,
        active_characters: Dict[str, CharacterStatus],
        current_turn: int,
    ) -> None:
        """Update last seen and death turns for all characters in the current state."""
        for name, status in active_characters.items():
            st_val = status.value if hasattr(status, "value") else str(status)
            if str(st_val).upper() == "DEAD":
                if name not in self.character_death_turn:
                    self.character_death_turn[name] = current_turn
            elif name not in self.character_last_active_turn:
                self.character_last_active_turn[name] = current_turn

    def prune_characters(
        self,
        characters: Dict[str, CharacterStatus],
        current_turn: int,
        player_name: str = "Hero",
    ) -> Tuple[Dict[str, CharacterStatus], List[str]]:
        """Prune characters that have been dead or absent for >10 turns or exceed max NPC capacity.

        Args:
            characters: Current active characters dictionary.
            current_turn: Current turn counter.
            player_name: Protected character name that must never be pruned.

        Returns:
            Tuple of (retained_characters_dict, list_of_pruned_character_names).
        """
        self.update_character_tracking(characters, current_turn)

        pruned_names: List[str] = []
        retained: Dict[str, CharacterStatus] = {}

        for name, status in characters.items():
            # Never prune the player
            if name.lower() == player_name.lower():
                retained[name] = status
                continue

            st_val = status.value if hasattr(status, "value") else str(status)
            is_dead = str(st_val).upper() == "DEAD"
            is_ally = str(st_val).upper() == "ALLY"

            # Check if dead for > threshold turns
            if is_dead:
                death_turn = self.character_death_turn.get(name, current_turn)
                if (current_turn - death_turn) >= self.config.npc_prune_turn_threshold:
                    pruned_names.append(name)
                    continue

            # Check if inactive/absent for > threshold turns (companions/allies are prioritized and protected)
            if not is_ally and not is_dead:
                last_active = self.character_last_active_turn.get(name, current_turn)
                if (current_turn - last_active) >= self.config.npc_prune_turn_threshold:
                    pruned_names.append(name)
                    continue

            retained[name] = status

        # If retained NPCs still exceed max_active_npcs, prune excess NPCs (prioritize retaining allies)
        non_player_count = sum(1 for n in retained.keys() if n.lower() != player_name.lower())
        if non_player_count > self.config.max_active_npcs:
            def prune_priority(n: str):
                s = retained[n]
                val = s.value if hasattr(s, "value") else str(s)
                ally_weight = 1 if str(val).upper() == "ALLY" else 0
                last_act = self.character_last_active_turn.get(n, 0)
                return (ally_weight, last_act)

            candidate_names = [
                n for n in retained.keys()
                if n.lower() != player_name.lower()
            ]
            candidate_names.sort(key=prune_priority)

            excess_count = non_player_count - self.config.max_active_npcs
            for n in candidate_names[:excess_count]:
                del retained[n]
                pruned_names.append(n)

        return retained, pruned_names

    def compress_summary(self, summary: str, max_sentences: int = 3) -> str:
        """Condense long story summary text into key recent narrative sentences.

        Args:
            summary: Input summary string.
            max_sentences: Maximum sentences to retain.

        Returns:
            Compressed summary string.
        """
        if not summary:
            return ""

        # Split sentences by period/exclamation/question mark followed by space
        sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
        if len(sentences) <= max_sentences:
            return summary.strip()

        # Retain first sentence if it anchors identity/goal, plus the last 2 sentences
        first_sentence = sentences[0]
        recent_sentences = sentences[-(max_sentences - 1):]
        condensed = f"{first_sentence} {' '.join(recent_sentences)}"
        return condensed.strip()
