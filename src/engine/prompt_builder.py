"""Context-first prompt builder and dynamic context injection pipeline."""

from typing import Any, Dict, List, Optional
from src.state.models import CharacterStatus, PlayerSetup, StoryContext


DM_SYSTEM_RULES = """You are the Dungeon Master of "Gemma Dungeon," a retro text adventure game.
Your primary goal is to maintain a coherent, engaging, and consistent story based strictly on the provided Story Context.

RULES:
1. COHERENCE: NEVER contradict the Story Summary or Active Characters status. If a character is marked "Dead", they cannot appear alive again. If an item is not in Key Items Held, the player does not have it.
2. STYLE: Keep narration atmospheric but concise (max 3 sentences). Write for a terminal interface (short paragraphs, high impact).
3. CHOICES: Present exactly 2-4 clear narrative choices based on the current situation. Choices must be actionable (e.g., "Open the door", not "Feel sad").
4. STATE MANAGEMENT: You MUST update the Story Context accurately after each turn. Only add to key_items_held if an item is crucial to the plot. Remove items when they are used or lost.
5. OUTPUT FORMAT: Respond strictly with a valid JSON object matching the schema below. Wrap the JSON in ```json ... ``` code fences. Do NOT output any thoughts, reasoning, planning, scratchpad analysis, bullet points, or self-corrections."""


OUTPUT_FORMAT_SCHEMA = """[OUTPUT FORMAT]
Respond ONLY with the JSON object wrapped in markdown code fences:
```json
{
  "narration": "string (2-3 sentences of descriptive text)",
  "choices": [
    "string (Action choice 1)",
    "string (Action choice 2)"
  ],
  "updated_story_context": {
    "current_location": "string (Current or new location)",
    "active_characters": {
      "Character Name": "Status (Alive/Dead/Hostile/Ally/Neutral)"
    },
    "key_items_held": ["item1", "item2"],
    "story_summary": "3-4 sentence summary of what just happened and current plot status"
  }
}
```"""


class PromptBuilder:
    """Constructs context-first prompts and message payloads for Gemma-Quest."""

    @staticmethod
    def format_characters(active_characters: Dict[str, CharacterStatus]) -> str:
        """Format the active characters map into a readable list for the prompt.

        Args:
            active_characters: Map of character name to CharacterStatus.

        Returns:
            Formatted bullet points or 'None' if empty.
        """
        if not active_characters:
            return "None"

        lines = []
        for name, status in active_characters.items():
            status_val = status.value if isinstance(status, CharacterStatus) else str(status)
            lines.append(f"- {name}: {status_val}")
        return "\n".join(lines)

    @staticmethod
    def format_items(key_items_held: List[str]) -> str:
        """Format the held key items into a comma-separated list.

        Args:
            key_items_held: List of item names.

        Returns:
            Comma-separated item string or 'None'.
        """
        if not key_items_held:
            return "None"
        return ", ".join(key_items_held)

    @classmethod
    def build_system_prompt(cls) -> str:
        """Construct the fixed system prompt with persona, rules, and output schema."""
        return f"{DM_SYSTEM_RULES}\n\n{OUTPUT_FORMAT_SCHEMA}"

    @classmethod
    def build_user_context_block(cls, context: StoryContext, player_input: str) -> str:
        """Construct the dynamic context injection and player action block.

        Args:
            context: Current validated StoryContext.
            player_input: Action text or choice selected by the player.

        Returns:
            Formatted context and player action string.
        """
        chars_str = cls.format_characters(context.active_characters)
        items_str = cls.format_items(context.key_items_held)

        return (
            f"[STORY CONTEXT]\n"
            f"Location: {context.current_location}\n"
            f"Tone: {context.world_tone}\n"
            f"Active Characters:\n{chars_str}\n"
            f"Key Items Held: {items_str}\n"
            f"Recent Story Summary: {context.story_summary}\n\n"
            f"[PLAYER ACTION]\n"
            f"{player_input.strip()}\n\n"
            f"Remember: Respond ONLY with the ```json ... ``` code block. Do NOT include any thoughts, analysis, or explanations."
        )

    @classmethod
    def build_chat_messages(
        cls, context: StoryContext, player_input: str
    ) -> List[Dict[str, str]]:
        """Construct role-separated chat messages for chat completion endpoints.

        Compatible with local Ollama chat API and Cloud Google AI Studio endpoints.

        Args:
            context: Current StoryContext.
            player_input: Player's selected action or input.

        Returns:
            List of message dicts with 'role' and 'content'.
        """
        system_content = cls.build_system_prompt()
        user_content = cls.build_user_context_block(context, player_input)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    @classmethod
    def build_monolithic_prompt(
        cls, context: StoryContext, player_input: str
    ) -> str:
        """Construct a single monolithic prompt string for completion-style models.

        Args:
            context: Current StoryContext.
            player_input: Player's selected action or input.

        Returns:
            Complete prompt string with rules, context, and output schema.
        """
        system_block = cls.build_system_prompt()
        user_block = cls.build_user_context_block(context, player_input)
        return f"{system_block}\n\n{user_block}"

    @classmethod
    def build_opening_messages(cls, setup: PlayerSetup) -> List[Dict[str, str]]:
        """Construct prompt messages to generate the initial game opening scene.

        Args:
            setup: Player setup choices containing character name, setting, and goal.

        Returns:
            List of role-based chat messages.
        """
        system_content = cls.build_system_prompt()

        setting_desc = setup.setting_type
        if setup.custom_setting_details:
            setting_desc += f" ({setup.custom_setting_details})"

        user_content = (
            f"[NEW GAME INITIALIZATION]\n"
            f"Player Character: {setup.player_name}\n"
            f"World Setting: {setting_desc}\n"
            f"Initial Quest/Goal: {setup.initial_goal or 'Survive and discover the world'}\n\n"
            f"Generate an evocative opening scene introducing the player character in their starting location. "
            f"Provide 2-4 starting choices to begin the adventure.\n"
            f"Return ONLY the requested JSON object matching the schema."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    @classmethod
    def build_json_repair_messages(
        cls, invalid_output: str, error_details: str
    ) -> List[Dict[str, str]]:
        """Construct a targeted correction prompt when model output failed JSON parsing.

        Args:
            invalid_output: The raw text that failed parsing.
            error_details: Reason or exception text describing the parse failure.

        Returns:
            List of chat messages directing the model to repair the JSON.
        """
        system_content = (
            "You are a JSON formatting assistant for a game engine. "
            "Your job is to convert malformed text into valid, strictly compliant JSON."
        )

        user_content = (
            f"The previous output failed JSON validation.\n"
            f"Validation Error: {error_details}\n\n"
            f"Malformed Output:\n{invalid_output}\n\n"
            f"Correct the syntax errors and return ONLY a valid JSON object matching this structure:\n"
            f"{OUTPUT_FORMAT_SCHEMA}\n"
            f"Do not include explanation, preamble, or markdown code fences."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    @classmethod
    def build_dead_character_correction_messages(
        cls, dead_character: str, original_narration: str
    ) -> List[Dict[str, str]]:
        """Construct a rewrite prompt when a dead character was inadvertently revived.

        Args:
            dead_character: The name of the deceased character.
            original_narration: The narrative that mistakenly revived the character.

        Returns:
            List of chat messages directing a rewrite adhering to the Dead Man's Switch.
        """
        system_content = cls.build_system_prompt()
        user_content = (
            f"Constraint Violation: Character '{dead_character}' is DEAD and cannot speak, act, or appear alive.\n\n"
            f"Original Scene:\n{original_narration}\n\n"
            f"Please rewrite the scene without reviving '{dead_character}', maintaining narrative continuity. "
            f"Return ONLY the valid JSON object matching the output schema."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
