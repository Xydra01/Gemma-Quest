"""Narrative consistency validator (The Guardian) for Gemma-Quest."""

import logging
import re
from typing import List, Optional
from src.state.models import CharacterStatus, GameResponse, StoryContext

logger = logging.getLogger(__name__)

MIN_SUMMARY_LENGTH = 10
DEFAULT_WAIT_CHOICE = "[2] Wait and assess the situation"


class NarrativeValidator:
    """The 'Guardian' validation layer enforcing narrative continuity and consistency invariants."""

    @classmethod
    def validate_dead_characters(
        cls, response: GameResponse, context: StoryContext
    ) -> GameResponse:
        """Enforce the Dead Man's Switch invariant across choices and updated context.

        Dead characters cannot be resurrected, and cannot be addressed as living in choices.

        Args:
            response: Candidate GameResponse.
            context: Verified local StoryContext.

        Returns:
            Sanitized GameResponse with invalid choices cleaned or replaced.
        """
        dead_characters = [
            name
            for name, status in context.active_characters.items()
            if status == CharacterStatus.DEAD
        ]

        if not dead_characters:
            return response

        cleaned_choices: List[str] = []
        for choice in response.choices:
            choice_has_dead_char = False
            violating_char = ""

            for dead_name in dead_characters:
                # Case-insensitive word boundary match
                pattern = r"\b" + re.escape(dead_name) + r"\b"
                if re.search(pattern, choice, flags=re.IGNORECASE):
                    choice_has_dead_char = True
                    violating_char = dead_name
                    break

            if choice_has_dead_char:
                logger.warning(
                    "Dead Man's Switch triggered: Choice '%s' references deceased character '%s'.",
                    choice,
                    violating_char,
                )
                # If we have enough choices without it, omit it
                # Otherwise sanitize the choice so the player does not speak to or fight a corpse
                remaining_candidates = [
                    c for c in response.choices if c != choice and c not in cleaned_choices
                ]
                if len(cleaned_choices) + len(remaining_candidates) >= 2:
                    # Skip adding this choice
                    continue
                else:
                    # Sanitize choice into inspect / search action
                    sanitized_choice = f"Inspect the remains of {violating_char}"
                    cleaned_choices.append(sanitized_choice)
            else:
                cleaned_choices.append(choice)

        # Ensure choice count invariant (2-4 choices)
        while len(cleaned_choices) < 2:
            cleaned_choices.append(DEFAULT_WAIT_CHOICE)
        if len(cleaned_choices) > 4:
            cleaned_choices = cleaned_choices[:4]

        # Invariant: Prevent resurrection and preserve dead character tracking in updated_story_context
        updated_characters = dict(response.updated_story_context.active_characters)
        for dead_name in dead_characters:
            if dead_name not in updated_characters or updated_characters[dead_name] != CharacterStatus.DEAD:
                logger.warning(
                    "Enforcing DEAD status for '%s' in active_characters.",
                    dead_name,
                )
                updated_characters[dead_name] = CharacterStatus.DEAD

        # Update updated_story_context with enforced character statuses
        new_context = response.updated_story_context.model_copy(
            update={"active_characters": updated_characters}
        )

        return response.model_copy(
            update={"choices": cleaned_choices, "updated_story_context": new_context}
        )

    @classmethod
    def validate_summary(
        cls, response: GameResponse, context: StoryContext
    ) -> GameResponse:
        """Ensure the story summary is not empty, truncated (< 10 chars), or contaminated with thinking traces.

        Args:
            response: Candidate GameResponse.
            context: Verified local StoryContext.

        Returns:
            GameResponse with fallback summary applied if needed.
        """
        proposed_summary = response.updated_story_context.story_summary.strip()
        is_contaminated = (
            proposed_summary.startswith("*")
            or proposed_summary.startswith("-")
            or any(k in proposed_summary for k in ["Role: Editor", "Task: Summarize", "Constraint", "Draft 1:"])
        )

        if len(proposed_summary) < MIN_SUMMARY_LENGTH or is_contaminated:
            if is_contaminated:
                logger.warning(
                    "Proposed story summary contains thinking/scratchpad contamination. Retaining previous summary."
                )
            else:
                logger.warning(
                    "Proposed story summary is too short (%d chars). Retaining previous summary.",
                    len(proposed_summary),
                )
            new_context = response.updated_story_context.model_copy(
                update={"story_summary": context.story_summary}
            )
            return response.model_copy(update={"updated_story_context": new_context})

        return response

    @classmethod
    def validate_location(
        cls, response: GameResponse, context: StoryContext
    ) -> GameResponse:
        """Ensure current location is non-empty, falling back to local location.

        Args:
            response: Candidate GameResponse.
            context: Verified local StoryContext.

        Returns:
            GameResponse with guaranteed valid location.
        """
        proposed_loc = response.updated_story_context.current_location.strip()
        if not proposed_loc:
            logger.warning("Empty location proposed by model. Retaining previous location.")
            new_context = response.updated_story_context.model_copy(
                update={"current_location": context.current_location}
            )
            return response.model_copy(update={"updated_story_context": new_context})

        return response

    @classmethod
    def validate_response(
        cls, response: GameResponse, context: StoryContext
    ) -> GameResponse:
        """Execute the complete Guardian validation suite against a GameResponse.

        Args:
            response: Proposed GameResponse from parser.
            context: Verified authoritative StoryContext.

        Returns:
            Fully validated and consistency-enforced GameResponse.
        """
        validated = cls.validate_dead_characters(response, context)
        validated = cls.validate_location(validated, context)
        validated = cls.validate_summary(validated, context)
        return validated
