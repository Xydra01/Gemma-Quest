"""Delta state reconciliation and narrative consistency enforcement."""

import logging
from typing import Optional, Set
from .models import CharacterStatus, GameResponse, StoryContext

logger = logging.getLogger(__name__)

# Maximum length for an item string to prevent hallucinated walls of text
MAX_ITEM_NAME_LENGTH = 50
# Minimum length for a valid narrative summary to avoid empty context
MIN_SUMMARY_LENGTH = 10


def sync_state(
    local_context: StoryContext,
    ai_response: GameResponse,
    protected_items: Optional[Set[str]] = None,
) -> StoryContext:
    """Reconciles local story context with an AI-generated turn response.

    Applies delta updates and enforces narrative consistency invariants:
    - Dead characters remain dead (Dead Man's Switch).
    - Item sanitization and retention of critical items.
    - Fallback preservation of story summary when AI provides empty text.
    - Incrementation of turn counter.

    Args:
        local_context: The verified, authoritative current game state.
        ai_response: The parsed structured response received from Gemma 4.
        protected_items: Optional set of critical quest items that cannot be dropped.

    Returns:
        A new, validated StoryContext instance representing the new game state.
    """
    proposed_context = ai_response.updated_story_context

    # 1. Location Update
    new_location = proposed_context.current_location.strip()
    updated_location = (
        new_location if new_location else local_context.current_location
    )

    # 2. Key Items Reconciliation & Sanitization
    sanitized_items = []
    seen_items = set()

    for item in proposed_context.key_items_held:
        cleaned_item = item.strip()
        if cleaned_item and len(cleaned_item) < MAX_ITEM_NAME_LENGTH:
            if cleaned_item.lower() not in seen_items:
                seen_items.add(cleaned_item.lower())
                sanitized_items.append(cleaned_item)

    # Protect critical quest items if specified
    if protected_items:
        for protected in protected_items:
            # If the player previously had it and it's missing, retain it
            if any(i.lower() == protected.lower() for i in local_context.key_items_held):
                if not any(i.lower() == protected.lower() for i in sanitized_items):
                    sanitized_items.append(protected)
                    seen_items.add(protected.lower())

    # 3. Character Status Sync & Narrative Invariants
    updated_characters = dict(local_context.active_characters)

    for name, proposed_status in proposed_context.active_characters.items():
        clean_name = name.strip()
        if not clean_name:
            continue

        current_status = updated_characters.get(clean_name)

        # Invariant: Dead characters cannot be resurrected by the model
        if current_status == CharacterStatus.DEAD and proposed_status != CharacterStatus.DEAD:
            logger.warning(
                "Prevented resurrection of dead character '%s' (attempted: %s)",
                clean_name,
                proposed_status.value,
            )
            # Retain DEAD status
            updated_characters[clean_name] = CharacterStatus.DEAD
        else:
            updated_characters[clean_name] = proposed_status

    # 4. Rolling Story Summary Fallback
    new_summary = proposed_context.story_summary.strip()
    is_contaminated = (
        new_summary.startswith("*")
        or new_summary.startswith("-")
        or any(k in new_summary for k in ["Role: Editor", "Task: Summarize", "Constraint", "Draft 1:"])
    )
    if len(new_summary) >= MIN_SUMMARY_LENGTH and not is_contaminated:
        updated_summary = new_summary
    else:
        if is_contaminated:
            logger.warning("Proposed summary contains thinking/scratchpad artifacts. Retaining previous summary.")
        else:
            logger.warning("Proposed summary too short/empty. Retaining previous summary.")
        updated_summary = local_context.story_summary

    # 5. World Tone
    updated_tone = (
        proposed_context.world_tone.strip() or local_context.world_tone
    )

    return StoryContext(
        current_location=updated_location,
        world_tone=updated_tone,
        active_characters=updated_characters,
        key_items_held=sanitized_items,
        story_summary=updated_summary,
        turn_count=local_context.turn_count + 1,
    )
