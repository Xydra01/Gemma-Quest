# Technical Design Document: Narrative State & Data Models

## 1. Overview
This document defines the core data structures that drive the game's narrative consistency. Unlike traditional RPGs that track numerical stats (HP, XP), this system tracks **Narrative Context**. This object is the single source of truth for the AI and is serialized/passed to the LLM (Gemma 4) on every turn to ensure coherence, character consistency, and plot logic.

## 2. Design Philosophy: "The Rolling Memory"

The LLM has a finite context window. We cannot pass the entire history of every sentence ever generated. Instead, we use a **Rolling Story Summary**.

*   **Mechanism:** Every turn (or every N turns), the system summarizes the last few events into a concise 3-4 sentence paragraph.
*   **Goal:** This summary replaces raw chat history in the prompt to save tokens while retaining plot-critical information (e.g., "The player killed the Goblin King" is more important than "The player swung his sword three times").

## 3. Core Data Models (Pydantic)

### 3.1 `StoryContext`
This is the primary state object. It represents the "current reality" of the game world as understood by the AI.

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum

class CharacterStatus(Enum):
    """Standardized states for NPCs to ensure consistency."""
    ALIVE = "Alive"
    DEAD = "Dead"
    HOSTILE = "Hostile"
    ALLY = "Ally"
    NEUTRAL = "Neutral"
    UNKNOWN = "Unknown"

class StoryContext(BaseModel):
    """
    The core memory of the story. 
    Passed to Gemma 4 every turn to maintain consistency.
    """
    
    # --- World State ---
    current_location: str = Field(
        default="The Rusty Gate",
        description="The specific physical location. Must be a noun phrase."
    )
    world_tone: str = Field(
        default="Dark Fantasy",
        description="High-level genre/tone modifier (e.g., 'Noir', 'Sci-Fi') to steer narrative voice."
    )

    # --- Character State ---
    active_characters: Dict[str, CharacterStatus] = Field(
        default_factory=dict,
        description="Map of NPC Name to Status. Only include NPCs who have interacted with the player or are currently relevant."
    )
    
    # Example: 
    # {
    #   "Goblin King": CharacterStatus.DEAD,
    #   "Old Mage": CharacterStatus.ALLEY,
    #   "Squirrel": CharacterStatus.DEAD
    # }

    # --- Inventory (Plot-Only) ---
    key_items_held: List[str] = Field(
        default_factory=list,
        description="List of items that significantly impact the plot. Avoid tracking consumables like 'bread' unless it's a quest item."
    )
    
    # Example: ["Map to Castle", "Goblin Key"]

    # --- Narrative Memory ---
    story_summary: str = Field(
        default="You are waking up in a dark cell.",
        description="A 3-4 sentence summary of recent events. This is the 'Rolling Memory' that prevents context window bloat."
    )
    
    # --- Meta Data ---
    turn_count: int = 0
```

### 3.2 `GameResponse`
This defines the expected output structure from the LLM. We force the LLM to return JSON so we can programmatically update our state.

```python
class GameResponse(BaseModel):
    """
    The expected output from Gemma 4.
    The system prompt instructs the model to end its response with this JSON structure.
    """

    narration: str = Field(
        description="The descriptive text shown to the user. Should be evocative and atmospheric."
    )

    choices: List[str] = Field(
        min_length=2,
        max_length=4,
        description="A list of 2-4 specific actions or dialogues the player can take. Must be distinct from each other."
    )
    
    updated_story_context: StoryContext = Field(
        description="The new state of the world after this turn's events. The model must update locations, items, and character statuses based on the narration."
    )
```

### 3.3 `PromptBuilder` Interface
A helper class to construct the final prompt string from these models.

```python
class PromptBuilder:
    @staticmethod
    def build_system_prompt(context: StoryContext) -> str:
        """Constructs the system message including all state constraints."""
        
        # Format characters for the prompt
        char_list = []
        for name, status in context.active_characters.items():
            char_list.append(f"- {name}: {status.value}")
        chars_str = "\n".join(char_list) if char_list else "None currently relevant."

        # Format items
        items_str = ", ".join(context.key_items_held) if context.key_items_held else "None"

        return f"""
You are a retro text adventure game engine.
Current Game State:
- Location: {context.current_location}
- Tone: {context.world_tone}
- Active Characters:
{chars_str}
- Key Items Held: {items_str}
- Story Summary (Recent Events): {context.story_summary}

Rules:
1. Maintain the tone of a 1980s terminal RPG.
2. Dead characters must remain dead. Do not resurrect them.
3. If an item is used, remove it from 'Key Items Held'.
4. Respond with narrative text followed by a JSON object in this exact format:
{{"narration": "...", "choices": ["...", "..."], "updated_story_context": {{"current_location": "...", ...}}}}
"""
```

## 4. State Update Logic (The "Sync" Process)

When the LLM returns a `GameResponse`, we must merge its `updated_story_context` with our local state to prevent drift.

### 4.1 Merge Strategy
We do **not** blindly overwrite the local state. We perform a **Delta Update**:

1.  **Location:** Always update `current_location` if it changes.
2.  **Items:** 
    *   If the LLM adds an item, add it to `key_items_held`.
    *   If the LLM removes an item (e.g., "You ate the apple"), remove it from `key_items_held`.
    *   *Safety Check:* Never allow items to be removed if they are critical quest items defined in `config.yaml` (optional advanced feature).
3.  **Characters:** 
    *   Merge dictionaries. If a character status changes, update it.
    *   *Constraint:* If a character is marked `DEAD`, the application logic should flag any future prompt where that character acts as alive, and reject/retry the LLM response.
4.  **Summary:** Update `story_summary` with the new text provided by the LLM.

### 4.2 Pseudocode for State Sync

```python
def sync_state(local_context: StoryContext, ai_response: GameResponse) -> StoryContext:
    updated = local_context.copy()
    
    # 1. Update Location
    if ai_response.updated_story_context.current_location != "":
        updated.current_location = ai_response.updated_story_context.current_location

    # 2. Sync Items (Add new, remove used)
    ai_items = set(ai_response.updated_story_context.key_items_held)
    local_items = set(updated.key_items_held)

    # Keep items that are in AI response OR were critical and not mentioned as used
    # For simplicity: Trust the AI's list but filter out invalid chars
    updated.key_items_held = [item for item in ai_response.updated_story_context.key_items_held if len(item) < 50]

    # 3. Sync Characters
    for name, status in ai_response.updated_story_context.active_characters.items():
        updated.active_characters[name] = status

    # 4. Update Summary (Rolling Memory)
    updated.story_summary = ai_response.updated_story_context.story_summary
    updated.turn_count += 1

    return updated
```

## 5. Consistency Enforcement Mechanisms

To strictly adhere to the "Consistency Enforcement" requirement:

### 5.1 The "Dead Man's Switch"
In the `PromptBuilder`, we explicitly list dead characters.
*   *Prompt:* "The following characters are DEAD and cannot speak or act: [List of Dead Chars]."
*   *Validation:* If the parser detects a dead character appearing in `choices` (e.g., `[1] Talk to Goblin King`), the UI automatically filters it out or replaces it with `[1] Look at the body`.

### 5.2 Context Window Management
If `turn_count > 10`, we do not just pass the `story_summary`. We also pass a **Condensed History** of the last 3 major plot points to help the LLM understand *why* things happened, preventing logical leaps.

## 6. Serialization & Persistence

`StoryContext` is fully serializable to JSON via Pydantic. This allows for:
1.  **Save/Load:** The entire game state can be saved as a single `.json` file.
2.  **Debugging:** Developers can inspect the current context at any time to see what the AI "thinks" is true.

```python
# Saving
state_data = local_context.model_dump_json(indent=2)
with open("savegame.json", "w") as f:
    f.write(state_data)

# Loading
loaded_context = StoryContext.model_validate_json(json_string)
```

## 7. Edge Cases & Validation

| Scenario | Handling |
| :--- | :--- |
| **LLM returns invalid JSON** | Catch exception, log error. Fall back to using `narration` as raw text and generate default choices `[1] Wait`, `[2] Look around`. Do NOT update state. |
| **LLM resurrects a dead character** | Detect in post-processing. If `active_characters[name] == DEAD` but narration implies they are acting, inject a correction prompt: *"You made a mistake. The Goblin King is dead. Please rewrite the scene."* |
| **Empty Story Summary** | If the AI returns an empty summary, retain the previous one to avoid losing context. |
| **Location Mismatch** | If `current_location` remains unchanged for 5 turns but the narrative implies movement, prompt the AI: *"Update your location state based on recent actions."* |

## 8. Example Data Flow

**Turn 1:**
*   *Input:* Player chooses `[2] Break the window`.
*   *AI Output JSON:*
    ```json
    {
      "narration": "Glass shatters... You escape into the forest.",
      "choices": ["Run to the castle", "Hide in bushes"],
      "updated_story_context": {
        "current_location": "The Forest Edge",
        "key_items_held": [],
        "active_characters": {},
        "story_summary": "You escaped your cell and broke into the forest."
      }
    }
    ```
*   *System Action:* Update `current_location` to "The Forest Edge". Save state. Render UI.