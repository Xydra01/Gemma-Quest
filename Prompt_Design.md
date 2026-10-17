# Technical Design Document: Prompt Engineering & LLM Strategy

## 1. Overview
This document details the prompt engineering strategy for interacting with Gemma 4 (or similar small language models). Because we are targeting local inference on limited hardware, we must minimize token usage and maximize instruction-following reliability. The core philosophy is **"Context-First" prompting**: forcing the model to ground its generation in the provided `StoryContext` object rather than relying on internal parametric knowledge or free-flowing creativity.

## 2. Design Principles for Small Models (E2B/E4B)

1.  **Constraint Satisfaction:** Small models are prone to "hallucinating" plot holes (e.g., bringing back dead characters). We mitigate this by explicitly listing constraints in the system prompt and validating outputs programmatically.
2.  **Token Efficiency:** We avoid long, open-ended system prompts. The `StoryContext` is injected dynamically. If the context is too large, we truncate or summarize it.
3.  **Structured Output Enforcement:** Free-form text generation from small models is unstable for structured data (JSON). We use a "Chain of Thought" style prompt where the model must *think* through the state update before outputting JSON, or simply enforce strict JSON-only output at the end.
4.  **Role-Playing Consistency:** The persona ("Dungeon Master") is kept simple to reduce cognitive load on the model.

## 3. System Prompt Architecture

The system prompt is divided into three distinct sections: **Persona & Rules**, **Dynamic Context Injection**, and **Output Schema**.

### 3.1 Fixed Persona & Rules Block
This block remains static across all sessions (unless the setting changes). It establishes the "contract" with the LLM.

```text
You are the Dungeon Master of "Gemma Dungeon," a retro text adventure game. 
Your primary goal is to maintain a coherent, engaging, and consistent story based strictly on the provided Story Context.

RULES:
1. COHERENCE: NEVER contradict the Story Summary or Active Characters status. If a character is marked "Dead", they cannot appear alive again. If an item is not in Key Items Held, the player does not have it.
2. STYLE: Keep narration atmospheric but concise (max 3 sentences). Write for a terminal interface (short paragraphs, high impact).
3. CHOICES: Present exactly 2-4 clear narrative choices based on the current situation. Choices must be actionable (e.g., "Open the door", not "Feel sad").
4. STATE MANAGEMENT: You MUST update the Story Context accurately after each turn. Only add to key_items_held if an item is crucial to the plot. Remove items when they are used or lost.
5. OUTPUT FORMAT: Your response MUST end with a valid JSON object matching the schema provided below. Do not include markdown code blocks (```json) around the final JSON.
```

### 3.2 Dynamic Context Injection
This section is populated by the `PromptBuilder` using the current `StoryContext` Pydantic model.

```text
[STORY CONTEXT]
Location: {{current_location}}
Active Characters: 
- {{char_name_1}}: {{status_1}}
- {{char_name_2}}: {{status_2}}
Key Items Held: {{item_1}}, {{item_2}} (or "None")
Recent Story Summary: {{story_summary}}

[PLAYER ACTION]
{{player_input}}
```

### 3.3 Output Schema Definition
We explicitly define the JSON structure to guide the parser.

```text
[OUTPUT FORMAT]
Return ONLY a JSON object with this exact structure:
{
  "narration": "string (2-3 sentences of descriptive text)",
  "choices": ["string", "string"],
  "updated_story_context": {
    "current_location": "string (New location or same as before)",
    "active_characters": {
      "Name": "Status (Alive/Dead/Hostile/Ally/Neutral)"
    },
    "key_items_held": ["item1", "item2"],
    "story_summary": "3-4 sentence summary of what just happened and current plot status"
  }
}
```

## 4. Prompt Assembly Logic (Code Implementation)

The `PromptBuilder` class handles the assembly of these parts into a single string passed to Ollama.

```python
class PromptBuilder:
    def build_full_prompt(self, context: StoryContext, player_input: str) -> str:
        # 1. Format Characters for readability
        chars = []
        for name, status in context.active_characters.items():
            chars.append(f"- {name}: {status.value}")
        chars_str = "\n".join(chars) if chars else "None"

        # 2. Format Items
        items_str = ", ".join(context.key_items_held) if context.key_items_held else "None"

        # 3. Assemble Final Prompt
        prompt = f"""You are the Dungeon Master of "Gemma Dungeon"... [FIXED RULES] ...

[STORY CONTEXT]
Location: {context.current_location}
Active Characters:
{chars_str}
Key Items Held: {items_str}
Recent Story Summary: {context.story_summary}

[PLAYER ACTION]
{player_input}

[OUTPUT FORMAT]
Return ONLY a valid JSON object... [SCHEMA] ...
"""
        return prompt
```

## 5. Post-Turn Memory Update Strategy (The "Refinement" Loop)

Small models often produce `story_summary` fields that are either too long, too vague, or repetitive. To keep the context window efficient and the narrative sharp, we implement an optional **Second-Pass Refinement**.

### 5.1 When to Trigger
*   If `len(updated_story_context.story_summary) > 200 characters`.
*   Every 5 turns (to prevent drift).
*   When a major plot point occurs (e.g., character death, location change).

### 5.2 Refinement Prompt Template
This is a lightweight call designed specifically for summarization, not storytelling.

```text
You are an editor for a text adventure game. 
Summarize the following interaction in exactly 3 sentences. 
Focus ONLY on plot changes, new locations, and character status changes. 
Do not include flavor text or dialogue. Output only the summary string.

[INPUT INTERACTION]
Player Action: {{player_input}}
Narration: {{last_narration}}
Current Summary: {{current_summary}}

[SUMMARY]
```

### 5.3 Implementation Flow
1.  Receive `GameResponse` from main LLM call.
2.  Check summary length/quality heuristics.
3.  If refinement needed, send `Refinement Prompt` to Gemma 4 (temperature set low, e.g., 0.2).
4.  Replace `updated_story_context.story_summary` with the refined output.
5.  Proceed to state sync.

## 6. Consistency Enforcement & Validation Layer

Even with good prompts, small models can fail. We add a **Post-Processing Validator** between the LLM response and the State Sync.

### 6.1 Dead Character Check
```python
def validate_dead_characters(narration: str, choices: List[str], context: StoryContext) -> bool:
    dead_chars = [name for name, status in context.active_characters.items() if status == "Dead"]

    # Simple keyword check (case-insensitive)
    lower_text = narration.lower() + " ".join(choices).lower()
    
    for char in dead_chars:
        if char.lower() in lower_text:
            # If a dead character is mentioned as acting/speaking, flag it.
            # Note: "He died" is okay. "He said hello" is bad.
            # For simplicity, we reject if the name appears in choices.
            if any(char.lower() in choice.lower() for choice in choices):
                return False 
    return True
```

### 6.2 Item Consistency Check
*   If an item is in `key_items_held` but not mentioned in the narration or summary, keep it.
*   If an item is *not* in `key_items_held` but the player tries to use it in the next turn, the UI should reject the action: *"You do not have that item."*

## 7. Temperature & Sampling Settings

| Parameter | Value | Reasoning |
| :--- | :--- | :--- |
| **Temperature** | 0.8 - 1.0 | High enough for creative narration, low enough to prevent total incoherence. |
| **Top-K** | 50 | Limits the sampling pool to top 50 tokens, reducing nonsensical outputs. |
| **Top-P** | 0.9 | Nucleus sampling threshold. |
| **Max Tokens** | 512 - 1024 | Enough for narration + JSON structure. Prevents rambling. |

## 8. Error Handling & Fallback Strategies

| Failure Mode | Detection | Recovery Action |
| :--- | :--- | :--- |
| **Invalid JSON** | `json.loads()` throws exception | **Fallback:** Treat entire response as `narration`. Generate default choices: `["Look around", "Wait"]`. Do **not** update state. Log error. |
| **Missing Fields** | Key fields (`current_location`) missing in JSON | Use previous state values for missing fields. Warn user in debug log. |
| **Incoherent Summary** | Summary is empty or < 10 chars | Retain the previous `story_summary`. Do not update. |
| **Dead Character Revival** | Validator detects dead char in choices/narration | Reject response. Retry generation with an added system prompt: *"You made a mistake. {{dead_char}} is dead. Rewrite the scene without them."* |

## 9. Example: Full Prompt Construction (Turn 2)

**Context:**
*   Location: The Forest Edge
*   Dead: Goblin King
*   Items: None
*   Summary: You escaped your cell and broke into the forest. The Goblin King is dead.

**Player Input:** `[1] Run to the castle`

**Generated Prompt:**
```text
You are the Dungeon Master... [RULES] ...

[STORY CONTEXT]
Location: The Forest Edge
Active Characters:
- Goblin King: Dead
Key Items Held: None
Recent Story Summary: You escaped your cell and broke into the forest. The Goblin King is dead.

[PLAYER ACTION]
[1] Run to the castle

[OUTPUT FORMAT]
Return ONLY a valid JSON object...
```

**Expected LLM Output:**
```json
{
  "narration": "You sprint through the trees, branches tearing at your clothes. The silhouette of the castle looms ahead in the fog.",
  "choices": ["Approach the gates", "Hide behind a tree"],
  "updated_story_context": {
    "current_location": "The Castle Approach",
    "active_characters": {"Goblin King": "Dead"},
    "key_items_held": [],
    "story_summary": "You escaped the cell and fled into the forest. You are now running towards the castle gates."
  }
}
```
