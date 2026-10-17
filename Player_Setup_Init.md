# Technical Design Document: Onboarding & Initialization Flow

## 1. Overview
This document defines the technical architecture for the **Player Setup & Initialization** phase of the game. Unlike a traditional character select screen, this module focuses on gathering minimal user inputs to seed the AI's narrative context and generate a consistent starting state.

## 2. Design Goals
*   **Low Friction:** Minimize input fields to keep the "retro" feel fast and immediate.
*   **Context Seeding:** Ensure all player inputs are explicitly mapped into the system prompt for the LLM (Gemma 4) to guarantee narrative consistency from turn one.
*   **Deterministic Start:** Use the inputs to deterministically generate or select the initial "Story Context" object before handing off to the main game loop.

## 3. User Flow & Input Schema

The onboarding sequence will consist of three simple, linear prompts in the TUI. No graphical menus; just text-based input fields with ASCII borders.

### Step 1: Player Identity
*   **Prompt:** `> ENTER YOUR NAME:`
*   **Input Type:** String
*   **Validation:**
    *   Trim whitespace.
    *   Sanitize HTML/JS characters (prevent prompt injection attempts via name field).
    *   Default fallback: `"Hero"` if empty or invalid.

### Step 2: Setting Selection
*   **Prompt:** `> SELECT SETTING [1] Fantasy [2] Sci-Fi [3] Cyberpunk [4] Custom:`
*   **Input Type:** Enum / String
*   **Validation:**
    *   If numeric (1-3): Map to predefined setting templates.
    *   If text: Accept free-form custom setting, append to `user_custom_context`.

### Step 3: Initial Goal/Motivation (Optional)
*   **Prompt:** `> WHAT IS YOUR FIRST GOAL? (optional)`
*   **Input Type:** String
*   **Validation:**
    *   Optional. If empty, the AI will generate a generic goal based on the setting.

## 4. Technical Architecture

### 4.1 Data Model: `PlayerSetup`
This object is created during onboarding and persists as part of the global `GameState`.

```typescript
interface PlayerSetup {
  playerName: string;
  settingType: 'fantasy' | 'scifi' | 'cyberpunk' | 'custom';
  customSettingDetails?: string; // Only if settingType === 'custom'
  initialGoal: string;
  timestamp: number;
}
```

### 4.2 Context Injection Pipeline

The core technical challenge is ensuring the LLM remembers these choices. We will implement a **Context Builder** function that runs once during initialization.

#### A. System Prompt Template (Initial)
The `PlayerSetup` data is interpolated into the system prompt *before* the first user turn.

```text
You are a retro text adventure game engine.
Current Game State:
- Player Name: {{playerName}}
- Setting: {{settingDescription}} // e.g., "High Fantasy Medieval" or "2145 Neo-Tokyo"
- Player's Current Goal: {{initialGoal}}

Constraints:
1. Maintain the tone of a 1980s terminal RPG.
2. Never break character.
3. Keep track of all NPCs mentioned and their status (alive/dead).
```

#### B. Setting Templates (Static Data)
To ensure consistency, we use static templates for predefined settings to reduce token usage and improve coherence.

*   **Fantasy:** "A mist-shrouded medieval kingdom where magic is fading."
*   **Sci-Fi:** "A sprawling space station drifting in the void of a dying star system."
*   **Cyberpunk:** "A neon-drenched metropolis ruled by corporate warlords and rogue AIs."

### 4.3 Initialization Sequence (Pseudocode)

```javascript
async function initializeGame() {
  // 1. Display DOS Boot Sequence (Nice-to-have)
  await renderDOSBoot();

  // 2. Gather Input
  const name = await prompt("ENTER YOUR NAME:");
  const settingChoice = await prompt("SELECT SETTING [1-4]:");
  const goal = await prompt("WHAT IS YOUR FIRST GOAL? (optional):");

  // 3. Construct PlayerSetup Object
  const setup = {
    playerName: sanitize(name) || "Hero",
    settingType: mapSettingChoice(settingChoice),
    initialGoal: goal || generateDefaultGoal(setup.settingType)
  };

  // 4. Build Initial Context String
  const systemPrompt = buildSystemPromptFromSetup(setup);

  // 5. Generate Opening Scene (First AI Turn)
  // We send a specific "START" command to the LLM to generate the intro
  const openingNarrative = await gemma4.generate({
    system: systemPrompt,
    userMessage: "Begin the story. Describe the player's immediate surroundings and introduce one key NPC. Present 3 choices."
  });

  // 6. Transition to Main Game Loop
  gameState.currentContext = { 
    setup, 
    history: [openingNarrative], 
    activeCharacters: extractCharacters(openingNarrative) 
  };
  
  renderMainUI(gameState);
}
```

## 5. State Management Integration

The `PlayerSetup` is not stored in a separate table; it is embedded into the **`Rolling Story Summary`** which is part of the global state.

*   **Why?** The requirement states: *"Tracks Current Location, Active Characters, Key Items, and a Rolling Story Summary."*
*   **Implementation:** The `initialGoal` and `settingType` are injected into the first entry of the `Rolling Story Summary`. As the game progresses, this summary is summarized/condensed by the LLM to keep token counts low, but the *initial* context remains the anchor.

## 6. Consistency Enforcement (Prompt Strategy)

To ensure the "dead characters stay dead" and setting consistency:
1.  **Explicit State Block:** The system prompt will always include a `<STATE>` block at the start of every turn.
2.  **Initialization Flag:** The first prompt includes a special instruction: *"You are starting a new game. Establish the world based on the provided Setup. Do not reference any previous games."*

## 7. Edge Cases & Error Handling

| Scenario | Handling |
| :--- | :--- |
| **User enters invalid setting** | Loop prompt until valid selection (1-4) or text is entered. |
| **User enters empty name** | Default to "Hero". Display a subtle ASCII blink effect if they leave it blank. |
| **LLM ignores setup** | Implement a post-processing check: If the opening narrative does not contain the `playerName` or match the `settingType` keywords, retry generation with a stricter prompt: *"You failed to include the player name. Try again."* |

## 8. UI/UX Specifications (Retro TUI)

*   **Visuals:**
    *   Use `\u003c=>` for input cursor.
    *   Frame inputs with ASCII box drawing characters: `+---+`.
    *   Blinking underscore `_` for input focus.
*   **Flow:**
    1.  Screen clears (or scrolls up).
    2.  Prompt appears at top center.
    3.  User types, hits Enter.
    4.  Input is echoed in gray/dim text above the prompt.
    5.  Next prompt appears below.

## 9. Future Enhancements (Nice-to-Have)

*   **Character Profile Preview:** After setup, show a small ASCII portrait generated from the name hash (deterministic).
*   **Save Slot Integration:** The `PlayerSetup` JSON becomes the seed for the save file header.