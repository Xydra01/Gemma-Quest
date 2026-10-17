# Design Doc: **Gemma Dungeon** – Narrative-First TUI Text Adventure

## 1. Executive Summary
**Gemma Dungeon** is a browser-based or terminal-based text adventure where **Gemma 4** acts as the **Dungeon Master**. Unlike traditional games with rigid stats (HP, XP), this project focuses on **narrative coherence**, **reactive storytelling**, and **character consistency**. The interface is a retro DOS-style TUI, emphasizing atmosphere and immersion over mechanical complexity.

**Core Philosophy:** *The AI is the Storyteller, not a Game Engine.*  
We prioritize consistent plot progression, memorable NPCs, and atmospheric writing over managing complex game states.

---

## 2. Core Features & Scope

### ✅ In Scope (Must-Have for Demo)
1. **Retro TUI Interface:** Terminal-style UI with green-on-black aesthetics, ASCII borders, and monospace fonts.
2. **Narrative-First State Management:**
   - Tracks `Current Location`, `Active Characters` (status), `Key Items`, and a `Rolling Story Summary`.
   - *No HP/XP systems.* Focus on plot-relevant state only.
3. **Dynamic Story Generation:** Gemma 4 generates narration, NPC dialogue, and outcomes based on player input.
4. **Choice Engine:** Players select from AI-generated options (numbered) or type free-form actions.
5. **Consistency Enforcement:** The model is prompted to strictly adhere to the provided `Story Context` (e.g., dead characters stay dead).

### 🚀 Nice-to-Have (If Time Permits)
1. **ASCII Art Headers:** Dynamic ASCII art for major locations (e.g., Castle, Forest).
2. **Sound Effects:** Subtle terminal beeps/blips via Web Audio API (if web-based) or terminal bell (if CLI).
3. **"DOS" Boot Sequence:** A fake DOS startup animation when the game begins.
4. **Character Profiles:** Brief descriptions of NPCs that update as their relationship with the player changes.

### ❌ Out of Scope
- Character creation screen (start with a default hero).
- Multiplayer support.
- Complex combat systems (keep it narrative-driven).
- Mobile app deployment.

---

## 3. Technical Architecture

### Option A: Pure Python CLI (Recommended)
*Best for speed of development and authentic DOS feel.*

```
┌─────────────────────┐       ┌──────────────────────┐       ┌─────────────────────┐
│   User Terminal     │       │    Python TUI App    │       │     Gemma 4         │
│                     │──────▶│  (Rich/Textual)      │──────▶│  (Ollama/Local)     │
│  • Type Choices     │◀─────│  • Render UI        │◀─────│  • Local Inference  │
│  • See Narration    │      │  • Narrative State  │       │  • Structured JSON  │
└─────────────────────┘      └──────────────────────┘       └─────────────────────┘
```

- **Frontend/Backend:** Single Python script.
- **TUI Library:** **Rich** (for beautiful terminal output).
- **LLM Integration:** **Ollama Python Client**.
- **State Management:** Pydantic models + JSON serialization.

### Option B: Web-Based TUI (If Remote Access is Critical)
*Use xterm.js in a browser.*

- **Frontend:** Vanilla JS/React with **xterm.js**.
- **Backend:** FastAPI + WebSockets.
- **Pros:** Shareable via URL.
- **Cons:** More complex setup than Python CLI.

**Recommendation:** Use **Option A (Python CLI with Rich)**. It is faster to build, has a more authentic DOS feel, and requires less infrastructure for a hackathon demo.

---

## 4. Data Model: Narrative State

We replace mechanical game states with **Narrative Context**. This object is passed to the LLM every turn to ensure coherence.

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class StoryContext(BaseModel):
    """
    The core memory of the story. Passed to Gemma 4 to maintain consistency.
    """
    current_location: str = "The Rusty Gate"

    # Track key NPCs and their status relative to the player
    active_characters: Dict[str, str] = {} 
    # Example: {"Goblin King": "Hostile", "Old Mage": "Ally", "Squirrel": "Dead"}

    # Only track items that significantly impact the plot
    key_items_held: List[str] = [] 
    # Example: ["Map to Castle", "Goblin Key"]
    
    # A 3-4 sentence summary of recent events (The Rolling Memory)
    story_summary: str = "You are waking up in a dark cell."
    
    # Optional: World tone or rules if needed for specific scenarios
    world_tone: str = "Dark Fantasy"

class GameResponse(BaseModel):
    """
    The expected output from Gemma 4.
    """
    narration: str
    choices: List[str]
    updated_story_context: StoryContext
```

---

## 5. Prompt Design & LLM Strategy

The prompt is designed to force the model to **reference** the context rather than just generate free text. This is critical for small models (E2B/E4B) on limited hardware.

### System Prompt (Fixed)
```text
You are the Dungeon Master of "Gemma Dungeon," a retro text adventure. Your primary goal is to maintain a coherent, engaging, and consistent story based on the provided Story Context.

RULES:
1. NEVER contradict the Story Summary or Active Characters status. If a character is marked "Dead", they cannot appear alive again.
2. Keep narration atmospheric but concise (2-3 sentences). Write for a terminal interface.
3. Present exactly 2-4 clear narrative choices based on the current situation.
4. Update the Story Context accurately after each turn. Only add to key_items_held if an item is crucial to the plot.
5. Output MUST be valid JSON matching the schema provided.

[STORY CONTEXT]
Location: {{story_context.current_location}}
Active Characters: {{story_context.active_characters}}
Key Items Held: {{', '.join(story_context.key_items_held)}}
Recent Story Summary: {{story_context.story_summary}}

[PLAYER ACTION]
{{player_input}}

[OUTPUT FORMAT]
{
  "narration": "string",
  "choices": ["string", "string"],
  "updated_story_context": {
    "current_location": "string",
    "active_characters": {"Name": "Status"},
    "key_items_held": ["item1"],
    "story_summary": "3-4 sentence summary of what just happened"
  }
}
```

### Post-Turn Memory Update (Optional but Recommended)
If the `updated_story_context.story_summary` from the JSON response is too long or vague, you can make a second, lightweight call to Gemma 4 to refine it:

```text
Summarize the following interaction in exactly 3 sentences. Focus on plot changes, new locations, and character status changes. Output only the summary string.

Interaction: {{last_narration + player_input}}
Current Summary: {{current_summary}}
```

---

## 6. TUI Design & Aesthetic (DOS Style)

### 6.1 Visual Style
- **Color Palette:**
  - Background: `#000000` (Black)
  - Text: `#00FF00` (Bright Green)
  - Highlights/Choices: `#FFFF00` (Yellow)
  - Errors/Negative Events: `#FF0000` (Red)
- **Font:** Monospace (Consolas, Courier New, or VT323 if embedded).
- **Borders:** Use Unicode box-drawing characters (`╔`, `╗`, `═`, `║`) to frame the screen.

### 6.2 Layout Example (Rich Library)
```text
╔══════════════════════════════════════════════╗
║   GEMMA DUNGEON v1.0                         │
╠══════════════════════════════════════════════╣
║                                              │
║  [LOCATION] The Rusty Gate                  │
║  [CHARACTERS] Goblin King (Hostile)         │
║  [ITEMS] Map to Castle                      │
║                                              │
║  You stand before a massive iron door.      │
║  The lock is rusted, but the hinges creak.  │
║                                              │
║  [1] Pick the lock                          │
║  [2] Search for another way                 │
║  [3] Type your own action...                │
║                                              │
╚══════════════════════════════════════════════╝
```

### 6.3 Interactive Elements
- **Numbered Choices:** Press `1`, `2`, or `3` to select.
- **Free Text:** Press `0` or type directly for custom actions.
- **Clear Screen:** Use ANSI escape codes or Rich’s `clear()` to reset the terminal between major scene changes (optional, but adds drama).

---

## 7. Implementation Sketch (Python + Rich)

```python
import ollama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import json

console = Console()

class Game:
    def __init__(self):
        self.context = StoryContext(
            current_location="The Rusty Gate",
            active_characters={},
            key_items_held=[],
            story_summary="You wake up in a dark cell. The bars are rusted."
        )
    
    def render_state(self):
        table = Table(title=self.context.current_location, border_style="cyan")
        table.add_column("Characters", style="yellow")
        table.add_column("Items", style="green")

        chars_str = ", ".join([f"{k} ({v})" for k, v in self.context.active_characters.items()]) or "None"
        items_str = ", ".join(self.context.key_items_held) or "None"

        table.add_row(chars_str, items_str)
        
        return Panel(
            Group(table),
            title="GEMMA DUNGEON",
            border_style="green",
            subtitle=f"[italic]Summary: {self.context.story_summary}[/italic]"
        )

    def build_prompt(self, player_input):
        # Serialize context to string for prompt injection
        ctx_str = f"""Location: {self.context.current_location}
Active Characters: {self.context.active_characters}
Key Items Held: {', '.join(self.context.key_items_held)}
Recent Story Summary: {self.context.story_summary}

Player Action: {player_input}"""

        return f"System Prompt... [STORY CONTEXT] {ctx_str} ..." # Use the prompt template from section 5

    def run(self):
        while True:
            console.print(self.render_state())
            
            # Get player input
            choices = ["Pick the lock", "Search for another way"] # Placeholder for demo
            choice_str = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(choices)])
            user_input = console.input(f"\n{choice_str}\n[0] Type action: ").strip()
            
            if not user_input: continue

            # Call LLM
            prompt = self.build_prompt(user_input)
            response = ollama.chat(model="gemma4:2b-q4_0", messages=[{"role": "user", "content": prompt}])
            
            # Parse JSON
            try:
                data = json.loads(response["message"]["content"])
                new_context = StoryContext(**data["updated_story_context"])
                self.context = new_context

                # Display Narration
                console.print(f"\n[bold green]{data['narration']}[/bold green]\n")
                
            except Exception as e:
                console.print(f"[red]Error parsing LLM response. Retrying...[/red]")

if __name__ == "__main__":
    game = Game()
    game.run()
```

---

## 8. Development Plan (1-Day Timeline)

| Time Block | Task | Deliverable |
|------------|------|-------------|
| **Hour 1** | Setup: Ollama + Gemma 4 E2B/E4B, Python env, install `rich` and `ollama`. | Working LLM call in terminal. |
| **Hour 2-3** | Build Prompt Templates + Pydantic models. Test consistency manually with a script. | Coherent story generation in CLI. |
| **Hour 4-5** | Implement Rich TUI: Panels, Tables, Input handling. | Functional game loop without LLM integration. |
| **Hour 6-7** | Connect TUI ↔ LLM. Handle loading states (e.g., "Thinking..."). | Playable end-to-end flow. |
| **Hour 8** | Polish: Add ASCII art headers, color codes for NPCs. | Visually appealing DOS-style UI. |
| **Hour 9** | Test: Play 3–5 scenarios. Fix edge cases (invalid JSON, dead character revival). | Stable demo. |
| **Hour 10-12** | Demo Prep: Record terminal video, prepare README. | Final submission ready. |

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Gemma 4 returns invalid JSON** | High | Medium | Pydantic validation + retry logic + fallback narration. |
| **Inconsistent story (e.g., dead character revives)** | Medium | High | Strict prompt rules: "If a character is marked Dead, they cannot appear alive." |
| **Slow LLM response (>10s)** | Medium | Medium | Show loading animation in terminal ("The dungeon hums with magic..."). Use E2B if E4B is too slow. |
| **Player inputs nonsensical actions** | High | Low | Prompt Gemma 4 to respond in-character with consequences. No need for strict validation. |

---

## 10. Demo Script (For Judges)

1.  **Boot Sequence:**
    ```text
    > Loading Gemma Dungeon...
    > Initializing Terminal...
    > Welcome, Adventurer.
    ```
2.  **Game Start:**
    - Green text appears in a bordered panel.
    - Narration: "You wake in a dark cell."
3.  **Choice Selection:**
    - Press `1` to pick the lock.
4.  **Twist:**
    - Goblin appears. Player types custom action: `0` then "Ask the goblin for help".
5.  **Coherence Test:**
    - Player kills the goblin. Later, tries to ask it for help again.
    - Gemma 4 responds: *"The goblin is dead. Its bones are picked clean by ravens."* **(Proves consistency)**
6.  **End:**
    - Game continues or ends based on story flags.

**Key Selling Point:** “It feels like playing a game from 1985, but the dungeon is generated by an AI that remembers everything you’ve done.”