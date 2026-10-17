# Technical Design Document: Core Runtime & TUI Architecture

## 1. Overview
This document details the technical implementation for the main runtime environment of the application. We are adopting **Option A: Pure Python CLI** as the primary architecture. This approach prioritizes development velocity, local-first execution, and an authentic "retro" terminal aesthetic without the overhead of a web server.

## 2. System Architecture Diagram

```text
┌─────────────────────┐       ┌───────────────────────────┐       ┌─────────────────────┐
│    User Terminal     │       │   Python TUI Application  │       │      Gemma 4        │
│                     │       │                         │       │                     │
│  • Type Choices      │────▶ │  • Render UI (Rich)      │────▶ │  • Local Inference  │
│  • See Narration     │◀──── │  • Narrative State Mgmt  │◀──── │  • Structured JSON  │
│  • ASCII Art         │       │  • Choice Engine Logic   │       │  • System Prompts   │
└─────────────────────┘       └───────────────────────────┘       └─────────────────────┘
```

## 3. Technology Stack & Dependencies

| Component | Library/Tool | Version (Target) | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.10+ | Core logic, state management, I/O. |
| **TUI Engine** | `rich` / `textual` | Latest Stable | Rendering ASCII borders, colors (green-on-black), live updates, input handling. |
| **LLM Client** | `ollama-python` | 0.2+ | Communicating with the local Ollama server running Gemma 4. |
| **State Models** | `pydantic` | 2.0+ | Data validation, serialization of game state to JSON. |
| **Async I/O** | `asyncio` | Built-in | Handling non-blocking LLM calls and UI updates simultaneously. |

## 4. Module Breakdown

### 4.1 The TUI Layer (`ui/`)
Responsible for all visual output and user input capture.

*   **Library Choice:** We will use **`rich`** for static components (headers, borders, logs) and **`textual`** if complex interactive widgets (like dynamic scrolling lists) are needed. *Note: For a simple text-based flow, `rich` alone may suffice with standard `input()` or raw terminal reading.*
*   **Theming:**
    *   Define a custom `Theme` object in Rich.
    *   Primary Color: `#00FF00` (Green).
    *   Background: `#000000` (Black).
    *   Accent: `#80FF80` (Light Green) for highlights/cursors.
*   **Components:**
    *   `StoryLog`: A scrollable panel displaying the narrative history.
    *   `InputPrompt`: A single-line input field at the bottom with a blinking cursor (`▌`).
    *   `StatusHeader`: Displays `Location`, `Character`, and `Items` in a compact ASCII bar.

### 4.2 The Narrative Engine (`engine/`)
The brain of the application that interfaces with the LLM.

*   **LLM Client Wrapper:**
    *   Encapsulates Ollama API calls.
    *   Handles retries, timeout management, and streaming responses (token-by-token display).
*   **Prompt Builder:**
    *   Function: `build_context(state: GameState) -> str`
    *   Assembles the System Prompt + Story Summary + Current Location + User Input into a single JSON or text block for Gemma 4.
*   **Output Parser:**
    *   The LLM is instructed to return structured JSON (see Section 5).
    *   Parser extracts: `narrative_text`, `choices[]`, `state_updates` (new items, location changes).

### 4.3 State Management (`state/`)
*   **Model:** Pydantic Base Class `GameState`.
*   **Fields:**
    ```python
    class GameState(BaseModel):
        current_location: str
        active_characters: List[Character]
        key_items: List[str]
        story_summary: str # Rolling summary for context window management
        turn_count: int
    ```
*   **Persistence:**
    *   `save_game(path)`: Serialize `GameState` to JSON.
    *   `load_game(path)`: Deserialize and validate.

## 5. Data Exchange Protocol (LLM ↔ App)

To ensure the "Choice Engine" works reliably, we enforce a strict output format from Gemma 4. The system prompt will instruct the model to **always** end its response with a JSON block.

### Example Prompt Instruction
> "Respond with narrative text followed by a JSON object in this exact format:
> `{\"narrative\": \"...\", \"choices\": [\"Option 1\", \"Option 2\"], \"state_updates\": {\"location\": \"New Place\", \"items_added\": [\"Sword\"]}}`"

### Parsing Logic (Python)
```python
import json

def parse_llm_response(response_text: str):
    # Simple regex or split to find the last JSON object
    try:
        json_start = response_text.rfind('{')
        json_end = response_text.rfind('}') + 1
        data = json.loads(response_text[json_start:json_end])
        
        narrative = data.get("narrative", "")
        choices = data.get("choices", [])
        updates = data.get("state_updates", {})
        
        return narrative, choices, updates

    except Exception as e:
        # Fallback: Treat entire response as narrative if JSON fails
        return response_text, [], {}
```

## 6. Async Execution Flow

To prevent the UI from freezing during LLM inference (which can take seconds):

1.  **User Input:** Captured via `input()` or Textual event loop.
2.  **Task Creation:** An `asyncio.Task` is created for the LLM call.
3.  **UI Update:** The TUI displays a "Thinking..." spinner (ASCII) in the status bar.
4.  **Streaming:** If Ollama supports streaming, tokens are yielded to the UI as they arrive (typewriter effect).
5.  **Completion:** Once the response is complete and parsed, the UI updates with the narrative and renders the numbered choices.

## 7. Error Handling & Resilience

*   **LLM Timeout:** If no token received within 30s, show `ERROR: Connection lost to AI engine`. Allow user to retry or quit.
*   **Invalid JSON:** If the LLM fails to provide valid JSON (common with creative models), fall back to treating the output as pure text and generate default choices (`[1] Look around`, `[2] Wait`).
*   **Missing Ollama Server:** On startup, ping `localhost:11434`. If unreachable, display a clear instruction: *"Please start your Ollama server first."*

## 8. File Structure

```text
/retro_tui_game
├── main.py               # Entry point, starts the loop
├── requirements.txt      # rich, ollama-python, pydantic, asyncio
├── config.yaml           # Model name (gemma4), temperature, max_tokens
├── src/
│   ├── ui/
│   │   ├── theme.py      # Rich color definitions
│   │   └── renderer.py   # Functions to print story log and prompts
│   ├── engine/
│   │   ├── llm_client.py # Ollama wrapper
│   │   └── parser.py     # JSON extraction logic
│   └── state/
│       └── models.py     # Pydantic GameState definitions
└── data/
    └── templates/        # Static setting descriptions (Fantasy, Sci-Fi)
```

## 9. Performance Considerations

*   **Token Count:** Keep the `story_summary` under 500 tokens to avoid context window overflow and reduce latency. Use a "summarize old history" prompt every 10 turns if necessary.
*   **Memory:** Since it's a local CLI, memory usage is low. The primary bottleneck is CPU/GPU inference time on the user's machine.
*   **Rendering:** Use `rich.live` or simple print statements with ANSI codes to avoid heavy re-rendering costs. Only redraw the "Story Log" area when new text arrives.