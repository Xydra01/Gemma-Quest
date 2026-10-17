# Technical Design Document: Risk Management & Mitigation Strategy

## 1. Overview
This document identifies potential technical and experiential risks associated with the **Gemma Dungeon** project, specifically focusing on the integration of a small local LLM (Gemma 4) within a real-time TUI environment. It outlines the likelihood and impact of each risk and provides concrete engineering mitigations to ensure a stable, coherent, and performant player experience.

## 2. Risk Assessment Matrix

| ID | Risk Description | Likelihood | Impact | Priority |
| :--- | :--- | :---: | :---: | :---: |
| **R1** | **Gemma 4 returns invalid JSON structure** | High | Medium | **Critical** |
| **R2** | **Narrative Inconsistency (Logic Hallucinations)**<br>e.g., Dead characters revive, items reappear. | Medium | High | **Critical** |
| **R3** | **Slow LLM Response Times (>10s latency)** | Medium | Medium | **High** |
| **R4** | **Player Inputs Nonsensical or Out-of-Scope Actions** | High | Low | **Medium** |
| **R5** | **Context Window Overflow (Token Limit Exceeded)** | Medium | High | **High** |
| **R6** | **User Environment Incompatibility (Terminal/OS)** | Low | Medium | **Low** |

---

## 3. Detailed Risk Analysis & Mitigations

### R1: Gemma 4 Returns Invalid JSON
**Risk:** Small models are prone to "breaking" structured output. They might add markdown code blocks (` ```json `), truncate the JSON, or include trailing text after the closing brace. If parsing fails, the game crashes or state is corrupted.

*   **Likelihood:** High (Especially with 2B/4B parameter models).
*   **Impact:** Medium (Game freezes or breaks; requires restart).

**Mitigation Strategy:**
1.  **Robust Parsing Layer:** Do not rely on `json.loads()` directly on the raw string. Implement a "Fuzzy JSON Extractor" that uses regex to find the first `{` and last `}` in the response, ignoring any surrounding text or markdown.
2.  **Pydantic Validation with Fallbacks:**
    *   Use Pydantic models for validation.
    *   If validation fails (e.g., missing `current_location`), do not crash. Instead, **retain the previous state** and use only the `narration` field if it is valid text.
3.  **Retry Logic:**
    *   If parsing fails, automatically retry the LLM call once with a stricter prompt: *"You failed to provide valid JSON. Try again. Output ONLY JSON."*
4.  **Graceful Degradation:**
    *   If all retries fail, display the raw text as narration and provide default choices (`[1] Look Around`, `[2] Wait`). Log the error for debugging but do not crash the app.

```python
def safe_parse_llm_response(response_text: str) -> Optional[GameResponse]:
    try:
        # Extract JSON block
        json_start = response_text.rfind('{')
        json_end = response_text.rfind('}') + 1
        data = json.loads(response_text[json_start:json_end])
        
        # Validate against Pydantic Model
        return GameResponse(**data)
        
    except (ValueError, KeyError):
        return None # Trigger fallback
```

### R2: Narrative Inconsistency (Logic Hallucinations)
**Risk:** The LLM may forget previous events. A character marked as "Dead" might speak again. An item the player lost might reappear in their inventory. This breaks immersion and trust in the game engine.

*   **Likelihood:** Medium (Small models have limited context retention).
*   **Impact:** High (Breaks narrative flow, confuses player).

**Mitigation Strategy:**
1.  **Explicit State Injection:** The `StoryContext` is injected into *every* prompt. This serves as a "ground truth" for the model.
2.  **Negative Constraints in Prompting:** Explicitly list dead characters in the system prompt:
    > *"The following characters are DEAD and cannot speak or act: [List]. If you accidentally mention them acting, rewrite the scene."*
3.  **Post-Processing Validator (The "Guardian" Script):**
    *   Before accepting the LLM's `updated_story_context`, run a check:
        *   If a character is `DEAD` in the new context but appeared in `narration` as acting, reject the update for that character.
        *   If an item is removed from inventory, ensure it isn't added back in the next turn unless explicitly "found" again.
4.  **Summary Refinement (Section 5):** The optional second-pass summarization helps condense memory, reducing the chance of the model getting lost in long histories.

### R3: Slow LLM Response Times (>10s Latency)
**Risk:** On CPU-only or low-end GPU hardware, Gemma 4 inference can be slow. A frozen terminal UI looks broken. Users may press Enter repeatedly or close the app.

*   **Likelihood:** Medium (Depends on user hardware).
*   **Impact:** Medium (Poor UX, perceived as "buggy").

**Mitigation Strategy:**
1.  **UI Loading States:**
    *   Never leave the UI blank while waiting. Display a dynamic ASCII loading animation or thematic flavor text:
        > `⏳ The dungeon hums with magic...`
        > `⏳ Waiting for the Oracle...`
2.  **Streaming Responses (If Supported):**
    *   Use Ollama's streaming API to render tokens as they arrive (typewriter effect). This makes the wait feel shorter and more engaging.
3.  **Model Selection Flexibility:**
    *   Provide a `config.yaml` option to switch between `gemma4:2b-q4_0` (faster) and `gemma4:7b-q4_0` (smarter but slower). Default to the smaller model for demo purposes.
4.  **Timeout Handling:**
    *   Implement a client-side timeout (e.g., 30 seconds). If exceeded, display `[ERROR] Connection to AI lost.` and allow retry.

### R4: Player Inputs Nonsensical Actions
**Risk:** The player types "Shoot the moon," "Kill God," or gibberish like "asdfjkl". The LLM might generate a confusing narrative or ignore the input entirely.

*   **Likelihood:** High (Players are unpredictable).
*   **Impact:** Low (Narrative is open-ended; weirdness can be part of the charm).

**Mitigation Strategy:**
1.  **In-Character Consequences:** Do not block inputs. Instead, prompt the LLM to handle it narratively.
    > *"If the player performs an impossible or nonsensical action, describe the consequences in-universe (e.g., 'The moon remains unaffected,' or 'You stumble and fall'). Maintain tone."*
2.  **Fallback Choices:** If the LLM fails to generate coherent choices from a weird input, default to `[1] Recover`, `[2] Look Around`.
3.  **Input Sanitization:** Strip HTML/JS characters to prevent prompt injection attempts (e.g., `<script>alert('xss')</script>`).

### R5: Context Window Overflow
**Risk:** The `story_summary` or `active_characters` list grows too large, exceeding the model's context window limit (4096 tokens for Gemma 4). This causes truncation or API errors.

*   **Likelihood:** Medium (Over long play sessions).
*   **Impact:** High (Game crashes or loses all memory).

**Mitigation Strategy:**
1.  **Aggressive Summarization:** The "Rolling Memory" strategy (Section 4) ensures the `story_summary` is always kept to ~3-4 sentences (~50 tokens).
2.  **Character Pruning:** Limit `active_characters` to the last 5-10 relevant NPCs. Remove characters who have been dead or absent for more than 10 turns.
3.  **Token Counter Monitoring:** Before sending the prompt, calculate the token length of the constructed prompt. If it exceeds 80% of the max context, force a "Memory Compression" event (summarize and discard older details).

### R6: User Environment Incompatibility
**Risk:** The user's terminal does not support Unicode box-drawing characters or color codes, resulting in garbled text (`[[1]]` instead of `[1]`).

*   **Likelihood:** Low (Modern terminals mostly support ANSI/Unicode).
*   **Impact:** Medium (Usability issue).

**Mitigation Strategy:**
1.  **Detect Terminal Capabilities:** Use Python's `shutil.get_terminal_size()` and check environment variables for terminal type.
2.  **Fallback to ASCII:** If Unicode is not supported, automatically switch the `Rich` theme to use ASCII box-drawing characters (`+`, `-`, `|`) instead of Unicode (`╔`, `═`).
3.  **Color Fallback:** If color is not supported, disable all color styling and rely on bold text for emphasis.

## 4. Monitoring & Debugging (Post-Launch)

To proactively manage these risks after the demo:

1.  **Log File Generation:**
    *   Log every prompt sent to the LLM and every response received to `debug_log.txt`.
    *   This allows developers to analyze *why* a hallucination occurred or why JSON failed.
2.  **Telemetry (Optional):**
    *   Track average latency, JSON parse failure rate, and user session duration.
3.  **User Feedback Loop:**
    *   Add a `[?]` command in the UI that displays the current `StoryContext` state, allowing users to verify if the AI "remembers" correctly.

## 5. Summary of Critical Path Mitigations

| Component | Primary Failure Mode | Primary Fix |
| :--- | :--- | :--- |
| **Parser** | Invalid JSON | Regex extraction + Pydantic fallback + Retry. |
| **Narrative Engine** | Logic Hallucination | Strict State Injection + Dead Character Validator. |
| **UI Layer** | Perceived Lag | Streaming tokens + Thematic Loading Animation. |
| **State Manager** | Context Overflow | Rolling Summary + Token Count Monitoring. |