# Agent Guidelines & Workflow Rules

Welcome to **Gemma-Quest**! When working on this repository, all AI agents and contributors must adhere to the following core guidelines:

---

## 1. Branching Strategy

- **Dedicated Branches Required:** All changes must be developed on a dedicated branch created from `main`. Do not push or commit directly to `main`.
- **Branch Naming Convention:** Name branches using the format:
  ```text
  <type>/<description>
  ```
  *(Note: All branch names must be lowercase, hyphen-separated, and categorized by type)*

- **Supported Types:**
  - `feat`: New features or capabilities (e.g., `feat/tui-retro-border`, `feat/ollama-client`)
  - `fix`: Bug fixes (e.g., `fix/narrative-state-desync`, `fix/choice-parser`)
  - `docs`: Documentation updates (e.g., `docs/prompt-design-update`, `docs/setup-guide`)
  - `refactor`: Code refactoring without changing user-facing behavior (e.g., `refactor/state-manager`)
  - `test`: Adding or updating test suites (e.g., `test/json-schema-validation`)
  - `chore`: Tooling, dependencies, or configuration changes (e.g., `chore/requirements-update`)

- **Description Style:** Use concise, lowercase, hyphen-separated descriptions for `<what-is-being-done>`.

---

## 2. Testing & Live Verification

- **Live Execution Required:** All changes must be tested and demonstrated to run properly **live** in the local environment before declaring any work complete.
- **Verification Standards:**
  - Run the application or CLI/TUI directly to verify interactive behavior, startup, and error handling.
  - Test edge cases (e.g., model disconnects, invalid player inputs, corrupted game state).
  - Ensure zero regressions in existing flows before marking a task complete.
  - Never mark a task as finished based solely on code review or static assumptions.

---

## 3. Communication & Clarification

- **Ask Open Questions:** Whenever requirements, architecture decisions, or next steps are underspecified or ambiguous, proactively ask open questions.
- **Surface Trade-offs:** When multiple valid technical approaches exist (e.g., TUI library choices, state serialization trade-offs, model prompt structure), present options and seek direction rather than making unverified assumptions.
- **Provide Status Updates:** Summarize what was tested, what passed, and any remaining questions or follow-ups.

---

## 4. Repository Context

- Refer to the project design documentation before modifying architectural patterns:
  - `GemmaQuestDesign.md`: Core vision and game loop architecture.
  - `Narrative_State.md`: State management specifications.
  - `TUI_Design.md` & `TUI-Runtime_Architecture.md`: Interface and runtime layout.
  - `Prompt_Design.md` & `API_Local.md`: LLM prompting and local inference setup.
