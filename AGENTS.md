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
  - **Dual-Mode LLM Testing (Cloud API vs. Local):** Once the Gemma 4 API is implemented, live tests can be conducted either through the Cloud API or locally via Ollama if the API fails (e.g. missing API keys, rate limits, or network connectivity issues). Pull requests must specify which backend was exercised during live verification.
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

---

## 5. Pull Request Guidelines & Template

When opening a Pull Request (PR), agents and contributors must use the following template to guarantee that changes are well-documented, conform to branch standards, and include concrete live verification evidence.

### Standard PR Template

```markdown
## Description
<!-- Provide a clear summary of what this PR does and why it was created. -->

## Type of Change
- [ ] `feat`: New feature or capability
- [ ] `fix`: Bug fix
- [ ] `docs`: Documentation update
- [ ] `refactor`: Refactoring without behavior change
- [ ] `test`: Test additions or updates
- [ ] `chore`: Dependencies, build scripts, configuration

## Live Verification & Testing Evidence
<!-- MANDATORY: Describe how the changes were tested live in the local environment. -->
<!-- Include exact commands executed, live terminal outputs, screenshots, or logs. -->
- **Live Test Command(s):** `...`
- **Inference Backend Tested:** `Cloud API` | `Local Ollama` | `N/A`
- **Observed Behavior:**
- **Edge Cases Tested:**

## Design & Architecture Alignment
<!-- Reference relevant docs if applicable (e.g., GemmaQuestDesign.md, Narrative_State.md) -->
- Related Docs:

## Open Questions & Trade-offs
<!-- Highlight any remaining ambiguities, tradeoffs made, or follow-ups needed. -->

## Checklist
- [ ] Branch follows the `<type>/<description>` naming convention.
- [ ] Code has been tested and verified **live** (not just statically reviewed).
- [ ] No regressions in existing game loop, TUI flow, or state tracking.
- [ ] Documentation updated where necessary.
```

