"""Second-pass story summary refinement loop for Gemma-Quest."""

import logging
import re
from typing import Any, Dict, List, Optional

from src.config import ContextConfig, GameConfig, load_config
from src.engine.base import InferenceError, InferenceProvider
from src.state.models import StoryContext

logger = logging.getLogger(__name__)

REFINEMENT_SYSTEM_PROMPT = """You are an editor for a text adventure game.
Your task is to update and condense the story summary to exactly 2-3 sentences based on the latest interaction.
Focus ONLY on plot progression, current location, and character status changes.
Respond ONLY with the clean 2-3 sentence summary paragraph.
Do NOT include any thoughts, reasoning, analysis, scratchpad notes, draft iterations, or bullet points."""


class SummaryRefiner:
    """Manages heuristic detection and execution for 2nd-pass story summary refinement."""

    def __init__(self, config: Optional[ContextConfig] = None):
        """Initialize the refiner with context configuration parameters."""
        if config is None:
            try:
                game_config = load_config()
                self.config = game_config.context
            except Exception:
                self.config = ContextConfig()
        else:
            self.config = config

    def should_refine(
        self,
        context: StoryContext,
        candidate_summary: Optional[str] = None,
        is_major_event: bool = False,
        char_threshold: Optional[int] = None,
        turn_interval: Optional[int] = None,
    ) -> bool:
        """Determine if summary refinement is required based on heuristics.

        Args:
            context: Current StoryContext.
            candidate_summary: Proposed summary to inspect (defaults to context.story_summary).
            is_major_event: Flag indicating character death, boss encounter, or major plot shift.
            char_threshold: Override for length threshold (default from config).
            turn_interval: Override for turn interval threshold (default from config).

        Returns:
            True if refinement should be triggered, False otherwise.
        """
        if is_major_event:
            return True

        summary_text = candidate_summary if candidate_summary is not None else context.story_summary
        threshold = char_threshold if char_threshold is not None else self.config.summary_refinement_char_threshold
        interval = turn_interval if turn_interval is not None else self.config.summary_refinement_turn_interval

        # Check character length heuristic
        if len(summary_text) > threshold:
            return True

        # Check periodic turn count heuristic (avoid triggering on turn 0 unless explicitly requested)
        if interval > 0 and context.turn_count > 0 and (context.turn_count % interval == 0):
            return True

        return False

    @classmethod
    def build_refinement_messages(
        cls, player_input: str, last_narration: str, current_summary: str
    ) -> List[Dict[str, str]]:
        """Construct role-based chat messages for summary refinement.

        Args:
            player_input: The player's last action.
            last_narration: Narration output from the turn.
            current_summary: Summary of previous events before this turn.

        Returns:
            List of message dicts for the LLM.
        """
        user_content = (
            f"[PREVIOUS SUMMARY]\n{current_summary.strip()}\n\n"
            f"[LATEST TURN]\n"
            f"Action: {player_input.strip()}\n"
            f"Narration: {last_narration.strip()}\n\n"
            f"Condense the updated story into 2-3 clean narrative sentences. "
            f"Output ONLY the summary paragraph without bullet points or analysis."
        )

        return [
            {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def refine_summary(
        self,
        provider: InferenceProvider,
        player_input: str,
        last_narration: str,
        current_summary: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Execute a 2nd-pass refinement call using the given provider.

        Args:
            provider: Active InferenceProvider (Local, Cloud, or Hybrid).
            player_input: Player's last action.
            last_narration: Narration text from the latest turn.
            current_summary: Summary to condense or update.
            options: Optional override for sampling parameters (defaults to low temp).

        Returns:
            Refined 2-3 sentence summary string, or current_summary if refinement fails.
        """
        messages = self.build_refinement_messages(
            player_input=player_input,
            last_narration=last_narration,
            current_summary=current_summary,
        )

        sampling_opts = {
            "temperature": 0.2,
            "max_tokens": 1024,
            "top_p": 0.9,
        }
        if options:
            sampling_opts.update(options)

        try:
            raw_response = provider.chat(messages, options=sampling_opts)
            cleaned = self._clean_summary_output(raw_response)

            # Heuristic safeguard: must be non-empty, minimum length, and free of thinking artifacts
            if not cleaned or len(cleaned) < 15 or cleaned.startswith("*") or cleaned.startswith("-"):
                logger.warning(
                    "Refinement output invalid or pure thinking ('%s'); retaining previous summary.",
                    cleaned[:80] if cleaned else "EMPTY",
                )
                return current_summary

            return cleaned

        except Exception as exc:
            logger.warning("Summary refinement failed (%s); retaining previous summary.", exc)
            return current_summary

    @staticmethod
    def _clean_summary_output(raw_text: str) -> str:
        """Strip markdown code blocks, prefixes, thought traces, and extraneous formatting."""
        if not raw_text:
            return ""

        text = raw_text.strip()

        # Remove markdown code blocks if wrapped
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Remove any leading [SUMMARY] label
        text = re.sub(r"^\[SUMMARY\]\s*", "", text, flags=re.IGNORECASE).strip()

        # Strip surrounding double or single quotes
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()

        # Check if the text contains an explicit draft label (e.g. "*Revised Draft:*" or "Summary:")
        draft_match = re.search(r'(?:Revised Draft|Final Summary|Summary):\s*([^\n*]+)', text, re.IGNORECASE)
        if draft_match:
            candidate = draft_match.group(1).strip()
            if len(candidate) > 20 and not candidate.startswith("*") and not candidate.startswith("-"):
                return candidate

        # Filter out reasoning bullet points / meta-language
        scratchpad_markers = (
            "*", "-", "1.", "2.", "3.", "Role:", "Task:", "Focus:", "Constraint",
            "Exclusions:", "Player Action:", "Narration:", "Current Summary:",
            "Wait", "Let's", "Revised Draft:", "Drafting Sentences:", "Interaction:",
        )
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        non_bullet_lines = []
        for line in lines:
            if any(line.startswith(marker) or line.lower().startswith(marker.lower()) for marker in scratchpad_markers):
                continue
            non_bullet_lines.append(line)

        if non_bullet_lines:
            candidate = " ".join(non_bullet_lines).strip()
            if len(candidate) > 20 and not candidate.startswith("*") and not candidate.startswith("-"):
                return candidate

        return ""
