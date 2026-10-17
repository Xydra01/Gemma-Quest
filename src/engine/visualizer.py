"""Scene visualizer service using Vertex AI Imagen 3 and retro ANSI rendering."""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Optional

from src.config import GameConfig, LLMConfig, VisualizerConfig
from src.ui.pixel_converter import (
    image_bytes_to_ascii,
    image_bytes_to_halfblocks,
    image_bytes_to_monochrome_halfblocks,
)

logger = logging.getLogger(__name__)

# Fallback retro ASCII landscape when offline or credentials are missing
FALLBACK_ASCII_SCENE = (
    "                  /\\                  \n"
    "                 /  \\    /\\           \n"
    "   /\\           /    \\  /  \\          \n"
    "  /  \\   /\\    /      \\/    \\         \n"
    " /    \\ /  \\  /              \\        \n"
    "/______V____\\/________________\\       \n"
    "  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░       \n"
    "   [ SCENE VISUALIZATION OFFLINE ]    \n"
    " (Set up Google Cloud ADC to enable)  "
)


class SceneVisualizer:
    """Manages scene image generation via Vertex AI / Imagen 3 and terminal conversion."""

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config: VisualizerConfig = (
            getattr(config, "visualizer", None)
            if config is not None
            else None
        ) or VisualizerConfig()
        self.llm_config: LLMConfig = (
            getattr(config, "llm", None)
            if config is not None
            else None
        ) or LLMConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._client_initialized = False
        self._init_error = None

    def _get_client(self):
        """Lazily initialize Google GenAI Client with Vertex AI / ADC or AI Studio."""
        if self._client_initialized:
            return self._client

        self._client_initialized = True
        try:
            from google import genai

            project_id = self.config.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
            location = self.config.location or os.getenv(
                "GOOGLE_CLOUD_LOCATION", "us-central1"
            )
            backend = self.config.backend

            # Determine whether to use Vertex AI with ADC
            use_vertex = False
            if backend == "vertex_ai":
                use_vertex = True
            elif backend == "auto":
                # If project_id or GOOGLE_GENAI_USE_VERTEXAI is set, prefer Vertex AI
                env_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in (
                    "true",
                    "1",
                    "yes",
                )
                if project_id or env_vertex:
                    use_vertex = True

            if use_vertex:
                # Vertex AI using Application Default Credentials (ADC)
                logger.info(
                    f"Initializing Vertex AI Image client (project={project_id}, location={location})"
                )
                self._client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location,
                )
            else:
                # Fallback to AI Studio API key if provided
                api_key = self.llm_config.get_api_key()
                if api_key:
                    logger.info("Initializing AI Studio Image client with API key")
                    self._client = genai.Client(api_key=api_key)
                else:
                    self._init_error = "No Vertex AI project or API key configured."
                    logger.warning(
                        "Visualizer: No Google Cloud Project (ADC) or API key found. Visuals will use offline fallback."
                    )
        except Exception as e:
            self._init_error = str(e)
            logger.warning(
                f"Visualizer client initialization error: {e}. Visuals will use offline fallback."
            )
            self._client = None

        return self._client

    def _compute_cache_key(
        self, location: str, narration: str, style: str, width: int, height: int
    ) -> str:
        """Generate SHA-256 cache key for the scene."""
        raw = f"{location.strip()}|{narration.strip()[:200]}|{style}|{width}x{height}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _build_pixel_art_prompt(
        self, location: str, narration: str, world_tone: Optional[str] = None
    ) -> str:
        """Construct evocative prompt conditioned for 16-bit retro pixel art."""
        tone = world_tone or "dark atmospheric fantasy"
        # Extract first 150 chars of narration to guide specific scene environment
        clean_narration = " ".join(narration.split())[:160]

        return (
            f"16-bit retro pixel art of {location}. {clean_narration}. "
            f"Setting tone: {tone}. Aesthetic: 1980s retro MS-DOS EGA adventure game, "
            f"rich lighting, dithering, limited palette, atmospheric scenery, "
            f"no text, no words, no user interface, no borders."
        )

    def generate_scene_art(
        self,
        narration: str,
        location: str = "Unknown Realm",
        world_tone: Optional[str] = None,
    ) -> Optional[str]:
        """Generate or retrieve cached ANSI / ASCII illustration for the current scene.

        Args:
            narration: Latest narration text.
            location: Current location name.
            world_tone: Narrative tone / atmosphere.

        Returns:
            Renderable ANSI or ASCII string, or None if disabled.
        """
        if not self.config.enabled:
            return None

        width = self.config.width
        height = self.config.height
        style = self.config.style

        cache_key = self._compute_cache_key(
            location, narration, style, width, height
        )
        cache_file = self.cache_dir / f"{cache_key}.ans"

        # Check local disk cache first
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.debug(f"Cache read error: {e}")

        # Attempt generation via Vertex AI / Imagen 3
        client = self._get_client()
        image_bytes = None

        if client:
            prompt = self._build_pixel_art_prompt(location, narration, world_tone)
            try:
                response = client.models.generate_images(
                    model=self.config.model,
                    prompt=prompt,
                    config={"number_of_images": 1, "aspect_ratio": "4:3"},
                )
                if response.generated_images and len(response.generated_images) > 0:
                    image_bytes = response.generated_images[0].image.image_bytes
            except Exception as e:
                logger.warning(f"Vertex AI image generation failed: {e}")
                image_bytes = None

        # Convert image bytes or return fallback
        if image_bytes:
            if style == "ascii":
                art = image_bytes_to_ascii(image_bytes, width=width, height=height)
            elif style == "monochrome":
                art = image_bytes_to_monochrome_halfblocks(
                    image_bytes, width=width, height=height
                )
            else:
                art = image_bytes_to_halfblocks(
                    image_bytes, width=width, height=height, quantize_colors=32
                )

            # Persist to local disk cache
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(art)
                # Also save the raw PNG alongside for reference
                png_file = self.cache_dir / f"{cache_key}.png"
                with open(png_file, "wb") as f:
                    f.write(image_bytes)
            except Exception as e:
                logger.debug(f"Cache write error: {e}")

            return art

        # Return fallback scene when credentials or network unavailable
        return FALLBACK_ASCII_SCENE
