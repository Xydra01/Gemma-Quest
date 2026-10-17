"""Configuration management for Gemma-Quest."""

import os
from pathlib import Path
from typing import Dict, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import yaml

# Base directory for the repository
BASE_DIR = Path(__file__).resolve().parent.parent


class AppConfig(BaseModel):
    """General application metadata and UI styling."""

    name: str = Field(default="Gemma Dungeon", description="Application title")
    version: str = Field(default="1.0.0", description="Application version")
    theme: str = Field(default="retro", description="TUI theme identifier")


class LLMConfig(BaseModel):
    """Inference and model parameters."""

    mode: Literal["local", "cloud", "hybrid"] = Field(
        default="local", description="Inference mode: 'local', 'cloud', or 'hybrid'"
    )
    local_model: str = Field(
        default="gemma4:e2b", description="Ollama model identifier"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Base URL for local Ollama server"
    )
    cloud_provider: str = Field(
        default="vertex_ai", description="Cloud provider identifier"
    )
    cloud_model_id: str = Field(
        default="gemma-4-26b-a4b-it", description="Cloud model identifier"
    )
    cloud_endpoint: str = Field(
        default="", description="Optional custom cloud endpoint URL"
    )
    api_key_env_var: str = Field(
        default="GEMMA_API_KEY", description="Environment variable holding the API key"
    )
    temperature: float = Field(
        default=0.8, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=50, ge=1, description="Top-k sampling")
    max_tokens: int = Field(default=1024, ge=64, description="Max tokens per response")
    timeout_seconds: int = Field(
        default=60, ge=1, description="Inference timeout in seconds"
    )

    def get_api_key(self) -> Optional[str]:
        """Retrieve the cloud API key from the environment."""
        return (
            os.getenv(self.api_key_env_var)
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )


class ResilienceConfig(BaseModel):
    """Fallback, circuit breaker, and retry settings."""

    hybrid_fallback_timeout: float = Field(
        default=5.0, description="Seconds before hybrid mode falls back to local"
    )
    circuit_breaker_threshold: int = Field(
        default=3, description="Consecutive cloud failures before disabling cloud"
    )
    max_parse_retries: int = Field(
        default=1, description="Maximum retries on invalid JSON responses"
    )


class ContextConfig(BaseModel):
    """Context window and narrative memory management."""

    summary_refinement_turn_interval: int = Field(
        default=5, description="Turns between automatic summary refinement"
    )
    summary_refinement_char_threshold: int = Field(
        default=450, description="Summary length triggering refinement"
    )
    context_window_warning_ratio: float = Field(
        default=0.8, description="Ratio of 4096 tokens triggering compression"
    )
    max_active_npcs: int = Field(
        default=10, description="Max NPCs tracked in active context"
    )
    npc_prune_turn_threshold: int = Field(
        default=10, description="Turns an inactive/dead NPC remains before pruning"
    )


class VisualizerConfig(BaseModel):
    """Scene visualizer and ANSI pixel art generation settings."""

    enabled: bool = Field(
        default=True, description="Enable visual scene rendering in the TUI"
    )
    backend: Literal["auto", "vertex_ai", "ai_studio"] = Field(
        default="auto", description="Image generation backend"
    )
    model: str = Field(
        default="gemini-2.5-flash-image",
        description="Image generation model identifier",
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Google Cloud Project ID for Vertex AI (falls back to GOOGLE_CLOUD_PROJECT)",
    )
    location: str = Field(
        default="us-central1", description="Google Cloud Region for Vertex AI"
    )
    style: Literal["pixel_art", "ascii", "monochrome"] = Field(
        default="pixel_art",
        description="Rendering style: 'pixel_art' (TrueColor ANSI), 'ascii', or 'monochrome'",
    )
    width: int = Field(
        default=60, ge=20, le=120, description="Visualizer character width"
    )
    height: int = Field(
        default=18, ge=6, le=40, description="Visualizer character height"
    )
    cache_dir: str = Field(
        default="data/cache/scenes", description="Directory to cache generated images"
    )


class SettingTemplate(BaseModel):
    """Predefined setting definition for onboarding."""

    name: str
    description: str
    default_goal: str
    starter_location: str


class GameConfig(BaseModel):
    """Root configuration model combining all sub-configs."""

    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    visualizer: VisualizerConfig = Field(default_factory=VisualizerConfig)


def load_config(config_path: Optional[str | Path] = None) -> GameConfig:
    """Load configuration from config.yaml and environment variables.

    Args:
        config_path: Optional path to config.yaml. Defaults to BASE_DIR / "config.yaml".

    Returns:
        Validated GameConfig instance.
    """
    # Load environment variables from .env if present
    load_dotenv(BASE_DIR / ".env")

    path = Path(config_path) if config_path else BASE_DIR / "config.yaml"

    config_data: dict = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                config_data = loaded

    # Allow environment variable override for LLM mode (e.g. GEMMA_MODE=cloud)
    env_mode = os.getenv("GEMMA_MODE")
    if env_mode and env_mode in ("local", "cloud", "hybrid"):
        config_data.setdefault("llm", {})["mode"] = env_mode

    # Allow environment variable override for cloud endpoint
    env_endpoint = os.getenv("GEMMA_CLOUD_ENDPOINT")
    if env_endpoint:
        config_data.setdefault("llm", {})["cloud_endpoint"] = env_endpoint

    # Visualizer environment variable overrides
    vis_data = config_data.setdefault("visualizer", {})
    if os.getenv("GOOGLE_CLOUD_PROJECT"):
        vis_data["project_id"] = os.getenv("GOOGLE_CLOUD_PROJECT")
    if os.getenv("GOOGLE_CLOUD_LOCATION"):
        vis_data["location"] = os.getenv("GOOGLE_CLOUD_LOCATION")
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes"):
        vis_data["backend"] = "vertex_ai"
    if os.getenv("GEMMA_VISUALS_ENABLED"):
        vis_data["enabled"] = os.getenv("GEMMA_VISUALS_ENABLED", "").lower() in (
            "true",
            "1",
            "yes",
        )

    return GameConfig(**config_data)


def load_setting_templates(
    templates_path: Optional[str | Path] = None,
) -> Dict[str, SettingTemplate]:
    """Load predefined world setting templates from YAML.

    Args:
        templates_path: Optional path to settings.yaml. Defaults to data/templates/settings.yaml.

    Returns:
        Mapping of setting key (e.g. 'fantasy') to SettingTemplate.
    """
    path = (
        Path(templates_path)
        if templates_path
        else BASE_DIR / "data" / "templates" / "settings.yaml"
    )

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "settings" not in data:
        return {}

    return {
        key: SettingTemplate(**setting_data)
        for key, setting_data in data["settings"].items()
    }
