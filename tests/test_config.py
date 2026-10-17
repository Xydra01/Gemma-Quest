"""Unit tests and verification for configuration and scaffolding."""

import os
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.config import (
    GameConfig,
    load_config,
    load_setting_templates,
    BASE_DIR,
)


def test_package_scaffolding():
    """Verify that all core project packages exist and are importable."""
    import src
    import src.ui
    import src.engine
    import src.state
    import src.config

    assert hasattr(src, "__version__")
    assert (BASE_DIR / "src" / "ui" / "__init__.py").exists()
    assert (BASE_DIR / "src" / "engine" / "__init__.py").exists()
    assert (BASE_DIR / "src" / "state" / "__init__.py").exists()
    assert (BASE_DIR / "data" / "templates" / "settings.yaml").exists()


def test_default_config_loading():
    """Verify loading default config from config.yaml."""
    config = load_config()
    assert isinstance(config, GameConfig)
    assert config.app.name == "Gemma Dungeon"
    assert config.app.theme == "retro"
    assert config.llm.mode == "local"
    assert config.llm.local_model == "gemma4:2b-q4_0"
    assert config.llm.temperature == 0.8
    assert config.resilience.circuit_breaker_threshold == 3
    assert config.context.summary_refinement_turn_interval == 5


def test_config_validation_constraints():
    """Verify Pydantic validation on out-of-range parameters."""
    with pytest.raises(ValidationError):
        # Invalid temperature > 2.0
        load_config_data = {
            "llm": {"temperature": 3.5}
        }
        GameConfig(**load_config_data)

    with pytest.raises(ValidationError):
        # Invalid mode not in ('local', 'cloud', 'hybrid')
        load_config_data = {
            "llm": {"mode": "invalid_mode"}
        }
        GameConfig(**load_config_data)


def test_environment_override(monkeypatch):
    """Verify environment variable overrides."""
    monkeypatch.setenv("GEMMA_MODE", "hybrid")
    monkeypatch.setenv("GEMMA_API_KEY", "test-secret-key")
    monkeypatch.setenv("GEMMA_CLOUD_ENDPOINT", "https://api.example.com/v1")

    config = load_config()
    assert config.llm.mode == "hybrid"
    assert config.llm.get_api_key() == "test-secret-key"
    assert config.llm.cloud_endpoint == "https://api.example.com/v1"


def test_load_setting_templates():
    """Verify loading predefined setting templates from data/templates/settings.yaml."""
    templates = load_setting_templates()
    assert "fantasy" in templates
    assert "scifi" in templates
    assert "cyberpunk" in templates

    fantasy = templates["fantasy"]
    assert fantasy.name == "Dark Fantasy"
    assert "medieval" in fantasy.description.lower()
    assert fantasy.starter_location == "The Rusty Gate"

    scifi = templates["scifi"]
    assert scifi.name == "Sci-Fi Void"
    assert scifi.starter_location == "Airlock Bay 4"

    cyberpunk = templates["cyberpunk"]
    assert cyberpunk.name == "Cyberpunk Metropolis"
    assert cyberpunk.starter_location == "Neon Alley Underpass"
