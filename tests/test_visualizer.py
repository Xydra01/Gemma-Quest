"""Unit tests for the scene visualizer, pixel converter, and TUI integration."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from src.config import GameConfig
from src.engine.visualizer import FALLBACK_ASCII_SCENE, SceneVisualizer
from src.state.models import StoryContext
from src.ui.layout import render_scene_panel
from src.ui.pixel_converter import (
    image_bytes_to_ascii,
    image_bytes_to_halfblocks,
    image_bytes_to_monochrome_halfblocks,
)
from src.ui.renderer import RetroTUI


@pytest.fixture
def synthetic_image_bytes() -> bytes:
    """Generate a 40x20 RGB PNG image in memory for testing."""
    img = Image.new("RGB", (40, 20), color=(45, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def test_config(tmp_path: Path) -> GameConfig:
    """Provide a test GameConfig with a temporary visualizer cache directory."""
    config = GameConfig()
    config.visualizer.cache_dir = str(tmp_path / "cache")
    config.visualizer.enabled = True
    config.visualizer.width = 30
    config.visualizer.height = 10
    return config


class TestPixelConverter:
    """Tests for ANSI half-block and ASCII pixel converters."""

    def test_image_bytes_to_halfblocks(self, synthetic_image_bytes: bytes):
        art = image_bytes_to_halfblocks(
            synthetic_image_bytes, width=20, height=8, quantize_colors=16
        )
        assert art != ""
        lines = art.splitlines()
        assert len(lines) == 8
        # Each line should contain the half-block character
        assert "▀" in lines[0]
        # Should contain 24-bit TrueColor ANSI escape sequences
        assert "\033[38;2;" in lines[0]
        assert "\033[48;2;" in lines[0]

    def test_image_bytes_to_ascii(self, synthetic_image_bytes: bytes):
        art = image_bytes_to_ascii(synthetic_image_bytes, width=25, height=6)
        assert art != ""
        lines = art.splitlines()
        assert len(lines) == 6
        assert len(lines[0]) == 25

    def test_image_bytes_to_monochrome(self, synthetic_image_bytes: bytes):
        art = image_bytes_to_monochrome_halfblocks(
            synthetic_image_bytes, width=20, height=8, tint_rgb=(0, 255, 65)
        )
        assert art != ""
        lines = art.splitlines()
        assert len(lines) == 8
        assert "▀" in lines[0]

    def test_empty_or_corrupt_bytes_returns_empty_string(self):
        assert image_bytes_to_halfblocks(b"") == ""
        assert image_bytes_to_ascii(b"") == ""
        assert image_bytes_to_monochrome_halfblocks(b"") == ""
        assert image_bytes_to_halfblocks(b"not an image file") == ""


class TestSceneVisualizer:
    """Tests for the SceneVisualizer service and caching logic."""

    def test_prompt_builder(self, test_config: GameConfig):
        viz = SceneVisualizer(test_config)
        prompt = viz._build_pixel_art_prompt(
            location="The Ancient Crypt",
            narration="Dust falls from cracked ceilings as torches flicker.",
            world_tone="gothic horror",
        )
        assert "16-bit retro pixel art" in prompt
        assert "The Ancient Crypt" in prompt
        assert "gothic horror" in prompt
        assert "1980s retro MS-DOS EGA" in prompt
        assert "no text" in prompt

    def test_disabled_visualizer_returns_none(self, test_config: GameConfig):
        test_config.visualizer.enabled = False
        viz = SceneVisualizer(test_config)
        result = viz.generate_scene_art(
            narration="Silence in the hall.", location="Chamber"
        )
        assert result is None

    def test_fallback_when_offline_or_unauthenticated(self, test_config: GameConfig):
        # When client is None (unauthenticated or offline), gracefully return FALLBACK_ASCII_SCENE
        viz = SceneVisualizer(test_config)
        viz._get_client = MagicMock(return_value=None)
        result = viz.generate_scene_art(
            narration="A stormy castle gate stands open.",
            location="Castle Gate",
        )
        assert result == FALLBACK_ASCII_SCENE

    def test_caching_mechanism(
        self, test_config: GameConfig, synthetic_image_bytes: bytes
    ):
        viz = SceneVisualizer(test_config)

        # Mock the client generate_images response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_generated_image = MagicMock()
        mock_generated_image.image.image_bytes = synthetic_image_bytes
        mock_response.generated_images = [mock_generated_image]
        mock_client.models.generate_images.return_value = mock_response

        viz._client = mock_client
        viz._client_initialized = True

        # First call: Cache miss, calls API
        art1 = viz.generate_scene_art(
            narration="A grand hall glows with candlelight.",
            location="Grand Hall",
        )
        assert art1 is not None
        assert "▀" in art1
        assert mock_client.models.generate_images.call_count == 1

        # Second call: Cache hit, reads from disk without calling API
        art2 = viz.generate_scene_art(
            narration="A grand hall glows with candlelight.",
            location="Grand Hall",
        )
        assert art2 == art1
        assert mock_client.models.generate_images.call_count == 1  # Unchanged!


class TestTUIIntegration:
    """Tests for UI panel layout and game frame rendering."""

    def test_render_scene_panel(self):
        panel = render_scene_panel("  [SCENE ART]  ")
        assert panel is not None
        assert "SCENE ILLUSTRATION" in str(panel.title)

    def test_render_game_frame_with_visual_art(self):
        tui = RetroTUI(force_ascii=True)
        ctx = StoryContext(
            current_location="Dungeon Entrance",
            world_tone="dark fantasy",
            active_characters={},
            key_items_held=[],
            story_summary="The journey begins.",
            turn_count=1,
        )

        with patch.object(tui.console, "print") as mock_print:
            tui.render_game_frame(
                context=ctx,
                narration="You step through the archway into the dark.",
                choices=["Light a torch", "Advance quietly"],
                visual_art="[ANSI PIXEL ART]",
            )
            # Should have printed header, status, scene illustration, narrative, actions
            assert mock_print.call_count >= 5
