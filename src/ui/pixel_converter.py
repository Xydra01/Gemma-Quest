"""Pixel art and ASCII converter for terminal rendering.

Converts image bytes into:
1. 24-bit TrueColor ANSI half-blocks ('▀', U+2580) for high-density terminal graphics.
2. Grayscale ASCII character art for teletype / minimal terminals.
3. Monochrome phosphor art (green / amber CRT aesthetic).
"""

import io
from typing import Optional
from PIL import Image

# Standard ASCII luminance ramp (dark to bright)
ASCII_RAMP = " .:-=+*#%@"


def image_bytes_to_halfblocks(
    image_bytes: bytes,
    width: int = 60,
    height: int = 18,
    quantize_colors: Optional[int] = None,
) -> str:
    """Convert raw image bytes into TrueColor ANSI half-block characters.

    Each character cell in a terminal is roughly 1:2 aspect ratio.
    Using Unicode upper half-block '▀' (U+2580), each character displays
    two vertical pixels:
      - Foreground: top pixel (x, 2*y)
      - Background: bottom pixel (x, 2*y + 1)

    Args:
        image_bytes: Raw PNG/JPEG bytes.
        width: Output character columns (default: 60).
        height: Output character lines (default: 18, giving 36 vertical pixels).
        quantize_colors: Optional retro color palette quantization (e.g. 16 or 32).

    Returns:
        ANSI-escaped multiline string suitable for Rich or direct terminal printing.
    """
    if not image_bytes:
        return ""

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            # Pixel dimensions: 2 pixels per line
            target_pixel_size = (width, height * 2)
            img = img.resize(target_pixel_size, Image.Resampling.LANCZOS)

            if quantize_colors and quantize_colors > 0:
                img = img.quantize(colors=quantize_colors).convert("RGB")

            lines = []
            for y in range(height):
                line_parts = []
                last_fg = None
                last_bg = None

                for x in range(width):
                    top_rgb = img.getpixel((x, 2 * y))
                    bottom_rgb = img.getpixel((x, 2 * y + 1))

                    fg_str = ""
                    bg_str = ""

                    if top_rgb != last_fg:
                        fg_str = f"\033[38;2;{top_rgb[0]};{top_rgb[1]};{top_rgb[2]}m"
                        last_fg = top_rgb

                    if bottom_rgb != last_bg:
                        bg_str = f"\033[48;2;{bottom_rgb[0]};{bottom_rgb[1]};{bottom_rgb[2]}m"
                        last_bg = bottom_rgb

                    line_parts.append(f"{fg_str}{bg_str}▀")

                line_parts.append("\033[0m")
                lines.append("".join(line_parts))

            return "\n".join(lines)
    except Exception:
        return ""


def image_bytes_to_ascii(
    image_bytes: bytes,
    width: int = 60,
    height: int = 18,
    ramp: str = ASCII_RAMP,
) -> str:
    """Convert raw image bytes into monochrome ASCII character art.

    Args:
        image_bytes: Raw PNG/JPEG bytes.
        width: Output character columns.
        height: Output character lines.
        ramp: Luminance character ramp from darkest to brightest.

    Returns:
        Multiline ASCII string.
    """
    if not image_bytes:
        return ""

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("L")  # Grayscale
            img = img.resize((width, height), Image.Resampling.LANCZOS)

            lines = []
            num_chars = len(ramp)
            for y in range(height):
                row = []
                for x in range(width):
                    lum = img.getpixel((x, y))
                    idx = int((lum / 255.0) * (num_chars - 1))
                    row.append(ramp[idx])
                lines.append("".join(row))

            return "\n".join(lines)
    except Exception:
        return ""


def image_bytes_to_monochrome_halfblocks(
    image_bytes: bytes,
    width: int = 60,
    height: int = 18,
    tint_rgb: tuple[int, int, int] = (0, 255, 65),  # Classic P1 green phosphor
) -> str:
    """Convert raw image bytes into vintage monochrome phosphor half-blocks.

    Args:
        image_bytes: Raw PNG/JPEG bytes.
        width: Output character columns.
        height: Output character lines.
        tint_rgb: Phosphor color tint (R, G, B).

    Returns:
        ANSI-escaped multiline string in vintage phosphor tint.
    """
    if not image_bytes:
        return ""

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("L")
            target_pixel_size = (width, height * 2)
            img = img.resize(target_pixel_size, Image.Resampling.LANCZOS)

            lines = []
            for y in range(height):
                line_parts = []
                last_fg = None
                last_bg = None

                for x in range(width):
                    lum_top = img.getpixel((x, 2 * y)) / 255.0
                    lum_bottom = img.getpixel((x, 2 * y + 1)) / 255.0

                    top_rgb = (
                        int(tint_rgb[0] * lum_top),
                        int(tint_rgb[1] * lum_top),
                        int(tint_rgb[2] * lum_top),
                    )
                    bottom_rgb = (
                        int(tint_rgb[0] * lum_bottom),
                        int(tint_rgb[1] * lum_bottom),
                        int(tint_rgb[2] * lum_bottom),
                    )

                    fg_str = ""
                    bg_str = ""

                    if top_rgb != last_fg:
                        fg_str = f"\033[38;2;{top_rgb[0]};{top_rgb[1]};{top_rgb[2]}m"
                        last_fg = top_rgb

                    if bottom_rgb != last_bg:
                        bg_str = f"\033[48;2;{bottom_rgb[0]};{bottom_rgb[1]};{bottom_rgb[2]}m"
                        last_bg = bottom_rgb

                    line_parts.append(f"{fg_str}{bg_str}▀")

                line_parts.append("\033[0m")
                lines.append("".join(line_parts))

            return "\n".join(lines)
    except Exception:
        return ""
