"""Generate the deterministic VectorForge sample set + per-image ground truth.

Usage: python samples/generate.py

Shapes are drawn at SUPERSAMPLE x resolution and downsampled with a box filter, which
gives realistic anti-aliased edges (the hard case for quantization and tracing) while
keeping the "true" flat colors known exactly. Each image gets a JSON file beside it
(e.g. 01_logo_4color.json) with its expected class, exact colors, stroke widths and the
thresholds the QA harness applies.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT_DIR = Path(__file__).resolve().parent
SUPERSAMPLE = 4
SEED = 1337

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 50, 47)
BLUE = (38, 139, 210)
YELLOW = (250, 200, 30)
GREEN = (60, 160, 80)
NAVY = (20, 40, 90)
SKIN = (245, 205, 170)
ORANGE = (240, 130, 40)


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _canvas(w: int, h: int, bg: tuple[int, ...], mode: str = "RGB") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new(mode, (w * SUPERSAMPLE, h * SUPERSAMPLE), bg)
    return img, ImageDraw.Draw(img)


def _down(img: Image.Image, w: int, h: int) -> Image.Image:
    return img.resize((w, h), Image.Resampling.BOX)


def _s(*vals: float) -> list[float]:
    """Scale logical coordinates to supersampled canvas."""
    return [v * SUPERSAMPLE for v in vals]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size * SUPERSAMPLE)


def _logo(w: int = 512, h: int = 512) -> Image.Image:
    img, d = _canvas(w, h, WHITE)
    d.ellipse(_s(56, 56, 296, 296), fill=RED)
    d.polygon(_s(300, 90, 460, 380, 140, 380), fill=BLUE)
    d.rounded_rectangle(_s(80, 330, 300, 460), radius=24 * SUPERSAMPLE, fill=YELLOW)
    return _down(img, w, h)


# ---------------------------------------------------------------------------- samples


def logo_4color() -> Image.Image:
    return _logo()


def lineart_black() -> Image.Image:
    w, h = 600, 600
    img, d = _canvas(w, h, WHITE)
    lw = 6 * SUPERSAMPLE
    d.ellipse(_s(100, 100, 500, 500), outline=BLACK, width=lw)
    d.line(_s(300, 100, 300, 500), fill=BLACK, width=lw)  # T/X junctions with circle
    d.line(_s(100, 300, 500, 300), fill=BLACK, width=lw)
    pts = [(300 + 180 * math.cos(t / 20), 300 + 180 * math.sin(t / 20) * math.cos(t / 40)) for t in range(0, 252)]
    d.line([tuple(_s(x, y)) for x, y in pts], fill=BLACK, width=3 * SUPERSAMPLE, joint="curve")
    d.arc(_s(40, 40, 200, 200), 180, 320, fill=BLACK, width=10 * SUPERSAMPLE)  # thick stroke
    d.polygon(_s(420, 60, 560, 60, 490, 170), outline=BLACK, width=4 * SUPERSAMPLE)  # sharp corners
    return _down(img, w, h)


def cartoon_outlined() -> Image.Image:
    w, h = 640, 480
    img, d = _canvas(w, h, (200, 230, 250))
    ol = 5 * SUPERSAMPLE
    d.rectangle(_s(0, 380, 640, 480), fill=GREEN, outline=BLACK, width=ol)
    d.ellipse(_s(220, 80, 420, 280), fill=SKIN, outline=BLACK, width=ol)  # face
    d.ellipse(_s(270, 140, 300, 175), fill=WHITE, outline=BLACK, width=3 * SUPERSAMPLE)
    d.ellipse(_s(340, 140, 370, 175), fill=WHITE, outline=BLACK, width=3 * SUPERSAMPLE)
    d.ellipse(_s(280, 152, 292, 166), fill=BLACK)
    d.ellipse(_s(350, 152, 362, 166), fill=BLACK)
    d.arc(_s(270, 180, 370, 250), 20, 160, fill=BLACK, width=4 * SUPERSAMPLE)  # smile
    d.polygon(_s(230, 290, 410, 290, 450, 400, 190, 400), fill=RED, outline=BLACK, width=ol)  # body
    d.ellipse(_s(500, 30, 600, 130), fill=YELLOW, outline=ORANGE, width=ol)  # sun
    return _down(img, w, h)


def text_sample() -> Image.Image:
    w, h = 800, 300
    img, d = _canvas(w, h, WHITE)
    d.text(_s(30, 20), "VectorForge", font=_font(96), fill=NAVY)
    d.text(_s(30, 150), "Sharp glyphs: a g R & 8 %", font=_font(48), fill=BLACK)
    d.text(_s(30, 230), "small text 14px — kerning, counters, serifs", font=_font(20), fill=RED)
    return _down(img, w, h)


def gradient() -> Image.Image:
    w, h = 512, 512
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    t = x / (w - 1)
    lin = np.stack([255 * (1 - t) + 30 * t, 80 + 100 * t, 200 * t + 40 * (1 - t)], axis=-1)
    r = np.hypot(x - 360, y - 360) / 180
    radial = np.clip(1 - r, 0, 1)[..., None]
    arr = lin * (1 - radial) + np.array(YELLOW, np.float32) * radial
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle((40, 40, 200, 200), fill=NAVY)  # flat region on top of gradient
    return img


def jpeg_artifacts() -> Image.Image:
    """Same art as the logo; main() writes it with JPEG q=20, 4:2:0 chroma subsampling."""
    return _logo()


def transparent_logo() -> Image.Image:
    w, h = 512, 512
    img, d = _canvas(w, h, (0, 0, 0, 0), mode="RGBA")
    d.ellipse(_s(40, 40, 472, 472), fill=(*NAVY, 255))
    d.regular_polygon((*_s(256, 256), 150 * SUPERSAMPLE), n_sides=5, rotation=90, fill=(*YELLOW, 255))
    d.ellipse(_s(206, 206, 306, 306), fill=(0, 0, 0, 0))  # true hole through all layers
    out = _down(img, w, h)
    # Semi-transparent corner square, pixel-aligned so its RGB stays exact (no premultiply rounding).
    ImageDraw.Draw(out).rectangle((0, 0, 59, 59), fill=(*RED, 128))
    return out


def thin_lines() -> Image.Image:
    w, h = 600, 400
    img = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(img)
    for i in range(12):  # 1-px grid (no AA: worst case for staircase edges)
        d.line((20 + i * 45, 20, 20 + i * 45, 180), fill=BLACK, width=1)
    for i, width in enumerate((1, 2, 3)):
        d.line((20, 220 + i * 40, 580, 240 + i * 40), fill=BLACK, width=width)  # shallow diagonals
    for k in range(6):
        d.arc((330 + k * 8, 200 + k * 8, 570 - k * 8, 390 - k * 8), 0, 300, fill=BLUE, width=1)
    return img


def large_2000() -> Image.Image:
    w, h = 2000, 2000
    img = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(SEED)
    colors = [RED, BLUE, YELLOW, GREEN, NAVY, ORANGE]
    for _ in range(140):
        x, y = rng.integers(0, w - 300, size=2)
        s = int(rng.integers(60, 300))
        c = colors[int(rng.integers(0, len(colors)))]
        if rng.random() < 0.5:
            d.ellipse((int(x), int(y), int(x) + s, int(y) + s), fill=c)
        else:
            d.rectangle((int(x), int(y), int(x) + s, int(y) + int(s * 0.6)), fill=c)
    # anti-alias via mild blur, like a real exported illustration
    return img.filter(ImageFilter.GaussianBlur(0.7))


def mixed_scene() -> Image.Image:
    w, h = 640, 480
    y, _ = np.mgrid[0:h, 0:w].astype(np.float32)
    sky = np.stack([120 + 100 * y / h, 170 + 60 * y / h, np.full_like(y, 250)], axis=-1)
    rng = np.random.default_rng(SEED)
    sky += rng.normal(0, 4, sky.shape)  # sensor-like noise
    img = Image.fromarray(np.clip(sky, 0, 255).astype(np.uint8)).resize((w * SUPERSAMPLE, h * SUPERSAMPLE))
    d = ImageDraw.Draw(img)
    ol = 4 * SUPERSAMPLE
    d.polygon(_s(0, 480, 200, 250, 420, 480), fill=GREEN, outline=BLACK, width=ol)
    d.polygon(_s(250, 480, 470, 200, 640, 400, 640, 480), fill=(90, 120, 70), outline=BLACK, width=ol)
    d.rectangle(_s(80, 330, 180, 440), fill=RED, outline=BLACK, width=ol)
    d.polygon(_s(70, 330, 130, 280, 190, 330), fill=NAVY, outline=BLACK, width=ol)
    return _down(img, w, h)


# ---------------------------------------------------------------------------- ground truth

Spec = dict[str, Any]
# stroke_widths_px: nominal width of every drawn stroke (lines are centered; Pillow outlines grow inward).
# stroke_components: number of separate connected strokes a correct skeleton must have.
SAMPLES: list[tuple[str, Callable[[], Image.Image], Spec]] = [
    (
        "01_logo_4color.png",
        logo_4color,
        {
            "expected_class": "flat_color",
            "palette": [WHITE, RED, BLUE, YELLOW],
            "notes": "4 flat colors, AA edges, curves + sharp triangle corners",
        },
    ),
    (
        "02_lineart_black.png",
        lineart_black,
        {
            "expected_class": "line_art",
            "palette": [WHITE, BLACK],
            "stroke_widths_px": {"circle": 6, "cross": 6, "curve": 3, "arc": 10, "triangle": 4},
            "median_stroke_width_px": 6,
            "stroke_components": 3,
            "notes": "Strokes 3-10px, X/T junctions, sharp polygon corners; test centerline + outline",
        },
    ),
    (
        "03_cartoon_outlined.png",
        cartoon_outlined,
        {
            "expected_class": "mixed",
            "palette": [(200, 230, 250), GREEN, SKIN, WHITE, BLACK, RED, YELLOW, ORANGE],
            "stroke_widths_px": {"outlines": 5, "eyes": 3, "smile": 4},
            "notes": "Fills with black outlines; outlines must stay continuous at junctions",
        },
    ),
    (
        "04_text.png",
        text_sample,
        {
            "expected_class": "flat_color",
            "palette": [WHITE, NAVY, BLACK, RED],
            "notes": "Glyph counters (holes), small 14-20px text",
        },
    ),
    (
        "05_gradient.png",
        gradient,
        {
            "expected_class": "mixed",
            "palette": None,
            "notes": "Linear+radial gradients (no exact palette). Stress test for banding; SSIM only",
        },
    ),
    (
        "06_jpeg_artifacts.jpg",
        jpeg_artifacts,
        {
            "expected_class": "flat_color",
            "palette": [WHITE, RED, BLUE, YELLOW],
            "notes": "Same art as 01 at JPEG q=20 with 4:2:0 chroma: ringing + blocks must not become colors",
        },
    ),
    (
        "07_transparent_logo.png",
        transparent_logo,
        {
            "expected_class": "flat_color",
            "palette": [NAVY, YELLOW, RED],
            "has_alpha": True,
            "notes": "Transparent bg, true hole in center, 50% alpha red square; no bg fill allowed",
        },
    ),
    (
        "08_thin_lines.png",
        thin_lines,
        {
            "expected_class": "line_art",
            "palette": [WHITE, BLACK, BLUE],
            "stroke_widths_px": {"grid": 1, "diagonal_1": 1, "diagonal_2": 2, "diagonal_3": 3, "arcs": 1},
            "median_stroke_width_px": 1,
            "notes": "1-3px aliased lines (12 vertical grid lines), shallow diagonals (staircase), concentric arcs",
        },
    ),
    (
        "09_large_2000.png",
        large_2000,
        {
            "expected_class": "flat_color",
            "palette": [WHITE, RED, BLUE, YELLOW, GREEN, NAVY, ORANGE],
            "notes": "2000x2000 performance test: must finish < 10 s",
        },
    ),
    (
        "10_mixed_scene.png",
        mixed_scene,
        {
            "expected_class": "mixed",
            "palette": None,
            "stroke_widths_px": {"outlines": 4},
            "notes": "Noisy gradient sky + outlined flat shapes",
        },
    ),
]

SSIM_MIN = {"flat_color": 0.90, "line_art": 0.85, "mixed": 0.85}


def main() -> None:
    for filename, fn, spec in SAMPLES:
        img = fn()
        path = OUT_DIR / filename
        if filename.endswith(".jpg"):
            img.save(path, format="JPEG", quality=20, subsampling=2)
        else:
            img.save(path, format="PNG", optimize=True)
        palette = spec.get("palette")
        truth = {
            "file": filename,
            "width": img.width,
            "height": img.height,
            "has_alpha": bool(spec.get("has_alpha", False)),
            "expected_class": spec["expected_class"],
            "palette_hex": [_hex(c) for c in palette] if palette else None,
            "stroke_widths_px": spec.get("stroke_widths_px"),
            "median_stroke_width_px": spec.get("median_stroke_width_px"),
            "stroke_components": spec.get("stroke_components"),
            "ssim_min": SSIM_MIN[spec["expected_class"]],
            "max_delta_e": 3.0,
            "max_mean_delta_e": 2.0,
            "seed": SEED,
            "notes": spec["notes"],
        }
        path.with_suffix(".json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
        print(f"{filename:28s} {img.width}x{img.height} {path.stat().st_size:>9,d} B")


if __name__ == "__main__":
    main()
