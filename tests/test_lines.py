"""Tests for pipeline/lines.py (stage 4: line & edge extraction).

Upstream stages are mocked: PreprocessResult objects are built directly from the sample files
(and upscaled x2 to mimic the preprocessor's small-image upscaling).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from contracts import fixtures as fx
from contracts.schemas import (
    DenoiseParams,
    DetailLevel,
    ImageClass,
    ImageClassLabel,
    ImageInput,
    ImageMode,
    LineMap,
    LineMode,
    PreprocessResult,
    Settings,
    SourceFormat,
)
from pipeline import lines as L
from pipeline.lines import extract_lines

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
LINE_ART = ImageClass(label=ImageClassLabel.LINE_ART, confidence=1.0)
MIXED = ImageClass(label=ImageClassLabel.MIXED, confidence=1.0)
FLAT = ImageClass(label=ImageClassLabel.FLAT_COLOR, confidence=1.0)


# ------------------------------------------------------------------ helpers


def make_pre(
    rgb: np.ndarray, alpha: np.ndarray | None = None, scale: float = 1.0, name: str = "x.png"
) -> PreprocessResult:
    """Build a PreprocessResult from source-space pixels, upscaling by ``scale`` (bicubic)."""
    h, w = rgb.shape[:2]
    if scale != 1.0:
        size = (round(w * scale), round(h * scale))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC)
        if alpha is not None:
            alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_CUBIC)
    src = ImageInput(
        path=Path(name),
        width=w,
        height=h,
        has_alpha=alpha is not None,
        mode=ImageMode.RGBA if alpha is not None else ImageMode.RGB,
        source_format=SourceFormat.PNG,
        file_size_bytes=1000,
    )
    return PreprocessResult(
        source=src,
        image=np.ascontiguousarray(rgb),
        alpha=alpha,
        scale_factor=scale,
        denoise=DenoiseParams(method="none", strength=0.0),
    )


def load_sample(stem: str, scale: float = 1.0) -> tuple[PreprocessResult, dict[str, object]]:
    path = next(SAMPLES.glob(f"{stem}*.png"))
    arr = np.asarray(Image.open(path).convert("RGBA"))
    alpha = arr[..., 3].copy()
    truth = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return make_pre(arr[..., :3].copy(), alpha if (alpha < 255).any() else None, scale, path.name), truth


def components(mask: np.ndarray) -> int:
    n, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(n) - 1


def source_coords(lm: LineMap, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Skeleton pixel coordinates in source space and their widths in source px."""
    ys, xs = np.nonzero(lm.skeleton)
    return (ys + 0.5) / scale - 0.5, (xs + 0.5) / scale - 0.5, lm.width_map[ys, xs] / scale


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def cartoon_outline_reference() -> np.ndarray:
    """Dark-outline reference for 03_cartoon_outlined: black strokes only (re-rendered at 4x,
    box-downsampled, >= 50 % coverage). Filled pupils and the orange sun outline are excluded."""
    ss, w, h = 4, 640, 480

    def s(*v: float) -> list[float]:
        return [x * ss for x in v]

    img = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(img)
    d.rectangle(s(0, 380, 640, 480), outline=255, width=5 * ss)
    d.ellipse(s(220, 80, 420, 280), outline=255, width=5 * ss)
    d.ellipse(s(270, 140, 300, 175), outline=255, width=3 * ss)
    d.ellipse(s(340, 140, 370, 175), outline=255, width=3 * ss)
    d.arc(s(270, 180, 370, 250), 20, 160, fill=255, width=4 * ss)
    d.polygon(s(230, 290, 410, 290, 450, 400, 190, 400), outline=255, width=5 * ss)
    return np.asarray(img.resize((w, h), Image.Resampling.BOX)) >= 128


def assert_valid_skeleton(lm: LineMap) -> None:
    sk = lm.skeleton
    assert not np.any(sk & ~lm.mask)
    # 1 px wide: no 2x2 fully-set blocks and no redundant staircase pixels.
    blocks = sk[:-1, :-1] & sk[1:, :-1] & sk[:-1, 1:] & sk[1:, 1:]
    assert not blocks.any()
    assert np.array_equal(L.remove_redundant_pixels(sk), sk)
    assert np.all(lm.width_map[sk] > 0)
    assert np.all(lm.width_map[~sk] == 0)


def region_median(values: np.ndarray, sel: np.ndarray) -> float:
    assert sel.sum() > 20, "region selection too small"
    return float(np.median(values[sel]))


# ------------------------------------------------------------------ 02 line art


@pytest.mark.golden
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_lineart_components_junctions_and_widths(scale: float) -> None:
    pre, truth = load_sample("02_", scale)
    lm = extract_lines(pre, LINE_ART, Settings(line_mode=LineMode.CENTERLINE))
    assert_valid_skeleton(lm)
    assert components(lm.skeleton) == truth["stroke_components"] == 3
    assert lm.color_rgb == (0, 0, 0)

    # No break at any X/T junction: the skeleton inside a window around each junction is one piece.
    r = round(14 * scale)
    for jx, jy in [(300, 300), (300, 100), (300, 500), (100, 300), (500, 300)]:
        cy, cx = round(jy * scale), round(jx * scale)
        win = lm.skeleton[cy - r : cy + r + 1, cx - r : cx + r + 1]
        assert win.sum() > 0 and components(win) == 1, (jx, jy)

    ys, xs, wd = source_coords(lm, scale)
    widths: dict[str, float] = truth["stroke_widths_px"]  # type: ignore[assignment]
    d_main = np.hypot(xs - 300, ys - 300)
    d_arc = np.hypot(xs - 120, ys - 120)
    on_cross = (np.abs(xs - 300) < 2) | (np.abs(ys - 300) < 2)
    measured = {
        "arc": region_median(wd, (np.abs(d_arc - 75) < 4) & (d_main > 215)),
        "triangle": region_median(wd, (xs > 410) & (ys < 180)),
        "circle": region_median(wd, (np.abs(d_main - 197) < 4) & ~on_cross),
        "cross": region_median(wd, on_cross & (d_main < 170)),
        "curve": region_median(wd, (d_main < 185) & (np.abs(xs - 300) > 6) & (np.abs(ys - 300) > 6)),
    }
    for key, value in measured.items():
        assert abs(value - widths[key]) <= 1.0, (key, value, widths[key])
    assert abs(lm.median_stroke_width / scale - float(truth["median_stroke_width_px"])) <= 1.0  # type: ignore[arg-type]


# ------------------------------------------------------------------ 08 thin lines


@pytest.mark.golden
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_thin_grid_lines_unbroken(scale: float) -> None:
    pre, truth = load_sample("08_", scale)
    lm = extract_lines(pre, LINE_ART, Settings(line_mode=LineMode.CENTERLINE))
    assert_valid_skeleton(lm)
    sk = lm.skeleton
    for i in range(12):
        x = 20 + i * 45
        x0, x1 = math.floor((x - 1) * scale), math.ceil((x + 2) * scale)
        band = sk[round(22 * scale) : round(179 * scale), x0:x1]
        assert band.any(axis=1).all(), f"grid line {i} broken"
        assert components(sk[round(15 * scale) : round(185 * scale), x0:x1]) == 1, f"grid line {i} split"

    ys, xs, wd = source_coords(lm, scale)
    widths: dict[str, float] = truth["stroke_widths_px"]  # type: ignore[assignment]
    grid = (ys > 25) & (ys < 175)
    measured = {"grid": region_median(wd, grid)}
    for i in range(3):
        yline = 220 + i * 40 + (xs - 20) * (20 / 560)
        measured[f"diagonal_{i + 1}"] = region_median(wd, (np.abs(ys - yline) < 3) & (xs > 30) & (xs < 320))
    near_diag = np.zeros(ys.shape, bool)
    for i in range(3):
        near_diag |= np.abs(ys - (220 + i * 40 + (xs - 20) * (20 / 560))) < 5
    measured["arcs"] = region_median(wd, (xs > 325) & (xs < 575) & (ys > 195) & (ys < 395) & ~near_diag)
    for key, value in measured.items():
        assert abs(value - widths[key]) <= 1.0, (key, value, widths[key])
    assert abs(lm.median_stroke_width / scale - float(truth["median_stroke_width_px"])) <= 1.0  # type: ignore[arg-type]


# ------------------------------------------------------------------ 03 / 10 mixed


@pytest.mark.golden
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_cartoon_fills_not_strokes(scale: float) -> None:
    pre, truth = load_sample("03_", scale)
    lm = extract_lines(pre, MIXED, Settings())
    assert_valid_skeleton(lm)
    mask = lm.mask
    if scale != 1.0:
        size = (pre.source.width, pre.source.height)
        mask = cv2.resize(lm.mask.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.5
    ref = cartoon_outline_reference()
    assert iou(mask, ref) >= 0.85
    # Fill interiors (sky, face, body, grass, sun) are never strokes.
    for x, y in [(50, 50), (320, 120), (320, 340), (100, 440), (550, 80)]:
        assert not mask[y - 5 : y + 6, x - 5 : x + 6].any(), (x, y)
    assert lm.color_rgb == (0, 0, 0)

    ys, xs, wd = source_coords(lm, scale)
    widths: dict[str, float] = truth["stroke_widths_px"]  # type: ignore[assignment]
    d_face = np.hypot(xs - 320, ys - 180)
    measured = {
        "outlines": region_median(wd, np.abs(d_face - 97.5) < 3),
        "eyes": region_median(wd, (ys > 135) & (ys < 180) & (d_face < 60)),
        "smile": region_median(wd, (ys > 225) & (ys < 255) & (xs > 280) & (xs < 360)),
    }
    for key, value in measured.items():
        assert abs(value - widths[key]) <= 1.0, (key, value, widths[key])


@pytest.mark.golden
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_mixed_scene_outlines(scale: float) -> None:
    pre, truth = load_sample("10_", scale)
    lm = extract_lines(pre, MIXED, Settings())
    assert_valid_skeleton(lm)
    assert components(lm.skeleton) == 1  # all outlined shapes touch
    widths: dict[str, float] = truth["stroke_widths_px"]  # type: ignore[assignment]
    assert abs(lm.median_stroke_width / scale - widths["outlines"]) <= 1.0
    # Noisy sky, dark-green hill and navy roof interiors are not strokes.
    for x, y in [(320, 60), (500, 440), (130, 318), (130, 390)]:
        sy, sx = round(y * scale), round(x * scale)
        r = round(4 * scale)
        assert not lm.mask[sy - r : sy + r + 1, sx - r : sx + r + 1].any(), (x, y)


# ------------------------------------------------------------------ synthetic behaviour


def _canvas(h: int = 120, w: int = 120) -> np.ndarray:
    return np.full((h, w, 3), 255, np.uint8)


def test_small_gap_at_t_junction_is_bridged() -> None:
    img = _canvas()
    img[29:32, 10:110] = 0  # bar: rows 29..31
    img[35:110, 59:62] = 0  # stem, 3-px gap (rows 32..34) below the bar
    lm = extract_lines(make_pre(img), LINE_ART, Settings())
    assert_valid_skeleton(lm)
    assert components(lm.skeleton) == 1
    assert components(lm.mask) == 1
    # Same at scale 2: the gap limit is in source px.
    lm2 = extract_lines(make_pre(img, scale=2.0), LINE_ART, Settings())
    assert components(lm2.skeleton) == 1


def test_large_gap_is_not_bridged() -> None:
    img = _canvas()
    img[29:32, 10:110] = 0
    img[42:110, 59:62] = 0  # 10-px gap
    lm = extract_lines(make_pre(img), LINE_ART, Settings())
    assert components(lm.skeleton) == 2


def test_y_junction_connected_and_spurs_pruned() -> None:
    pil = Image.new("RGB", (480, 480), (255, 255, 255))
    draw = ImageDraw.Draw(pil)
    for end in [(80, 80), (400, 80), (240, 440)]:
        draw.line([(240, 240), end], fill=(0, 0, 0), width=28)  # 7 px after 4x box downsampling
    img = np.asarray(pil.resize((120, 120), Image.Resampling.BOX)).copy()
    lm = extract_lines(make_pre(img), LINE_ART, Settings())
    assert_valid_skeleton(lm)
    assert components(lm.skeleton) == 1
    assert len(L.endpoints(lm.skeleton)) == 3
    assert abs(lm.median_stroke_width - 7) <= 1


def test_spur_pruning_removes_short_branch_only() -> None:
    sk = np.zeros((30, 40), bool)
    sk[15, 5:35] = True  # main line
    sk[12:15, 20] = True  # 3-px spur
    sk[1:15, 25] = True  # 14-px real branch (the 9-px tail to its right is kept too)
    width = np.where(sk, 4.0, 0.0).astype(np.float32)
    out = L.prune_spurs(sk, width, 1.5)
    assert not out[12:15, 20].any()
    assert out[1:15, 25].all()
    assert out[15, 5:35].all()
    assert np.array_equal(L.prune_spurs(np.zeros((5, 5), bool), np.zeros((5, 5), np.float32), 1.5), np.zeros((5, 5)))


def test_redundant_staircase_pixels_removed() -> None:
    sk = np.zeros((10, 10), bool)
    sk[2, 2:5] = True
    sk[3, 4:8] = True  # L-shaped step: (2,4) and (3,4) make a redundant corner
    out = L.remove_redundant_pixels(sk)
    assert out.sum() == sk.sum() - 1
    assert components(out) == 1


def test_noise_specks_removed() -> None:
    img = _canvas()
    cv2.line(img, (10, 60), (110, 60), (0, 0, 0), 3)
    rng = np.random.default_rng(0)
    for y, x in rng.integers(5, 115, size=(30, 2)):
        if abs(y - 60) > 6:
            img[y, x] = 0
    lm = extract_lines(make_pre(img), LINE_ART, Settings(detail_level=DetailLevel.LOW))
    assert components(lm.mask) == 1
    assert components(lm.skeleton) == 1


def test_transparent_pixels_excluded() -> None:
    img = _canvas()
    cv2.line(img, (10, 60), (110, 60), (0, 0, 0), 5)
    alpha = np.full(img.shape[:2], 255, np.uint8)
    alpha[:, 60:] = 0
    lm = extract_lines(make_pre(img, alpha), LINE_ART, Settings())
    assert lm.mask.any()
    assert not lm.mask[:, 60:].any()
    assert not lm.skeleton[:, 60:].any()


def test_transparent_background_line_art_uses_alpha() -> None:
    img = np.zeros((100, 100, 3), np.uint8)
    alpha = np.zeros((100, 100), np.uint8)
    cv2.circle(alpha, (50, 50), 30, 255, 4)
    img[alpha > 0] = (30, 60, 90)
    lm = extract_lines(make_pre(img, alpha), LINE_ART, Settings())
    assert components(lm.skeleton) == 1
    assert lm.color_rgb == (30, 60, 90)
    assert abs(lm.median_stroke_width - 5) <= 1.5


def test_colored_strokes_kept_in_line_art_but_fill_blobs_removed() -> None:
    img = _canvas(200, 200)
    cv2.line(img, (10, 20), (190, 20), (210, 139, 38), 2)  # blue stroke (BGR->RGB irrelevant)
    cv2.circle(img, (100, 120), 60, (0, 0, 0), -1)  # solid disk, 120 px wide: a fill, not a stroke
    lm = extract_lines(make_pre(img), LINE_ART, Settings())
    assert lm.mask[18:23, 20:180].any(axis=0).all()
    assert not lm.mask[100:140, 80:120].any()


def test_empty_and_uniform_images() -> None:
    blank = make_pre(_canvas())
    for cls in (LINE_ART, MIXED, FLAT):
        lm = extract_lines(blank, cls, Settings())
        assert not lm.mask.any() and not lm.skeleton.any()
        assert lm.median_stroke_width == 0.0
        assert lm.color_rgb == (0, 0, 0)
    # Fully transparent image
    lm = extract_lines(make_pre(_canvas(), np.zeros((120, 120), np.uint8)), MIXED, Settings())
    assert not lm.mask.any()
    # Uniform black image: a "stroke" wider than any real stroke is a fill -> nothing.
    lm = extract_lines(make_pre(np.zeros((50, 50, 3), np.uint8)), MIXED, Settings())
    assert not lm.mask.any()


def test_mixed_ignores_light_only_images() -> None:
    img = _canvas()
    img[:, :60] = (250, 200, 30)
    img[:, 60:] = (60, 160, 80)
    lm = extract_lines(make_pre(img), MIXED, Settings())
    assert not lm.mask.any()


def test_fixture_preprocess_result_flat_color_centerline() -> None:
    pre = fx.make_preprocess_result(64, has_alpha=True)
    lm = extract_lines(pre, FLAT, Settings(line_mode=LineMode.CENTERLINE))
    assert isinstance(lm, LineMap)
    assert lm.mask.shape == (64, 64)
    assert not lm.mask[:4].any()  # transparent rows


def test_bridge_gaps_helper_noop_cases() -> None:
    mask = np.zeros((20, 20), bool)
    out, n = L.bridge_gaps(mask, mask, np.zeros((20, 20), np.float32), 3)
    assert n == 0 and out is mask
    mask[5, 2:18] = True
    out, n = L.bridge_gaps(mask, mask, mask.astype(np.float32), 3)
    assert n == 0


def test_dominant_color_and_ink_estimation_edge_cases() -> None:
    rgb = _canvas(10, 10)
    none = np.zeros((10, 10), bool)
    assert tuple(L.dominant_color(rgb, none)) == (255, 255, 255)
    assert L.estimate_ink_color(rgb, none) is None
    assert L.estimate_ink_color(rgb, ~none) is None


# ------------------------------------------------------------------ performance


@pytest.mark.slow
def test_runtime_2000x2000() -> None:
    pre02, _ = load_sample("02_")
    big = cv2.resize(np.asarray(pre02.image), (2000, 2000), interpolation=cv2.INTER_CUBIC)
    pre_big = make_pre(big)
    pre09, _ = load_sample("09_")
    extract_lines(make_pre(_canvas()), LINE_ART, Settings())  # warm-up (imports, LUTs)
    for pre, cls in [(pre_big, LINE_ART), (pre09, MIXED), (pre09, LINE_ART)]:
        t0 = time.perf_counter()
        lm = extract_lines(pre, cls, Settings(line_mode=LineMode.CENTERLINE))
        elapsed = time.perf_counter() - t0
        assert elapsed <= 1.5, (cls.label, elapsed)
        assert lm.mask.shape == (2000, 2000)
