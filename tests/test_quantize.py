"""Tests for pipeline.quantize (color quantization stage).

Upstream stages are mocked: PreprocessResult objects are built directly from the sample
files (scale 1.0, no denoise) or from contracts.fixtures.
"""

from __future__ import annotations

import json
import time
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from contracts.fixtures import make_image_class, make_preprocess_result
from contracts.schemas import (
    DenoiseParams,
    DetailLevel,
    ImageClass,
    ImageClassLabel,
    ImageInput,
    ImageMode,
    Palette,
    PreprocessResult,
    Settings,
    SourceFormat,
    StageError,
)
from pipeline import quantize as q
from pipeline.quantize import quantize

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
MANIFEST = {e["file"]: e for e in (json.loads(p.read_text(encoding="utf-8")) for p in sorted(SAMPLES.glob("*.json")))}
WITH_PALETTE = [f for f, e in MANIFEST.items() if e["palette_hex"]]
FLAT = ImageClass(label=ImageClassLabel.FLAT_COLOR, confidence=1.0)


# ----------------------------------------------------------------------------- helpers


def make_pre(
    rgb: np.ndarray, alpha: np.ndarray | None = None, fmt: SourceFormat = SourceFormat.PNG
) -> PreprocessResult:
    """Wrap raw arrays in a PreprocessResult (processing space == source space)."""
    h, w = rgb.shape[:2]
    return PreprocessResult(
        source=ImageInput(
            path=Path("synthetic.png"),
            width=w,
            height=h,
            has_alpha=alpha is not None,
            mode=ImageMode.RGBA if alpha is not None else ImageMode.RGB,
            source_format=fmt,
            file_size_bytes=1,
        ),
        image=np.ascontiguousarray(rgb, dtype=np.uint8),
        alpha=None if alpha is None else np.ascontiguousarray(alpha, dtype=np.uint8),
        scale_factor=1.0,
        denoise=DenoiseParams(method="none", strength=0.0),
    )


@cache
def load_sample(name: str) -> PreprocessResult:
    with Image.open(SAMPLES / name) as img:
        arr = np.asarray(img.convert("RGBA" if "A" in img.getbands() else "RGB"))
    alpha = arr[..., 3] if arr.shape[2] == 4 else None
    fmt = SourceFormat.JPEG if name.endswith(".jpg") else SourceFormat.PNG
    return make_pre(arr[..., :3], alpha, fmt)


def sample_class(name: str) -> ImageClass:
    return ImageClass(label=ImageClassLabel(MANIFEST[name]["expected_class"]), confidence=1.0)


@cache
def run_sample(name: str) -> tuple[Palette, float]:
    start = time.perf_counter()
    palette = quantize(load_sample(name), sample_class(name), Settings())
    return palette, time.perf_counter() - start


def gt_lab(name: str) -> np.ndarray:
    rgbs = np.array([q._hex_to_rgb(h) for h in MANIFEST[name]["palette_hex"]], dtype=np.uint8)
    return q._rgb_to_lab(rgbs)


def palette_lab(palette: Palette) -> np.ndarray:
    return np.array([c.lab for c in palette.colors])


def min_component_area(labels: np.ndarray) -> int:
    """Smallest region area, with regions bridged across 1-px gaps (the quantizer's speckle notion)."""
    best = labels.size
    for c in np.unique(labels[labels >= 0]):
        mask = (labels == c).astype(np.uint8)
        n, bridged = cv2.connectedComponents(cv2.dilate(mask, np.ones((3, 3), np.uint8)), connectivity=8)
        area = np.bincount(bridged[mask.astype(bool)], minlength=n)
        best = min(best, int(area[area > 0].min()))
    return best


# ----------------------------------------------------------------------------- performance (runs first)


@pytest.mark.slow
@pytest.mark.golden
def test_large_image_under_two_seconds() -> None:
    """Best of 3 runs. Defined first in the module on purpose: on laptops the CPU drops to a
    much lower clock after a few seconds of sustained load (turbo budget), which would make
    this wall-clock check measure the thermal state left by earlier tests, not the code."""
    pre = load_sample("09_large_2000.png")
    quantize(make_pre(np.zeros((8, 8, 3), np.uint8)), FLAT, Settings())  # warm imports
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        palette = quantize(pre, FLAT, Settings())
        timings.append(time.perf_counter() - start)
    assert min(timings) <= 2.0, f"took {timings}"
    assert len(palette.colors) == 7


# ----------------------------------------------------------------------------- golden samples


@pytest.mark.golden
@pytest.mark.parametrize("name", WITH_PALETTE)
def test_ground_truth_colors_matched(name: str) -> None:
    palette, _ = run_sample(name)
    de = q._delta_e2000(gt_lab(name)[:, None, :], palette_lab(palette)[None, :, :])
    best = de.min(axis=1)
    assert best.max() < 3.0, f"unmatched ground-truth colors: {best}"
    assert best.mean() < 2.0
    matched = set(de.argmin(axis=1).tolist())
    total = sum(c.pixel_count for c in palette.colors)
    extras = [c.hex for c in palette.colors if c.index not in matched and c.pixel_count / total > 0.005]
    assert not extras, f"extra colors above 0.5% share: {extras}"


@pytest.mark.golden
@pytest.mark.parametrize("name", ["01_logo_4color.png", "06_jpeg_artifacts.jpg"])
def test_logo_yields_exactly_four_colors(name: str) -> None:
    palette, _ = run_sample(name)
    assert len(palette.colors) == 4


@pytest.mark.golden
@pytest.mark.parametrize("name", WITH_PALETTE)
def test_color_count_equals_ground_truth(name: str) -> None:
    palette, _ = run_sample(name)
    assert len(palette.colors) == len(MANIFEST[name]["palette_hex"])


@pytest.mark.golden
@pytest.mark.parametrize("name", sorted(MANIFEST))
def test_every_sample_produces_valid_palette(name: str) -> None:
    palette, _ = run_sample(name)
    pre = load_sample(name)
    assert palette.label_map.shape == (pre.height, pre.width)
    assert np.array_equal(palette.label_map >= 0, pre.opaque_mask)
    assert all(c.pixel_count > 0 for c in palette.colors)
    counts = [c.pixel_count for c in palette.colors]
    assert counts == sorted(counts, reverse=True)
    assert min_component_area(np.asarray(palette.label_map)) >= Settings().preset.speckle_min_area_px


@pytest.mark.golden
def test_line_art_is_ink_and_paper() -> None:
    palette, _ = run_sample("02_lineart_black.png")
    assert sorted(c.hex for c in palette.colors) == ["#000000", "#ffffff"]
    assert palette.colors[palette.background_index or 0].hex == "#ffffff"


@pytest.mark.golden
def test_thin_one_pixel_lines_survive() -> None:
    palette, _ = run_sample("08_thin_lines.png")
    pre = load_sample("08_thin_lines.png")
    hexes = [c.hex for c in palette.colors]
    black = hexes.index("#000000")
    src_black = np.all(np.asarray(pre.image) == 0, axis=2)
    agree = (np.asarray(palette.label_map)[src_black] == black).mean()
    assert agree > 0.99


@pytest.mark.golden
def test_background_detection() -> None:
    logo, _ = run_sample("01_logo_4color.png")
    assert logo.background_index is not None
    assert logo.colors[logo.background_index].hex == "#ffffff"
    transparent, _ = run_sample("07_transparent_logo.png")
    assert transparent.background_index is None


# ----------------------------------------------------------------------------- transparency


def test_transparent_pixels_get_minus_one_and_contribute_no_color() -> None:
    base = load_sample("07_transparent_logo.png")
    rgb = np.asarray(base.image).copy()
    alpha = np.asarray(base.alpha)
    rgb[alpha < 128] = (0, 255, 0)  # vivid green hidden under transparency
    palette = quantize(make_pre(rgb, alpha), FLAT, Settings())
    labels = np.asarray(palette.label_map)
    assert np.array_equal(labels == -1, alpha < 128)
    greens = [
        c.hex
        for c in palette.colors
        if q._delta_e2000(np.array(c.lab), q._rgb_to_lab(np.array([[0, 255, 0]], np.uint8))[0]) < 30
    ]
    assert not greens
    assert sorted(c.hex for c in palette.colors) == sorted(MANIFEST["07_transparent_logo.png"]["palette_hex"])


def test_fully_transparent_image_raises() -> None:
    rgb = np.full((10, 10, 3), 200, np.uint8)
    with pytest.raises(StageError):
        quantize(make_pre(rgb, np.zeros((10, 10), np.uint8)), FLAT, Settings())


def test_contract_fixture_with_alpha() -> None:
    pre = make_preprocess_result(has_alpha=True)
    palette = quantize(pre, make_image_class(), Settings())
    assert np.array_equal(np.asarray(palette.label_map) == -1, ~pre.opaque_mask)
    assert sorted(c.hex for c in palette.colors) == ["#000000", "#dc322f", "#ffffff"]


def test_speckle_island_surrounded_by_transparency_is_kept() -> None:
    rgb = np.full((40, 40, 3), 255, np.uint8)
    alpha = np.full((40, 40), 255, np.uint8)
    alpha[:, 20:] = 0
    rgb[5:7, 30:32] = (220, 50, 47)
    alpha[5:7, 30:32] = 255  # 4-px opaque island, no opaque neighbour
    palette = quantize(make_pre(rgb, alpha), FLAT, Settings())
    labels = np.asarray(palette.label_map)
    island = labels[5:7, 30:32]
    assert (island >= 0).all()
    assert palette.colors[int(island[0, 0])].hex == "#dc322f"


# ----------------------------------------------------------------------------- anti-aliasing


def test_antialiased_pixels_join_a_neighbouring_color() -> None:
    name = "01_logo_4color.png"
    pre = load_sample(name)
    palette, _ = run_sample(name)
    rgb = np.asarray(pre.image)
    labels = np.asarray(palette.label_map)
    gts = [q._hex_to_rgb(h) for h in MANIFEST[name]["palette_hex"]]
    kernel = np.ones((9, 9), np.uint8)
    exact_any = np.zeros(labels.shape, bool)
    near = {}
    for rgb_gt in gts:
        exact = np.all(rgb == rgb_gt, axis=2)
        exact_any |= exact
        near[rgb_gt] = cv2.dilate(exact.astype(np.uint8), kernel).astype(bool)
    aa = ~exact_any
    assert aa.sum() > 500  # the sample really is anti-aliased
    label_rgb = np.array([c.rgb for c in palette.colors])[labels]
    ok = np.zeros(labels.shape, bool)
    for rgb_gt in gts:
        ok |= np.all(label_rgb == rgb_gt, axis=2) & near[rgb_gt]
    assert ok[aa].all(), "an AA pixel took a color that is not present next to it"


def test_one_pixel_blend_column_is_not_a_color() -> None:
    rgb = np.full((60, 60, 3), 255, np.uint8)
    rgb[:, :30] = (220, 50, 47)
    rgb[:, 30] = (238, 153, 151)  # 50% red/white anti-aliased column
    palette = quantize(make_pre(rgb), FLAT, Settings())
    assert sorted(c.hex for c in palette.colors) == ["#dc322f", "#ffffff"]


def test_thin_mixture_band_is_dropped() -> None:
    rgb = np.full((80, 80, 3), 255, np.uint8)
    rgb[:, :40] = (38, 139, 210)
    rgb[:, 40:42] = (147, 197, 233)  # 2-px band of an exact blue/white blend: pure, but a mixture cluster
    palette = quantize(make_pre(rgb), FLAT, Settings())
    assert sorted(c.hex for c in palette.colors) == ["#268bd2", "#ffffff"]


def test_is_blend_detects_mixtures_only() -> None:
    rgbs = np.array([[255, 255, 255], [220, 50, 47], [238, 153, 151], [38, 139, 210]], np.uint8)
    lab = q._rgb_to_lab(rgbs)
    assert q._is_blend(2, lab)
    assert not q._is_blend(3, lab)
    assert not q._is_blend(0, q._rgb_to_lab(np.array([[0, 0, 0], [0, 0, 0], [9, 9, 9]], np.uint8)))


def test_wide_blend_colored_region_is_kept() -> None:
    rgb = np.full((90, 90, 3), 255, np.uint8)
    rgb[:, :30] = (220, 50, 47)
    rgb[:, 30:60] = (238, 153, 151)  # same blend color, but a solid region -> real color
    palette = quantize(make_pre(rgb), FLAT, Settings())
    assert len(palette.colors) == 3


# ----------------------------------------------------------------------------- colors


def test_final_color_is_median_not_mean() -> None:
    rgb = np.full((60, 60, 3), 255, np.uint8)
    region = np.full((60, 30, 3), (200, 40, 40), np.uint8)
    region[::5] = (230, 60, 60)  # 20% lighter rows: shifts the mean, not the median
    rgb[:, :30] = region
    palette = quantize(make_pre(rgb), FLAT, Settings())
    assert "#c82828" in [c.hex for c in palette.colors]


def test_weighted_median_lab() -> None:
    ulab = np.array([[10.0, 0, 0], [20.0, 5, 5], [90.0, 9, 9]])
    med = q._weighted_median_lab(ulab, np.array([5, 1, 3]))
    assert med.tolist() == [10.0, 0.0, 0.0]


def test_merge_close_centers() -> None:
    centers = np.array([[50.0, 0, 0], [51.0, 0, 0], [90.0, 0, 0]])
    merged = q._merge_close(centers, np.array([3.0, 1.0, 1.0]))
    assert merged.shape == (2, 3)
    assert merged[0, 0] == pytest.approx(50.25)


def test_near_duplicate_colors_are_merged() -> None:
    rgb = np.full((60, 60, 3), 255, np.uint8)
    rgb[:, :30] = (200, 40, 40)
    rgb[:, 15:30] = (202, 41, 41)  # ΔE2000 < 3 from its neighbour
    palette = quantize(make_pre(rgb), FLAT, Settings())
    assert len(palette.colors) == 2


def test_single_color_image() -> None:
    palette = quantize(make_pre(np.full((20, 30, 3), (12, 34, 56), np.uint8)), FLAT, Settings())
    assert len(palette.colors) == 1
    color = palette.colors[0]
    assert color.hex == "#0c2238" and color.is_background and color.pixel_count == 600


def test_output_is_deterministic() -> None:
    pre = load_sample("06_jpeg_artifacts.jpg")
    a = quantize(pre, FLAT, Settings())
    b = quantize(pre, FLAT, Settings())
    assert [c.model_dump() for c in a.colors] == [c.model_dump() for c in b.colors]
    assert np.array_equal(a.label_map, b.label_map)


def test_lab_field_is_lab_of_rgb() -> None:
    palette, _ = run_sample("03_cartoon_outlined.png")
    for c in palette.colors:
        expected = q._rgb_to_lab(np.array([c.rgb], np.uint8))[0]
        assert np.allclose(c.lab, expected)


# ----------------------------------------------------------------------------- settings


def test_max_colors_caps_auto_detection() -> None:
    pre = load_sample("03_cartoon_outlined.png")
    palette = quantize(pre, FLAT, Settings(max_colors=3))
    assert 1 <= len(palette.colors) <= 3


def test_noise_image_respects_class_cap() -> None:
    rgb = np.random.default_rng(1).integers(0, 256, (48, 48, 3), dtype=np.uint8)
    palette = quantize(make_pre(rgb), ImageClass(label=ImageClassLabel.LINE_ART, confidence=1.0), Settings())
    assert len(palette.colors) <= q._AUTO_MAX_COLORS[ImageClassLabel.LINE_ART]


def test_fit_points_grid_coarsens_when_too_many(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(q, "_MAX_FIT_POINTS", 50)
    lab = np.random.default_rng(2).uniform(0, 100, (2000, 3))
    pts, wts = q._fit_points(lab, np.ones(len(lab)))
    assert len(pts) <= 50
    assert wts.sum() == pytest.approx(2000)


@pytest.mark.parametrize(("level", "kept"), [(DetailLevel.HIGH, True), (DetailLevel.MEDIUM, False)])
def test_speckles_absorbed_by_detail_level(level: DetailLevel, kept: bool) -> None:
    rgb = np.full((80, 80, 3), 255, np.uint8)
    rgb[:, 40:] = (38, 139, 210)
    for y in range(5, 75, 10):
        rgb[y : y + 3, 10:13] = (0, 0, 0)  # 3x3 = 9 px black dots in the white half
    palette = quantize(make_pre(rgb), FLAT, Settings(detail_level=level))
    hexes = [c.hex for c in palette.colors]
    assert ("#000000" in hexes) == kept
    if not kept:
        labels = np.asarray(palette.label_map)
        assert palette.colors[int(labels[6, 11])].hex == "#ffffff"


def test_speckle_goes_to_closest_colored_neighbour() -> None:
    labels = np.zeros((20, 20), np.int32)
    labels[:, 10:] = 1
    labels[9:11, 9:11] = 2  # 4-px speckle touching both regions
    colors = q._rgb_to_lab(np.array([[255, 255, 255], [220, 50, 47], [200, 40, 40]], np.uint8))
    out = q._remove_speckles(labels, colors, 16)
    assert set(np.unique(out[9:11, 9:11]).tolist()) == {1}
    assert q._remove_speckles(labels, colors, 1) is labels


def test_palette_override() -> None:
    pre = load_sample("01_logo_4color.png")
    override = ["#00ff00", "#ffffff", "#e00000", "#2080d0", "#ffd000"]
    palette = quantize(pre, FLAT, Settings(palette_override=override, max_colors=2))
    hexes = [c.hex for c in palette.colors]
    assert hexes == override[1:]  # override order kept, unused green dropped, max_colors ignored
    assert palette.colors[2].rgb == (32, 128, 208)  # exact override color, not a median
    assert palette.colors[0].is_background
    assert min_component_area(np.asarray(palette.label_map)) >= Settings().preset.speckle_min_area_px


def test_palette_override_respects_transparency() -> None:
    pre = load_sample("07_transparent_logo.png")
    palette = quantize(pre, FLAT, Settings(palette_override=["#000000"]))
    assert [c.hex for c in palette.colors] == ["#000000"]
    assert np.array_equal(np.asarray(palette.label_map) == -1, ~pre.opaque_mask)


def test_nested_speckles_resolve_in_later_rounds() -> None:
    labels = np.zeros((20, 20), np.int32)
    labels[8:11, 8:11] = 1  # 8-px ring of label 1 ...
    labels[9, 9] = 2  # ... around a 1-px speckle that only touches the ring
    colors = q._rgb_to_lab(np.array([[255, 255, 255], [38, 139, 210], [0, 0, 0]], np.uint8))
    out = q._remove_speckles(labels, colors, 16)
    assert (out == 0).all()


def test_crossed_thin_line_is_not_a_speckle() -> None:
    labels = np.zeros((30, 30), np.int32)
    labels[15, :] = 1  # 1-px line ...
    labels[:, 5::6] = 2  # ... cut every 6 px by crossing 1-px lines of another color
    colors = q._rgb_to_lab(np.array([[255, 255, 255], [0, 0, 0], [38, 139, 210]], np.uint8))
    out = q._remove_speckles(labels, colors, 16)
    assert np.array_equal(out, labels)
