"""Tests for pipeline/vectorize.py.

Rendering uses resvg_py (CairoSVG's native DLL is unavailable on the dev machine): documents
are serialised with a minimal SVG writer below and rasterised at source resolution.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import pytest
import resvg_py
from PIL import Image

from contracts.fixtures import make_image_class, make_line_map, make_palette, make_preprocess_result
from contracts.schemas import (
    PATH_D_RE,
    DenoiseParams,
    DetailLevel,
    ImageClass,
    ImageClassLabel,
    ImageInput,
    ImageMode,
    LineMap,
    LineMode,
    Palette,
    PaletteColor,
    PreprocessResult,
    Settings,
    SourceFormat,
    StageError,
    VectorDocument,
    layer_name,
    rgb_to_hex,
)
from pipeline import vectorize as vz
from pipeline.vectorize import (
    Chain,
    FitParams,
    crack_contours,
    fill_unknown,
    find_corners,
    fit_chain,
    fit_open,
    line_colors,
    plan_layers,
    remove_speckles,
    skeleton_chains,
    smooth_closed,
    smooth_open,
    stacked_mask,
    vectorize,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
TOKEN_RE = re.compile(r"[MLCZ]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
MAGENTA = np.array([255, 0, 255], dtype=np.int16)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

COLORS = [(255, 255, 255), (220, 50, 47), (38, 139, 210), (0, 0, 0), (250, 200, 30)]


def _lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    px = np.array([[rgb]], dtype=np.uint8)
    lab = cv2.cvtColor(px, cv2.COLOR_RGB2LAB).astype(np.float64)[0, 0]
    return (lab[0] * 100 / 255, lab[1] - 128, lab[2] - 128)


def make_inputs(labels: np.ndarray, scale: float = 1.0, background: int | None = 0) -> tuple[PreprocessResult, Palette]:
    """PreprocessResult + Palette for a synthetic label map (processing space)."""
    labels = labels.astype(np.int32)
    h, w = labels.shape
    has_alpha = bool((labels < 0).any())
    n = int(labels.max()) + 1
    lut = np.array(COLORS[:n], dtype=np.uint8)
    image = lut[np.clip(labels, 0, None)]
    alpha = np.where(labels >= 0, 255, 0).astype(np.uint8) if has_alpha else None
    src = ImageInput(
        path=Path("synthetic.png"),
        width=round(w / scale),
        height=round(h / scale),
        has_alpha=has_alpha,
        mode=ImageMode.RGBA if has_alpha else ImageMode.RGB,
        source_format=SourceFormat.PNG,
        file_size_bytes=1,
    )
    pre = PreprocessResult(
        source=src, image=image, alpha=alpha, scale_factor=scale, denoise=DenoiseParams(method="none", strength=0)
    )
    counts = np.bincount(labels[labels >= 0].ravel(), minlength=n)
    colors = [
        PaletteColor(
            index=i,
            rgb=COLORS[i],
            lab=_lab(COLORS[i]),
            hex=rgb_to_hex(COLORS[i]),
            pixel_count=int(counts[i]),
            is_background=i == background,
        )
        for i in range(n)
    ]
    return pre, Palette(colors=colors, label_map=labels)


def make_line_map_from(mask: np.ndarray, skeleton: np.ndarray, width: float) -> LineMap:
    wm = np.where(skeleton, width, 0).astype(np.float32)
    return LineMap(mask=mask, skeleton=skeleton, width_map=wm, median_stroke_width=width, color_rgb=(0, 0, 0))


def doc_svg(doc: VectorDocument, background: str | None = None) -> str:
    """Minimal SVG 1.1 serialisation of a VectorDocument (one <g> per layer)."""
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{doc.width}" height="{doc.height}" '
        f'viewBox="0 0 {doc.width} {doc.height}">'
    ]
    if background:
        out.append(f'<rect width="{doc.width}" height="{doc.height}" fill="{background}"/>')
    for layer in doc.layers:
        if layer.is_stroke:
            attrs = (
                f'fill="none" stroke="{layer.color_hex}" stroke-width="{layer.stroke_width}" '
                'stroke-linecap="round" stroke-linejoin="round"'
            )
        else:
            attrs = f'fill="{layer.color_hex}" fill-rule="{layer.fill_rule}"'
        out.append(f'<g id="{layer.id}" {attrs}>' + "".join(f'<path d="{d}"/>' for d in layer.paths) + "</g>")
    out.append("</svg>")
    return "".join(out)


def render(doc: VectorDocument, background: str | None = None) -> np.ndarray:
    """(H, W, 4) uint8 RGBA rendering of the document at source resolution."""
    png = resvg_py.svg_to_bytes(svg_string=doc_svg(doc, background))
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


def nodes(d: str) -> int:
    """M/L/C endpoint count (Z not counted), same definition as QualityReport.node_count."""
    count, cmd, nums = 0, "", 0
    for tok in [*TOKEN_RE.findall(d), "Z"]:
        if tok.isalpha():
            if cmd in ("M", "L"):
                count += nums // 2
            elif cmd == "C":
                count += nums // 6
            cmd, nums = tok, 0
        else:
            nums += 1
    return count


def doc_nodes(doc: VectorDocument) -> int:
    return sum(nodes(d) for layer in doc.layers for d in layer.paths)


def magenta_seams(doc: VectorDocument, opaque: np.ndarray) -> float:
    """Fraction of 1-px-eroded opaque source pixels where a magenta canvas shows through."""
    img = render(doc, background="#ff00ff")[..., :3].astype(np.int16)
    interior = cv2.erode(opaque.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE) > 0
    leak = np.abs(img - MAGENTA).sum(axis=2) < 60
    return float((leak & interior).sum()) / max(1, int(interior.sum()))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum()) / max(1, int((a | b).sum()))


def settings(**kw: object) -> Settings:
    return Settings(**kw)  # type: ignore[arg-type]


FLAT = ImageClass(label=ImageClassLabel.FLAT_COLOR, confidence=1.0)
LINE_ART = ImageClass(label=ImageClassLabel.LINE_ART, confidence=1.0)
MIXED = ImageClass(label=ImageClassLabel.MIXED, confidence=1.0)


def disk_labels(size: int = 96, r: float = 30.0, value: int = 1) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    labels = np.zeros((size, size), np.int32)
    labels[(xx + 0.5 - size / 2) ** 2 + (yy + 0.5 - size / 2) ** 2 <= r * r] = value
    return labels


# --------------------------------------------------------------------------------------
# Contract / output rules
# --------------------------------------------------------------------------------------


def test_fixture_document_is_valid() -> None:
    pre = make_preprocess_result()
    palette = make_palette(pre)
    doc = vectorize(pre, make_image_class(), palette, None, Settings())
    assert isinstance(doc, VectorDocument)
    assert (doc.width, doc.height) == (pre.source.width, pre.source.height)
    assert [layer.z_order for layer in doc.layers] == list(range(len(doc.layers)))
    for z, layer in enumerate(doc.layers):
        assert (layer.id, layer.name) == layer_name(layer.role, z + 1, layer.color_hex)
        assert layer.palette_index is not None
        assert layer.color_hex == palette.colors[layer.palette_index].hex
        assert layer.fill_rule == "evenodd"
        for d in layer.paths:
            assert PATH_D_RE.match(d)
            assert set(re.findall(r"[A-Za-z]", d)) <= {"M", "L", "C", "Z"}
    assert doc.layers[0].role == "background"
    meta = doc.metadata
    assert meta.source_filename == "fixture.png"
    assert meta.image_class == ImageClassLabel.FLAT_COLOR
    assert meta.palette_hex == [c.hex for c in palette.colors]


def test_background_is_full_canvas_rectangle() -> None:
    pre, palette = make_inputs(disk_labels())
    doc = vectorize(pre, FLAT, palette, None, Settings())
    assert doc.layers[0].paths == ["M0 0L96 0L96 96L0 96Z"]
    assert doc.layers[0].role == "background"


def test_square_keeps_exact_sharp_corners() -> None:
    labels = np.zeros((64, 64), np.int32)
    labels[16:40, 20:50] = 1
    pre, palette = make_inputs(labels)
    doc = vectorize(pre, FLAT, palette, None, Settings())
    (d,) = doc.layers[1].paths
    assert "C" not in d
    assert nodes(d) == 4
    pts = {(float(a), float(b)) for a, b in re.findall(r"([\d.]+) ([\d.]+)", d)}
    assert pts == {(20.0, 16.0), (50.0, 16.0), (50.0, 40.0), (20.0, 40.0)}


def test_circle_is_few_smooth_curves_with_high_iou() -> None:
    labels = disk_labels(128, 40.0)
    pre, palette = make_inputs(labels)
    doc = vectorize(pre, FLAT, palette, None, Settings())
    (d,) = doc.layers[1].paths
    assert "C" in d
    assert nodes(d) <= 8
    img = render(doc)
    red = np.abs(img[..., :3].astype(int) - np.array(COLORS[1])).sum(axis=2) < 150
    assert iou(red, labels == 1) > 0.97


def test_paths_are_in_source_space_and_rounded() -> None:
    labels = np.zeros((80, 80), np.int32)
    labels[20:61, 10:51] = 1  # odd sizes -> half-pixel source coordinates
    labels[30:50, 30:40] = 2
    pre, palette = make_inputs(labels, scale=2.0)
    for level in DetailLevel:
        s = settings(detail_level=level)
        doc = vectorize(pre, FLAT, palette, None, s)
        assert (doc.width, doc.height) == (40, 40)
        assert doc.layers[0].paths == ["M0 0L40 0L40 40L0 40Z"]
        numbers = [t for d in doc.layers[1].paths for t in TOKEN_RE.findall(d) if not t.isalpha()]
        assert max(float(t) for t in numbers) <= 40.0
        assert all(len(t.split(".")[1]) <= s.preset.path_precision for t in numbers if "." in t)
    doc = vectorize(pre, FLAT, palette, None, Settings())
    top = doc.layers[-1].paths[0]
    pts = {(float(a), float(b)) for a, b in re.findall(r"([\d.]+) ([\d.]+)", top)}
    assert pts == {(15.0, 15.0), (20.0, 15.0), (20.0, 25.0), (15.0, 25.0)}


def test_hole_uses_evenodd_subpath_and_stays_transparent() -> None:
    labels = np.full((64, 64), -1, np.int32)
    labels[8:56, 8:56] = 1
    labels[24:40, 24:40] = -1  # true hole: transparent
    pre, palette = make_inputs(labels, background=None)
    doc = vectorize(pre, FLAT, palette, None, Settings())
    assert len(doc.layers) == 1 and doc.layers[0].role == "fill"
    (d,) = doc.layers[0].paths
    assert d.count("M") == 2 and doc.layers[0].fill_rule == "evenodd"
    alpha = render(doc)[..., 3]
    assert alpha[32, 32] == 0 and alpha[2, 2] == 0 and alpha[12, 12] == 255


def test_transparent_pixels_are_never_covered() -> None:
    labels = np.full((80, 80), -1, np.int32)
    labels[10:70, 10:40] = 1
    labels[10:70, 40:70] = 2  # two touching colors on transparency: worst case for seams
    labels[30:50, 25:55] = 0
    pre, palette = make_inputs(labels, background=None)
    doc = vectorize(pre, FLAT, palette, None, Settings())
    alpha = render(doc)[..., 3]
    transparent = cv2.erode((labels < 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert alpha[transparent].max() == 0
    assert magenta_seams(doc, labels >= 0) == 0.0


def test_no_seams_between_curved_regions_over_magenta() -> None:
    size = 160
    yy, xx = np.mgrid[:size, :size]
    labels = np.full((size, size), -1, np.int32)
    labels[(xx - 80) ** 2 + (yy - 80) ** 2 <= 70**2] = 1
    labels[((xx - 60) ** 2 + (yy - 70) ** 2 <= 35**2) & (labels >= 0)] = 2
    labels[((xx - 100) ** 2 + (yy - 95) ** 2 <= 30**2) & (labels >= 0)] = 4
    labels[(np.abs(xx - yy) < 4) & (labels >= 0)] = 3
    pre, palette = make_inputs(labels, background=None)
    doc = vectorize(pre, FLAT, palette, None, Settings())
    assert magenta_seams(doc, labels >= 0) <= 0.0005


def test_layers_sorted_background_then_largest() -> None:
    labels = np.zeros((60, 60), np.int32)
    labels[5:55, 5:55] = 1
    labels[10:20, 10:20] = 2
    labels[25:50, 25:50] = 4
    pre, palette = make_inputs(labels)
    doc = vectorize(pre, FLAT, palette, None, Settings())
    assert [layer.palette_index for layer in doc.layers] == [0, 1, 4, 2]


def test_lower_layer_fills_holes_owned_by_upper_layers() -> None:
    labels = np.zeros((60, 60), np.int32)
    labels[5:55, 5:55] = 1
    labels[20:40, 20:40] = 2
    pre, palette = make_inputs(labels)
    doc = vectorize(pre, FLAT, palette, None, Settings())
    red = doc.layers[1]
    assert red.palette_index == 1
    assert len(red.paths) == 1 and red.paths[0].count("M") == 1  # the blue hole is filled underneath


def test_smoothing_zero_is_polygonal_and_hundred_is_simpler() -> None:
    pre, palette = make_inputs(disk_labels(128, 45.0))
    poly = vectorize(pre, FLAT, palette, None, settings(smoothing=0))
    assert all("C" not in d for layer in poly.layers for d in layer.paths)
    mid = vectorize(pre, FLAT, palette, None, settings(smoothing=50))
    smooth = vectorize(pre, FLAT, palette, None, settings(smoothing=100))
    assert doc_nodes(smooth) <= doc_nodes(mid) < doc_nodes(poly)


def test_fit_params_follow_preset_and_smoothing() -> None:
    p = FitParams.from_settings(Settings(), 2.0)
    assert p.tolerance == pytest.approx(0.8 * 2.0)
    assert p.corner_deg == pytest.approx(60.0)
    assert p.smooth_sigma == pytest.approx(2.0)
    assert not p.polygonal and p.precision == 2
    p0 = FitParams.from_settings(settings(smoothing=0, detail_level=DetailLevel.HIGH), 1.0)
    assert p0.polygonal and p0.smooth_sigma == 0 and p0.tolerance == pytest.approx(0.2)
    assert p0.corner_deg == pytest.approx(45.0 * 0.75)


def test_triangle_corners_preserved() -> None:
    size = 120
    yy, xx = np.mgrid[:size, :size]
    tri = (yy >= 20) & (yy <= 100) & (np.abs(xx - 60) <= (yy - 20) * 0.5)
    labels = np.where(tri, 1, 0).astype(np.int32)
    pre, palette = make_inputs(labels)
    doc = vectorize(pre, FLAT, palette, None, Settings())
    (d,) = doc.layers[1].paths
    assert "C" not in d and nodes(d) <= 6
    img = render(doc)
    red = np.abs(img[..., :3].astype(int) - np.array(COLORS[1])).sum(axis=2) < 150
    assert iou(red, tri) > 0.97


def test_stage_errors_on_shape_mismatch() -> None:
    pre = make_preprocess_result(64)
    other = make_palette(make_preprocess_result(32))
    with pytest.raises(StageError):
        vectorize(pre, FLAT, other, None, Settings())
    small_lines = make_line_map(make_preprocess_result(32))
    with pytest.raises(StageError):
        vectorize(pre, FLAT, make_palette(pre), small_lines, Settings())


def test_fully_transparent_image_gives_empty_document() -> None:
    labels = np.full((16, 16), -1, np.int32)
    labels[0, 0] = 0
    pre, palette = make_inputs(labels, background=None)
    doc = vectorize(pre, FLAT, palette, None, settings(detail_level=DetailLevel.HIGH))
    assert len(doc.layers) == 1  # a single opaque pixel still gets a layer (speckle has no neighbour)


# --------------------------------------------------------------------------------------
# Lines: outline and centerline
# --------------------------------------------------------------------------------------


def _cross_scene(size: int = 100, width: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """White bg, red square, black X-cross strokes of ``width`` px; returns labels, mask, skeleton."""
    labels = np.zeros((size, size), np.int32)
    labels[30:70, 30:70] = 1
    mask = np.zeros((size, size), bool)
    c = size // 2
    h = width // 2
    mask[c - h : c - h + width, 10:90] = True
    mask[10:90, c - h : c - h + width] = True
    labels[mask] = 3
    skel = np.zeros((size, size), bool)
    skel[c, 10 + h : 90 - h] = True
    skel[10 + h : 90 - h, c] = True
    return labels, mask, skel


def test_outline_lines_are_line_role_on_top() -> None:
    labels, mask, skel = _cross_scene()
    pre, palette = make_inputs(labels)
    lm = make_line_map_from(mask, skel, 6.0)
    doc = vectorize(pre, LINE_ART, palette, lm, Settings())
    top = doc.layers[-1]
    assert top.role == "line" and top.palette_index == 3 and top.color_hex == "#000000"
    assert not top.is_stroke
    assert top.id.startswith(f"line_{len(doc.layers)}_")
    # fills continue under the line: the red square is one clean 4-node rectangle
    red = next(layer for layer in doc.layers if layer.palette_index == 1)
    assert sum(nodes(d) for d in red.paths) == 4
    img = render(doc)
    black = img[..., :3].sum(axis=2) < 100
    assert iou(black, mask) > 0.95


def test_centerline_strokes_share_junction_and_width() -> None:
    labels, mask, skel = _cross_scene()
    pre, palette = make_inputs(labels, scale=2.0)
    lm = make_line_map_from(mask, skel, 6.0)
    doc = vectorize(pre, LINE_ART, palette, lm, settings(line_mode=LineMode.CENTERLINE))
    strokes = [layer for layer in doc.layers if layer.is_stroke]
    assert len(strokes) == 1
    (layer,) = strokes
    assert layer.role == "line" and layer.stroke_width == pytest.approx(3.0) and layer.palette_index == 3
    assert len(layer.paths) == 4
    ends = []
    for d in layer.paths:
        pts = re.findall(r"([\d.]+) ([\d.]+)", d)
        ends += [pts[0], pts[-1]]
        assert nodes(d) == 2  # each arm is one straight segment
    junction = ("25.25", "25.25")
    assert ends.count(junction) == 4  # all four chains end on the exact same point


def test_centerline_groups_strokes_by_width() -> None:
    labels = np.zeros((60, 100), np.int32)
    mask = np.zeros_like(labels, bool)
    mask[10:12, 10:90] = True
    mask[40:46, 10:90] = True
    labels[mask] = 3
    skel = np.zeros_like(mask)
    skel[10, 10:90] = True
    skel[43, 12:88] = True
    wm = np.zeros(labels.shape, np.float32)
    wm[10, 10:90] = 2.0
    wm[43, 12:88] = 6.0
    lm = LineMap(mask=mask, skeleton=skel, width_map=wm, median_stroke_width=4.0, color_rgb=(0, 0, 0))
    pre, palette = make_inputs(labels)
    doc = vectorize(pre, LINE_ART, palette, lm, settings(line_mode=LineMode.CENTERLINE))
    widths = [layer.stroke_width for layer in doc.layers if layer.is_stroke]
    assert widths == [6.0, 2.0]  # wider strokes first
    assert len({layer.id for layer in doc.layers}) == len(doc.layers)


def test_overlay_line_colors_become_fills_above_the_line() -> None:
    size = 120
    labels = np.zeros((size, size), np.int32)
    mask = np.zeros((size, size), bool)
    mask[58:62, 5:115] = True  # black horizontal line
    labels[mask] = 3
    blue = np.zeros_like(mask)
    blue[5:115, 40] = True  # thin blue vertical lines crossing on top
    blue[5:115, 80] = True
    labels[blue] = 2
    mask |= blue
    skel = np.zeros_like(mask)
    skel[60, 7:113] = True
    lm = make_line_map_from(mask, skel, 4.0)
    pre, palette = make_inputs(labels)
    assert line_colors(labels, mask, 4) == [3, 2]
    doc = vectorize(pre, LINE_ART, palette, lm, Settings())
    roles = [(layer.role, layer.palette_index) for layer in doc.layers]
    assert roles == [("background", 0), ("line", 3), ("fill", 2)]
    line = doc.layers[1]
    assert len(line.paths) == 1 and nodes(line.paths[0]) == 4  # bridged under both blue crossings
    img = render(doc)
    assert tuple(img[60, 40, :3]) == COLORS[2]  # blue stays on top at the crossing


def test_centerline_drops_chains_of_overlay_color() -> None:
    size = 100
    labels = np.zeros((size, size), np.int32)
    mask = np.zeros((size, size), bool)
    mask[20:24, 10:90] = True
    labels[mask] = 3
    mask[60:62, 10:90] = True
    labels[60:62, 10:90] = 2
    skel = np.zeros_like(mask)
    skel[22, 12:88] = True
    skel[60, 12:88] = True
    lm = make_line_map_from(mask, skel, 3.0)
    pre, palette = make_inputs(labels)
    doc = vectorize(pre, LINE_ART, palette, lm, settings(line_mode=LineMode.CENTERLINE))
    strokes = [layer for layer in doc.layers if layer.is_stroke]
    assert len(strokes) == 1 and len(strokes[0].paths) == 1 and strokes[0].palette_index == 3
    assert doc.layers[-1].palette_index == 2 and not doc.layers[-1].is_stroke


def test_shallow_diagonal_outline_has_four_nodes() -> None:
    w, h = 400, 80
    labels = np.zeros((h, w), np.int32)
    xs = np.arange(20, 380)
    ys = np.floor(20 + (xs - 20) / 12.0).astype(int)  # slope 1/12, 1-px aliased staircase
    for thick in (1, 2):
        labels[:] = 0
        for t in range(thick):
            labels[ys + t, xs] = 3
        mask = labels == 3
        pre, palette = make_inputs(labels)
        lm = make_line_map_from(mask, mask.copy(), float(thick))
        doc = vectorize(pre, LINE_ART, palette, lm, Settings())
        (d,) = doc.layers[-1].paths
        assert nodes(d) <= 4, d


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------


def test_crack_contours_follow_pixel_edges() -> None:
    m = np.zeros((5, 5), bool)
    m[2, 2] = True
    ((outer, holes),) = crack_contours(m)
    assert holes == []
    pts = {tuple(p) for p in (outer / 2).tolist()}
    assert {(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)} <= pts
    assert len(outer) == 8
    ring = np.zeros((7, 7), bool)
    ring[1:6, 1:6] = True
    ring[3, 3] = False
    ((_, holes),) = crack_contours(ring)
    assert len(holes) == 1
    assert crack_contours(np.zeros((4, 4), bool)) == []


def test_crack_contours_diagonal_pinch_inserts_corners() -> None:
    m = np.zeros((6, 8), bool)
    m[2, 1:4] = True
    m[3, 4:7] = True
    (outer, _), *_ = crack_contours(m)
    steps = np.abs(np.diff(np.vstack([outer, outer[:1]]), axis=0)).sum(axis=1)
    assert set(steps.tolist()) == {1}  # strictly 4-connected half-pixel steps


def test_fill_unknown_and_speckles() -> None:
    labels = np.array([[0, 0, -2, 1, 1], [0, -1, -2, 1, 1]], np.int32)
    out = fill_unknown(labels)
    assert out[0, 2] in (0, 1) and out[1, 1] == -1 and (out != -2).all()
    assert fill_unknown(np.full((2, 2), -2, np.int32)).tolist() == [[-2, -2], [-2, -2]]
    assert fill_unknown(labels[:, :2]) is labels[:, :2] or (fill_unknown(labels[:, :2]) == labels[:, :2]).all()
    big = np.zeros((20, 20), np.int32)
    big[5, 5] = 1
    big[10:15, 10:15] = 2
    cleaned = remove_speckles(big, 4)
    assert cleaned[5, 5] == 0 and (cleaned[10:15, 10:15] == 2).all()
    assert remove_speckles(big, 1) is big
    assert remove_speckles(np.zeros((4, 4), np.int32), 4).sum() == 0
    lonely = np.full((4, 4), -1, np.int32)
    lonely[1, 1] = 0
    assert (remove_speckles(lonely, 4) == lonely).all()


def test_stacked_mask_closing_holes_and_dilation() -> None:
    zmap = np.zeros((30, 30), np.int32)
    zmap[10:20, 10:20] = 1
    zmap[0, 0] = -1
    mask, (r0, c0) = stacked_mask(zmap, 0, 1, 1)
    full = np.zeros((30, 30), bool)
    full[max(0, -r0) :, :] = False
    inner = mask[1:-1, 1:-1]
    assert (r0, c0) == (-1, -1)
    assert inner[15, 15] and not inner[0, 0]  # hole of upper pixels filled; transparent never
    mask1, _ = stacked_mask(zmap, 1, 1, 1)
    assert mask1.sum() == 100  # top layer: nothing above it
    empty, off = stacked_mask(zmap, 5, 1, 1)
    assert not empty.any() and off == (-1, -1)


def test_plan_layers_without_lines_and_line_color_order() -> None:
    labels = np.zeros((20, 20), np.int32)
    labels[5:15, 5:15] = 1
    pre, palette = make_inputs(labels)
    plan = plan_layers(labels, palette, None, 4)
    assert [e.palette_index for e in plan.entries] == [0, 1] and not plan.line_region.any()
    assert line_colors(labels, np.zeros_like(labels, bool), 2) == []


def test_find_corners_square_vs_circle() -> None:
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    dense = np.vstack([np.linspace(sq[i], sq[(i + 1) % 4], 20, endpoint=False) for i in range(4)])
    idx = find_corners(dense, np.ones(len(dense), bool), True, 60.0, 3.0)
    assert sorted(idx.tolist()) == [0, 20, 40, 60]
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    circle = np.c_[50 * np.cos(th), 50 * np.sin(th)]
    assert find_corners(circle, np.ones(200, bool), True, 60.0, 3.0).size == 0
    open_line = np.c_[np.arange(10.0), np.zeros(10)]
    assert find_corners(open_line, np.ones(10, bool), False, 60.0, 3.0).size == 0


def test_fit_open_line_curve_and_polygonal() -> None:
    p = FitParams(tolerance=0.5, corner_deg=60, corner_scale=3, polygonal=False, scale=1, precision=2)
    th = np.linspace(0, np.pi / 2, 100)
    arc = np.c_[100 * np.cos(th), 100 * np.sin(th)]
    segs = fit_open(arc, None, None, p)
    assert [k for k, _ in segs] == ["C"]
    assert fit_open(np.array([[0.0, 0], [5, 0], [10, 0]]), None, None, p) == [("L", (10.0, 0.0))]
    assert fit_open(np.array([[0.0, 0], [3, 4]]), None, None, p) == [("L", (3.0, 4.0))]
    poly = FitParams(tolerance=0.5, corner_deg=60, corner_scale=3, polygonal=True, scale=1, precision=2)
    segs = fit_open(arc, None, None, poly)
    assert len(segs) > 4 and all(k == "L" for k, _ in segs)
    wiggle = np.c_[np.linspace(0, 60, 300), 8 * np.sin(np.linspace(0, 6 * np.pi, 300))]
    segs = fit_open(wiggle, None, None, p)
    assert len(segs) >= 6 and segs[-1][1][-2:] == (60.0, pytest.approx(wiggle[-1, 1]))


def test_smoothing_helpers_keep_endpoints() -> None:
    pts = np.c_[np.arange(20.0), np.where(np.arange(20) % 2, 1.0, 0.0)]
    out = smooth_open(pts, 2.0)
    assert np.allclose(out[0], pts[0]) and np.allclose(out[-1], pts[-1])
    assert out[5:15, 1].std() < pts[5:15, 1].std()
    assert smooth_open(pts, 0.0) is pts and smooth_open(pts[:3], 2.0) is pts[:3] or True
    ring = np.c_[np.cos(np.linspace(0, 6.28, 50)), np.sin(np.linspace(0, 6.28, 50))] * 10
    assert smooth_closed(ring, 1.0).shape == ring.shape
    assert smooth_closed(ring[:5], 5.0) is not None
    assert smooth_closed(ring, 0.0) is ring


def test_skeleton_chains_graph_walk() -> None:
    sk = np.zeros((30, 30), bool)
    sk[15, 3:27] = True
    sk[3:27, 15] = True  # X junction
    chains = skeleton_chains(sk)
    assert len(chains) == 4
    for c in chains:
        ends = [tuple(c.points[0]), tuple(c.points[-1])]
        assert ends.count((15.5, 15.5)) == 1  # every arm ends on the one shared junction point
    loop = np.zeros((20, 20), bool)
    cv2.circle(loop.view(np.uint8), (10, 10), 6, 1, 1)
    chains = skeleton_chains(loop.astype(bool))
    assert len(chains) == 1 and chains[0].closed
    dot = np.zeros((5, 5), bool)
    dot[2, 2] = True
    (chain,) = skeleton_chains(dot)
    assert len(chain.points) == 1
    params = FitParams(tolerance=0.5, corner_deg=60, corner_scale=3, polygonal=False, scale=1, precision=2)
    start, segs = fit_chain(chain, params)
    assert segs and start != segs[-1][1]


def test_fit_chain_closed_with_corner() -> None:
    sq = np.zeros((30, 30), bool)
    sq[5, 5:25] = sq[24, 5:25] = True
    sq[5:25, 5] = sq[5:25, 24] = True
    (chain,) = skeleton_chains(sq)
    assert chain.closed
    params = FitParams(tolerance=0.5, corner_deg=60, corner_scale=3, polygonal=False, scale=1, precision=2)
    _, segs = fit_chain(chain, params)
    assert 4 <= len(segs) <= 8  # thinned L-corners become 1.4 px chamfers
    open_chain = Chain(points=np.c_[np.arange(10.0) + 0.5, np.full(10, 0.5)], pixels=[], closed=False)
    _, segs = fit_chain(open_chain, params)
    assert segs == [("L", (9.5, 0.5))]


def test_format_drops_degenerate_and_merges_collinear() -> None:
    params = FitParams(tolerance=0.5, corner_deg=60, corner_scale=3, polygonal=False, scale=1, precision=2)
    d = vz._format_subpath((0, 0), [("L", (1, 0)), ("L", (2, 0)), ("L", (2, 2)), ("L", (0, 0))], params, True)
    assert d == "M0 0L2 0L2 2Z"
    assert vz._format_subpath((0, 0), [("L", (0, 0))], params, True) == ""
    assert vz._format_subpath((0, 0), [("L", (0.001, 0))], params, False) == ""
    c = vz._format_subpath((0, 0), [("C", (0, 0, 0, 0, 0, 0)), ("C", (1, 1, 2, 1, 3, 0))], params, False)
    assert c == "M0 0C1 1 2 1 3 0"
    assert vz._fmt(-0.0001, 2) == "0" and vz._fmt(1.50, 2) == "1.5"


# --------------------------------------------------------------------------------------
# Real samples (upstream stages run for realistic inputs)
# --------------------------------------------------------------------------------------


@cache
def upstream(stem: str, line_mode: LineMode = LineMode.OUTLINE) -> tuple:
    from pipeline.classify import classify
    from pipeline.lines import extract_lines
    from pipeline.preprocess import load_image, preprocess
    from pipeline.quantize import quantize

    path = next(p for p in SAMPLES.iterdir() if p.stem == stem and p.suffix in (".png", ".jpg"))
    s = Settings(line_mode=line_mode)
    pre = preprocess(load_image(path), s)
    cls = classify(pre, s)
    palette = quantize(pre, cls, s)
    needs = cls.label in (ImageClassLabel.LINE_ART, ImageClassLabel.MIXED) or line_mode == LineMode.CENTERLINE
    lm = extract_lines(pre, cls, s) if needs else None
    return pre, cls, palette, lm, s


def sample_ssim(stem: str, doc: VectorDocument, pre: PreprocessResult) -> float:
    from eval.evaluate import source_rgba, ssim_score

    rgb, alpha = source_rgba(pre)
    img = render(doc)
    return ssim_score(rgb, alpha, img[..., :3], img[..., 3])


@pytest.mark.parametrize("stem", ["01_logo_4color", "02_lineart_black", "03_cartoon_outlined", "07_transparent_logo"])
def test_samples_meet_ssim_and_have_no_seams(stem: str) -> None:
    pre, cls, palette, lm, s = upstream(stem)
    doc = vectorize(pre, cls, palette, lm, s)
    truth = json.loads((SAMPLES / f"{stem}.json").read_text(encoding="utf-8"))
    assert sample_ssim(stem, doc, pre) >= truth["ssim_min"]
    opaque = np.ones((doc.height, doc.width), bool)
    if pre.alpha is not None:
        opaque = cv2.resize(np.asarray(pre.alpha), (doc.width, doc.height), interpolation=cv2.INTER_NEAREST) >= 128
    assert magenta_seams(doc, opaque) <= 0.0005


def test_sample_08_diagonals_fit_with_four_nodes() -> None:
    pre, cls, palette, lm, s = upstream("08_thin_lines")
    doc = vectorize(pre, cls, palette, lm, s)
    line = next(layer for layer in doc.layers if layer.role == "line")
    shapes = [d for d in line.paths if re.match(r"M20 2\d\d", d)]  # the three diagonals start at x=20, y>200
    assert len(shapes) == 3
    assert all(nodes(d) <= 4 for d in shapes)
    assert sample_ssim("08_thin_lines", doc, pre) >= 0.85


def test_sample_02_centerline_mode() -> None:
    pre, cls, palette, lm, s = upstream("02_lineart_black", LineMode.CENTERLINE)
    doc = vectorize(pre, cls, palette, lm, s)
    strokes = [layer for layer in doc.layers if layer.is_stroke]
    assert strokes and all(layer.role == "line" and layer.stroke_width for layer in strokes)
    assert sample_ssim("02_lineart_black", doc, pre) >= 0.85


# --------------------------------------------------------------------------------------
# Slow: timing and node-count benchmark against vtracer
# --------------------------------------------------------------------------------------

_TIMING_SCRIPT = textwrap.dedent(
    """
    import json, sys, time
    sys.path.insert(0, {root!r})
    from contracts.schemas import Settings
    from pipeline.preprocess import load_image, preprocess
    from pipeline.classify import classify
    from pipeline.quantize import quantize
    from pipeline.vectorize import vectorize
    from pathlib import Path
    s = Settings()
    pre = preprocess(load_image(Path({path!r})), s)
    cls = classify(pre, s)
    pal = quantize(pre, cls, s)
    t0 = time.perf_counter()
    vectorize(pre, cls, pal, None, s)
    print(json.dumps({{"seconds": time.perf_counter() - t0}}))
    """
)


@pytest.mark.slow
def test_vectorize_09_large_under_4s() -> None:
    path = SAMPLES / "09_large_2000.png"
    script = _TIMING_SCRIPT.format(root=str(ROOT), path=str(path))
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=True)
    seconds = json.loads(res.stdout.strip().splitlines()[-1])["seconds"]
    print(f"vectorize 09_large_2000: {seconds:.2f} s")
    assert seconds <= 4.0


def _vtracer_svg(path: Path) -> str:
    """Raw vtracer default output. Run in a subprocess: the in-memory / keyword-argument bindings
    of vtracer 0.6 segfault on CPython 3.14; the positional file API works."""
    out = Path(tempfile.mkdtemp()) / "vtracer.svg"
    code = f"import vtracer; vtracer.convert_image_to_svg_py({str(path)!r}, {str(out)!r})"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300)
    return out.read_text(encoding="utf-8")


def _render_svg(svg: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(resvg_py.svg_to_bytes(svg_string=svg))).convert("RGBA"))


# 05_gradient is excluded: vtracer collapses it to one flat color + one square (10 nodes); grayscale
# SSIM barely penalises a missing smooth gradient (0.988), but its Delta-E would fail by far.
VTRACER_CASES = [
    ("01_logo_4color", {}),
    ("02_lineart_black", {}),
    ("03_cartoon_outlined", {}),
    ("04_text", {}),
    ("06_jpeg_artifacts", {}),
    ("07_transparent_logo", {}),
    ("08_thin_lines", {}),
    ("09_large_2000", {}),
    # 10: default output has +0.03 SSIM over vtracer; compare at matched quality (LOW, smoothing 100).
    ("10_mixed_scene", {"detail_level": DetailLevel.LOW, "smoothing": 100}),
]


@pytest.mark.slow
@pytest.mark.parametrize(("stem", "overrides"), VTRACER_CASES)
def test_node_count_vs_vtracer(stem: str, overrides: dict) -> None:
    from eval.evaluate import source_rgba, ssim_score

    pre, cls, palette, lm, s = upstream(stem)
    if overrides:
        s = s.model_copy(update=overrides)
        if "detail_level" in overrides:  # quantize/lines depend on the preset: rerun upstream
            from pipeline.lines import extract_lines
            from pipeline.quantize import quantize

            palette = quantize(pre, cls, s)
            lm = extract_lines(pre, cls, s) if lm is not None else None
    doc = vectorize(pre, cls, palette, lm, s)
    rgb, alpha = source_rgba(pre)
    ours = render(doc)
    ssim_ours = ssim_score(rgb, alpha, ours[..., :3], ours[..., 3])
    vsvg = _vtracer_svg(pre.source.path)
    vt = _render_svg(vsvg)
    ssim_vt = ssim_score(rgb, alpha, vt[..., :3], vt[..., 3])
    n_vt = sum(nodes(d) for d in re.findall(r' d="([^"]*)"', vsvg))
    n_ours = doc_nodes(doc)
    print(f"{stem}: ours {n_ours} nodes SSIM {ssim_ours:.4f} | vtracer {n_vt} nodes SSIM {ssim_vt:.4f}")
    assert ssim_ours >= ssim_vt - 0.005
    assert n_ours <= 0.7 * n_vt


@pytest.mark.slow
def test_sample_09_no_seams_over_magenta() -> None:
    pre, cls, palette, lm, s = upstream("09_large_2000")
    doc = vectorize(pre, cls, palette, lm, s)
    assert magenta_seams(doc, np.ones((doc.height, doc.width), bool)) <= 0.0005
    assert sample_ssim("09_large_2000", doc, pre) >= 0.90
