"""Quality metrics for a finished conversion (pipeline stage 8, ``evaluate``).

All metrics follow the definitions in ``contracts.schemas.QualityReport``. ``preview.png``
from the :class:`ExportBundle` *is* the rendering of the SVG, so no SVG renderer is needed
here: SSIM, gap ratio and alpha IoU are measured on it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from lxml import etree
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

from contracts.schemas import (
    ExportBundle,
    LineMap,
    MetricCheck,
    MetricName,
    Palette,
    PreprocessResult,
    QualityReport,
    QualityThresholds,
    VectorDocument,
    VectorLayer,
)
from eval.raster import document_node_count, hex_to_rgb

SVG_NS = "http://www.w3.org/2000/svg"

P95_SAMPLE_CAP = 200_000
"""Max interior pixels (evenly strided) used for the p95 pixel Delta-E diagnostic."""

UNCOVERED_MIN_FRACTION = 0.005
"""Palette regions with at least this share of opaque pixels must be represented by a layer.
If none is, the region is scored with the color the preview actually renders there."""

MEDIAN_SAMPLE_CAP = 60_000
"""Regions larger than this are sub-sampled (evenly strided) before taking the LAB median."""


# --------------------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------------------


def source_rgba(pre: PreprocessResult) -> tuple[np.ndarray, np.ndarray | None]:
    """Source RGB (H, W, 3) uint8 and alpha (H, W) uint8 or None, at SOURCE resolution.

    RGB comes from the original file when it is readable (SSIM is defined against the
    input); otherwise ``pre.image`` is resized to source size. Alpha always comes from
    ``pre.alpha`` so background removal done by preprocess is honoured.
    """
    w, h = pre.source.width, pre.source.height
    rgb: np.ndarray | None = None
    path = pre.source.path
    if path.is_file():
        try:
            with Image.open(path) as img:
                if img.size == (w, h):
                    rgb = np.asarray(img.convert("RGBA"))[..., :3].copy()
        except OSError:
            rgb = None
    if rgb is None:
        rgb = _resize(np.asarray(pre.image), w, h)
    alpha = None if pre.alpha is None else _resize(np.asarray(pre.alpha), w, h)
    return rgb, alpha


def _resize(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    if arr.shape[1] == w and arr.shape[0] == h:
        return arr.copy()
    interp = cv2.INTER_AREA if arr.shape[1] > w else cv2.INTER_LINEAR
    return cv2.resize(arr, (w, h), interpolation=interp)


def load_preview(path: Path, width: int, height: int) -> np.ndarray:
    """preview.png as (H, W, 4) uint8 RGBA at source resolution."""
    with Image.open(path) as img:
        rgba = np.asarray(img.convert("RGBA")).copy()
    return _resize(rgba, width, height)


def composite_over_white(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """(H, W, 3) float32 in [0, 1]: straight-alpha RGB composited over white."""
    out = rgb.astype(np.float32) / np.float32(255.0)
    if alpha is None:
        return out
    a = alpha.astype(np.float32)[..., None] / np.float32(255.0)
    return out * a + (np.float32(1.0) - a)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


GRAY_WEIGHTS = np.array([0.2125, 0.7154, 0.0721], dtype=np.float32)
"""skimage.color.rgb2gray weights (Rec. 709 luma). They sum to 1, so graying commutes with
compositing over white: gray(rgb * a + (1 - a)) == gray(rgb) * a + (1 - a)."""


def gray_over_white(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """(H, W) float32 in [0, 1]: skimage rgb2gray of the image composited over white."""
    gray = (rgb.astype(np.float32) @ GRAY_WEIGHTS) / np.float32(255.0)
    if alpha is None:
        return gray
    a = alpha.astype(np.float32) / np.float32(255.0)
    return gray * a + (np.float32(1.0) - a)


def ssim_score(
    src_rgb: np.ndarray, src_alpha: np.ndarray | None, prev_rgb: np.ndarray, prev_alpha: np.ndarray | None
) -> float:
    """Grayscale SSIM of source vs preview, both composited over white, same resolution."""
    a = gray_over_white(src_rgb, src_alpha).astype(np.float64)
    b = gray_over_white(prev_rgb, prev_alpha).astype(np.float64)
    win = min(7, min(a.shape))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return float(1.0 - np.abs(a - b).mean())
    value = float(structural_similarity(a, b, data_range=1.0, win_size=win))
    return float(np.clip(value, -1.0, 1.0))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 sRGB -> (N, 3) float64 CIELAB (D65, 2 deg)."""
    arr = np.asarray(rgb, dtype=np.float64).reshape(-1, 1, 3) / 255.0
    return rgb2lab(arr).reshape(-1, 3)


def median_lab(pixels_rgb: np.ndarray) -> np.ndarray:
    """Per-channel median in LAB of (N, 3) uint8 pixels (sub-sampled above MEDIAN_SAMPLE_CAP)."""
    if pixels_rgb.shape[0] > MEDIAN_SAMPLE_CAP:
        step = int(np.ceil(pixels_rgb.shape[0] / MEDIAN_SAMPLE_CAP))
        pixels_rgb = pixels_rgb[::step]
    return np.median(rgb_to_lab(pixels_rgb), axis=0)


def delta_e(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 between broadcastable LAB arrays (last axis = 3)."""
    return np.asarray(deltaE_ciede2000(np.asarray(lab1, np.float64), np.asarray(lab2, np.float64)))


@dataclass(frozen=True)
class RegionDeltaE:
    """Delta-E diagnostics for one layer."""

    layer_id: str
    role: str
    palette_index: int | None
    pixels: int
    layer_lab: tuple[float, float, float]
    source_lab: tuple[float, float, float]
    delta_e: float


def processing_space_source(pre: PreprocessResult, source_rgb: np.ndarray | None = None) -> np.ndarray:
    """Original RGB on the processing grid (nearest neighbour: no new blended colors)."""
    rgb = source_rgba(pre)[0] if source_rgb is None else source_rgb
    if rgb.shape[:2] == (pre.height, pre.width):
        return rgb
    return cv2.resize(rgb, (pre.width, pre.height), interpolation=cv2.INTER_NEAREST)


def _layer_palette_index(layer: VectorLayer, palette: Palette) -> int | None:
    """Palette index for a layer: explicit, else exact hex match, else nearest in LAB."""
    if layer.palette_index is not None and layer.palette_index < len(palette.colors):
        return layer.palette_index
    for color in palette.colors:
        if color.hex == layer.color_hex:
            return color.index
    pal_lab = rgb_to_lab(np.array([c.rgb for c in palette.colors], dtype=np.uint8))
    lab = rgb_to_lab(np.array([hex_to_rgb(layer.color_hex)], dtype=np.uint8))
    return int(np.argmin(delta_e(np.broadcast_to(lab, pal_lab.shape), pal_lab)))


def region_delta_es(
    pre: PreprocessResult,
    palette: Palette,
    line_map: LineMap | None,
    doc: VectorDocument,
    preview_rgb: np.ndarray | None = None,
    source_rgb: np.ndarray | None = None,
) -> list[RegionDeltaE]:
    """CIEDE2000 of each layer's color vs the median LAB of the source pixels it represents.

    "Source pixels" are the ORIGINAL input (``source_rgb`` at source resolution, default
    :func:`source_rgba`) nearest-resampled onto the processing grid, the same reference SSIM
    uses. Measuring against ``pre.image`` would hide color damage done by preprocessing
    (e.g. interpolated upscaling turning 1-px black lines gray).

    Fill/background layers use their palette region (``label_map == index``); stroke/line
    layers use LineMap-covered pixels when a LineMap exists. Only opaque pixels count.
    Layers whose region is empty are skipped.

    Uncovered regions: if ``preview_rgb`` (source resolution) is given, every palette region
    holding >= UNCOVERED_MIN_FRACTION of the opaque pixels that no layer represents (e.g. a
    dropped layer) is scored as the median LAB the preview renders over that region vs its
    source median, with ``layer_id = "uncovered:<hex>"``. Without this a missing layer
    would be invisible to Delta-E.
    """
    image = processing_space_source(pre, source_rgb)
    opaque = pre.opaque_mask
    labels = np.asarray(palette.label_map)
    results: list[RegionDeltaE] = []
    represented: set[int] = set()
    for layer in doc.layers:
        is_line = layer.role == "line" or layer.is_stroke
        index: int | None = None
        if is_line and line_map is not None and line_map.mask.shape == opaque.shape:
            region = np.asarray(line_map.mask) & opaque
            if layer.palette_index is not None:
                represented.add(layer.palette_index)
        else:
            index = _layer_palette_index(layer, palette)
            if index is None or labels.shape != opaque.shape:
                continue
            represented.add(index)
            region = (labels == index) & opaque
        count = int(region.sum())
        if count == 0:
            continue
        layer_lab = rgb_to_lab(np.array([hex_to_rgb(layer.color_hex)], dtype=np.uint8))[0]
        results.append(_region_result(layer.id, layer.role, index, count, layer_lab, median_lab(image[region])))
    if preview_rgb is None or labels.shape != opaque.shape:
        return results
    total = max(int(opaque.sum()), 1)
    preview_proc = _resize(preview_rgb, labels.shape[1], labels.shape[0])
    for color in palette.colors:
        if color.index in represented:
            continue
        region = (labels == color.index) & opaque
        count = int(region.sum())
        if count < UNCOVERED_MIN_FRACTION * total:
            continue
        rendered, source = median_lab(preview_proc[region]), median_lab(image[region])
        results.append(_region_result(f"uncovered:{color.hex}", "uncovered", color.index, count, rendered, source))
    return results


def _region_result(
    layer_id: str, role: str, index: int | None, count: int, layer_lab: np.ndarray, src_lab: np.ndarray
) -> RegionDeltaE:
    return RegionDeltaE(
        layer_id=layer_id,
        role=role,
        palette_index=index,
        pixels=count,
        layer_lab=(float(layer_lab[0]), float(layer_lab[1]), float(layer_lab[2])),
        source_lab=(float(src_lab[0]), float(src_lab[1]), float(src_lab[2])),
        delta_e=float(delta_e(layer_lab, src_lab)),
    )


def interior_mask(src_alpha: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """Source opaque area (alpha >= 128) eroded by 1 px; the image border is not eroded."""
    if src_alpha is None:
        return np.ones(shape, dtype=bool)
    opaque = (src_alpha >= 128).astype(np.uint8)
    eroded = cv2.erode(opaque, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
    return eroded.astype(bool)


def gap_mask(src_alpha: np.ndarray | None, prev_alpha: np.ndarray) -> np.ndarray:
    """Pixels inside the eroded source opaque area that the preview leaves transparent."""
    return interior_mask(src_alpha, prev_alpha.shape) & (prev_alpha < 128)


def gap_ratio(src_alpha: np.ndarray | None, prev_alpha: np.ndarray) -> float:
    """Fraction of interior opaque source pixels whose preview alpha < 128."""
    interior = interior_mask(src_alpha, prev_alpha.shape)
    total = int(interior.sum())
    if total == 0:
        return 0.0
    return float((interior & (prev_alpha < 128)).sum()) / total


def alpha_iou(src_alpha: np.ndarray, prev_alpha: np.ndarray) -> float:
    """IoU of (source alpha >= 128) and (preview alpha >= 128)."""
    a, b = src_alpha >= 128, prev_alpha >= 128
    union = int((a | b).sum())
    return 1.0 if union == 0 else float((a & b).sum()) / union


def pixel_delta_e_p95(
    src_rgb: np.ndarray, src_alpha: np.ndarray | None, prev_rgb: np.ndarray, prev_alpha: np.ndarray | None
) -> float:
    """DIAGNOSTIC ONLY (not a QualityReport metric): 95th-percentile per-pixel CIEDE2000.

    Source and preview are composited over white and compared pixel by pixel over the
    1-px-eroded source opaque interior (at most P95_SAMPLE_CAP evenly strided pixels).
    Unlike the region-median Delta-E it sees minority colors merged into a larger region.
    """
    interior = interior_mask(src_alpha, prev_rgb.shape[:2]).ravel()
    idx = np.flatnonzero(interior)
    if idx.size == 0:
        return 0.0
    if idx.size > P95_SAMPLE_CAP:
        idx = idx[:: int(np.ceil(idx.size / P95_SAMPLE_CAP))]

    def lab_at(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
        px = rgb.reshape(-1, 3)[idx].astype(np.float32) / np.float32(255.0)
        if alpha is not None:
            a = alpha.ravel()[idx].astype(np.float32)[:, None] / np.float32(255.0)
            px = px * a + (np.float32(1.0) - a)
        return rgb2lab(px.astype(np.float64).reshape(-1, 1, 3)).reshape(-1, 3)

    return float(np.percentile(delta_e(lab_at(src_rgb, src_alpha), lab_at(prev_rgb, prev_alpha)), 95))


def pixel_delta_e_p95_for(pre: PreprocessResult, bundle: ExportBundle) -> float:
    """:func:`pixel_delta_e_p95` loading the original source and preview.png at source resolution."""
    src_rgb, src_alpha = source_rgba(pre)
    preview = load_preview(bundle.preview_png_path, pre.source.width, pre.source.height)
    return pixel_delta_e_p95(src_rgb, src_alpha, preview[..., :3], preview[..., 3])


def svg_problems(svg_path: Path, doc: VectorDocument) -> list[str]:
    """Structural problems with output.svg (empty list = valid).

    Checks: well-formed XML, root ``<svg>`` in the SVG namespace, ``viewBox == 0 0 W H``,
    every VectorLayer id present, and at least one drawable element if there are layers.
    """
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True)
        root = etree.parse(str(svg_path), parser).getroot()
    except (etree.XMLSyntaxError, OSError) as exc:
        return [f"not well-formed XML: {exc}"]
    problems: list[str] = []
    if root.tag != f"{{{SVG_NS}}}svg":
        problems.append(f"root element is {root.tag!r}, expected svg in the SVG namespace")
    view_box = root.get("viewBox")
    try:
        vb = [float(v) for v in (view_box or "").replace(",", " ").split()]
    except ValueError:
        vb = []
    if len(vb) != 4 or not np.allclose(vb, doc.view_box, atol=1e-6):
        problems.append(f"viewBox {view_box!r} != '0 0 {doc.width} {doc.height}'")
    ids = {el.get("id") for el in root.iter() if isinstance(el.tag, str) and el.get("id")}
    missing = [layer.id for layer in doc.layers if layer.id not in ids]
    if missing:
        problems.append(f"layer ids missing from SVG: {missing}")
    drawables = {f"{{{SVG_NS}}}{t}" for t in ("path", "rect", "circle", "ellipse", "polygon", "polyline", "line")}
    if doc.layers and not any(el.tag in drawables for el in root.iter()):
        problems.append("no drawable elements")
    return problems


# --------------------------------------------------------------------------------------
# Checks + entrypoint
# --------------------------------------------------------------------------------------


def _check(name: MetricName, value: float, threshold: float, comparator: str) -> MetricCheck:
    ops = {"<": value < threshold, "<=": value <= threshold, ">=": value >= threshold, ">": value > threshold}
    return MetricCheck(
        name=name, value=float(value), threshold=float(threshold), comparator=comparator, passed=ops[comparator]
    )


def build_checks(
    *,
    ssim: float,
    ssim_min: float,
    mean_de: float,
    max_de: float,
    gap: float,
    iou: float | None,
    time_s: float,
    time_budget_s: float,
    svg_ok: bool,
) -> list[MetricCheck]:
    """One MetricCheck per QualityThresholds threshold, plus svg_valid (and alpha_iou if applicable)."""
    checks = [
        _check(MetricName.SSIM, ssim, ssim_min, ">="),
        _check(MetricName.MEAN_DELTA_E, mean_de, QualityThresholds.MAX_MEAN_DELTA_E, "<"),
        _check(MetricName.MAX_DELTA_E, max_de, QualityThresholds.MAX_DELTA_E, "<"),
        _check(MetricName.GAP_RATIO, gap, QualityThresholds.MAX_GAP_RATIO, "<="),
    ]
    if iou is not None:
        checks.append(_check(MetricName.ALPHA_IOU, iou, QualityThresholds.MIN_ALPHA_IOU, ">="))
    checks.append(_check(MetricName.PROCESSING_TIME, time_s, time_budget_s, "<"))
    checks.append(_check(MetricName.SVG_VALID, 1.0 if svg_ok else 0.0, 1.0, ">="))
    return checks


def evaluate(
    pre: PreprocessResult,
    palette: Palette,
    line_map: LineMap | None,
    doc: VectorDocument,
    bundle: ExportBundle,
    processing_time_s: float,
) -> QualityReport:
    """Compute SSIM, Delta-E, gap ratio, alpha IoU, node count, size and threshold checks."""
    return evaluate_detailed(pre, palette, line_map, doc, bundle, processing_time_s)[0]


def evaluate_detailed(
    pre: PreprocessResult,
    palette: Palette,
    line_map: LineMap | None,
    doc: VectorDocument,
    bundle: ExportBundle,
    processing_time_s: float,
) -> tuple[QualityReport, list[RegionDeltaE]]:
    """:func:`evaluate` plus the per-region Delta-E diagnostics behind mean/max Delta-E."""
    w, h = pre.source.width, pre.source.height
    src_rgb, src_alpha = source_rgba(pre)
    preview = load_preview(bundle.preview_png_path, w, h)
    prev_rgb, prev_alpha = preview[..., :3], preview[..., 3]

    ssim = ssim_score(src_rgb, src_alpha, prev_rgb, prev_alpha)
    regions = region_delta_es(pre, palette, line_map, doc, preview_rgb=prev_rgb, source_rgb=src_rgb)
    des = [r.delta_e for r in regions]
    mean_de = float(np.mean(des)) if des else 0.0
    max_de = float(np.max(des)) if des else 0.0
    gap = gap_ratio(src_alpha, prev_alpha)
    iou = None if src_alpha is None else alpha_iou(src_alpha, prev_alpha)
    svg_ok = not svg_problems(bundle.svg_path, doc)

    checks = build_checks(
        ssim=ssim,
        ssim_min=QualityThresholds.SSIM_MIN[doc.metadata.image_class],
        mean_de=mean_de,
        max_de=max_de,
        gap=gap,
        iou=iou,
        time_s=processing_time_s,
        time_budget_s=QualityThresholds.time_budget_s(w, h),
        svg_ok=svg_ok,
    )
    report = QualityReport(
        ssim=ssim,
        mean_delta_e=mean_de,
        max_delta_e=max_de,
        gap_ratio=gap,
        alpha_iou=iou,
        node_count=document_node_count(doc),
        file_size_bytes=bundle.svg_path.stat().st_size,
        processing_time_s=max(0.0, float(processing_time_s)),
        checks=checks,
    )
    return report, regions
