"""Stage 5: trace palette regions and strokes into cubic Bezier paths (``vectorize``).

Pipeline inside this stage
--------------------------
1. **Layer plan.** The line layer takes the DOMINANT palette color inside ``LineMap.mask``
   (the contract has one ``LineMap.color_rgb``, and QA scores every line layer against all
   mask pixels). Other colors holding >= 5 % of the mask (the blue arcs in 08) become
   *overlay* fills stacked ABOVE the line, so the line can pass under them at crossings.
   Base fills sit below: background first, then by area (largest first). Fills are re-grown
   under the line pixels from the nearest base-fill pixel, so they meet under the line.
   Speckles below ``preset.speckle_min_area_px`` are merged into a neighbour. For MIXED
   images the area is scaled by ``scale_factor**2 * 2``, because gradients and photographic
   areas quantise into noisy islands.
2. **Stacked masks.** Each layer's mask is its own pixels plus, restricted to pixels owned
   by layers *above* it, a closing (square kernel; for the line layer also horizontal and
   vertical line kernels that bridge crossings at staircase steps), every enclosed hole made
   only of upper-layer pixels, and (fills) a dilation of ``1 px + fit tolerance`` (source px).
   The bottom layer extends under EVERY opaque pixel (on opaque images it is the full canvas
   rectangle), which makes hairline gaps impossible by construction. Transparent (-1) pixels
   are never added, so they stay transparent.
3. **Crack-exact contours.** Each mask is traced with ``cv2.findContours`` on a
   half-pixel grid (pixel corners / edge midpoints / centers), so the contour runs exactly
   along pixel edges (not through boundary pixel centers, which would shrink every shape by
   0.5 px).
4. **Corners.** A crack vertex is a sharp corner if its *turn* (as vtracer's
   ``corner_threshold``) reaches ``corner_threshold_deg`` with chords of 3 AND 6 source px.
   The larger scale ignores aliasing notches and staircase steps, and both corners of a 1-px
   line cap still survive.
5. **Fitting.** Between corners, the edge midpoints are smoothed along the contour
   (Gaussian, sigma = 1 source px at smoothing 50) and fitted with Schneider's algorithm
   (least-squares cubic, windowed-centroid G1 tangents, Newton re-parameterisation, split at
   the worst point). A straight ``L`` is tried first. Long, nearly straight runs may deviate
   by an extra 0.5 source px (the staircase of an aliased edge), so shallow diagonals become
   a single segment. Collinear and duplicate nodes are removed after rounding.
6. **Centerline mode.** The skeleton (redundant L-corner pixels thinned) is walked as a graph:
   nodes are endpoints and junction clusters. Every chain touching a junction ends on the
   cluster's single shared centroid. Chains of the line color are fitted like open contours
   and grouped into stroke layers by width quantised to 0.5 source px. Overlay-color chains
   are left to their fill layers.

Tolerance and smoothing
-----------------------
``tol = simplify_tolerance_px * (0.5 + smoothing / 100)`` source px (``smoothing = 50`` gives the
preset value). ``smoothing = 0`` is polygonal: only ``L`` segments, via Douglas-Peucker splits,
with no contour smoothing. The corner threshold is scaled by ``0.75 + 0.5 * smoothing / 100``
and the along-contour sigma by ``smoothing / 50``, so smoother output has fewer corners and nodes.

Tracer benchmark (09_large_2000, 7 palette masks, this dev machine)
------------------------------------------------------------------
============================  ==========  =====================================================
tracer (per color mask)       time        notes
============================  ==========  =====================================================
vtracer 0.6 binary (file API) 2.06 s      In-process ``convert_pixels_to_svg`` / keyword args
                                          segfault on CPython 3.14; only the positional file
                                          API works (PNG round trip). No control over the
                                          stacking or the fit tolerance, and it traces pixel
                                          centers.
potracer 0.0.4 (pure Python)  2.02 s      For ONE 0.4 MP mask (~14 s for all 7). Best curve
                                          quality, but far over the 4 s budget.
skimage.find_contours         1.91 s      Contours only (no fitting); Python assembly loop.
cv2.findContours (this module) 0.10 s     Contours only; crack-exact on a half-pixel grid.
============================  ==========  =====================================================

So contours come from OpenCV and fitting is done here in NumPy. That also gives exact control over
corners, tolerance, junction sharing and stacking, which neither library offers.
Node counts against raw vtracer default output are in ``tests/test_vectorize.py``
(``test_node_count_vs_vtracer``, marked slow).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from contracts.schemas import (
    DocumentMetadata,
    ImageClass,
    ImageClassLabel,
    LineMap,
    LineMode,
    Palette,
    PreprocessResult,
    Settings,
    StageError,
    VectorDocument,
    VectorLayer,
    layer_name,
)

STAGE = "vectorize"

LINE_COLOR_MIN_SHARE = 0.05
"""A palette color is a line color if it holds at least this share of LineMap.mask pixels."""
UNDERLAP_SRC_PX = 1.0
"""Lower layers extend this far (plus the fit tolerance) under upper layers (source px)."""
FILL_CLOSE_SRC_PX = 1.0
"""Closing radius (source px) used to bridge notches cut into a fill by upper layers."""
LINE_CLOSE_MAX_PX = 8
"""Upper bound (processing px) of the closing radius that bridges crossings between line colors."""
CORNER_SCALE_SRC_PX = 3.0
"""Chord length (source px) over which turning angles are measured for corner detection."""
ALIAS_SRC_PX = 0.5
"""Extra deviation (source px) allowed for straight runs: the amplitude of an aliased staircase."""
STRAIGHT_RATIO = 0.02
"""...but only if the deviation is at most this fraction of the run length (keeps curves curved)."""
MIXED_SPECKLE_FACTOR = 2.0
"""MIXED images: speckle area is scaled to source px and multiplied by this."""
SMOOTH_SIGMA_SRC_PX = 1.0
"""Along-contour Gaussian sigma (source px) at smoothing = 50; removes aliasing stairs before fitting."""
WIDTH_QUANTUM_SRC_PX = 0.5
"""Stroke widths are quantised to multiples of this (source px) when grouping stroke layers."""
MAX_SPLIT_DEPTH = 40

Segment = tuple[str, tuple[float, ...]]
"""('L', (x, y)) or ('C', (x1, y1, x2, y2, x, y)) in processing space."""


# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FitParams:
    """Numeric fitting parameters in PROCESSING space, derived from Settings."""

    tolerance: float
    """Max deviation (processing px) of the fitted path from the traced boundary."""
    corner_deg: float
    """Minimum turn (degrees) for a vertex to be kept as a sharp corner."""
    corner_scale: float
    """Chord length (processing px) used to measure turning angles."""
    polygonal: bool
    """True for smoothing = 0: straight segments only."""
    scale: float
    """Processing px per source px (PreprocessResult.scale_factor)."""
    precision: int
    """Decimal places for output coordinates."""
    smooth_sigma: float = 0.0
    """Gaussian sigma (processing px, along the contour) applied to fit data between corners."""

    @classmethod
    def from_settings(cls, settings: Settings, scale: float) -> FitParams:
        """Map DetailPreset + smoothing onto processing-space parameters."""
        preset = settings.preset
        s = settings.smoothing / 100.0
        tol_src = preset.simplify_tolerance_px * (0.5 + s)
        return cls(
            tolerance=tol_src * scale,
            corner_deg=min(170.0, preset.corner_threshold_deg * (0.75 + 0.5 * s)),
            corner_scale=max(2.0, CORNER_SCALE_SRC_PX * scale),
            polygonal=settings.smoothing == 0,
            scale=scale,
            precision=preset.path_precision,
            smooth_sigma=SMOOTH_SIGMA_SRC_PX * scale * settings.smoothing / 50.0,
        )


@dataclass
class _LayerSpec:
    """A layer before it becomes a VectorLayer (ids are assigned once z-order is known)."""

    role: Literal["background", "fill", "line"]
    palette_index: int
    color_hex: str
    paths: list[str]
    stroke_width: float | None = None
    extra: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def vectorize(
    pre: PreprocessResult,
    image_class: ImageClass,
    palette: Palette,
    line_map: LineMap | None,
    settings: Settings,
) -> VectorDocument:
    """Trace palette regions (and strokes) into a stacked, gap-free VectorDocument in source space."""
    labels = np.array(palette.label_map, dtype=np.int32)  # writable copy
    if labels.shape != (pre.height, pre.width):
        raise StageError(STAGE, f"label_map shape {labels.shape} != processing size {(pre.height, pre.width)}")
    if line_map is not None and line_map.mask.shape != labels.shape:
        raise StageError(STAGE, f"line_map shape {line_map.mask.shape} != label_map shape {labels.shape}")
    params = FitParams.from_settings(settings, pre.scale_factor)
    speckle = settings.preset.speckle_min_area_px
    if image_class.label == ImageClassLabel.MIXED:
        # Gradient/photo regions quantise into noisy islands: judge their size in SOURCE px.
        speckle = round(speckle * max(1.0, pre.scale_factor) ** 2 * MIXED_SPECKLE_FACTOR)
    plan = plan_layers(labels, palette, line_map, speckle)
    centerline = settings.line_mode == LineMode.CENTERLINE
    h, w = labels.shape
    fill_close = max(1, round(FILL_CLOSE_SRC_PX * params.scale))
    fill_dilate = max(1, math.ceil(UNDERLAP_SRC_PX * params.scale + params.tolerance))
    line_close = 0
    if line_map is not None:
        line_close = int(min(LINE_CLOSE_MAX_PX, max(1, 2 * math.ceil(line_map.median_stroke_width) + 4)))

    specs: list[_LayerSpec] = []
    for z, entry in enumerate(plan.entries):
        if entry.is_line and centerline and line_map is not None:
            specs += _centerline_layers(line_map, plan, entry.palette_index, palette, params)
            continue
        extra = None if entry.is_line else plan.line_region & (plan.under == entry.palette_index)
        if z == 0 and not entry.is_line:
            extra = plan.zmap >= 0  # the bottom layer runs under every opaque pixel: no gaps possible
        close_r, dilate_r = (line_close, 0) if entry.is_line else (fill_close, fill_dilate)
        mask, offset = stacked_mask(
            plan.zmap,
            z,
            close_r,
            dilate_r,
            extra_own=extra,
            directional=entry.is_line,
            no_grow=None if entry.is_line else plan.line_region,
        )
        if _is_full_canvas(mask, offset, (h, w)):
            paths = [_rect_path(params, h, w)]
        else:
            paths = trace_mask(mask, offset, params)
        if not paths:
            continue
        role: Literal["background", "fill", "line"] = (
            "line" if entry.is_line else "background" if entry.palette_index == palette.background_index else "fill"
        )
        color_hex = palette.colors[entry.palette_index].hex
        specs.append(_LayerSpec(role=role, palette_index=entry.palette_index, color_hex=color_hex, paths=paths))

    layers: list[VectorLayer] = []
    for z, spec in enumerate(specs):
        lid, name = layer_name(spec.role, z + 1, spec.color_hex)
        layers.append(
            VectorLayer(
                id=lid,
                name=name,
                role=spec.role,
                color_hex=spec.color_hex,
                paths=spec.paths,
                z_order=z,
                is_stroke=spec.stroke_width is not None,
                stroke_width=spec.stroke_width,
                fill_rule="evenodd",
                palette_index=spec.palette_index,
            )
        )
    return VectorDocument(
        width=pre.source.width,
        height=pre.source.height,
        layers=layers,
        metadata=DocumentMetadata(
            source_filename=pre.source.filename,
            image_class=image_class.label,
            settings=settings,
            palette_hex=[c.hex for c in palette.colors],
        ),
    )


# --------------------------------------------------------------------------------------
# Layer planning (label preparation + z-order)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanEntry:
    """One z-slot of the stack: a palette color's fill, or the line layer."""

    palette_index: int
    is_line: bool = False


@dataclass
class LayerPlan:
    """Top-owner z-map plus what lies under the line layer."""

    entries: list[PlanEntry]
    """Bottom to top."""
    zmap: np.ndarray
    """(H, W) int32: index into ``entries`` of the layer visible at each pixel; -1 transparent."""
    line_region: np.ndarray
    """(H, W) bool: pixels owned by the line layer (LineMap.mask AND the line color)."""
    under: np.ndarray
    """(H, W) int32: palette index of the fill re-grown under line pixels (-1 elsewhere/none)."""


LINE = -10
"""Pseudo-label for line-layer pixels inside plan_layers."""


def line_colors(labels: np.ndarray, mask: np.ndarray, n_colors: int) -> list[int]:
    """Palette indices holding >= LINE_COLOR_MIN_SHARE of the opaque mask pixels, most frequent first."""
    sel = labels[mask & (labels >= 0)]
    if sel.size == 0:
        return []
    counts = np.bincount(sel, minlength=n_colors)
    keep = np.flatnonzero(counts >= LINE_COLOR_MIN_SHARE * sel.size)
    return [int(i) for i in keep[np.argsort(-counts[keep], kind="stable")]]


def plan_layers(labels: np.ndarray, palette: Palette, line_map: LineMap | None, speckle: int) -> LayerPlan:
    """Decide the layer stack.

    The line layer holds the DOMINANT line color inside LineMap.mask (the contract has a single
    ``LineMap.color_rgb``, and QA scores every line layer against all mask pixels). Other colors
    that make up >= 5 % of the mask (e.g. the blue arcs in 08) become "overlay" fill layers ABOVE
    the line layer, so the line can pass under them at crossings. Base fills (background first,
    then largest first) sit below and are re-grown under line pixels.
    """
    n = len(palette.colors)
    top = labels.copy()
    overlay: list[int] = []
    line_color: int | None = None
    if line_map is not None and bool(np.any(line_map.mask)):
        mask = np.asarray(line_map.mask)
        colors = line_colors(labels, mask, n)
        if colors:
            line_color, overlay = colors[0], colors[1:]
            top[mask & (labels == line_color)] = LINE
    top = remove_speckles(top, speckle)
    line_region = top == LINE
    under = np.full(labels.shape, -1, dtype=np.int32)
    if line_region.any():
        tmp = top.copy()
        tmp[line_region] = -2
        seeds = tmp >= 0
        if overlay:
            seeds &= ~np.isin(tmp, overlay)
        grown = fill_unknown(tmp, seeds)
        under[line_region] = grown[line_region]
    counts = np.bincount(top[top >= 0].ravel(), minlength=n)
    base = [int(i) for i in np.argsort(-counts, kind="stable") if counts[i] > 0 and int(i) not in overlay]
    bg = palette.background_index
    if bg is not None and bg in base:
        base.remove(bg)
        base.insert(0, bg)
    entries = [PlanEntry(i) for i in base]
    if line_color is not None and line_region.any():
        entries.append(PlanEntry(line_color, is_line=True))
    entries += [PlanEntry(i) for i in overlay if counts[i] > 0]
    rank = np.full(n, -1, dtype=np.int32)
    zmap = np.full(labels.shape, -1, dtype=np.int32)
    for z, e in enumerate(entries):
        if e.is_line:
            zmap[line_region] = z
        else:
            rank[e.palette_index] = z
    valid = top >= 0
    zmap[valid] = rank[top[valid]]
    return LayerPlan(entries=entries, zmap=zmap, line_region=line_region, under=under)


def fill_unknown(labels: np.ndarray, seeds: np.ndarray | None = None) -> np.ndarray:
    """Replace -2 ("unknown") pixels with the label of the nearest seed pixel.

    Seeds default to every pixel with a label >= 0. Other negative labels (e.g. -1 transparent)
    are neither seeds nor filled. Without any seed the input is returned unchanged.
    """
    unknown = labels == -2
    if not unknown.any():
        return labels
    seeds = labels >= 0 if seeds is None else seeds
    if not seeds.any():
        return labels
    src = np.where(seeds, 0, 255).astype(np.uint8)
    _, nearest = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(int(nearest.max()) + 1, dtype=np.int32)
    lut[nearest[seeds]] = labels[seeds]
    out = labels.copy()
    out[unknown] = lut[nearest[unknown]]
    return out


def remove_speckles(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Merge 8-connected components smaller than ``min_area`` into the nearest other label.

    Every label except -1 (transparent) takes part, including the LINE pseudo-label.
    """
    if min_area <= 1:
        return labels
    out = labels.copy()
    changed = False
    for value in np.unique(labels):
        if value == -1:
            continue
        n, comp, stats, _ = cv2.connectedComponentsWithStats((labels == value).astype(np.uint8), connectivity=8)
        small = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < min_area) + 1
        if small.size == 0:
            continue
        out[np.isin(comp, small)] = -2
        changed = True
    if not changed:
        return labels
    seeds = (out != -2) & (out != -1)
    if not seeds.any():
        return labels
    return fill_unknown(out, seeds)


# --------------------------------------------------------------------------------------
# Stacked masks
# --------------------------------------------------------------------------------------


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def stacked_mask(
    zmap: np.ndarray,
    z: int,
    close_r: int,
    dilate_r: int,
    extra_own: np.ndarray | None = None,
    directional: bool = False,
    no_grow: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Mask of layer ``z`` extended under the layers above it (zmap > z); never onto zmap < 0.

    Own pixels are ``zmap == z`` plus ``extra_own`` (hidden pixels the layer should cover, e.g.
    its fill re-grown under a line). The mask is then extended into upper-layer pixels by:

    * a closing with a square of half-size ``close_r`` (a disk's 1-px tip slips into narrow gaps,
      so a disk never bridges a line cut by a crossing stroke), plus horizontal/vertical line
      closings of the 1-px-thickened mask when ``directional`` (offset gaps at staircase steps);
    * filling enclosed holes that contain only upper-layer pixels;
    * a dilation by ``dilate_r``.

    Closing and dilation never add ``no_grow`` pixels (fills do not bulge under the line layer;
    they are re-grown there instead). Returns (mask cropped to its bounding box plus margin with
    an empty 1-px border, (row, col) offset of mask[0, 0] in the full image).
    """
    own_full = zmap == z
    if extra_own is not None:
        own_full = own_full | (extra_own & (zmap > z))
    rows = np.flatnonzero(own_full.any(axis=1))
    if rows.size == 0:
        return np.zeros((3, 3), dtype=bool), (-1, -1)
    cols = np.flatnonzero(own_full.any(axis=0))
    margin = close_r + dilate_r + 2
    r0, r1 = max(0, rows[0] - margin), min(zmap.shape[0], rows[-1] + 1 + margin)
    c0, c1 = max(0, cols[0] - margin), min(zmap.shape[1], cols[-1] + 1 + margin)
    own = np.pad(own_full[r0:r1, c0:c1], 1).astype(np.uint8)
    higher = np.pad(zmap[r0:r1, c0:c1] > z, 1)
    growable = higher if no_grow is None else higher & ~np.pad(no_grow[r0:r1, c0:c1], 1)
    mask = own.copy()
    if higher.any():
        if close_r > 0:
            closed = cv2.morphologyEx(
                own,
                cv2.MORPH_CLOSE,
                np.ones((2 * close_r + 1, 2 * close_r + 1), np.uint8),
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            if directional:
                # Shallow lines are cut by crossing strokes at a staircase step; a square cannot
                # bridge such an offset gap, a line kernel along the (thickened) stroke can.
                thick = cv2.dilate(own, np.ones((3, 3), np.uint8))
                length = 2 * close_r + 1
                for kernel in (np.ones((1, length), np.uint8), np.ones((length, 1), np.uint8)):
                    closed |= cv2.morphologyEx(
                        thick, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
                    )
            mask |= (closed.astype(bool) & growable).astype(np.uint8)
        # Holes enclosed by the mask that contain nothing but upper-layer pixels are filled.
        n, comp = cv2.connectedComponents((mask == 0).astype(np.uint8), connectivity=4)
        if n > 2:
            outside = np.unique(np.concatenate([comp[0], comp[-1], comp[:, 0], comp[:, -1]]))
            fillable = ~(np.bincount(comp[(mask == 0) & ~higher], minlength=n) > 0)
            fillable[0] = False
            fillable[outside] = False
            if fillable.any():
                mask |= fillable[comp].astype(np.uint8)
        if dilate_r > 0:
            grown = cv2.dilate(mask, _disk(dilate_r), borderType=cv2.BORDER_CONSTANT, borderValue=0)
            mask |= (grown.astype(bool) & growable).astype(np.uint8)
    mask[0, :] = mask[-1, :] = 0
    mask[:, 0] = mask[:, -1] = 0
    return mask.astype(bool), (r0 - 1, c0 - 1)


def _is_full_canvas(mask: np.ndarray, offset: tuple[int, int], shape: tuple[int, int]) -> bool:
    h, w = shape
    if offset != (-1, -1) or mask.shape != (h + 2, w + 2):
        return False
    return bool(mask[1:-1, 1:-1].all())


def _rect_path(params: FitParams, h: int, w: int) -> str:
    x1, y1 = _fmt(w / params.scale, params.precision), _fmt(h / params.scale, params.precision)
    return f"M0 0L{x1} 0L{x1} {y1}L0 {y1}Z"


# --------------------------------------------------------------------------------------
# Contour tracing (crack-exact) and path assembly
# --------------------------------------------------------------------------------------


def crack_contours(mask: np.ndarray) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Trace a bool mask (with an empty 1-px border) along pixel edges.

    Returns [(outer, [holes...]), ...]; each contour is an (N, 2) int array of HALF-pixel
    coordinates (x2, y2), 4-connected and closed (last point connects to the first). Even/even
    points are pixel corners, points with one odd coordinate are edge midpoints.
    """
    h, w = mask.shape
    grid = np.zeros((2 * h + 1, 2 * w + 1), dtype=np.uint8)
    grid[1::2, 1::2] = mask
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hier = hierarchy[0]
    out: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for i, c in enumerate(contours):
        if hier[i][3] != -1:
            continue
        holes = []
        child = hier[i][2]
        while child != -1:
            holes.append(_four_connect(contours[child].reshape(-1, 2)))
            child = hier[child][0]
        out.append((_four_connect(c.reshape(-1, 2)), holes))
    return out


def _four_connect(pts: np.ndarray) -> np.ndarray:
    """Insert the pixel-corner point skipped by each diagonal step so every step is axis-aligned."""
    pts = pts.astype(np.int64)
    if len(pts) < 2:
        return pts
    d = np.roll(pts, -1, axis=0) - pts
    diag = np.flatnonzero((d[:, 0] != 0) & (d[:, 1] != 0))
    if diag.size == 0:
        return pts
    p = pts[diag]
    cand_a = np.stack([p[:, 0] + d[diag, 0], p[:, 1]], axis=1)
    cand_b = np.stack([p[:, 0], p[:, 1] + d[diag, 1]], axis=1)
    use_a = (cand_a[:, 0] % 2 == 0) & (cand_a[:, 1] % 2 == 0)
    extra = np.where(use_a[:, None], cand_a, cand_b)
    return np.insert(pts, diag + 1, extra, axis=0)


def trace_mask(mask: np.ndarray, offset: tuple[int, int], params: FitParams) -> list[str]:
    """Trace a (cropped, bordered) mask into path strings: one path per outer contour + holes."""
    r0, c0 = offset
    paths = []
    for outer, holes in crack_contours(mask):
        parts = []
        for contour in [outer, *holes]:
            if len(contour) < 4:
                continue
            pts = contour.astype(np.float64) / 2.0
            pts[:, 0] += c0
            pts[:, 1] += r0
            corner_mask = (contour[:, 0] % 2 == 0) & (contour[:, 1] % 2 == 0)
            start, segs = fit_closed(pts, corner_mask, params)
            d = _format_subpath(start, segs, params, closed=True)
            if d:
                parts.append(d)
        if parts:
            paths.append("".join(parts))
    return paths


# --------------------------------------------------------------------------------------
# Corner detection
# --------------------------------------------------------------------------------------


def _turn_angles(pts: np.ndarray, closed: bool, scale: float) -> np.ndarray:
    """Absolute turning angle (degrees) at every point, chords of arclength ``scale`` each side."""
    n = len(pts)
    if n < 3:
        return np.zeros(n)
    if closed:
        ext = np.vstack([pts, pts[:1]])
        seg = np.linalg.norm(np.diff(ext, axis=0), axis=1)
        total = float(seg.sum())
        s = np.concatenate([[0.0], np.cumsum(seg[:-1])])
        reach = min(scale, total / 4.0)
        s3 = np.concatenate([s - total, s, s + total])
        p3 = np.vstack([pts, pts, pts])
        fwd = np.clip(np.searchsorted(s3, s + reach, side="left"), 0, 3 * n - 1)
        back = np.clip(np.searchsorted(s3, s - reach, side="right") - 1, 0, 3 * n - 1)
        a = pts - p3[back]
        b = p3[fwd] - pts
    else:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        reach = min(scale, float(s[-1]) / 4.0) if s[-1] > 0 else scale
        fwd = np.clip(np.searchsorted(s, s + reach, side="left"), 0, n - 1)
        back = np.clip(np.searchsorted(s, s - reach, side="right") - 1, 0, n - 1)
        a = pts - pts[back]
        b = pts[fwd] - pts
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = (a * b).sum(axis=1)
    ang = np.degrees(np.abs(np.arctan2(cross, dot)))
    valid = (np.linalg.norm(a, axis=1) > 1e-9) & (np.linalg.norm(b, axis=1) > 1e-9)
    return np.where(valid, ang, 0.0)


def find_corners(
    pts: np.ndarray, candidates: np.ndarray, closed: bool, threshold: float, scale: float, margin: float = 10.0
) -> np.ndarray:
    """Indices of sharp corners.

    A candidate is a corner if its turn reaches ``threshold`` at chord lengths ``scale`` AND
    ``2 * scale`` (so aliasing notches are ignored) and it is within ``margin`` degrees
    of the strongest candidate within ``scale / 2`` arclength (so both corners of a thin
    line cap survive, while staircase vertices next to a big corner do not).
    """
    ang = _turn_angles(pts, closed, scale)
    # A real corner keeps its turn at twice the chord length; a 1-2 px notch does not.
    ang = np.minimum(ang, _turn_angles(pts, closed, 2.0 * scale))
    cand = np.flatnonzero(candidates & (ang >= threshold))
    if cand.size <= 1:
        return cand
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]) if closed else pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    pos = s[cand]
    keep = []
    half = scale / 2.0
    for k, i in enumerate(cand):
        dist = np.abs(pos - pos[k])
        if closed:
            dist = np.minimum(dist, total - dist)
        near = cand[dist <= half]
        if ang[i] >= ang[near].max() - margin:
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


# --------------------------------------------------------------------------------------
# Curve fitting (Schneider)
# --------------------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-12 else np.zeros(2)


def _tangent(pts: np.ndarray, i: int, reach: float) -> np.ndarray:
    """Direction of travel at pts[i], robust to pixel staircases.

    Uses the centroid of the points within ``reach`` arclength after pts[i] minus the centroid of
    those before it (pts[i] itself at an end), which averages out aliasing steps.
    """
    n = len(pts)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    j1 = int(np.searchsorted(s, s[i] + reach, side="right"))
    j0 = int(np.searchsorted(s, s[i] - reach, side="left"))
    fwd = pts[i + 1 : max(j1, i + 2)] if i < n - 1 else pts[i : i + 1]
    back = pts[min(j0, i - 1) : i] if i > 0 else pts[i : i + 1]
    return _unit(fwd.mean(axis=0) - back.mean(axis=0))


def _chord_deviation(pts: np.ndarray) -> tuple[float, int]:
    """Max distance of pts from the segment pts[0]-pts[-1] and the index where it occurs."""
    p0, p1 = pts[0], pts[-1]
    d = p1 - p0
    length = float(np.hypot(d[0], d[1]))
    rel = pts - p0
    if length < 1e-9:
        dist = np.hypot(rel[:, 0], rel[:, 1])
    else:
        t = np.clip((rel @ d) / (length * length), 0.0, 1.0)
        proj = p0 + t[:, None] * d
        dist = np.hypot(*(pts - proj).T)
    i = int(np.argmax(dist))
    return float(dist[i]), i


def _bezier_eval(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    mt = 1.0 - u
    return (
        (mt**3)[:, None] * ctrl[0]
        + (3 * mt**2 * u)[:, None] * ctrl[1]
        + (3 * mt * u**2)[:, None] * ctrl[2]
        + (u**3)[:, None] * ctrl[3]
    )


def _generate(pts: np.ndarray, u: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Least-squares cubic with fixed end tangents (Schneider, Graphics Gems I)."""
    p0, p3 = pts[0], pts[-1]
    mt = 1.0 - u
    b0, b1, b2, b3 = mt**3, 3 * mt**2 * u, 3 * mt * u**2, u**3
    a1 = b1[:, None] * t0
    a2 = b2[:, None] * t1
    c00 = float((a1 * a1).sum())
    c01 = float((a1 * a2).sum())
    c11 = float((a2 * a2).sum())
    tmp = pts - ((b0 + b1)[:, None] * p0 + (b2 + b3)[:, None] * p3)
    x0 = float((a1 * tmp).sum())
    x1 = float((a2 * tmp).sum())
    det = c00 * c11 - c01 * c01
    seg_len = float(np.hypot(*(p3 - p0)))
    alpha1 = alpha2 = seg_len / 3.0
    if abs(det) > 1e-12:
        a_l = (x0 * c11 - x1 * c01) / det
        a_r = (c00 * x1 - c01 * x0) / det
        eps = 1e-6 * max(seg_len, 1e-9)
        if a_l > eps and a_r > eps and a_l < 2.0 * seg_len + 1.0 and a_r < 2.0 * seg_len + 1.0:
            alpha1, alpha2 = a_l, a_r
    return np.array([p0, p0 + t0 * alpha1, p3 + t1 * alpha2, p3])


def _reparameterize(pts: np.ndarray, ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    """One Newton-Raphson step per point towards the closest curve parameter."""
    q = _bezier_eval(ctrl, u)
    d1 = 3.0 * (ctrl[1:] - ctrl[:-1])
    d2 = 2.0 * (d1[1:] - d1[:-1])
    mt = 1.0 - u
    q1 = (mt**2)[:, None] * d1[0] + (2 * mt * u)[:, None] * d1[1] + (u**2)[:, None] * d1[2]
    q2 = mt[:, None] * d2[0] + u[:, None] * d2[1]
    diff = q - pts
    num = (diff * q1).sum(axis=1)
    den = (q1 * q1).sum(axis=1) + (diff * q2).sum(axis=1)
    step = np.where(np.abs(den) > 1e-12, num / np.where(np.abs(den) > 1e-12, den, 1.0), 0.0)
    out = np.clip(u - step, 0.0, 1.0)
    out[0], out[-1] = 0.0, 1.0
    return out


def _chord_params(pts: np.ndarray) -> np.ndarray:
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s / s[-1] if s[-1] > 0 else np.linspace(0.0, 1.0, len(pts))


def _max_error(pts: np.ndarray, ctrl: np.ndarray, u: np.ndarray) -> tuple[float, int]:
    dist = np.hypot(*(_bezier_eval(ctrl, u) - pts).T)
    i = int(np.argmax(dist))
    return float(dist[i]), i


def fit_open(
    pts: np.ndarray,
    t0: np.ndarray | None,
    t1: np.ndarray | None,
    params: FitParams,
    out: list[Segment] | None = None,
    depth: int = 0,
) -> list[Segment]:
    """Fit pts (first/last fixed) with lines and cubics; append segments to ``out``.

    ``t0`` is the unit tangent leaving pts[0], ``t1`` the unit tangent leaving pts[-1] backwards
    (towards the interior). None means "estimate one-sided from the data".
    """
    out = [] if out is None else out
    n = len(pts)
    end = (float(pts[-1, 0]), float(pts[-1, 1]))
    if n <= 2:
        out.append(("L", end))
        return out
    tol = params.tolerance
    dev, far = _chord_deviation(pts)
    chord = float(np.hypot(*(pts[-1] - pts[0])))
    if dev <= tol or (dev <= tol + ALIAS_SRC_PX * params.scale and dev <= STRAIGHT_RATIO * chord):
        # Long, nearly straight runs may deviate by an extra half source pixel: that is the
        # staircase of an aliased straight edge, not shape.
        out.append(("L", end))
        return out
    reach = params.corner_scale
    split = far
    if not params.polygonal and n >= 4 and depth < MAX_SPLIT_DEPTH:
        t0v = t0 if t0 is not None else _tangent(pts, 0, reach)
        t1v = t1 if t1 is not None else -_tangent(pts, n - 1, reach)
        u = _chord_params(pts)
        ctrl = _generate(pts, u, t0v, t1v)
        err, split = _max_error(pts, ctrl, u)
        if err > tol and err < 4.0 * tol:
            for _ in range(4):
                u = _reparameterize(pts, ctrl, u)
                ctrl = _generate(pts, u, t0v, t1v)
                err, split = _max_error(pts, ctrl, u)
                if err <= tol:
                    break
        if err <= tol:
            out.append(("C", (*map(float, ctrl[1]), *map(float, ctrl[2]), *end)))
            return out
    if depth >= MAX_SPLIT_DEPTH:
        out.append(("L", end))
        return out
    split = int(min(max(split, 1), n - 2))
    tc = None if params.polygonal else _tangent(pts, split, reach)
    fit_open(pts[: split + 1], t0, None if tc is None else -tc, params, out, depth + 1)
    fit_open(pts[split:], tc, t1, params, out, depth + 1)
    return out


def _gauss(sigma: float) -> np.ndarray:
    r = max(1, math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return k / k.sum()


def smooth_open(pts: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth an open polyline along its index; both endpoints stay fixed (odd reflection)."""
    n = len(pts)
    if sigma <= 0 or n < 4:
        return pts
    k = _gauss(sigma)
    r = min(len(k) // 2, n - 1)
    k = _gauss(sigma)[len(k) // 2 - r : len(k) // 2 + r + 1]
    k = k / k.sum()
    left = 2 * pts[0] - pts[r:0:-1]
    right = 2 * pts[-1] - pts[-2 : -r - 2 : -1]
    ext = np.vstack([left, pts, right])
    out = np.stack([np.convolve(ext[:, 0], k, mode="valid"), np.convolve(ext[:, 1], k, mode="valid")], axis=1)
    out[0], out[-1] = pts[0], pts[-1]
    return out


def smooth_closed(pts: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a closed polyline along its index (wrap-around)."""
    n = len(pts)
    if sigma <= 0 or n < 4:
        return pts
    k = _gauss(sigma)
    r = len(k) // 2
    if r >= n:
        return pts
    ext = np.vstack([pts[-r:], pts, pts[:r]])
    return np.stack([np.convolve(ext[:, 0], k, mode="valid"), np.convolve(ext[:, 1], k, mode="valid")], axis=1)


def fit_closed(
    pts: np.ndarray, corner_mask: np.ndarray, params: FitParams
) -> tuple[tuple[float, float], list[Segment]]:
    """Fit a closed crack contour (half-pixel spaced, in processing px).

    ``corner_mask`` marks pixel-corner points (the only corner candidates); edge midpoints are
    the fitting data. Returns (start point, segments) with the last segment ending at the start.
    """
    n = len(pts)
    prev_d = pts - np.roll(pts, 1, axis=0)
    next_d = np.roll(pts, -1, axis=0) - pts
    turning = np.abs(prev_d[:, 0] * next_d[:, 1] - prev_d[:, 1] * next_d[:, 0]) > 1e-9
    corners = find_corners(pts, corner_mask & turning, True, params.corner_deg, params.corner_scale)
    keep = ~corner_mask
    if corners.size == 0:
        data = pts[keep]
        if len(data) < 3:
            data = pts
        data = smooth_closed(data, params.smooth_sigma)
        start = data[0]
        loop = np.vstack([data, data[:1]])
        reach = params.corner_scale
        wrap = np.vstack([data[-len(data) // 2 :], data, data[: len(data) // 2 + 1]])
        t = _tangent(wrap, len(data) - len(data) // 2 if len(data) > 1 else 0, reach)
        t = t if not params.polygonal else None
        segs = fit_open(loop, t, None if t is None else -t, params)
        return (float(start[0]), float(start[1])), segs
    segs: list[Segment] = []
    rolled_idx = np.roll(np.arange(n), -int(corners[0]))
    corner_set = np.zeros(n, dtype=bool)
    corner_set[corners] = True
    order_corner = corner_set[rolled_idx]
    order_keep = keep[rolled_idx]
    order_pts = pts[rolled_idx]
    cpos = np.flatnonzero(order_corner)
    bounds = [*cpos.tolist(), n]
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        idx = np.arange(a, b + 1)
        inner = idx[1:-1]
        inner = inner[order_keep[inner % n]]
        chunk = np.vstack([order_pts[a], order_pts[inner % n], order_pts[b % n]])
        fit_open(smooth_open(chunk, params.smooth_sigma), None, None, params, segs)
    start = order_pts[0]
    return (float(start[0]), float(start[1])), segs


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------


def _fmt(v: float, precision: int) -> str:
    s = f"{v:.{precision}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


def _format_subpath(start: tuple[float, float], segs: list[Segment], params: FitParams, closed: bool) -> str:
    """Scale to source space, round, drop degenerate/collinear nodes and emit 'M..L..C..Z'."""
    inv = 1.0 / params.scale
    p = params.precision

    def q(x: float, y: float) -> tuple[float, float]:
        return round(x * inv, p), round(y * inv, p)

    cur = q(*start)
    first = cur
    items: list[tuple[str, tuple[float, ...]]] = []
    for kind, vals in segs:
        if kind == "L":
            pt = q(vals[0], vals[1])
            if pt == cur:
                continue
            items.append(("L", pt))
            cur = pt
        else:
            c1, c2, pt = q(vals[0], vals[1]), q(vals[2], vals[3]), q(vals[4], vals[5])
            if pt == cur and c1 == cur and c2 == cur:
                continue
            items.append(("C", (*c1, *c2, *pt)))
            cur = pt
    items = _merge_collinear(first, items, 10.0 ** (-p))
    if closed and items and items[-1][0] == "L" and items[-1][1] == first:
        items.pop()  # Z draws the closing line
    if closed and len(items) < 2 and not any(k == "C" for k, _ in items):
        return ""  # degenerate (zero-area) contour
    if not closed and not items:
        return ""
    out = [f"M{_fmt(first[0], p)} {_fmt(first[1], p)}"]
    for kind, vals in items:
        out.append(kind + " ".join(_fmt(v, p) for v in vals))
    if closed:
        out.append("Z")
    return "".join(out)


def _merge_collinear(
    start: tuple[float, float], items: list[tuple[str, tuple[float, ...]]], eps: float
) -> list[tuple[str, tuple[float, ...]]]:
    """Drop the middle node of consecutive L-L runs that are collinear (within rounding)."""
    out: list[tuple[str, tuple[float, ...]]] = []
    for kind, vals in items:
        if kind == "L" and out and out[-1][0] == "L":
            a = np.array(_prev_end(out, start))
            b = np.array(out[-1][1])
            c = np.array(vals)
            ac = c - a
            length = float(np.hypot(*ac))
            if length > 0:
                dist = abs(ac[0] * (b - a)[1] - ac[1] * (b - a)[0]) / length
                t = float(np.dot(b - a, ac)) / (length * length)
                if dist <= max(eps, 0.02) and 0.0 < t < 1.0:
                    out[-1] = ("L", tuple(vals))
                    continue
        out.append((kind, tuple(vals)))
    return out


def _prev_end(out: list[tuple[str, tuple[float, ...]]], start: tuple[float, float]) -> tuple[float, float]:
    """End point of the segment before the last one in ``out`` (or the subpath start)."""
    if len(out) < 2:
        return start
    vals = out[-2][1]
    return (vals[-2], vals[-1])


# --------------------------------------------------------------------------------------
# Centerline strokes
# --------------------------------------------------------------------------------------

_NEIGHBORS = [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
"""4-neighbours first, so walks prefer axis steps over diagonal shortcuts."""


@dataclass
class Chain:
    """One skeleton chain between graph nodes (or a closed loop)."""

    points: np.ndarray
    """(N, 2) x, y processing px (pixel centers; junction ends replaced by the shared point)."""
    pixels: list[tuple[int, int]]
    closed: bool


def thin_corners(sk: np.ndarray) -> np.ndarray:
    """Remove redundant L-corner pixels (exactly two neighbours: one vertical, one horizontal).

    Such a pixel is not needed for 8-connectivity, and keeping it makes its two neighbours look
    like degree-3 junctions. One pass, and only pixels whose neighbours are not themselves
    being removed.
    """
    p = np.pad(sk, 1)
    c = p[1:-1, 1:-1]
    n, s_, w_, e = p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
    diag = p[:-2, :-2].astype(np.int8) + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:]
    four = n.astype(np.int8) + s_ + w_ + e
    corner = c & (four == 2) & (n ^ s_) & (w_ ^ e) & (diag == 0)
    if not corner.any():
        return sk
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    crowded = cv2.filter2D(corner.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT) > 0
    return sk & ~(corner & ~crowded)


def skeleton_chains(skeleton: np.ndarray) -> list[Chain]:
    """Walk a 1-px 8-connected skeleton into chains.

    Nodes are endpoints (degree 1), isolated pixels and junction clusters (8-connected groups of
    degree >= 3 pixels). Each junction cluster has one representative point (its centroid)
    that every chain touching the cluster starts or ends on, so chains share it exactly.
    """
    sk = thin_corners(np.asarray(skeleton, dtype=bool))
    h, w = sk.shape
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    deg = cv2.filter2D(sk.astype(np.uint8), cv2.CV_16S, k, borderType=cv2.BORDER_CONSTANT)
    node = sk & (deg != 2)
    ncl, cluster = cv2.connectedComponents(node.astype(np.uint8), connectivity=8)
    reps: dict[int, tuple[float, float]] = {}
    if ncl > 1:
        ys, xs = np.nonzero(node)
        cl = cluster[ys, xs]
        cnt = np.bincount(cl, minlength=ncl)
        mx = np.bincount(cl, weights=xs, minlength=ncl) / np.maximum(cnt, 1)
        my = np.bincount(cl, weights=ys, minlength=ncl) / np.maximum(cnt, 1)
        reps = {int(c): (float(mx[c]) + 0.5, float(my[c]) + 0.5) for c in range(1, ncl)}
    visited = np.zeros_like(sk)

    def nbrs(y: int, x: int) -> list[tuple[int, int]]:
        res = []
        for dy, dx in _NEIGHBORS:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and sk[yy, xx]:
                res.append((yy, xx))
        return res

    def point(p: tuple[int, int]) -> tuple[float, float]:
        if node[p]:
            return reps[int(cluster[p])]
        return (p[1] + 0.5, p[0] + 0.5)

    chains: list[Chain] = []
    node_pixels = list(zip(*np.nonzero(node), strict=True))
    for start in node_pixels:
        start = (int(start[0]), int(start[1]))
        c_start = int(cluster[start])
        for first in nbrs(*start):
            if node[first]:  # same cluster (adjacent node pixels always share one)
                continue
            if visited[first]:
                continue
            path = [start, first]
            visited[first] = True
            prev, cur = start, first
            while True:
                cand = [p for p in nbrs(*cur) if p != prev]
                ends = [p for p in cand if node[p] and not (int(cluster[p]) == c_start and len(path) < 4)]
                if ends:
                    path.append(ends[0])
                    break
                nxt = [p for p in cand if not node[p] and not visited[p]]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                visited[cur] = True
                path.append(cur)
            chains.append(Chain(np.array([point(p) for p in path]), path, closed=False))
        if not nbrs(*start):
            chains.append(Chain(np.array([point(start)]), [start], closed=False))
    # Loops without any node.
    rest = sk & ~node & ~visited
    for y, x in zip(*np.nonzero(rest), strict=True):
        start = (int(y), int(x))
        if visited[start]:
            continue
        path = [start]
        visited[start] = True
        prev, cur = start, start
        while True:
            nxt = [p for p in nbrs(*cur) if p != prev and not visited[p]]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            visited[cur] = True
            path.append(cur)
        closed = len(path) > 2 and start in nbrs(*path[-1])
        chains.append(Chain(np.array([point(p) for p in path]), path, closed=closed))
    return chains


def fit_chain(chain: Chain, params: FitParams) -> tuple[tuple[float, float], list[Segment]]:
    """Fit an open or closed skeleton chain with corners preserved; returns (start, segments)."""
    pts = chain.points.astype(np.float64)
    if len(pts) == 1:
        x, y = pts[0]
        return (x - 0.25, y), [("L", (x + 0.25, y))]
    if chain.closed:
        corners = find_corners(pts, np.ones(len(pts), bool), True, params.corner_deg, params.corner_scale)
        if corners.size == 0:
            pts = smooth_closed(pts, params.smooth_sigma)
            loop = np.vstack([pts, pts[:1]])
            return (float(pts[0, 0]), float(pts[0, 1])), fit_open(loop, None, None, params)
        pts = np.roll(pts, -int(corners[0]), axis=0)
        cidx = [*(np.sort((corners - corners[0]) % len(pts))).tolist(), len(pts)]
        pts = np.vstack([pts, pts[:1]])
    else:
        corners = find_corners(pts, np.ones(len(pts), bool), False, params.corner_deg, params.corner_scale)
        cidx = sorted({0, *corners.tolist(), len(pts) - 1})
    segs: list[Segment] = []
    for a, b in zip(cidx[:-1], cidx[1:], strict=True):
        if b > a:
            fit_open(smooth_open(pts[a : b + 1], params.smooth_sigma), None, None, params, segs)
    return (float(pts[0, 0]), float(pts[0, 1])), segs


def _centerline_layers(
    line_map: LineMap, plan: LayerPlan, color: int, palette: Palette, params: FitParams
) -> list[_LayerSpec]:
    """Skeleton chains of the line color -> stroke layers grouped by quantised width.

    The whole skeleton is walked (so junctions are found even where another color crosses),
    then chains lying mostly outside the line layer's pixels (e.g. strokes of an overlay color,
    which are traced as fills above) are dropped.
    """
    width_map = np.asarray(line_map.width_map)
    groups: dict[float, list[str]] = {}
    for chain in skeleton_chains(np.asarray(line_map.skeleton)):
        ys = np.array([p[0] for p in chain.pixels])
        xs = np.array([p[1] for p in chain.pixels])
        if plan.line_region[ys, xs].mean() < 0.5:
            continue
        widths = width_map[ys, xs]
        widths = widths[widths > 0]
        w_proc = float(np.median(widths)) if widths.size else max(1.0, line_map.median_stroke_width)
        w_src = max(WIDTH_QUANTUM_SRC_PX, round(w_proc / params.scale / WIDTH_QUANTUM_SRC_PX) * WIDTH_QUANTUM_SRC_PX)
        start, segs = fit_chain(chain, params)
        d = _format_subpath(start, segs, params, closed=chain.closed)
        if d:
            groups.setdefault(w_src, []).append(d)
    # Wider strokes first so thin detail stays visible on top.
    hex_ = palette.colors[color].hex
    return [
        _LayerSpec(role="line", palette_index=color, color_hex=hex_, paths=paths, stroke_width=float(width))
        for width, paths in sorted(groups.items(), key=lambda kv: -kv[0])
    ]


__all__ = [
    "Chain",
    "FitParams",
    "crack_contours",
    "fill_unknown",
    "find_corners",
    "fit_chain",
    "fit_closed",
    "fit_open",
    "line_colors",
    "remove_speckles",
    "skeleton_chains",
    "thin_corners",
    "plan_layers",
    "stacked_mask",
    "trace_mask",
    "vectorize",
]
