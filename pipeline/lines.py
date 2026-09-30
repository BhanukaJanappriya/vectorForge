"""Stage 4: line & edge extraction.

``extract_lines(pre, image_class, settings) -> LineMap``

Produces, in processing space:

* ``mask``      -- binary stroke mask (anti-aliased edges resolved at 50 % ink coverage),
* ``skeleton``  -- 1-px wide, 8-connected centerline (subset of ``mask``), spurs pruned,
* ``width_map`` -- ``2 x distance-transform`` of the mask on skeleton pixels, 0 elsewhere,
* ``median_stroke_width`` and the dominant stroke ``color_rgb``.

Two ink models are used:

* **Background-relative** (LINE_ART, and FLAT_COLOR when centerline mode is requested): every
  pixel that differs enough from the dominant background colour is ink. This keeps coloured
  strokes (e.g. blue arcs) as well as black ones. A pixel is kept when its distance from the
  background is at least half the maximum distance in its 3x3 neighbourhood, i.e. >= 50 %
  coverage of the adjacent ink.
* **Ink-relative** (MIXED): only pixels close to the darkest ink colour form stroke cores;
  anti-aliased fringe pixels next to a core are added when they are >= 50 % ink with respect to
  the brightest adjacent (fill) colour. Fills of any colour are therefore never strokes.

In every mode, regions wider than a maximum stroke width are removed (distance-transform
width filter), so solid blobs never count as strokes. Gaps of up to 3 source px between a
stroke end and another stroke are bridged, small specks are dropped, and transparent pixels
(``pre.opaque_mask`` false) are never part of a stroke.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

from contracts.schemas import (
    ImageClass,
    ImageClassLabel,
    LineMap,
    PreprocessResult,
    Settings,
    StageError,
)

STAGE = "lines"

MAX_STROKE_WIDTH_MIXED_SRC = 10.0
"""MIXED: dark regions wider than this (source px) are fills, not outline strokes."""
MAX_STROKE_WIDTH_LINE_ART_SRC = 32.0
"""LINE_ART: regions wider than this (source px) are solid fills, not strokes."""
GAP_CLOSE_SRC = 3.0
"""Maximum gap (source px) bridged between a stroke end and another stroke."""
SPUR_FACTOR = 1.5
"""Skeleton branches shorter than SPUR_FACTOR x local stroke width are pruned."""
INK_CORE_DIST = 60.0
"""MIXED: max RGB distance from the ink colour for a pixel to be a stroke core."""
DARK_L_MAX = 40.0
"""MIXED: L* above which a pixel is never a dark-ink candidate."""
TRANSPARENT_BG_FRACTION = 0.25
"""LINE_ART: if at least this fraction is transparent, the opaque pixels are the ink."""

_OFFSETS: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_KERNEL3 = np.ones((3, 3), np.uint8)


# --------------------------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------------------------


def extract_lines(pre: PreprocessResult, image_class: ImageClass, settings: Settings) -> LineMap:
    """Extract the stroke mask, centerline skeleton and stroke widths of ``pre``.

    Args:
        pre: Preprocessed pixels in processing space.
        image_class: Detected class; MIXED selects the dark-outline ink model.
        settings: User settings (``detail_level`` controls speck removal).

    Returns:
        A validated :class:`LineMap` in processing space.

    Raises:
        StageError: if an unexpected numerical failure occurs.
    """
    try:
        return _extract(pre, image_class, settings)
    except StageError:
        raise
    except (ValueError, cv2.error) as exc:  # pragma: no cover - defensive wrapper
        raise StageError(STAGE, f"line extraction failed: {exc}") from exc


def _extract(pre: PreprocessResult, image_class: ImageClass, settings: Settings) -> LineMap:
    rgb = np.asarray(pre.image)
    opaque = np.asarray(pre.opaque_mask, dtype=bool)
    scale = float(pre.scale_factor)
    unit = max(1.0, scale)

    if image_class.label == ImageClassLabel.MIXED:
        mask = mixed_stroke_mask(rgb, opaque)
        max_width = MAX_STROKE_WIDTH_MIXED_SRC * scale
    else:
        mask = line_art_stroke_mask(rgb, opaque)
        max_width = MAX_STROKE_WIDTH_LINE_ART_SRC * scale

    mask = remove_wide_regions(mask, max_width)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, _KERNEL3).astype(bool) & opaque
    mask = remove_specks(mask, settings.preset.speckle_min_area_px)

    skeleton = skeletonize_mask(mask)
    dist = distance_to_edge(mask)
    gap = int(math.ceil(GAP_CLOSE_SRC * unit))
    bridged, n_bridges = bridge_gaps(mask, skeleton, dist, gap)
    if n_bridges:
        mask = bridged & opaque
        skeleton = skeletonize_mask(mask)
        dist = distance_to_edge(mask)

    width_full = stroke_width_from_distance(dist)
    skeleton = prune_spurs(skeleton, width_full, SPUR_FACTOR)
    skeleton = remove_redundant_pixels(skeleton) & mask

    width_map = np.where(skeleton, width_full, np.float32(0.0)).astype(np.float32)
    widths = width_map[skeleton]
    median_width = float(np.median(widths)) if widths.size else 0.0
    return LineMap(
        mask=mask,
        skeleton=skeleton,
        width_map=width_map,
        median_stroke_width=median_width,
        color_rgb=stroke_color(rgb, mask),
    )


# --------------------------------------------------------------------------------------
# Stroke masks
# --------------------------------------------------------------------------------------


def _max3(values: np.ndarray) -> np.ndarray:
    """3x3 maximum filter (float32)."""
    return cv2.dilate(values, _KERNEL3, borderType=cv2.BORDER_REPLICATE)


def _rgb_distance(rgb: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Euclidean sRGB distance of every pixel from ``color`` as float32 (H, W)."""
    diff = rgb.astype(np.float32) - color.astype(np.float32)
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff, dtype=np.float32))


def dominant_color(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Most frequent colour (5-bit bins, refined by the bin's median) among ``valid`` pixels."""
    px = rgb[valid]
    if px.shape[0] == 0:
        return np.array([255, 255, 255], np.uint8)
    q = (px >> 3).astype(np.int32)
    codes = (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
    top = int(np.argmax(np.bincount(codes, minlength=1 << 15)))
    return np.median(px[codes == top], axis=0).astype(np.uint8)


def line_art_stroke_mask(rgb: np.ndarray, opaque: np.ndarray) -> np.ndarray:
    """Background-relative ink mask for line art.

    Ink = pixels far enough from the dominant background colour, thresholded at 50 % of the
    local (3x3) maximum contrast so anti-aliased edges resolve to the nominal stroke width.
    """
    if opaque.mean() <= 1.0 - TRANSPARENT_BG_FRACTION:
        # Transparent background: the opaque artwork itself is the ink.
        return opaque.copy()
    bg = dominant_color(rgb, opaque)
    s = _rgb_distance(rgb, bg)
    s[~opaque] = 0.0
    fg = s[s > 20.0]
    if fg.size == 0:
        return np.zeros(opaque.shape, bool)
    s_hi = float(np.percentile(fg, 90))
    t_min = float(np.clip(0.35 * s_hi, 30.0, 120.0))
    return (s >= t_min) & (s >= 0.5 * _max3(s)) & opaque


def estimate_ink_color(rgb: np.ndarray, opaque: np.ndarray) -> np.ndarray | None:
    """Colour of the darkest ink (median of the darkest pixels), or None if nothing is dark."""
    lab_l = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[..., 0].astype(np.float32) * (100.0 / 255.0)
    vals = lab_l[opaque]
    if vals.size == 0:
        return None
    sample = vals[:: max(1, vals.size // 200_000)]
    thr = DARK_L_MAX
    if float(sample.min()) < float(sample.max()):
        thr = min(float(threshold_otsu(sample)), DARK_L_MAX)
    dark = opaque & (lab_l < thr)
    if int(dark.sum()) < 4:
        return None
    l0 = float(np.percentile(lab_l[dark], 5))
    core = dark & (lab_l <= l0 + 10.0)
    return np.median(rgb[core], axis=0).astype(np.uint8)


def mixed_stroke_mask(rgb: np.ndarray, opaque: np.ndarray) -> np.ndarray:
    """Ink-relative mask: only uniformly dark stroke pixels (plus >= 50 % AA fringe)."""
    ink = estimate_ink_color(rgb, opaque)
    if ink is None:
        return np.zeros(opaque.shape, bool)
    t = _rgb_distance(rgb, ink)
    t[~opaque] = 0.0
    core = (t <= INK_CORE_DIST) & opaque
    near_core = cv2.dilate(core.astype(np.uint8), _KERNEL3).astype(bool)
    fringe = near_core & (t <= 0.5 * _max3(t))
    return (core | fringe) & opaque


def remove_wide_regions(mask: np.ndarray, max_width: float) -> np.ndarray:
    """Distance-transform width filter: drop every part of ``mask`` wider than ``max_width``.

    Pixels deeper than ``max_width / 2`` seed a "fat" region, which is grown back to the
    region's full extent (a morphological opening), then removed from the mask.
    """
    dist = distance_to_edge(mask)
    half = max_width / 2.0
    seeds = dist > half
    if not seeds.any():
        return mask
    r = int(math.ceil(half)) + 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    fat = cv2.dilate(seeds.astype(np.uint8), disk).astype(bool)
    return mask & ~fat


def remove_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove 8-connected components smaller than ``min_area`` pixels (at least 2)."""
    min_area = max(2, int(min_area))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels]


def distance_to_edge(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (float32) from each mask pixel to the nearest non-mask pixel.

    The image border counts as an edge, so strokes clipped by the border are not inflated.
    """
    padded = np.pad(mask.astype(np.uint8), 1)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dist[1:-1, 1:-1]


def stroke_width_from_distance(dist: np.ndarray) -> np.ndarray:
    """Local stroke width = ``2 x DT`` of the mask (float32, 0 outside the mask).

    ``DT`` is measured between pixel centres, so this is exact for even widths and +1 px for
    odd ones; it matches the nominal width of aliased (Pillow/Bresenham) lines, which render
    thinner than their nominal width.
    """
    return (2.0 * dist).astype(np.float32)


def stroke_color(rgb: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    """Per-channel median colour of the mask pixels; black if the mask is empty."""
    px = rgb[mask]
    if px.shape[0] == 0:
        return (0, 0, 0)
    med = np.median(px, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


# --------------------------------------------------------------------------------------
# Skeleton
# --------------------------------------------------------------------------------------


def _bbox(mask: np.ndarray) -> tuple[slice, slice] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """``skimage.morphology.skeletonize`` restricted to the mask's bounding box, made 1-px thin."""
    out = np.zeros(mask.shape, bool)
    box = _bbox(mask)
    if box is None:
        return out
    out[box] = skeletonize(mask[box])
    return remove_redundant_pixels(out)


def neighbor_count(skel: np.ndarray) -> np.ndarray:
    """Number of 8-neighbours set for every pixel (uint8)."""
    s = skel.astype(np.uint8)
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    return cv2.filter2D(s, -1, k, borderType=cv2.BORDER_CONSTANT)


def _build_redundant_lut() -> np.ndarray:
    """LUT over the 8-neighbour code: True where the centre is a redundant staircase pixel.

    Bit order follows ``_OFFSETS`` (NW, N, NE, W, E, SW, S, SE). A pixel is redundant when it
    is a simple point (8-connectivity number 1), it has >= 2 neighbours, and at least two of
    them are 4-neighbours (an L-corner that the diagonal already connects).
    """
    lut = np.zeros(256, bool)
    ring = (0, 1, 2, 4, 7, 6, 5, 3)  # NW, N, NE, E, SE, S, SW, W in circular order
    for code in range(256):
        bits = [(code >> i) & 1 for i in range(8)]
        x = [bits[i] for i in ring]  # x[0]=NW, x[1]=N, x[2]=NE, x[3]=E, ...
        xb = [1 - v for v in x]
        # 8-connectivity number (Yokoi): sum over 4-neighbours k of xb_k - xb_k*xb_k+1*xb_k+2
        c8 = sum(xb[k] - xb[k] * xb[(k + 1) % 8] * xb[(k + 2) % 8] for k in (1, 3, 5, 7))
        n4 = x[1] + x[3] + x[5] + x[7]
        lut[code] = c8 == 1 and sum(x) >= 2 and n4 >= 2
    return lut


_REDUNDANT_LUT = _build_redundant_lut()


def remove_redundant_pixels(skel: np.ndarray) -> np.ndarray:
    """Delete staircase pixels so the skeleton is strictly 1 px wide, keeping 8-connectivity.

    Pixels are processed in four parity classes; pixels of one class are never 8-adjacent, so
    removing a whole class at once cannot break connectivity.
    """
    out = np.zeros(skel.shape, bool)
    box = _bbox(skel)
    if box is None:
        return out
    padded = np.pad(skel[box], 1)
    changed = True
    while changed:
        changed = False
        ys_all, xs_all = np.nonzero(padded)
        parity = (ys_all & 1) * 2 + (xs_all & 1)
        for cls in range(4):
            sel = parity == cls
            ys, xs = ys_all[sel], xs_all[sel]
            if ys.size == 0:
                continue
            code = np.zeros(ys.size, np.int32)
            for bit, (dy, dx) in enumerate(_OFFSETS):
                code |= padded[ys + dy, xs + dx].astype(np.int32) << bit
            drop = _REDUNDANT_LUT[code]
            if drop.any():
                padded[ys[drop], xs[drop]] = False
                changed = True
    out[box] = padded[1:-1, 1:-1]
    return out


def _neighbors(sk: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    return [(y + dy, x + dx) for dy, dx in _OFFSETS if sk[y + dy, x + dx]]


def _one_group(points: list[tuple[int, int]]) -> bool:
    """True if the given pixels form a single 8-connected group."""
    group = [points[0]]
    rest = points[1:]
    grew = True
    while rest and grew:
        grew = False
        for q in list(rest):
            if any(abs(q[0] - g[0]) <= 1 and abs(q[1] - g[1]) <= 1 for g in group):
                group.append(q)
                rest.remove(q)
                grew = True
    return not rest


def prune_spurs(skel: np.ndarray, width: np.ndarray, factor: float) -> np.ndarray:
    """Remove branches that run from an endpoint to a junction and are shorter than
    ``factor`` x the stroke width at that junction. Isolated segments are kept."""
    if not skel.any():
        return skel.copy()
    sk = np.pad(skel, 1)
    wd = np.pad(width, 1)
    max_len = int(math.ceil(factor * float(width[skel].max()))) + 1
    for _ in range(2):
        ends = np.argwhere(sk & (neighbor_count(sk) == 1))
        removed = False
        for y0, x0 in ends:
            y, x = int(y0), int(x0)
            if not sk[y, x]:
                continue
            path = [(y, x)]
            seen = {(y, x)}
            junction: tuple[int, int] | None = None
            while len(path) <= max_len:
                nxt = [p for p in _neighbors(sk, y, x) if p not in seen]
                if not nxt:
                    break  # reached the other end of an isolated segment
                if len(nxt) == 1:
                    y, x = nxt[0]
                    path.append((y, x))
                    seen.add((y, x))
                    continue
                # Branch point. If the onward pixels touch each other, (y, x) merely sits beside
                # the other stroke and belongs to the spur; otherwise (y, x) is the junction.
                junction = max(nxt, key=lambda q: float(wd[q])) if _one_group(nxt) else path.pop()
                break
            if junction is None:
                continue
            if len(path) < factor * float(wd[junction]):
                for py, px in path:
                    sk[py, px] = False
                removed = True
        if not removed:
            break
    return sk[1:-1, 1:-1].copy()


def endpoints(skel: np.ndarray) -> np.ndarray:
    """(N, 2) array of (y, x) skeleton endpoints (exactly one 8-neighbour)."""
    return np.argwhere(skel & (neighbor_count(skel) == 1))


def bridge_gaps(mask: np.ndarray, skel: np.ndarray, dist: np.ndarray, gap: int) -> tuple[np.ndarray, int]:
    """Bridge gaps of up to ``gap`` px between a stroke end and a *different* stroke.

    For every skeleton endpoint, the nearest mask pixel of another mask component is searched
    within ``gap`` px of the stroke's end cap (the cap extends ~DT beyond the skeleton end).
    Bridges are drawn into the mask with the stroke's local width, so re-skeletonizing
    connects the strokes. Returns the (possibly) extended mask and the number of bridges.
    """
    ends = endpoints(skel)
    if ends.shape[0] == 0:
        return mask, 0
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask, 0
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    h, w = mask.shape
    out = mask.astype(np.uint8)
    count = 0
    for y0, x0 in ends:
        y, x = int(y0), int(x0)
        own = int(labels[y, x])
        radius = gap + int(math.ceil(float(dist[y, x])))
        ya, yb, xa, xb = max(0, y - radius), min(h, y + radius + 1), max(0, x - radius), min(w, x + radius + 1)
        win = labels[ya:yb, xa:xb]
        cand = np.argwhere((win > 0) & (win != own))
        if cand.shape[0] == 0:
            continue
        d2 = (cand[:, 0] + ya - y) ** 2 + (cand[:, 1] + xa - x) ** 2
        best = int(np.argmin(d2))
        if d2[best] > radius * radius:
            continue
        ty, tx = int(cand[best, 0] + ya), int(cand[best, 1] + xa)
        a, b = find(own), find(int(labels[ty, tx]))
        if a == b:
            continue
        parent[a] = b
        thickness = max(1, int(round(2.0 * float(dist[y, x]))))
        cv2.line(out, (x, y), (tx, ty), 1, thickness=thickness)
        count += 1
    return out.astype(bool), count
