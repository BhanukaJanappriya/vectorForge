"""Stage 2: classify an image as LINE_ART, FLAT_COLOR or MIXED.

Features are computed on a working copy at roughly *source* resolution (processing space
divided by ``scale_factor``), capped at ``WORK_MAX_SIDE`` px, so thresholds are independent
of preprocessing's up/downscaling and runtime stays well below 150 ms at 2 MP.
If preprocess removed a background (``background_removed``), the recorded background color
(``PreprocessResult.background_lab``) is composited back first, so the label
describes the artwork and does not depend on the remove_background setting.

Features (all fractions are relative to opaque pixels):

* ``n_colors`` / ``colors_95``: coarse LAB histogram (10 L x 12 a x 12 b units) -- number of
  bins holding >= 0.2 % of pixels / needed to cover 95 % of pixels.
* ``gradient_area``: share of 8x8-px cells, away from edges, whose area-averaged LAB color
  changes by > 0.5 per cell -- smooth, large-scale color ramps (gradients / continuous tone).
  Robust to JPEG residue, which averages out within a cell.
* ``smooth_ratio`` / ``flat_ratio`` (diagnostic): pixels with a small non-zero / ~zero
  per-pixel luminance gradient.
* ``edge_density``: Canny edge pixels. ``edge_per_ink``: edges per non-background pixel
  (high for thin strokes).
* ``dark_ratio``: pixels with L* < 35.
* ``bg_share``: share of the dominant color (the background; transparency if the image has
  a substantial transparent area). ``ink_ratio``: non-background pixels / *all* pixels.
* ``fill_ratio``: non-background, non-dark pixels surviving an 11x11 opening (solid fills).
* ``thin_ink_ratio``: share of ink removed by the same opening (stroke thinness).
* ``ink_extent``: area-weighted bounding-box diagonal of ink components / image diagonal.
  Line drawings consist of long connected strokes; text consists of many small glyphs.
* ``outline_ratio`` / ``outline_fraction``: thin dark pixels next to a solid fill, relative to
  thin dark pixels / to the image (black outlines around colored regions => MIXED).

Decision model (factorized logistic; each score is a sigmoid of one or more features):

    g = gradient score (gradient_area, colors_95), l = line-art score, o = outline score
    P(MIXED)      = g + (1 - g) * (1 - l) * o
    P(LINE_ART)   = (1 - g) * l
    P(FLAT_COLOR) = (1 - g) * (1 - l) * (1 - o)

The label is the argmax and ``confidence`` its probability. The sigmoid centres were placed
between the per-class feature values measured on samples/ (all 10 samples are classified
with confidence >= 0.94), so the probabilities fall off smoothly near decision boundaries.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from contracts.schemas import ImageClass, ImageClassLabel, PreprocessResult, ProcessingMode, Settings

WORK_MAX_SIDE = 768
"""Long-side cap of the working copy used for feature extraction."""
OPEN_KERNEL = 11
"""Structures thinner than this (working px) count as strokes, thicker as fills."""
DARK_L = 35.0
BG_DELTA_E = 12.0
GRADIENT_CELL = 8
"""Cell size (working px) for the large-scale gradient measurement."""
GRADIENT_MIN_SLOPE = 0.5
"""Minimum LAB change per cell for a cell to count as part of a color ramp."""
TRANSPARENT_BG_SHARE = 0.2
"""If at least this fraction of pixels is transparent, transparency is the background."""

_BIN_LUT_L = (np.minimum(np.arange(256) * 100.0 / 255.0 / 10.0, 10).astype(np.int32) * 484).astype(np.int32)
_BIN_LUT_AB = (np.arange(256) // 12).astype(np.int32)

_FORCED: dict[ProcessingMode, ImageClassLabel] = {
    ProcessingMode.LINE_ART: ImageClassLabel.LINE_ART,
    ProcessingMode.FLAT_COLOR: ImageClassLabel.FLAT_COLOR,
    ProcessingMode.MIXED: ImageClassLabel.MIXED,
}


def _sigmoid(x: float) -> float:
    """Numerically safe logistic function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _removed_background_rgb(pre: PreprocessResult) -> np.ndarray | None:
    """sRGB of the background preprocess removed (PreprocessResult.background_lab), or None."""
    if not pre.background_removed or pre.background_lab is None:
        return None
    lab = np.array([[pre.background_lab]], dtype=np.float32)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[0, 0]
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _working_copy(pre: PreprocessResult) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb, opaque mask) resampled to ~source resolution, long side <= WORK_MAX_SIDE."""
    img = np.asarray(pre.image)
    h, w = img.shape[:2]
    scale = min(1.0 / pre.scale_factor, WORK_MAX_SIDE / max(h, w))
    opaque = np.asarray(pre.opaque_mask)
    bg_rgb = _removed_background_rgb(pre)
    if bg_rgb is not None:
        # Classify the content, not the user's remove_background choice: put the removed
        # background color back so enclosed regions of that color remain "background".
        img = img.copy()
        img[~opaque] = bg_rgb
        opaque = np.ones_like(opaque)
    if scale < 1.0:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        opaque = cv2.resize(opaque.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    return img, opaque


def _open(mask: np.ndarray, size: int = OPEN_KERNEL) -> np.ndarray:
    """Morphological opening with a size x size square: keeps only structures >= size wide."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)


def _gradient_area(lab: np.ndarray, gray: np.ndarray, opaque: np.ndarray) -> float:
    """Fraction of the image covered by smooth, large-scale color ramps (gradients).

    LAB is area-averaged over GRADIENT_CELL x GRADIENT_CELL cells, which averages residual
    noise / JPEG ringing away while a ramp's per-cell slope grows. Cells touching an edge
    (Canny) or transparency are ignored, so flat art -- even heavily compressed -- scores ~0.
    """
    h, w = gray.shape
    rows, cols = h // GRADIENT_CELL, w // GRADIENT_CELL
    if rows < 3 or cols < 3:
        return 0.0
    crop = (slice(0, rows * GRADIENT_CELL), slice(0, cols * GRADIENT_CELL))
    coarse = cv2.resize(lab[crop], (cols, rows), interpolation=cv2.INTER_AREA)
    blocked = (cv2.Canny(gray, 30, 90) > 0) | ~opaque
    blocked_cells = cv2.resize(blocked[crop].astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA) > 0
    blocked_cells = cv2.dilate(blocked_cells.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    slope2 = np.zeros((rows, cols), np.float32)
    for channel in cv2.split(coarse):
        gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0) / 8.0
        gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1) / 8.0
        slope2 += gx * gx + gy * gy
    ramp = (slope2 > GRADIENT_MIN_SLOPE**2) & ~blocked_cells
    return float(ramp.sum()) / (rows * cols)


def extract_features(pre: PreprocessResult) -> dict[str, float]:
    """Compute the diagnostic features used by :func:`classify` (deterministic)."""
    img, opaque = _working_copy(pre)
    n_opaque = int(opaque.sum())
    features: dict[str, float] = {"opaque_fraction": n_opaque / opaque.size}
    if n_opaque < 16:
        return features

    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    lum = lab[..., 0]

    # Coarse LAB histogram (bincount is much faster than np.unique on ~1 MP).
    # 8-bit LAB: L8 = L*·255/100, a8 = a*+128, b8 = b*+128; LUTs map straight to bin offsets.
    lab8 = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    bins = _BIN_LUT_L[lab8[0]] + _BIN_LUT_AB[lab8[1]] * 22 + _BIN_LUT_AB[lab8[2]]
    counts = np.bincount(bins[opaque], minlength=11 * 22 * 22)
    shares = np.sort(counts)[::-1] / n_opaque
    features["n_colors"] = float(np.count_nonzero(shares >= 0.002))
    features["colors_95"] = float(min(int(np.searchsorted(np.cumsum(shares), 0.95)) + 1, len(shares)))

    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = cv2.magnitude(gx, gy)[opaque]
    features["smooth_ratio"] = float(np.mean((grad > 0.15) & (grad < 3.0)))
    features["flat_ratio"] = float(np.mean(grad <= 0.15))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features["gradient_area"] = _gradient_area(lab, gray, opaque)
    edges = cv2.Canny(gray, 50, 150) > 0
    n_edges = int(np.count_nonzero(edges & opaque))
    features["edge_density"] = n_edges / n_opaque

    dark = (lum < DARK_L) & opaque
    features["dark_ratio"] = float(dark.sum()) / n_opaque

    transparent_share = 1.0 - n_opaque / opaque.size
    if transparent_share >= TRANSPARENT_BG_SHARE:
        background = np.zeros_like(opaque)
        features["bg_share"] = 0.0
    else:
        dominant = int(np.argmax(counts))
        members = lab[(bins == dominant) & opaque]
        bg_lab = np.median(members[:: max(1, len(members) // 20000)], axis=0)
        diff = cv2.subtract(lab, tuple(float(v) for v in bg_lab) + (0.0,))
        dist2 = cv2.transform(cv2.multiply(diff, diff), np.ones((1, 3), np.float32))
        background = (dist2 < BG_DELTA_E**2) & opaque
        features["bg_share"] = float(background.sum()) / n_opaque
    features["transparent_bg"] = float(transparent_share >= TRANSPARENT_BG_SHARE)

    ink = opaque & ~background
    n_ink = int(ink.sum())
    features["ink_ratio"] = n_ink / opaque.size  # of the canvas, so transparent backgrounds count as empty
    features["edge_per_ink"] = n_edges / max(n_ink, 1)

    fill_core = _open(ink & ~dark)
    features["fill_ratio"] = float(fill_core.sum()) / n_opaque
    features["thin_ink_ratio"] = 1.0 - float((_open(ink) & ink).sum()) / max(n_ink, 1)

    dark_thin = dark & ~_open(dark)
    near_fill = cv2.dilate(fill_core.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    outline = int((dark_thin & near_fill).sum())
    features["outline_ratio"] = outline / max(int(dark_thin.sum()), 1)
    features["outline_fraction"] = outline / n_opaque

    n_comp, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    if n_comp > 1:
        area = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        diag = np.hypot(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) / math.hypot(*img.shape[:2])
        features["ink_extent"] = float((area * diag).sum() / area.sum())
    else:
        features["ink_extent"] = 0.0
    features["ink_components"] = float(n_comp - 1)
    return {k: round(v, 5) for k, v in features.items()}


def score(features: dict[str, float]) -> dict[ImageClassLabel, float]:
    """Map features to class probabilities (see module docstring)."""
    if "gradient_area" not in features:  # (almost) nothing opaque
        return {ImageClassLabel.FLAT_COLOR: 0.5, ImageClassLabel.LINE_ART: 0.25, ImageClassLabel.MIXED: 0.25}
    gradient = max(
        _sigmoid((features["gradient_area"] - 0.25) / 0.05),
        _sigmoid((features["colors_95"] - 20.0) / 3.0),
    )
    thinness = math.sqrt(
        _sigmoid((features["thin_ink_ratio"] - 0.8) / 0.05) * _sigmoid((features["edge_per_ink"] - 0.15) / 0.04)
    )
    line = (
        _sigmoid((0.03 - features["fill_ratio"]) / 0.008)
        * thinness
        * _sigmoid((features["ink_extent"] - 0.25) / 0.05)
        * _sigmoid((0.3 - features["ink_ratio"]) / 0.05)
    )
    outline = _sigmoid((features["outline_fraction"] - 0.008) / 0.002) * _sigmoid(
        (features["outline_ratio"] - 0.35) / 0.07
    )
    return {
        ImageClassLabel.MIXED: gradient + (1 - gradient) * (1 - line) * outline,
        ImageClassLabel.LINE_ART: (1 - gradient) * line,
        ImageClassLabel.FLAT_COLOR: (1 - gradient) * (1 - line) * (1 - outline),
    }


def classify(pre: PreprocessResult, settings: Settings) -> ImageClass:
    """Detect LINE_ART / FLAT_COLOR / MIXED. If settings.mode != AUTO, return that
    label with confidence=1.0 and forced=True."""
    forced = _FORCED.get(settings.mode)
    if forced is not None:
        return ImageClass(label=forced, confidence=1.0, forced=True, features={})
    features = extract_features(pre)
    probs = score(features)
    label = max(probs, key=lambda k: (probs[k], k == ImageClassLabel.FLAT_COLOR))
    for key, value in probs.items():
        features[f"p_{key.value}"] = round(value, 5)
    return ImageClass(label=label, confidence=round(min(max(probs[label], 0.0), 1.0), 4), features=features)
