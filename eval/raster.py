"""Pure-NumPy path utilities for QA: path parsing, node counting and a reference rasterizer.

The rasterizer exists because the QA harness must not depend on a native SVG renderer
(CairoSVG needs the Cairo DLL). It renders a :class:`VectorDocument` directly from its
absolute ``M/L/C/Z`` paths with an exact scanline algorithm:

* fills: pixel-center sampling with ``nonzero`` / ``evenodd`` winding, optionally
  supersampled (``ss`` x ``ss`` samples per pixel) for anti-aliasing;
* strokes: OpenCV polylines on the supersampled grid;
* layers are composited bottom to top with premultiplied "over".

Documents whose coordinates are all integers (e.g. the pixel-run oracle) are rendered at
``ss = 1`` which is exact: every pixel is either fully covered or not, so adjacent
rectangles never produce anti-aliasing seams.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import cv2
import numpy as np

from contracts.schemas import VectorDocument, VectorLayer

TOKEN_RE = re.compile(r"[MLCZ]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_ARITY = {"M": 2, "L": 2, "C": 6}


@dataclass(frozen=True)
class Subpath:
    """A flattened subpath in source space."""

    points: np.ndarray
    """(N, 2) float64 x, y."""
    closed: bool


def _tokens(d: str) -> list[str]:
    tokens = TOKEN_RE.findall(d)
    if not tokens or tokens[0] != "M":
        raise ValueError(f"path must start with M: {d[:40]!r}")
    return tokens


def node_count(d: str) -> int:
    """Number of M/L/C segment endpoints in an absolute path (Z is not counted).

    Implicit repeats (``L 1 2 3 4``) count one node per coordinate group.
    """
    count = 0
    cmd = ""
    numbers = 0
    for tok in _tokens(d):
        if tok.isalpha():
            if cmd in _ARITY:
                count += numbers // _ARITY[cmd]
            cmd, numbers = tok, 0
        else:
            numbers += 1
    if cmd in _ARITY:
        count += numbers // _ARITY[cmd]
    return count


def document_node_count(doc: VectorDocument) -> int:
    """Total node count over all layers of a document."""
    return sum(node_count(d) for layer in doc.layers for d in layer.paths)


def _bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Flatten a cubic Bezier to points (excluding p0), adaptive to control-polygon length."""
    length = float(np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2))
    n = int(min(64, max(2, math.ceil(length / 1.5))))
    t = np.linspace(0.0, 1.0, n + 1)[1:, None]
    mt = 1.0 - t
    return mt**3 * p0 + 3 * mt**2 * t * p1 + 3 * mt * t**2 * p2 + t**3 * p3


def parse_path(d: str) -> list[Subpath]:
    """Parse an absolute M/L/C/Z path into flattened subpaths."""
    subpaths: list[Subpath] = []
    current: list[tuple[float, float]] = []

    def finish(closed: bool) -> None:
        if len(current) > 1:
            subpaths.append(Subpath(np.asarray(current, dtype=np.float64), closed=closed))

    cmd = ""
    nums: list[float] = []

    def flush() -> None:
        nonlocal current
        if cmd == "M" and len(nums) >= 2:
            finish(False)
            current = [(nums[0], nums[1])]
            current.extend(zip(nums[2::2], nums[3::2], strict=False))  # implicit lineto
        elif cmd == "L":
            current.extend(zip(nums[0::2], nums[1::2], strict=False))
        elif cmd == "C":
            for i in range(0, len(nums) - 5, 6):
                p0 = np.array(current[-1], dtype=np.float64)
                c = np.array(nums[i : i + 6], dtype=np.float64).reshape(3, 2)
                current.extend(map(tuple, _bezier(p0, c[0], c[1], c[2]).tolist()))

    for tok in _tokens(d):
        if tok.isalpha():
            flush()
            nums = []
            cmd = tok
            if cmd == "Z" and current:
                finish(True)
                current = [current[0]]  # a following L/C continues from the subpath start
        else:
            nums.append(float(tok))
    flush()
    finish(False)
    return subpaths


def _all_integer(layers: list[VectorLayer]) -> bool:
    for layer in layers:
        for d in layer.paths:
            for tok in TOKEN_RE.findall(d):
                if not tok.isalpha() and float(tok) != int(float(tok)):
                    return False
    return True


def choose_supersampling(doc: VectorDocument) -> int:
    """1 for integer-aligned documents (exact), else 3 (or 2 for large images)."""
    if _all_integer(doc.layers) and not any(layer.is_stroke for layer in doc.layers):
        return 1
    return 3 if doc.width * doc.height <= 1_500_000 else 2


def fill_coverage(subpaths: list[Subpath], width: int, height: int, rule: str = "evenodd", ss: int = 1) -> np.ndarray:
    """Per-pixel coverage in [0, 1] of the filled subpaths (all implicitly closed)."""
    w2, h2 = width * ss, height * ss
    polys = [sp.points for sp in subpaths if sp.points.shape[0] >= 2]
    if not polys:
        return np.zeros((height, width), dtype=np.float32)
    lengths = np.array([p.shape[0] for p in polys])
    pts = np.vstack(polys) * ss
    nxt_idx = np.arange(pts.shape[0]) + 1
    ends = np.cumsum(lengths)
    nxt_idx[ends - 1] = ends - lengths  # close each subpath back to its first point
    x0, y0 = pts[:, 0], pts[:, 1]
    x1, y1 = pts[nxt_idx, 0], pts[nxt_idx, 1]
    keep = y0 != y1
    x0, y0, x1, y1 = x0[keep], y0[keep], x1[keep], y1[keep]
    direction = np.where(y1 > y0, 1, -1).astype(np.int64)
    ymin, ymax = np.minimum(y0, y1), np.maximum(y0, y1)
    # Sample rows at centers r + 0.5 with ymin <= r + 0.5 < ymax.
    r_start = np.clip(np.ceil(ymin - 0.5), 0, h2).astype(np.int64)
    r_end = np.clip(np.ceil(ymax - 0.5), 0, h2).astype(np.int64)
    counts = np.maximum(r_end - r_start, 0)
    total = int(counts.sum())
    acc_w = w2 + 1
    if total == 0:
        return np.zeros((height, width), dtype=np.float32)
    edge = np.repeat(np.arange(counts.size), counts)
    offsets = np.cumsum(counts) - counts
    rows = r_start[edge] + (np.arange(total) - offsets[edge])
    yc = rows + 0.5
    xc = x0[edge] + (yc - y0[edge]) * (x1[edge] - x0[edge]) / (y1[edge] - y0[edge])
    cols = np.clip(np.ceil(xc - 0.5), 0, w2).astype(np.int64)
    weights = direction[edge] if rule == "nonzero" else np.ones(total, dtype=np.int64)
    acc = np.bincount(rows * acc_w + cols, weights=weights, minlength=h2 * acc_w)
    winding = np.cumsum(acc.reshape(h2, acc_w)[:, :w2], axis=1)
    inside = (winding != 0) if rule == "nonzero" else (winding.astype(np.int64) % 2 == 1)
    if ss == 1:
        return inside.astype(np.float32)
    return inside.reshape(height, ss, width, ss).mean(axis=(1, 3), dtype=np.float32)


def stroke_coverage(subpaths: list[Subpath], width: int, height: int, stroke_width: float, ss: int = 2) -> np.ndarray:
    """Per-pixel coverage in [0, 1] of stroked subpaths (round-ish joins, OpenCV polylines)."""
    shift = 4
    canvas = np.zeros((height * ss, width * ss), dtype=np.uint8)
    thickness = max(1, round(stroke_width * ss))
    polys = []
    closed_flags = []
    for sp in subpaths:
        pts = np.round((sp.points * ss - 0.5) * (1 << shift)).astype(np.int32)
        polys.append(pts.reshape(-1, 1, 2))
        closed_flags.append(sp.closed)
    for pts, closed in zip(polys, closed_flags, strict=True):
        cv2.polylines(canvas, [pts], closed, 255, thickness=thickness, lineType=cv2.LINE_8, shift=shift)
    cov = canvas.astype(np.float32) / 255.0
    if ss == 1:
        return cov
    return cov.reshape(height, ss, width, ss).mean(axis=(1, 3), dtype=np.float32)


def hex_to_rgb(color_hex: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (r, g, b)."""
    return int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)


def layer_coverage(layer: VectorLayer, width: int, height: int, ss: int) -> np.ndarray:
    """Coverage of one layer at source resolution."""
    subpaths = [sp for d in layer.paths for sp in parse_path(d)]
    if layer.is_stroke:
        return stroke_coverage(subpaths, width, height, float(layer.stroke_width or 1.0), ss=max(ss, 2))
    return fill_coverage(subpaths, width, height, rule=layer.fill_rule, ss=ss)


def render_document(doc: VectorDocument, ss: int | None = None) -> np.ndarray:
    """Render a VectorDocument to an (H, W, 4) uint8 straight-alpha RGBA image over transparency."""
    ss = choose_supersampling(doc) if ss is None else ss
    h, w = doc.height, doc.width
    premult = np.zeros((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)
    for layer in doc.layers:
        cov = layer_coverage(layer, w, h, ss)
        rows = np.flatnonzero(cov.any(axis=1))
        if rows.size == 0:
            continue
        cols = np.flatnonzero(cov.any(axis=0))
        box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        a = cov[box] * np.float32(layer.opacity)
        color = np.array(hex_to_rgb(layer.color_hex), dtype=np.float32) / 255.0
        pm, al = premult[box], alpha[box]
        if layer.opacity == 1.0 and ss == 1 and not layer.is_stroke:  # binary coverage: plain assignment
            hit = a > 0
            pm[hit] = color
            al[hit] = 1.0
            continue
        inv = 1.0 - a
        pm *= inv[..., None]
        pm += color * a[..., None]
        al *= inv
        al += a
    out = np.empty((h, w, 4), dtype=np.uint8)
    safe = np.maximum(alpha, 1e-6)[..., None]
    np.divide(premult, safe, out=premult)
    premult[alpha == 0] = 0.0
    out[..., :3] = np.clip(np.rint(premult * 255.0), 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    return out
