"""Color quantization stage: opaque pixels -> a small LAB palette + per-pixel label map.

Algorithm (auto mode)
---------------------
1. Collapse the opaque pixels to their unique sRGB colors (per-pixel work becomes a table
   lookup) and convert those to CIELAB (D65, 2 deg). Transparent pixels are ignored
   entirely and get label -1.
2. Flag *mixed* pixels: pixels whose color lies (in sRGB, where anti-aliasing blends) on the
   segment between two opposite neighbours that differ strongly. These are anti-aliased edge
   pixels; they are excluded from palette fitting and from the final median colors.
3. Fit a weighted k-means on the remaining ("pure") pixels, collapsed onto a LAB grid (the
   fitting subsample). k grows from 1 until the weight of pure pixels farther than
   ``_FIT_TOL`` from every center is negligible, i.e. the elbow of the outlier-distortion
   curve, capped by ``max_colors``. Each new center is seeded at the heaviest poorly fitted
   bin and refined with weighted Lloyd iterations. There is no random step at all (no
   random init, no random subsampling), so the output is fully deterministic.
4. Merge centers closer than ΔE2000 3, then prune clusters that are spatially thin AND
   either a *mixture* (center ≈ sRGB blend of two other centers) or a near-duplicate of
   another center (compression ringing, chroma bleeding). Assign every opaque pixel to its
   nearest center (all pixels, not just the subsample).
5. Anti-aliasing cleanup: a pixel whose label is not anchored by a pure pixel of that label
   nearby is moved to the nearest label that is anchored nearby, so edge pixels always
   join one of the two colors they blend and never form a region of their own.
6. Speckle removal: 8-connected components smaller than ``speckle_min_area_px`` are absorbed
   into the neighbouring region with the closest color.
7. Final color of each entry = per-channel MEDIAN in LAB of its pure member pixels; final
   colors within ΔE2000 3 are merged. ``is_background`` = the label covering >= 50% of all
   border pixels (transparent border pixels count against every label).

With ``settings.palette_override`` steps 2-5 and 7 are replaced by nearest-override-color
assignment (LAB) + speckle removal; the palette is exactly the override colors that are
used, in override order.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab

from contracts.schemas import (
    ImageClass,
    ImageClassLabel,
    Palette,
    PaletteColor,
    PreprocessResult,
    Settings,
    StageError,
    rgb_to_hex,
)

STAGE = "quantize"

MERGE_DELTA_E = 3.0
"""Centers / final colors closer than this (CIEDE2000) are merged."""
_AUTO_MAX_COLORS: dict[ImageClassLabel, int] = {
    ImageClassLabel.LINE_ART: 8,
    ImageClassLabel.FLAT_COLOR: 16,
    ImageClassLabel.MIXED: 24,
}
"""Upper bound on the auto-detected color count when Settings.max_colors is None."""

_FIT_TOL = 12.0
"""LAB distance (ΔE76) above which a pure pixel counts as poorly represented by the palette."""
_OUTLIER_SHARE = 0.0004
"""Stop growing k once poorly represented pure pixels are below this share of opaque pixels."""
_GRID_STEP = 1.0
"""LAB bin size used to collapse the fitting subsample."""
_MAX_FIT_POINTS = 20_000
"""If the LAB grid still has more bins, the bin size doubles until it fits."""
_KMEANS_ITERS = 60

_EDGE_MIN = 24.0
"""Minimum sRGB distance between two opposite neighbours for the pair to count as an edge."""
_MIX_MIN = 6.0
"""A mixed pixel must differ from both neighbours by more than this (sRGB distance)."""
_MIX_SLACK = 10.0
"""Allowed detour (sRGB distance) of the path n1 -> p -> n2 versus n1 -> n2."""

_ANCHOR_TOL = 10.0
"""A pure pixel within this ΔE76 of its center anchors its label in the neighbourhood."""
_ANCHOR_RADIUS = 3
"""Neighbourhood radius (px) searched for anchoring pixels."""

_MIXTURE_TOL = 6.0
"""ΔE76 tolerance between a center and the best sRGB blend of two other centers."""
_THIN_SURVIVAL = 0.4
"""A cluster is spatially thin if less than this share of its pixels survives a 3x3 erosion."""
_ARTIFACT_DELTA_E = 20.0
"""A thin cluster within this ΔE2000 of another center is treated as an artifact of it."""

_MEDIAN_SAMPLE = 400_000
"""Max member pixels used for a median (deterministic stride subsample)."""


# ----------------------------------------------------------------------------- color math


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 sRGB -> (N, 3) float64 CIELAB (D65, 2 deg)."""
    arr = np.asarray(rgb, dtype=np.float64).reshape(-1, 1, 3) / 255.0
    return np.asarray(rgb2lab(arr), dtype=np.float64).reshape(-1, 3)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """(N, 3) CIELAB -> (N, 3) uint8 sRGB (clipped to gamut, rounded)."""
    arr = np.asarray(lab, dtype=np.float64).reshape(-1, 1, 3)
    rgb = np.clip(np.asarray(lab2rgb(arr)).reshape(-1, 3), 0.0, 1.0)
    out: np.ndarray = np.round(rgb * 255.0).astype(np.uint8)
    return out


def _delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Broadcasting CIEDE2000 between (..., 3) LAB arrays."""
    a, b = np.broadcast_arrays(np.asarray(lab1, dtype=np.float64), np.asarray(lab2, dtype=np.float64))
    return np.asarray(deltaE_ciede2000(a, b), dtype=np.float64)  # type: ignore[no-untyped-call]


def _pairwise_delta_e(lab: np.ndarray) -> np.ndarray:
    """(K, K) CIEDE2000 matrix with +inf on the diagonal."""
    de = _delta_e2000(lab[:, None, :], lab[None, :, :])
    np.fill_diagonal(de, np.inf)
    return de


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (r, g, b)."""
    return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)


def _nearest(points: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of and Euclidean distance to the nearest center for each point."""
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    d2 = (points**2).sum(axis=1)[:, None] - 2.0 * points @ centers.T + (centers**2).sum(axis=1)[None, :]
    idx = np.argmin(d2, axis=1)
    best = np.maximum(d2[np.arange(len(points)), idx], 0.0)
    return idx.astype(np.int32), np.sqrt(best)


# ----------------------------------------------------------------------------- pixel analysis


class _Pixels:
    """Opaque pixels of an image collapsed to unique colors."""

    def __init__(self, pre: PreprocessResult) -> None:
        self.shape: tuple[int, int] = (pre.height, pre.width)
        self.opaque: np.ndarray = np.ascontiguousarray(pre.opaque_mask)
        self.rgb_img: np.ndarray = np.asarray(pre.image)
        img = self.rgb_img
        codes = (img[..., 0].astype(np.int32) << 16) | (img[..., 1].astype(np.int32) << 8) | img[..., 2]
        self.flat_idx: np.ndarray = np.flatnonzero(self.opaque.ravel())
        uniq, inv = np.unique(codes.ravel()[self.flat_idx], return_inverse=True)
        self.inv: np.ndarray = inv.astype(np.int32).ravel()
        self.urgb: np.ndarray = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.uint8)
        self.ulab: np.ndarray = _rgb_to_lab(self.urgb)
        self.n: int = int(self.flat_idx.size)

    def label_image(self, per_pixel: np.ndarray) -> np.ndarray:
        """(H, W) int32 label map from per-opaque-pixel labels; transparent pixels = -1."""
        labels = np.full(self.shape[0] * self.shape[1], -1, dtype=np.int32)
        labels[self.flat_idx] = per_pixel
        return labels.reshape(self.shape)

    def pixel_labels(self, labels: np.ndarray) -> np.ndarray:
        """Per-opaque-pixel labels of an (H, W) label map."""
        out: np.ndarray = labels.ravel()[self.flat_idx]
        return out


def _mixed_mask(rgb: np.ndarray, opaque: np.ndarray) -> np.ndarray:
    """(H, W) bool: pixels whose color is a blend of two opposite, clearly different neighbours.

    Only pixels with a large local range are tested (a necessary condition), which keeps
    this cheap on large flat images.
    """
    h, w = opaque.shape
    k3 = np.ones((3, 3), np.uint8)
    spread = np.zeros((h, w), dtype=np.uint8)
    for c in range(3):
        ch = np.ascontiguousarray(rgb[..., c])
        spread = np.maximum(spread, cv2.dilate(ch, k3) - cv2.erode(ch, k3))
    ys, xs = np.nonzero((spread >= _EDGE_MIN / np.sqrt(3.0)) & opaque)
    mixed = np.zeros((h, w), dtype=bool)
    if ys.size == 0:
        return mixed
    pad = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge").astype(np.float32)
    opad = np.pad(opaque, 1, mode="constant", constant_values=False)
    py, px_ = ys + 1, xs + 1
    p = pad[py, px_]
    hit = np.zeros(ys.size, dtype=bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n1, n2 = pad[py + dy, px_ + dx], pad[py - dy, px_ - dx]
        valid = opad[py + dy, px_ + dx] & opad[py - dy, px_ - dx]
        d1, d2, d12 = (np.sqrt(np.einsum("ij,ij->i", v, v)) for v in (p - n1, p - n2, n1 - n2))
        hit |= valid & (d12 > _EDGE_MIN) & (np.minimum(d1, d2) > _MIX_MIN) & (d1 + d2 <= d12 + _MIX_SLACK)
    mixed[ys[hit], xs[hit]] = True
    return mixed


def _fit_points(lab: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse weighted LAB points onto a grid so k-means runs on a bounded subsample."""
    step = _GRID_STEP
    while True:
        keys = np.round(lab / step).astype(np.int64)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        if len(uniq) <= _MAX_FIT_POINTS:
            break
        step *= 2.0
    inv = inv.ravel()
    wsum = np.bincount(inv, weights=weights, minlength=len(uniq))
    pts = np.stack([np.bincount(inv, weights=weights * lab[:, c], minlength=len(uniq)) for c in range(3)], axis=1)
    return pts / wsum[:, None], wsum


def _kmeans(points: np.ndarray, weights: np.ndarray, init: np.ndarray) -> np.ndarray:
    """Weighted Lloyd k-means from a given initialisation (deterministic)."""
    centers = init.astype(np.float64).copy()
    k = len(centers)
    for _ in range(_KMEANS_ITERS):
        idx, _ = _nearest(points, centers)
        wsum = np.bincount(idx, weights=weights, minlength=k)
        new = centers.copy()
        filled = wsum > 0
        for c in range(3):
            new[filled, c] = np.bincount(idx, weights=weights * points[:, c], minlength=k)[filled] / wsum[filled]
        shift = float(np.abs(new - centers).max())
        centers = new
        if shift < 1e-3:
            break
    return centers


def _select_centers(points: np.ndarray, weights: np.ndarray, cap: int, min_outliers: float) -> np.ndarray:
    """Grow k until the weight of poorly fitted points drops below ``min_outliers`` (or k == cap)."""
    centers = points[[int(np.argmax(weights))]]
    max_k = min(cap, len(points))
    while True:
        _, dist = _nearest(points, centers)
        outlier = dist > _FIT_TOL
        if len(centers) >= max_k or weights[outlier].sum() <= min_outliers:
            return centers
        seed = points[int(np.argmax(np.where(outlier, weights, -1.0)))]
        centers = _kmeans(points, weights, np.vstack([centers, seed]))


def _merge_close(centers: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Repeatedly merge the closest pair of centers with ΔE2000 < MERGE_DELTA_E (weighted mean)."""
    centers, weights = centers.copy(), weights.astype(np.float64).copy()
    while len(centers) > 1:
        de = _pairwise_delta_e(centers)
        i, j = np.unravel_index(int(np.argmin(de)), de.shape)
        if de[i, j] >= MERGE_DELTA_E:
            break
        wt = weights[i] + weights[j]
        centers[i] = (centers[i] * weights[i] + centers[j] * weights[j]) / max(wt, 1e-12)
        weights[i] = wt
        centers = np.delete(centers, j, axis=0)
        weights = np.delete(weights, j)
    return centers


def _is_blend(c: int, centers_lab: np.ndarray) -> bool:
    """True if center ``c`` is (in sRGB) close to a 10-90% blend of two other centers."""
    rgb = _lab_to_rgb(centers_lab).astype(np.float64)
    others = [i for i in range(len(centers_lab)) if i != c]
    for ai, a in enumerate(others):
        for b in others[ai + 1 :]:
            ab = rgb[a] - rgb[b]
            denom = float(ab @ ab)
            if denom <= 0:
                continue
            t = float((rgb[c] - rgb[b]) @ ab) / denom
            if not 0.1 <= t <= 0.9:
                continue
            blend = np.clip(np.round(rgb[b] + t * ab), 0, 255).astype(np.uint8)
            if float(np.linalg.norm(_rgb_to_lab(blend[None])[0] - centers_lab[c])) < _MIXTURE_TOL:
                return True
    return False


def _survival(mask: np.ndarray) -> float:
    """Share of mask pixels surviving a 3x3 erosion (1.0 = solid, 0.0 = only thin structures)."""
    total = int(np.count_nonzero(mask))
    if total == 0:
        return 0.0
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
    return float(np.count_nonzero(eroded)) / total


def _is_artifact(c: int, centers: np.ndarray, de: np.ndarray, labels: np.ndarray) -> bool:
    """Thin cluster that is a near-duplicate of another center or a blend of two others."""
    close = de[c].min() < _ARTIFACT_DELTA_E or _is_blend(c, centers)
    return close and _survival(labels == c) < _THIN_SURVIVAL


def _prune_clusters(centers: np.ndarray, px: _Pixels) -> np.ndarray:
    """Drop spatially thin clusters that are mixtures of two others or artifacts of a close one."""
    while len(centers) > 1:
        labels = px.label_image(_nearest(px.ulab, centers)[0][px.inv])
        de = _pairwise_delta_e(centers)
        counts = np.bincount(labels[labels >= 0], minlength=len(centers))
        bad = [c for c in range(len(centers)) if counts[c] == 0 or _is_artifact(c, centers, de, labels)]
        if not bad:
            break
        centers = np.delete(centers, min(bad, key=lambda c: (int(counts[c]), c)), axis=0)
    return centers


def _reassign_unanchored(
    labels: np.ndarray, px: _Pixels, centers: np.ndarray, pure: np.ndarray, dist_u: np.ndarray
) -> np.ndarray:
    """Move pixels whose label has no anchoring pure pixel nearby to the nearest anchored label.

    ``dist_u`` is the LAB distance of every unique color to its own (nearest) center.
    """
    k = len(centers)
    anchor = pure & px.label_image(dist_u[px.inv] < _ANCHOR_TOL).astype(bool) & (labels >= 0)
    size = 2 * _ANCHOR_RADIUS + 1
    kernel = np.ones((size, size), np.uint8)
    near = [cv2.dilate((anchor & (labels == c)).astype(np.uint8), kernel).astype(bool) for c in range(k)]
    fix = np.zeros(labels.shape, dtype=bool)
    for c in range(k):
        fix |= (labels == c) & ~near[c]
    ys, xs = np.nonzero(fix)
    if ys.size == 0:
        return labels
    cand = np.stack([n[ys, xs] for n in near], axis=1)
    ok = cand.any(axis=1)
    ys, xs, cand = ys[ok], xs[ok], cand[ok]
    lab = _rgb_to_lab_cached(px, ys, xs)
    d2 = ((lab[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    d2[~cand] = np.inf
    out = labels.copy()
    out[ys, xs] = np.argmin(d2, axis=1).astype(np.int32)
    return out


def _rgb_to_lab_cached(px: _Pixels, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """LAB of the opaque pixels at (ys, xs), looked up from the unique-color table."""
    codes = px.rgb_img[ys, xs].astype(np.int64)
    key = (codes[:, 0] << 16) | (codes[:, 1] << 8) | codes[:, 2]
    ukey = (px.urgb[:, 0].astype(np.int64) << 16) | (px.urgb[:, 1].astype(np.int64) << 8) | px.urgb[:, 2]
    out: np.ndarray = px.ulab[np.searchsorted(ukey, key)]
    return out


def _small_components(labels: np.ndarray, k: int, min_area: int) -> tuple[np.ndarray, np.ndarray]:
    """Find speckles: 8-connected components whose *bridged* area is below ``min_area``.

    The area is measured on components of the label mask dilated by 1 px (counting only
    real member pixels), so a thin line cut into short pieces by a crossing stroke of
    another color is not mistaken for speckles. Returns an (H, W) map of speckle ids
    (-1 elsewhere) and the label of each speckle.
    """
    comp_map = np.full(labels.shape, -1, dtype=np.int32)
    owners: list[np.ndarray] = []
    next_id = 0
    k3 = np.ones((3, 3), np.uint8)
    for c in range(k):
        mask = (labels == c).astype(np.uint8)
        if not mask.any():
            continue
        member = mask.astype(bool)
        nb, bridged = cv2.connectedComponents(cv2.dilate(mask, k3), connectivity=8)
        area = np.bincount(bridged[member], minlength=nb)
        small = (area > 0) & (area < min_area)
        if not small.any():
            continue
        hole = member & small[bridged]
        _, comp = cv2.connectedComponents(hole.astype(np.uint8), connectivity=8)
        uniq, inv = np.unique(comp[hole], return_inverse=True)
        comp_map[hole] = next_id + inv.astype(np.int32).ravel()
        owners.append(np.full(len(uniq), c, dtype=np.int32))
        next_id += len(uniq)
    owner = np.concatenate(owners) if owners else np.zeros(0, dtype=np.int32)
    return comp_map, owner


def _remove_speckles(labels: np.ndarray, colors_lab: np.ndarray, min_area: int) -> np.ndarray:
    """Absorb regions smaller than ``min_area`` into the neighbouring region with the closest color.

    Each speckle (see ``_small_components``) is relabelled as a whole to the adjacent settled
    label with the smallest ΔE2000 to its own color. Speckles that only touch other speckles
    are resolved in later rounds; speckles with no opaque neighbour at all (islands
    surrounded by transparency) keep their label.
    """
    k = len(colors_lab)
    if min_area <= 1 or k < 2:
        return labels
    comp_map, owner = _small_components(labels, k, min_area)
    if owner.size == 0:
        return labels
    de = _pairwise_delta_e(colors_lab)
    h, w = labels.shape
    out = np.pad(labels, 1, mode="constant", constant_values=-1)
    cmap = np.pad(comp_map, 1, mode="constant", constant_values=-1)
    ys, xs = np.nonzero(cmap >= 0)
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    while ys.size:
        cid = cmap[ys, xs]
        pair_c = np.concatenate([cid] * len(offsets))
        pair_l = np.concatenate([out[ys + dy, xs + dx] for dy, dx in offsets])
        settled = np.concatenate([cmap[ys + dy, xs + dx] < 0 for dy, dx in offsets]) & (pair_l >= 0)
        pair_c, pair_l = pair_c[settled], pair_l[settled]
        if pair_c.size == 0:
            break
        cost = de[owner[pair_c], pair_l]
        order = np.lexsort((pair_l, cost, pair_c))
        first = np.ones(order.size, dtype=bool)
        first[1:] = pair_c[order][1:] != pair_c[order][:-1]
        target = np.full(owner.size, -1, dtype=np.int32)
        target[pair_c[order][first]] = pair_l[order][first]
        done = target[cid] >= 0
        out[ys[done], xs[done]] = target[cid[done]]
        cmap[ys[done], xs[done]] = -1
        ys, xs = ys[~done], xs[~done]
    return np.ascontiguousarray(out[1 : h + 1, 1 : w + 1])


def _weighted_median_lab(ulab: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-channel weighted median of unique LAB colors with the given pixel counts."""
    out = np.empty(3)
    nz = counts > 0
    vals, wts = ulab[nz], counts[nz].astype(np.float64)
    half = wts.sum() / 2.0
    for c in range(3):
        order = np.argsort(vals[:, c], kind="stable")
        cum = np.cumsum(wts[order])
        out[c] = vals[order[int(np.searchsorted(cum, half))], c]
    return out


def _final_colors(per_pixel: np.ndarray, used: list[int], px: _Pixels, pure_px: np.ndarray) -> np.ndarray:
    """(len(used), 3) LAB: median of each label's pure members (all members if too few pure)."""
    medians = []
    for c in used:
        members = per_pixel == c
        sel = members & pure_px
        if np.count_nonzero(sel) < max(8, 0.02 * np.count_nonzero(members)):
            sel = members
        inv = px.inv[sel]
        if inv.size > _MEDIAN_SAMPLE:
            inv = inv[:: inv.size // _MEDIAN_SAMPLE + 1]
        medians.append(_weighted_median_lab(px.ulab, np.bincount(inv, minlength=len(px.ulab))))
    return np.array(medians)


def _background_label(labels: np.ndarray, k: int) -> int | None:
    """Label covering >= 50% of all border pixels, if any."""
    border = np.concatenate([labels[0], labels[-1], labels[1:-1, 0], labels[1:-1, -1]])
    counts = np.bincount(border[border >= 0], minlength=k)
    best = int(np.argmax(counts))
    return best if counts[best] * 2 >= border.size else None


def _build_palette(labels: np.ndarray, rgbs: dict[int, tuple[int, int, int]], order: list[int]) -> Palette:
    """Build the Palette: label ``order[i]`` becomes index i with color ``rgbs[order[i]]``."""
    remap = np.full(int(labels.max()) + 2, -1, dtype=np.int32)
    for new, old in enumerate(order):
        remap[old] = new
    out = remap[labels]  # label -1 indexes the last (sentinel) slot and stays -1
    k = len(order)
    counts = np.bincount(out[out >= 0], minlength=k)
    bg = _background_label(out, k)
    labs = _rgb_to_lab(np.array([rgbs[o] for o in order], dtype=np.uint8))
    colors = [
        PaletteColor(
            index=new,
            rgb=rgbs[old],
            lab=(float(labs[new, 0]), float(labs[new, 1]), float(labs[new, 2])),
            hex=rgb_to_hex(rgbs[old]),
            pixel_count=int(counts[new]),
            is_background=bg == new,
        )
        for new, old in enumerate(order)
    ]
    return Palette(colors=colors, label_map=out)


# ----------------------------------------------------------------------------- entrypoints


def _quantize_override(px: _Pixels, override: list[str], min_area: int) -> Palette:
    """Assign every opaque pixel to the nearest override color, clean speckles, keep used colors."""
    rgbs = [_hex_to_rgb(h) for h in override]
    lab = _rgb_to_lab(np.array(rgbs, dtype=np.uint8))
    labels = px.label_image(_nearest(px.ulab, lab)[0][px.inv])
    labels = _remove_speckles(labels, lab, min_area)
    used = np.bincount(labels[labels >= 0], minlength=len(rgbs))
    order = [i for i in range(len(rgbs)) if used[i] > 0]
    return _build_palette(labels, dict(enumerate(rgbs)), order)


def _quantize_auto(px: _Pixels, cap: int, min_area: int) -> Palette:
    """Auto-detected palette (see module docstring)."""
    mixed = _mixed_mask(px.rgb_img, px.opaque)
    pure_px = ~px.pixel_labels(mixed.astype(np.int32)).astype(bool)
    fit_inv = px.inv[pure_px] if pure_px.any() else px.inv
    ucount = np.bincount(fit_inv, minlength=len(px.urgb)).astype(np.float64)
    keep = ucount > 0
    points, weights = _fit_points(px.ulab[keep], ucount[keep])

    centers = _select_centers(points, weights, cap, _OUTLIER_SHARE * px.n)
    idx, _ = _nearest(points, centers)
    centers = _merge_close(centers, np.bincount(idx, weights=weights, minlength=len(centers)))
    centers = _prune_clusters(centers, px)

    label_u, dist_u = _nearest(px.ulab, centers)
    labels = px.label_image(label_u[px.inv])
    labels = _reassign_unanchored(labels, px, centers, ~mixed, dist_u)
    labels = _remove_speckles(labels, centers, min_area)

    while True:
        per_pixel = px.pixel_labels(labels)
        counts = np.bincount(per_pixel, minlength=len(centers))
        used = [c for c in range(len(centers)) if counts[c] > 0]
        rgb_used = _lab_to_rgb(_final_colors(per_pixel, used, px, pure_px))
        if len(used) < 2:
            break
        de = _pairwise_delta_e(_rgb_to_lab(rgb_used))
        i, j = np.unravel_index(int(np.argmin(de)), de.shape)
        if de[i, j] >= MERGE_DELTA_E:
            break
        big, small = sorted((used[i], used[j]), key=lambda c: (-int(counts[c]), c))
        labels = np.where(labels == small, big, labels).astype(np.int32)

    rgbs = {c: (int(r[0]), int(r[1]), int(r[2])) for c, r in zip(used, rgb_used, strict=True)}
    order = sorted(used, key=lambda c: (-int(counts[c]), c))
    return _build_palette(labels, rgbs, order)


def quantize(pre: PreprocessResult, image_class: ImageClass, settings: Settings) -> Palette:
    """Quantize the opaque pixels of ``pre`` into a palette; transparent pixels get label -1.

    Colors are ordered by descending pixel count. Honours ``settings.palette_override``
    (nearest override color per pixel, auto-detection skipped) and ``settings.max_colors``
    (upper bound on the auto-detected count). Raises StageError if no pixel is opaque.
    """
    px = _Pixels(pre)
    if px.n == 0:
        raise StageError(STAGE, "image has no opaque pixels to quantize")
    min_area = settings.preset.speckle_min_area_px
    if settings.palette_override:
        return _quantize_override(px, list(settings.palette_override), min_area)
    cap = settings.max_colors if settings.max_colors is not None else _AUTO_MAX_COLORS[image_class.label]
    return _quantize_auto(px, cap, min_area)
