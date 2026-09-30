"""Ground-truth "oracle" stand-ins for pipeline stages (eval-internal test utilities).

They let the QA harness produce real metric numbers before the pipeline stages exist, and
give the metric self-tests a known-good reference:

* :func:`oracle_load_image` / :func:`oracle_preprocess`: Pillow decode, no denoising.
* :func:`oracle_classify`: the sample's ground-truth class.
* :func:`oracle_quantize`: every opaque pixel -> nearest ground-truth palette color
  (CIEDE2000), or k-means in LAB when the sample has no exact palette.
* :func:`oracle_document`: exact pixel-run rectangles of the label map (merged vertically),
  one layer per color, background stacked underneath as a full-canvas rectangle.
* :func:`oracle_svg` / :func:`oracle_export`: lxml SVG + preview.png rendered by
  :mod:`eval.raster` (no Cairo).

None of this is pipeline code; it never replaces a stage in production.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from lxml import etree
from PIL import Image
from skimage.color import lab2rgb

from contracts.schemas import (
    DenoiseParams,
    DocumentMetadata,
    ExportBundle,
    ImageClass,
    ImageClassLabel,
    ImageInput,
    ImageMode,
    Palette,
    PaletteColor,
    PreprocessResult,
    Settings,
    SourceFormat,
    VectorDocument,
    VectorLayer,
    layer_name,
    rgb_to_hex,
)
from eval.evaluate import SVG_NS, delta_e, rgb_to_lab
from eval.raster import hex_to_rgb, render_document

INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
RECTS_PER_PATH = 1000
AUTO_COLORS = 16


# --------------------------------------------------------------------------------------
# load / preprocess / classify
# --------------------------------------------------------------------------------------


def oracle_load_image(path: Path) -> ImageInput:
    """Describe an image file with Pillow (no validation beyond decodability)."""
    with Image.open(path) as img:
        fmt = SourceFormat.JPEG if (img.format or "").upper() in {"JPEG", "JPG"} else SourceFormat.PNG
        mode = ImageMode(img.mode) if img.mode in ImageMode.__members__.values() else ImageMode.RGB
        has_alpha = False
        if "A" in img.getbands() or "transparency" in img.info:
            has_alpha = bool(np.asarray(img.convert("RGBA"))[..., 3].min() < 255)
        width, height = img.size
    return ImageInput(
        path=path,
        width=width,
        height=height,
        has_alpha=has_alpha,
        mode=mode,
        source_format=fmt,
        file_size_bytes=path.stat().st_size,
    )


def oracle_preprocess(image: ImageInput, settings: Settings | None = None) -> PreprocessResult:
    """Decode to sRGB + alpha at scale 1.0 without denoising."""
    with Image.open(image.path) as img:
        rgba = np.asarray(img.convert("RGBA"))
    return PreprocessResult(
        source=image,
        image=np.ascontiguousarray(rgba[..., :3]),
        alpha=np.ascontiguousarray(rgba[..., 3]) if image.has_alpha else None,
        scale_factor=1.0,
        denoise=DenoiseParams(method="none", strength=0.0),
    )


def oracle_classify(label: str | ImageClassLabel) -> ImageClass:
    """The ground-truth class with full confidence."""
    return ImageClass(label=ImageClassLabel(label), confidence=1.0, features={"oracle": 1.0})


# --------------------------------------------------------------------------------------
# quantize
# --------------------------------------------------------------------------------------


def _kmeans_palette(uniq_lab: np.ndarray, counts: np.ndarray, n_colors: int, iterations: int = 25) -> np.ndarray:
    """Pixel-count-weighted k-means (Lloyd, farthest-point init) in LAB -> (K, 3) LAB centers.

    Fits on at most 50k unique colors: the 5k most frequent plus a uniform random rest,
    so large flat regions are never dropped from the sample. Deterministic.
    """
    n = int(min(n_colors, uniq_lab.shape[0]))
    idx = np.arange(uniq_lab.shape[0])
    if idx.size > 50_000:
        top = np.argsort(-counts, kind="stable")[:5_000]
        rest = np.setdiff1d(idx, top)
        idx = np.concatenate([top, np.random.default_rng(0).choice(rest, 45_000, replace=False)])
    x, wts = uniq_lab[idx], counts[idx].astype(np.float64)
    centers = [x[int(np.argmax(wts))]]
    d2 = ((x - centers[0]) ** 2).sum(1)
    for _ in range(1, n):
        centers.append(x[int(np.argmax(d2 * np.sqrt(wts)))])
        d2 = np.minimum(d2, ((x - centers[-1]) ** 2).sum(1))
    c = np.array(centers)
    for _ in range(iterations):
        assign = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        sums = np.zeros_like(c)
        np.add.at(sums, assign, x * wts[:, None])
        mass = np.bincount(assign, weights=wts, minlength=n)
        new = np.where(mass[:, None] > 0, sums / np.maximum(mass, 1e-12)[:, None], c)
        if np.allclose(new, c, atol=1e-3):
            c = new
            break
        c = new
    return c


def _lab_to_rgb8(lab: np.ndarray) -> np.ndarray:
    rgb = lab2rgb(np.asarray(lab, dtype=np.float64).reshape(-1, 1, 3)).reshape(-1, 3)
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-column weighted median of (N, 3) values."""
    out = np.empty(values.shape[1])
    for ch in range(values.shape[1]):
        order = np.argsort(values[:, ch], kind="stable")
        cw = np.cumsum(weights[order])
        out[ch] = values[order[np.searchsorted(cw, 0.5 * cw[-1])], ch]
    return out


def _border_background(labels: np.ndarray) -> int | None:
    """Label covering >= 50% of the image border, if any."""
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    border = border[border >= 0]
    if border.size == 0:
        return None
    counts = np.bincount(border)
    best = int(np.argmax(counts))
    return best if counts[best] >= 0.5 * border.size else None


def oracle_quantize(pre: PreprocessResult, palette_hex: Sequence[str] | None, n_auto: int = AUTO_COLORS) -> Palette:
    """Assign each opaque pixel to the nearest ground-truth color (or k-means colors)."""
    image = np.asarray(pre.image)
    opaque = pre.opaque_mask
    packed = (image[..., 0].astype(np.int32) << 16) | (image[..., 1].astype(np.int32) << 8) | image[..., 2]
    uniq, inverse, counts = np.unique(packed[opaque], return_inverse=True, return_counts=True)
    uniq_rgb = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.uint8)
    uniq_lab = rgb_to_lab(uniq_rgb)
    if palette_hex:
        pal_rgb = np.array([hex_to_rgb(h) for h in palette_hex], dtype=np.uint8)
        pal_lab = rgb_to_lab(pal_rgb)
        uniq_label = np.argmin(delta_e(uniq_lab[:, None, :], pal_lab[None, :, :]), axis=1).astype(np.int32)
    else:
        centers = _kmeans_palette(uniq_lab, counts.astype(np.float64), n_auto)
        uniq_label = np.argmin(((uniq_lab[:, None, :] - centers[None]) ** 2).sum(-1), axis=1).astype(np.int32)
        # Oracle colors = median LAB of each cluster (what the Delta-E metric compares against).
        medians = np.array(
            [
                _weighted_median(uniq_lab[uniq_label == k], counts[uniq_label == k].astype(np.float64))
                if np.any(uniq_label == k)
                else centers[k]
                for k in range(centers.shape[0])
            ]
        )
        pal_rgb = _lab_to_rgb8(medians)
        pal_lab = rgb_to_lab(pal_rgb)
    labels = np.full(image.shape[:2], -1, dtype=np.int32)
    labels[opaque] = uniq_label[inverse.ravel()]
    # Drop palette entries that ended up unused and re-index densely.
    used = np.unique(labels[labels >= 0])
    remap = np.full(len(pal_rgb), -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    labels = np.where(labels >= 0, remap[np.maximum(labels, 0)], -1).astype(np.int32)
    pal_rgb, pal_lab = pal_rgb[used], pal_lab[used]
    bg = _border_background(labels) if pre.alpha is None else None
    pixel_counts = np.bincount(labels[labels >= 0].ravel(), minlength=used.size)
    colors = [
        PaletteColor(
            index=i,
            rgb=(int(rgb[0]), int(rgb[1]), int(rgb[2])),
            lab=(float(lab[0]), float(lab[1]), float(lab[2])),
            hex=rgb_to_hex((int(rgb[0]), int(rgb[1]), int(rgb[2]))),
            pixel_count=int(pixel_counts[i]),
            is_background=i == bg,
        )
        for i, (rgb, lab) in enumerate(zip(pal_rgb, pal_lab, strict=True))
    ]
    return Palette(colors=colors, label_map=labels)


# --------------------------------------------------------------------------------------
# vectorize (pixel-run rectangles)
# --------------------------------------------------------------------------------------


def label_rectangles(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a label map into axis-aligned rectangles (x0, y0, x1, y1, label), labels >= 0.

    Horizontal runs are found per row; identical runs in consecutive rows are merged.
    """
    h, w = labels.shape
    starts = np.ones((h, w), dtype=bool)
    starts[:, 1:] = labels[:, 1:] != labels[:, :-1]
    rows, c0 = np.nonzero(starts)
    order_end = np.empty_like(c0)
    order_end[:-1] = c0[1:]
    order_end[-1] = w
    same_row_next = np.zeros(rows.size, dtype=bool)
    same_row_next[:-1] = rows[1:] == rows[:-1]
    c1 = np.where(same_row_next, order_end, w)
    lab = labels[rows, c0]
    keep = lab >= 0
    rows, c0, c1, lab = rows[keep], c0[keep], c1[keep], lab[keep]
    order = np.lexsort((rows, c1, c0, lab))
    rows, c0, c1, lab = rows[order], c0[order], c1[order], lab[order]
    cont = np.zeros(rows.size, dtype=bool)
    cont[1:] = (lab[1:] == lab[:-1]) & (c0[1:] == c0[:-1]) & (c1[1:] == c1[:-1]) & (rows[1:] == rows[:-1] + 1)
    group_start = np.nonzero(~cont)[0]
    group_end = np.append(group_start[1:], rows.size) - 1
    return c0[group_start], rows[group_start], c1[group_start], rows[group_end] + 1, lab[group_start]


def _fmt(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _rect_paths(x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray, inv_scale: float) -> list[str]:
    subpaths = []
    for a, b, c, d in zip(x0.tolist(), y0.tolist(), x1.tolist(), y1.tolist(), strict=True):
        xa, ya, xb, yb = (_fmt(v * inv_scale) for v in (a, b, c, d))
        subpaths.append(f"M{xa} {ya}L{xb} {ya}L{xb} {yb}L{xa} {yb}Z")
    return ["".join(subpaths[i : i + RECTS_PER_PATH]) for i in range(0, len(subpaths), RECTS_PER_PATH)]


def oracle_document(
    pre: PreprocessResult,
    palette: Palette,
    image_class: ImageClassLabel,
    *,
    labels: np.ndarray | None = None,
    stack_background: bool = True,
    settings: Settings | None = None,
) -> VectorDocument:
    """Exact pixel-run vectorization of ``labels`` (default: the palette label map).

    With ``stack_background`` the background color is one full-canvas rectangle at z=0 and
    its runs are omitted (the stacking convention). Other layers are ordered by pixel count.
    """
    labels = np.asarray(palette.label_map if labels is None else labels)
    w, h = pre.source.width, pre.source.height
    inv_scale = 1.0 / pre.scale_factor
    bg = palette.background_index if stack_background else None
    x0, y0, x1, y1, lab = label_rectangles(labels)
    counts = np.bincount(lab, minlength=len(palette.colors))
    order = [i for i in np.argsort(-counts, kind="stable").tolist() if counts[i] > 0 and i != bg]
    layers: list[VectorLayer] = []
    if bg is not None:
        color = palette.colors[bg]
        lid, name = layer_name("background", 1, color.hex)
        layers.append(
            VectorLayer(
                id=lid,
                name=name,
                role="background",
                color_hex=color.hex,
                paths=[f"M0 0L{w} 0L{w} {h}L0 {h}Z"],
                z_order=0,
                palette_index=bg,
            )
        )
    for index in order:
        sel = lab == index
        color = palette.colors[index]
        z = len(layers)
        role = "background" if index == palette.background_index and bg is None else "fill"
        lid, name = layer_name(role, z + 1, color.hex)
        layers.append(
            VectorLayer(
                id=lid,
                name=name,
                role=role,
                color_hex=color.hex,
                paths=_rect_paths(x0[sel], y0[sel], x1[sel], y1[sel], inv_scale),
                z_order=z,
                palette_index=index,
            )
        )
    return VectorDocument(
        width=w,
        height=h,
        layers=layers,
        metadata=DocumentMetadata(
            source_filename=pre.source.filename,
            image_class=image_class,
            settings=settings or Settings(),
            palette_hex=[c.hex for c in palette.colors],
        ),
    )


# --------------------------------------------------------------------------------------
# assemble / export
# --------------------------------------------------------------------------------------


def oracle_svg(doc: VectorDocument) -> str:
    """Serialize a VectorDocument to SVG 1.1 with one Inkscape layer <g> per VectorLayer."""
    root = etree.Element(f"{{{SVG_NS}}}svg", nsmap={None: SVG_NS, "inkscape": INKSCAPE_NS})
    root.set("version", "1.1")
    root.set("width", str(doc.width))
    root.set("height", str(doc.height))
    root.set("viewBox", f"0 0 {doc.width} {doc.height}")
    for layer in doc.layers:
        g = etree.SubElement(root, f"{{{SVG_NS}}}g", id=layer.id)
        g.set(f"{{{INKSCAPE_NS}}}groupmode", "layer")
        g.set(f"{{{INKSCAPE_NS}}}label", layer.name)
        if layer.is_stroke:
            g.set("fill", "none")
            g.set("stroke", layer.color_hex)
            g.set("stroke-width", _fmt(float(layer.stroke_width or 1.0)))
            g.set("stroke-linecap", "round")
            g.set("stroke-linejoin", "round")
        else:
            g.set("fill", layer.color_hex)
            g.set("fill-rule", layer.fill_rule)
        if layer.opacity < 1.0:
            g.set("opacity", _fmt(layer.opacity))
        for d in layer.paths:
            etree.SubElement(g, f"{{{SVG_NS}}}path", d=d)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8").decode("utf-8")


def write_preview(doc: VectorDocument, path: Path) -> Path:
    """Render ``doc`` with eval.raster and save an RGBA PNG."""
    Image.fromarray(render_document(doc), mode="RGBA").save(path, compress_level=1)
    return path


def oracle_export(svg: str, doc: VectorDocument, out_dir: Path) -> ExportBundle:
    """Write output.svg and preview.png (rasterized from ``doc``) into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "output.svg"
    svg_path.write_text(svg, encoding="utf-8")
    preview = write_preview(doc, out_dir / "preview.png")
    return ExportBundle(
        svg_path=svg_path,
        preview_png_path=preview,
        warnings=["eval oracle export: preview.png rasterized by eval.raster (no Cairo); no .ai/.eps"],
    )
