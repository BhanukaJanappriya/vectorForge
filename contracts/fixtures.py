"""Factories producing small, valid contract objects for mocking upstream stages.

Agents use these in unit tests until the real upstream module lands, e.g.:

    pre = make_preprocess_result()
    palette = make_palette(pre)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from contracts.schemas import (
    DenoiseParams,
    DocumentMetadata,
    ImageClass,
    ImageClassLabel,
    ImageInput,
    ImageMode,
    LineMap,
    Palette,
    PaletteColor,
    PreprocessResult,
    Settings,
    SourceFormat,
    VectorDocument,
    VectorLayer,
    rgb_to_hex,
)

# Two flat colors: white background, red square in the middle, 1-px black frame line.
_COLORS: list[tuple[tuple[int, int, int], tuple[float, float, float]]] = [
    ((255, 255, 255), (100.0, 0.0, 0.0)),
    ((220, 50, 47), (48.3, 67.9, 46.3)),
    ((0, 0, 0), (0.0, 0.0, 0.0)),
]


def _labels(size: int) -> np.ndarray:
    labels = np.zeros((size, size), dtype=np.int32)
    q = size // 4
    labels[q : 3 * q, q : 3 * q] = 1
    labels[size // 8, size // 8 : size - size // 8] = 2
    return labels


def make_image_input(size: int = 64, has_alpha: bool = False, path: Path | None = None) -> ImageInput:
    return ImageInput(
        path=path or Path("fixture.png"),
        width=size,
        height=size,
        has_alpha=has_alpha,
        mode=ImageMode.RGBA if has_alpha else ImageMode.RGB,
        source_format=SourceFormat.PNG,
        file_size_bytes=1024,
    )


def make_preprocess_result(size: int = 64, has_alpha: bool = False) -> PreprocessResult:
    labels = _labels(size)
    lut = np.array([c for c, _ in _COLORS], dtype=np.uint8)
    alpha = None
    if has_alpha:
        alpha = np.full((size, size), 255, dtype=np.uint8)
        alpha[: size // 16] = 0
    return PreprocessResult(
        source=make_image_input(size, has_alpha),
        image=lut[labels],
        alpha=alpha,
        scale_factor=1.0,
        denoise=DenoiseParams(method="none", strength=0.0),
    )


def make_image_class(label: ImageClassLabel = ImageClassLabel.FLAT_COLOR) -> ImageClass:
    return ImageClass(label=label, confidence=0.9)


def make_palette(pre: PreprocessResult | None = None) -> Palette:
    pre = pre or make_preprocess_result()
    labels = _labels(pre.width)
    labels[~pre.opaque_mask] = -1
    counts = np.bincount(labels[labels >= 0].ravel(), minlength=len(_COLORS))
    colors = [
        PaletteColor(index=i, rgb=rgb, lab=lab, hex=rgb_to_hex(rgb), pixel_count=int(counts[i]), is_background=i == 0)
        for i, (rgb, lab) in enumerate(_COLORS)
    ]
    return Palette(colors=colors, label_map=labels)


def make_line_map(pre: PreprocessResult | None = None) -> LineMap:
    pre = pre or make_preprocess_result()
    mask = _labels(pre.width) == 2
    width_map = np.where(mask, 1.0, 0.0).astype(np.float32)
    return LineMap(mask=mask, skeleton=mask.copy(), width_map=width_map, median_stroke_width=1.0, color_rgb=(0, 0, 0))


def make_vector_document(size: int = 64, settings: Settings | None = None) -> VectorDocument:
    q, e = size // 4, size // 8
    return VectorDocument(
        width=size,
        height=size,
        layers=[
            VectorLayer(
                id="color_1_FFFFFF",
                name="color_1_#FFFFFF",
                role="background",
                color_hex="#ffffff",
                paths=[f"M0 0L{size} 0L{size} {size}L0 {size}Z"],
                z_order=0,
                palette_index=0,
            ),
            VectorLayer(
                id="color_2_DC322F",
                name="color_2_#DC322F",
                role="fill",
                color_hex="#dc322f",
                paths=[f"M{q} {q}L{3 * q} {q}C{3 * q} {2 * q} {3 * q} {2 * q} {3 * q} {3 * q}L{q} {3 * q}Z"],
                z_order=1,
                palette_index=1,
            ),
            VectorLayer(
                id="line_3_000000",
                name="line_3_#000000",
                role="line",
                color_hex="#000000",
                paths=[f"M{e} {e + 0.5}L{size - e} {e + 0.5}"],
                z_order=2,
                is_stroke=True,
                stroke_width=1.0,
                palette_index=2,
            ),
        ],
        metadata=DocumentMetadata(
            source_filename="fixture.png",
            image_class=ImageClassLabel.FLAT_COLOR,
            settings=settings or Settings(),
            palette_hex=[rgb_to_hex(c) for c, _ in _COLORS],
        ),
    )
