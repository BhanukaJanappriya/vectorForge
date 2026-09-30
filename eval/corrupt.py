"""Deliberately broken variants of an oracle conversion, used to prove the metrics discriminate.

Each corruption declares the check(s) it must flip from pass (on the clean oracle) to fail:

=============  ==========================================  =====================
corruption     what it does                                 expected failing check
=============  ==========================================  =====================
shift_colors   every layer color darkened by dL = 5 (LAB)   mean/max_delta_e
missing_layer  the largest non-background layer is dropped  max_delta_e
seam           1-px transparent seam along region borders   gap_ratio
blur           preview.png Gaussian-blurred (sigma = 3 px)  ssim
=============  ==========================================  =====================
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.color import lab2rgb

from contracts.schemas import (
    ExportBundle,
    ImageClassLabel,
    MetricName,
    Palette,
    PreprocessResult,
    QualityReport,
    VectorDocument,
    rgb_to_hex,
)
from eval.evaluate import evaluate, rgb_to_lab
from eval.oracle import (
    oracle_document,
    oracle_export,
    oracle_load_image,
    oracle_preprocess,
    oracle_quantize,
    oracle_svg,
)
from eval.raster import hex_to_rgb

SHIFT_DL = -5.0
BLUR_SIGMA = 3.0


def shift_colors(doc: VectorDocument, d_l: float = SHIFT_DL) -> VectorDocument:
    """Shift every layer's lightness by ``d_l`` in CIELAB (by ``-d_l`` where that would leave 0..100)."""
    layers = []
    for layer in doc.layers:
        lab = rgb_to_lab(np.array([hex_to_rgb(layer.color_hex)], dtype=np.uint8))[0]
        shifted = lab[0] + d_l
        lab[0] = shifted if 0.0 <= shifted <= 100.0 else lab[0] - d_l  # flip at the gamut edge
        rgb = np.clip(np.round(lab2rgb(lab.reshape(1, 1, 3)).reshape(3) * 255), 0, 255).astype(int)
        layers.append(layer.model_copy(update={"color_hex": rgb_to_hex((int(rgb[0]), int(rgb[1]), int(rgb[2])))}))
    return doc.model_copy(update={"layers": layers})


def _path_area_hint(layer_paths: list[str]) -> int:
    return sum(len(d) for d in layer_paths)


def drop_layer(doc: VectorDocument, layer_id: str | None = None) -> VectorDocument:
    """Remove ``layer_id`` (default: the non-background layer with the most path data)."""
    if layer_id is None:
        fills = [layer for layer in doc.layers if layer.role != "background"] or list(doc.layers)
        layer_id = max(fills, key=lambda layer: _path_area_hint(layer.paths)).id
    return doc.model_copy(update={"layers": [layer for layer in doc.layers if layer.id != layer_id]})


def seam_labels(labels: np.ndarray) -> np.ndarray:
    """Mark every pixel whose right or lower neighbour has a different opaque label as -1."""
    out = np.array(labels, dtype=np.int32, copy=True)
    seam = np.zeros(labels.shape, dtype=bool)
    right = (labels[:, :-1] != labels[:, 1:]) & (labels[:, :-1] >= 0) & (labels[:, 1:] >= 0)
    down = (labels[:-1, :] != labels[1:, :]) & (labels[:-1, :] >= 0) & (labels[1:, :] >= 0)
    seam[:, :-1] |= right
    seam[:-1, :] |= down
    out[seam] = -1
    return out


def blur_preview(bundle: ExportBundle, sigma: float = BLUR_SIGMA) -> None:
    """Gaussian-blur preview.png in place."""
    with Image.open(bundle.preview_png_path) as img:
        rgba = np.asarray(img.convert("RGBA")).copy()
    Image.fromarray(cv2.GaussianBlur(rgba, (0, 0), sigma), mode="RGBA").save(bundle.preview_png_path)


@dataclass
class CorruptionResult:
    """Outcome of one corruption vs its clean baseline."""

    sample: str
    corruption: str
    expected: list[str]
    baseline_failed: list[str]
    failed: list[str]
    flipped: list[str] = field(default_factory=list)
    ok: bool = False
    metrics: dict[str, float | None] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _failed(report: QualityReport) -> list[str]:
    return [c.name.value for c in report.checks if not c.passed]


def _metrics(report: QualityReport) -> dict[str, float | None]:
    return {
        "ssim": report.ssim,
        "mean_delta_e": report.mean_delta_e,
        "max_delta_e": report.max_delta_e,
        "gap_ratio": report.gap_ratio,
        "alpha_iou": report.alpha_iou,
    }


EXPECTED: dict[str, list[str]] = {
    "shift_colors": [MetricName.MEAN_DELTA_E.value, MetricName.MAX_DELTA_E.value],
    "missing_layer": [MetricName.MAX_DELTA_E.value],
    "seam": [MetricName.GAP_RATIO.value],
    "blur": [MetricName.SSIM.value],
}

DEFAULT_PLAN: list[tuple[str, str]] = [
    ("01_logo_4color", "shift_colors"),
    ("01_logo_4color", "missing_layer"),
    ("01_logo_4color", "seam"),
    ("04_text", "blur"),
]
"""(sample, corruption). Blur uses the edge-dense text sample: on flat logos a sigma-3 blur
keeps SSIM ~0.96, so SSIM alone cannot flag it there (see report notes)."""


def corrupt_and_evaluate(
    pre: PreprocessResult,
    palette: Palette,
    image_class: ImageClassLabel,
    corruption: str,
    out_dir: Path,
) -> tuple[QualityReport, QualityReport]:
    """(clean oracle report, corrupted report) for one corruption."""
    doc = oracle_document(pre, palette, image_class)
    clean = evaluate(pre, palette, None, doc, oracle_export(oracle_svg(doc), doc, out_dir / "clean"), 0.0)
    builders: dict[str, Callable[[], VectorDocument]] = {
        "shift_colors": lambda: shift_colors(doc),
        "missing_layer": lambda: drop_layer(doc),
        "seam": lambda: oracle_document(
            pre, palette, image_class, labels=seam_labels(np.asarray(palette.label_map)), stack_background=False
        ),
        "blur": lambda: doc,
    }
    if corruption not in builders:
        raise ValueError(f"unknown corruption {corruption!r}; choose from {sorted(builders)}")
    bad_doc = builders[corruption]()
    bundle = oracle_export(oracle_svg(bad_doc), bad_doc, out_dir / corruption)
    if corruption == "blur":
        blur_preview(bundle)
    return clean, evaluate(pre, palette, None, bad_doc, bundle, 0.0)


def run_corruption(sample_png: Path, corruption: str, out_dir: Path) -> CorruptionResult:
    """Run one corruption on a sample through the oracle chain and compare with the clean baseline."""
    truth = json.loads(sample_png.with_suffix(".json").read_text(encoding="utf-8"))
    pre = oracle_preprocess(oracle_load_image(sample_png))
    palette = oracle_quantize(pre, truth.get("palette_hex"))
    clean, bad = corrupt_and_evaluate(
        pre, palette, ImageClassLabel(truth["expected_class"]), corruption, out_dir / sample_png.stem
    )
    base_failed, failed = _failed(clean), _failed(bad)
    flipped = [name for name in failed if name not in base_failed]
    expected = EXPECTED[corruption]
    return CorruptionResult(
        sample=sample_png.stem,
        corruption=corruption,
        expected=expected,
        baseline_failed=base_failed,
        failed=failed,
        flipped=flipped,
        ok=all(name in flipped for name in expected),
        metrics=_metrics(bad),
    )


def run_plan(samples_dir: Path, out_dir: Path, plan: list[tuple[str, str]] | None = None) -> list[CorruptionResult]:
    """Run each (sample, corruption) pair; samples are looked up by stem in ``samples_dir``."""
    results = []
    for stem, corruption in DEFAULT_PLAN if plan is None else plan:
        matches = sorted(p for p in samples_dir.glob(f"{stem}.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not matches:
            raise FileNotFoundError(f"sample {stem!r} not found in {samples_dir}")
        results.append(run_corruption(matches[0], corruption, out_dir / "corruptions"))
    return results
