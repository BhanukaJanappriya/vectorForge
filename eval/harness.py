"""Run samples through the pipeline, falling back to oracle stubs for missing stages.

Stages are resolved lazily from ``contracts.stages.STAGE_ENTRYPOINTS`` at run time (never at
import time). A stage whose module or function does not exist yet is reported as
``SKIPPED (stage missing)``; one that fails to import or raises is reported as ``ERROR`` /
``FAILED``. In every one of these cases the oracle stand-in from :mod:`eval.oracle` is used
so the downstream stages and the metrics still run.
"""

from __future__ import annotations

import importlib
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from contracts.schemas import (
    ExportBundle,
    ImageClass,
    ImageClassLabel,
    ImageInput,
    LineMap,
    LineMode,
    Palette,
    PreprocessResult,
    QualityReport,
    QualityThresholds,
    Settings,
    VectorDocument,
)
from contracts.stages import STAGE_ENTRYPOINTS
from eval.evaluate import (
    RegionDeltaE,
    delta_e,
    evaluate_detailed,
    pixel_delta_e_p95_for,
    rgb_to_lab,
    svg_problems,
)
from eval.oracle import (
    oracle_classify,
    oracle_document,
    oracle_export,
    oracle_load_image,
    oracle_preprocess,
    oracle_quantize,
    oracle_svg,
)
from eval.raster import hex_to_rgb

PIPELINE_STAGES = [
    "load_image",
    "preprocess",
    "classify",
    "quantize",
    "extract_lines",
    "vectorize",
    "assemble_svg",
    "export",
]

OK = "OK"
MISSING = "SKIPPED (stage missing)"
IMPORT_ERROR = "ERROR (import failed)"
FAILED = "FAILED (raised)"
NOT_NEEDED = "N/A (not needed)"
ORACLE = "ORACLE (forced)"

SAMPLE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass
class StageRun:
    """How one stage was executed for one sample."""

    status: str
    source: str
    """'pipeline', 'oracle' or 'none'."""
    seconds: float = 0.0
    detail: str = ""


@dataclass
class SampleResult:
    """Everything the harness learned about one sample."""

    sample: str
    path: str
    truth: dict[str, Any]
    stages: dict[str, StageRun] = field(default_factory=dict)
    image_class: str | None = None
    n_colors: int | None = None
    report: QualityReport | None = None
    regions: list[RegionDeltaE] = field(default_factory=list)
    ground_truth: dict[str, Any] = field(default_factory=dict)
    svg_problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    out_dir: str = ""
    svg_path: str | None = None
    preview_path: str | None = None
    eval_seconds: float = 0.0
    stub_seconds: float = 0.0
    p95_pixel_delta_e: float | None = None
    """Diagnostic only ('p95 px dE (diag)'); not a MetricCheck and not in QualityReport."""

    @property
    def passed(self) -> bool:
        return self.report is not None and self.report.passed and self.error is None

    @property
    def real_stages(self) -> int:
        return sum(1 for s in self.stages.values() if s.source == "pipeline")

    def to_json(self) -> dict[str, Any]:
        return {
            "sample": self.sample,
            "path": self.path,
            "expected_class": self.truth.get("expected_class"),
            "image_class": self.image_class,
            "n_colors": self.n_colors,
            "passed": self.passed,
            "stages": {k: asdict(v) for k, v in self.stages.items()},
            "report": None if self.report is None else self.report.model_dump(mode="json"),
            "regions": [asdict(r) for r in self.regions],
            "ground_truth": self.ground_truth,
            "svg_problems": self.svg_problems,
            "warnings": self.warnings,
            "error": self.error,
            "svg_path": self.svg_path,
            "preview_path": self.preview_path,
            "eval_seconds": self.eval_seconds,
            "stub_seconds": self.stub_seconds,
            "diagnostics": {"p95_pixel_delta_e": self.p95_pixel_delta_e},
        }


# --------------------------------------------------------------------------------------
# Stage resolution
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolution:
    fn: Callable[..., Any] | None
    status: str
    detail: str = ""


def resolve_stage(name: str) -> Resolution:
    """Import the stage entrypoint named in STAGE_ENTRYPOINTS; never raises."""
    target = STAGE_ENTRYPOINTS[name]
    module_name, attr = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and (module_name == exc.name or module_name.startswith(exc.name + ".")):
            return Resolution(None, MISSING, f"module {module_name} not found")
        return Resolution(None, IMPORT_ERROR, f"{module_name}: {exc}")
    except Exception as exc:  # noqa: BLE001 - a half-written module must not crash the harness
        return Resolution(None, IMPORT_ERROR, f"{module_name}: {type(exc).__name__}: {exc}")
    fn = getattr(module, attr, None)
    if not callable(fn):
        return Resolution(None, MISSING, f"{target} not defined")
    return Resolution(fn, OK, target)


def _call(
    result: SampleResult,
    name: str,
    real: Resolution | None,
    expected: type | tuple[type, ...],
    real_args: tuple[Any, ...],
    oracle: Callable[[], Any] | None,
) -> Any:
    """Run the real stage if resolvable, else (or on failure) the oracle; record the outcome."""
    status, detail = (real.status, real.detail) if real is not None else (ORACLE, "oracle forced")
    if real is not None and real.fn is not None:
        t0 = time.perf_counter()
        try:
            value = real.fn(*real_args)
            if not isinstance(value, expected):
                raise TypeError(f"returned {type(value).__name__}, expected {expected}")
            result.stages[name] = StageRun(OK, "pipeline", time.perf_counter() - t0, real.detail)
            return value
        except Exception as exc:  # noqa: BLE001 - report and fall back
            status = FAILED
            detail = f"{type(exc).__name__}: {exc}".splitlines()[0][:300]
    if oracle is None:
        result.stages[name] = StageRun(status, "none", 0.0, detail)
        return None
    t0 = time.perf_counter()
    value = oracle()
    result.stages[name] = StageRun(status, "oracle", time.perf_counter() - t0, detail)
    return value


# --------------------------------------------------------------------------------------
# Ground truth comparison
# --------------------------------------------------------------------------------------


def _blend_explanation(rgb: np.ndarray, gt_rgb: np.ndarray, gt_hex: list[str]) -> str | None:
    """If ``rgb`` is (within Delta-E 3) an sRGB mix of two GT colors, name them.

    Anti-aliasing (box-filter downsampling) blends linearly in sRGB, so the projection
    onto each GT pair is done in RGB and the residual is measured with CIEDE2000.
    """
    best: tuple[float, str] | None = None
    c = rgb.astype(np.float64)
    for i in range(len(gt_hex)):
        for j in range(i + 1, len(gt_hex)):
            a, b = gt_rgb[i].astype(np.float64), gt_rgb[j].astype(np.float64)
            ab = b - a
            t = float(np.clip(np.dot(c - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0.0, 1.0))
            mix = np.clip(np.round(a + t * ab), 0, 255).astype(np.uint8)
            d = float(delta_e(rgb_to_lab(rgb[None].astype(np.uint8))[0], rgb_to_lab(mix[None])[0]))
            if 0.05 < t < 0.95 and (best is None or d < best[0]):
                best = (d, f"AA blend of {gt_hex[i]}/{gt_hex[j]} (t={t:.2f})")
    return best[1] if best and best[0] < QualityThresholds.MAX_DELTA_E else None


def compare_palette(truth_hex: list[str], palette: Palette) -> dict[str, Any]:
    """Match ground-truth colors to palette colors with CIEDE2000 (tolerance MAX_DELTA_E)."""
    tol = QualityThresholds.MAX_DELTA_E
    gt_rgb = np.array([hex_to_rgb(h) for h in truth_hex], dtype=np.uint8)
    gt_lab = rgb_to_lab(gt_rgb)
    pal_hex = [c.hex for c in palette.colors]
    pal_rgb = np.array([c.rgb for c in palette.colors], dtype=np.uint8)
    pal_lab = rgb_to_lab(pal_rgb)
    d = delta_e(gt_lab[:, None, :], pal_lab[None, :, :])
    gt_best = d.min(axis=1)
    pal_best = d.min(axis=0)
    unmatched = [truth_hex[i] for i in np.flatnonzero(gt_best >= tol)]
    extra = []
    for j in np.flatnonzero(pal_best >= tol):
        why = _blend_explanation(pal_rgb[j], gt_rgb, truth_hex)
        extra.append({"hex": pal_hex[j], "pixels": palette.colors[j].pixel_count, "explanation": why})
    return {
        "expected_n": len(truth_hex),
        "got_n": len(pal_hex),
        "max_gt_delta_e": float(gt_best.max()),
        "unmatched_gt": unmatched,
        "extra": extra,
        "match": not unmatched and not extra,
    }


def expected_stroke_width(truth: dict[str, Any]) -> float | None:
    """Ground-truth median stroke width (explicit, else median of the listed widths)."""
    if truth.get("median_stroke_width_px") is not None:
        return float(truth["median_stroke_width_px"])
    widths = truth.get("stroke_widths_px")
    if widths:
        return float(np.median(list(widths.values())))
    return None


def compare_ground_truth(
    truth: dict[str, Any],
    image_class: ImageClass,
    palette: Palette,
    line_map: LineMap | None,
    pre: PreprocessResult,
    stages: dict[str, StageRun],
) -> dict[str, Any]:
    """Class, palette and stroke-width comparison against samples/*.json."""
    out: dict[str, Any] = {
        "class": {
            "expected": truth["expected_class"],
            "got": image_class.label.value,
            "match": image_class.label.value == truth["expected_class"],
            "source": stages["classify"].source,
        }
    }
    if truth.get("palette_hex"):
        out["palette"] = compare_palette(truth["palette_hex"], palette) | {"source": stages["quantize"].source}
    else:
        out["palette"] = {"match": None, "note": "no exact ground-truth palette", "got_n": len(palette.colors)}
    gt_width = expected_stroke_width(truth)
    if gt_width is None:
        out["strokes"] = {"match": None, "note": "no ground-truth stroke width"}
    elif line_map is None:
        out["strokes"] = {"match": None, "expected": gt_width, "note": "no LineMap (extract_lines not run)"}
    else:
        got = float(line_map.median_stroke_width) / pre.scale_factor
        tol = max(1.0, 0.25 * gt_width)
        out["strokes"] = {
            "expected": gt_width,
            "got": got,
            "tolerance": tol,
            "match": abs(got - gt_width) <= tol,
            "source": stages["extract_lines"].source,
        }
    return out


# --------------------------------------------------------------------------------------
# Running samples
# --------------------------------------------------------------------------------------


def list_samples(samples_dir: Path) -> list[Path]:
    """Sample images that have a ground-truth JSON beside them, sorted by name."""
    return sorted(
        p for p in samples_dir.iterdir() if p.suffix.lower() in SAMPLE_SUFFIXES and p.with_suffix(".json").is_file()
    )


def find_sample(samples_dir: Path, key: str) -> Path:
    """Match by stem ('01_logo_4color'), file name, or numeric prefix ('01', '1')."""
    samples = list_samples(samples_dir)
    for p in samples:
        if key in (p.stem, p.name):
            return p
    prefix = key.zfill(2) + "_" if key.isdigit() else None
    matches = [p for p in samples if (prefix and p.name.startswith(prefix)) or p.stem.startswith(key)]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"no unique sample matches {key!r}; available: {[p.stem for p in samples]}")


def run_sample(
    path: Path, out_dir: Path, settings: Settings | None = None, *, use_pipeline: bool = True
) -> SampleResult:
    """Convert one sample (real stages where available, oracle otherwise) and evaluate it."""
    settings = settings or Settings()
    truth = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    result = SampleResult(sample=path.stem, path=str(path), truth=truth, out_dir=str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    def res(name: str) -> Resolution | None:
        return resolve_stage(name) if use_pipeline else None

    try:
        image: ImageInput = _call(
            result, "load_image", res("load_image"), ImageInput, (path,), lambda: oracle_load_image(path)
        )
        pre: PreprocessResult = _call(
            result,
            "preprocess",
            res("preprocess"),
            PreprocessResult,
            (image, settings),
            lambda: oracle_preprocess(image, settings),
        )
        cls: ImageClass = _call(
            result,
            "classify",
            res("classify"),
            ImageClass,
            (pre, settings),
            lambda: oracle_classify(truth["expected_class"]),
        )
        result.image_class = cls.label.value
        palette: Palette = _call(
            result,
            "quantize",
            res("quantize"),
            Palette,
            (pre, cls, settings),
            lambda: oracle_quantize(pre, truth.get("palette_hex")),
        )
        result.n_colors = len(palette.colors)
        needs_lines = cls.label in (ImageClassLabel.LINE_ART, ImageClassLabel.MIXED) or (
            settings.line_mode == LineMode.CENTERLINE
        )
        line_map: LineMap | None = None
        if needs_lines:
            line_map = _call(result, "extract_lines", res("extract_lines"), LineMap, (pre, cls, settings), None)
            if line_map is None:
                result.stages["extract_lines"].detail += "; no oracle, line_map=None"
        else:
            result.stages["extract_lines"] = StageRun(NOT_NEEDED, "none", 0.0, f"class {cls.label.value}")
        doc: VectorDocument = _call(
            result,
            "vectorize",
            res("vectorize"),
            VectorDocument,
            (pre, cls, palette, line_map, settings),
            lambda: oracle_document(pre, palette, cls.label, settings=settings),
        )
        svg: str = _call(result, "assemble_svg", res("assemble_svg"), str, (doc, settings), lambda: oracle_svg(doc))
        export_dir = out_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        bundle: ExportBundle = _call(
            result,
            "export",
            res("export"),
            ExportBundle,
            (svg, doc, settings, export_dir),
            lambda: oracle_export(svg, doc, export_dir),
        )
        result.svg_path, result.preview_path = str(bundle.svg_path), str(bundle.preview_png_path)
        result.warnings = list(bundle.warnings)
        # The time check covers pipeline stages only; oracle stub time is reported separately.
        elapsed = sum(s.seconds for s in result.stages.values() if s.source == "pipeline")
        result.stub_seconds = sum(s.seconds for s in result.stages.values() if s.source == "oracle")
        t0 = time.perf_counter()
        result.report, result.regions = evaluate_detailed(pre, palette, line_map, doc, bundle, elapsed)
        result.eval_seconds = time.perf_counter() - t0
        result.p95_pixel_delta_e = pixel_delta_e_p95_for(pre, bundle)
        result.svg_problems = svg_problems(bundle.svg_path, doc)
        result.ground_truth = compare_ground_truth(truth, cls, palette, line_map, pre, result.stages)
    except Exception as exc:  # noqa: BLE001 - one broken sample must not stop the run
        result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
    return result
