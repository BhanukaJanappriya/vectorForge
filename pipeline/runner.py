"""Pipeline composition root: the ONLY module that imports and chains the stages.

Stages are resolved lazily from :data:`contracts.stages.STAGE_ENTRYPOINTS` (via importlib)
when :func:`run_pipeline` is called, never at import time, so this module imports cleanly
while individual stages are still being written. Callers (the API worker, the eval CLI)
may override any stage through the ``stages`` mapping, e.g. to inject stand-ins for a
stage that does not exist yet or fakes in tests.

Order and branching follow ``contracts/stages.py``::

    image   = load_image(path)
    pre     = preprocess(image, settings)
    cls     = classify(pre, settings)
    palette = quantize(pre, cls, settings)
    lines   = extract_lines(pre, cls, settings)   # LINE_ART, MIXED or line_mode == CENTERLINE, else None
    doc     = vectorize(pre, cls, palette, lines, settings)
    svg     = assemble_svg(doc, settings)
    bundle  = export(svg, doc, settings, out_dir)
    report  = evaluate(pre, palette, lines, doc, bundle, elapsed_s)

Any exception raised by a stage is wrapped in :class:`PipelineError`, whose ``detail`` is
the HTTP-facing :class:`contracts.api.ErrorDetail` (``code`` = ``VectorForgeError.code``).
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contracts.api import ErrorDetail, PipelineStage
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
    Settings,
    StageError,
    VectorDocument,
    VectorForgeError,
)
from contracts.stages import STAGE_ENTRYPOINTS

StageFn = Callable[..., Any]
ProgressCallback = Callable[[PipelineStage, float], None]

STAGE_ORDER: tuple[str, ...] = (
    "load_image",
    "preprocess",
    "classify",
    "quantize",
    "extract_lines",
    "vectorize",
    "assemble_svg",
    "export",
    "evaluate",
)
"""Stage names (keys of STAGE_ENTRYPOINTS) in execution order."""

STAGE_TO_PIPELINE_STAGE: dict[str, PipelineStage] = {
    "load_image": PipelineStage.PREPROCESS,
    "preprocess": PipelineStage.PREPROCESS,
    "classify": PipelineStage.CLASSIFY,
    "quantize": PipelineStage.QUANTIZE,
    "extract_lines": PipelineStage.EXTRACT_LINES,
    "vectorize": PipelineStage.VECTORIZE,
    "assemble_svg": PipelineStage.ASSEMBLE,
    "export": PipelineStage.EXPORT,
    "evaluate": PipelineStage.EVALUATE,
}
"""Maps a stage function name onto the coarser API progress stage."""

STAGE_PROGRESS: dict[str, float] = {
    "load_image": 0.0,
    "preprocess": 0.02,
    "classify": 0.10,
    "quantize": 0.15,
    "extract_lines": 0.30,
    "vectorize": 0.40,
    "assemble_svg": 0.70,
    "export": 0.75,
    "evaluate": 0.90,
}
"""Overall progress fraction reported when each stage STARTS (rough share of run time)."""

_EXPECTED_TYPES: dict[str, type | tuple[type, ...]] = {
    "load_image": ImageInput,
    "preprocess": PreprocessResult,
    "classify": ImageClass,
    "quantize": Palette,
    "extract_lines": LineMap,
    "vectorize": VectorDocument,
    "assemble_svg": str,
    "export": ExportBundle,
    "evaluate": QualityReport,
}


@dataclass(frozen=True)
class PipelineResult:
    """Every contract object produced by one successful run, plus timings."""

    image: ImageInput
    pre: PreprocessResult
    image_class: ImageClass
    palette: Palette
    line_map: LineMap | None
    doc: VectorDocument
    svg: str
    bundle: ExportBundle
    report: QualityReport
    timings: dict[str, float]
    """Wall-clock seconds per stage name (keys of STAGE_ENTRYPOINTS) that actually ran."""
    elapsed_s: float
    """Wall-clock seconds for the full run, including evaluation."""
    processing_time_s: float
    """Seconds from load_image through export (the value passed to evaluate)."""


class PipelineError(VectorForgeError):
    """A run failed. ``detail`` is ready for the API; ``__cause__`` is the original exception."""

    def __init__(
        self,
        detail: ErrorDetail,
        stage_name: str,
        timings: dict[str, float],
        partial: dict[str, Any],
    ) -> None:
        super().__init__(detail.message)
        self.detail = detail
        self.code = detail.code
        self.stage_name = stage_name
        """Stage function name (key of STAGE_ENTRYPOINTS) that failed."""
        self.timings = timings
        self.partial = partial
        """Outputs of the stages that completed, keyed by stage name."""


def needs_lines(image_class: ImageClass, settings: Settings) -> bool:
    """Whether extract_lines runs: LINE_ART, MIXED, or centerline mode requested."""
    return (
        image_class.label in (ImageClassLabel.LINE_ART, ImageClassLabel.MIXED)
        or settings.line_mode == LineMode.CENTERLINE
    )


def resolve_stage(name: str) -> StageFn:
    """Import the callable for stage ``name`` from STAGE_ENTRYPOINTS.

    Raises StageError(name, ...) (never ImportError) if the module or function is missing or
    the module fails to import.
    """
    try:
        target = STAGE_ENTRYPOINTS[name]
    except KeyError:
        raise StageError(name, f"unknown stage {name!r}") from None
    module_name, attr = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and (module_name == exc.name or module_name.startswith(exc.name + ".")):
            raise StageError(name, f"stage module {module_name} is not available") from exc
        raise StageError(name, f"stage module {module_name} failed to import: {exc}") from exc
    except Exception as exc:  # a broken stage module must surface as a stage failure
        raise StageError(name, f"stage module {module_name} failed to import: {type(exc).__name__}: {exc}") from exc
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise StageError(name, f"stage entrypoint {target} is not defined")
    return fn


def resolve_stages(overrides: Mapping[str, StageFn] | None = None) -> dict[str, StageFn]:
    """Resolve every stage, preferring ``overrides``. Raises StageError for the first missing one."""
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(STAGE_ORDER))
    if unknown:
        raise StageError(unknown[0], f"unknown stage override(s): {', '.join(unknown)}")
    return {name: overrides[name] if name in overrides else resolve_stage(name) for name in STAGE_ORDER}


def error_detail(exc: BaseException, stage_name: str | None) -> ErrorDetail:
    """Map any exception raised while running ``stage_name`` onto the API ErrorDetail."""
    stage = STAGE_TO_PIPELINE_STAGE.get(stage_name) if stage_name else None
    if isinstance(exc, VectorForgeError):
        message = str(exc) or type(exc).__name__
        return ErrorDetail(code=exc.code, message=message, stage=stage)
    return ErrorDetail(code="internal_error", message=f"{type(exc).__name__}: {exc}", stage=stage)


@dataclass
class _Run:
    """Mutable state of one run (timings, partial outputs, progress reporting)."""

    on_progress: ProgressCallback | None
    timings: dict[str, float] = field(default_factory=dict)
    partial: dict[str, Any] = field(default_factory=dict)

    def report(self, stage: PipelineStage, fraction: float) -> None:
        if self.on_progress is not None:
            self.on_progress(stage, fraction)

    def call(self, name: str, fn: StageFn, *args: Any) -> Any:
        """Run one stage: report progress, time it, type-check its output, wrap errors."""
        self.report(STAGE_TO_PIPELINE_STAGE[name], STAGE_PROGRESS[name])
        t0 = time.perf_counter()
        try:
            value = fn(*args)
            expected = _EXPECTED_TYPES[name]
            if not isinstance(value, expected):
                raise StageError(name, f"returned {type(value).__name__}, expected {_type_name(expected)}")
        except Exception as exc:
            self.timings[name] = time.perf_counter() - t0
            raise PipelineError(error_detail(exc, name), name, dict(self.timings), dict(self.partial)) from exc
        self.timings[name] = time.perf_counter() - t0
        self.partial[name] = value
        return value


def _type_name(expected: type | tuple[type, ...]) -> str:
    if isinstance(expected, tuple):
        return " | ".join(t.__name__ for t in expected)
    return expected.__name__


def run_pipeline(
    path: Path,
    settings: Settings | None,
    out_dir: Path,
    *,
    on_progress: ProgressCallback | None = None,
    stages: Mapping[str, StageFn] | None = None,
) -> PipelineResult:
    """Convert the image at ``path`` into ``out_dir`` and evaluate the result.

    Args:
        path: Uploaded PNG/JPEG file.
        settings: Conversion settings; None means defaults.
        out_dir: Directory for export outputs (created if missing).
        on_progress: Called with (stage, overall fraction in [0, 1]) as each stage starts,
            and with (DONE, 1.0) at the end.
        stages: Optional stage-name -> callable overrides for STAGE_ENTRYPOINTS.

    Raises:
        PipelineError: any stage failed (including a missing stage module), with ``detail``
            mapped from the original exception.
    """
    settings = settings or Settings()
    run = _Run(on_progress)
    t_start = time.perf_counter()
    try:
        fns = resolve_stages(stages)
    except StageError as exc:
        name = exc.stage if exc.stage in STAGE_TO_PIPELINE_STAGE else None
        raise PipelineError(error_detail(exc, name), exc.stage, {}, {}) from exc
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PipelineError(error_detail(exc, "export"), "export", {}, {}) from exc

    image: ImageInput = run.call("load_image", fns["load_image"], Path(path))
    pre: PreprocessResult = run.call("preprocess", fns["preprocess"], image, settings)
    image_class: ImageClass = run.call("classify", fns["classify"], pre, settings)
    palette: Palette = run.call("quantize", fns["quantize"], pre, image_class, settings)
    line_map: LineMap | None = None
    if needs_lines(image_class, settings):
        line_map = run.call("extract_lines", fns["extract_lines"], pre, image_class, settings)
    doc: VectorDocument = run.call("vectorize", fns["vectorize"], pre, image_class, palette, line_map, settings)
    svg: str = run.call("assemble_svg", fns["assemble_svg"], doc, settings)
    bundle: ExportBundle = run.call("export", fns["export"], svg, doc, settings, Path(out_dir))
    processing_time_s = time.perf_counter() - t_start
    report: QualityReport = run.call(
        "evaluate", fns["evaluate"], pre, palette, line_map, doc, bundle, processing_time_s
    )
    elapsed_s = time.perf_counter() - t_start
    run.report(PipelineStage.DONE, 1.0)
    return PipelineResult(
        image=image,
        pre=pre,
        image_class=image_class,
        palette=palette,
        line_map=line_map,
        doc=doc,
        svg=svg,
        bundle=bundle,
        report=report,
        timings=dict(run.timings),
        elapsed_s=elapsed_s,
        processing_time_s=processing_time_s,
    )
