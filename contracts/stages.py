"""Call signatures of every pipeline stage.

Each owning agent exposes the function(s) matching the Protocols below, at the import
path listed in STAGE_ENTRYPOINTS. The pipeline runner (pipeline/runner.py, owned by
backend-devops) is the ONLY code allowed to import more than one stage module.

Pipeline order and branching
----------------------------
    image   = load_image(path)
    pre     = preprocess(image, settings)
    cls     = classify(pre, settings)
    palette = quantize(pre, cls, settings)                          # always runs
    lines   = extract_lines(pre, cls, settings)                     # LINE_ART, MIXED, or line_mode == CENTERLINE
              else None
    doc     = vectorize(pre, cls, palette, lines, settings)
    svg     = assemble_svg(doc, settings)
    bundle  = export(svg, doc, settings, out_dir)
    report  = evaluate(pre, palette, lines, doc, bundle, elapsed_s)
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from contracts.schemas import (
    ExportBundle,
    ImageClass,
    ImageInput,
    LineMap,
    Palette,
    PreprocessResult,
    QualityReport,
    Settings,
    VectorDocument,
)


class LoadImage(Protocol):
    def __call__(self, path: Path) -> ImageInput:
        """Validate (magic bytes, size, decodability) and describe an uploaded file.
        Raises InvalidImageError."""
        ...


class Preprocess(Protocol):
    def __call__(self, image: ImageInput, settings: Settings) -> PreprocessResult:
        """Decode, convert to sRGB, split alpha, denoise/deblock, optionally rescale,
        and (if settings.remove_background) make the background transparent."""
        ...


class Classify(Protocol):
    def __call__(self, pre: PreprocessResult, settings: Settings) -> ImageClass:
        """Detect LINE_ART / FLAT_COLOR / MIXED. If settings.mode != AUTO, return that
        label with confidence=1.0 and forced=True."""
        ...


class Quantize(Protocol):
    def __call__(self, pre: PreprocessResult, image_class: ImageClass, settings: Settings) -> Palette:
        """Quantize opaque pixels in LAB; transparent pixels get label -1.
        Honours settings.palette_override."""
        ...


class ExtractLines(Protocol):
    def __call__(self, pre: PreprocessResult, image_class: ImageClass, settings: Settings) -> LineMap:
        """Extract stroke mask, skeleton and stroke widths."""
        ...


class Vectorize(Protocol):
    def __call__(
        self,
        pre: PreprocessResult,
        image_class: ImageClass,
        palette: Palette,
        line_map: LineMap | None,
        settings: Settings,
    ) -> VectorDocument:
        """Trace regions/strokes into Bezier paths in SOURCE space, stacked layers."""
        ...


class AssembleSvg(Protocol):
    def __call__(self, doc: VectorDocument, settings: Settings) -> str:
        """Serialize to an optimized, valid SVG 1.1 string with one <g> layer per VectorLayer."""
        ...


class Export(Protocol):
    def __call__(self, svg: str, doc: VectorDocument, settings: Settings, out_dir: Path) -> ExportBundle:
        """Write output.svg, preview.png (always) and requested .ai/.eps into out_dir."""
        ...


class Evaluate(Protocol):
    def __call__(
        self,
        pre: PreprocessResult,
        palette: Palette,
        line_map: LineMap | None,
        doc: VectorDocument,
        bundle: ExportBundle,
        processing_time_s: float,
    ) -> QualityReport:
        """Compute SSIM, Delta-E, gap ratio, alpha IoU, node count, size and threshold checks."""
        ...


STAGE_ENTRYPOINTS: dict[str, str] = {
    "load_image": "pipeline.preprocess:load_image",
    "preprocess": "pipeline.preprocess:preprocess",
    "classify": "pipeline.classify:classify",
    "quantize": "pipeline.quantize:quantize",
    "extract_lines": "pipeline.lines:extract_lines",
    "vectorize": "pipeline.vectorize:vectorize",
    "assemble_svg": "pipeline.assemble:assemble_svg",
    "export": "pipeline.export:export",
    "evaluate": "eval.evaluate:evaluate",
}
