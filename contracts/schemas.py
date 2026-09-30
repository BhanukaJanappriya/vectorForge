"""VectorForge shared pipeline contracts.

Every pipeline module communicates ONLY through the types defined here. Modules must
not import each other. Changing anything in this file requires Orchestrator approval
(see CLAUDE.md -> Rules) and a SCHEMA_VERSION bump.

Coordinate systems
------------------
* "Source space": pixel grid of the original uploaded image (ImageInput.width x height).
* "Processing space": pixel grid of all arrays inside PreprocessResult, Palette and
  LineMap. processing_size = round(source_size * PreprocessResult.scale_factor).
* VectorDocument paths are ALWAYS in source space; viewBox = "0 0 W H" of the source.
  The vectorizer is responsible for undoing scale_factor.

Array conventions
-----------------
* Arrays are row-major, shape (H, W[, C]), origin top-left.
* Arrays stored in contract objects are read-only views. Copy before mutating.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainValidator,
    computed_field,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "1.0.0"

# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class VectorForgeError(Exception):
    """Base class for all errors raised by pipeline modules."""

    code: str = "internal_error"


class InvalidImageError(VectorForgeError):
    """Input file is unreadable, unsupported, too large, or corrupt."""

    code = "invalid_image"


class StageError(VectorForgeError):
    """A pipeline stage failed on otherwise valid input."""

    code = "stage_failed"

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


class ExportToolUnavailableError(VectorForgeError):
    """An external export tool (Inkscape, potrace, ...) is missing at runtime."""

    code = "export_tool_unavailable"


# --------------------------------------------------------------------------------------
# NumPy field helpers
# --------------------------------------------------------------------------------------


def _ndarray(dtypes: tuple[type, ...], ndim: int, channels: int | None = None) -> Any:
    """Build an annotated ndarray type that checks dtype/ndim and stores a read-only view."""

    allowed = tuple(np.dtype(d) for d in dtypes)

    def check(value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"expected numpy.ndarray, got {type(value).__name__}")
        if value.dtype not in allowed:
            raise ValueError(f"dtype {value.dtype} not in {[str(d) for d in allowed]}")
        if value.ndim != ndim:
            raise ValueError(f"expected ndim={ndim}, got shape {value.shape}")
        if channels is not None and value.shape[-1] != channels:
            raise ValueError(f"expected {channels} channels, got shape {value.shape}")
        if value.size == 0:
            raise ValueError("array must not be empty")
        view = value.view()
        view.flags.writeable = False
        return view

    return Annotated[np.ndarray, PlainValidator(check)]


RGBImage = _ndarray((np.uint8,), ndim=3, channels=3)
"""(H, W, 3) uint8 sRGB."""
AlphaMask = _ndarray((np.uint8,), ndim=2)
"""(H, W) uint8, 0 = fully transparent, 255 = opaque."""
BoolMask = _ndarray((np.bool_,), ndim=2)
"""(H, W) bool."""
LabelMap = _ndarray((np.int32,), ndim=2)
"""(H, W) int32 palette indices; -1 = transparent / not assigned."""
FloatMap = _ndarray((np.float32,), ndim=2)
"""(H, W) float32."""

HEX_RE = re.compile(r"^#[0-9a-f]{6}$")
XML_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
PATH_D_RE = re.compile(r"^\s*M[MLCZ0-9eE.,+\-\s]*$")
"""VectorDocument paths use ONLY absolute M, L, C, Z so they map 1:1 onto PDF/EPS operators
(m, l, c, h). The SVG optimizer may rewrite them to relative/short forms in output.svg."""

RGB = tuple[
    Annotated[int, Field(ge=0, le=255)],
    Annotated[int, Field(ge=0, le=255)],
    Annotated[int, Field(ge=0, le=255)],
]
LAB = tuple[float, float, float]


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Return the canonical lowercase '#rrggbb' form of an RGB triple."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _check_hex(value: str) -> str:
    if not HEX_RE.match(value):
        raise ValueError(f"color must be lowercase '#rrggbb', got {value!r}")
    return value


HexColor = Annotated[str, AfterValidator(_check_hex)]


class _Contract(BaseModel):
    """Base config: immutable, strict about unknown fields, numpy-friendly."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


# --------------------------------------------------------------------------------------
# Enums & user settings
# --------------------------------------------------------------------------------------


class ImageMode(StrEnum):
    """Pillow mode of the decoded source file."""

    L = "L"
    LA = "LA"
    P = "P"
    RGB = "RGB"
    RGBA = "RGBA"
    CMYK = "CMYK"


class SourceFormat(StrEnum):
    PNG = "png"
    JPEG = "jpeg"


class ProcessingMode(StrEnum):
    AUTO = "auto"
    LINE_ART = "line_art"
    FLAT_COLOR = "flat_color"
    MIXED = "mixed"


class ImageClassLabel(StrEnum):
    LINE_ART = "line_art"
    FLAT_COLOR = "flat_color"
    MIXED = "mixed"


class DetailLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LineMode(StrEnum):
    OUTLINE = "outline"
    CENTERLINE = "centerline"


class OutputFormat(StrEnum):
    SVG = "svg"
    AI = "ai"
    EPS = "eps"
    PNG = "png"


class DetailPreset(_Contract):
    """Numeric parameters a DetailLevel maps to. Shared so every module agrees."""

    speckle_min_area_px: int = Field(
        ge=0, description="Regions smaller than this (processing px) are merged into a neighbour."
    )
    path_precision: int = Field(ge=0, le=4, description="Decimal places for SVG coordinates.")
    simplify_tolerance_px: float = Field(gt=0, description="Max deviation (source px) allowed when removing nodes.")
    corner_threshold_deg: float = Field(
        gt=0, lt=180, description="Angle below which a vertex is kept as a sharp corner."
    )


DETAIL_PRESETS: dict[DetailLevel, DetailPreset] = {
    DetailLevel.LOW: DetailPreset(
        speckle_min_area_px=64, path_precision=2, simplify_tolerance_px=1.5, corner_threshold_deg=60.0
    ),
    DetailLevel.MEDIUM: DetailPreset(
        speckle_min_area_px=16, path_precision=2, simplify_tolerance_px=0.8, corner_threshold_deg=60.0
    ),
    DetailLevel.HIGH: DetailPreset(
        speckle_min_area_px=4, path_precision=2, simplify_tolerance_px=0.4, corner_threshold_deg=45.0
    ),
}


class Settings(_Contract):
    """User-controllable conversion settings (API body + UI form)."""

    mode: ProcessingMode = ProcessingMode.AUTO
    max_colors: int | None = Field(default=None, ge=2, le=64, description="None = auto-detect.")
    detail_level: DetailLevel = DetailLevel.MEDIUM
    line_mode: LineMode = LineMode.OUTLINE
    smoothing: int = Field(default=50, ge=0, le=100, description="0 = polygonal, 100 = maximally smooth curves.")
    remove_background: bool = Field(
        default=False,
        description="Preprocess makes the detected background transparent (PreprocessResult.background_removed).",
    )
    palette_override: list[HexColor] | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="User-edited palette for a re-run (UI edit/merge). When set, quantize assigns every opaque pixel "
        "to the nearest of these colors in LAB and skips auto-detection; max_colors is ignored.",
    )
    output_formats: list[OutputFormat] = Field(default_factory=lambda: list(OutputFormat))

    @field_validator("palette_override")
    @classmethod
    def _dedupe_palette(cls, value: list[str] | None) -> list[str] | None:
        return None if value is None else list(dict.fromkeys(value))

    @field_validator("output_formats")
    @classmethod
    def _svg_always_first(cls, value: list[OutputFormat]) -> list[OutputFormat]:
        """SVG is the canonical artefact: always produced, de-duplicated, stable order."""
        wanted = set(value) | {OutputFormat.SVG}
        return [f for f in OutputFormat if f in wanted]

    @property
    def preset(self) -> DetailPreset:
        return DETAIL_PRESETS[self.detail_level]


# --------------------------------------------------------------------------------------
# Stage 0: ingest
# --------------------------------------------------------------------------------------


class ImageInput(_Contract):
    """Metadata of the uploaded source file (pixels are loaded by Preprocess)."""

    path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    has_alpha: bool = Field(description="True only if the file has an alpha channel with at least one pixel < 255.")
    mode: ImageMode
    source_format: SourceFormat
    file_size_bytes: int = Field(gt=0)

    @property
    def filename(self) -> str:
        return self.path.name


# --------------------------------------------------------------------------------------
# Stage 1: preprocess
# --------------------------------------------------------------------------------------


class DenoiseParams(_Contract):
    """Record of what preprocessing actually did (for reproducibility/debugging)."""

    method: Literal["none", "bilateral", "median", "nl_means", "edge_preserving"]
    strength: float = Field(ge=0, description="Method-specific strength (e.g. sigmaColor, h).")
    jpeg_deblock: bool = False
    extra: dict[str, float] = Field(default_factory=dict)


class PreprocessResult(_Contract):
    """Cleaned pixels in processing space."""

    source: ImageInput
    image: RGBImage
    alpha: AlphaMask | None = Field(description="None iff the source is opaque and no background was removed.")
    background_removed: bool = Field(
        default=False, description="True if remove_background made background pixels alpha=0."
    )
    scale_factor: float = Field(gt=0, description="processing_size / source_size.")
    denoise: DenoiseParams

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def opaque_mask(self) -> np.ndarray:
        """(H, W) bool: pixels that must be vectorized (alpha >= 128, or all if opaque)."""
        if self.alpha is None:
            return np.ones(self.image.shape[:2], dtype=bool)
        return self.alpha >= 128

    @model_validator(mode="after")
    def _consistent(self) -> PreprocessResult:
        if self.alpha is not None and self.alpha.shape != self.image.shape[:2]:
            raise ValueError(f"alpha shape {self.alpha.shape} != image shape {self.image.shape[:2]}")
        if (self.source.has_alpha or self.background_removed) != (self.alpha is not None):
            raise ValueError("alpha must be provided iff source.has_alpha or background_removed")
        for src, proc in ((self.source.width, self.width), (self.source.height, self.height)):
            if abs(round(src * self.scale_factor) - proc) > 1:
                raise ValueError(f"scale_factor {self.scale_factor} inconsistent with sizes {src}->{proc}")
        return self


# --------------------------------------------------------------------------------------
# Stage 2: classify
# --------------------------------------------------------------------------------------


class ImageClass(_Contract):
    label: ImageClassLabel
    confidence: float = Field(ge=0, le=1)
    forced: bool = Field(default=False, description="True when Settings.mode overrode auto-detection.")
    features: dict[str, float] = Field(default_factory=dict, description="Diagnostic features used to decide.")


# --------------------------------------------------------------------------------------
# Stage 3: color quantization
# --------------------------------------------------------------------------------------


class PaletteColor(_Contract):
    index: int = Field(ge=0)
    rgb: RGB
    lab: LAB = Field(description="CIELAB (D65, 2 deg) of rgb.")
    hex: HexColor
    pixel_count: int = Field(ge=0, description="Number of label_map pixels with this index.")
    is_background: bool = False

    @model_validator(mode="after")
    def _hex_matches_rgb(self) -> PaletteColor:
        if self.hex != rgb_to_hex(self.rgb):
            raise ValueError(f"hex {self.hex} does not match rgb {self.rgb}")
        return self


class Palette(_Contract):
    """Quantized colors + per-pixel assignment in processing space."""

    colors: list[PaletteColor] = Field(min_length=1, max_length=64)
    label_map: LabelMap

    @property
    def background_index(self) -> int | None:
        return next((c.index for c in self.colors if c.is_background), None)

    @model_validator(mode="after")
    def _consistent(self) -> Palette:
        if [c.index for c in self.colors] != list(range(len(self.colors))):
            raise ValueError("colors[i].index must equal i")
        if sum(c.is_background for c in self.colors) > 1:
            raise ValueError("at most one background color")
        lo, hi = int(self.label_map.min()), int(self.label_map.max())
        if lo < -1 or hi >= len(self.colors):
            raise ValueError(f"label_map values must be in [-1, {len(self.colors) - 1}], got [{lo}, {hi}]")
        counts = np.bincount(self.label_map[self.label_map >= 0].ravel(), minlength=len(self.colors))
        for c in self.colors:
            if int(counts[c.index]) != c.pixel_count:
                raise ValueError(f"color {c.index}: pixel_count {c.pixel_count} != label_map count {counts[c.index]}")
        return self


# --------------------------------------------------------------------------------------
# Stage 4: line extraction
# --------------------------------------------------------------------------------------


class LineMap(_Contract):
    """Stroke pixels in processing space."""

    mask: BoolMask = Field(description="True where a stroke covers the pixel.")
    skeleton: BoolMask = Field(description="1-px wide, 8-connected centerline; subset of mask.")
    width_map: FloatMap = Field(description="Local stroke width (px) at skeleton pixels, 0 elsewhere.")
    median_stroke_width: float = Field(ge=0)
    color_rgb: RGB = Field(description="Dominant stroke color.")

    @model_validator(mode="after")
    def _consistent(self) -> LineMap:
        if not (self.mask.shape == self.skeleton.shape == self.width_map.shape):
            raise ValueError("mask, skeleton and width_map must share a shape")
        if np.any(self.skeleton & ~self.mask):
            raise ValueError("skeleton must be a subset of mask")
        if np.any(self.width_map[~self.skeleton] != 0):
            raise ValueError("width_map must be 0 outside the skeleton")
        return self


# --------------------------------------------------------------------------------------
# Stage 5: vectorization
# --------------------------------------------------------------------------------------


def layer_name(role: Literal["background", "fill", "line"], number: int, color_hex: str) -> tuple[str, str]:
    """Canonical (id, name) for a layer, e.g. ('color_1_E53935', 'color_1_#E53935').

    number is 1-based in z-order. Roles map to prefixes: background/fill -> 'color', line -> 'line'.
    The name is what Illustrator/Inkscape show as the layer label.
    """
    prefix = "line" if role == "line" else "color"
    upper = _check_hex(color_hex)[1:].upper()
    return f"{prefix}_{number}_{upper}", f"{prefix}_{number}_#{upper}"


class VectorLayer(_Contract):
    """One color layer. Painted in ascending z_order (0 = bottom).

    Hairline-gap strategy: the vectorizer uses *stacking* -- lower layers extend under
    the layers above them (dilated 0.5-1 px), so anti-aliasing seams never expose the canvas.
    Use layer_name() for id/name.
    """

    id: str = Field(description="XML id, unique within the document.")
    name: str = Field(min_length=1, description="Human-readable name; becomes the Illustrator layer name.")
    role: Literal["background", "fill", "line"]
    color_hex: HexColor
    paths: list[str] = Field(min_length=1, description="SVG path 'd' strings in source space; absolute M/L/C/Z only.")
    z_order: int = Field(ge=0)
    is_stroke: bool = False
    stroke_width: float | None = Field(default=None, gt=0, description="Source px; required iff is_stroke.")
    fill_rule: Literal["nonzero", "evenodd"] = "evenodd"
    opacity: float = Field(default=1.0, ge=0, le=1)
    palette_index: int | None = Field(
        default=None, ge=0, description="Palette color this layer was traced from (for Delta-E)."
    )

    @field_validator("id")
    @classmethod
    def _xml_id(cls, value: str) -> str:
        if not XML_ID_RE.match(value):
            raise ValueError(f"invalid XML id {value!r}")
        return value

    @field_validator("paths")
    @classmethod
    def _path_syntax(cls, value: list[str]) -> list[str]:
        for d in value:
            if not PATH_D_RE.match(d):
                raise ValueError(f"invalid path data: {d[:60]!r}")
        return value

    @model_validator(mode="after")
    def _stroke(self) -> VectorLayer:
        if self.is_stroke != (self.stroke_width is not None):
            raise ValueError("stroke_width must be set iff is_stroke")
        return self


class DocumentMetadata(_Contract):
    source_filename: str
    image_class: ImageClassLabel
    settings: Settings
    palette_hex: list[HexColor] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    generator: str = "VectorForge"


class VectorDocument(_Contract):
    width: int = Field(gt=0, description="Source width in px.")
    height: int = Field(gt=0, description="Source height in px.")
    layers: list[VectorLayer] = Field(description="Sorted by z_order ascending.")
    metadata: DocumentMetadata

    @computed_field  # type: ignore[prop-decorator]
    @property
    def view_box(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, float(self.width), float(self.height))

    @model_validator(mode="after")
    def _consistent(self) -> VectorDocument:
        ids = [layer.id for layer in self.layers]
        if len(ids) != len(set(ids)):
            raise ValueError("layer ids must be unique")
        z = [layer.z_order for layer in self.layers]
        if z != sorted(z) or len(z) != len(set(z)):
            raise ValueError("layers must be sorted by strictly increasing z_order")
        return self


# --------------------------------------------------------------------------------------
# Stage 6/7: SVG assembly & export
# --------------------------------------------------------------------------------------


class ExportBundle(_Contract):
    """Files written to the job output directory. All paths must exist."""

    svg_path: Path
    preview_png_path: Path = Field(description="Always rendered (needed for QA), even if PNG not requested.")
    ai_path: Path | None = None
    eps_path: Path | None = None
    warnings: list[str] = Field(default_factory=list, description="e.g. 'inkscape unavailable, EPS skipped'.")

    @model_validator(mode="after")
    def _files_exist(self) -> ExportBundle:
        for name in ("svg_path", "preview_png_path", "ai_path", "eps_path"):
            p: Path | None = getattr(self, name)
            if p is not None and not p.is_file():
                raise ValueError(f"{name} does not exist: {p}")
        return self


# --------------------------------------------------------------------------------------
# Stage 8: quality evaluation
# --------------------------------------------------------------------------------------


class QualityThresholds:
    """Non-negotiable acceptance thresholds from the product spec."""

    MAX_DELTA_E: float = 3.0
    MAX_MEAN_DELTA_E: float = 2.0
    MAX_GAP_RATIO: float = 0.0005
    """Fraction of interior source-opaque pixels that render transparent in preview.png."""
    SSIM_MIN: dict[ImageClassLabel, float] = {
        ImageClassLabel.FLAT_COLOR: 0.90,
        ImageClassLabel.LINE_ART: 0.85,
        ImageClassLabel.MIXED: 0.85,
    }
    MAX_PROCESSING_S_4MP: float = 10.0
    """Budget for a 2000x2000 image; scaled linearly by pixel count (min 2 s)."""

    @classmethod
    def time_budget_s(cls, width: int, height: int) -> float:
        return max(2.0, cls.MAX_PROCESSING_S_4MP * (width * height) / 4_000_000)


class MetricName(StrEnum):
    SSIM = "ssim"
    MAX_DELTA_E = "max_delta_e"
    MEAN_DELTA_E = "mean_delta_e"
    GAP_RATIO = "gap_ratio"
    ALPHA_IOU = "alpha_iou"
    PROCESSING_TIME = "processing_time_s"
    SVG_VALID = "svg_valid"


_COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    "<": lambda v, t: v < t,
    "<=": lambda v, t: v <= t,
    ">=": lambda v, t: v >= t,
    ">": lambda v, t: v > t,
}


class MetricCheck(_Contract):
    name: MetricName
    value: float
    threshold: float
    comparator: Literal["<", "<=", ">=", ">"]
    passed: bool

    @model_validator(mode="after")
    def _passed_consistent(self) -> MetricCheck:
        if self.passed != _COMPARATORS[self.comparator](self.value, self.threshold):
            raise ValueError(
                f"{self.name}: passed={self.passed} contradicts {self.value} {self.comparator} {self.threshold}"
            )
        return self


class QualityReport(_Contract):
    """Metric definitions:

    * ssim: SSIM on grayscale after alpha-compositing source and preview over white,
      computed at source resolution.
    * mean/max_delta_e: CIEDE2000 between each fill layer's color_hex and the dominant
      (median in LAB) source color of its palette region; stroke/line layers compare
      against LineMap-covered pixels. Background layers are included.
    * gap_ratio: render preview.png over transparency; among source pixels with alpha >= 128
      after a 1-px erosion (so silhouette anti-aliasing is ignored), the fraction whose preview
      alpha < 128. Detects hairline gaps between color regions.
    * alpha_iou: IoU of (source alpha >= 128) vs (preview alpha >= 128); None if opaque.
    * node_count: total M/L/C segment endpoints over all VectorDocument layers (Z not counted).
    * file_size_bytes: size of the optimized output.svg.
    """

    ssim: float = Field(ge=-1, le=1)
    mean_delta_e: float = Field(ge=0)
    max_delta_e: float = Field(ge=0)
    gap_ratio: float = Field(ge=0, le=1)
    alpha_iou: float | None = Field(default=None, ge=0, le=1)
    node_count: int = Field(ge=0)
    file_size_bytes: int = Field(ge=0)
    processing_time_s: float = Field(ge=0)
    checks: list[MetricCheck]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)
