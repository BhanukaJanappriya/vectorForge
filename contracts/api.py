"""HTTP API models (source of truth for api/openapi.yaml and frontend TypeScript types).

Regenerate the OpenAPI file after any change: `python scripts/export_openapi.py`.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from contracts.schemas import SCHEMA_VERSION, ImageClass, QualityReport, Settings

API_PREFIX = "/api/v1"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 36_000_000
ACCEPTED_MEDIA_TYPES = ["image/png", "image/jpeg"]
JOB_TTL_SECONDS = 3600
"""Job files (upload + outputs) are deleted this long after the job finishes."""


class _ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PipelineStage(StrEnum):
    UPLOAD = "upload"
    PREPROCESS = "preprocess"
    CLASSIFY = "classify"
    QUANTIZE = "quantize"
    EXTRACT_LINES = "extract_lines"
    VECTORIZE = "vectorize"
    ASSEMBLE = "assemble"
    EXPORT = "export"
    EVALUATE = "evaluate"
    DONE = "done"


class FileKind(StrEnum):
    ORIGINAL = "original"
    SVG = "svg"
    AI = "ai"
    EPS = "eps"
    PNG = "png"


class JobFile(_ApiModel):
    kind: FileKind
    url: str = Field(description="Relative URL, e.g. /api/v1/jobs/{id}/files/svg")
    media_type: str
    size_bytes: int = Field(ge=0)


class JobResult(_ApiModel):
    width: int
    height: int
    image_class: ImageClass
    layer_count: int = Field(ge=0)
    palette_hex: list[str]
    quality: QualityReport
    files: list[JobFile]
    warnings: list[str] = Field(default_factory=list)


class ErrorDetail(_ApiModel):
    code: str = Field(description="Machine-readable, e.g. invalid_image, stage_failed, too_large.")
    message: str
    stage: PipelineStage | None = None


class JobResponse(_ApiModel):
    job_id: str
    status: JobStatus
    stage: PipelineStage
    progress: float = Field(ge=0, le=1)
    created_at: datetime
    updated_at: datetime
    expires_at: datetime = Field(description="When the job and its files will be deleted.")
    filename: str
    settings: Settings
    source_job_id: str | None = Field(default=None, description="Set when this job is a re-run of another job.")
    result: JobResult | None = None
    error: ErrorDetail | None = None


class RerunRequest(_ApiModel):
    """Re-run a finished job's upload with new settings (e.g. an edited palette_override)."""

    settings: Settings


class Limits(_ApiModel):
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    job_ttl_seconds: int = JOB_TTL_SECONDS
    max_pixels: int = MAX_PIXELS
    accepted_media_types: list[str] = Field(default_factory=lambda: list(ACCEPTED_MEDIA_TYPES))


class ConfigResponse(_ApiModel):
    defaults: Settings
    limits: Limits


class HealthResponse(_ApiModel):
    status: str = "ok"
    version: str
    schema_version: str = SCHEMA_VERSION
    capabilities: dict[str, bool] = Field(description="e.g. {'inkscape': true, 'vtracer': true, 'potrace': false}")
