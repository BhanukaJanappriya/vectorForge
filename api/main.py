"""FastAPI application implementing ``api/openapi.yaml``.

Run with ``uvicorn api.main:app`` (single server process; pipeline concurrency comes from
the worker pool, see api.worker). Configuration: environment variables, see api.config.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import importlib.util
import logging
import shutil
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi import Path as PathParam
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from api import __version__
from api.config import AppConfig
from api.errors import ApiError, ErrorResponse, install_error_handlers, not_found
from api.openapi import build_openapi
from api.store import JobStore, StoredJob, canonical_job_id, utcnow
from api.uploads import EXTENSIONS, check_image, discard, receive_upload, safe_filename
from api.worker import MEDIA_TYPES, JobManager
from contracts.api import (
    ACCEPTED_MEDIA_TYPES,
    API_PREFIX,
    MAX_UPLOAD_BYTES,
    ConfigResponse,
    ErrorDetail,
    FileKind,
    HealthResponse,
    JobResponse,
    JobStatus,
    Limits,
    PipelineStage,
    RerunRequest,
)
from contracts.schemas import Settings
from pipeline.runner import StageFn

log = logging.getLogger("vectorforge.api")

SVG_CSP = "default-src 'none'; style-src 'unsafe-inline'"
FILE_KINDS = tuple(k.value for k in FileKind)
"""Strict whitelist for the ``kind`` path parameter."""
UPLOAD_PART = "upload.part"

JobIdParam = Annotated[str, PathParam(json_schema_extra={"format": "uuid"})]


def _error(description: str) -> dict[str, Any]:
    return {"model": ErrorResponse, "description": description}


def _importable(module: str) -> bool:
    """True if ``module`` imports (cairosvg needs its native library, not just the package)."""
    try:
        importlib.import_module(module)
    except Exception:  # OSError when the native library is missing
        return False
    return True


@lru_cache(maxsize=1)
def detect_capabilities() -> dict[str, bool]:
    """Which external tools / native-backed libraries are usable in this environment."""
    return {
        "inkscape": shutil.which("inkscape") is not None,
        "potrace": shutil.which("potrace") is not None or importlib.util.find_spec("potrace") is not None,
        "vtracer": importlib.util.find_spec("vtracer") is not None,
        "cairosvg": _importable("cairosvg"),
        "ghostscript": any(shutil.which(n) for n in ("gs", "gswin64c", "gswin32c")),
    }


def _new_job(
    store: JobStore,
    job_id: str,
    filename: str,
    settings: Settings,
    upload_name: str,
    media_type: str,
    source_job_id: str | None = None,
) -> StoredJob:
    now = utcnow()
    job = JobResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        stage=PipelineStage.UPLOAD,
        progress=0.0,
        created_at=now,
        updated_at=now,
        expires_at=now + store.ttl,
        filename=filename,
        settings=settings,
        source_job_id=source_job_id,
    )
    stored = StoredJob(job=job, upload_name=upload_name, upload_media_type=media_type)
    stored.outputs = {FileKind.ORIGINAL: upload_name}
    store.save(stored, create=True)
    return stored


def _parse_settings(raw: str | None) -> Settings:
    if raw is None or not raw.strip():
        return Settings()
    try:
        return Settings.model_validate_json(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'settings'}: {e['msg']}" for e in exc.errors()[:5]
        )
        raise ApiError(422, "invalid_settings", f"Invalid settings: {problems}", PipelineStage.UPLOAD) from exc


def _download_name(filename: str, kind: FileKind, upload_name: str) -> str:
    stem = Path(filename).stem or "output"
    if kind == FileKind.ORIGINAL:
        return f"{stem}{Path(upload_name).suffix}"
    if kind == FileKind.PNG:
        return f"{stem}_preview.png"
    return f"{stem}.{kind.value}"


def create_app(config: AppConfig | None = None, stages: Mapping[str, StageFn] | None = None) -> FastAPI:
    """Build the app.

    Args:
        config: Runtime configuration (default: from environment variables).
        stages: Optional pipeline stage overrides passed to the runner (tests, eval stand-ins).
            Must be picklable (module-level functions) when the process executor is used.
    """
    config = config or AppConfig.from_env()
    store = JobStore(config.jobs_dir, config.job_ttl_seconds)
    manager = JobManager(config, store, stages)

    async def _cleanup_loop() -> None:
        while True:
            await asyncio.sleep(config.cleanup_interval_seconds)
            try:
                deleted = await run_in_threadpool(store.sweep)
                if deleted:
                    log.info("deleted %d expired job(s)", len(deleted))
            except Exception:
                log.exception("job cleanup failed")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        store.ensure()
        store.fail_unfinished("Interrupted by a server restart; please convert again.")
        store.sweep()
        manager.start()
        cleaner = asyncio.create_task(_cleanup_loop())
        try:
            yield
        finally:
            cleaner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleaner
            await run_in_threadpool(manager.shutdown)

    app = FastAPI(
        title="VectorForge API",
        version=__version__,
        lifespan=lifespan,
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=None,
        openapi_url=f"{API_PREFIX}/openapi.json",
    )
    app.state.config = config
    app.state.store = store
    app.state.manager = manager
    app.openapi = lambda: build_openapi(app)  # type: ignore[method-assign]
    install_error_handlers(app)

    def load_job(job_id: str) -> StoredJob:
        stored = store.load(job_id) if canonical_job_id(job_id) else None
        if stored is None:
            raise not_found()
        return stored

    def dispatch(job_id: str) -> None:
        try:
            manager.submit(job_id)
        except Exception as exc:
            log.exception("could not dispatch job %s", job_id)
            manager.mark_failed_now(job_id, ErrorDetail(code="internal_error", message=f"Dispatch failed: {exc}"))

    @app.get(
        f"{API_PREFIX}/health",
        operation_id="getHealth",
        summary="Liveness + available external tools",
        response_model=HealthResponse,
        response_description="OK",
    )
    async def get_health() -> HealthResponse:
        """\fLiveness probe; also reports which external tools are installed."""
        capabilities = await run_in_threadpool(detect_capabilities)
        return HealthResponse(version=__version__, capabilities=capabilities)

    @app.get(
        f"{API_PREFIX}/config",
        operation_id="getConfig",
        summary="Default settings and upload limits (drives the UI form)",
        response_model=ConfigResponse,
        response_description="OK",
    )
    async def get_config() -> ConfigResponse:
        """\fDefault Settings and the limits enforced by this server."""
        limits = Limits(
            max_upload_bytes=config.max_upload_bytes,
            job_ttl_seconds=config.job_ttl_seconds,
            max_pixels=config.max_pixels,
            accepted_media_types=list(ACCEPTED_MEDIA_TYPES),
        )
        return ConfigResponse(defaults=Settings(), limits=limits)

    @app.post(
        f"{API_PREFIX}/convert",
        operation_id="convert",
        summary="Upload an image and start a conversion job",
        status_code=202,
        response_model=JobResponse,
        response_description="Job accepted",
        responses={
            413: _error("File too large"),
            415: _error("Unsupported media type"),
            422: _error("Invalid settings or undecodable image"),
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {
                                "file": {
                                    "type": "string",
                                    "format": "binary",
                                    "description": f"PNG or JPEG, <= {MAX_UPLOAD_BYTES} bytes",
                                },
                                "settings": {
                                    "type": "string",
                                    "contentMediaType": "application/json",
                                    "description": "JSON-encoded Settings; omitted fields use defaults",
                                },
                            },
                        },
                        "encoding": {"settings": {"contentType": "application/json"}},
                    }
                },
            }
        },
    )
    async def convert(request: Request, background: BackgroundTasks) -> JobResponse:
        """\fStream the upload to disk, validate it and queue a conversion job.

        The body is parsed manually (not via File()/Form()) so the size limit is enforced
        while streaming; ``settings`` is a plain JSON string field.
        """
        job_id, job_dir = await run_in_threadpool(store.create_dir)
        part = job_dir / UPLOAD_PART
        try:
            upload = await receive_upload(request, part, config.max_upload_bytes)
            media_type = await run_in_threadpool(check_image, part, config.max_pixels)
            settings = _parse_settings(upload.settings_raw)
            upload_name = f"upload{EXTENSIONS[media_type]}"
            part.replace(job_dir / upload_name)
            stored = _new_job(
                store, job_id, safe_filename(upload.filename), settings, upload_name, media_type
            )
        except BaseException:
            discard(part)
            await run_in_threadpool(shutil.rmtree, job_dir, True)
            raise
        background.add_task(dispatch, job_id)
        return stored.job

    @app.post(
        f"{API_PREFIX}/jobs/{{job_id}}/rerun",
        operation_id="rerunJob",
        summary="Start a new job from the same upload with new settings (palette edit/merge)",
        status_code=202,
        response_model=JobResponse,
        response_description="New job accepted",
        responses={404: _error("Unknown or expired job"), 422: _error("Invalid settings")},
    )
    async def rerun_job(job_id: JobIdParam, body: RerunRequest, background: BackgroundTasks) -> JobResponse:
        """\fCopy the source job's upload into a new job and queue it with the new settings."""
        source = load_job(job_id)
        src_file = store.job_dir(job_id) / source.upload_name
        new_id, new_dir = await run_in_threadpool(store.create_dir)
        try:
            await run_in_threadpool(shutil.copyfile, src_file, new_dir / source.upload_name)
            stored = _new_job(
                store,
                new_id,
                source.job.filename,
                body.settings,
                source.upload_name,
                source.upload_media_type,
                source_job_id=job_id,
            )
        except FileNotFoundError as exc:  # source deleted concurrently
            await run_in_threadpool(shutil.rmtree, new_dir, True)
            raise not_found() from exc
        except BaseException:
            await run_in_threadpool(shutil.rmtree, new_dir, True)
            raise
        background.add_task(dispatch, new_id)
        return stored.job

    @app.get(
        f"{API_PREFIX}/jobs/{{job_id}}",
        operation_id="getJob",
        summary="Poll job status/result",
        response_model=JobResponse,
        response_description="OK",
        responses={404: _error("Unknown job")},
    )
    async def get_job(job_id: JobIdParam) -> JobResponse:
        """\fCurrent status, progress and (when finished) the result or error."""
        stored = await run_in_threadpool(load_job, job_id)
        return stored.job

    @app.delete(
        f"{API_PREFIX}/jobs/{{job_id}}",
        operation_id="deleteJob",
        summary="Delete a job and its files",
        status_code=204,
        response_class=Response,
        response_description="Deleted",
        responses={404: _error("Unknown job")},
    )
    async def delete_job(job_id: JobIdParam) -> Response:
        """\fCancel the job if still queued and delete its directory."""
        load_job(job_id)
        manager.cancel(job_id)
        if not await run_in_threadpool(store.delete, job_id):
            raise not_found()
        return Response(status_code=204)

    @app.get(
        f"{API_PREFIX}/jobs/{{job_id}}/files/{{kind}}",
        operation_id="getJobFile",
        summary="Download an input/output file",
        response_class=Response,
        responses={
            200: {
                "description": "File contents",
                "content": {
                    "image/svg+xml": {"schema": {"type": "string"}},
                    "image/png": {"schema": {"type": "string", "format": "binary"}},
                    "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
                    "application/postscript": {"schema": {"type": "string", "format": "binary"}},
                    "application/illustrator": {"schema": {"type": "string", "format": "binary"}},
                },
            },
            404: _error("Unknown job, or file not produced / not ready"),
        },
    )
    async def get_job_file(
        job_id: JobIdParam,
        kind: Annotated[str, PathParam(json_schema_extra={"enum": list(FILE_KINDS)})],
        download: Annotated[bool, Query(description="Send Content-Disposition: attachment")] = False,
    ) -> Response:
        """\fServe one whitelisted file of a job with its media type (SVG gets a strict CSP)."""
        stored = await run_in_threadpool(load_job, job_id)
        if kind not in FILE_KINDS:
            raise not_found(f"Unknown file kind {kind!r}.")
        file_kind = FileKind(kind)
        if file_kind != FileKind.ORIGINAL and stored.job.status != JobStatus.SUCCEEDED:
            raise not_found("The job has not produced its files yet.")
        rel = stored.outputs.get(file_kind)
        job_dir = store.job_dir(job_id).resolve()
        path = (job_dir / rel).resolve() if rel else None
        if path is None or not path.is_relative_to(job_dir) or not path.is_file():
            raise not_found(f"This job has no {kind} file.")
        media_type = stored.upload_media_type if file_kind == FileKind.ORIGINAL else MEDIA_TYPES[file_kind]
        headers = {"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=3600"}
        if file_kind == FileKind.SVG:
            headers["Content-Security-Policy"] = SVG_CSP
        return FileResponse(
            path,
            media_type=media_type,
            headers=headers,
            filename=_download_name(stored.job.filename, file_kind, stored.upload_name),
            content_disposition_type="attachment" if download else "inline",
        )

    return app


app = create_app()
