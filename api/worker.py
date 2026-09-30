"""Background execution of pipeline jobs.

v1 design: the ``/convert`` and ``/rerun`` routes schedule :meth:`JobManager.submit` as a
FastAPI BackgroundTask, which hands the job to a :class:`~concurrent.futures.ProcessPoolExecutor`
(the pipeline is CPU-bound, so threads would serialize on the GIL). ``MAX_WORKERS`` caps
concurrency. The worker process runs :func:`execute_job`, which reads the job from the
on-disk store, runs :func:`pipeline.runner.run_pipeline` and writes status, progress and
the result back to ``job.json``. The API process never holds job state in memory.

Upgrade path (ARQ or Celery + Redis): :func:`execute_job` only needs ``(data_dir, job_id)``
and the store on disk, so it can be registered unchanged as an ARQ/Celery task. Replace
:class:`JobManager` with a thin client that enqueues ``execute_job`` in Redis, run the
workers as a separate compose service sharing the DATA_DIR volume (or move uploads/outputs
to object storage), and move the sweeper to an ARQ cron / Celery beat job. Routes,
responses and the job store format stay the same.
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
import traceback
from collections.abc import Mapping
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from api.config import AppConfig
from api.store import OUT_DIR, TERMINAL, JobGoneError, JobStore, StoredJob, utcnow
from contracts.api import API_PREFIX, ErrorDetail, FileKind, JobFile, JobResult, JobStatus, PipelineStage
from pipeline.runner import PipelineError, PipelineResult, StageFn, run_pipeline

log = logging.getLogger("vectorforge.worker")

MEDIA_TYPES: dict[FileKind, str] = {
    FileKind.SVG: "image/svg+xml",
    FileKind.AI: "application/illustrator",
    FileKind.EPS: "application/postscript",
    FileKind.PNG: "image/png",
}
"""Media types of the pipeline outputs (``original`` uses the sniffed upload type)."""

MAX_TASKS_PER_CHILD = 25
"""Recycle worker processes periodically to bound memory growth from native libraries."""


def file_url(job_id: str, kind: FileKind) -> str:
    """Relative download URL of a job file."""
    return f"{API_PREFIX}/jobs/{job_id}/files/{kind.value}"


def _output_paths(result: PipelineResult) -> dict[FileKind, Path | None]:
    bundle = result.bundle
    return {
        FileKind.SVG: bundle.svg_path,
        FileKind.PNG: bundle.preview_png_path,
        FileKind.AI: bundle.ai_path,
        FileKind.EPS: bundle.eps_path,
    }


def build_result(stored: StoredJob, job_dir: Path, result: PipelineResult) -> JobResult:
    """Record the produced files in ``stored.outputs`` and build the public JobResult.

    Only formats requested in ``settings.output_formats`` are listed (svg is always
    requested). Outputs must be inside the job directory.
    """
    job_id = stored.job.job_id
    root = job_dir.resolve()
    requested = {fmt.value for fmt in stored.job.settings.output_formats}
    files = [
        JobFile(
            kind=FileKind.ORIGINAL,
            url=file_url(job_id, FileKind.ORIGINAL),
            media_type=stored.upload_media_type,
            size_bytes=(job_dir / stored.upload_name).stat().st_size,
        )
    ]
    outputs: dict[FileKind, str] = {FileKind.ORIGINAL: stored.upload_name}
    for kind, path in _output_paths(result).items():
        if path is None or kind.value not in requested:
            continue
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(root):
            raise PipelineError(
                ErrorDetail(code="stage_failed", message=f"export wrote {kind.value} outside the job directory",
                            stage=PipelineStage.EXPORT),
                "export",
                result.timings,
                {},
            )
        if not resolved.is_file():
            continue
        outputs[kind] = resolved.relative_to(root).as_posix()
        files.append(
            JobFile(kind=kind, url=file_url(job_id, kind), media_type=MEDIA_TYPES[kind],
                    size_bytes=resolved.stat().st_size)
        )
    stored.outputs = outputs
    return JobResult(
        width=result.image.width,
        height=result.image.height,
        image_class=result.image_class,
        layer_count=len(result.doc.layers),
        palette_hex=[color.hex for color in result.palette.colors],
        quality=result.report,
        files=files,
        warnings=list(result.bundle.warnings),
    )


def execute_job(
    data_dir: str,
    job_id: str,
    ttl_seconds: int,
    stages: Mapping[str, StageFn] | None = None,
) -> JobStatus | None:
    """Run one job to completion inside a worker process (or thread).

    Returns the final status, or None if the job no longer exists (deleted/expired).
    Never raises for pipeline failures: they are recorded in ``job.json``.
    """
    store = JobStore(Path(data_dir) / "jobs", ttl_seconds)
    stored = store.load(job_id)
    if stored is None or stored.job.status in TERMINAL:
        return None if stored is None else stored.job.status
    job_dir = store.job_dir(job_id)
    try:
        stored.job.status = JobStatus.RUNNING
        stored.job.stage = PipelineStage.PREPROCESS
        stored.job.progress = 0.0
        store.touch(stored)
        store.save(stored)

        def on_progress(stage: PipelineStage, fraction: float) -> None:
            if stage == PipelineStage.DONE:
                return  # the terminal write below records completion atomically with the result
            stored.job.stage = stage
            stored.job.progress = max(stored.job.progress, min(1.0, fraction))
            store.touch(stored)
            store.save(stored)

        try:
            result = run_pipeline(
                job_dir / stored.upload_name,
                stored.job.settings,
                job_dir / OUT_DIR,
                on_progress=on_progress,
                stages=stages,
            )
            stored.job.result = build_result(stored, job_dir, result)
            store.finish(stored, JobStatus.SUCCEEDED)
        except PipelineError as exc:
            log.info("job %s failed at %s: %s", job_id, exc.stage_name, exc.detail.message)
            store.finish(stored, JobStatus.FAILED, error=exc.detail)
        except JobGoneError:
            raise
        except Exception as exc:
            log.error("job %s crashed:\n%s", job_id, traceback.format_exc())
            detail = ErrorDetail(code="internal_error", message=f"{type(exc).__name__}: {exc}", stage=stored.job.stage)
            store.finish(stored, JobStatus.FAILED, error=detail)
        store.save(stored)
        return stored.job.status
    except JobGoneError:
        log.info("job %s was deleted while running", job_id)
        return None


class JobManager:
    """Dispatches jobs to a bounded executor and records worker crashes."""

    def __init__(self, config: AppConfig, store: JobStore, stages: Mapping[str, StageFn] | None = None) -> None:
        self.config = config
        self.store = store
        self.stages = dict(stages) if stages else None
        self._executor: Executor | None = None
        self._futures: dict[str, Future[JobStatus | None]] = {}
        self._lock = threading.Lock()

    def _new_executor(self) -> Executor:
        if self.config.executor == "thread":
            return ThreadPoolExecutor(max_workers=self.config.max_workers, thread_name_prefix="vf-job")
        # spawn: never fork the multi-threaded server process
        return ProcessPoolExecutor(
            max_workers=self.config.max_workers,
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=MAX_TASKS_PER_CHILD,
        )

    def start(self) -> None:
        """Create the executor (idempotent)."""
        with self._lock:
            if self._executor is None:
                self._executor = self._new_executor()

    def submit(self, job_id: str) -> None:
        """Queue a job. Called from a FastAPI BackgroundTask after the 202 response."""
        self.start()
        args = (str(self.config.data_dir), job_id, self.config.job_ttl_seconds, self.stages)
        with self._lock:
            assert self._executor is not None
            try:
                future = self._executor.submit(execute_job, *args)
            except BrokenProcessPool:
                log.warning("worker pool was broken; recreating it")
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = self._new_executor()
                future = self._executor.submit(execute_job, *args)
            self._futures[job_id] = future
        future.add_done_callback(lambda f, jid=job_id: self._on_done(jid, f))

    def _on_done(self, job_id: str, future: Future[JobStatus | None]) -> None:
        with self._lock:
            self._futures.pop(job_id, None)
        if future.cancelled():
            return
        exc = future.exception()
        if exc is None:
            return
        log.error("worker for job %s died: %r", job_id, exc)
        if isinstance(exc, BrokenProcessPool):
            with self._lock:
                broken, self._executor = self._executor, None
            if broken is not None:
                broken.shutdown(wait=False, cancel_futures=True)
        stored = self.store.load(job_id)
        if stored is None or stored.job.status in TERMINAL:
            return
        detail = ErrorDetail(code="worker_crashed", message=f"Worker process failed: {exc!r}", stage=stored.job.stage)
        self.store.finish(stored, JobStatus.FAILED, error=detail)
        try:
            self.store.save(stored)
        except JobGoneError:
            return

    def cancel(self, job_id: str) -> None:
        """Cancel a job that has not started yet (a running one stops at its next progress write)."""
        with self._lock:
            future = self._futures.pop(job_id, None)
        if future is not None:
            future.cancel()

    def active(self) -> int:
        """Number of queued or running futures."""
        with self._lock:
            return len(self._futures)

    def shutdown(self) -> None:
        """Stop accepting work, cancel queued jobs and wait for running ones."""
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def mark_failed_now(self, job_id: str, detail: ErrorDetail) -> None:
        """Record a failure for a job that could not be dispatched."""
        stored = self.store.load(job_id)
        if stored is None:
            return
        self.store.finish(stored, JobStatus.FAILED, error=detail, now=utcnow())
        try:
            self.store.save(stored)
        except JobGoneError:
            return
