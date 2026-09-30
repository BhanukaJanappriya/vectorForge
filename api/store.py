"""On-disk job store: ``DATA_DIR/jobs/<uuid>/`` holds ``job.json``, the upload and ``out/``.

The directory is the single source of truth shared by the API process and the worker
processes: the API creates the job, the worker updates status/progress in ``job.json``
(atomic write + rename), and the sweeper deletes it ``JOB_TTL_SECONDS`` after it finished.
Because state lives on disk, not in API memory, swapping the executor for ARQ/Celery
workers on other hosts only needs a shared volume (or an object store) at DATA_DIR.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from contracts.api import ErrorDetail, FileKind, JobResponse, JobStatus, PipelineStage

JOB_FILE = "job.json"
OUT_DIR = "out"
TERMINAL = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED})
_RETRIES = 20
_RETRY_SLEEP_S = 0.02
P = ParamSpec("P")
R = TypeVar("R")


_COMPUTED_FIELDS: dict[str, Any] = {"job": {"result": {"quality": {"passed"}}}}
"""Computed (output-only) fields; the contract models forbid them as input, so they are not persisted."""
log = logging.getLogger("vectorforge.store")


class JobGoneError(Exception):
    """The job was deleted (or expired) while something was still writing to it."""


class StoredJob(BaseModel):
    """What ``job.json`` contains: the public JobResponse plus private bookkeeping."""

    model_config = ConfigDict(extra="forbid")

    job: JobResponse
    upload_name: str = Field(description="File name of the upload inside the job dir.")
    upload_media_type: str
    outputs: dict[FileKind, str] = Field(
        default_factory=dict, description="Downloadable kind -> path relative to the job dir."
    )
    finished_at: datetime | None = None


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(UTC)


def canonical_job_id(value: str) -> str | None:
    """Return ``value`` if it is a canonical lowercase UUID string, else None."""
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None
    return value if str(parsed) == value else None


def _retry_os(fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Retry transient Windows sharing violations (a reader holding the file open)."""
    for attempt in range(_RETRIES):
        try:
            return fn(*args, **kwargs)
        except PermissionError:
            if attempt == _RETRIES - 1:
                raise
            time.sleep(_RETRY_SLEEP_S)
    raise AssertionError("unreachable")  # pragma: no cover


class JobStore:
    """File-system job store rooted at ``jobs_dir``."""

    def __init__(self, jobs_dir: Path, ttl_seconds: int) -> None:
        self.jobs_dir = Path(jobs_dir)
        self.ttl = timedelta(seconds=ttl_seconds)

    # ---------------------------------------------------------------- paths

    def ensure(self) -> None:
        """Create the jobs directory."""
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        """Directory of a job (``job_id`` must already be validated)."""
        return self.jobs_dir / job_id

    def create_dir(self) -> tuple[str, Path]:
        """Allocate a new job id and its (empty) directory."""
        self.ensure()
        job_id = str(uuid.uuid4())
        path = self.job_dir(job_id)
        path.mkdir()
        return job_id, path

    def ids(self) -> Iterator[str]:
        """All job ids currently on disk (including half-created ones)."""
        if not self.jobs_dir.is_dir():
            return
        for entry in self.jobs_dir.iterdir():
            if entry.is_dir() and canonical_job_id(entry.name):
                yield entry.name

    # ------------------------------------------------------------- read/write

    def load(self, job_id: str) -> StoredJob | None:
        """Read a job; None if the id is invalid, unknown, deleted or unreadable."""
        if canonical_job_id(job_id) is None:
            return None
        path = self.job_dir(job_id) / JOB_FILE
        try:
            raw = _retry_os(path.read_bytes)
        except (FileNotFoundError, NotADirectoryError):
            return None
        try:
            return StoredJob.model_validate_json(raw)
        except ValidationError as exc:
            log.warning("unreadable job file %s: %s", path, exc)
            return None

    def save(self, stored: StoredJob, *, create: bool = False) -> None:
        """Atomically write ``job.json``.

        Unless ``create`` is set, the job must still exist: a job deleted while a worker is
        running raises JobGoneError instead of being resurrected.
        """
        job_dir = self.job_dir(stored.job.job_id)
        target = job_dir / JOB_FILE
        if not create and not target.exists():
            raise JobGoneError(stored.job.job_id)
        tmp = job_dir / f".{JOB_FILE}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            tmp.write_text(stored.model_dump_json(exclude=_COMPUTED_FIELDS), encoding="utf-8")
            _retry_os(os.replace, tmp, target)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise JobGoneError(stored.job.job_id) from exc
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

    def touch(self, stored: StoredJob, now: datetime | None = None) -> None:
        """Bump updated_at and push expires_at out while the job is still active."""
        now = now or utcnow()
        stored.job.updated_at = now
        base = stored.finished_at or now
        stored.job.expires_at = base + self.ttl

    def finish(
        self,
        stored: StoredJob,
        status: JobStatus,
        *,
        error: ErrorDetail | None = None,
        now: datetime | None = None,
    ) -> None:
        """Mark a job terminal; it expires ``ttl`` after this moment."""
        now = now or utcnow()
        stored.job.status = status
        stored.job.error = error
        if status == JobStatus.SUCCEEDED:
            stored.job.stage = PipelineStage.DONE
            stored.job.progress = 1.0
        elif error is not None and error.stage is not None:
            stored.job.stage = error.stage
        stored.finished_at = now
        self.touch(stored, now)

    # ---------------------------------------------------------------- delete

    def delete(self, job_id: str) -> bool:
        """Delete a job. ``job.json`` goes first so the job disappears even if files are locked."""
        if canonical_job_id(job_id) is None:
            return False
        job_dir = self.job_dir(job_id)
        if not job_dir.is_dir():
            return False
        existed = (job_dir / JOB_FILE).exists()
        with contextlib.suppress(FileNotFoundError):
            _retry_os((job_dir / JOB_FILE).unlink)
        shutil.rmtree(job_dir, ignore_errors=True)
        return existed

    def sweep(self, now: datetime | None = None) -> list[str]:
        """Delete expired jobs, and orphan directories (no job.json) older than the TTL."""
        now = now or utcnow()
        deleted: list[str] = []
        for job_id in list(self.ids()):
            stored = self.load(job_id)
            if stored is not None:
                if stored.job.expires_at <= now:
                    self.delete(job_id)
                    deleted.append(job_id)
                continue
            job_dir = self.job_dir(job_id)
            try:
                mtime = datetime.fromtimestamp(job_dir.stat().st_mtime, UTC)
            except FileNotFoundError:
                continue
            if mtime + self.ttl <= now:
                shutil.rmtree(job_dir, ignore_errors=True)
                deleted.append(job_id)
        return deleted

    def fail_unfinished(self, reason: str, now: datetime | None = None) -> list[str]:
        """Mark queued/running jobs failed (their worker died with the previous server process)."""
        failed: list[str] = []
        for job_id in list(self.ids()):
            stored = self.load(job_id)
            if stored is None or stored.job.status in TERMINAL:
                continue
            detail = ErrorDetail(code="interrupted", message=reason, stage=stored.job.stage)
            self.finish(stored, JobStatus.FAILED, error=detail, now=now)
            with contextlib.suppress(JobGoneError):
                self.save(stored)
                failed.append(job_id)
        return failed
