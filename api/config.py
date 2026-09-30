"""Runtime configuration read from environment variables.

| Variable                   | Default                   | Meaning                                            |
|----------------------------|---------------------------|----------------------------------------------------|
| DATA_DIR                   | ./data                    | Job store root; jobs live in DATA_DIR/jobs/<uuid>/ |
| MAX_WORKERS                | 2                         | Max concurrent pipeline runs (worker processes)    |
| WORKER_EXECUTOR            | process                   | ``process`` (production) or ``thread`` (tests/dev) |
| JOB_TTL_SECONDS            | contracts.api value (3600) | Delete a job this long after it finishes          |
| CLEANUP_INTERVAL_SECONDS   | 60                        | How often the expired-job sweeper runs             |
| MAX_UPLOAD_BYTES           | contracts.api value (20 MiB) | Upload size limit (413 above it)                |
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from contracts.api import JOB_TTL_SECONDS, MAX_PIXELS, MAX_UPLOAD_BYTES

ExecutorKind = Literal["process", "thread"]


def _int(env: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class AppConfig:
    """Settings of one API instance."""

    data_dir: Path
    max_workers: int = 2
    executor: ExecutorKind = "process"
    job_ttl_seconds: int = JOB_TTL_SECONDS
    cleanup_interval_seconds: float = 60.0
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    max_pixels: int = MAX_PIXELS

    @property
    def jobs_dir(self) -> Path:
        """Directory holding one sub-directory per job."""
        return self.data_dir / "jobs"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AppConfig:
        """Build a config from environment variables (see module docstring)."""
        env = os.environ if env is None else env
        executor = env.get("WORKER_EXECUTOR", "process").strip().lower() or "process"
        if executor not in ("process", "thread"):
            raise ValueError(f"WORKER_EXECUTOR must be 'process' or 'thread', got {executor!r}")
        return cls(
            data_dir=Path(env.get("DATA_DIR", "data")).resolve(),
            max_workers=_int(env, "MAX_WORKERS", 2, 1),
            executor=executor,  # type: ignore[arg-type]
            job_ttl_seconds=_int(env, "JOB_TTL_SECONDS", JOB_TTL_SECONDS, 1),
            cleanup_interval_seconds=float(_int(env, "CLEANUP_INTERVAL_SECONDS", 60, 1)),
            max_upload_bytes=_int(env, "MAX_UPLOAD_BYTES", MAX_UPLOAD_BYTES, 1),
        )
