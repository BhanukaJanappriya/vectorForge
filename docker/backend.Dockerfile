# syntax=docker/dockerfile:1.7
# VectorForge backend: FastAPI + pipeline + evaluator on python:3.12-slim.
# Build context: repository root (see docker-compose.yml).

# ---------------------------------------------------------------- wheel builder
# pycairo publishes no Linux wheels, so it (and any other sdist-only dependency) is
# compiled here against the cairo headers; the runtime stage installs only the wheels.
FROM python:3.12-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config libcairo2-dev \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip wheel --wheel-dir /wheels -r /tmp/requirements.txt

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    MAX_WORKERS=2 \
    WORKER_EXECUTOR=process \
    JOB_TTL_SECONDS=3600 \
    CLEANUP_INTERVAL_SECONDS=60
# inkscape: .ai/.eps export; potrace: bitmap tracing; ghostscript: EPS/PDF tooling;
# libcairo2 (+ pango/gdk-pixbuf, fonts): CairoSVG preview rendering and pycairo;
# libglib2.0-0: OpenCV runtime dependency.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      inkscape potrace ghostscript \
      libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 \
      libglib2.0-0 fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 vectorforge \
 && mkdir -p /data && chown vectorforge:vectorforge /data
COPY --from=builder /wheels /wheels
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-index --find-links /wheels -r /tmp/requirements.txt && rm -rf /wheels

WORKDIR /app
COPY contracts/ contracts/
COPY pipeline/ pipeline/
COPY eval/ eval/
COPY api/ api/
# Inkscape needs a writable HOME for its profile directory.
ENV HOME=/home/vectorforge
USER vectorforge
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status == 200 else 1)"
# One server process: concurrency comes from the MAX_WORKERS process pool, and the
# startup hook that fails interrupted jobs assumes a single API process per DATA_DIR.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
