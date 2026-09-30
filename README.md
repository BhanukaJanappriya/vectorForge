# VectorForge

Convert raster images (PNG/JPG) into clean, layered vector graphics: **SVG**, **.ai**, optional
**EPS**, and a **PNG preview**, with a quality report (SSIM, CIEDE2000, gap ratio, node count).

> **About .ai:** native Adobe Illustrator files are proprietary. `output.ai` is an Illustrator-compatible
> **PDF** saved with the `.ai` extension, with one named layer per color. Illustrator and Inkscape open it directly.

## Project status

| Part | Status | How to verify |
|---|---|---|
| Contracts (`contracts/`) | ✅ Done | `python -m pytest -q tests/test_contracts.py` |
| API spec (`api/openapi.yaml`) | ✅ Done | `python scripts/export_openapi.py --check` |
| Sample images + ground truth (`samples/`) | ✅ Done | `python -m pytest -q tests/test_samples.py` |
| Pipeline stages (`pipeline/`) | ✅ Done (all 8 stages) | `python -m eval run --all` |
| Evaluation CLI (`eval/`) | ✅ Done | `python -m eval run --all` |
| API + job worker (`api/`) | ✅ Done, verified locally | see "Run the app locally" |
| Docker (`docker/`, `docker-compose.yml`) | ⚠️ Written, not yet run | `docker compose up --build` |
| Frontend (`frontend/`) | ✅ Done | `cd frontend && npm test && npm run test:e2e` |

Live per-module status is in [PROGRESS.md](PROGRESS.md).

## Pipeline

```
upload → preprocess → classify → quantize → lines → vectorize → assemble → export → evaluate
```

Each stage is a pure function in `pipeline/<stage>.py`. Stages share data only through the typed
contracts in [`contracts/`](contracts/). Project rules and ownership are in [CLAUDE.md](CLAUDE.md), and the
specialist agent definitions are in [`.claude/agents/`](.claude/agents/).

## Repository layout

```
contracts/         shared types, stage signatures, API models, test fixtures (frozen)
api/openapi.yaml   generated from contracts/api.py; the FastAPI app lives in api/
pipeline/          one module per stage + runner.py
eval/              metrics + `python -m eval run --all`
frontend/          React + Vite + TS + Tailwind
samples/           10 generated test images, each with a ground-truth .json
scripts/           tooling (OpenAPI generator)
tests/             pytest suite (test_<module>.py)
```

## Run and verify

### 1. Set up

Requires Python 3.11+ and Git. Node 20+ and Docker are needed later for the frontend and the full app.

```bash
git clone https://github.com/BhanukaJanappriya/vectorForge.git
cd vectorForge
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

Check that the key libraries import:

```bash
python -c "import numpy, cv2, skimage, vtracer, pydantic; print('core dependencies OK')"
python -c "import cairosvg; print('cairo OK')"
```

Expected output: `core dependencies OK` and `cairo OK`.

> **Windows:** CairoSVG needs the native Cairo library. If the second command fails with
> `no library called "cairo-2" was found`, install the GTK3 runtime (for example `choco install gtk-runtime`),
> make sure its `bin` folder is on `PATH`, and open a new terminal. Alternatively, run everything in Docker.
> Phase 0 tests do not need Cairo; rendering previews (export and eval) does.

### 2. Run the test suite

```bash
python -m pytest -q
```

Expected output: every test passes (currently `602 passed, 7 skipped`). The skipped tests need CairoSVG,
Inkscape or Ghostscript and run inside Docker.

Timing tests are marked `slow` and excluded by default. Run them on an otherwise idle machine:

```bash
python -m pytest -q -m slow
```

On laptops that throttle under sustained load, a long run can push individual timings over budget. If one
fails, re-run it alone, for example `python -m pytest -q -m slow tests/test_quantize.py`. A failure names the broken rule. For example, a
contract validator rejects inconsistent data, or a sample no longer matches its ground truth.

Check test coverage (it must stay at or above 80%):

```bash
python -m pytest -q --cov=contracts --cov=pipeline --cov=eval
```

### 3. Verify the API spec is in sync with the contracts

```bash
python scripts/export_openapi.py --check
```

No output and exit code 0 means `api/openapi.yaml` matches `contracts/api.py`. If you edited
`contracts/api.py`, regenerate the spec with `python scripts/export_openapi.py`.

### 4. Verify the sample images are reproducible

```bash
python samples/generate.py
git status --short samples
```

The generator prints 10 lines (file, size, bytes). `git status` should show **no changes**, which proves the
samples and their ground-truth JSON files are deterministic.

### 5. Lint

```bash
python -m ruff check contracts scripts samples tests pipeline eval
```

Expected output: `All checks passed!`

### 6. Measure conversion quality

```bash
python -m eval run --all
```

This runs every sample through the real pipeline and prints one row per sample (SSIM, mean/max ΔE,
gap ratio, node count, file size, time, PASS/FAIL). It also writes an HTML report with side-by-side and diff
images to `eval/reports/<timestamp>/report.html`. The program is working when every row says **PASS**
against the thresholds below. Use `python -m eval run 01` for a single sample.

| Metric | Threshold |
|---|---|
| SSIM (input vs. preview.png) | ≥ 0.90 flat color, ≥ 0.85 line art and mixed |
| CIEDE2000 per region | mean < 2, max < 3 |
| Gap ratio (hairline gaps) | ≤ 0.0005 |
| Time for 2000×2000 | < 10 s |

The last lines of the output show a self-check: deliberately corrupted outputs (shifted colors, a missing layer,
a seam, a blurred preview) must each fail on the right metric and show `OK`.

### 7. Run the app locally (without Docker)

Terminal 1, the API on port 8000:

```bash
python -m uvicorn api.main:app --port 8000
```

Terminal 2, the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and convert `samples/01_logo_4color.png`. Without Cairo and Inkscape on the machine,
the export uses built-in fallbacks (resvg for the preview, a direct PDF/EPS writer). The job's warnings say so.

Check the API directly:

```bash
curl http://localhost:8000/api/v1/health
curl -F "file=@samples/01_logo_4color.png" http://localhost:8000/api/v1/convert
curl http://localhost:8000/api/v1/jobs/<job_id>
curl -o output.svg "http://localhost:8000/api/v1/jobs/<job_id>/files/svg?download=true"
```

A finished job has `"status": "succeeded"` and `result.quality.passed: true`.

### 8. Run the full app with Docker

```bash
docker compose up --build
```

Then verify it:

```bash
curl http://localhost:8080/api/v1/health
```

The expected response looks like `{"status":"ok", ..., "capabilities":{"inkscape":true, ...}}`. Open
http://localhost:8080, upload `samples/01_logo_4color.png`, and wait for the result. Then check that:

- the side-by-side view shows the vector matching the original,
- the quality badge shows PASS,
- the SVG, AI, EPS and PNG downloads open (the .ai opens in Illustrator or Inkscape with layers named like `color_1_#DC322F`).

You can also convert an image from the command line against the running app:

```bash
curl -F "file=@samples/01_logo_4color.png" http://localhost:8080/api/v1/convert
curl http://localhost:8080/api/v1/jobs/<job_id>
curl -o output.svg "http://localhost:8080/api/v1/jobs/<job_id>/files/svg?download=true"
```

> The Docker setup has been validated with `docker compose config` but not yet built and run. Report any build
> problem as an issue.
