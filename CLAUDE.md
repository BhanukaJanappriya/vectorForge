# VectorForge — Project Rules

## Goal
Convert PNG/JPG into clean vectors: output.svg, output.ai (PDF-compatible),
output.eps, preview.png. Auto-detect line art / flat color / mixed.

## Quality thresholds (must pass)
Canonical values live in `contracts/schemas.py::QualityThresholds`.
- SSIM(input, preview.png) >= 0.90 flat color, >= 0.85 line art and mixed
- CIEDE2000 per region: mean < 2, max < 3
- No hairline gaps between color regions (gap_ratio <= 0.0005)
- Transparency preserved; no background fill unless present in the source
- < 10 s for 2000x2000 on CPU (budget scales with pixel count, min 2 s)

## Stack
Python 3.11+, FastAPI, Pydantic v2, OpenCV, scikit-image, scikit-learn, vtracer, potrace,
CairoSVG, Inkscape CLI, scour, lxml, pikepdf. Frontend: React + Vite + TS + Tailwind.
pytest, Playwright. Docker + docker compose.

## Architecture
preprocess -> classify -> quantize -> lines -> vectorize -> assemble -> export -> evaluate

| Stage | File | Entrypoint(s) | Owner (subagent) |
|---|---|---|---|
| preprocess | `pipeline/preprocess.py` | `load_image`, `preprocess` | preprocessor |
| classify | `pipeline/classify.py` | `classify` | preprocessor |
| quantize | `pipeline/quantize.py` | `quantize` | color-quantizer |
| lines | `pipeline/lines.py` | `extract_lines` | line-extractor |
| vectorize | `pipeline/vectorize.py` | `vectorize` | vectorizer |
| assemble | `pipeline/assemble.py` | `assemble_svg` | svg-exporter |
| export | `pipeline/export.py` | `export` | svg-exporter |
| evaluate | `eval/evaluate.py` + `eval/` CLI | `evaluate` | qa-evaluator |
| runner + API | `pipeline/runner.py`, `api/` | FastAPI app | backend-devops |
| UI | `frontend/` | — | frontend-builder |

Contracts (read before touching any stage):
- `contracts/schemas.py`: every data type passed between stages, `Settings`, `DETAIL_PRESETS`,
  `QualityThresholds`, `layer_name()`, and errors (`InvalidImageError`, `StageError`, …)
- `contracts/stages.py`: the exact signature of each stage, `STAGE_ENTRYPOINTS`, and pipeline branching
- `contracts/api.py`: HTTP models. `api/openapi.yaml` is GENERATED from it (`python scripts/export_openapi.py`)
- `contracts/fixtures.py`: valid mock objects for testing a stage without its upstream stages

Key conventions (all enforced by validators):
- Arrays in contracts are (H, W[, C]), in *processing space*, and read-only. Copy before mutating.
- Label map value -1 means transparent. Transparent pixels are never vectorized.
- VectorDocument paths are in *source space* and use absolute `M/L/C/Z` only.
- Layers are ordered bottom to top by `z_order`, stacked (lower layers extend under upper layers) so there are no gaps.
  Layer ids and names come from `layer_name()`, e.g. `color_1_#E53935`.
- Stage modules never import each other. Only `pipeline/runner.py` composes stages.

## Rules
1. `/contracts/` and `api/openapi.yaml` are frozen. Only the orchestrator (main session) edits them.
   Subagents that need a change STOP and report it under CONTRACT CHANGES NEEDED.
2. Each subagent edits only its own module files and `tests/test_<module>.py` (plus any listed helpers).
   New dependencies go into `requirements*.txt` and are listed in the report.
3. Type hints, docstrings, and tests (>= 80% coverage) on everything. No TODO stubs, placeholder
   code, or `NotImplementedError`.
4. Run `pytest` for your module (and the contract tests) before reporting done. Paste the real output.
5. Report format: `DONE: … | TESTS: … | METRICS: … | BLOCKERS: … | CONTRACT CHANGES NEEDED: …`
6. .ai output = Illustrator-compatible PDF with .ai extension (native .ai is proprietary),
   with one named layer per VectorLayer.
7. Progress is tracked in PROGRESS.md. Update it after each task.
8. Samples: `samples/NN_*.png|jpg` with ground truth in `samples/NN_*.json`. Regenerate with
   `python samples/generate.py`. Never hand-edit.

## Commands
- Install: `pip install -r requirements-dev.txt`
- Tests: `pytest -q` (single module: `pytest tests/test_quantize.py --cov=pipeline.quantize`)
- Eval: `python -m eval run --all`
- Regenerate OpenAPI: `python scripts/export_openapi.py` (a test fails if it is stale)
- App: `docker compose up` (then http://localhost:8080)
