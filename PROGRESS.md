# VectorForge — Progress

Contract version: **1.0.0**. Last updated: 2026-09-30.

| Phase | Subagent | Module(s) | Status | Tests | Key metrics | Notes |
|---|---|---|---|---|---|---|
| 0 | main session | contracts, OpenAPI, samples, skeleton, CLAUDE.md, subagents | ✅ Done | 44 contract + 11 sample tests | contracts coverage 98% | |
| 1 | preprocessor | pipeline/preprocess.py, pipeline/classify.py | ✅ Done | 160 + 3 slow, 99% cov | 10/10 classes correct; preprocess 0.07 s, classify 0.05 s on 09; JPEG 06: 68.6% fewer colors, edge sharpness +7.2% | 2x upscale is nearest-neighbour (bilinear blended thin-line colors) |
| 1 | color-quantizer | pipeline/quantize.py | ✅ Done | 57 + 1 slow, 99% cov | exact color count on all 8 ground-truth samples; max ΔE 0.00 (06 raw JPEG 1.26); 09 in 0.7–0.9 s | Own deterministic k-means (sklearn overhead too high) |
| 1 | line-extractor | pipeline/lines.py | ✅ Done | 22 + 1 slow, 98% cov | 02: 3/3 strokes, junctions intact; 08: 12/12 grid lines; 03 outline IoU 0.913; 0.27–0.46 s @ 4 MP | Width estimate runs +1 px on odd widths (within tolerance) |
| 1 | frontend-builder | frontend/ | ✅ Done | Vitest 88, Playwright 6 (1440 + 375 px) | 89% statements; JS 57.9 kB gzip | Runs against MSW mock of the OpenAPI spec |
| 1 | qa-evaluator | eval/ | ✅ Done | 96, 99% cov | all 4 corruption self-checks caught; Sharma CIEDE2000 pairs to 1e-4; full run ~10 s | vectorize/assemble/export use eval stand-ins until Phase 2 |
| 2 | vectorizer | pipeline/vectorize.py | ⏳ Next | — | — | |
| 2 | svg-exporter | pipeline/assemble.py, pipeline/export.py | ⏳ Next | — | — | |
| 2 | backend-devops | api/, pipeline/runner.py, Docker | ⏳ Next | — | — | |
| 3 | all | integration & tuning loop | ⏳ | — | — | |

## Results: `python -m eval run --all` (2026-09-30, end of Phase 1)

Real stages: load_image, preprocess, classify, quantize, extract_lines. **Vectorize, assemble and export are eval
stand-ins (pixel-run rectangles)**, so SSIM and node counts below measure the upstream stages, not final output.

| Sample | Class | Colors | SSIM | Mean ΔE | Max ΔE | Gap | Pass |
|---|---|---|---|---|---|---|---|
| 01_logo_4color | flat_color ✓ | 4/4 | 0.9962 | 0.00 | 0.00 | 0 | ✅ |
| 02_lineart_black | line_art ✓ | 2/2 | 0.9888 | 0.00 | 0.00 | 0 | ✅ |
| 03_cartoon_outlined | mixed ✓ | 8/8 | 0.9945 | 0.00 | 0.00 | 0 | ✅ |
| 04_text | flat_color ✓ | 4/4 | 0.9786 | 0.59 | 2.34 | 0 | ✅ |
| 05_gradient | mixed ✓ | 19 | 0.9790 | 0.13 | 0.49 | 0 | ✅ |
| 06_jpeg_artifacts | flat_color ✓ | 4/4 | 0.9810 | 0.00 | 0.00 | 0 | ✅ |
| 07_transparent_logo | flat_color ✓ | 3/3 | 0.9894 | 0.00 | 0.00 | 0 | ✅ |
| 08_thin_lines | line_art ✓ | 3/3 | 1.0000 | 0.00 | 0.00 | 0 | ✅ |
| 09_large_2000 | flat_color ✓ | 7/7 | 0.9890 | 0.00 | 0.00 | 0 | ✅ |
| 10_mixed_scene | mixed ✓ | 9 | 0.9279 | 0.07 | 0.21 | 0 | ✅ |

## Decisions log

| # | Decision | Rationale |
|---|---|---|
| D1 | `VectorLayer.paths` limited to absolute `M/L/C/Z` | Maps 1:1 onto PDF/EPS operators, so svg-exporter can write .ai directly with named layers (OCGs) if Inkscape's PDF export drops them. |
| D2 | .ai goes through the Inkscape CLI first, with the direct PDF writer as fallback | Follows the spec. The acceptance test is that layers survive (pikepdf OCG check). |
| D3 | SSIM threshold for MIXED = 0.85 | The spec only defines flat (0.90) and line art (0.85). |
| D4 | Time budget scales with pixel count | 10 s at 4 MP, minimum 2 s (`QualityThresholds.time_budget_s`). |
| D5 | `preview.png` always rendered | QA needs it regardless of the requested formats. |
| D6 | Arrays in contracts are read-only views | Enforces pure stages; callers keep writeable arrays. |
| D7 | OpenAPI generated from `contracts/api.py`; frontend types generated from the OpenAPI file | Single source of truth; drift tests on both sides. |
| D8 | Background removal happens in preprocess (sets alpha + `background_removed`) | Downstream stages treat it as transparency. |
| D9 | Palette edit/merge uses `Settings.palette_override` + `POST /jobs/{id}/rerun` | Frontend "edit/merge colors and re-run" feature. |
| D10 | `gap_ratio` metric (≤ 0.0005) and mean ΔE < 2 check | Makes "no hairline gaps" and mean-ΔE measurable. |
| D11 | `pipeline/runner.py` owned by backend-devops; eval composes stages itself until it exists | Single composition root. |
| D12 | Job TTL = 1 hour | backend-devops spec. |
| D13 | `PreprocessResult.background_lab` added (Phase 1) | Replaces hidden `DenoiseParams.extra` keys that classify depended on. |
| D14 | `QualityThresholds.MIN_ALPHA_IOU = 0.98` (Phase 1) | Proposed by qa-evaluator; alpha IoU had no threshold. |
| D15 | ΔE is measured against ORIGINAL source pixels, not `pre.image` | Using `pre.image` hid the preprocess color-blending bug on 08. |
| D16 | Timing tests are `@pytest.mark.slow`, excluded from `pytest -q`, run with `pytest -m slow` | This laptop throttles under sustained load; in-suite timing was flaky. The eval time check stays authoritative. |

## Open questions / risks

- **p95 per-pixel ΔE** (diagnostic column in eval) catches merged colors that the region-median ΔE misses,
  but is non-zero on gradients/JPEG/noise (05: 6.08, 06: 4.77, 10: 3.04). Decide in Phase 3 whether it becomes
  a class-dependent contract metric.
- Sample 05 (pure gradient) may not reach SSIM 0.85 with real flat-fill vectorization at a sane node count.
- Centerline tracing of 1-px aliased lines (sample 08) is the highest Phase 2 risk.
- Eval time checks can fail spuriously when the dev machine is low on RAM (seen once on 10: quantize 2.0 s vs 0.5 s solo).
- The native Cairo DLL is missing on the Windows dev machine (`import cairosvg` fails). svg-exporter needs it
  locally (GTK3 runtime) or must test in Docker.
- Inkscape is not installed on the dev machine, so .ai/.eps via Inkscape can only be verified in Docker.

## Notes for Phase 2 (from Phase 1 reports)

- frontend sends `settings` as a plain multipart string field; backend should use `Form(str)`. Error bodies must use
  the spec shape `{"error": {...}}`. Media types: `.ai` → `application/illustrator`, `.eps` → `application/postscript`.
- LineMap widths run ~1 px high on odd-width strokes; LINE_ART solid ink regions wider than 32 px are left to the fill path.
- eval harness needs a per-stage hook when switching to `pipeline/runner.py`, to keep SKIPPED rows and stand-ins.
