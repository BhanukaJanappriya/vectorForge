# VectorForge — Progress

Contract version: **1.0.0**. Last updated: 2026-09-29.

| Phase | Subagent | Module(s) | Status | Tests | Key metrics | Blockers |
|---|---|---|---|---|---|---|
| 0 | main session | contracts, OpenAPI, samples, skeleton, CLAUDE.md, subagents | ✅ Done — awaiting human review + commit | 55 passed | contracts coverage 98% | — |
| 1 | preprocessor | pipeline/preprocess.py, pipeline/classify.py | ⏳ Not started | — | — | — |
| 1 | color-quantizer | pipeline/quantize.py | ⏳ Not started | — | — | — |
| 1 | line-extractor | pipeline/lines.py | ⏳ Not started | — | — | — |
| 1 | frontend-builder | frontend/ | ⏳ Not started | — | — | — |
| 1 | qa-evaluator | eval/ | ⏳ Not started | — | — | — |
| 2 | vectorizer | pipeline/vectorize.py | ⏳ Waiting for Phase 1 | — | — | — |
| 2 | svg-exporter | pipeline/assemble.py, pipeline/export.py | ⏳ Waiting for Phase 1 | — | — | — |
| 2 | backend-devops | api/, pipeline/runner.py, Docker | ⏳ Waiting for Phase 1 | — | — | — |
| 3 | all | integration & tuning loop | ⏳ | — | — | — |

## Results (filled by `python -m eval run --all`)

_No results yet._

## Decisions log

| # | Decision | Rationale |
|---|---|---|
| D1 | `VectorLayer.paths` limited to absolute `M/L/C/Z` | Maps 1:1 onto PDF/EPS operators, so svg-exporter can write .ai directly with named layers (OCGs) if Inkscape's PDF export drops them. vtracer and potrace already emit only these commands. |
| D2 | .ai goes through the Inkscape CLI first, with the direct PDF writer as fallback | Follows the spec. The acceptance test is that layers survive (pikepdf OCG check). |
| D3 | SSIM threshold for MIXED = 0.85 | The spec only defines flat (0.90) and line art (0.85). |
| D4 | Time budget scales with pixel count | 10 s at 4 MP, minimum 2 s (`QualityThresholds.time_budget_s`). |
| D5 | `preview.png` always rendered | QA needs it regardless of the requested formats. |
| D6 | Arrays in contracts are read-only views | Enforces pure stages; callers keep writeable arrays. |
| D7 | OpenAPI generated from `contracts/api.py` | Single source of truth; a test fails on drift. |
| D8 | Background removal happens in preprocess (sets alpha + `background_removed`) | Downstream stages then treat the removed background as transparency with no special cases. |
| D9 | Palette edit/merge uses `Settings.palette_override` + `POST /jobs/{id}/rerun` | Required by the frontend "edit/merge colors and re-run" feature. |
| D10 | `gap_ratio` metric (≤ 0.0005) and mean ΔE < 2 check added to QualityReport | Makes "no hairline gaps" and the mean-ΔE requirement measurable. |
| D11 | `pipeline/runner.py` owned by backend-devops; eval uses it once it exists | Single composition root; stage modules never import each other. |
| D12 | Job TTL = 1 hour | backend-devops spec. |

## Open questions / risks

- Sample 05 (pure gradient) may not reach SSIM 0.85 with flat fills at a sane node count. It is currently
  in pass/fail as the spec says "all samples". Revisit in Phase 3.
- Centerline tracing of 1-px aliased lines (sample 08) is the highest algorithmic risk.
- The native Cairo DLL is missing on the Windows dev machine (`import cairosvg` fails). svg-exporter and
  qa-evaluator need it locally (GTK3 runtime) or must test in Docker.
- Inkscape is not installed on the dev machine, so .ai/.eps via Inkscape can only be verified in Docker.
