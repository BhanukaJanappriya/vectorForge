# VectorForge frontend

React 18 + Vite + TypeScript (strict) + Tailwind v4. Owned by the `frontend-builder` subagent.
API types are **generated** from `../api/openapi.yaml` into `src/api/schema.d.ts` and never hand-written.

## Run

```bash
npm install
npx playwright install chromium   # once, for E2E

npm run dev:mock    # http://localhost:5173 against the MSW mock API (no backend needed)
npm run dev         # http://localhost:5173, /api proxied to VITE_API_TARGET (default http://localhost:8000)
npm run build       # tsc --noEmit + production build to dist/ (the mock is NOT included)
npm run preview     # serve dist/ with the same /api proxy
```

In Docker, nginx serves `dist/` and proxies `/api` on the same origin, so the app always calls relative
`/api/v1/...` URLs.

## Verify

```bash
npm run lint          # ESLint (typescript-eslint type-checked, react-hooks, jsx-a11y strict), 0 warnings
npx tsc --noEmit      # strict type check of src/, e2e/ and configs
npm test              # Vitest component/unit tests (jsdom + MSW node server)
npm run test:coverage # same, with v8 coverage
npm run test:e2e      # Playwright (Chromium) at 1440 px and 375 px against `vite --mode mock`
npm run gen:api       # regenerate src/api/schema.d.ts after the OpenAPI spec changes
```

`src/api/schema.test.ts` fails if `schema.d.ts` is stale relative to `api/openapi.yaml`.

## Mock API (`src/mocks/`)

MSW handlers implement every path in the spec (`/health`, `/config`, `/convert`, `/jobs/{id}`, `/jobs/{id}/rerun`,
`/jobs/{id}/files/{kind}`, `DELETE /jobs/{id}`) and return spec-valid `JobResponse` payloads that walk through the
`PipelineStage` values (`extract_lines` is skipped for flat-color/outline jobs, as in `contracts/stages.py`).
The "vector result" is a hand-written SVG colored with the job's palette (one `<g>` per color, ids from `layer_name()`),
and the uploaded image stands in for `preview.png`.

Deterministic triggers:
- files larger than `limits.max_upload_bytes` -> 413; media types other than PNG/JPEG -> 415;
- bytes that are not a PNG/JPEG header -> 422 `invalid_image`; invalid settings -> 422 `invalid_settings`;
- a filename containing `fail` -> the job fails at `vectorize` with `stage_failed`;
- palette edits change the mock's ΔE/SSIM, so merging distinct colors makes the quality checks fail.

`mock-public/mockServiceWorker.js` is only served in `--mode mock` (regenerate with `npm run msw:init`).

## Structure

```
src/
  api/          schema.d.ts (generated), types.ts (aliases), enums.ts (runtime enum lists, exhaustiveness-checked),
                client.ts (fetch wrapper + ApiError), poll.ts (300 ms -> 2 s backoff)
  hooks/        useConfig (GET /config), useJobRunner (convert/rerun -> poll state machine)
  components/   UploadDropzone, SettingsPanel, ProgressIndicator, ComparisonViewer (slider/side-by-side,
                synchronized zoom/pan, checkerboard, SVG via <img> only), PaletteEditor (edit/merge/remove -> rerun),
                QualityBadge, Downloads, ResultSummary, ErrorBanner, Icons
  lib/          color, format, upload validation, viewport (pan/zoom math), download, errors
  mocks/        MSW handlers + in-memory job store (browser worker and node server)
e2e/            Playwright specs
```
