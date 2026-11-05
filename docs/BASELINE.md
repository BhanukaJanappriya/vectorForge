# Baseline (2026-11-05)

Measured before any optimization work, from commit `f034969`. Reproduce with
`python scripts/benchmark.py --repeats 3`.

## Environment

Linux container, 4 CPUs, 16 GB RAM, Python 3.13, numpy 2.5, OpenCV headless, pycairo 1.18 (cairo PostScript
surface). Inkscape is not installed, so every export used the local fallbacks (resvg preview, direct PDF
writer for `.ai`, pycairo for `.eps`). Ghostscript 10.02 was installed for independent EPS rendering.

## Test and check results

| Check | Result |
|---|---|
| `pytest -q --cov=pipeline --cov=api --cov=eval --cov=contracts` | 604 passed, 5 skipped, 20 slow deselected, coverage 98.5%, 75 s |
| `python -m eval run --all` | 9/10 pass; sample 10 fails only `processing_time_s` (3.54 s); 4/4 corruption self-checks caught |
| Frontend | eslint clean, `tsc` + `vite build` OK (JS 57.9 kB gzip), Vitest 88/88, Playwright 6/6 (1440 px and 375 px) |

## Pipeline benchmark

Method: each sample runs in a fresh subprocess (cold start, so imports are included), 3 repeats, median of
time and peak RSS. Time is the full run including `evaluate`. SSIM, CIEDE2000 and gap are the pipeline's own
`QualityReport` values. The same method must be used for the "after" numbers.

| Sample | Time s | Peak MB | Nodes | EPS KB | SVG KB | SSIM | Mean dE | Max dE | Gap | Slowest stages |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 01_logo_4color | 1.29 | 167 | 36 | 3.5 | 1.9 | 0.9962 | 0.00 | 0.00 | 0.00000 | export 0.27, preprocess 0.15 |
| 02_lineart_black | 1.69 | 187 | 104 | 5.2 | 2.7 | 0.9894 | 0.00 | 0.00 | 0.00000 | export 0.30, vectorize 0.16 |
| 03_cartoon_outlined | 2.73 | 189 | 125 | 6.6 | 4.4 | 0.9949 | 0.00 | 0.00 | 0.00000 | quantize 0.86, vectorize 0.36 |
| 04_text | 2.47 | 172 | 779 | 19.9 | 9.6 | 0.9789 | 0.59 | 2.34 | 0.00000 | vectorize 0.56, quantize 0.53 |
| 05_gradient | 2.41 | 171 | 616 | 16.3 | 10.3 | 0.9810 | 0.13 | 0.49 | 0.00000 | vectorize 0.58, quantize 0.43 |
| 06_jpeg_artifacts | 1.61 | 167 | 89 | 4.6 | 2.5 | 0.9833 | 0.00 | 0.00 | 0.00000 | export 0.36, preprocess 0.32 |
| 07_transparent_logo | 1.13 | 169 | 46 | 3.9 | 2.0 | 0.9844 | 0.00 | 0.00 | 0.00000 | export 0.25, classify 0.22 |
| 08_thin_lines | 2.11 | 178 | 178 | 7.2 | 4.3 | 0.9615 | 0.00 | 0.00 | 0.00000 | vectorize 0.37, export 0.32 |
| 09_large_2000 | 7.16 | 726 | 1332 | 42.1 | 25.3 | 0.9901 | 0.00 | 0.00 | 0.00000 | vectorize 2.44, evaluate 1.30 |
| 10_mixed_scene | 5.44 | 194 | 5417 | 81.8 | 42.1 | 0.9321 | 0.07 | 0.21 | 0.00000 | vectorize 2.38, quantize 0.92 |
| **mean** | 2.81 | 232 | 872 | 19.1 | 10.5 | 0.9792 | | | | |

Cold-start timings are noisier and higher than a warm in-process run (sample 09: 7.16 s cold median vs 4.99 s
for a single warm run), so compare only like with like.

## EPS observations

- All 10 EPS files start with `%!PS-Adobe-3.0 EPSF-3.0`, carry a `%%BoundingBox`, contain vector operators
  (`m l c f*`) and no raster data (the only `image` text is cairo's prologue macro definition).
- Ghostscript (`-dEPSCrop`) renders all 10 with exit code 0.
- cairo writes a tight BoundingBox around the drawn content, not the full canvas. Sample 07 (transparent
  margins) gets `0 30 355 384` instead of the full page, so the EPS dimensions differ from the source.
- Cairo emits an opaque white full-canvas `rectfill` in some outputs (PostScript has no alpha), which needs a
  check against sources that are transparent.
