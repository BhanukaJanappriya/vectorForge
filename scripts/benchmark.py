"""Reproducible pipeline benchmark: per-stage time, peak memory, geometry and file sizes.

    python scripts/benchmark.py                    # every sample, 3 repeats, markdown table
    python scripts/benchmark.py --repeats 5 --json out.json
    python scripts/benchmark.py --samples 09 10    # numeric prefixes or stems

Each (sample, repeat) runs in a fresh subprocess so ``ru_maxrss`` is that run's own peak
resident memory, not a process-wide high-water mark. Time and memory columns are medians
over the repeats. SSIM, CIEDE2000 and gap come from the pipeline's own ``QualityReport``.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def _worker(sample: Path) -> dict[str, Any]:
    """Run the full pipeline once in this process and return its measurements."""
    sys.path.insert(0, str(ROOT))
    from contracts.schemas import Settings  # noqa: PLC0415
    from eval.raster import document_node_count  # noqa: PLC0415
    from pipeline.runner import run_pipeline  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        result = run_pipeline(sample, Settings(), Path(tmp))
        bundle, report = result.bundle, result.report
        sizes = {
            name: path.stat().st_size
            for name, path in (("svg", bundle.svg_path), ("eps", bundle.eps_path), ("ai", bundle.ai_path))
            if path is not None
        }
        return {
            "sample": sample.stem,
            "elapsed_s": result.elapsed_s,
            "processing_time_s": result.processing_time_s,
            "stages_s": result.timings,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "nodes": document_node_count(result.doc),
            "layers": len(result.doc.layers),
            "sizes_bytes": sizes,
            "ssim": report.ssim,
            "mean_delta_e": report.mean_delta_e,
            "max_delta_e": report.max_delta_e,
            "gap_ratio": report.gap_ratio,
            "passed": report.passed,
            "warnings": list(bundle.warnings),
        }


def _select(prefixes: list[str] | None) -> list[Path]:
    paths = sorted(p for p in SAMPLES.glob("[0-9][0-9]_*") if p.suffix in {".png", ".jpg"})
    if not prefixes:
        return paths
    return [p for p in paths if any(p.stem == q or p.stem.startswith(q) for q in prefixes)]


def _run_once(sample: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(sample)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{sample.stem} failed:\n{proc.stderr[-800:]}")
    return json.loads(proc.stdout.splitlines()[-1])


def _summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Median the time and memory of repeated runs; geometry and quality are deterministic."""
    first = runs[0]
    stage_names = first["stages_s"].keys()
    return {
        **first,
        "repeats": len(runs),
        "elapsed_s": statistics.median(r["elapsed_s"] for r in runs),
        "processing_time_s": statistics.median(r["processing_time_s"] for r in runs),
        "peak_rss_mb": statistics.median(r["peak_rss_mb"] for r in runs),
        "stages_s": {k: statistics.median(r["stages_s"][k] for r in runs) for k in stage_names},
    }


def _table(rows: list[dict[str, Any]]) -> str:
    head = "| Sample | Time s | Peak MB | Nodes | EPS KB | SVG KB | SSIM | Mean dE | Max dE | Gap | Slowest stages |"
    lines = [head, "|" + "|".join(["---"] + ["---:"] * 9 + ["---"]) + "|"]
    for r in rows:
        slow = sorted(r["stages_s"].items(), key=lambda kv: kv[1], reverse=True)[:2]
        eps = r["sizes_bytes"].get("eps")
        lines.append(
            f"| {r['sample']} | {r['elapsed_s']:.2f} | {r['peak_rss_mb']:.0f} | {r['nodes']} "
            f"| {eps / 1024:.1f} | {r['sizes_bytes']['svg'] / 1024:.1f} | {r['ssim']:.4f} "
            f"| {r['mean_delta_e']:.2f} | {r['max_delta_e']:.2f} | {r['gap_ratio']:.5f} "
            f"| {', '.join(f'{k} {v:.2f}' for k, v in slow)} |"
            if eps is not None
            else f"| {r['sample']} | {r['elapsed_s']:.2f} | EPS missing |"
        )
    n = len(rows)
    lines.append(
        f"| **mean** | {sum(r['elapsed_s'] for r in rows) / n:.2f} | {sum(r['peak_rss_mb'] for r in rows) / n:.0f} "
        f"| {sum(r['nodes'] for r in rows) / n:.0f} "
        f"| {sum(r['sizes_bytes'].get('eps', 0) for r in rows) / n / 1024:.1f} "
        f"| {sum(r['sizes_bytes']['svg'] for r in rows) / n / 1024:.1f} "
        f"| {sum(r['ssim'] for r in rows) / n:.4f} | | | | |"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--samples", nargs="*", help="sample stems or numeric prefixes (default: all)")
    parser.add_argument("--repeats", type=int, default=3, help="runs per sample (default 3)")
    parser.add_argument("--json", type=Path, help="also write the raw summary to this file")
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(_worker(args.worker)))
        return 0
    paths = _select(args.samples)
    if not paths:
        print("no matching samples", file=sys.stderr)
        return 1
    rows = []
    for path in paths:
        rows.append(_summarise([_run_once(path) for _ in range(args.repeats)]))
        print(f"done {path.stem}", file=sys.stderr, flush=True)
    print(_table(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
