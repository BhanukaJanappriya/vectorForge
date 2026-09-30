"""``python -m eval`` command line.

    python -m eval run --all              # every sample in samples/
    python -m eval run 01_logo_4color     # one sample (stem, file name or numeric prefix)
    python -m eval run --all --oracle     # ignore pipeline stages, oracle stubs only
    python -m eval list                   # list samples

Prints the results table, the stage status table, the ground-truth comparison and the
corrupted-variant self-check, and writes eval/reports/<timestamp>/{report.html,results.json}.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from eval.corrupt import DEFAULT_PLAN, CorruptionResult, run_plan
from eval.harness import SampleResult, find_sample, list_samples, run_sample
from eval.report import corruption_table, ground_truth_table, results_table, stage_table, write_report

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = ROOT / "samples"
DEFAULT_REPORTS = Path(__file__).resolve().parent / "reports"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval", description="VectorForge quality evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="convert and evaluate samples")
    target = run.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="run every sample")
    target.add_argument("sample", nargs="?", help="sample stem, file name or numeric prefix")
    run.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES, help="samples directory")
    run.add_argument("--out", type=Path, default=None, help="report directory (default eval/reports/<timestamp>)")
    run.add_argument("--oracle", action="store_true", help="use oracle stubs for every stage")
    run.add_argument("--no-corruptions", action="store_true", help="skip the corrupted-variant self-check")
    run.add_argument("--strict", action="store_true", help="exit 1 if any sample or self-check fails")
    ls = sub.add_parser("list", help="list samples")
    ls.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    return parser


def _run(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    samples_dir: Path = args.samples
    try:
        paths = list_samples(samples_dir) if args.all else [find_sample(samples_dir, args.sample)]
    except (FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report_dir: Path = args.out or DEFAULT_REPORTS / datetime.now().strftime("%Y%m%d-%H%M%S")
    results: list[SampleResult] = []
    for path in paths:
        print(f"running {path.stem} ...", flush=True)
        results.append(run_sample(path, report_dir / path.stem, use_pipeline=not args.oracle))
    corruptions: list[CorruptionResult] = []
    if not args.no_corruptions:
        available = {p.stem for p in list_samples(samples_dir)}
        plan = [(stem, name) for stem, name in DEFAULT_PLAN if stem in available]
        for stem, name in DEFAULT_PLAN:
            if stem not in available:
                print(f"self-check {name}: skipped, sample {stem} not in {samples_dir}")
        print("running corrupted-variant self-check ...", flush=True)
        corruptions = run_plan(samples_dir, report_dir, plan) if plan else []
    total = time.perf_counter() - t0
    html_path = write_report(results, corruptions, report_dir, total)
    print()
    print("RESULTS ('*' = produced by an eval oracle stub; real = pipeline stages used; time = pipeline stages only;")
    print("'p95 px dE (diag)' = 95th-percentile per-pixel CIEDE2000, diagnostic only, not a pass/fail check)")
    print(results_table(results))
    print()
    print("STAGES")
    print(stage_table(results))
    print()
    print("GROUND TRUTH")
    print(ground_truth_table(results))
    for r in results:
        if r.error:
            print(f"\n{r.sample}: ERROR {r.error}")
    if corruptions:
        print()
        print("METRIC SELF-CHECK (corrupted oracle variants)")
        print(corruption_table(corruptions))
    passed = sum(r.passed for r in results)
    print()
    print(f"{passed}/{len(results)} samples pass; total {time.perf_counter() - t0:.1f}s")
    print(f"report: {html_path}")
    print(f"json:   {html_path.parent / 'results.json'}")
    failed = passed < len(results) or any(not c.ok for c in corruptions)
    return 1 if (args.strict and failed) else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        for p in list_samples(args.samples):
            print(p.stem)
        return 0
    return _run(args)
