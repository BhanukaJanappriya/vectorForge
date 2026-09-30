"""Console tables, results.json and report.html for an eval run."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from contracts.schemas import SCHEMA_VERSION, MetricName
from contracts.stages import STAGE_ENTRYPOINTS
from eval.corrupt import CorruptionResult
from eval.evaluate import composite_over_white, gap_mask, load_preview
from eval.harness import PIPELINE_STAGES, SampleResult

THUMB_MAX = 720


# --------------------------------------------------------------------------------------
# Text tables (ASCII only: Windows consoles are often cp1252)
# --------------------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows]
    return "\n".join([line, sep, *body])


def _size(n: int) -> str:
    return f"{n / 1024:.1f}KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.2f}MB"


def _diag(r: SampleResult) -> str:
    """p95 per-pixel Delta-E diagnostic (not a contract metric)."""
    return "-" if r.p95_pixel_delta_e is None else f"{r.p95_pixel_delta_e:.2f}"


def _failed_checks(r: SampleResult) -> list[str]:
    return [] if r.report is None else [c.name.value for c in r.report.checks if not c.passed]


def results_table(results: list[SampleResult]) -> str:
    """One row per sample. '*' marks values produced by an oracle stub instead of the pipeline."""
    headers = [
        "sample",
        "class (got/gt)",
        "colors",
        "SSIM",
        "mean dE",
        "max dE",
        "p95 px dE (diag)",
        "gap",
        "nodes",
        "size",
        "time",
        "real",
        "pass",
    ]
    rows = []
    for r in results:
        cls_mark = "*" if r.stages.get("classify") and r.stages["classify"].source != "pipeline" else ""
        q_mark = "*" if r.stages.get("quantize") and r.stages["quantize"].source != "pipeline" else ""
        gt_n = len(r.truth["palette_hex"]) if r.truth.get("palette_hex") else "-"
        cls = f"{r.image_class or '?'}{cls_mark}/{r.truth.get('expected_class')}"
        colors = f"{r.n_colors if r.n_colors is not None else '?'}{q_mark}/{gt_n}"
        real = f"{r.real_stages}/{len(PIPELINE_STAGES)}"
        if r.report is None:
            rows.append([r.sample, cls, colors, "-", "-", "-", "-", "-", "-", "-", "-", real, "ERROR"])
            continue
        q = r.report
        verdict = "PASS" if r.passed else "FAIL: " + ",".join(_failed_checks(r))
        rows.append(
            [
                r.sample,
                cls,
                colors,
                f"{q.ssim:.4f}",
                f"{q.mean_delta_e:.2f}",
                f"{q.max_delta_e:.2f}",
                _diag(r),
                f"{q.gap_ratio:.5f}",
                str(q.node_count),
                _size(q.file_size_bytes),
                f"{q.processing_time_s:.2f}s",
                real,
                verdict,
            ]
        )
    return _table(headers, rows)


def stage_table(results: list[SampleResult]) -> str:
    """One row per stage: aggregated status across samples."""
    rows = []
    for name in PIPELINE_STAGES:
        statuses = Counter(r.stages[name].status for r in results if name in r.stages)
        sources = Counter(r.stages[name].source for r in results if name in r.stages)
        details = sorted({r.stages[name].detail for r in results if name in r.stages and r.stages[name].detail})
        status = ", ".join(f"{s} x{n}" if len(statuses) > 1 else s for s, n in statuses.items()) or "-"
        used = ", ".join(f"{s} x{n}" for s, n in sources.items())
        rows.append([name, STAGE_ENTRYPOINTS[name], status, used, (details[0] if details else "")[:70]])
    return _table(["stage", "entrypoint", "status", "used", "detail"], rows)


def ground_truth_table(results: list[SampleResult]) -> str:
    rows = []
    for r in results:
        gt = r.ground_truth
        if not gt:
            rows.append([r.sample, "-", "-", "-"])
            continue
        c = gt["class"]
        origin = ", oracle" if c["source"] != "pipeline" else ""
        cls = f"{'ok' if c['match'] else 'MISMATCH'} ({c['got']} vs {c['expected']}{origin})"
        p = gt["palette"]
        if p.get("match") is None:
            pal = f"n/a ({p.get('note')})"
        else:
            pal = f"{'ok' if p['match'] else 'MISMATCH'} {p['got_n']} vs {p['expected_n']}"
            if p["unmatched_gt"]:
                pal += f"; missing {p['unmatched_gt']}"
            if p["extra"]:
                pal += "; extra " + ", ".join(f"{e['hex']}({e['explanation'] or 'unexplained'})" for e in p["extra"])
            if p.get("source") != "pipeline":
                pal += " (oracle)"
        s = gt["strokes"]
        if s.get("match") is None:
            strokes = f"n/a ({s.get('note')})"
        else:
            strokes = f"{'ok' if s['match'] else 'MISMATCH'} {s['got']:.2f} vs {s['expected']:.2f}px"
        rows.append([r.sample, cls, pal, strokes])
    return _table(["sample", "class", "palette", "stroke width"], rows)


def corruption_table(corruptions: list[CorruptionResult]) -> str:
    rows = [
        [
            c.sample,
            c.corruption,
            ",".join(c.expected),
            ",".join(c.flipped) or "-",
            ",".join(c.baseline_failed) or "-",
            f"{c.metrics['ssim']:.4f}",
            f"{c.metrics['mean_delta_e']:.2f}/{c.metrics['max_delta_e']:.2f}",
            f"{c.metrics['gap_ratio']:.5f}",
            "OK" if c.ok else "NOT DETECTED",
        ]
        for c in corruptions
    ]
    headers = [
        "sample",
        "corruption",
        "must fail",
        "newly failing",
        "failing on clean",
        "SSIM",
        "dE mean/max",
        "gap",
        "result",
    ]
    return _table(headers, rows)


# --------------------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------------------


def _thumb(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = min(1.0, THUMB_MAX / max(h, w))
    if scale >= 1.0:
        return arr
    return cv2.resize(arr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def write_images(r: SampleResult, out_dir: Path) -> dict[str, str]:
    """original.png, preview.png, diff.png (|source - preview| heatmap) and gaps.png."""
    if r.preview_path is None:
        return {}
    with Image.open(r.path) as img:
        src = np.asarray(img.convert("RGBA"))
    h, w = src.shape[:2]
    prev = load_preview(Path(r.preview_path), w, h)
    src_c = composite_over_white(src[..., :3], src[..., 3])
    prev_c = composite_over_white(prev[..., :3], prev[..., 3])
    diff = np.abs(src_c - prev_c).max(axis=2)
    heat = cv2.applyColorMap(np.clip(diff * 4 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    gaps = gap_mask(src[..., 3] if r.truth.get("has_alpha") else None, prev[..., 3])
    gray = (composite_over_white(src[..., :3], src[..., 3]).mean(axis=2) * 160 + 60).astype(np.uint8)
    gap_img = np.stack([gray] * 3, axis=-1)
    if gaps.any():
        gap_img[cv2.dilate(gaps.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = (255, 0, 255)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"original": src, "preview": prev, "diff": heat, "gaps": gap_img}
    names = {}
    for key, arr in files.items():
        name = f"{key}.png"
        Image.fromarray(_thumb(np.ascontiguousarray(arr))).save(out_dir / name, compress_level=1)
        names[key] = f"{out_dir.name}/{name}"
    return names


# --------------------------------------------------------------------------------------
# JSON + HTML
# --------------------------------------------------------------------------------------


def results_json(
    results: list[SampleResult], corruptions: list[CorruptionResult], total_seconds: float
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "total_seconds": total_seconds,
        "passed": sum(r.passed for r in results),
        "total": len(results),
        "samples": [r.to_json() for r in results],
        "corruptions": [c.to_json() for c in corruptions],
    }


_CSS = """
body{font-family:system-ui,Segoe UI,sans-serif;margin:24px;color:#222}
table{border-collapse:collapse;margin:8px 0 24px}td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}
th{background:#f2f2f2}.pass{background:#dff5e1}.fail{background:#fbe0e0}.skip{background:#fff4d6}
.imgs{display:flex;gap:8px;flex-wrap:wrap}.imgs figure{margin:0}.imgs img{max-width:360px;border:1px solid #ccc;
background:repeating-conic-gradient(#ddd 0 25%,#fff 0 50%) 0 0/16px 16px}
code,pre{font-size:12px}figcaption{font-size:12px;color:#555}
"""


def _cell(text: str, ok: bool | None) -> str:
    cls = "" if ok is None else ("pass" if ok else "fail")
    return f'<td class="{cls}">{html.escape(text)}</td>'


def render_html(
    results: list[SampleResult],
    corruptions: list[CorruptionResult],
    images: dict[str, dict[str, str]],
    total_seconds: float,
) -> str:
    """Self-contained HTML report (images referenced relatively)."""
    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'><title>VectorForge eval</title><style>{_CSS}</style>"
        "</head><body>",
        "<h1>VectorForge quality report</h1>",
        f"<p>{sum(r.passed for r in results)}/{len(results)} samples pass. Total {total_seconds:.1f} s. "
        "Values marked * come from eval oracle stubs, not the pipeline.</p>",
    ]
    parts.append(
        "<h2>Results</h2><table><tr>"
        + "".join(
            f"<th>{h}</th>"
            for h in [
                "sample",
                "class",
                "colors",
                "SSIM",
                "mean ΔE",
                "max ΔE",
                "p95 px ΔE (diag)",
                "gap ratio",
                "alpha IoU",
                "nodes",
                "size",
                "time",
                "real stages",
                "pass",
            ]
        )
        + "</tr>"
    )
    for r in results:
        parts.append("<tr>" + f'<td><a href="#{r.sample}">{html.escape(r.sample)}</a></td>')
        if r.report is None:
            parts.append(f'<td colspan="12">{html.escape((r.error or "")[:200])}</td>' + _cell("ERROR", False))
            parts.append("</tr>")
            continue
        chk = {c.name: c.passed for c in r.report.checks}
        q = r.report
        gt_cls = r.ground_truth.get("class", {})
        parts.append(_cell(f"{r.image_class} / {r.truth['expected_class']}", gt_cls.get("match")))
        parts.append(_cell(str(r.n_colors), r.ground_truth.get("palette", {}).get("match")))
        parts.append(_cell(f"{q.ssim:.4f}", chk.get(MetricName.SSIM)))
        parts.append(_cell(f"{q.mean_delta_e:.2f}", chk.get(MetricName.MEAN_DELTA_E)))
        parts.append(_cell(f"{q.max_delta_e:.2f}", chk.get(MetricName.MAX_DELTA_E)))
        parts.append(_cell(_diag(r), None))
        parts.append(_cell(f"{q.gap_ratio:.5f}", chk.get(MetricName.GAP_RATIO)))
        parts.append(_cell("-" if q.alpha_iou is None else f"{q.alpha_iou:.4f}", chk.get(MetricName.ALPHA_IOU)))
        parts.append(_cell(str(q.node_count), None))
        parts.append(_cell(_size(q.file_size_bytes), chk.get(MetricName.SVG_VALID)))
        parts.append(_cell(f"{q.processing_time_s:.2f}s", chk.get(MetricName.PROCESSING_TIME)))
        parts.append(_cell(f"{r.real_stages}/{len(PIPELINE_STAGES)}", None))
        parts.append(_cell("PASS" if r.passed else "FAIL", r.passed) + "</tr>")
    parts.append("</table>")

    parts.append(
        "<h2>Stages</h2><table><tr><th>sample</th>" + "".join(f"<th>{s}</th>" for s in PIPELINE_STAGES) + "</tr>"
    )
    for r in results:
        parts.append(f"<tr><td>{html.escape(r.sample)}</td>")
        for s in PIPELINE_STAGES:
            run = r.stages.get(s)
            if run is None:
                parts.append("<td>-</td>")
                continue
            ok = True if run.source == "pipeline" else (None if run.status.startswith("N/A") else False)
            label = f"{run.status} [{run.source}] {run.seconds:.2f}s"
            cls = "pass" if ok else ("" if ok is None else "skip")
            parts.append(f'<td class="{cls}" title="{html.escape(run.detail)}">{html.escape(label)}</td>')
        parts.append("</tr>")
    parts.append("</table>")

    if corruptions:
        parts.append(
            "<h2>Metric self-check: corrupted variants</h2><table><tr>"
            + "".join(
                f"<th>{h}</th>"
                for h in [
                    "sample",
                    "corruption",
                    "must fail",
                    "newly failing",
                    "failing on clean",
                    "SSIM",
                    "ΔE mean/max",
                    "gap",
                    "result",
                ]
            )
            + "</tr>"
        )
        for c in corruptions:
            parts.append(
                "<tr>"
                + "".join(
                    _cell(t, None)
                    for t in [
                        c.sample,
                        c.corruption,
                        ", ".join(c.expected),
                        ", ".join(c.flipped) or "-",
                        ", ".join(c.baseline_failed) or "-",
                        f"{c.metrics['ssim']:.4f}",
                        f"{c.metrics['mean_delta_e']:.2f} / {c.metrics['max_delta_e']:.2f}",
                        f"{c.metrics['gap_ratio']:.5f}",
                    ]
                )
                + _cell("OK" if c.ok else "NOT DETECTED", c.ok)
                + "</tr>"
            )
        parts.append("</table>")

    parts.append("<h2>Samples</h2>")
    for r in results:
        parts.append(f'<h3 id="{r.sample}">{html.escape(r.sample)}</h3>')
        parts.append(f"<p>{html.escape(r.truth.get('notes', ''))}</p>")
        if r.error:
            parts.append(f"<pre>{html.escape(r.error)}</pre>")
        imgs = images.get(r.sample, {})
        if imgs:
            parts.append(
                '<div class="imgs">'
                + "".join(
                    f'<figure><img src="{html.escape(src)}" alt="{k}"><figcaption>{k}</figcaption></figure>'
                    for k, src in imgs.items()
                )
                + "</div>"
            )
        if r.regions:
            parts.append(
                "<table><tr><th>layer</th><th>role</th><th>pixels</th><th>layer LAB</th><th>source median"
                " LAB</th><th>ΔE</th></tr>"
            )
            for g in sorted(r.regions, key=lambda g: -g.delta_e):
                parts.append(
                    "<tr>"
                    + "".join(
                        _cell(t, None)
                        for t in [
                            g.layer_id,
                            g.role,
                            str(g.pixels),
                            ", ".join(f"{v:.1f}" for v in g.layer_lab),
                            ", ".join(f"{v:.1f}" for v in g.source_lab),
                        ]
                    )
                    + _cell(f"{g.delta_e:.2f}", g.delta_e < 3.0)
                    + "</tr>"
                )
            parts.append("</table>")
        extras = {"ground truth": r.ground_truth, "svg problems": r.svg_problems, "warnings": r.warnings}
        parts.append(f"<pre>{html.escape(json.dumps(extras, indent=1, default=str))}</pre>")
    parts.append("</body></html>")
    return "\n".join(parts)


def write_report(
    results: list[SampleResult], corruptions: list[CorruptionResult], report_dir: Path, total_seconds: float
) -> Path:
    """Write images, results.json and report.html into report_dir; return the HTML path."""
    report_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, dict[str, str]] = {}
    for r in results:
        try:
            images[r.sample] = write_images(r, report_dir / r.sample)
        except (OSError, ValueError) as exc:
            r.warnings.append(f"report images failed: {exc}")
    (report_dir / "results.json").write_text(
        json.dumps(results_json(results, corruptions, total_seconds), indent=2, default=str), encoding="utf-8"
    )
    out = report_dir / "report.html"
    out.write_text(render_html(results, corruptions, images, total_seconds), encoding="utf-8")
    return out
