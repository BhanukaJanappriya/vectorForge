"""Harness, ground-truth comparison, report and CLI (eval/harness.py, eval/report.py, eval/cli.py)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from contracts import stages as contract_stages
from contracts.schemas import ImageClassLabel, LineMap, StageError
from eval import harness
from eval.cli import main
from eval.oracle import oracle_classify, oracle_load_image, oracle_preprocess, oracle_quantize
from eval.report import corruption_table, ground_truth_table, render_html, results_table, stage_table
from tests.test_eval_oracle import make_sample

FAKE = "vf_eval_fake_stages"
PIPELINE_KEYS = harness.PIPELINE_STAGES


@pytest.fixture
def entrypoints(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Point every pipeline stage at a fake module whose functions tests install."""
    module = types.ModuleType(FAKE)
    monkeypatch.setitem(sys.modules, FAKE, module)
    for key in PIPELINE_KEYS:
        monkeypatch.setitem(contract_stages.STAGE_ENTRYPOINTS, key, f"{FAKE}:{key}")
    return contract_stages.STAGE_ENTRYPOINTS


def _install(**fns: object) -> None:
    for name, fn in fns.items():
        setattr(sys.modules[FAKE], name, fn)


# ---------------------------------------------------------------------------- resolution


def test_resolve_stage_statuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    entry = contract_stages.STAGE_ENTRYPOINTS
    monkeypatch.setitem(entry, "vectorize", "pipeline.vf_does_not_exist:vectorize")
    assert harness.resolve_stage("vectorize").status == harness.MISSING
    monkeypatch.setitem(entry, "vectorize", "eval.oracle:no_such_function")
    res = harness.resolve_stage("vectorize")
    assert res.status == harness.MISSING and "not defined" in res.detail
    monkeypatch.setitem(entry, "vectorize", "eval.oracle:oracle_classify")
    assert harness.resolve_stage("vectorize").status == harness.OK
    # Module exists but one of ITS imports is missing -> import error, not "stage missing".
    (tmp_path / "vf_broken_dep.py").write_text("import vf_uninstalled_package_xyz\n", encoding="utf-8")
    (tmp_path / "vf_broken_code.py").write_text("raise RuntimeError('half written')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(entry, "vectorize", "vf_broken_dep:vectorize")
    assert harness.resolve_stage("vectorize").status == harness.IMPORT_ERROR
    monkeypatch.setitem(entry, "vectorize", "vf_broken_code:vectorize")
    res = harness.resolve_stage("vectorize")
    assert res.status == harness.IMPORT_ERROR and "half written" in res.detail


def test_all_stages_missing_falls_back_to_oracle(entrypoints: dict[str, str], tmp_path: Path) -> None:
    path = make_sample(tmp_path / "s")
    result = harness.run_sample(path, tmp_path / "out")
    assert result.error is None, result.error
    assert result.passed and result.real_stages == 0
    assert {s.status for k, s in result.stages.items() if k != "extract_lines"} == {harness.MISSING}
    assert result.stages["extract_lines"].status == harness.NOT_NEEDED
    assert result.report is not None and result.report.processing_time_s == 0.0
    assert result.stub_seconds > 0
    assert result.ground_truth["palette"]["match"] and result.ground_truth["class"]["source"] == "oracle"


def test_real_failing_and_wrong_type_stages(entrypoints: dict[str, str], tmp_path: Path) -> None:
    path = make_sample(tmp_path / "s")

    def bad_preprocess(image: object, settings: object) -> None:
        raise StageError("preprocess", "boom")

    _install(
        load_image=oracle_load_image,
        preprocess=bad_preprocess,
        classify=lambda pre, settings: "flat_color",  # wrong type
        quantize=lambda pre, cls, settings: oracle_quantize(pre, ["#ffffff", "#dc322f", "#268bd2", "#fac81e"]),
    )
    result = harness.run_sample(path, tmp_path / "out")
    assert result.error is None
    st = result.stages
    assert st["load_image"].status == harness.OK and st["load_image"].source == "pipeline"
    assert st["preprocess"].status == harness.FAILED and "boom" in st["preprocess"].detail
    assert st["preprocess"].source == "oracle"
    assert st["classify"].status == harness.FAILED and "returned str" in st["classify"].detail
    assert st["quantize"].source == "pipeline" and result.real_stages == 2
    assert result.report is not None and result.report.processing_time_s > 0


def test_line_art_with_and_without_lines(entrypoints: dict[str, str], tmp_path: Path) -> None:
    path = make_sample(tmp_path / "s")
    truth = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    truth.update(expected_class="line_art", median_stroke_width_px=3)
    path.with_suffix(".json").write_text(json.dumps(truth), encoding="utf-8")
    no_lines = harness.run_sample(path, tmp_path / "a")
    assert no_lines.stages["extract_lines"].status == harness.MISSING
    assert "line_map=None" in no_lines.stages["extract_lines"].detail
    assert no_lines.ground_truth["strokes"]["match"] is None

    def lines(pre: object, cls: object, settings: object) -> LineMap:
        h, w = pre.height, pre.width  # type: ignore[attr-defined]
        mask = np.zeros((h, w), bool)
        mask[5, 5:50] = True
        width_map = np.where(mask, 3.0, 0.0).astype(np.float32)
        return LineMap(
            mask=mask, skeleton=mask.copy(), width_map=width_map, median_stroke_width=3.0, color_rgb=(0, 0, 0)
        )

    _install(extract_lines=lines)
    with_lines = harness.run_sample(path, tmp_path / "b")
    assert with_lines.stages["extract_lines"].source == "pipeline"
    strokes = with_lines.ground_truth["strokes"]
    assert strokes["match"] and strokes["got"] == 3.0


def test_forced_oracle_and_broken_sample(tmp_path: Path) -> None:
    path = make_sample(tmp_path / "s")
    result = harness.run_sample(path, tmp_path / "o", use_pipeline=False)
    assert {s.status for s in result.stages.values()} == {harness.ORACLE, harness.NOT_NEEDED}
    truth = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    truth["palette_hex"] = ["#zzzzzz"]
    path.with_suffix(".json").write_text(json.dumps(truth), encoding="utf-8")
    broken = harness.run_sample(path, tmp_path / "b", use_pipeline=False)
    assert broken.error and not broken.passed and broken.report is None
    as_json = broken.to_json()
    assert as_json["report"] is None and as_json["error"]
    assert "ERROR" in results_table([broken])
    assert "ERROR" in render_html([broken], [], {}, 1.0)
    assert broken.sample in ground_truth_table([broken])


# ---------------------------------------------------------------------------- ground truth


def test_compare_palette_flags_missing_and_aa_blends() -> None:
    pre = oracle_preprocess(oracle_load_image(Path(__file__).resolve().parents[1] / "samples" / "01_logo_4color.png"))
    palette = oracle_quantize(pre, ["#ffffff", "#dc322f", "#ee9897", "#268bd2"])  # #ee9897 = red/white blend
    out = harness.compare_palette(["#ffffff", "#dc322f", "#268bd2", "#fac81e"], palette)
    assert not out["match"]
    assert out["unmatched_gt"] == ["#fac81e"]
    extra = {e["hex"]: e["explanation"] for e in out["extra"]}
    assert "AA blend of #ffffff/#dc322f" in (extra["#ee9897"] or "")


def test_expected_stroke_width_and_class_mismatch(tmp_path: Path) -> None:
    assert harness.expected_stroke_width({"median_stroke_width_px": 6}) == 6.0
    assert harness.expected_stroke_width({"stroke_widths_px": {"a": 5, "b": 3, "c": 4}}) == 4.0
    assert harness.expected_stroke_width({}) is None
    pre = oracle_preprocess(oracle_load_image(make_sample(tmp_path)))
    palette = oracle_quantize(pre, None, n_auto=4)
    runs = {k: harness.StageRun(harness.OK, "pipeline") for k in PIPELINE_KEYS}
    truth = {"expected_class": "flat_color", "palette_hex": None}
    gt = harness.compare_ground_truth(truth, oracle_classify(ImageClassLabel.MIXED), palette, None, pre, runs)
    assert gt["class"]["match"] is False and gt["palette"]["match"] is None
    assert gt["strokes"]["match"] is None


def test_find_and_list_samples(tmp_path: Path) -> None:
    make_sample(tmp_path, "90_synthetic")
    make_sample(tmp_path, "91_other")
    (tmp_path / "93_no_truth.png").write_bytes(b"x")
    assert [p.stem for p in harness.list_samples(tmp_path)] == ["90_synthetic", "91_other"]
    assert harness.find_sample(tmp_path, "90").stem == "90_synthetic"
    assert harness.find_sample(tmp_path, "91_other.png").stem == "91_other"
    assert harness.find_sample(tmp_path, "90_syn").stem == "90_synthetic"
    with pytest.raises(FileNotFoundError):
        harness.find_sample(tmp_path, "9")


# ---------------------------------------------------------------------------- CLI + report


def test_cli_list_and_unknown_sample(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_sample(tmp_path, "90_synthetic")
    assert main(["list", "--samples", str(tmp_path)]) == 0
    assert "90_synthetic" in capsys.readouterr().out
    assert main(["run", "nope", "--samples", str(tmp_path), "--out", str(tmp_path / "r")]) == 1


def test_cli_run_all_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    samples = tmp_path / "samples"
    make_sample(samples, "90_synthetic")
    bad = make_sample(samples, "91_bad_truth")
    truth = json.loads(bad.with_suffix(".json").read_text(encoding="utf-8"))
    truth["palette_hex"] = ["#ffffff", "#268bd2", "#fac81e"]  # red merged into white -> SSIM failure
    bad.with_suffix(".json").write_text(json.dumps(truth), encoding="utf-8")
    out = tmp_path / "report"
    args = ["run", "--all", "--samples", str(samples), "--out", str(out), "--oracle"]
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "RESULTS" in text and "STAGES" in text and "GROUND TRUTH" in text
    assert "skipped, sample 01_logo_4color not in" in text
    assert "1/2 samples pass" in text
    assert (out / "report.html").is_file()
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert data["total"] == 2 and data["passed"] == 1
    assert "p95 px dE (diag)" in text
    # Diagnostic only: present in results.json, never a MetricCheck.
    for sample in data["samples"]:
        assert sample["diagnostics"]["p95_pixel_delta_e"] >= 0
        assert "p95" not in json.dumps(sample["report"])
    bad_diag = next(s for s in data["samples"] if s["sample"] == "91_bad_truth")["diagnostics"]
    assert bad_diag["p95_pixel_delta_e"] > 3  # red merged into white is visible per pixel
    assert "p95 px ΔE (diag)" in (out / "report.html").read_text(encoding="utf-8")
    for name in ("original.png", "preview.png", "diff.png", "gaps.png"):
        assert (out / "90_synthetic" / name).is_file()
    assert main([*args, "--strict", "--no-corruptions"]) == 1


def test_cli_single_sample_with_corruptions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "r"
    assert main(["run", "01", "--oracle", "--out", str(out), "--strict"]) == 0
    text = capsys.readouterr().out
    assert "METRIC SELF-CHECK" in text and "NOT DETECTED" not in text
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "corrupted variants" in html and "missing_layer" in html


def test_tables_with_gaps_and_stage_mix(entrypoints: dict[str, str], tmp_path: Path) -> None:
    path = make_sample(tmp_path / "s", alpha=True)
    result = harness.run_sample(path, tmp_path / "o")
    other = harness.run_sample(path, tmp_path / "p", use_pipeline=False)
    table = stage_table([result, other])
    assert harness.MISSING in table and "x1" in table
    assert "alpha" not in corruption_table([])
    from eval.report import write_images

    images = write_images(result, tmp_path / "imgs" / result.sample)
    assert set(images) == {"original", "preview", "diff", "gaps"}
    result.preview_path = None
    assert write_images(result, tmp_path / "none") == {}
