"""Tests for eval/eps_validate.py: structure, vector content, raster detection, canvas and render checks."""

from __future__ import annotations

import re
from pathlib import Path

import cairo
import numpy as np
import pytest

from contracts.fixtures import make_vector_document
from eval.cli import main as eval_main
from eval.eps_validate import (
    PX_TO_PT,
    EpsCheck,
    find_ghostscript,
    format_report,
    render_eps,
    render_on_canvas,
    render_similarity,
    validate_eps,
)
from pipeline.export import write_cairo_eps

needs_gs = pytest.mark.skipif(find_ghostscript() is None, reason="ghostscript not installed")


def status(report, name: str) -> str:
    return next(c.status for c in report.checks if c.name == name)


@pytest.fixture()
def good_eps(tmp_path: Path) -> Path:
    doc = make_vector_document(64)
    path = tmp_path / "ok.eps"
    write_cairo_eps(doc, path)
    return path


def test_valid_cairo_eps_passes(good_eps: Path) -> None:
    report = validate_eps(good_eps, expected_size=(64, 64), render=False)
    assert report.passed, [c for c in report.checks if c.status == "fail"]
    assert report.path_ops > 0 and report.paint_ops > 0
    assert status(report, "vector_content") == "pass"
    assert status(report, "no_raster") == "pass"
    assert report.to_dict()["passed"] is True
    assert "PASS" in format_report(report)


def test_macro_definitions_in_prologue_are_not_raster(good_eps: Path) -> None:
    text = good_eps.read_text(encoding="latin-1")
    assert "cairo_image" in text.split("%%EndProlog")[0]
    assert status(validate_eps(good_eps, render=False), "no_raster") == "pass"


def test_empty_file_fails(tmp_path: Path) -> None:
    path = tmp_path / "e.eps"
    path.write_bytes(b"")
    report = validate_eps(path)
    assert not report.passed and report.size_bytes == 0


@pytest.mark.parametrize(
    ("mutate", "check"),
    [
        (lambda t: t.replace("%!PS-Adobe-3.0 EPSF-3.0", "%!PS-Adobe-3.0", 1), "header"),
        (lambda t: t.replace("%%BoundingBox:", "%%Bounding:", 1), "bounding_box"),
        (lambda t: re.sub(r"%%BoundingBox: .*", "%%BoundingBox: 50 50 10 10", t, count=1), "bounding_box"),
        (lambda t: t.rsplit("%%EOF", 1)[0], "terminator"),
        (lambda t: t.replace("%%EndProlog", "", 1), "prolog_balanced"),
        (lambda t: t.replace("showpage", "Q showpage", 1), "graphics_state"),
    ],
)
def test_structural_corruptions_are_caught(good_eps: Path, tmp_path: Path, mutate, check: str) -> None:
    bad = tmp_path / "bad.eps"
    bad.write_text(mutate(good_eps.read_text(encoding="latin-1")), encoding="latin-1")
    report = validate_eps(bad, render=False)
    assert status(report, check) == "fail"
    assert not report.passed


def test_atend_bounding_box_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "a.eps"
    path.write_text("%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: (atend)\n%%EndComments\n%%Trailer\n%%EOF\n")
    check = next(c for c in validate_eps(path, render=False).checks if c.name == "bounding_box")
    assert check.status == "fail" and "atend" in check.detail


def test_dos_binary_wrapper_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "d.eps"
    path.write_bytes(b"\xc5\xd0\xd3\xc6" + b"\x00" * 30 + b"%!PS-Adobe-3.0 EPSF-3.0\n%%EOF\n")
    assert status(validate_eps(path, render=False), "header") == "fail"


def test_geometry_free_eps_fails_vector_check(tmp_path: Path) -> None:
    path = tmp_path / "n.eps"
    path.write_text(
        "%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\n%%EndComments\n%%BeginProlog\n%%EndProlog\n"
        "%%Page: 1 1\n%%EndPageSetup\nshowpage\n%%Trailer\n%%EOF\n"
    )
    assert status(validate_eps(path, render=False), "vector_content") == "fail"


def test_embedded_raster_is_detected(tmp_path: Path) -> None:
    surface = cairo.PSSurface(str(tmp_path / "r.eps"), 40, 40)
    surface.set_eps(True)
    ctx = cairo.Context(surface)
    img = cairo.ImageSurface(cairo.FORMAT_ARGB32, 8, 8)
    pix = cairo.Context(img)
    pix.set_source_rgb(0.2, 0.4, 0.9)
    pix.paint()
    ctx.set_source_surface(img, 0, 0)
    ctx.paint()
    ctx.rectangle(5, 5, 20, 20)
    ctx.fill()
    surface.finish()
    report = validate_eps(tmp_path / "r.eps", render=False)
    assert status(report, "no_raster") == "fail"
    assert not report.passed


def test_canvas_size_pass_warn_fail(tmp_path: Path) -> None:
    def write(bbox: str) -> Path:
        p = tmp_path / "c.eps"
        p.write_text(
            f"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: {bbox}\n%%EndComments\n%%BeginProlog\n%%EndProlog\n"
            "%%EndPageSetup\n0 0 m 5 5 l 5 0 l f\nshowpage\n%%Trailer\n%%EOF\n"
        )
        return p

    full = round(100 * PX_TO_PT)
    assert (
        status(validate_eps(write(f"0 0 {full} {full}"), expected_size=(100, 100), render=False), "canvas_size")
        == "pass"
    )
    assert status(validate_eps(write("0 20 70 75"), expected_size=(100, 100), render=False), "canvas_size") == "warn"
    assert status(validate_eps(write("0 0 200 200"), expected_size=(100, 100), render=False), "canvas_size") == "fail"


def test_full_page_background_warns_only_for_transparent_source(tmp_path: Path) -> None:
    path = tmp_path / "b.eps"
    path.write_text(
        "%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 50 40\n%%EndComments\n%%BeginProlog\n%%EndProlog\n"
        "%%EndPageSetup\n1 g\n0 0 50 40 rectfill\n0 0 m 5 5 l 5 0 l f\nshowpage\n%%Trailer\n%%EOF\n"
    )
    assert status(validate_eps(path, source_has_alpha=True, render=False), "background_fill") == "warn"
    assert status(validate_eps(path, source_has_alpha=False, render=False), "background_fill") == "pass"


def test_report_collections() -> None:
    from eval.eps_validate import EpsReport

    rep = EpsReport(
        "x", 1, None, 0, 0, [EpsCheck("a", "warn", ""), EpsCheck("b", "fail", ""), EpsCheck("c", "skip", "")]
    )
    assert [c.name for c in rep.warnings] == ["a"] and [c.name for c in rep.failed] == ["b"] and not rep.passed


def test_missing_ghostscript_skips_render(good_eps: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("eval.eps_validate.find_ghostscript", lambda: None)
    assert status(validate_eps(good_eps), "render") == "skip"
    assert render_eps(good_eps)[0] is None
    assert render_on_canvas(good_eps, (0, 0, 48, 48), (64, 64)) is None


@needs_gs
def test_ghostscript_render_passes_and_covers_canvas(good_eps: Path) -> None:
    report = validate_eps(good_eps, expected_size=(64, 64))
    assert status(report, "render") == "pass"
    canvas = render_on_canvas(good_eps, report.bbox, (64, 64))
    assert canvas is not None and canvas.shape == (64, 64, 3) and canvas.min() < 250


@needs_gs
def test_garbage_postscript_fails_render(tmp_path: Path) -> None:
    path = tmp_path / "g.eps"
    path.write_text(
        "%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 20 20\n%%EndComments\n%%BeginProlog\n%%EndProlog\n"
        "%%EndPageSetup\n0 0 m 5 5 l f\nbogus_operator\nshowpage\n%%Trailer\n%%EOF\n"
    )
    assert status(validate_eps(path), "render") == "fail"


@needs_gs
def test_blank_render_fails(tmp_path: Path) -> None:
    path = tmp_path / "blank.eps"
    path.write_text(
        "%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 20 20\n%%EndComments\n%%BeginProlog\n%%EndProlog\n"
        "%%EndPageSetup\n0 g 100 100 moveto 110 100 lineto 110 110 lineto fill\nshowpage\n%%Trailer\n%%EOF\n"
    )
    assert status(validate_eps(path), "render") == "fail"


@needs_gs
def test_render_similarity_against_source(good_eps: Path, tmp_path: Path) -> None:
    import cv2

    bbox = validate_eps(good_eps, render=False).bbox
    ref = tmp_path / "ref.png"
    cv2.imwrite(str(ref), np.full((64, 64, 3), 255, np.uint8))
    score = render_similarity(good_eps, bbox, ref)
    assert score is not None and 0.0 <= score <= 1.0
    assert render_similarity(good_eps, bbox, tmp_path / "missing.png") is None
    report = validate_eps(good_eps, reference_path=ref)
    assert report.render_ssim is not None
    assert status(report, "render_similarity") == "fail"  # white reference vs coloured artwork


def test_cli_eps_command(good_eps: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert eval_main(["eps", str(good_eps), "--size", "64x64", "--no-render"]) == 0
    assert "PASS" in capsys.readouterr().out
    assert eval_main(["eps", str(good_eps.with_name("nope.eps"))]) == 1


def test_cli_eps_command_exit_code_on_failure(tmp_path: Path) -> None:
    bad = tmp_path / "bad.eps"
    bad.write_text("not postscript")
    assert eval_main(["eps", str(bad), "--no-render"]) == 1
