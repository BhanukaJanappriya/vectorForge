"""Tests for pipeline/runner.py using injected fake stages (contracts.fixtures) and real upstream stages."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from contracts.api import PipelineStage
from contracts.fixtures import (
    make_image_class,
    make_image_input,
    make_line_map,
    make_palette,
    make_preprocess_result,
    make_vector_document,
)
from contracts.schemas import (
    ExportBundle,
    ExportToolUnavailableError,
    ImageClass,
    ImageClassLabel,
    InvalidImageError,
    LineMode,
    Palette,
    PreprocessResult,
    ProcessingMode,
    QualityReport,
    Settings,
    StageError,
    VectorDocument,
)
from contracts.stages import STAGE_ENTRYPOINTS
from pipeline import runner
from pipeline.runner import (
    STAGE_ORDER,
    STAGE_PROGRESS,
    PipelineError,
    PipelineResult,
    error_detail,
    needs_lines,
    resolve_stage,
    resolve_stages,
    run_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "01_logo_4color.png"


# ------------------------------------------------------------------------------ fakes


def fake_load_image(path: Path) -> Any:
    return make_image_input(64, path=path)


def fake_preprocess(image: Any, settings: Settings) -> PreprocessResult:
    return make_preprocess_result(64)


def fake_classify(pre: PreprocessResult, settings: Settings) -> ImageClass:
    return make_image_class(ImageClassLabel.FLAT_COLOR)


def fake_classify_line_art(pre: PreprocessResult, settings: Settings) -> ImageClass:
    return make_image_class(ImageClassLabel.LINE_ART)


def fake_quantize(pre: PreprocessResult, image_class: ImageClass, settings: Settings) -> Palette:
    return make_palette(pre)


def fake_extract_lines(pre: PreprocessResult, image_class: ImageClass, settings: Settings) -> Any:
    return make_line_map(pre)


def fake_vectorize(pre: PreprocessResult, cls: ImageClass, palette: Palette, lines: Any, settings: Settings) -> Any:
    return make_vector_document(64, settings)


def fake_assemble(doc: VectorDocument, settings: Settings) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {doc.width} {doc.height}"/>'


def fake_export(svg: str, doc: VectorDocument, settings: Settings, out_dir: Path) -> ExportBundle:
    svg_path = out_dir / "output.svg"
    svg_path.write_text(svg, encoding="utf-8")
    png = out_dir / "preview.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return ExportBundle(svg_path=svg_path, preview_png_path=png)


def fake_evaluate(pre: Any, palette: Any, lines: Any, doc: Any, bundle: ExportBundle, elapsed: float) -> QualityReport:
    return QualityReport(
        ssim=1.0,
        mean_delta_e=0.0,
        max_delta_e=0.0,
        gap_ratio=0.0,
        node_count=4,
        file_size_bytes=bundle.svg_path.stat().st_size,
        processing_time_s=elapsed,
        checks=[],
    )


FAKES: dict[str, Any] = {
    "load_image": fake_load_image,
    "preprocess": fake_preprocess,
    "classify": fake_classify,
    "quantize": fake_quantize,
    "extract_lines": fake_extract_lines,
    "vectorize": fake_vectorize,
    "assemble_svg": fake_assemble,
    "export": fake_export,
    "evaluate": fake_evaluate,
}


class Recorder:
    """Wraps the fakes to record call order."""

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        self.calls: list[str] = []
        self.stages = {name: self._wrap(name, fn) for name, fn in {**FAKES, **(overrides or {})}.items()}

    def _wrap(self, name: str, fn: Any) -> Any:
        def call(*args: Any) -> Any:
            self.calls.append(name)
            return fn(*args)

        return call


# ------------------------------------------------------------------------------ happy paths


def test_flat_color_skips_lines_and_reports_progress(tmp_path: Path) -> None:
    rec = Recorder()
    progress: list[tuple[PipelineStage, float]] = []
    result = run_pipeline(
        SAMPLE, Settings(), tmp_path / "out", on_progress=lambda s, f: progress.append((s, f)), stages=rec.stages
    )
    assert isinstance(result, PipelineResult)
    assert rec.calls == [n for n in STAGE_ORDER if n != "extract_lines"]
    assert result.line_map is None
    assert result.image_class.label == ImageClassLabel.FLAT_COLOR
    assert set(result.timings) == set(rec.calls)
    assert all(t >= 0 for t in result.timings.values())
    assert result.elapsed_s >= result.processing_time_s >= 0
    assert result.report.processing_time_s == pytest.approx(result.processing_time_s)
    assert result.bundle.svg_path.parent == tmp_path / "out"
    assert result.svg.startswith("<svg")
    fractions = [f for _, f in progress]
    assert fractions == sorted(fractions)
    assert progress[0] == (PipelineStage.PREPROCESS, 0.0)
    assert progress[-1] == (PipelineStage.DONE, 1.0)
    assert PipelineStage.EXTRACT_LINES not in [s for s, _ in progress]


@pytest.mark.parametrize(
    ("classify", "settings", "expect_lines"),
    [
        (fake_classify_line_art, Settings(), True),
        (fake_classify, Settings(line_mode=LineMode.CENTERLINE), True),
        (fake_classify, Settings(mode=ProcessingMode.FLAT_COLOR), False),
    ],
)
def test_line_branching(tmp_path: Path, classify: Any, settings: Settings, expect_lines: bool) -> None:
    rec = Recorder({"classify": classify})
    result = run_pipeline(SAMPLE, settings, tmp_path, stages=rec.stages)
    assert ("extract_lines" in rec.calls) is expect_lines
    assert (result.line_map is not None) is expect_lines
    if expect_lines:
        assert rec.calls.index("extract_lines") == rec.calls.index("quantize") + 1


def test_needs_lines_for_mixed() -> None:
    assert needs_lines(make_image_class(ImageClassLabel.MIXED), Settings())
    assert not needs_lines(make_image_class(ImageClassLabel.FLAT_COLOR), Settings())


def test_default_settings_and_stage_arguments(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def vectorize(pre: Any, cls: Any, palette: Any, lines: Any, settings: Settings) -> Any:
        seen.update(pre=pre, cls=cls, palette=palette, lines=lines, settings=settings)
        return fake_vectorize(pre, cls, palette, lines, settings)

    def export(svg: str, doc: Any, settings: Settings, out_dir: Path) -> ExportBundle:
        seen["out_dir"] = out_dir
        return fake_export(svg, doc, settings, out_dir)

    stages = {**FAKES, "vectorize": vectorize, "export": export}
    result = run_pipeline(SAMPLE, None, tmp_path / "a" / "b", stages=stages)
    assert seen["settings"] == Settings()
    assert seen["pre"] is result.pre and seen["palette"] is result.palette and seen["lines"] is None
    assert seen["out_dir"] == tmp_path / "a" / "b" and seen["out_dir"].is_dir()


def test_real_upstream_stages_with_fake_downstream(tmp_path: Path) -> None:
    """Real load_image/preprocess/classify/quantize resolved via STAGE_ENTRYPOINTS."""
    overrides = {k: FAKES[k] for k in ("vectorize", "assemble_svg", "export", "evaluate")}
    result = run_pipeline(SAMPLE, Settings(), tmp_path, stages=overrides)
    assert result.image.width == 512 and result.image.height == 512
    assert result.image_class.label == ImageClassLabel.FLAT_COLOR
    assert len(result.palette.colors) == 4


# ------------------------------------------------------------------------------ resolution


def test_resolve_stage_returns_real_entrypoint() -> None:
    from pipeline.preprocess import preprocess

    assert resolve_stage("preprocess") is preprocess


def test_missing_stage_module_is_stage_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(STAGE_ENTRYPOINTS, "vectorize", "pipeline.does_not_exist_xyz:vectorize")
    with pytest.raises(StageError) as info:
        resolve_stage("vectorize")
    assert info.value.stage == "vectorize"
    assert "pipeline.does_not_exist_xyz" in str(info.value)
    assert not isinstance(info.value, ImportError)

    overrides = {k: v for k, v in FAKES.items() if k != "vectorize"}
    with pytest.raises(PipelineError) as perr:
        run_pipeline(SAMPLE, Settings(), tmp_path, stages=overrides)
    assert perr.value.detail.code == "stage_failed"
    assert perr.value.detail.stage == PipelineStage.VECTORIZE
    assert "vectorize" in perr.value.detail.message
    assert perr.value.stage_name == "vectorize"
    assert isinstance(perr.value.__cause__, StageError)


def test_missing_attribute_and_unknown_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(STAGE_ENTRYPOINTS, "vectorize", "pipeline.preprocess:no_such_function")
    with pytest.raises(StageError, match="not defined"):
        resolve_stage("vectorize")
    with pytest.raises(StageError, match="unknown stage"):
        resolve_stage("nope")
    with pytest.raises(StageError, match="unknown stage override"):
        resolve_stages({"bogus": fake_assemble})


def test_unknown_override_fails_run(tmp_path: Path) -> None:
    with pytest.raises(PipelineError) as info:
        run_pipeline(SAMPLE, Settings(), tmp_path, stages={**FAKES, "bogus": fake_assemble})
    assert info.value.detail.stage is None
    assert info.value.detail.code == "stage_failed"


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("raise RuntimeError('half written')\n", "RuntimeError: half written"),
        ("import vf_dependency_that_is_missing_123\n", "failed to import"),
    ],
)
def test_broken_stage_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str, match: str) -> None:
    name = f"vf_broken_stage_{abs(hash(source))}"
    (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(STAGE_ENTRYPOINTS, "export", f"{name}:export")
    try:
        with pytest.raises(StageError, match=match) as info:
            resolve_stage("export")
        assert info.value.stage == "export"
    finally:
        sys.modules.pop(name, None)


# ------------------------------------------------------------------------------ error mapping


def _raiser(exc: Exception) -> Any:
    def stage(*_: Any) -> Any:
        raise exc

    return stage


@pytest.mark.parametrize(
    ("stage", "exc", "code", "api_stage"),
    [
        ("load_image", InvalidImageError("not an image"), "invalid_image", PipelineStage.PREPROCESS),
        ("quantize", StageError("quantize", "no colors"), "stage_failed", PipelineStage.QUANTIZE),
        ("export", ExportToolUnavailableError("inkscape missing"), "export_tool_unavailable", PipelineStage.EXPORT),
        ("assemble_svg", ValueError("bad"), "internal_error", PipelineStage.ASSEMBLE),
        ("evaluate", RuntimeError("boom"), "internal_error", PipelineStage.EVALUATE),
    ],
)
def test_exceptions_map_to_error_detail(
    tmp_path: Path, stage: str, exc: Exception, code: str, api_stage: PipelineStage
) -> None:
    with pytest.raises(PipelineError) as info:
        run_pipeline(SAMPLE, Settings(), tmp_path, stages={**FAKES, stage: _raiser(exc)})
    err = info.value
    assert err.detail.code == code == err.code
    assert err.detail.stage == api_stage
    assert err.stage_name == stage
    assert err.__cause__ is exc
    assert stage in err.timings
    done = STAGE_ORDER[: STAGE_ORDER.index(stage)]
    assert set(err.partial) == {n for n in done if n != "extract_lines"}


def test_wrong_return_type_is_stage_failure(tmp_path: Path) -> None:
    with pytest.raises(PipelineError) as info:
        run_pipeline(SAMPLE, Settings(), tmp_path, stages={**FAKES, "classify": lambda *a: "flat"})
    assert info.value.detail.code == "stage_failed"
    assert "expected ImageClass" in info.value.detail.message


def test_error_detail_without_stage_and_empty_message() -> None:
    detail = error_detail(InvalidImageError(), None)
    assert detail.stage is None and detail.code == "invalid_image" and detail.message == "InvalidImageError"
    assert error_detail(KeyError("x"), "nope").stage is None


def test_out_dir_that_cannot_be_created(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(PipelineError) as info:
        run_pipeline(SAMPLE, Settings(), blocker / "out", stages=FAKES)
    assert info.value.detail.stage == PipelineStage.EXPORT


def test_progress_table_covers_every_stage() -> None:
    assert set(STAGE_PROGRESS) == set(STAGE_ORDER) == set(STAGE_ENTRYPOINTS)
    assert list(STAGE_PROGRESS.values()) == sorted(STAGE_PROGRESS.values())
    assert runner.STAGE_TO_PIPELINE_STAGE.keys() == STAGE_PROGRESS.keys()
