"""End-to-end pipeline runs with the network and the LLM stubbed out.

Proves the orchestration actually executes: intake, per-video production,
error isolation, resume, and book assembly — the behaviour the ADK
``SequentialAgent``/``LoopAgent`` graph provided, without the framework.
"""

from __future__ import annotations

import dspy
import pytest

from bookforge.dspy_runtime.program import BookForgeProgram
from bookforge.pipeline import orchestrator
from bookforge.pipeline.orchestrator import run_pipeline
from bookforge.pipeline.progress import Reporter
from bookforge.schemas import MediaBundle
from tests.conftest import good_chapter_tex

CHANNEL_URL = "https://www.youtube.com/@testchannel"


@pytest.fixture
def stub_youtube(monkeypatch, transcript_segments):
    """Replace every network call with deterministic local data."""

    def fake_list(channel_url, max_videos=None, min_duration_sec=0,
                  max_duration_sec=10**9, auth=None):
        videos = [
            {
                "video_id": f"vid{i:03d}",
                "title": f"Lecture {i}: Kubernetes Networking Deep Dive",
                "url": f"https://www.youtube.com/watch?v=vid{i:03d}",
                "duration_sec": 1200,
                "upload_date": f"2025010{i}",
            }
            for i in range(1, 4)
        ]
        return "Test Channel", videos[: max_videos or len(videos)]

    def fake_acquire(record, workspace, settings):
        video_dir = workspace.video_dir(record.video_id)
        transcript_path = video_dir / "transcript.txt"
        transcript_path.write_text(
            "\n".join(f"[{s.timestamp}] {s.text}" for s in transcript_segments),
            encoding="utf-8",
        )
        bundle = MediaBundle(
            video_id=record.video_id,
            video_path=str(video_dir / "video.mp4"),
            transcript_path=str(transcript_path),
            transcript_source="captions",
            duration_sec=1200.0,
            frames=[],
        )
        (video_dir / "media.json").write_text(bundle.model_dump_json(), encoding="utf-8")
        return bundle, list(transcript_segments)

    # Intake goes through bookforge.ingest now, so patch the tool the YouTube
    # source actually calls rather than a name intake no longer imports.
    monkeypatch.setattr("bookforge.tools.youtube.list_channel_videos", fake_list)
    monkeypatch.setattr("bookforge.pipeline.chapter.acquire_media", fake_acquire)
    monkeypatch.setattr(orchestrator, "acquire_media", fake_acquire)
    return fake_acquire


class StubAnalyst(dspy.Module):
    def __init__(self, analysis, fail_on: set[str] | None = None) -> None:
        super().__init__()
        self.analysis = analysis
        self.fail_on = fail_on or set()
        self.calls = 0

    def forward(self, video_title: str = "", **kwargs) -> dspy.Prediction:
        self.calls += 1
        if video_title in self.fail_on:
            raise RuntimeError("simulated analysis failure")
        return dspy.Prediction(
            analysis=self.analysis, grounding_score=0.95,
            grounding_report=None, windows=1,
        )


class StubChapter(dspy.Module):
    def __init__(self, tex_for) -> None:
        super().__init__()
        self.tex_for = tex_for
        self.calls = 0

    def forward(self, manifest=None, **kwargs) -> dspy.Prediction:
        from bookforge.dspy_runtime.modules.chapter import ChapterAttemptLog, ChapterResult
        from bookforge.schemas import ChapterVerdict

        self.calls += 1
        tex = self.tex_for(manifest)
        verdict = ChapterVerdict(compiles=True, compile_skipped=False, score=0.94)
        result = ChapterResult(
            tex=tex, verdict=verdict, iterations=1,
            history=[ChapterAttemptLog(1, 0.94, True, 0, 0, "abc", "ok")],
            accepted=True,
        )
        return dspy.Prediction(
            chapter_tex=tex, verdict=verdict, result=result, score=0.94, accepted=True
        )


def _program(settings, analysis, tex_for, fail_on=None) -> BookForgeProgram:
    program = BookForgeProgram(settings, load_optimized=False)
    program.analyst = StubAnalyst(analysis, fail_on)
    program.captioner = None  # no frames in these fixtures
    program.chapter = StubChapter(tex_for)
    return program


class TestHappyPath:
    def test_full_run_produces_a_book(self, settings, analysis, stub_youtube):
        settings.max_videos = 2
        program = _program(settings, analysis, good_chapter_tex)
        result = run_pipeline(CHANNEL_URL, settings, Reporter(), program)

        assert result.error == ""
        assert len(result.outcomes) == 2
        assert result.verified == 2
        assert result.book["chapters_included"] == 2
        assert result.book["mean_quality_score"] > 0.9

    def test_artifacts_land_on_disk(self, settings, analysis, stub_youtube):
        settings.max_videos = 1
        program = _program(settings, analysis, good_chapter_tex)
        run_pipeline(CHANNEL_URL, settings, Reporter(), program)

        root = settings.workspace_root_abs / "test-channel"
        chapter_dir = next((root / "chapters").iterdir())
        for name in ("chapter.tex", "analysis.json", "assets.json", "verdict.json"):
            assert (chapter_dir / name).exists(), f"{name} was not written"
        assert (root / "main.tex").exists()
        assert (root / "preamble.tex").exists()
        assert (root / "book" / "book_manifest.json").exists()

    def test_main_tex_inputs_each_chapter(self, settings, analysis, stub_youtube):
        settings.max_videos = 2
        program = _program(settings, analysis, good_chapter_tex)
        run_pipeline(CHANNEL_URL, settings, Reporter(), program)
        main_tex = (settings.workspace_root_abs / "test-channel" / "main.tex").read_text()
        assert main_tex.count(r"\input{chapters/") == 2

    def test_duplicate_labels_across_chapters_are_renamed(
        self, settings, analysis, stub_youtube
    ):
        """Two chapters from the same template share every label."""
        settings.max_videos = 2
        program = _program(settings, analysis, good_chapter_tex)
        run_pipeline(CHANNEL_URL, settings, Reporter(), program)

        chapters = sorted((settings.workspace_root_abs / "test-channel" / "chapters").iterdir())
        second = (chapters[1] / "chapter.tex").read_text()
        assert "ch:01-kubernetes-2" in second, "a colliding label survived into the book"


class TestErrorIsolation:
    def test_one_bad_video_does_not_stop_the_book(self, settings, analysis, stub_youtube):
        settings.max_videos = 3
        program = _program(
            settings, analysis, good_chapter_tex,
            fail_on={"Lecture 2: Kubernetes Networking Deep Dive"},
        )
        result = run_pipeline(CHANNEL_URL, settings, Reporter(), program)

        assert result.failed == 1
        assert result.verified == 2
        assert result.book["chapters_included"] == 2
        assert len(result.book["chapters_failed"]) == 1

    def test_failure_is_recorded_in_the_manifest(self, settings, analysis, stub_youtube):
        settings.max_videos = 2
        program = _program(
            settings, analysis, good_chapter_tex,
            fail_on={"Lecture 1: Kubernetes Networking Deep Dive"},
        )
        run_pipeline(CHANNEL_URL, settings, Reporter(), program)

        from bookforge.workspace import Workspace

        manifest = Workspace(settings.workspace_root_abs, "test-channel").load_manifest()
        failed = [v for v in manifest.videos if v.status == "failed"]
        assert len(failed) == 1
        assert "simulated analysis failure" in failed[0].error

    def test_all_videos_failing_raises_a_useful_error(
        self, settings, analysis, stub_youtube
    ):
        settings.max_videos = 2
        titles = {f"Lecture {i}: Kubernetes Networking Deep Dive" for i in (1, 2)}
        program = _program(settings, analysis, good_chapter_tex, fail_on=titles)
        result = run_pipeline(CHANNEL_URL, settings, Reporter(), program)
        assert "nothing to compile" in result.error
        assert "verdict.json" in result.error, "the error should say where to look"


class TestResume:
    def test_verified_chapters_are_skipped_on_rerun(
        self, settings, analysis, stub_youtube
    ):
        settings.max_videos = 2
        first = _program(settings, analysis, good_chapter_tex)
        run_pipeline(CHANNEL_URL, settings, Reporter(), first)
        assert first.chapter.calls == 2

        second = _program(settings, analysis, good_chapter_tex)
        result = run_pipeline(CHANNEL_URL, settings, Reporter(), second)
        assert second.chapter.calls == 0, "re-did work that was already verified"
        assert result.book["chapters_included"] == 2

    def test_failed_chapters_are_retried(self, settings, analysis, stub_youtube):
        settings.max_videos = 2
        failing = {"Lecture 2: Kubernetes Networking Deep Dive"}
        run_pipeline(
            CHANNEL_URL, settings, Reporter(),
            _program(settings, analysis, good_chapter_tex, fail_on=failing),
        )
        retry = _program(settings, analysis, good_chapter_tex)
        result = run_pipeline(CHANNEL_URL, settings, Reporter(), retry)
        assert retry.chapter.calls == 1, "the failed video was not retried"
        assert result.book["chapters_included"] == 2


class TestReporting:
    def test_events_are_emitted(self, settings, analysis, stub_youtube):
        settings.max_videos = 1
        events = []
        reporter = Reporter([events.append])
        run_pipeline(
            CHANNEL_URL, settings, reporter, _program(settings, analysis, good_chapter_tex)
        )
        kinds = {event.kind for event in events}
        assert "stage" in kinds
        assert any("Intake complete" in event.message for event in events)
        assert any("Book compiled" in event.message for event in events)
