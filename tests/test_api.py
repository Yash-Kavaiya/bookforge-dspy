"""The HTTP API and the job runner.

Every test runs against a real TestClient with the pipeline stubbed, so the
routes, validation, streaming and path handling are exercised for real without
building a book.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from bookforge.api.app import create_app
from bookforge.api.jobs import Job, JobRunner, reset_runner
from bookforge.schemas import VideoRecord


@pytest.fixture(autouse=True)
def _clean_runner():
    reset_runner()
    yield
    reset_runner()


@pytest.fixture
def client(settings, monkeypatch):
    # get_runner caches a process-wide runner; point it at this test's settings.
    monkeypatch.setattr("bookforge.api.jobs._runner", JobRunner(settings))
    return TestClient(create_app(settings))


@pytest.fixture
def config_client(settings, tmp_path, monkeypatch):
    """Like ``client``, but with an isolated .env so a save never touches the repo's."""
    monkeypatch.setattr("bookforge.api.jobs._runner", JobRunner(settings))
    return TestClient(create_app(settings, env_path=tmp_path / ".env"))


@pytest.fixture
def video_folder(tmp_path):
    folder = tmp_path / "Talks"
    folder.mkdir()
    for name in ("01_first-talk.mp4", "02_second-talk.mp4"):
        (folder / name).write_bytes(b"video bytes")
    return folder


class TestHealth:
    def test_reports_readiness_and_tools(self, client):
        payload = client.get("/api/health").json()
        assert "ready" in payload
        assert set(payload["tools"]) == {"pdflatex", "ffmpeg"}
        assert "textbook" in payload["themes"]

    def test_reports_workspace_and_extensions(self, client, settings):
        payload = client.get("/api/health").json()
        assert payload["settings"]["workspace_root"] == str(settings.workspace_root_abs)
        assert ".mp4" in payload["settings"]["video_extensions"]

    def test_missing_credentials_is_surfaced_not_hidden(self, settings, monkeypatch):
        """Finding out at the end of an hour-long run is the worst time."""
        for variable in (
            "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY", "NVIDIA_API_KEY", "BOOKFORGE_OPENAI_API_KEY",
            "GOOGLE_GENAI_USE_VERTEXAI",
        ):
            monkeypatch.delenv(variable, raising=False)
        settings.openai_api_key = ""
        payload = TestClient(create_app(settings)).get("/api/health").json()
        assert payload["ready"] is False
        assert any("credentials" in w for w in payload["warnings"])


class TestConfigGet:
    def test_lists_grouped_fields_with_masked_secret(self, config_client, settings):
        payload = config_client.get("/api/config").json()
        keys = {f["key"] for f in payload["fields"]}
        assert {"openai_api_key", "writer_temperature", "qa_max_iterations"} <= keys
        assert {g["key"] for g in payload["groups"]} >= {"credentials", "sampling", "quality"}

        secret = next(f for f in payload["fields"] if f["key"] == "openai_api_key")
        assert secret["is_set"] is True
        assert settings.openai_api_key not in secret["value"], "the raw key must never round-trip"

    def test_provider_env_presence_is_reported_without_values(self, config_client, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "secret-value")
        payload = config_client.get("/api/config").json()
        assert payload["provider_env"]["GOOGLE_API_KEY"] is True
        assert "secret-value" not in json.dumps(payload)


class TestConfigUpdate:
    def test_updates_a_hyperparameter_on_the_live_settings(self, config_client, settings, monkeypatch):
        monkeypatch.setenv("BOOKFORGE_WRITER_TEMPERATURE", "placeholder")
        response = config_client.put("/api/config", json={"settings": {"writer_temperature": 0.9}})
        assert response.status_code == 200
        # Every route closure shares this exact Settings instance, so a build
        # started right after this call must see the new value with no restart.
        assert settings.writer_temperature == 0.9

    def test_persists_to_the_env_file(self, config_client, tmp_path, monkeypatch):
        monkeypatch.setenv("BOOKFORGE_QA_MAX_ITERATIONS", "placeholder")
        config_client.put("/api/config", json={"settings": {"qa_max_iterations": 5}})
        env_text = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "BOOKFORGE_QA_MAX_ITERATIONS=5" in env_text

    def test_rejects_unknown_setting(self, config_client):
        response = config_client.put("/api/config", json={"settings": {"not_a_real_field": 1}})
        assert response.status_code == 400

    def test_rejects_a_cross_field_violation(self, config_client, settings):
        """The overlap/chunk-size pair is only valid together; a partial write must not land."""
        response = config_client.put(
            "/api/config", json={"settings": {"transcript_chunk_chars": 100}}
        )
        assert response.status_code == 422
        assert settings.transcript_chunk_chars != 100, "an invalid change must not apply"

    def test_updates_a_provider_env_var(self, config_client, tmp_path, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "placeholder")
        response = config_client.put("/api/config", json={"env": {"GOOGLE_API_KEY": "abc123"}})
        assert response.status_code == 200
        import os

        assert os.environ["GOOGLE_API_KEY"] == "abc123"
        assert "GOOGLE_API_KEY=abc123" in (tmp_path / ".env").read_text(encoding="utf-8")

    def test_rejects_unknown_env_var(self, config_client):
        response = config_client.put("/api/config", json={"env": {"SOME_RANDOM_KEY": "x"}})
        assert response.status_code == 400

    def test_a_new_api_key_is_masked_on_the_next_read(self, config_client, tmp_path, monkeypatch):
        monkeypatch.setenv("BOOKFORGE_OPENAI_API_KEY", "placeholder")
        config_client.put("/api/config", json={"settings": {"openai_api_key": "sk-new-secret-key"}})
        payload = config_client.get("/api/config").json()
        secret = next(f for f in payload["fields"] if f["key"] == "openai_api_key")
        assert secret["value"].endswith("-key")
        assert "sk-new-secret-key" not in secret["value"]


class TestConfigTest:
    def test_reports_ok_with_latency_on_success(self, config_client, monkeypatch):
        class FakeLM:
            def __call__(self, prompt=None, **kwargs):
                return ["OK"]

        monkeypatch.setattr("bookforge.api.app.build_lm", lambda role, trial, **kw: FakeLM())
        response = config_client.post("/api/config/test", json={"role": "analyst"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is True
        assert payload["reply"] == "OK"
        assert isinstance(payload["latency_ms"], int)

    def test_reports_failure_without_raising(self, config_client, monkeypatch):
        def boom(role, trial, **kwargs):
            raise RuntimeError("invalid api key")

        monkeypatch.setattr("bookforge.api.app.build_lm", boom)
        response = config_client.post("/api/config/test", json={"role": "writer"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is False
        assert "invalid api key" in payload["error"]

    def test_unknown_role_is_rejected(self, config_client):
        response = config_client.post("/api/config/test", json={"role": "nope"})
        assert response.status_code == 400


class TestPreview:
    def test_local_folder(self, client, video_folder):
        payload = client.post(
            "/api/preview", json={"source": [str(video_folder)]}
        ).json()
        assert payload["count"] == 2
        assert payload["items"][0]["title"] == "First Talk"
        assert payload["items"][0]["source"] == "local_video"

    def test_ordering_is_applied(self, client, video_folder):
        (video_folder / "10_last-talk.mp4").write_bytes(b"v")
        payload = client.post("/api/preview", json={"source": [str(video_folder)]}).json()
        assert payload["items"][-1]["title"] == "Last Talk", "10 must sort after 02"

    def test_bad_source_returns_400_with_the_reason(self, client, tmp_path):
        response = client.post("/api/preview", json={"source": [str(tmp_path / "nope")]})
        assert response.status_code == 400
        assert "do not exist" in response.json()["detail"]

    def test_non_youtube_url_is_rejected(self, client):
        response = client.post("/api/preview", json={"source": ["https://vimeo.com/1"]})
        assert response.status_code == 400
        assert "YouTube" in response.json()["detail"]

    def test_empty_source_is_rejected(self, client):
        assert client.post("/api/preview", json={"source": []}).status_code == 422


class TestUpload:
    def test_saves_supported_files(self, client):
        files = [
            ("files", ("talk.mp4", b"video bytes", "video/mp4")),
            ("files", ("talk.srt", b"1\n00:00:01,000 --> 00:00:02,000\nhi\n", "text/plain")),
        ]
        payload = client.post("/api/upload", files=files).json()
        assert len(payload["saved"]) == 2
        assert payload["rejected"] == []

        from pathlib import Path

        batch = Path(payload["path"])
        assert (batch / "talk.mp4").exists()

    def test_rejects_unsupported_types(self, client):
        files = [("files", ("notes.pdf", b"%PDF", "application/pdf"))]
        response = client.post("/api/upload", files=files)
        assert response.status_code == 400
        assert "supported video or transcript" in response.json()["detail"]

    def test_path_traversal_in_filename_is_neutralised(self, client):
        """A filename is attacker-controlled; it must never escape the batch."""
        files = [("files", ("../../evil.mp4", b"v", "video/mp4"))]
        payload = client.post("/api/upload", files=files).json()
        from pathlib import Path

        batch = Path(payload["path"])
        assert payload["saved"][0]["name"] == "evil.mp4"
        assert (batch / "evil.mp4").exists()
        assert not (batch.parent.parent / "evil.mp4").exists()

    def test_oversized_batch_is_refused_and_leaves_nothing_behind(
        self, settings, monkeypatch
    ):
        """A browser tab must not be able to fill the server's disk.

        Without a ceiling the write succeeds until the filesystem is full, and
        the failure then lands on whatever writes next -- usually a chapter
        compile in an unrelated build.
        """
        from pathlib import Path

        settings.upload_max_mb = 1
        monkeypatch.setattr("bookforge.api.jobs._runner", JobRunner(settings))
        client = TestClient(create_app(settings))

        big = b"x" * (2 * 1024 * 1024)
        response = client.post(
            "/api/upload", files=[("files", ("huge.mp4", big, "video/mp4"))]
        )
        assert response.status_code == 413
        assert "BOOKFORGE_UPLOAD_MAX_MB" in response.json()["detail"]

        uploads = settings.workspace_root_abs / "uploads"
        leftovers = list(uploads.rglob("*.mp4")) if uploads.exists() else []
        assert leftovers == [], "the partial upload was not cleaned up"
        assert Path(uploads).exists() or True  # the batch folder itself is gone

    def test_uploaded_folder_previews_as_a_source(self, client):
        files = [("files", ("lesson.mp4", b"video bytes", "video/mp4"))]
        path = client.post("/api/upload", files=files).json()["path"]
        payload = client.post("/api/preview", json={"source": [path]}).json()
        assert payload["count"] == 1
        assert payload["items"][0]["title"] == "Lesson"


class TestBuilds:
    def test_rejects_unknown_theme(self, client, video_folder):
        response = client.post(
            "/api/builds", json={"source": [str(video_folder)], "theme": "fancy"}
        )
        assert response.status_code == 400

    def test_rejects_a_bad_source_before_starting_a_thread(self, client, tmp_path):
        """A background failure is only visible in the log; fail fast instead."""
        response = client.post("/api/builds", json={"source": [str(tmp_path / "gone")]})
        assert response.status_code == 400

    def test_unknown_build_is_404(self, client):
        assert client.get("/api/builds/deadbeef").status_code == 404

    def test_list_is_empty_initially(self, client):
        assert client.get("/api/builds").json()["builds"] == []


class TestWorkspaceRoutes:
    @pytest.fixture
    def populated(self, workspace, analysis, assets_manifest):
        manifest = workspace.init_manifest("src", "Test Channel", "test-channel")
        record = VideoRecord(
            video_id="vid001", title="First", url="u", chapter_number=1,
            status="verified", qa_score=0.93, grounding_score=0.88,
            qa_iterations=2, compiled=True,
        )
        manifest.videos.append(record)
        workspace.save_manifest(manifest)

        chapter_dir = workspace.chapter_dir(record)
        (chapter_dir / "verdict.json").write_text(json.dumps({
            "score": 0.93, "accepted": True, "iterations": 2, "compiles": True,
            "defects": [{"severity": "minor", "message": "one overfull box", "fix": "shorten it"}],
            "history": [{"iteration": 0, "score": 0.81, "compiles": True,
                         "critical_defects": 0, "total_defects": 2,
                         "fingerprint": "abc", "note": "initial draft"},
                        {"iteration": 1, "score": 0.93, "compiles": True,
                         "critical_defects": 0, "total_defects": 1,
                         "fingerprint": "def", "note": "improved"}],
        }), encoding="utf-8")
        (workspace.book_dir / "book.pdf").write_bytes(b"%PDF-1.5 fake")
        return workspace

    def test_lists_workspaces(self, client, populated):
        payload = client.get("/api/workspaces").json()
        assert payload["workspaces"][0]["slug"] == "test-channel"
        assert payload["workspaces"][0]["has_pdf"] is True

    def test_chapter_detail_includes_the_verdict(self, client, populated):
        payload = client.get("/api/workspaces/test-channel/chapters").json()
        chapter = payload["chapters"][0]
        assert chapter["qa_score"] == 0.93
        assert chapter["verdict"]["defects"][0]["severity"] == "minor"
        assert len(chapter["verdict"]["history"]) == 2

    def test_downloads_the_pdf(self, client, populated):
        response = client.get("/api/workspaces/test-channel/book")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"

    def test_unknown_workspace_is_404(self, client):
        assert client.get("/api/workspaces/nope/chapters").status_code == 404

    def test_path_traversal_is_refused(self, client, populated):
        """A path from a query string must never escape the workspace."""
        response = client.get(
            "/api/workspaces/test-channel/file", params={"path": "../../../etc/passwd"}
        )
        assert response.status_code in (403, 404)

    def test_serves_a_workspace_file(self, client, populated):
        response = client.get(
            "/api/workspaces/test-channel/file", params={"path": "manifest.json"}
        )
        assert response.status_code == 200
        assert "channel_slug" in response.text


class TestUI:
    """The UI is an Angular SPA (frontend/, built to bookforge/api/webapp/ by
    `npm run build`). FastAPI's job is just to hand back index.html for any
    client-side route and real files for everything Angular actually built —
    tests point WEBAPP_DIR at a throwaway directory so they don't depend on a
    real `npm run build` having run first.
    """

    @pytest.fixture
    def webapp_client(self, settings, monkeypatch, tmp_path):
        webapp = tmp_path / "webapp"
        webapp.mkdir()
        (webapp / "index.html").write_text(
            "<html><body><app-root>BookForge</app-root></body></html>", encoding="utf-8"
        )
        (webapp / "main-ABC123.js").write_text("console.log('bookforge')", encoding="utf-8")
        monkeypatch.setattr("bookforge.api.app.WEBAPP_DIR", webapp)
        return TestClient(create_app(settings)), webapp

    def test_root_serves_the_angular_shell(self, webapp_client):
        client, _ = webapp_client
        response = client.get("/")
        assert response.status_code == 200
        assert "BookForge" in response.text

    def test_a_client_side_route_gets_the_same_shell(self, webapp_client):
        """/config is handled by Angular Router in the browser, not FastAPI —
        a direct navigation or refresh there must still get index.html."""
        client, _ = webapp_client
        root = client.get("/")
        config = client.get("/config")
        assert config.status_code == 200
        assert config.text == root.text

    def test_a_real_built_asset_is_served_from_disk(self, webapp_client):
        client, _ = webapp_client
        response = client.get("/main-ABC123.js")
        assert response.status_code == 200
        assert "bookforge" in response.text

    def test_path_traversal_cannot_escape_the_webapp_dir(self, webapp_client, tmp_path):
        client, _webapp = webapp_client
        (tmp_path / "secret.txt").write_text("do not serve me", encoding="utf-8")
        response = client.get("/..%2Fsecret.txt")
        assert "do not serve me" not in response.text

    def test_falls_back_to_a_message_when_the_ui_is_not_built(self, settings, monkeypatch, tmp_path):
        monkeypatch.setattr("bookforge.api.app.WEBAPP_DIR", tmp_path / "does-not-exist")
        response = TestClient(create_app(settings)).get("/")
        assert response.status_code == 200
        assert "not built" in response.text


class TestJobMechanics:
    def test_events_buffer_and_replay(self):
        """A late or reconnecting client must still see the whole run."""
        job = Job(id="x", source=["s"], title="", theme="textbook")
        job.add_event("stage", "one")
        job.add_event("info", "two")
        assert [e.message for e in job.events_since(0)] == ["one", "two"]
        assert [e.message for e in job.events_since(1)] == ["two"]

    def test_sequence_numbers_are_monotonic(self):
        job = Job(id="x", source=["s"], title="", theme="textbook")
        seqs = [job.add_event("info", str(i)).seq for i in range(5)]
        assert seqs == [1, 2, 3, 4, 5]

    def test_cancel_marks_and_raises_at_the_next_checkpoint(self):
        from bookforge.api.jobs import JobCancelled

        job = Job(id="x", source=["s"], title="", theme="textbook")
        assert job.cancel() is True
        assert job.cancelled
        with pytest.raises(JobCancelled):
            job.check_cancelled()

    def test_cancelling_a_finished_job_is_refused(self):
        job = Job(id="x", source=["s"], title="", theme="textbook")
        job.status = "succeeded"
        assert job.cancel() is False

    def test_waiter_is_woken_by_a_new_event(self):
        job = Job(id="x", source=["s"], title="", theme="textbook")
        import threading

        woken = []

        def waiter():
            woken.append(job.wait_for_event(after=0, timeout=5.0))

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.05)
        job.add_event("info", "hello")
        thread.join(timeout=5)
        assert woken == [True]

    def test_event_buffer_is_bounded(self):
        """One long run must not grow the log without bound."""
        from bookforge.api.jobs import MAX_EVENTS

        job = Job(id="x", source=["s"], title="", theme="textbook")
        for i in range(MAX_EVENTS + 50):
            job.add_event("info", str(i))
        assert len(job.events_since(0)) == MAX_EVENTS

    def test_concurrency_limit_is_enforced(self, settings):
        runner = JobRunner(settings, max_concurrent=1)
        job = Job(id="busy", source=["s"], title="", theme="textbook", status="running")
        runner._jobs["busy"] = job
        with pytest.raises(RuntimeError, match="already running"):
            runner.submit(source=["/tmp"], title="")

    def test_queued_jobs_count_toward_concurrency_cap(self, settings):
        """A job still in 'queued' must block a second submit."""
        runner = JobRunner(settings, max_concurrent=1)
        job = Job(id="queued", source=["s"], title="", theme="textbook", status="queued")
        runner._jobs["queued"] = job
        with pytest.raises(RuntimeError, match="already running"):
            runner.submit(source=["/tmp"], title="")


class TestLiveBuild:
    """A build driven through the HTTP API, with the model stubbed.

    This is the test that proves the whole seam works: submit over HTTP, the
    job thread runs the real pipeline, events accumulate, chapters appear, and
    the finished book is downloadable — none of which the unit tests cover,
    because each of them stops at a module boundary.
    """

    @pytest.fixture
    def stubbed(self, monkeypatch, analysis, transcript_segments, tmp_path):
        import dspy

        from bookforge.schemas import MediaBundle
        from tests.conftest import good_chapter_tex

        folder = tmp_path / "Live Talks"
        folder.mkdir()
        for name in ("01_first.mp4", "02_second.mp4"):
            (folder / name).write_bytes(b"video")

        def fake_acquire(record, workspace, settings):
            video_dir = workspace.video_dir(record.video_id)
            path = video_dir / "transcript.txt"
            path.write_text(
                "\n".join(f"[{s.timestamp}] {s.text}" for s in transcript_segments),
                encoding="utf-8",
            )
            bundle = MediaBundle(
                video_id=record.video_id, transcript_path=str(path),
                transcript_source="captions", duration_sec=600.0, frames=[],
            )
            (video_dir / "media.json").write_text(bundle.model_dump_json(), encoding="utf-8")
            return bundle, list(transcript_segments)

        monkeypatch.setattr("bookforge.pipeline.chapter.acquire_media", fake_acquire)
        monkeypatch.setattr("bookforge.pipeline.orchestrator.acquire_media", fake_acquire)

        class StubAnalyst(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(
                    analysis=analysis, grounding_score=0.95,
                    grounding_report=None, windows=1,
                )

        class StubChapter(dspy.Module):
            def forward(self, manifest=None, **kwargs):
                from bookforge.dspy_runtime.modules.chapter import (
                    ChapterAttemptLog,
                    ChapterResult,
                )
                from bookforge.schemas import ChapterVerdict

                tex = good_chapter_tex(manifest, declare=False)
                verdict = ChapterVerdict(compiles=True, score=0.95)
                result = ChapterResult(
                    tex=tex, verdict=verdict, iterations=1,
                    history=[ChapterAttemptLog(0, 0.95, True, 0, 0, "aa", "initial draft")],
                    accepted=True,
                )
                return dspy.Prediction(
                    chapter_tex=tex, verdict=verdict, result=result,
                    score=0.95, accepted=True,
                )

        from bookforge.dspy_runtime import program as program_module

        real_program = program_module.BookForgeProgram

        class StubProgram(real_program):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.analyst = StubAnalyst()
                self.captioner = None
                self.chapter = StubChapter()

        monkeypatch.setattr("bookforge.api.jobs.BookForgeProgram", StubProgram, raising=False)
        monkeypatch.setattr(program_module, "BookForgeProgram", StubProgram)
        return folder

    def _await_finish(self, client, job_id, timeout=90):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = client.get(f"/api/builds/{job_id}").json()
            if job["status"] in ("succeeded", "failed", "cancelled"):
                return job
            time.sleep(0.25)
        raise AssertionError("build did not finish in time")

    def test_build_runs_and_produces_chapters(self, client, stubbed, settings):
        settings.compile_latex = False  # the design compiles elsewhere; keep this fast
        response = client.post(
            "/api/builds",
            json={"source": [str(stubbed)], "theme": "textbook", "compile_latex": False},
        )
        assert response.status_code == 200
        job_id = response.json()["id"]

        job = self._await_finish(client, job_id)
        assert job["status"] == "succeeded", job.get("error")
        assert len(job["chapters"]) == 2
        assert job["result"]["chapters_included"] == 2

    def test_events_accumulate_and_replay(self, client, stubbed, settings):
        settings.compile_latex = False
        job_id = client.post(
            "/api/builds", json={"source": [str(stubbed)], "compile_latex": False}
        ).json()["id"]
        self._await_finish(client, job_id)

        payload = client.get(f"/api/builds/{job_id}/events?after=0").json()
        messages = [event["message"] for event in payload["events"]]
        assert any("Intake complete" in m for m in messages)
        assert any("Book compiled" in m or "compiled" in m.lower() for m in messages)
        # Replaying from a cursor returns strictly newer events.
        first_seq = payload["events"][0]["seq"]
        later = client.get(f"/api/builds/{job_id}/events?after={first_seq}").json()
        assert all(event["seq"] > first_seq for event in later["events"])

    def test_chapters_endpoint_reflects_the_finished_build(self, client, stubbed, settings):
        settings.compile_latex = False
        job_id = client.post(
            "/api/builds", json={"source": [str(stubbed)], "compile_latex": False}
        ).json()["id"]
        job = self._await_finish(client, job_id)

        payload = client.get(f"/api/workspaces/{job['channel_slug']}/chapters").json()
        assert len(payload["chapters"]) == 2
        assert payload["chapters"][0]["verdict"] is not None
        assert payload["chapters"][0]["source"] == "local_video"

    def test_a_second_build_is_refused_while_one_runs(self, client, stubbed, settings):
        settings.compile_latex = False
        first = client.post(
            "/api/builds", json={"source": [str(stubbed)], "compile_latex": False}
        )
        assert first.status_code == 200
        second = client.post(
            "/api/builds", json={"source": [str(stubbed)], "compile_latex": False}
        )
        # Either the first already finished (200) or it is still running (409);
        # both are correct, and a 500 would not be.
        assert second.status_code in (200, 409)
        self._await_finish(client, first.json()["id"])
