"""Configuration, intake and the command line."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bookforge.cli import build_parser
from bookforge.config import Settings, preflight
from bookforge.pipeline.intake import extract_channel_url


class TestSettings:
    def test_defaults_are_sane(self, tmp_path):
        settings = Settings(workspace_root=tmp_path)
        assert settings.qa_max_iterations >= 1
        assert settings.transcript_chunk_overlap_chars < settings.transcript_chunk_chars
        assert settings.qa_min_acceptable_score <= settings.qa_accept_score

    def test_env_prefix(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BOOKFORGE_QA_MAX_ITERATIONS", "7")
        monkeypatch.setenv("BOOKFORGE_WRITER_MODEL", "gemini/gemini-2.5-pro")
        settings = Settings(workspace_root=tmp_path)
        assert settings.qa_max_iterations == 7
        assert settings.writer_model == "gemini/gemini-2.5-pro"

    def test_overlap_must_be_smaller_than_chunk(self, tmp_path):
        with pytest.raises(ValidationError, match="smaller than"):
            Settings(
                workspace_root=tmp_path,
                transcript_chunk_chars=1000,
                transcript_chunk_overlap_chars=1000,
            )

    def test_acceptance_thresholds_must_be_ordered(self, tmp_path):
        with pytest.raises(ValidationError, match="cannot exceed"):
            Settings(
                workspace_root=tmp_path,
                qa_accept_score=0.5,
                qa_min_acceptable_score=0.9,
            )

    def test_zero_iterations_is_clamped(self, tmp_path):
        assert Settings(workspace_root=tmp_path, qa_max_iterations=0).qa_max_iterations == 1

    def test_credentials_detected_from_any_provider(self, monkeypatch, tmp_path):
        for variable in (
            "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY", "NVIDIA_API_KEY", "BOOKFORGE_OPENAI_API_KEY",
            "GOOGLE_GENAI_USE_VERTEXAI",
        ):
            monkeypatch.delenv(variable, raising=False)
        assert not Settings(workspace_root=tmp_path).has_llm_credentials()

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert Settings(workspace_root=tmp_path).has_llm_credentials()

    def test_paths_resolve_to_absolute(self, tmp_path):
        settings = Settings(workspace_root=tmp_path / "data")
        assert settings.workspace_root_abs.is_absolute()
        assert settings.artifacts_dir_abs.is_absolute()


class TestPreflight:
    def test_missing_credentials_is_fatal(self, monkeypatch, tmp_path):
        for variable in (
            "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY", "NVIDIA_API_KEY", "BOOKFORGE_OPENAI_API_KEY",
            "GOOGLE_GENAI_USE_VERTEXAI",
        ):
            monkeypatch.delenv(variable, raising=False)
        with pytest.raises(RuntimeError, match="No LLM credentials"):
            preflight(Settings(workspace_root=tmp_path))

    def test_returns_warnings_without_mutating_settings(self, settings, monkeypatch):
        """The old preflight silently flipped compile_latex as a side effect."""
        monkeypatch.setattr("shutil.which", lambda name: None)
        settings.compile_latex = True
        warnings = preflight(settings)
        assert any("pdflatex" in w for w in warnings)
        assert any("ffmpeg" in w for w in warnings)
        assert settings.compile_latex is True, "preflight mutated the caller's settings"

    def test_no_warnings_when_tools_present(self, settings, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        assert preflight(settings) == []


class TestUrlExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("https://www.youtube.com/@abc", "https://www.youtube.com/@abc"),
            ("Make a book from https://youtu.be/xyz please", "https://youtu.be/xyz"),
            ("see https://www.youtube.com/@abc.", "https://www.youtube.com/@abc"),
            ("go to https://www.youtube.com/@abc/", "https://www.youtube.com/@abc"),
        ],
    )
    def test_extracts_and_trims(self, text, expected):
        assert extract_channel_url(text) == expected

    @pytest.mark.parametrize("text", ["", "no url here", "https://vimeo.com/1", None])
    def test_returns_none_without_a_youtube_url(self, text):
        assert extract_channel_url(text) is None


class TestParser:
    def test_build_command(self):
        args = build_parser().parse_args(
            ["build", "https://youtube.com/@x", "--max-videos", "3", "--no-latex"]
        )
        assert args.command == "build"
        assert args.max_videos == 3
        assert args.no_latex is True

    def test_optimize_command(self):
        args = build_parser().parse_args(
            ["optimize", "analyst", "--optimizer", "gepa", "--budget", "medium"]
        )
        assert args.target == "analyst"
        assert args.optimizer == "gepa"
        assert args.budget == "medium"

    def test_evaluate_command(self):
        args = build_parser().parse_args(["evaluate", "chapter", "--optimized"])
        assert args.target == "chapter" and args.optimized is True

    def test_verify_command(self):
        args = build_parser().parse_args(["verify", "a/chapter.tex", "--json"])
        assert args.chapter == "a/chapter.tex" and args.json is True

    def test_artifacts_command(self):
        assert build_parser().parse_args(["artifacts"]).command == "artifacts"

    def test_invalid_optimizer_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["optimize", "analyst", "--optimizer", "nope"])

    def test_command_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


class TestExtensionEnvParsing:
    """Extension lists are tuples, and tuples come from the environment as JSON.

    ``BOOKFORGE_VIDEO_EXTENSIONS=.mp4,.mkv`` -- the spelling anyone would try --
    used to raise a SettingsError naming the field but not the expected format.
    """

    def test_comma_separated_list(self, monkeypatch):
        from bookforge.config import Settings

        monkeypatch.setenv("BOOKFORGE_VIDEO_EXTENSIONS", ".mp4,.mkv")
        assert Settings().video_extensions == (".mp4", ".mkv")

    def test_leading_dot_and_case_are_normalised(self, monkeypatch):
        from bookforge.config import Settings

        monkeypatch.setenv("BOOKFORGE_TRANSCRIPT_EXTENSIONS", "srt, VTT")
        assert Settings().transcript_extensions == (".srt", ".vtt")

    def test_json_list_still_works(self, monkeypatch):
        from bookforge.config import Settings

        monkeypatch.setenv("BOOKFORGE_VIDEO_EXTENSIONS", '[".mov"]')
        assert Settings().video_extensions == (".mov",)
