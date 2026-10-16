"""The DSPy layer: signatures, LM routing, artifacts and the program container."""

from __future__ import annotations

import json

import pytest

from bookforge.dspy_runtime.artifacts import (
    describe_artifacts,
    load_metadata,
    load_program,
    save_program,
    signature_fingerprint,
)
from bookforge.dspy_runtime.lm import USAGE, LMRouter, UsageTracker, build_lm
from bookforge.dspy_runtime.modules import (
    ChapterProgram,
    FrameCaptioner,
    TranscriptAnalyst,
)
from bookforge.dspy_runtime.program import BookForgeProgram
from bookforge.dspy_runtime.signatures import (
    AnalyzeTranscriptChunk,
    CaptionFrames,
    CritiqueChapter,
    MergeChunkAnalyses,
    RefineChapter,
    SingleShotAnalysis,
    WriteChapter,
)
from bookforge.schemas import ChapterAnalysis

ALL_SIGNATURES = [
    AnalyzeTranscriptChunk,
    MergeChunkAnalyses,
    SingleShotAnalysis,
    CaptionFrames,
    WriteChapter,
    RefineChapter,
    CritiqueChapter,
]


class TestSignatures:
    @pytest.mark.parametrize("signature", ALL_SIGNATURES, ids=lambda s: s.__name__)
    def test_has_instructions(self, signature):
        assert signature.instructions and len(signature.instructions) > 80

    @pytest.mark.parametrize("signature", ALL_SIGNATURES, ids=lambda s: s.__name__)
    def test_has_inputs_and_outputs(self, signature):
        assert signature.input_fields, f"{signature.__name__} has no inputs"
        assert signature.output_fields, f"{signature.__name__} has no outputs"

    def test_analysis_output_is_typed(self):
        assert SingleShotAnalysis.output_fields["analysis"].annotation is ChapterAnalysis

    def test_chapter_signatures_agree_on_field_names(self):
        """The refiner must accept what the writer produces."""
        assert "chapter_tex" in WriteChapter.output_fields
        assert "chapter_tex" in RefineChapter.output_fields
        assert "current_chapter_tex" in RefineChapter.input_fields

    def test_writer_instructions_carry_the_latex_contract(self):
        text = WriteChapter.instructions
        for required in ("Overview", "Learning Objectives", "Glossary", "Solutions"):
            assert required in text
        assert r"\includegraphics" in text
        assert "manifest" in text.lower()

    def test_instructions_may_contain_literal_braces(self):
        r"""ADK's {placeholder} injection banned literal braces from prompts.

        That is why the original writer prompt had to *describe* LaTeX in
        English instead of showing it. DSPy passes inputs as arguments, so real
        markup can appear — and does.
        """
        assert r"\begin{figure}" in WriteChapter.instructions


class TestModuleConstruction:
    def test_analyst_exposes_three_predictors(self, settings):
        names = {name for name, _ in TranscriptAnalyst(settings).named_predictors()}
        assert any("analyze_chunk" in n for n in names)
        assert any("merge" in n for n in names)
        assert any("single_shot" in n for n in names)

    def test_chapter_program_exposes_writer_and_refiner(self, settings):
        names = {name for name, _ in ChapterProgram(settings).named_predictors()}
        assert any("writer" in n for n in names)
        assert any("refiner" in n for n in names)
        assert any("critic" in n for n in names)

    def test_critic_can_be_disabled(self, settings):
        program = ChapterProgram(settings, use_semantic_critic=False)
        assert program.critic is None
        assert not any("critic" in n for n, _ in program.named_predictors())

    def test_captioner_has_one_predictor(self, settings):
        assert len(list(FrameCaptioner(settings).named_predictors())) == 1


class TestLMRouting:
    def test_role_temperatures_differ(self, settings):
        assert build_lm("analyst", settings).kwargs["temperature"] == settings.analyst_temperature
        assert build_lm("critic", settings).kwargs["temperature"] == 0.0

    def test_openai_base_url_is_applied(self, settings):
        lm = build_lm("writer", settings)
        assert lm.kwargs["api_base"] == settings.openai_api_base
        assert lm.kwargs["api_key"] == "test-key-not-used"

    def test_gemini_models_skip_openai_credentials(self, settings):
        settings.writer_model = "gemini/gemini-2.5-pro"
        lm = build_lm("writer", settings)
        assert "api_base" not in lm.kwargs

    @pytest.mark.parametrize(
        "model", ["openai/gpt-6-sol", "openai/gpt-5.6-sol", "openai/o3", "openai/gpt-5"]
    )
    def test_reasoning_models_get_default_temperature_and_completion_budget(
        self, settings, model
    ):
        """gpt-6-sol 400s on max_tokens and on any temperature but 1."""
        settings.critic_model = model
        lm = build_lm("critic", settings)
        assert lm.kwargs.get("temperature") is None
        assert not lm.kwargs.get("max_tokens")
        assert lm.kwargs["max_completion_tokens"] >= 32_000

    @pytest.mark.parametrize(
        "model", ["openai/gpt-4o", "openai/gpt-4.1-mini", "openai/gpt-5-chat-latest",
                  "openai/nvidia/nemotron-3-ultra-550b-a55b"]
    )
    def test_ordinary_models_keep_role_temperature(self, settings, model):
        from bookforge.dspy_runtime.lm import is_reasoning_model

        assert not is_reasoning_model(model)
        settings.critic_model = model
        assert build_lm("critic", settings).kwargs["temperature"] == 0.0

    def test_resample_temperature_never_reaches_a_reasoning_model(self, settings):
        settings.writer_model = "openai/gpt-6-sol"
        lm = build_lm("writer", settings).copy(rollout_id=1, temperature=0.7)
        assert lm._sanitize({"temperature": 0.7})["temperature"] is None

    def test_usage_is_read_from_the_provider_response(self, settings):
        """on_lm_end only sees completion text; tokens must come from forward."""
        from types import SimpleNamespace

        USAGE.reset()
        lm = build_lm("writer", settings)
        lm._record_usage(SimpleNamespace(usage={"prompt_tokens": 11, "completion_tokens": 4}))
        totals = USAGE.totals()
        assert (totals.prompt_tokens, totals.completion_tokens, totals.calls) == (11, 4, 0)
        USAGE.reset()

    def test_unknown_role_rejected(self, settings):
        with pytest.raises(ValueError, match="unknown LM role"):
            build_lm("nonsense", settings)

    def test_router_caches_per_role(self, settings):
        router = LMRouter(settings)
        assert router.get("analyst") is router.get("analyst")
        assert router.get("analyst") is not router.get("writer")


class TestUsageTracking:
    def test_records_and_totals(self):
        tracker = UsageTracker()
        tracker.record("analyst", 100, 50)
        tracker.record("analyst", 200, 60)
        tracker.record("writer", 300, 90, failed=True)
        total = tracker.totals()
        assert total.calls == 3
        assert total.prompt_tokens == 600
        assert total.completion_tokens == 200
        assert total.total_tokens == 800
        assert total.failures == 1

    def test_summary_lists_each_role(self):
        tracker = UsageTracker()
        tracker.record("analyst", 10, 5)
        tracker.record("writer", 20, 8)
        summary = tracker.summary()
        assert "analyst" in summary and "writer" in summary and "TOTAL" in summary

    def test_empty_summary(self):
        assert "No LLM calls" in UsageTracker().summary()

    def test_global_tracker_resets(self):
        USAGE.reset()
        assert USAGE.totals().calls == 0


class TestArtifacts:
    def test_fingerprint_is_stable(self, settings):
        assert signature_fingerprint(TranscriptAnalyst(settings)) == signature_fingerprint(
            TranscriptAnalyst(settings)
        )

    def test_fingerprint_differs_between_programs(self, settings):
        assert signature_fingerprint(TranscriptAnalyst(settings)) != signature_fingerprint(
            FrameCaptioner(settings)
        )

    def test_save_writes_program_and_metadata(self, settings, tmp_path):
        program = TranscriptAnalyst(settings)
        save_program(
            program, tmp_path, "analyst", optimizer="gepa",
            val_score=0.81, baseline_score=0.63, trainset_size=10, valset_size=6,
        )
        assert (tmp_path / "analyst.json").exists()
        meta = json.loads((tmp_path / "analyst.meta.json").read_text())
        assert meta["optimizer"] == "gepa"
        assert meta["signature_fingerprint"] == signature_fingerprint(program)

    def test_metadata_reports_improvement(self, settings, tmp_path):
        save_program(
            TranscriptAnalyst(settings), tmp_path, "analyst",
            optimizer="gepa", val_score=0.81, baseline_score=0.63,
        )
        assert load_metadata(tmp_path, "analyst").improvement == pytest.approx(0.18)

    def test_roundtrip_restores_instructions(self, settings, tmp_path):
        source = TranscriptAnalyst(settings)
        source.single_shot.predict.signature = source.single_shot.predict.signature.with_instructions(
            "OPTIMIZED INSTRUCTION TEXT"
        )
        save_program(source, tmp_path, "analyst", optimizer="gepa")

        target = TranscriptAnalyst(settings)
        assert load_program(target, tmp_path, "analyst")
        assert "OPTIMIZED" in target.single_shot.predict.signature.instructions

    def test_missing_artifact_is_not_an_error(self, settings, tmp_path):
        assert load_program(TranscriptAnalyst(settings), tmp_path, "absent") is False

    def test_signature_mismatch_is_refused(self, settings, tmp_path):
        """Old instructions can name fields that no longer exist."""
        save_program(TranscriptAnalyst(settings), tmp_path, "analyst", optimizer="gepa")
        meta_path = tmp_path / "analyst.meta.json"
        meta = json.loads(meta_path.read_text())
        meta["signature_fingerprint"] = "deadbeefdeadbeef"
        meta_path.write_text(json.dumps(meta))

        assert load_program(TranscriptAnalyst(settings), tmp_path, "analyst") is False
        with pytest.raises(RuntimeError, match="Re-run optimization"):
            load_program(TranscriptAnalyst(settings), tmp_path, "analyst", strict=True)

    def test_describe_lists_saved_programs(self, settings, tmp_path):
        save_program(
            TranscriptAnalyst(settings), tmp_path, "analyst",
            optimizer="gepa", val_score=0.8, baseline_score=0.6,
        )
        listing = describe_artifacts(tmp_path)
        assert "analyst" in listing and "gepa" in listing and "+0.200" in listing

    def test_describe_empty_directory(self, tmp_path):
        assert "No optimized programs" in describe_artifacts(tmp_path)


class TestProgramContainer:
    def test_builds_all_components(self, settings):
        program = BookForgeProgram(settings, load_optimized=False)
        assert set(program.components()) == {"analyst", "captioner", "chapter"}
        assert program.loaded_artifacts == []

    def test_assigns_role_specific_lms(self, settings):
        settings.critic_model = "openai/critic-model"
        settings.writer_model = "openai/writer-model"
        program = BookForgeProgram(settings, load_optimized=False)
        assert program.chapter.writer.get_lm().model == "openai/writer-model"
        assert program.chapter.critic.get_lm().model == "openai/critic-model"

    def test_loads_artifacts_when_present(self, settings):
        directory = settings.artifacts_dir_abs
        save_program(TranscriptAnalyst(settings), directory, "analyst", optimizer="gepa")
        program = BookForgeProgram(settings, load_optimized=True)
        assert "analyst" in program.loaded_artifacts

    def test_lms_survive_artifact_loading(self, settings):
        """Loading state can replace predictors; the LMs must be re-attached."""
        save_program(TranscriptAnalyst(settings), settings.artifacts_dir_abs, "analyst", optimizer="gepa")
        program = BookForgeProgram(settings, load_optimized=True)
        assert program.analyst.get_lm() is not None

    def test_describe_mentions_models(self, settings):
        text = BookForgeProgram(settings, load_optimized=False).describe()
        assert settings.writer_model in text
        assert "optimized: none" in text
