"""The analyst's map-reduce, and the writer/critic/refiner modules.

The analyst is where the ADK pipeline lost the middle of long videos. These
tests drive its windowing logic with scripted predictors so the control flow is
exercised without a live model.
"""

from __future__ import annotations

import dspy
import pytest

from bookforge.dspy_runtime.modules.analyst import TranscriptAnalyst
from bookforge.dspy_runtime.modules.critic import SemanticCritic
from bookforge.dspy_runtime.modules.writer import ChapterRefiner, ChapterWriter
from bookforge.dspy_runtime.signatures import (
    ChunkAnalysis,
    SemanticCritique,
    SemanticDefect,
)
from bookforge.schemas import ChapterVerdict, ConceptNote, TranscriptSegment


def _long_segments(n: int = 400) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            start_sec=i * 5.0,
            text=(
                f"At this point the speaker explains topic {i}, covering how the "
                f"cluster handles case {i} and why that matters in practice."
            ),
        )
        for i in range(n)
    ]


class ScriptedChunk(dspy.Module):
    def __init__(self, fail_indices: set[int] | None = None) -> None:
        super().__init__()
        self.fail_indices = fail_indices or set()
        self.seen: list[int] = []

    def forward(self, chunk_index: int = 1, **kwargs) -> dspy.Prediction:
        self.seen.append(chunk_index)
        if chunk_index in self.fail_indices:
            raise RuntimeError("simulated window failure")
        return dspy.Prediction(
            analysis=ChunkAnalysis(
                concepts=[
                    ConceptNote(
                        heading=f"Concept from window {chunk_index}",
                        explanation=(
                            "The speaker explains how the cluster handles this case "
                            "and why the behaviour matters when operating at scale."
                        ),
                    )
                ],
                segment_summary=f"window {chunk_index}",
            )
        )


class ScriptedMerge(dspy.Module):
    def __init__(self, analysis) -> None:
        super().__init__()
        self.analysis = analysis
        self.received: list[ChunkAnalysis] = []

    def forward(self, chunk_analyses=None, **kwargs) -> dspy.Prediction:
        self.received = list(chunk_analyses or [])
        return dspy.Prediction(analysis=self.analysis)


class ScriptedSingle(dspy.Module):
    def __init__(self, analysis) -> None:
        super().__init__()
        self.analysis = analysis
        self.calls = 0
        self.seen_transcript = ""

    def forward(self, transcript: str = "", **kwargs) -> dspy.Prediction:
        self.calls += 1
        self.seen_transcript = transcript
        return dspy.Prediction(analysis=self.analysis)


def _analyst(settings, analysis, fail_indices=None) -> TranscriptAnalyst:
    module = TranscriptAnalyst(settings)
    module.analyze_chunk = ScriptedChunk(fail_indices)
    module.merge = ScriptedMerge(analysis)
    module.single_shot = ScriptedSingle(analysis)
    return module


class TestAnalystRouting:
    def test_short_transcript_uses_single_shot(self, settings, analysis, transcript_segments):
        module = _analyst(settings, analysis)
        result = module(video_title="K8s", segments=transcript_segments)
        assert module.single_shot.calls == 1
        assert module.analyze_chunk.seen == []
        assert result.windows == 1

    def test_visual_insights_reach_the_model_but_not_grounding(
        self, settings, analysis, transcript_segments
    ):
        analysis.tables, analysis.charts = [], []
        analysis.summary = "Accuracy was 99.7 percent on the slide."
        for concept in analysis.concepts:
            concept.explanation = (
                "The speaker discussed accuracy and the slide showed a 99.7 percent "
                "figure that was never spoken aloud in the recording itself."
            )
            concept.key_points = []
        module = _analyst(settings, analysis)
        result = module(
            video_title="K8s",
            segments=transcript_segments,
            visual_insights="Slide at 01:12 shows 99.7% top-1 accuracy.",
        )
        assert "Visual screen briefing" in module.single_shot.seen_transcript
        assert "99.7" in module.single_shot.seen_transcript
        assert "99.7" in result.grounding_report.ungrounded

    def test_long_transcript_uses_map_reduce(self, settings, analysis):
        settings.transcript_chunk_chars = 3_000
        settings.transcript_chunk_overlap_chars = 300
        module = _analyst(settings, analysis)
        result = module(video_title="K8s", segments=_long_segments())
        assert result.windows > 1
        assert module.single_shot.calls == 0
        assert len(module.analyze_chunk.seen) == result.windows

    def test_every_window_reaches_the_merge(self, settings, analysis):
        """No window may be silently dropped — that was the truncation bug."""
        settings.transcript_chunk_chars = 3_000
        settings.transcript_chunk_overlap_chars = 300
        module = _analyst(settings, analysis)
        result = module(video_title="K8s", segments=_long_segments())
        assert len(module.merge.received) == result.windows

    def test_one_failed_window_does_not_lose_the_rest(self, settings, analysis):
        settings.transcript_chunk_chars = 3_000
        settings.transcript_chunk_overlap_chars = 300
        module = _analyst(settings, analysis, fail_indices={2})
        result = module(video_title="K8s", segments=_long_segments())
        assert len(module.merge.received) == result.windows - 1
        assert result.analysis is analysis

    def test_all_windows_failing_raises(self, settings, analysis):
        settings.transcript_chunk_chars = 3_000
        settings.transcript_chunk_overlap_chars = 300
        module = _analyst(settings, analysis, fail_indices=set(range(1, 100)))
        with pytest.raises(RuntimeError, match="every transcript window failed"):
            module(video_title="K8s", segments=_long_segments())

    def test_plain_text_falls_back_to_single_shot(self, settings, analysis, transcript_text):
        module = _analyst(settings, analysis)
        module(video_title="K8s", transcript=transcript_text)
        assert module.single_shot.calls == 1
        assert "Kubernetes" in module.single_shot.seen_transcript

    def test_no_input_is_rejected(self, settings, analysis):
        with pytest.raises(ValueError, match="segments or a transcript"):
            _analyst(settings, analysis)(video_title="K8s")

    def test_grounding_is_reported(self, settings, analysis, transcript_segments):
        result = _analyst(settings, analysis)(video_title="K8s", segments=transcript_segments)
        assert 0.0 <= result.grounding_score <= 1.0
        assert result.grounding_report is not None


class TestWriter:
    def test_output_is_sanitized(self, settings, analysis, assets_manifest):
        writer = ChapterWriter(settings)

        class Fenced(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(
                    chapter_tex="```latex\n\\documentclass{book}\n\\chapter{X}\n```"
                )

        writer.write = Fenced()
        tex = writer(
            video_title="K8s", chapter_number=1, chapter_label_slug="k8s",
            analysis=analysis, manifest=assets_manifest,
        ).chapter_tex
        assert "documentclass" not in tex
        assert "```" not in tex
        assert r"\chapter{X}" in tex

    def test_manifest_is_rendered_for_the_prompt(self, settings, analysis, assets_manifest):
        writer = ChapterWriter(settings)
        captured = {}

        class Capture(dspy.Module):
            def forward(self, assets_manifest: str = "", **kwargs):
                captured["manifest"] = assets_manifest
                return dspy.Prediction(chapter_tex=r"\chapter{X}")

        writer.write = Capture()
        writer(
            video_title="K8s", chapter_number=1, chapter_label_slug="k8s",
            analysis=analysis, manifest=assets_manifest,
        )
        assert "chart_1.pdf" in captured["manifest"]
        assert "fig:ch01-chart1" in captured["manifest"]
        assert "table_1.tex" in captured["manifest"]


class TestRefiner:
    def test_truncated_revision_is_discarded(self, settings, analysis, assets_manifest):
        """A refiner returning a stub has failed; keeping it would lose content."""
        refiner = ChapterRefiner(settings)

        class Truncating(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(chapter_tex=r"\chapter{X}")

        refiner.refine = Truncating()
        original = r"\chapter{X}" + "\nword" * 500
        result = refiner(
            video_title="K8s", chapter_number=1, chapter_label_slug="k8s",
            analysis=analysis, manifest=assets_manifest,
            current_chapter_tex=original, critique="fix things",
        )
        assert result.degenerate is True
        assert result.chapter_tex == original

    def test_substantial_revision_is_kept(self, settings, analysis, assets_manifest):
        refiner = ChapterRefiner(settings)
        revised = r"\chapter{Revised}" + "\nword" * 500

        class Good(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(chapter_tex=revised)

        refiner.refine = Good()
        result = refiner(
            video_title="K8s", chapter_number=1, chapter_label_slug="k8s",
            analysis=analysis, manifest=assets_manifest,
            current_chapter_tex=r"\chapter{X}" + "\nword" * 500,
            critique="fix things",
        )
        assert result.degenerate is False
        assert "Revised" in result.chapter_tex


class TestSemanticCritic:
    def test_defects_are_normalized(self, settings, analysis):
        critic = SemanticCritic(settings)

        class Scripted(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(
                    critique=SemanticCritique(
                        defects=[
                            SemanticDefect(severity="CRITICAL", message="Wrong claim.", fix="Cut it."),
                            SemanticDefect(severity="nonsense", message="Vague.", fix=""),
                            SemanticDefect(severity="minor", message="   ", fix="x"),
                        ]
                    )
                )

        critic.critique = Scripted()
        defects = critic(
            video_title="K8s", analysis=analysis,
            chapter_tex=r"\chapter{X}", verdict=ChapterVerdict(compiles=True),
        ).defects
        assert [d.severity for d in defects] == ["critical", "minor"]
        assert all(d.message.strip() for d in defects)

    def test_mechanical_findings_are_passed_through(self, settings, analysis):
        critic = SemanticCritic(settings)
        captured = {}

        class Capture(dspy.Module):
            def forward(self, mechanical_findings: str = "", **kwargs):
                captured["findings"] = mechanical_findings
                return dspy.Prediction(critique=SemanticCritique())

        critic.critique = Capture()
        critic(
            video_title="K8s", analysis=analysis, chapter_tex=r"\chapter{X}",
            verdict=ChapterVerdict(
                compiles=False, score=0.3,
                invented_assets=["figures/ghost.pdf"],
                missing_sections=["Glossary"],
            ),
        )
        findings = captured["findings"]
        assert "Compiled: NO" in findings
        assert "figures/ghost.pdf" in findings
        assert "Glossary" in findings

    def test_backend_failure_yields_no_defects(self, settings, analysis):
        """A failed critique means 'nothing extra found', never a failed chapter."""
        critic = SemanticCritic(settings)

        class Failing(dspy.Module):
            def forward(self, **kwargs):
                raise RuntimeError("model unavailable")

        critic.critique = Failing()
        result = critic(
            video_title="K8s", analysis=analysis,
            chapter_tex=r"\chapter{X}", verdict=ChapterVerdict(compiles=True),
        )
        assert result.defects == [] and result.failed is True
