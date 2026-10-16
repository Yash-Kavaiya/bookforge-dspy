"""DSPy-facing metrics: scalar rewards and GEPA feedback.

The feedback string is not decoration — GEPA's reflection model reads it to
decide how to rewrite an instruction. Feedback that says "score was low" teaches
nothing; feedback that names the invented filename teaches the fix. These tests
assert the feedback is specific enough to act on.
"""

from __future__ import annotations

import dspy
import pytest

from bookforge.metrics.composite import (
    feedback_analysis_metric,
    feedback_chapter_metric,
    get_metric,
    scalar_analysis_metric,
    scalar_chapter_metric,
    score_analysis,
)
from tests.conftest import good_chapter_tex


def _analysis_example(transcript: str) -> dspy.Example:
    return dspy.Example(
        video_title="K8s", transcript_chunk=transcript, transcript=transcript
    ).with_inputs("video_title", "transcript_chunk")


def _chapter_example(analysis, manifest) -> dspy.Example:
    return dspy.Example(
        video_title="K8s",
        chapter_number=1,
        chapter_label_slug="k8s",
        analysis=analysis,
        assets_manifest=manifest.describe_for_prompt(),
        assets_manifest_obj=manifest,
    ).with_inputs("video_title", "chapter_number", "chapter_label_slug", "analysis", "assets_manifest")


class TestAnalysisScoring:
    def test_good_analysis_scores_well(self, analysis, transcript_text):
        score, feedback = score_analysis(analysis, transcript_text)
        assert score > 0.55
        # A grounded analysis must never be accused of inventing numbers, even
        # when the feedback suggests it could be richer.
        assert "do not appear in the transcript" not in feedback

    def test_flawless_analysis_gets_reinforcing_feedback(self, analysis, transcript_text):
        from bookforge.schemas import ConceptNote, GlossaryItem

        analysis.concepts.append(
            ConceptNote(
                heading="Service Discovery",
                explanation=(
                    "Service discovery in the cluster is mediated by CoreDNS, which "
                    "answers name lookups for services and caches those answers so "
                    "repeated resolution does not hit the control plane. The cache "
                    "lifetime bounds how quickly a client notices an endpoint change."
                ),
                key_points=["CoreDNS answers service names"],
                evidence_timestamps=["00:01:30"],
            )
        )
        analysis.exercises = analysis.exercises * 5
        # score_analysis's "flawless" message requires every dimension to
        # clear its bar, including >= 4 glossary terms — not just concepts
        # and exercises.
        analysis.glossary += [
            GlossaryItem(term="CoreDNS", definition="The cluster's internal DNS server."),
            GlossaryItem(term="Service discovery", definition="Finding a pod by service name."),
            GlossaryItem(term="Control plane", definition="The components that manage cluster state."),
        ]
        _, feedback = score_analysis(analysis, transcript_text)
        assert "Grounded" in feedback

    def test_missing_analysis_scores_zero(self, transcript_text):
        score, feedback = score_analysis(None, transcript_text)
        assert score == 0.0
        assert "ChapterAnalysis" in feedback

    def test_hallucination_lowers_the_score_and_names_it(self, analysis, transcript_text):
        clean, _ = score_analysis(analysis, transcript_text)
        analysis.concepts[0].explanation += " Throughput hit 88888 megabits per second."
        dirty, feedback = score_analysis(analysis, transcript_text)
        assert dirty < clean
        assert "88888" in feedback

    def test_thin_analysis_is_told_why(self, analysis, transcript_text):
        analysis.concepts = analysis.concepts[:1]
        analysis.exercises = []
        analysis.tables, analysis.charts, analysis.diagrams = [], [], []
        _, feedback = score_analysis(analysis, transcript_text)
        assert "concept" in feedback.lower()
        assert "exercises" in feedback.lower()

    def test_missing_evidence_timestamps_are_flagged(self, analysis, transcript_text):
        for concept in analysis.concepts:
            concept.evidence_timestamps = []
        _, feedback = score_analysis(analysis, transcript_text)
        assert "evidence_timestamps" in feedback

    def test_richer_analysis_outscores_thinner(self, analysis, transcript_text):
        rich, _ = score_analysis(analysis, transcript_text)
        analysis.tables, analysis.charts, analysis.diagrams = [], [], []
        analysis.exercises = []
        thin, _ = score_analysis(analysis, transcript_text)
        assert rich > thin


class TestAnalysisMetricAdapters:
    def test_scalar_metric_returns_a_float(self, analysis, transcript_text):
        value = scalar_analysis_metric(
            _analysis_example(transcript_text), dspy.Prediction(analysis=analysis)
        )
        assert isinstance(value, float) and 0.0 <= value <= 1.0

    def test_scalar_metric_returns_bool_when_bootstrapping(self, analysis, transcript_text):
        value = scalar_analysis_metric(
            _analysis_example(transcript_text),
            dspy.Prediction(analysis=analysis),
            trace=object(),
        )
        assert isinstance(value, bool)

    def test_gepa_metric_returns_score_and_feedback(self, analysis, transcript_text):
        result = feedback_analysis_metric(
            _analysis_example(transcript_text), dspy.Prediction(analysis=analysis)
        )
        assert hasattr(result, "score") and hasattr(result, "feedback")
        assert isinstance(result.feedback, str) and result.feedback

    def test_malformed_prediction_scores_zero(self, transcript_text):
        assert scalar_analysis_metric(
            _analysis_example(transcript_text), dspy.Prediction(analysis="not an analysis")
        ) == 0.0

    def test_missing_output_field_scores_zero(self, transcript_text):
        assert scalar_analysis_metric(
            _analysis_example(transcript_text), dspy.Prediction(something_else=1)
        ) == 0.0


class TestChapterMetric:
    def test_good_chapter_scores_above_a_stub(self, analysis, assets_manifest):
        example = _chapter_example(analysis, assets_manifest)
        good = scalar_chapter_metric(
            example, dspy.Prediction(chapter_tex=good_chapter_tex(assets_manifest))
        )
        stub = scalar_chapter_metric(example, dspy.Prediction(chapter_tex=r"\chapter{X}"))
        assert good > stub

    def test_feedback_names_the_invented_file(self, analysis, assets_manifest):
        tex = good_chapter_tex(assets_manifest) + "\n\\bfFigure{fig:hallucinated}\n"
        result = feedback_chapter_metric(
            _chapter_example(analysis, assets_manifest), dspy.Prediction(chapter_tex=tex)
        )
        assert "fig:hallucinated" in result.feedback
        assert result.score < 1.0

    def test_feedback_leads_with_compilation_status(self, analysis, assets_manifest):
        result = feedback_chapter_metric(
            _chapter_example(analysis, assets_manifest),
            dspy.Prediction(chapter_tex=good_chapter_tex(assets_manifest)),
        )
        assert "compil" in result.feedback.lower()

    def test_example_without_context_is_reported_not_crashed(self):
        result = feedback_chapter_metric(
            dspy.Example(video_title="x"), dspy.Prediction(chapter_tex="whatever")
        )
        assert result.score == 0.0
        assert "missing" in result.feedback

    def test_empty_chapter_scores_zero(self, analysis, assets_manifest):
        assert scalar_chapter_metric(
            _chapter_example(analysis, assets_manifest), dspy.Prediction(chapter_tex="")
        ) == 0.0


class TestRegistry:
    @pytest.mark.parametrize("target", ["analysis", "chapter"])
    @pytest.mark.parametrize("feedback", [True, False])
    def test_lookup(self, target, feedback):
        assert callable(get_metric(target, feedback=feedback))

    def test_unknown_target(self):
        with pytest.raises(KeyError):
            get_metric("nonsense")

    def test_scalar_and_feedback_agree(self, analysis, transcript_text):
        """The optimizer's objective must equal what production ships against."""
        example = _analysis_example(transcript_text)
        prediction = dspy.Prediction(analysis=analysis)
        assert scalar_analysis_metric(example, prediction) == pytest.approx(
            feedback_analysis_metric(example, prediction).score
        )
