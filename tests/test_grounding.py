"""Grounding — turning "don't hallucinate" from a prompt into a measurement."""

from __future__ import annotations

from pathlib import Path

from bookforge.metrics.grounding import check_chapter_grounding, check_grounding


class TestAnalysisGrounding:
    def test_grounded_analysis_scores_high(self, analysis, transcript_text):
        report = check_grounding(analysis, transcript_text)
        assert report.ungrounded == []
        assert report.score > 0.9

    def test_invented_number_is_caught(self, analysis, transcript_text):
        """The exact failure the ADK critic was asked to catch by reading."""
        analysis.concepts[0].explanation += (
            " Benchmarks showed a 4382 percent improvement in latency."
        )
        report = check_grounding(analysis, transcript_text)
        assert "4382" in report.ungrounded
        assert report.numeric_score < 1.0
        assert "4382" in report.feedback()

    def test_fabricated_chart_values_are_caught(self, analysis, transcript_text):
        analysis.charts[0].values = [1234.0, 5678.0]
        analysis.charts[0].categories = ["Host", "Overlay"]
        report = check_grounding(analysis, transcript_text)
        assert {"1234", "5678"} <= set(report.ungrounded)

    def _single_claim(self, analysis, claim: str):
        """Reduce the fixture to exactly one numeric assertion."""
        analysis.tables, analysis.charts, analysis.diagrams = [], [], []
        analysis.glossary = []
        analysis.learning_objectives = ["The reader can read."] * 2
        analysis.chapter_title = "A Chapter"
        analysis.summary = claim
        analysis.concepts = analysis.concepts[:1]
        analysis.concepts[0].heading = "The Measurement"
        analysis.concepts[0].explanation = (
            f"{claim} The speaker reported this while walking through the results, "
            "and it is the only figure this section relies upon."
        )
        analysis.concepts[0].key_points = []
        return analysis

    def test_word_numbers_in_transcript_count_as_grounded(self, analysis):
        """Auto-captions spell numbers out; a digit-only check under-reports badly."""
        analysis = self._single_claim(analysis, "The cluster had 25 nodes.")
        report = check_grounding(analysis, "The cluster had twenty five nodes running.")
        assert "25" not in report.ungrounded

    def test_percentage_forms_are_reconciled(self, analysis):
        analysis = self._single_claim(analysis, "Throughput fell to 0.17 of baseline.")
        report = check_grounding(analysis, "Throughput dropped by 17 percent under load.")
        assert "0.17" not in report.ungrounded

    def test_x100_without_percent_cue_is_not_auto_grounded(self, analysis):
        """Bare ×100 matching made '2' and '200' look interchangeable."""
        analysis = self._single_claim(analysis, "The cluster had 200 nodes.")
        report = check_grounding(analysis, "The speaker mentioned the number 2 twice.")
        assert "200" in report.ungrounded

    def test_bare_word_number_without_unit_is_not_vocabulary(self, analysis):
        # extract_numbers ignores single-digit literals by design.
        analysis = self._single_claim(analysis, "There were 50 replicas.")
        report = check_grounding(
            analysis, "Fifty ideas keep the control plane small."
        )
        assert "50" in report.ungrounded

    def test_visual_briefing_numbers_do_not_ground_analysis(self, analysis):
        """Gemini on-screen figures must not launder invented claims."""
        analysis = self._single_claim(
            analysis, "The slide showed a 99.7 percent accuracy figure."
        )
        spoken = "The speaker discussed accuracy without stating a number."
        briefing = "At 01:12 the slide shows 99.7% top-1 accuracy."
        # Spoken-only grounding (what TranscriptAnalyst._finalize must use).
        report = check_grounding(analysis, spoken)
        assert "99.7" in report.ungrounded
        # Contaminated grounding would incorrectly pass — prove the contrast.
        contaminated = check_grounding(analysis, spoken + "\n" + briefing)
        assert "99.7" not in contaminated.ungrounded

    def test_scaled_numbers_are_reconciled(self, analysis):
        analysis = self._single_claim(analysis, "The system served 1500 requests per second.")
        report = check_grounding(analysis, "We served 1.5k requests per second at peak.")
        assert "1500" not in report.ungrounded

    def test_off_topic_concept_is_flagged(self, analysis, transcript_text):
        analysis.concepts[0].heading = "Baroque Counterpoint"
        analysis.concepts[0].explanation = (
            "Baroque counterpoint interweaves independent melodic voices according "
            "to strict harmonic conventions established during the seventeenth "
            "century, particularly in fugal composition and canonic imitation."
        )
        report = check_grounding(analysis, transcript_text)
        assert report.unsupported_concepts
        assert "Baroque" in report.feedback()

    def test_empty_transcript_is_neutral(self, analysis):
        report = check_grounding(analysis, "")
        assert report.checked == 0
        assert report.numeric_score == 1.0

    def test_no_numbers_asserted_is_not_a_failure(self, analysis, transcript_text):
        analysis.tables, analysis.charts = [], []
        analysis.summary = "A discussion of networking concepts."
        for concept in analysis.concepts:
            concept.explanation = (
                "Pods communicate directly across a flat address space, so no port "
                "mapping layer sits between them and their peers in the cluster."
            )
            concept.key_points = []
        report = check_grounding(analysis, transcript_text)
        assert report.numeric_score == 1.0


class TestChapterGrounding:
    def test_chapter_matching_analysis_is_grounded(self, analysis, assets_manifest):
        from tests.conftest import good_chapter_tex

        report = check_chapter_grounding(good_chapter_tex(assets_manifest), analysis)
        assert report.ungrounded == []

    def test_writer_invented_number_is_caught(self, analysis, assets_manifest):
        from tests.conftest import good_chapter_tex

        tex = good_chapter_tex(assets_manifest) + "\nLatency improved by 9999 percent.\n"
        report = check_chapter_grounding(tex, analysis)
        assert "9999" in report.ungrounded

    def test_latex_structural_numbers_are_not_flagged(self, analysis, assets_manifest):
        """0.85\\textwidth and label suffixes are markup, not claims."""
        from tests.conftest import good_chapter_tex

        report = check_chapter_grounding(good_chapter_tex(assets_manifest), analysis)
        assert "85" not in report.ungrounded
        assert "01" not in report.ungrounded

    def test_declaration_paths_are_not_numeric_claims(self, analysis, assets_manifest):
        """Asset declaration paths (screen-recording timestamps in folder names)
        must not fail QA as invented numbers."""
        from bookforge.design.layout import declaration_block
        from tests.conftest import good_chapter_tex

        # Simulate a local-upload slug with date fragments in the path.
        figures = []
        for fig in assets_manifest.figures:
            figures.append(
                fig.model_copy(
                    update={
                        "filename": (
                            "chapters/01_01-27-13-25-40/figures/"
                            + Path(fig.filename).name
                        )
                    }
                )
            )
        from bookforge.schemas import AssetsManifest

        mangled = AssetsManifest(
            chapter_slug="01_01-27-13-25-40",
            chapter_dir="chapters/01_01-27-13-25-40",
            figures=figures,
            tables=assets_manifest.tables,
        )
        tex = declaration_block(mangled) + "\n" + good_chapter_tex(mangled)
        report = check_chapter_grounding(tex, analysis)
        for fragment in ("27", "13", "25", "40"):
            assert fragment not in report.ungrounded
