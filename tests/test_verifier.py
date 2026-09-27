"""Structure checking and the composite verifier.

These are the deterministic checks that the ADK critic performed by *reading* —
moving them into code is what makes the chapter score reproducible, and a
reproducible score is what makes the refinement loop and the optimizers work.
"""

from __future__ import annotations

from bookforge.metrics.structure import REQUIRED_SECTIONS, check_structure
from bookforge.metrics.verifier import (
    VerificationContext,
    merge_semantic_defects,
    verify_chapter,
)
from bookforge.schemas import Defect
from tests.conftest import good_chapter_tex


def _context(analysis, manifest, settings=None) -> VerificationContext:
    return VerificationContext(analysis=analysis, manifest=manifest, settings=settings)


class TestStructure:
    def test_good_chapter_passes(self, analysis, assets_manifest):
        report = check_structure(
            good_chapter_tex(assets_manifest), analysis, assets_manifest
        )
        assert report.has_chapter_command
        assert report.missing_sections == []
        assert report.invented_assets == []
        assert report.unused_assets == []
        assert report.missing_concepts == []
        assert report.score > 0.9

    def test_undeclared_label_is_critical(self, analysis, assets_manifest):
        """Placing a label that was never declared is a hard LaTeX error."""
        tex = good_chapter_tex(assets_manifest) + "\n\\bfFigure{fig:imaginary}\n"
        report = check_structure(tex, analysis, assets_manifest)
        assert "fig:imaginary" in report.invented_assets
        assert any(d.severity == "critical" for d in report.defects())

    def test_raw_path_reference_is_flagged(self, analysis, assets_manifest):
        """Hand-rolling \\includegraphics bypasses width, caption and placement."""
        tex = good_chapter_tex(assets_manifest) + (
            "\n\\includegraphics[width=3cm]{chapters/01_x/figures/sneaky.pdf}\n"
        )
        report = check_structure(tex, analysis, assets_manifest)
        assert any("sneaky.pdf" in path for path in report.raw_asset_references)
        assert any("raw path" in d.message for d in report.defects())

    def test_missing_section_is_major(self, analysis, assets_manifest):
        tex = good_chapter_tex(assets_manifest).replace(r"\bfsection{Glossary}", "")
        report = check_structure(tex, analysis, assets_manifest)
        assert "Glossary" in report.missing_sections
        assert any(d.severity == "major" for d in report.defects())

    def test_all_required_sections_are_checked(self, analysis, assets_manifest):
        report = check_structure("\\chapter{X}", analysis, assets_manifest)
        assert set(report.missing_sections) == set(REQUIRED_SECTIONS)

    def test_dropped_concept_is_detected(self, analysis, assets_manifest):
        tex = good_chapter_tex(assets_manifest).replace(
            r"\section{Overlay Networking Costs}", ""
        )
        report = check_structure(tex, analysis, assets_manifest)
        assert "Overlay Networking Costs" in report.missing_concepts

    def test_unplaced_asset_is_flagged(self, analysis, assets_manifest):
        """An asset rendered and then never placed is wasted work."""
        label = assets_manifest.tables[0].label
        tex = good_chapter_tex(assets_manifest).replace(f"\\bfTable{{{label}}}", "")
        report = check_structure(tex, analysis, assets_manifest)
        assert label in report.unused_assets

    def test_dangling_ref_is_detected(self, analysis, assets_manifest):
        tex = good_chapter_tex(assets_manifest) + "\nSee Figure~\\ref{fig:nonexistent}.\n"
        report = check_structure(tex, analysis, assets_manifest)
        assert "fig:nonexistent" in report.dangling_refs

    def test_duplicate_label_is_detected(self, analysis, assets_manifest):
        tex = good_chapter_tex(assets_manifest) + (
            "\n\\label{ch:01-kubernetes}\n"
        )
        report = check_structure(tex, analysis, assets_manifest)
        assert "ch:01-kubernetes" in report.duplicate_labels

    def test_missing_diagram_is_detected(self, analysis, assets_manifest):
        """A diagram spec the chapter never renders is a major omission."""
        tex = good_chapter_tex(assets_manifest)
        start = tex.index(r"\bfDiagram")
        end = tex.index(r"\bfsection{Key Takeaways}")
        tex = tex[:start] + tex[end:]
        report = check_structure(tex, analysis, assets_manifest)
        assert report.missing_diagrams == ["Request path"]

    def test_short_chapter_is_flagged(self, analysis, assets_manifest):
        report = check_structure(r"\chapter{X} tiny", analysis, assets_manifest)
        assert report.word_count < 700
        assert any("words" in d.message for d in report.defects())

    def test_missing_file_on_disk_detected(self, analysis, assets_manifest):
        report = check_structure(
            good_chapter_tex(assets_manifest),
            analysis,
            assets_manifest,
            existing_paths=set(),  # nothing exists
        )
        assert len(report.missing_assets) == 3


class TestVerifier:
    def test_empty_chapter_scores_zero(self, analysis, assets_manifest):
        verdict = verify_chapter("", _context(analysis, assets_manifest))
        assert verdict.score == 0.0
        assert not verdict.is_acceptable
        assert verdict.critical_defects

    def test_good_chapter_scores_well(self, analysis, assets_manifest, settings):
        verdict = verify_chapter(
            good_chapter_tex(assets_manifest),
            _context(analysis, assets_manifest, settings),
        )
        # compile_latex is off in tests, so the ceiling is below 1.0 by design:
        # we never award full marks for verification we did not perform.
        assert verdict.score > 0.5
        assert verdict.invented_assets == []
        assert verdict.ungrounded_numbers == []

    def test_invented_number_produces_critical_defect(
        self, analysis, assets_manifest, settings
    ):
        tex = good_chapter_tex(assets_manifest) + "\nThroughput reached 55555 Mbps.\n"
        verdict = verify_chapter(tex, _context(analysis, assets_manifest, settings))
        assert "55555" in verdict.ungrounded_numbers
        assert any("55555" in d.message for d in verdict.critical_defects)

    def test_verdict_is_deterministic(self, analysis, assets_manifest, settings):
        """Comparable scores across iterations is the loop's whole premise."""
        tex = good_chapter_tex(assets_manifest)
        context = _context(analysis, assets_manifest, settings)
        scores = {verify_chapter(tex, context).score for _ in range(5)}
        assert len(scores) == 1

    def test_worse_chapter_scores_lower(self, analysis, assets_manifest, settings):
        context = _context(analysis, assets_manifest, settings)
        good = verify_chapter(good_chapter_tex(assets_manifest), context).score
        bad = verify_chapter(r"\chapter{X} not much here", context).score
        assert bad < good

    def test_feedback_text_is_actionable(self, analysis, assets_manifest, settings):
        verdict = verify_chapter(r"\chapter{X}", _context(analysis, assets_manifest, settings))
        feedback = verdict.feedback_text()
        assert "CRITICAL" in feedback or "MAJOR" in feedback
        assert "FIX:" in feedback

    def test_defects_are_ordered_by_severity(self, analysis, assets_manifest, settings):
        tex = good_chapter_tex(assets_manifest) + "\n\\bfFigure{fig:fake}\n"
        verdict = verify_chapter(tex, _context(analysis, assets_manifest, settings))
        assert verdict.feedback_text().startswith("1. [CRITICAL]")


class TestFalsePositiveRegressions:
    """Three defects found only by running the real pipeline end to end.

    Each one made a perfectly good chapter fail. A verifier that invents
    defects is worse than a lax one: the loop burns its budget chasing them,
    and the optimizer learns to avoid behaviour that was correct.
    """

    def test_manifest_caption_numbers_are_not_inventions(
        self, analysis, assets_manifest, settings
    ):
        """Frame captions carry timestamps that BookForge itself generated.

        The writer is required to copy the manifest caption verbatim, so
        flagging its numbers accused the writer of inventing what we handed it.
        """
        assets_manifest.figures[1].caption = (
            "The speaker's slide at 137 seconds, showing the flat pod network model."
        )
        tex = good_chapter_tex(assets_manifest)
        verdict = verify_chapter(tex, _context(analysis, assets_manifest, settings))
        assert "137" not in verdict.ungrounded_numbers

    def test_table_fragment_labels_count_as_defined(
        self, analysis, assets_manifest, settings
    ):
        """The label lives inside the \\input'ed fragment, not the chapter body."""
        report = check_structure(
            good_chapter_tex(assets_manifest), analysis, assets_manifest
        )
        assert assets_manifest.tables[0].label not in report.dangling_refs

    def test_single_pass_compile_reports_no_undefined_refs(self, tmp_path):
        """LaTeX resolves \\ref via the .aux file on the *next* pass.

        Reporting undefined references after one pass is a guaranteed false
        positive on every chapter that cross-references anything — which is
        every chapter, since the contract requires floats to be referenced.
        """
        import pytest as _pytest

        from bookforge.tools import latex as lx

        if not lx.pdflatex_available():
            _pytest.skip("pdflatex not installed")

        body = (
            r"\chapter{X}\label{ch:x}" "\n"
            r"See Figure~\ref{fig:later} below." "\n"
            r"\begin{figure}[htbp]\centering\rule{2cm}{1cm}"
            r"\caption{c}\label{fig:later}\end{figure}"
        )
        source = lx.chapter_wrapper(lx.PREAMBLE, body)
        (tmp_path / "one.tex").write_text(source, encoding="utf-8")
        (tmp_path / "two.tex").write_text(source, encoding="utf-8")

        one = lx.compile_tex(tmp_path / "one.tex", tmp_path / "b1", passes=1, cwd=tmp_path)
        assert one.ok
        assert one.undefined_references == [], (
            "a forward reference is undefined on pass 1 by definition; "
            "reporting it is a false defect"
        )

        two = lx.compile_tex(tmp_path / "two.tex", tmp_path / "b2", passes=2, cwd=tmp_path)
        assert two.ok and two.undefined_references == []


class TestSemanticMerge:
    def test_semantic_defects_reduce_the_score(self, analysis, assets_manifest, settings):
        verdict = verify_chapter(
            good_chapter_tex(assets_manifest),
            _context(analysis, assets_manifest, settings),
        )
        before = verdict.score
        merged = merge_semantic_defects(
            verdict, [Defect(severity="major", message="Prose is circular.", fix="Rewrite.")]
        )
        assert merged.score < before
        assert len(merged.defects) >= 1

    def test_empty_semantic_list_is_a_noop(self, analysis, assets_manifest, settings):
        verdict = verify_chapter(
            good_chapter_tex(assets_manifest),
            _context(analysis, assets_manifest, settings),
        )
        before = verdict.score
        assert merge_semantic_defects(verdict, []).score == before

    def test_penalty_is_capped(self, analysis, assets_manifest, settings):
        verdict = verify_chapter(
            good_chapter_tex(assets_manifest),
            _context(analysis, assets_manifest, settings),
        )
        before = verdict.score
        merged = merge_semantic_defects(
            verdict,
            [Defect(severity="critical", message=f"defect {i}") for i in range(20)],
        )
        # The cap is exactly 0.35; compare with a tolerance so the assertion is
        # about the cap, not about float representation.
        assert merged.score >= before - 0.35 - 1e-9

    def test_critical_semantic_defect_blocks_acceptance(
        self, analysis, assets_manifest, settings
    ):
        verdict = verify_chapter(
            good_chapter_tex(assets_manifest),
            _context(analysis, assets_manifest, settings),
        )
        merged = merge_semantic_defects(
            verdict, [Defect(severity="critical", message="States a fact the analysis lacks.")]
        )
        assert not merged.is_acceptable
