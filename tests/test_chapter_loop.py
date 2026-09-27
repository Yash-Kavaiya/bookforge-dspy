"""The refinement loop — regression tests for the ADK bug this replaces.

The original ``LoopAgent(sub_agents=[critic, refiner], max_iterations=3)``
stopped *on the refiner* when its budget ran out, and that final rewrite went
straight into the book without ever being compiled or checked. The chapter that
shipped could therefore be worse than the first draft, and could fail to build.

The property asserted here is monotonicity: the returned chapter's score is the
maximum over every candidate the loop produced. If that ever breaks, the loop
has regressed to the old behaviour.
"""

from __future__ import annotations

import dspy
import pytest

from bookforge.dspy_runtime.modules.chapter import ChapterProgram
from tests.conftest import good_chapter_tex


class ScriptedWriter(dspy.Module):
    """Returns pre-written drafts in order, recording what it was asked."""

    def __init__(self, drafts: list[str]) -> None:
        super().__init__()
        self.drafts = list(drafts)
        self.calls = 0

    def forward(self, **kwargs) -> dspy.Prediction:
        index = min(self.calls, len(self.drafts) - 1)
        self.calls += 1
        return dspy.Prediction(chapter_tex=self.drafts[index])


class ScriptedRefiner(dspy.Module):
    """Returns pre-written revisions in order, recording the critiques it saw."""

    def __init__(self, revisions: list[str]) -> None:
        super().__init__()
        self.revisions = list(revisions)
        self.calls = 0
        self.critiques: list[str] = []

    def forward(self, current_chapter_tex: str = "", critique: str = "", **kwargs):
        self.critiques.append(critique)
        if self.calls >= len(self.revisions):
            self.calls += 1
            return dspy.Prediction(chapter_tex=current_chapter_tex, degenerate=True)
        revision = self.revisions[self.calls]
        self.calls += 1
        return dspy.Prediction(chapter_tex=revision, degenerate=False)


def _program(settings, drafts, revisions) -> ChapterProgram:
    program = ChapterProgram(settings, use_semantic_critic=False)
    program.writer = ScriptedWriter(drafts)
    program.refiner = ScriptedRefiner(revisions)
    return program


def _run(program, analysis, manifest):
    return program(
        video_title="Kubernetes Networking Deep Dive",
        chapter_number=1,
        chapter_label_slug="kubernetes",
        analysis=analysis,
        manifest=manifest,
    )


# A draft placing a label that was never declared: mechanically broken, and
# LaTeX would refuse it outright.
def _broken(manifest) -> str:
    return good_chapter_tex(manifest) + "\n\\bfFigure{fig:does-not-exist}\n"


def _stub(manifest) -> str:
    return r"\chapter{Stub}" + "\n" + "word " * 20


class TestMonotonicity:
    def test_returns_the_best_candidate_not_the_last(
        self, settings, analysis, assets_manifest
    ):
        """THE regression test. Good draft, then two worse revisions.

        The ADK loop would ship the final revision because it stopped there.
        This loop must ship the good draft.
        """
        good = good_chapter_tex(assets_manifest)
        program = _program(
            settings,
            drafts=[good],
            revisions=[_broken(assets_manifest), _stub(assets_manifest)],
        )
        result = _run(program, analysis, assets_manifest)

        assert result.chapter_tex == good, "the loop shipped a worse revision"
        assert result.score == max(entry.score for entry in result.result.history)

    def test_score_never_decreases_across_iterations(
        self, settings, analysis, assets_manifest
    ):
        program = _program(
            settings,
            drafts=[_broken(assets_manifest)],
            revisions=[_stub(assets_manifest), good_chapter_tex(assets_manifest)],
        )
        result = _run(program, analysis, assets_manifest)
        best_so_far = 0.0
        for entry in result.result.history:
            best_so_far = max(best_so_far, entry.score)
        assert result.score == best_so_far

    def test_improvement_is_adopted(self, settings, analysis, assets_manifest):
        good = good_chapter_tex(assets_manifest)
        program = _program(
            settings, drafts=[_stub(assets_manifest)], revisions=[good]
        )
        result = _run(program, analysis, assets_manifest)
        assert result.chapter_tex == good
        assert result.result.history[-1].score > result.result.history[0].score

    def test_every_candidate_is_verified(self, settings, analysis, assets_manifest):
        """No candidate may reach the book unscored — including the last one."""
        program = _program(
            settings,
            drafts=[_stub(assets_manifest)],
            revisions=[_broken(assets_manifest), good_chapter_tex(assets_manifest)],
        )
        result = _run(program, analysis, assets_manifest)
        assert all(entry.fingerprint for entry in result.result.history)
        assert all(entry.score >= 0.0 for entry in result.result.history)


class TestTermination:
    def test_high_scoring_draft_skips_refinement(
        self, settings, analysis, assets_manifest
    ):
        # Any mechanically sound draft is acceptable.
        settings.qa_min_acceptable_score = 0.1
        settings.qa_accept_score = 0.1
        program = _program(
            settings, drafts=[good_chapter_tex(assets_manifest)], revisions=[]
        )
        result = _run(program, analysis, assets_manifest)
        assert result.accepted
        assert program.refiner.calls == 0, "refined a chapter that was already good"

    def test_unchanged_revision_stops_the_loop(
        self, settings, analysis, assets_manifest
    ):
        """A refiner echoing its input signals it has nothing left to fix."""
        settings.qa_accept_score = 0.99
        stub = _stub(assets_manifest)
        program = _program(settings, drafts=[stub], revisions=[stub])
        result = _run(program, analysis, assets_manifest)
        assert any("no change" in entry.note for entry in result.result.history)
        assert program.refiner.calls == 1

    def test_iteration_budget_is_respected(self, settings, analysis, assets_manifest):
        settings.qa_accept_score = 0.999
        settings.qa_max_iterations = 2
        program = _program(
            settings,
            drafts=[_stub(assets_manifest)],
            revisions=[
                _stub(assets_manifest) + "\n% a",
                _stub(assets_manifest) + "\n% b",
                _stub(assets_manifest) + "\n% c",
            ],
        )
        _run(program, analysis, assets_manifest)
        assert program.refiner.calls <= 2

    def test_stalling_stops_early(self, settings, analysis, assets_manifest):
        """Two consecutive non-improvements: stop paying for more attempts."""
        settings.qa_accept_score = 0.999
        settings.qa_max_iterations = 6
        stub = _stub(assets_manifest)
        program = _program(
            settings,
            drafts=[good_chapter_tex(assets_manifest)],
            revisions=[stub + "\n% a", stub + "\n% b", stub + "\n% c", stub + "\n% d"],
        )
        _run(program, analysis, assets_manifest)
        assert program.refiner.calls <= 3, "kept refining after it stopped helping"


class TestFeedbackLoop:
    def test_refiner_receives_actionable_defects(
        self, settings, analysis, assets_manifest
    ):
        settings.qa_accept_score = 0.999
        program = _program(
            settings,
            drafts=[_broken(assets_manifest)],
            revisions=[good_chapter_tex(assets_manifest)],
        )
        _run(program, analysis, assets_manifest)
        critique = program.refiner.critiques[0]
        assert "fig:does-not-exist" in critique
        assert "CRITICAL" in critique
        assert "FIX:" in critique


class TestResult:
    def test_result_reports_its_history(self, settings, analysis, assets_manifest):
        program = _program(
            settings,
            drafts=[_stub(assets_manifest)],
            revisions=[good_chapter_tex(assets_manifest)],
        )
        result = _run(program, analysis, assets_manifest).result
        assert result.iterations == len(result.history)
        summary = result.summary()
        assert "->" in summary and "score" in summary

    def test_unacceptable_chapter_is_not_accepted(
        self, settings, analysis, assets_manifest
    ):
        settings.qa_accept_score = 0.99
        settings.qa_min_acceptable_score = 0.95
        program = _program(settings, drafts=[_stub(assets_manifest)], revisions=[])
        result = _run(program, analysis, assets_manifest)
        assert not result.accepted

    @pytest.mark.parametrize("draft", ["", "   ", "```\n```"])
    def test_empty_draft_scores_zero(self, settings, analysis, assets_manifest, draft):
        program = _program(settings, drafts=[draft], revisions=[])
        result = _run(program, analysis, assets_manifest)
        assert result.score == 0.0
        assert not result.accepted
