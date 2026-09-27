"""ChapterProgram — the write/verify/refine loop, made monotonic.

The bug this replaces
---------------------
The ADK pipeline ran ``LoopAgent(sub_agents=[critic, refiner], max_iterations=3)``.
The critic could approve and escalate out of the loop; otherwise the refiner
rewrote the chapter and the loop went round again. When the iteration budget ran
out, the loop simply stopped — and it stopped *on the refiner*. That final
rewrite was written straight to ``chapter.tex`` and typeset into the book
without ever being compiled, asset-checked or read by anything.

So the pipeline's worst case was not "we ran out of attempts and shipped our
best effort". It was "we ran out of attempts and shipped the one draft nobody
looked at" — which could be worse than the first draft, and might not compile.
Because the critic's verdict lived only in the model's prose, nothing could
compare iteration 3 against iteration 1 to notice.

The fix
-------
Every candidate — first draft and every revision — goes through the same
deterministic verifier and gets a comparable score. The loop keeps the best
candidate it has ever seen and returns that. Formally: the output score is
``max`` over all candidates, so refinement can never make the shipped chapter
worse than the draft it started from. Refinement stops early when the score
clears the acceptance bar, when the refiner stops changing anything, or when
two consecutive attempts fail to improve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import dspy

from bookforge.config import Settings, get_settings
from bookforge.design.layout import declaration_block
from bookforge.dspy_runtime.modules.critic import SemanticCritic
from bookforge.dspy_runtime.modules.writer import ChapterRefiner, ChapterWriter
from bookforge.metrics.verifier import (
    VerificationContext,
    merge_semantic_defects,
    verify_chapter,
)
from bookforge.schemas import (
    AssetsManifest,
    ChapterAnalysis,
    ChapterCandidate,
    ChapterVerdict,
)
from bookforge.tools.latex import strip_asset_declarations
from bookforge.workspace import Workspace

logger = logging.getLogger(__name__)


@dataclass
class ChapterAttemptLog:
    """Per-iteration record, persisted next to the chapter for auditability."""

    iteration: int
    score: float
    compiles: bool
    critical_defects: int
    total_defects: int
    fingerprint: str
    note: str = ""


@dataclass
class ChapterResult:
    """The winning candidate plus the full history of how it was reached."""

    tex: str
    verdict: ChapterVerdict
    iterations: int
    history: list[ChapterAttemptLog] = field(default_factory=list)
    accepted: bool = False

    @property
    def score(self) -> float:
        return self.verdict.score

    def summary(self) -> str:
        trail = " -> ".join(f"{entry.score:.2f}" for entry in self.history)
        status = "accepted" if self.accepted else "best-effort"
        return (
            f"{status} after {self.iterations} iteration(s), "
            f"score {self.verdict.score:.2f} "
            f"(compiles={self.verdict.compiles}) [{trail}]"
        )


class ChapterProgram(dspy.Module):
    """Draft, verify, and refine one chapter — returning the best verified draft.

    Predictors exposed for optimization: ``writer.write``, ``refiner.refine``,
    ``critic.critique``.
    """

    def __init__(self, settings: Settings | None = None, use_semantic_critic: bool = True) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.writer = ChapterWriter(self.settings)
        self.refiner = ChapterRefiner(self.settings)
        self.critic = SemanticCritic(self.settings) if use_semantic_critic else None

    # -- verification -------------------------------------------------------

    def _verify(
        self,
        tex: str,
        video_title: str,
        context: VerificationContext,
        run_semantic: bool,
    ) -> ChapterVerdict:
        """Deterministic verdict, optionally enriched with semantic judgement.

        Asset declarations are BookForge-owned and only get written to disk
        after the loop (see pipeline.chapter). QA must still compile against
        them, otherwise every ``\\bfFigure{...}`` fails as "never declared"
        even when the label is in the manifest — which is exactly what was
        capping every chapter at ~0.60 with compiles=False.
        """
        body = strip_asset_declarations(tex)
        if context.manifest is not None:
            body = declaration_block(context.manifest) + "\n" + body
        verdict = verify_chapter(body, context)
        # Only spend a critic call on a draft that is mechanically sound. A
        # chapter that does not compile already has its instructions; asking a
        # model to also dislike its prose adds cost and no signal.
        if run_semantic and self.critic is not None and verdict.compiles:
            semantic = self.critic(
                video_title=video_title,
                analysis=context.analysis,
                chapter_tex=tex,
                verdict=verdict,
            ).defects
            verdict = merge_semantic_defects(verdict, semantic)
        return verdict

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        video_title: str,
        chapter_number: int,
        chapter_label_slug: str,
        analysis: ChapterAnalysis,
        manifest: AssetsManifest,
        workspace: Workspace | None = None,
        build_dir: Path | None = None,
    ) -> dspy.Prediction:
        settings = self.settings
        context = VerificationContext(
            analysis=analysis,
            manifest=manifest,
            workspace=workspace,
            build_dir=build_dir,
            settings=settings,
        )

        history: list[ChapterAttemptLog] = []
        seen_fingerprints: set[str] = set()

        # --- first draft ---------------------------------------------------
        draft = self.writer(
            video_title=video_title,
            chapter_number=chapter_number,
            chapter_label_slug=chapter_label_slug,
            analysis=analysis,
            manifest=manifest,
        ).chapter_tex

        verdict = self._verify(draft, video_title, context, run_semantic=True)
        best = ChapterCandidate(tex=draft, iteration=0, verdict=verdict)
        seen_fingerprints.add(best.fingerprint)
        history.append(_log_entry(best, "initial draft"))
        logger.info(
            "chapter %02d '%s' draft scored %.2f (compiles=%s)",
            chapter_number, video_title, verdict.score, verdict.compiles,
        )

        if verdict.score >= settings.qa_accept_score and verdict.is_acceptable:
            return _result(best, history, accepted=True)

        # --- bounded refinement --------------------------------------------
        stalled = 0
        current = best

        for iteration in range(1, settings.qa_max_iterations + 1):
            critique = current.verdict.feedback_text()
            revision = self.refiner(
                video_title=video_title,
                chapter_number=chapter_number,
                chapter_label_slug=chapter_label_slug,
                analysis=analysis,
                manifest=manifest,
                current_chapter_tex=current.tex,
                critique=critique,
            )
            revised_tex = revision.chapter_tex

            candidate = ChapterCandidate(
                tex=revised_tex,
                iteration=iteration,
                verdict=ChapterVerdict(),
            )
            if candidate.fingerprint in seen_fingerprints:
                # The refiner reproduced a draft we have already scored, so
                # another verification would cost a compile for no information.
                history.append(
                    ChapterAttemptLog(
                        iteration=iteration,
                        score=current.verdict.score,
                        compiles=current.verdict.compiles,
                        critical_defects=len(current.verdict.critical_defects),
                        total_defects=len(current.verdict.defects),
                        fingerprint=candidate.fingerprint,
                        note="refiner produced no change; stopping",
                    )
                )
                logger.info(
                    "chapter %02d: refiner returned an unchanged draft, stopping",
                    chapter_number,
                )
                break
            seen_fingerprints.add(candidate.fingerprint)

            # Every candidate is verified — including the last one. This is the
            # invariant the ADK loop lacked.
            candidate.verdict = self._verify(
                revised_tex, video_title, context, run_semantic=True
            )
            improved = candidate.verdict.score > best.verdict.score
            history.append(
                _log_entry(
                    candidate,
                    "improved" if improved else "no improvement (best retained)",
                )
            )
            logger.info(
                "chapter %02d iteration %d scored %.2f (best %.2f, compiles=%s)",
                chapter_number, iteration, candidate.verdict.score,
                best.verdict.score, candidate.verdict.compiles,
            )

            if improved:
                best = candidate
                stalled = 0
            else:
                stalled += 1

            current = candidate

            if best.verdict.score >= settings.qa_accept_score and best.verdict.is_acceptable:
                return _result(best, history, accepted=True)

            if stalled >= 2:
                # Two consecutive non-improvements: more attempts are unlikely
                # to help and each one costs a full generate-and-compile cycle.
                logger.info(
                    "chapter %02d: refinement stalled after %d iterations",
                    chapter_number, iteration,
                )
                break

        accepted = (
            best.verdict.is_acceptable
            and best.verdict.score >= settings.qa_min_acceptable_score
        )
        return _result(best, history, accepted=accepted)


def _log_entry(candidate: ChapterCandidate, note: str) -> ChapterAttemptLog:
    return ChapterAttemptLog(
        iteration=candidate.iteration,
        score=candidate.verdict.score,
        compiles=candidate.verdict.compiles,
        critical_defects=len(candidate.verdict.critical_defects),
        total_defects=len(candidate.verdict.defects),
        fingerprint=candidate.fingerprint,
        note=note,
    )


def _result(
    best: ChapterCandidate, history: list[ChapterAttemptLog], accepted: bool
) -> dspy.Prediction:
    result = ChapterResult(
        tex=best.tex,
        verdict=best.verdict,
        iterations=len(history),
        history=history,
        accepted=accepted,
    )
    return dspy.Prediction(
        chapter_tex=best.tex,
        verdict=best.verdict,
        result=result,
        score=best.verdict.score,
        accepted=accepted,
    )
