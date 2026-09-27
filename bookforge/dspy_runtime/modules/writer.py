"""ChapterWriter and ChapterRefiner — LaTeX composition and repair."""

from __future__ import annotations

import logging

import dspy

from bookforge.config import Settings, get_settings
from bookforge.dspy_runtime.signatures import RefineChapter, WriteChapter
from bookforge.schemas import AssetsManifest, ChapterAnalysis
from bookforge.tools import latex as latex_tools

logger = logging.getLogger(__name__)


class ChapterWriter(dspy.Module):
    """Compose one LaTeX chapter body from a validated analysis and its assets."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.write = dspy.ChainOfThought(WriteChapter)

    def forward(
        self,
        video_title: str,
        chapter_number: int,
        chapter_label_slug: str,
        analysis: ChapterAnalysis,
        manifest: AssetsManifest,
    ) -> dspy.Prediction:
        prediction = self.write(
            video_title=video_title,
            chapter_number=chapter_number,
            chapter_label_slug=chapter_label_slug,
            analysis=analysis,
            assets_manifest=manifest.describe_for_prompt(),
        )
        # Sanitize before anything else sees it: the verifier scores what will
        # actually be written to disk, never a prettier intermediate form.
        return dspy.Prediction(
            chapter_tex=latex_tools.sanitize_chapter_tex(prediction.chapter_tex)
        )


class ChapterRefiner(dspy.Module):
    """Repair a chapter against a numbered defect list."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.refine = dspy.ChainOfThought(RefineChapter)

    def forward(
        self,
        video_title: str,
        chapter_number: int,
        chapter_label_slug: str,
        analysis: ChapterAnalysis,
        manifest: AssetsManifest,
        current_chapter_tex: str,
        critique: str,
    ) -> dspy.Prediction:
        prediction = self.refine(
            video_title=video_title,
            chapter_number=chapter_number,
            chapter_label_slug=chapter_label_slug,
            analysis=analysis,
            assets_manifest=manifest.describe_for_prompt(),
            current_chapter_tex=current_chapter_tex,
            critique=critique,
        )
        revised = latex_tools.sanitize_chapter_tex(prediction.chapter_tex)

        # A refiner that returns almost nothing has failed, not succeeded.
        # Falling back to the input keeps the loop's best-candidate logic honest
        # instead of scoring an empty string as a regression.
        if len(revised.strip()) < 0.35 * len(current_chapter_tex.strip() or "x"):
            logger.warning(
                "refiner returned %d chars against an input of %d — discarding the "
                "truncated revision",
                len(revised.strip()), len(current_chapter_tex.strip()),
            )
            return dspy.Prediction(chapter_tex=current_chapter_tex, degenerate=True)
        return dspy.Prediction(chapter_tex=revised, degenerate=False)
