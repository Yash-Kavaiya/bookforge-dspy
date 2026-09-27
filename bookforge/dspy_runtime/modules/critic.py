"""SemanticCritic — the judgement half of chapter QA.

The ADK critic was one LLM asked to do four jobs: call a compile tool, diff
asset references against a manifest, check section presence, and check numeric
fidelity. Three of those are exact string work that a language model performs
probabilistically and a regex performs perfectly — and when the model skipped a
step, nothing noticed.

Here the deterministic work lives in ``metrics.verifier`` and always runs. This
module handles only what genuinely needs reading comprehension: silently
dropped content, prose that contradicts its own figures, explanations that
teach nothing, wrong exercise answers. It is told what the verifier already
found so it does not spend its attention re-deriving it.
"""

from __future__ import annotations

import logging

import dspy

from bookforge.config import Settings, get_settings
from bookforge.dspy_runtime.signatures import CritiqueChapter
from bookforge.schemas import ChapterAnalysis, ChapterVerdict, Defect

logger = logging.getLogger(__name__)

_VALID_SEVERITIES = {"critical", "major", "minor"}


class SemanticCritic(dspy.Module):
    """Find the defects a machine check cannot see."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.critique = dspy.ChainOfThought(CritiqueChapter)

    @staticmethod
    def _mechanical_summary(verdict: ChapterVerdict) -> str:
        lines = [
            f"Compiled: {'yes' if verdict.compiles else 'NO'}",
            f"Composite score: {verdict.score:.2f}",
        ]
        if verdict.invented_assets:
            lines.append(f"Invented asset paths: {', '.join(verdict.invented_assets)}")
        if verdict.missing_assets:
            lines.append(f"Missing files on disk: {', '.join(verdict.missing_assets)}")
        if verdict.missing_sections:
            lines.append(f"Missing sections: {', '.join(verdict.missing_sections)}")
        if verdict.ungrounded_numbers:
            lines.append(
                f"Numbers absent from the analysis: {', '.join(verdict.ungrounded_numbers[:10])}"
            )
        if verdict.compile_log and not verdict.compiles:
            lines.append(f"Compiler output:\n{verdict.compile_log[:800]}")
        return "\n".join(lines)

    def forward(
        self,
        video_title: str,
        analysis: ChapterAnalysis,
        chapter_tex: str,
        verdict: ChapterVerdict,
    ) -> dspy.Prediction:
        """Return semantic defects as validated ``Defect`` objects."""
        try:
            prediction = self.critique(
                video_title=video_title,
                analysis=analysis,
                chapter_tex=chapter_tex,
                mechanical_findings=self._mechanical_summary(verdict),
            )
            raw_defects = prediction.critique.defects
        except Exception as exc:
            # A failed critique means "no extra findings", never a failed
            # chapter — the deterministic verdict still stands on its own.
            logger.warning("semantic critique failed for '%s': %s", video_title, exc)
            return dspy.Prediction(defects=[], failed=True)

        defects: list[Defect] = []
        for item in raw_defects:
            severity = str(item.severity or "").strip().lower()
            if severity not in _VALID_SEVERITIES:
                severity = "minor"
            message = (item.message or "").strip()
            if not message:
                continue
            defects.append(
                Defect(severity=severity, message=message, fix=(item.fix or "").strip())
            )
        return dspy.Prediction(defects=defects, failed=False)
