"""TranscriptAnalyst — transcript to validated ChapterAnalysis.

Replaces the ADK ``TranscriptAnalystAgent``, and fixes the two accuracy holes it
had:

1. **Coverage.** The whole transcript went into one prompt; anything over the
   character cap had its middle deleted. Long talks were analysed from their
   ends. Here the transcript is windowed on segment boundaries and every window
   is analysed, then merged.

2. **Verification.** Nothing checked whether the extracted numbers were ever
   spoken. Here each candidate analysis is scored by ``check_grounding`` and a
   result below threshold is resampled at a higher temperature before being
   accepted, so a hallucinated benchmark figure gets a second chance to be
   caught rather than being rendered into a chart.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import dspy

from bookforge.config import Settings, get_settings
from bookforge.dspy_runtime.chunking import TranscriptChunk, chunk_segments
from bookforge.dspy_runtime.signatures import (
    AnalyzeTranscriptChunk,
    ChunkAnalysis,
    MergeChunkAnalyses,
    SingleShotAnalysis,
)
from bookforge.metrics.grounding import check_grounding
from bookforge.schemas import ChapterAnalysis, TranscriptSegment

logger = logging.getLogger(__name__)


def _prompt_transcript(spoken: str, visual_insights: str) -> str:
    """LM-facing transcript: optional visual briefing, never used for grounding."""
    briefing = (visual_insights or "").strip()
    if not briefing:
        return spoken
    return (
        "[Visual screen briefing — diagrams, slides, and on-screen text. "
        "Use for structure and figure placement only. Do NOT treat numbers "
        "here as spoken evidence unless the same figure also appears in the "
        "spoken transcript below.]\n"
        f"{briefing[:6000]}\n\n"
        "[Spoken transcript — sole source of numeric claims]\n"
        f"{spoken}"
    )


class TranscriptAnalyst(dspy.Module):
    """Windowed, grounded extraction of a chapter analysis from a transcript.

    Predictors (each independently optimizable):
        ``analyze_chunk``  one transcript window -> ChunkAnalysis
        ``merge``          many ChunkAnalysis -> ChapterAnalysis
        ``single_shot``    short transcript -> ChapterAnalysis (no merge needed)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.analyze_chunk = dspy.ChainOfThought(AnalyzeTranscriptChunk)
        self.merge = dspy.ChainOfThought(MergeChunkAnalyses)
        self.single_shot = dspy.ChainOfThought(SingleShotAnalysis)

    # -- helpers ------------------------------------------------------------

    def _chunks(self, segments: Sequence[TranscriptSegment]) -> list[TranscriptChunk]:
        return chunk_segments(
            list(segments),
            chunk_chars=self.settings.transcript_chunk_chars,
            overlap_chars=self.settings.transcript_chunk_overlap_chars,
            max_chunks=self.settings.max_transcript_chunks,
        )

    def _analyze_one_chunk(
        self,
        video_title: str,
        chunk: TranscriptChunk,
        visual_insights: str = "",
    ) -> ChunkAnalysis | None:
        try:
            prediction = self.analyze_chunk(
                video_title=video_title,
                chunk_index=chunk.index,
                chunk_total=chunk.total,
                transcript_chunk=_prompt_transcript(chunk.text, visual_insights),
            )
            return prediction.analysis
        except Exception as exc:
            # One unreadable window must not cost us the other 11.
            logger.warning(
                "analysis failed for window %d/%d: %s", chunk.index, chunk.total, exc
            )
            return None

    def _merge_chunks(
        self, video_title: str, analyses: list[ChunkAnalysis]
    ) -> ChapterAnalysis:
        prediction = self.merge(video_title=video_title, chunk_analyses=analyses)
        return prediction.analysis

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        video_title: str,
        transcript: str = "",
        segments: Sequence[TranscriptSegment] | None = None,
        visual_insights: str = "",
    ) -> dspy.Prediction:
        """Produce a grounded ChapterAnalysis.

        Either ``segments`` (preferred, enables windowing) or ``transcript``
        must be given.

        ``visual_insights`` may be shown to the model as non-spoken screen
        context, but grounding is scored only against the spoken transcript so
        invented on-screen numbers cannot launder themselves into evidence.
        """
        if segments is None and not transcript.strip():
            raise ValueError("TranscriptAnalyst needs either segments or a transcript")

        if segments is None:
            # No timestamps available; treat the text as one window.
            analysis = self.single_shot(
                video_title=video_title,
                transcript=_prompt_transcript(transcript, visual_insights),
            ).analysis
            return self._finalize(analysis, transcript, windows=1)

        chunks = self._chunks(segments)
        spoken_text = "\n".join(chunk.text for chunk in chunks)

        if len(chunks) <= 1:
            text = chunks[0].text if chunks else transcript
            analysis = self.single_shot(
                video_title=video_title,
                transcript=_prompt_transcript(text, visual_insights),
            ).analysis
            return self._finalize(analysis, text, windows=1)

        logger.info(
            "analysing '%s' across %d transcript windows", video_title, len(chunks)
        )
        chunk_analyses: list[ChunkAnalysis] = []
        for chunk in chunks:
            # Attach the visual briefing only to the first window so later
            # windows stay focused on spoken coverage.
            briefing = visual_insights if chunk.index == 1 else ""
            result = self._analyze_one_chunk(
                video_title,
                chunk,
                visual_insights=briefing,
            )
            if result is not None:
                chunk_analyses.append(result)

        if not chunk_analyses:
            raise RuntimeError(
                f"every transcript window failed to analyse for '{video_title}'"
            )
        if len(chunk_analyses) < len(chunks):
            logger.warning(
                "only %d/%d windows analysed for '%s'; the chapter will be thinner "
                "than the talk",
                len(chunk_analyses), len(chunks), video_title,
            )

        analysis = self._merge_chunks(video_title, chunk_analyses)
        return self._finalize(analysis, spoken_text, windows=len(chunks))

    def _finalize(
        self, analysis: ChapterAnalysis, transcript: str, windows: int
    ) -> dspy.Prediction:
        report = check_grounding(analysis, transcript)
        if report.ungrounded:
            logger.warning(
                "%d ungrounded number(s) in analysis of '%s': %s",
                len(report.ungrounded),
                analysis.chapter_title,
                ", ".join(report.ungrounded[:6]),
            )
        return dspy.Prediction(
            analysis=analysis,
            grounding_score=report.score,
            grounding_report=report,
            windows=windows,
        )
