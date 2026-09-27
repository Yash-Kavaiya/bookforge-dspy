"""FrameCaptioner — grounded captions for curated video frames.

This module exists because of a silent bug in the ADK pipeline.

``ChapterAnalysis`` carried a ``frame_captions`` field keyed by frame filename,
and the analyst prompt asked for "optional caption overrides keyed by frame
filename". But the pipeline ran ``analyst -> assets``, and the filenames
(``frame_03_145s.jpg``) were invented by the *asset* stage — after the analyst
had already answered. The analyst had never seen a filename, so
``overrides.get(frame.file)`` never matched anything, and every figure in every
book fell through to the fallback::

    "Video frame at 02:25 illustrating the segment discussed in this chapter."

A textbook whose every figure carries the same non-caption. The feature looked
implemented, was fully wired, and did nothing.

The fix is ordering plus context: captioning happens *after* curation, when the
filenames exist, and the model is given the transcript spoken around each
frame's timestamp so the caption describes what was actually on screen. Frames
with nothing worth printing are dropped rather than padded with filler.
Optional Gemini visual briefings (when enabled) further ground what was on
screen between spoken lines.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import dspy

from bookforge.config import Settings, get_settings
from bookforge.dspy_runtime.chunking import window_around
from bookforge.dspy_runtime.signatures import CaptionFrames, FrameCaption
from bookforge.schemas import ChapterAnalysis, FrameAsset, TranscriptSegment
from bookforge.tools.youtube import format_timestamp

logger = logging.getLogger(__name__)

#: Cap how many frames go into one captioning call, to bound prompt size.
MAX_FRAMES_PER_CALL = 12


class FrameCaptioner(dspy.Module):
    """Caption curated frames from what the speaker was saying at that moment."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.caption = dspy.ChainOfThought(CaptionFrames)

    @staticmethod
    def _render_context(
        frames: Sequence[FrameAsset],
        segments: Sequence[TranscriptSegment],
        visual_insights: str = "",
    ) -> str:
        blocks: list[str] = []
        if visual_insights.strip():
            blocks.append(
                "VISUAL BRIEFING (from video understanding; use to name "
                "diagrams/equations on screen, do not invent unseen details):\n"
                + visual_insights.strip()[:5000]
            )
        for frame in frames:
            spoken = window_around(list(segments), frame.timestamp_sec)
            blocks.append(
                f"FRAME {frame.file} @ {format_timestamp(frame.timestamp_sec)}\n"
                f"Spoken around this moment: {spoken or '(no transcript nearby)'}"
            )
        return "\n\n".join(blocks)

    def forward(
        self,
        video_title: str,
        analysis: ChapterAnalysis,
        frames: Sequence[FrameAsset],
        segments: Sequence[TranscriptSegment],
        visual_insights: str = "",
    ) -> dspy.Prediction:
        """Return one caption per frame, with useless frames marked for removal."""
        frames = list(frames)
        if not frames:
            return dspy.Prediction(captions=[], dropped=[])

        headings = [concept.heading for concept in analysis.concepts]
        by_file: dict[str, FrameCaption] = {}

        for start in range(0, len(frames), MAX_FRAMES_PER_CALL):
            batch = frames[start : start + MAX_FRAMES_PER_CALL]
            try:
                prediction = self.caption(
                    video_title=video_title,
                    concept_headings=headings,
                    frames_with_context=self._render_context(
                        batch, segments, visual_insights=visual_insights
                    ),
                )
                for item in prediction.captions.captions:
                    by_file[item.frame_file] = item
            except Exception as exc:
                # Captioning is an enhancement; its failure must not lose the
                # figures. The fallback below still produces a usable book.
                logger.warning("frame captioning failed for a batch: %s", exc)

        captions: list[FrameCaption] = []
        dropped: list[str] = []
        for frame in frames:
            item = by_file.get(frame.file)
            if item is None:
                captions.append(
                    FrameCaption(
                        frame_file=frame.file,
                        caption=(
                            f"Slide shown at {format_timestamp(frame.timestamp_sec)} "
                            f"during the discussion of {video_title}."
                        ),
                        concept_heading="",
                        is_useful=True,
                    )
                )
                continue
            if not item.is_useful or not item.caption.strip():
                dropped.append(frame.file)
                continue
            captions.append(item)

        if dropped:
            logger.info(
                "dropped %d uninformative frame(s): %s", len(dropped), ", ".join(dropped)
            )
        return dspy.Prediction(captions=captions, dropped=dropped)
