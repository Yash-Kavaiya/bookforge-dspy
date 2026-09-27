"""Transcript windowing for map-reduce analysis.

The ADK pipeline injected the whole transcript into one prompt and, when that
exceeded ``max_chars``, kept the first and last halves with a
``...[middle truncated]...`` marker between them. For a three-hour lecture that
threw away the entire middle of the talk while the analyst was still instructed
to "be faithful" — so the model faithfully described a video it had only seen
the ends of, and nothing in the pipeline could tell.

Chunking here splits on segment boundaries with overlap, so every spoken word
reaches the analyst exactly once (plus a little context on each side) and the
merge step reconciles the windows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from bookforge.schemas import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptChunk:
    """A contiguous window of transcript, aligned to segment boundaries."""

    index: int  # 1-based
    total: int
    segments: list[TranscriptSegment]

    @property
    def start_sec(self) -> float:
        return self.segments[0].start_sec if self.segments else 0.0

    @property
    def end_sec(self) -> float:
        return self.segments[-1].start_sec if self.segments else 0.0

    @property
    def text(self) -> str:
        return "\n".join(f"[{seg.timestamp}] {seg.text}" for seg in self.segments)

    def __len__(self) -> int:
        return len(self.text)


def chunk_segments(
    segments: list[TranscriptSegment],
    chunk_chars: int = 24_000,
    overlap_chars: int = 1_200,
    max_chunks: int = 0,
) -> list[TranscriptChunk]:
    """Split segments into overlapping windows of roughly ``chunk_chars``.

    Splits only at segment boundaries, so a sentence is never cut in half.
    Overlap carries the tail of each window into the next, which stops a concept
    explained across a boundary from being lost by both.

    ``max_chunks`` caps cost. When the transcript would exceed it, windows are
    widened to cover the whole talk rather than dropping the tail — full
    coverage at lower resolution beats a silent gap.
    """
    if not segments:
        return []
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars")

    def _line_len(segment: TranscriptSegment) -> int:
        return len(segment.timestamp) + len(segment.text) + 4

    total_chars = sum(_line_len(seg) for seg in segments)
    effective_chunk = chunk_chars
    if max_chunks and total_chars > 0:
        stride = max(1, chunk_chars - overlap_chars)
        projected = max(1, -(-total_chars // stride))  # ceil
        if projected > max_chunks:
            effective_chunk = max(
                chunk_chars, -(-total_chars // max_chunks) + overlap_chars
            )
            logger.info(
                "transcript needs %d windows at %d chars; widening to %d chars "
                "to stay within the %d-window budget",
                projected, chunk_chars, effective_chunk, max_chunks,
            )

    windows: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    current_len = 0

    for segment in segments:
        length = _line_len(segment)
        if current and current_len + length > effective_chunk:
            windows.append(current)
            # Carry the tail forward as overlap context.
            carry: list[TranscriptSegment] = []
            carry_len = 0
            for prev in reversed(current):
                prev_len = _line_len(prev)
                if carry_len + prev_len > overlap_chars:
                    break
                carry.insert(0, prev)
                carry_len += prev_len
            current = carry
            current_len = carry_len
        current.append(segment)
        current_len += length

    if current:
        windows.append(current)

    total = len(windows)
    return [
        TranscriptChunk(index=i, total=total, segments=window)
        for i, window in enumerate(windows, start=1)
    ]


def window_around(
    segments: list[TranscriptSegment],
    timestamp_sec: float,
    before_sec: float = 25.0,
    after_sec: float = 25.0,
    max_chars: int = 900,
) -> str:
    """Transcript text spoken around a moment — context for frame captioning.

    This is what makes a figure caption say what the slide actually shows
    instead of "Video frame at 03:15".
    """
    lower = timestamp_sec - before_sec
    upper = timestamp_sec + after_sec
    picked = [seg for seg in segments if lower <= seg.start_sec <= upper]

    if not picked and segments:
        nearest = min(segments, key=lambda s: abs(s.start_sec - timestamp_sec))
        picked = [nearest]

    text = " ".join(seg.text for seg in picked).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
    return text
