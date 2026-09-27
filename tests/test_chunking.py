"""Transcript windowing — the fix for silent middle-of-video truncation."""

from __future__ import annotations

import pytest

from bookforge.dspy_runtime.chunking import chunk_segments, window_around
from bookforge.schemas import TranscriptSegment


def _long_transcript(n: int = 400) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            start_sec=i * 5.0,
            text=f"Segment {i} discusses topic number {i} in some technical detail.",
        )
        for i in range(n)
    ]


class TestChunking:
    def test_short_transcript_is_one_window(self, transcript_segments):
        chunks = chunk_segments(transcript_segments, chunk_chars=24_000)
        assert len(chunks) == 1
        assert chunks[0].index == 1 and chunks[0].total == 1

    def test_long_transcript_splits(self):
        chunks = chunk_segments(_long_transcript(), chunk_chars=2_000, overlap_chars=200)
        assert len(chunks) > 1
        assert [c.index for c in chunks] == list(range(1, len(chunks) + 1))
        assert all(c.total == len(chunks) for c in chunks)

    def test_every_segment_is_covered(self):
        """The regression this module exists for: no content may be dropped.

        The ADK loader kept the head and tail of a long transcript and deleted
        the middle. Here every segment must appear in at least one window.
        """
        segments = _long_transcript(400)
        chunks = chunk_segments(segments, chunk_chars=2_000, overlap_chars=200)
        covered = {seg.start_sec for chunk in chunks for seg in chunk.segments}
        assert covered == {seg.start_sec for seg in segments}

    def test_no_truncation_marker_is_ever_emitted(self):
        chunks = chunk_segments(_long_transcript(400), chunk_chars=2_000)
        assert not any("truncated" in chunk.text for chunk in chunks)

    def test_windows_overlap(self):
        chunks = chunk_segments(_long_transcript(200), chunk_chars=2_000, overlap_chars=400)
        first = {s.start_sec for s in chunks[0].segments}
        second = {s.start_sec for s in chunks[1].segments}
        assert first & second, "consecutive windows must share context"

    def test_max_chunks_widens_rather_than_dropping(self):
        """Hitting the budget must not silently lose the end of the talk."""
        segments = _long_transcript(600)
        chunks = chunk_segments(
            segments, chunk_chars=1_000, overlap_chars=100, max_chunks=4
        )
        assert len(chunks) <= 4
        covered = {seg.start_sec for chunk in chunks for seg in chunk.segments}
        assert covered == {seg.start_sec for seg in segments}

    def test_splits_only_on_segment_boundaries(self):
        chunks = chunk_segments(_long_transcript(100), chunk_chars=1_500, overlap_chars=100)
        for chunk in chunks:
            for line in chunk.text.splitlines():
                assert line.startswith("["), "a line was cut mid-sentence"

    def test_empty_input(self):
        assert chunk_segments([]) == []

    def test_invalid_overlap_rejected(self):
        with pytest.raises(ValueError, match="smaller than chunk_chars"):
            chunk_segments(_long_transcript(10), chunk_chars=100, overlap_chars=100)

    def test_chunk_boundaries_reported(self):
        chunks = chunk_segments(_long_transcript(100), chunk_chars=1_500)
        assert chunks[0].start_sec == 0.0
        assert chunks[-1].end_sec > chunks[0].end_sec


class TestWindowAround:
    def test_returns_nearby_speech(self, transcript_segments):
        text = window_around(transcript_segments, timestamp_sec=48, before_sec=20, after_sec=20)
        assert "940 megabits" in text

    def test_falls_back_to_nearest_when_window_is_empty(self, transcript_segments):
        text = window_around(transcript_segments, timestamp_sec=9999, before_sec=1, after_sec=1)
        assert text, "must return the closest segment rather than nothing"

    def test_respects_max_chars(self):
        text = window_around(_long_transcript(200), timestamp_sec=500, max_chars=120)
        assert len(text) <= 130  # allows for the ellipsis

    def test_empty_segments(self):
        assert window_around([], timestamp_sec=10) == ""
