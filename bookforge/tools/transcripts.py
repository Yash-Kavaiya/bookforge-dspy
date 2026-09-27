"""Importing transcripts that already exist.

SubRip (.srt), WebVTT (.vtt) and plain text all end up in the same normalized
form the rest of the pipeline reads: one ``[HH:MM:SS] text`` line per segment.

Plain text without timestamps is the awkward case. Everything downstream —
chunking, frame captioning, grounding evidence — is keyed on time, so a
timestamp-free transcript is given synthetic ones estimated from reading speed.
They are approximate and labelled as such; without them the chapter would lose
chunk boundaries and evidence links entirely, which costs more than the
imprecision does.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from bookforge.schemas import TranscriptSegment

logger = logging.getLogger(__name__)

#: Words per minute used to estimate timings for untimed text. Conversational
#: technical speech runs 130-160; 145 sits in the middle.
ASSUMED_WPM = 145

_SRT_TIME_RE = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{1,3})\s*-->"
)
_VTT_TIME_RE = re.compile(r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})[.,]\d+\s*-->")
#: WebVTT allows MM:SS.mmm as well as HH:MM:SS.mmm.
_VTT_SHORT_TIME_RE = re.compile(r"(?P<m>\d{1,2}):(?P<s>\d{2})[.,]\d+\s*-->")
_TAG_RE = re.compile(r"<[^>]+>")
_BRACKET_TS_RE = re.compile(r"^\[(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})\]\s*(?P<text>.*)$")


def detect_format(path: Path) -> str:
    """``srt`` | ``vtt`` | ``timestamped`` | ``plain``."""
    suffix = Path(path).suffix.lower()
    head = Path(path).read_text(encoding="utf-8", errors="ignore")[:4000]

    if suffix == ".vtt" or head.lstrip().startswith("WEBVTT"):
        return "vtt"
    if suffix == ".srt" or _SRT_TIME_RE.search(head):
        return "srt"
    if _BRACKET_TS_RE.search(head.splitlines()[0] if head.splitlines() else ""):
        return "timestamped"
    if _VTT_TIME_RE.search(head):
        return "vtt"
    return "plain"


def parse_srt(text: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    start: float | None = None
    buffer: list[str] = []

    def flush() -> None:
        if start is not None and buffer:
            body = " ".join(buffer).strip()
            if body:
                segments.append(TranscriptSegment(start_sec=start, text=body))

    for raw in text.splitlines():
        line = raw.strip()
        match = _SRT_TIME_RE.search(line)
        if match:
            flush()
            buffer = []
            start = (
                int(match.group("h")) * 3600
                + int(match.group("m")) * 60
                + int(match.group("s"))
            )
            continue
        if not line:
            flush()
            buffer = []
            start = None if start is None else start
            continue
        if line.isdigit() and not buffer:
            continue  # cue index
        buffer.append(_TAG_RE.sub("", line))
    flush()
    return segments


def parse_vtt(text: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    last_text = ""
    start = 0.0

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE")):
            continue
        match = _VTT_TIME_RE.search(line)
        if match:
            start = float(
                int(match.group("h")) * 3600
                + int(match.group("m")) * 60
                + int(match.group("s"))
            )
            continue
        short = _VTT_SHORT_TIME_RE.search(line)
        if short:
            start = float(int(short.group("m")) * 60 + int(short.group("s")))
            continue
        if line.isdigit():
            continue
        body = _TAG_RE.sub("", line).strip()
        # Auto-captions scroll, repeating the previous line verbatim.
        if not body or body == last_text:
            continue
        segments.append(TranscriptSegment(start_sec=start, text=body))
        last_text = body
    return segments


def parse_timestamped(text: str) -> list[TranscriptSegment]:
    """Our own ``[HH:MM:SS] text`` format, or anything close to it."""
    segments: list[TranscriptSegment] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _BRACKET_TS_RE.match(line)
        if match:
            start = (
                int(match.group("h")) * 3600
                + int(match.group("m")) * 60
                + int(match.group("s"))
            )
            segments.append(
                TranscriptSegment(start_sec=float(start), text=match.group("text").strip())
            )
        elif segments:
            segments[-1].text = f"{segments[-1].text} {line}".strip()
        else:
            segments.append(TranscriptSegment(start_sec=0.0, text=line))
    return segments


def parse_plain(text: str, wpm: int = ASSUMED_WPM) -> list[TranscriptSegment]:
    """Split untimed prose into sentence groups with estimated timings.

    The timings are fabricated and BookForge says so wherever it matters. They
    exist because chunking, captioning and evidence links are all keyed on time;
    an untimed transcript would otherwise degrade far more than the estimate
    error costs.
    """
    #: Group into ~40-word segments, roughly a caption cue's worth of speech.
    words_per_segment = 40
    seconds_per_word = 60.0 / max(wpm, 1)

    # Split on sentence ends so a segment does not start mid-clause.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    segments: list[TranscriptSegment] = []
    elapsed = 0.0
    buffer: list[str] = []
    buffered_words = 0

    def flush(at: float) -> None:
        nonlocal buffer, buffered_words
        if buffer:
            segments.append(
                TranscriptSegment(start_sec=round(at, 1), text=" ".join(buffer))
            )
            buffer = []
            buffered_words = 0

    for sentence in sentences:
        count = len(sentence.split())
        if buffered_words and buffered_words + count > words_per_segment:
            flush(elapsed - buffered_words * seconds_per_word)
        buffer.append(sentence)
        buffered_words += count
        elapsed += count * seconds_per_word
    flush(elapsed - buffered_words * seconds_per_word)

    if segments:
        logger.info(
            "estimated timings for %d untimed segments at %d wpm; they are "
            "approximate",
            len(segments), wpm,
        )
    return segments


def load_segments(path: Path) -> list[TranscriptSegment]:
    """Read any supported transcript file into normalized segments."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []

    kind = detect_format(path)
    parsers = {
        "srt": parse_srt,
        "vtt": parse_vtt,
        "timestamped": parse_timestamped,
        "plain": parse_plain,
    }
    segments = parsers[kind](text)
    logger.info("parsed %s as %s: %d segments", path.name, kind, len(segments))
    return segments


def write_normalized(segments: list[TranscriptSegment], dest: Path) -> Path:
    """Write segments in the pipeline's canonical ``[HH:MM:SS] text`` form."""
    from bookforge.workspace import Workspace

    body = "\n".join(f"[{seg.timestamp}] {seg.text}" for seg in segments)
    return Workspace.write_text(Path(dest), body + "\n")
