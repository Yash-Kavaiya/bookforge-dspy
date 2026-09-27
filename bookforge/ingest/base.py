"""Source abstraction: where the recordings come from.

The ADK pipeline had YouTube hard-wired into intake, media acquisition and the
CLI. That was fine until it met the way YouTube actually behaves from a server:
flat channel listings succeed while every media and caption request is answered
with a bot challenge. The pipeline discovered a whole channel and then failed on
every single video — with intake reporting success, which is the worst place for
a failure to hide.

Four sources now sit behind one interface, and a workspace can mix them: a
manifest may hold YouTube videos, local files and bare transcripts together,
because ``VideoRecord.source`` says how to acquire each one.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from bookforge.schemas import SourceKind, VideoRecord


@dataclass
class DiscoveredItem:
    """One recording a source found, before it becomes a manifest record."""

    video_id: str
    title: str
    source: SourceKind
    url: str = ""
    source_path: str = ""
    duration_sec: int = 0
    upload_date: str = ""
    #: Sort key within the source. Lower comes first.
    order_key: tuple = ()

    def to_record(self, chapter_number: int) -> VideoRecord:
        return VideoRecord(
            video_id=self.video_id,
            title=self.title,
            url=self.url,
            source=self.source,
            source_path=self.source_path,
            duration_sec=self.duration_sec,
            upload_date=self.upload_date,
            chapter_number=chapter_number,
        )


@dataclass
class Discovery:
    """What a source found, plus what the reader should know about it."""

    title: str
    items: list[DiscoveredItem] = field(default_factory=list)
    #: Human-readable provenance, printed on the book's colophon.
    source_line: str = ""
    #: Non-fatal problems worth surfacing (skipped files, duration filters).
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)


class Source(Protocol):
    """Anything that can enumerate recordings to turn into chapters."""

    kind: SourceKind

    def discover(self) -> Discovery:
        """Enumerate the recordings, cheaply. No downloads, no transcription."""
        ...


# ---------------------------------------------------------------------------
# Identifier helpers
# ---------------------------------------------------------------------------

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_\-]")


def stable_id(value: str, prefix: str = "loc", max_len: int = 40) -> str:
    """A filesystem-safe, stable id derived from a path or name.

    Stability matters more than readability: the id becomes a directory name in
    the workspace, so it must survive a re-run unchanged or every resume would
    re-download and re-transcribe into a fresh directory. A content-independent
    hash of the path gives that, and a readable prefix keeps the workspace
    navigable by eye.
    """
    stem = _SAFE_ID_RE.sub("-", Path(value).stem)[:max_len].strip("-") or "item"
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{stem}-{digest}"[:64]


def humanize_filename(path: Path) -> str:
    """Turn ``03_the-harness-boundary.mp4`` into ``The Harness Boundary``.

    Local files are usually named by a numbering convention the author already
    imposed, so stripping it gives a serviceable chapter title. It is a guess,
    and the analyst renames the chapter anyway — this is what the progress log
    and the manifest show while that is still running.
    """
    stem = path.stem
    stem = re.sub(r"^[\s_\-.]*\d{1,4}[\s_\-.]+", "", stem)  # leading index
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    if not stem:
        return path.stem or path.name
    # Preserve deliberate capitalisation; only fix all-lowercase names.
    if stem.islower():
        stem = stem.title()
    return stem


def sort_key_for(path: Path, order: str) -> tuple:
    """Ordering key for local files."""
    if order == "modified":
        return (path.stat().st_mtime, path.name.lower())
    if order == "created":
        stat = path.stat()
        return (getattr(stat, "st_birthtime", stat.st_ctime), path.name.lower())
    # "name" — natural sort, so ep2 precedes ep10 the way a human expects.
    return (natural_key(path.name),)


def natural_key(name: str) -> tuple:
    """Split digits from text so ``ep2`` sorts before ``ep10``."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
        if part != ""
    )
