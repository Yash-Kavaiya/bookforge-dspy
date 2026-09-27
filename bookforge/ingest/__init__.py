"""Ingest: resolve whatever the user pointed at into a list of recordings.

``resolve_source`` takes the same argument the CLI and the API accept — a URL, a
file, a folder, or several of them — and returns the right source. Callers do
not branch on input type; that decision lives here.
"""

from __future__ import annotations

import re
from pathlib import Path

from bookforge.config import Settings
from bookforge.ingest.base import (
    DiscoveredItem,
    Discovery,
    Source,
    humanize_filename,
    natural_key,
    stable_id,
)
from bookforge.ingest.sources import (
    LocalVideoSource,
    MixedFolderSource,
    TranscriptSource,
    YouTubeSource,
)

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_YOUTUBE_RE = re.compile(r"(youtube\.com|youtu\.be)", re.IGNORECASE)


def looks_like_url(value: str) -> bool:
    return bool(_URL_RE.match(str(value).strip()))


def resolve_source(
    spec: str | Path | list[str | Path],
    settings: Settings,
    title: str = "",
) -> Source:
    """Pick the source for whatever the user supplied.

    Accepts a YouTube URL, a single video or transcript file, a folder, or a
    list mixing files and folders. Raises with a specific message rather than
    guessing when the input cannot be interpreted — a wrong guess here wastes a
    long pipeline run.
    """
    specs = spec if isinstance(spec, list) else [spec]
    if not specs:
        raise ValueError("no input was provided")

    texts = [str(item).strip() for item in specs if str(item).strip()]
    if not texts:
        raise ValueError("no input was provided")

    urls = [text for text in texts if looks_like_url(text)]
    if urls:
        if len(urls) != len(texts):
            raise ValueError(
                "Mixing URLs and local paths in one run is not supported. Build "
                "the book from one kind of source, or run twice into the same "
                "workspace."
            )
        if len(urls) > 1:
            raise ValueError(
                "Give one channel, playlist or video URL at a time. A playlist "
                "URL already covers many videos."
            )
        url = urls[0]
        if not _YOUTUBE_RE.search(url):
            raise ValueError(
                f"{url} is not a YouTube URL. BookForge ingests YouTube links, "
                "local video files, or transcript files."
            )
        return YouTubeSource(url=url, settings=settings)

    paths = [Path(text).expanduser() for text in texts]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise ValueError(
            "These paths do not exist: " + ", ".join(str(p) for p in missing)
        )

    # A single folder is the common case: take videos and transcripts together.
    if len(paths) == 1 and paths[0].is_dir():
        return MixedFolderSource(path=paths[0], settings=settings, title_override=title)

    video_exts = {ext.lower() for ext in settings.video_extensions}
    transcript_exts = {ext.lower() for ext in settings.transcript_extensions}
    suffixes = {p.suffix.lower() for p in paths if p.is_file()}

    has_video = bool(suffixes & video_exts)
    has_transcript = bool(suffixes & transcript_exts)

    if has_video and not has_transcript:
        return LocalVideoSource(paths=paths, settings=settings, title_override=title)
    if has_transcript and not has_video:
        return TranscriptSource(paths=paths, settings=settings, title_override=title)
    if has_video and has_transcript:
        # Files of both kinds: treat their common parent as a mixed folder so a
        # transcript beside its video is paired rather than duplicated.
        parents = {p.parent for p in paths}
        if len(parents) == 1:
            return MixedFolderSource(
                path=parents.pop(), settings=settings, title_override=title
            )
        return LocalVideoSource(paths=paths, settings=settings, title_override=title)

    raise ValueError(
        "None of the given files is a recognised video or transcript. "
        f"Video types: {', '.join(settings.video_extensions)}. "
        f"Transcript types: {', '.join(settings.transcript_extensions)}."
    )


__all__ = [
    "DiscoveredItem",
    "Discovery",
    "LocalVideoSource",
    "MixedFolderSource",
    "Source",
    "TranscriptSource",
    "YouTubeSource",
    "humanize_filename",
    "looks_like_url",
    "natural_key",
    "resolve_source",
    "stable_id",
]
