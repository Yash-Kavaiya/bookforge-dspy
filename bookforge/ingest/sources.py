"""The four concrete sources: YouTube, local video, transcripts, mixed folders."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from bookforge.config import Settings
from bookforge.ingest.base import (
    DiscoveredItem,
    Discovery,
    humanize_filename,
    sort_key_for,
    stable_id,
)
from bookforge.tools import youtube as yt

logger = logging.getLogger(__name__)

#: The phrase a mixed folder filters out once it has found transcripts instead.
#: Kept in one place so the filter cannot drift from the message.
_NO_VIDEOS = "contains no files with a known video extension"


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------


@dataclass
class YouTubeSource:
    """A channel, playlist or single video URL."""

    url: str
    settings: Settings
    kind = "youtube"

    def auth(self) -> dict:
        return yt.auth_options(
            cookies_file=self.settings.youtube_cookies_file,
            cookies_from_browser=self.settings.youtube_cookies_from_browser,
            proxy=self.settings.youtube_proxy,
        )

    def discover(self) -> Discovery:
        title, videos = yt.list_channel_videos(
            self.url,
            max_videos=self.settings.max_videos,
            min_duration_sec=self.settings.min_video_duration_sec,
            max_duration_sec=self.settings.max_video_duration_sec,
            auth=self.auth(),
        )
        items = [
            DiscoveredItem(
                video_id=video["video_id"],
                title=video["title"],
                source="youtube",
                url=video["url"],
                duration_sec=video["duration_sec"],
                upload_date=video["upload_date"],
                # Chronological, so the book reads the way the channel developed.
                order_key=(video["upload_date"] == "", video["upload_date"]),
            )
            for video in videos
        ]
        warnings: list[str] = []
        if not self.auth():
            warnings.append(
                "No YouTube cookies configured. Channel listing works without "
                "them, but downloads from a server IP are usually refused with a "
                "bot check. Set BOOKFORGE_YOUTUBE_COOKIES_FILE if downloads fail."
            )
        return Discovery(
            title=title,
            items=items,
            source_line=self.url,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Local video files
# ---------------------------------------------------------------------------


@dataclass
class LocalVideoSource:
    """One or more local video files, or a folder of them."""

    paths: list[Path]
    settings: Settings
    title_override: str = ""
    kind = "local_video"

    def _collect(self) -> tuple[list[Path], list[str]]:
        found: list[Path] = []
        warnings: list[str] = []
        extensions = {ext.lower() for ext in self.settings.video_extensions}

        for raw in self.paths:
            path = Path(raw).expanduser()
            if not path.exists():
                warnings.append(f"{path} does not exist and was skipped.")
                continue
            if path.is_dir():
                # Recursive: a folder of talks is often one directory per event.
                matches = [
                    child
                    for child in sorted(path.rglob("*"))
                    if child.is_file() and child.suffix.lower() in extensions
                ]
                if not matches:
                    warnings.append(
                        f"{path} {_NO_VIDEOS} "
                        f"({', '.join(sorted(extensions))})."
                    )
                found.extend(matches)
            elif path.suffix.lower() in extensions:
                found.append(path)
            else:
                warnings.append(
                    f"{path.name} is not a recognised video type and was skipped."
                )

        # A file reachable by two of the given paths must not become two chapters.
        unique: dict[Path, Path] = {}
        for path in found:
            unique.setdefault(path.resolve(), path)
        return list(unique.values()), warnings

    def discover(self) -> Discovery:
        files, warnings = self._collect()
        order = self.settings.local_order

        items: list[DiscoveredItem] = []
        skipped_short = 0
        for path in files:
            duration = int(yt.probe_duration(path))
            if duration and duration < self.settings.min_video_duration_sec:
                skipped_short += 1
                continue
            items.append(
                DiscoveredItem(
                    video_id=stable_id(str(path.resolve()), prefix="vid"),
                    title=humanize_filename(path),
                    source="local_video",
                    source_path=str(path.resolve()),
                    duration_sec=duration,
                    order_key=sort_key_for(path, order),
                )
            )
        if skipped_short:
            warnings.append(
                f"{skipped_short} file(s) shorter than "
                f"{self.settings.min_video_duration_sec}s were skipped."
            )
        if self.settings.max_videos:
            items = sorted(items, key=lambda i: i.order_key)[: self.settings.max_videos]

        roots = ", ".join(str(Path(p).expanduser()) for p in self.paths)
        return Discovery(
            title=self.title_override or _title_from_paths(self.paths),
            items=items,
            source_line=f"Local recordings: {roots}",
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Existing transcripts
# ---------------------------------------------------------------------------


@dataclass
class TranscriptSource:
    """A folder of .srt/.vtt/.txt transcripts, with no video at all.

    The fastest path by a wide margin: no download, no ffmpeg, no Whisper. It is
    also the only path that works when the recording is gone and the text is all
    that survives. Chapters get no screenshots, so the book carries charts,
    tables and diagrams but no frames.
    """

    paths: list[Path]
    settings: Settings
    title_override: str = ""
    kind = "transcript"

    def discover(self) -> Discovery:
        extensions = {ext.lower() for ext in self.settings.transcript_extensions}
        found: list[Path] = []
        warnings: list[str] = []

        for raw in self.paths:
            path = Path(raw).expanduser()
            if not path.exists():
                warnings.append(f"{path} does not exist and was skipped.")
                continue
            if path.is_dir():
                found.extend(
                    child
                    for child in sorted(path.rglob("*"))
                    if child.is_file() and child.suffix.lower() in extensions
                )
            elif path.suffix.lower() in extensions:
                found.append(path)
            else:
                warnings.append(f"{path.name} is not a recognised transcript type.")

        unique: dict[Path, Path] = {}
        for path in found:
            unique.setdefault(path.resolve(), path)

        items = [
            DiscoveredItem(
                video_id=stable_id(str(path), prefix="txt"),
                title=humanize_filename(path),
                source="transcript",
                source_path=str(path),
                order_key=sort_key_for(path, self.settings.local_order),
            )
            for path in unique.values()
        ]
        if self.settings.max_videos:
            items = sorted(items, key=lambda i: i.order_key)[: self.settings.max_videos]

        roots = ", ".join(str(Path(p).expanduser()) for p in self.paths)
        return Discovery(
            title=self.title_override or _title_from_paths(self.paths),
            items=items,
            source_line=f"Transcripts: {roots}",
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Mixed folder
# ---------------------------------------------------------------------------


@dataclass
class MixedFolderSource:
    """A folder holding videos and/or transcripts, taking whichever it finds.

    What "point BookForge at my folder" should mean. A transcript sitting beside
    a video of the same stem is treated as that video's transcript rather than a
    chapter of its own — otherwise the same talk would be typeset twice.
    """

    path: Path
    settings: Settings
    title_override: str = ""
    kind = "local_video"

    def discover(self) -> Discovery:
        root = Path(self.path).expanduser()
        video = LocalVideoSource([root], self.settings, self.title_override).discover()

        video_stems = {
            Path(item.source_path).with_suffix("").resolve()
            for item in video.items
        }
        transcripts = TranscriptSource(
            [root], self.settings, self.title_override
        ).discover()
        standalone = [
            item
            for item in transcripts.items
            if Path(item.source_path).with_suffix("").resolve() not in video_stems
        ]

        items = video.items + standalone
        warnings = list(video.warnings)
        if standalone:
            # A transcripts-only folder is a supported input, not a mistake, so
            # the video scan's "no files with a known video extension" complaint
            # is noise here -- and it renders as the loudest thing on a preview
            # that in fact succeeded.
            warnings = [w for w in warnings if _NO_VIDEOS not in w]
        paired = len(transcripts.items) - len(standalone)
        if paired:
            warnings.append(
                f"{paired} transcript(s) were matched to a video of the same name "
                "and will be used instead of transcribing it."
            )
        if not items:
            warnings.append(
                f"No videos or transcripts found under {root}. Supported video "
                f"types: {', '.join(self.settings.video_extensions)}; transcripts: "
                f"{', '.join(self.settings.transcript_extensions)}."
            )
        return Discovery(
            title=self.title_override or _title_from_paths([root]),
            items=items,
            source_line=f"Local folder: {root}",
            warnings=warnings,
        )


def _title_from_paths(paths: list[Path | str]) -> str:
    """A book title from the folder the material came from."""
    if not paths:
        return "Untitled Collection"
    first = Path(paths[0]).expanduser()
    base = first if first.is_dir() else first.parent
    name = base.name or base.resolve().name
    return humanize_filename(Path(name)) if name else "Untitled Collection"
