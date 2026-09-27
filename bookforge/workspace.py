"""Filesystem workspace + manifest management.

The workspace is the durable memory of the pipeline; in-memory state carries
only small pointers into it, so any run can be killed and resumed.

Layout under ``<root>/<channel_slug>/``::

    manifest.json
    videos/<video_id>/     video.mp4 audio.wav transcript.txt media.json
                           frames_raw/ frames/
    chapters/<NN>_<slug>/  chapter.tex verdict.json figures/ tables/
    build/                 qa_<video_id>/ ...
    book/                  main.tex book.pdf book_manifest.json

Concurrency
-----------
The original implementation guarded manifest writes with an ``asyncio.Lock``
while the actual concurrent writers were OS threads (``asyncio.to_thread``) —
an asyncio lock provides no mutual exclusion across threads, so a
read-modify-write on manifest.json could lose a status update and make a
finished chapter look pending forever. Here every mutation goes through a
process-wide ``threading.RLock`` *and* an on-disk lock file, so concurrent
threads and concurrent ``bookforge`` processes sharing a workspace are both
safe.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bookforge.schemas import (
    TERMINAL_OK_STATUSES,
    ChannelManifest,
    VideoRecord,
)

logger = logging.getLogger(__name__)

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _process_lock(key: str) -> threading.RLock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


@contextmanager
def _file_lock(path: Path, timeout: float = 30.0) -> Iterator[None]:
    """Cross-process advisory lock via exclusive file creation.

    Deliberately dependency-free and portable. A stale lock (owner crashed) is
    reclaimed after ``timeout`` seconds rather than deadlocking the pipeline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                logger.warning("reclaiming stale workspace lock at %s", path)
                with suppress(OSError):
                    path.unlink()
                deadline = time.monotonic() + timeout
            time.sleep(0.05)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        fd = None
        yield
    finally:
        if fd is not None:
            os.close(fd)
        with suppress(OSError):
            path.unlink()


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug[:max_len].strip("-") or "untitled"


class Workspace:
    """Durable, resumable, concurrency-safe project directory."""

    def __init__(self, root: Path, channel_slug: str) -> None:
        self.channel_slug = channel_slug
        self.root = Path(root) / channel_slug
        self.videos_dir = self.root / "videos"
        self.chapters_dir = self.root / "chapters"
        self.build_dir = self.root / "build"
        self.book_dir = self.root / "book"
        for directory in (
            self.root,
            self.videos_dir,
            self.chapters_dir,
            self.build_dir,
            self.book_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._lock = _process_lock(str(self.root.resolve()))

    # -- manifest -----------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def lock_path(self) -> Path:
        return self.root / ".manifest.lock"

    def manifest_exists(self) -> bool:
        return self.manifest_path.exists()

    def load_manifest(self) -> ChannelManifest:
        with self._lock:
            return ChannelManifest.model_validate_json(
                self.manifest_path.read_text(encoding="utf-8")
            )

    def save_manifest(self, manifest: ChannelManifest) -> None:
        with self._lock, _file_lock(self.lock_path):
            self._atomic_write_json(
                self.manifest_path, manifest.model_dump(mode="json")
            )

    def init_manifest(
        self, channel_url: str, channel_title: str, slug: str
    ) -> ChannelManifest:
        manifest = ChannelManifest(
            channel_url=channel_url,
            channel_title=channel_title,
            channel_slug=slug,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.save_manifest(manifest)
        return manifest

    def update_video(self, video_id: str, **changes: Any) -> VideoRecord:
        """Atomic read-modify-write of one video record.

        Safe against concurrent threads and concurrent processes. This is the
        ONLY supported way to mutate the manifest during a run.
        """
        with self._lock, _file_lock(self.lock_path):
            manifest = ChannelManifest.model_validate_json(
                self.manifest_path.read_text(encoding="utf-8")
            )
            record = manifest.by_id(video_id)
            if record is None:
                raise KeyError(f"video {video_id} is not in the manifest")
            for key, value in changes.items():
                if not hasattr(record, key):
                    raise AttributeError(f"VideoRecord has no field {key!r}")
                setattr(record, key, value)
            self._atomic_write_json(
                self.manifest_path, manifest.model_dump(mode="json")
            )
            return record

    def pending_videos(self) -> list[VideoRecord]:
        """Videos whose chapter is not yet verified — the resume source of truth."""
        return [
            v
            for v in self.load_manifest().ordered()
            if v.status not in TERMINAL_OK_STATUSES
        ]

    # -- per-video paths ----------------------------------------------------

    def video_dir(self, video_id: str) -> Path:
        directory = self.videos_dir / video_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def chapter_dir(self, record: VideoRecord) -> Path:
        directory = (
            self.chapters_dir / f"{record.chapter_number:02d}_{slugify(record.title)}"
        )
        for sub in (directory, directory / "figures", directory / "tables"):
            sub.mkdir(parents=True, exist_ok=True)
        return directory

    def rel(self, path: Path) -> str:
        """Workspace-root-relative POSIX path, as used inside LaTeX."""
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, delete=False, encoding="utf-8", suffix=".tmp"
        ) as tmp:
            json.dump(payload, tmp, indent=2, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)

    @staticmethod
    def write_text(path: Path, content: str) -> Path:
        """Atomic text write — a killed process never leaves a half-written .tex."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, delete=False, encoding="utf-8", suffix=".tmp"
        ) as tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
        return path

    @staticmethod
    def read_text(path: Path) -> str:
        return Path(path).read_text(encoding="utf-8", errors="replace")
