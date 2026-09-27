"""The orchestrator — intake, per-chapter production, book compilation.

This replaces ADK's ``SequentialAgent`` / ``LoopAgent`` / ``BaseAgent`` graph
with plain Python. That is not a downgrade: the graph was always sequential
with a prefetch thread pool bolted on, and expressing it directly removes a
framework dependency, an async event protocol, and the session-state dance that
made ``frame_captions`` fail silently in the first place.

Error isolation is preserved and strengthened: a video that raises is marked
``failed`` and the run continues, so one broken source never costs the book.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from bookforge.config import Settings, get_settings
from bookforge.dspy_runtime.lm import USAGE
from bookforge.dspy_runtime.program import BookForgeProgram
from bookforge.pipeline.chapter import ChapterOutcome, produce_chapter
from bookforge.pipeline.compile import compile_book
from bookforge.pipeline.intake import run_intake
from bookforge.pipeline.media import acquire_media
from bookforge.pipeline.progress import PipelineCancelled, ProgressTracker, Reporter
from bookforge.schemas import VideoRecord
from bookforge.workspace import Workspace

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    channel_slug: str
    workspace_root: str
    outcomes: list[ChapterOutcome] = field(default_factory=list)
    book: dict | None = None
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return self.book is not None and not self.error

    @property
    def verified(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "verified")

    @property
    def failed(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "failed")

    def summary(self) -> str:
        if self.error:
            return f"Run failed: {self.error}"
        pdf = (self.book or {}).get("pdf") or "(no PDF)"
        return (
            f"{len(self.outcomes)} chapters produced "
            f"({self.verified} verified, {self.failed} failed). PDF: {pdf}"
        )


class MediaPrefetcher:
    """Downloads and transcribes upcoming videos while the current one is in LLM stages.

    Media acquisition is IO-bound and idempotent, so overlapping it with the
    LLM work is free latency. Threads only ever *write cache files*; every
    manifest mutation stays on the main thread, which is what keeps the
    read-modify-write of the manifest serialized.
    """

    def __init__(
        self,
        workspace: Workspace,
        settings: Settings,
        records: list[VideoRecord],
        concurrency: int,
    ) -> None:
        self.workspace = workspace
        self.settings = settings
        self.records = records
        self.concurrency = max(1, concurrency)
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._futures: list[concurrent.futures.Future] = []
        self._stop = threading.Event()

    def start(self) -> None:
        if not self.records:
            return
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.concurrency, thread_name_prefix="bf-prefetch"
        )
        self._futures = [
            self._pool.submit(self._prefetch_one, record) for record in self.records
        ]

    def _prefetch_one(self, record: VideoRecord) -> None:
        if self._stop.is_set():
            return
        try:
            acquire_media(record, self.workspace, self.settings)
            logger.info("prefetched media for %s", record.video_id)
        except Exception as exc:
            # The main pass will retry this video and report the real error.
            logger.info("media prefetch failed for %s: %s", record.video_id, exc)

    def stop(self) -> None:
        self._stop.set()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None


def run_pipeline(
    source: str | Path | list,
    settings: Settings | None = None,
    reporter: Reporter | None = None,
    program: BookForgeProgram | None = None,
    title: str = "",
) -> RunResult:
    """Produce a complete book from any supported source.

    ``source`` is a YouTube URL, a video file, a transcript file, a folder, or
    a list of those — see ``bookforge.ingest.resolve_source``.
    """
    settings = settings or get_settings()
    reporter = reporter or Reporter()
    program = program or BookForgeProgram(settings)

    reporter.info(program.describe())

    workspace, _manifest = run_intake(source, settings, reporter, title=title)
    result = RunResult(
        channel_slug=workspace.channel_slug, workspace_root=str(workspace.root)
    )

    pending = workspace.pending_videos()
    if not pending:
        reporter.info("Every chapter is already verified; compiling the book.")
        result.book = compile_book(workspace, settings, reporter)
        return result

    tracker = ProgressTracker(len(pending))
    reporter.stage(
        f"Producing {len(pending)} chapter(s) with media prefetch x"
        f"{settings.media_prefetch_concurrency}. Initial ETA "
        f"~{_eta(tracker)}."
    )

    prefetcher = MediaPrefetcher(
        workspace, settings, pending[1:], settings.media_prefetch_concurrency
    )
    prefetcher.start()

    try:
        for index, record in enumerate(pending):
            reporter.checkpoint()
            reporter.stage(tracker.start_chapter(index, record.title))
            try:
                outcome = produce_chapter(
                    record, workspace, program, settings, reporter
                )
            except Exception as exc:
                # Cooperative cancel must not be recorded as a chapter failure.
                if isinstance(exc, PipelineCancelled):
                    prefetcher.stop()
                    raise
                logger.exception("chapter pipeline failed for %s", record.video_id)
                workspace.update_video(
                    record.video_id, status="failed", error=str(exc)[:500]
                )
                outcome = ChapterOutcome(
                    video_id=record.video_id,
                    chapter_number=record.chapter_number,
                    title=record.title,
                    status="failed",
                    error=str(exc)[:500],
                )
                reporter.error(
                    tracker.finish_chapter(index, record.title, "failed")
                    + f" Error: {exc}"
                )
            else:
                reporter.stage(
                    tracker.finish_chapter(
                        index, record.title, outcome.status, outcome.score
                    )
                )
            result.outcomes.append(outcome)
    finally:
        prefetcher.stop()

    reporter.info(tracker.summary())
    reporter.info(USAGE.summary())

    try:
        result.book = compile_book(workspace, settings, reporter)
    except Exception as exc:
        logger.exception("book compilation failed")
        result.error = str(exc)
    return result


def _eta(tracker: ProgressTracker) -> str:
    from bookforge.pipeline.progress import format_duration

    return format_duration(tracker.eta_sec())
