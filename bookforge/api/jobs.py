"""Background job runner: one book build per job, observable while it runs.

The pipeline is synchronous and long-running — a book takes minutes to hours.
A web UI cannot hold an HTTP request open for that, so a job is started, given
an id, and then polled or streamed.

Design notes worth stating:

* **Jobs run in a thread, not a task queue.** BookForge is a single-tenant tool
  someone runs on their own machine or one server; adding Redis and Celery
  would triple the deployment surface to solve a problem this does not have.
  The workspace is the durable state, so a crashed process resumes from disk —
  which is the property a queue would otherwise be providing.
* **Events are buffered, not just streamed.** A browser that connects late, or
  reconnects after a dropped WebSocket, still gets the full history. Streaming
  without a buffer means refreshing the page loses everything that happened.
* **Cancellation is cooperative.** The pipeline checks a flag between chapters
  rather than being killed mid-write, so a cancelled run leaves a consistent
  workspace that can be resumed rather than a half-written chapter.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from bookforge.config import Settings
from bookforge.pipeline.progress import PipelineCancelled, PipelineEvent, Reporter

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]

#: Events kept per job. Enough for a long run's worth of chapter transitions
#: without letting one job's log grow without bound.
MAX_EVENTS = 2000


class JobCancelled(PipelineCancelled):
    """Raised inside the pipeline thread when a cancel is requested."""


@dataclass
class JobEvent:
    seq: int
    kind: str
    message: str
    at: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "message": self.message,
            "at": self.at,
            "data": self.data,
        }


@dataclass
class Job:
    """One book build."""

    id: str
    source: list[str]
    title: str
    theme: str
    status: JobStatus = "queued"
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    workspace_root: str = ""
    channel_slug: str = ""
    result: dict | None = None
    chapters: list[dict] = field(default_factory=list)

    _events: deque[JobEvent] = field(default_factory=lambda: deque(maxlen=MAX_EVENTS))
    _seq: int = 0
    _cancel: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _waiters: list[threading.Event] = field(default_factory=list)

    # -- events -------------------------------------------------------------

    def add_event(self, kind: str, message: str, **data) -> JobEvent:
        with self._lock:
            self._seq += 1
            event = JobEvent(
                seq=self._seq,
                kind=kind,
                message=message,
                at=datetime.now(timezone.utc).isoformat(),
                data=data,
            )
            self._events.append(event)
            waiters = list(self._waiters)
            self._waiters.clear()
        # Wake streamers outside the lock so a slow consumer cannot block the
        # pipeline thread that is producing events.
        for waiter in waiters:
            waiter.set()
        return event

    def events_since(self, after: int = 0) -> list[JobEvent]:
        with self._lock:
            return [event for event in self._events if event.seq > after]

    def wait_for_event(self, after: int, timeout: float = 20.0) -> bool:
        """Block until an event newer than ``after`` exists, or timeout."""
        with self._lock:
            if any(event.seq > after for event in self._events):
                return True
            if self.is_finished:
                return False
            waiter = threading.Event()
            self._waiters.append(waiter)
        return waiter.wait(timeout)

    # -- lifecycle ----------------------------------------------------------

    @property
    def is_finished(self) -> bool:
        return self.status in ("succeeded", "failed", "cancelled")

    def cancel(self) -> bool:
        if self.is_finished:
            return False
        self._cancel.set()
        self.add_event(
            "warning",
            "Cancellation requested; stopping at the next stage boundary.",
        )
        return True

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled("the run was cancelled")

    def to_dict(self, include_events: bool = False) -> dict:
        # Lazy: dspy is a heavy, import-order-sensitive dependency (see the
        # numpy guard in cli.py), so it must not load just because something
        # asked for a job's status — only building/serving actually needs it.
        from bookforge.dspy_runtime.lm import USAGE

        payload: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "theme": self.theme,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "workspace_root": self.workspace_root,
            "channel_slug": self.channel_slug,
            "result": self.result,
            "chapters": self.chapters,
            "last_seq": self._seq,
            # Global, not per-job — fine here because only one job runs at a
            # time (see JobRunner's single-tenant note above); reset at start.
            "usage": USAGE.to_dict(),
        }
        if include_events:
            payload["events"] = [event.to_dict() for event in self.events_since(0)]
        return payload


class JobRunner:
    """Owns the jobs and the threads running them."""

    def __init__(self, settings: Settings, max_concurrent: int = 1) -> None:
        self.settings = settings
        self.max_concurrent = max_concurrent
        self._jobs: dict[str, Job] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    # -- registry -----------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(
                self._jobs.values(), key=lambda job: job.created_at, reverse=True
            )

    def running_count(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.status == "running")

    def active_count(self) -> int:
        """Queued + running — used to enforce the concurrency cap."""
        with self._lock:
            return sum(
                1
                for job in self._jobs.values()
                if job.status in ("queued", "running")
            )

    # -- submission ---------------------------------------------------------

    def submit(
        self,
        source: list[str],
        title: str = "",
        theme: str = "",
        max_videos: int | None = None,
        compile_latex: bool | None = None,
        qa_iterations: int | None = None,
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            source=[str(item) for item in source],
            title=title,
            theme=theme or self.settings.book_theme,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        overrides = {
            "book_theme": job.theme,
            "max_videos": max_videos,
            "compile_latex": compile_latex,
            "qa_max_iterations": qa_iterations,
        }
        thread = threading.Thread(
            target=self._run,
            args=(job, overrides),
            name=f"bookforge-job-{job.id}",
            daemon=True,
        )
        # Cap check + registration must be atomic: two POSTs that both see
        # status=="running" can otherwise both start while jobs are still queued.
        with self._lock:
            active = sum(
                1
                for existing in self._jobs.values()
                if existing.status in ("queued", "running")
            )
            if active >= self.max_concurrent:
                raise RuntimeError(
                    f"{self.max_concurrent} build(s) already running. Wait for one to "
                    "finish or cancel it — book builds are CPU and API heavy, so "
                    "running many at once makes all of them slower."
                )
            self._jobs[job.id] = job
            self._threads[job.id] = thread
        thread.start()
        return job

    # -- execution ----------------------------------------------------------

    def _run(self, job: Job, overrides: dict) -> None:
        from bookforge.dspy_runtime.lm import USAGE
        from bookforge.dspy_runtime.program import BookForgeProgram
        from bookforge.pipeline.orchestrator import run_pipeline

        job.status = "running"
        job.started_at = datetime.now(timezone.utc).isoformat()
        USAGE.reset()
        job.add_event("stage", f"Build {job.id} started.")

        # Each job gets its own Settings copy: two jobs must not fight over a
        # shared theme or video cap.
        settings = self.settings.model_copy(deep=True)
        for key, value in overrides.items():
            if value is not None:
                setattr(settings, key, value)

        reporter = Reporter(
            [lambda event: self._on_event(job, event)],
            cancel_check=job.check_cancelled,
        )

        try:
            program = BookForgeProgram(settings)
            result = run_pipeline(
                job.source if len(job.source) > 1 else job.source[0],
                settings,
                reporter,
                program,
                title=job.title,
            )
            job.channel_slug = result.channel_slug
            job.workspace_root = result.workspace_root
            job.chapters = [
                {
                    "video_id": outcome.video_id,
                    "chapter_number": outcome.chapter_number,
                    "title": outcome.title,
                    "status": outcome.status,
                    "score": outcome.score,
                    "grounding_score": outcome.grounding_score,
                    "iterations": outcome.iterations,
                    "compiled": outcome.compiled,
                    "error": outcome.error,
                }
                for outcome in result.outcomes
            ]
            job.result = result.book
            if result.error:
                job.status = "failed"
                job.error = result.error
                job.add_event("error", result.error)
            else:
                job.status = "succeeded"
                job.add_event("done", result.summary(), result=result.book)
        except JobCancelled:
            job.status = "cancelled"
            job.add_event("warning", "Build cancelled. Re-run to resume from where it stopped.")
        except Exception as exc:
            logger.exception("job %s failed", job.id)
            job.status = "failed"
            job.error = str(exc)
            job.add_event("error", f"Build failed: {exc}")
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat()
            # Wake any streamer still waiting, so it sees the terminal status
            # instead of hanging until its timeout.
            job.add_event("info", f"Build {job.status}.")

    def _on_event(self, job: Job, event: PipelineEvent) -> None:
        job.add_event(event.kind, event.message, **event.data)
        # Cancellation is checked here because the reporter is called between
        # every stage: it is the natural, safe interruption point.
        job.check_cancelled()


_runner: JobRunner | None = None
_runner_lock = threading.Lock()


def get_runner(settings: Settings | None = None) -> JobRunner:
    """Process-wide runner, created on first use."""
    global _runner
    with _runner_lock:
        if _runner is None:
            from bookforge.config import get_settings

            _runner = JobRunner(settings or get_settings())
        return _runner


def reset_runner() -> None:
    """Drop the runner. Tests only."""
    global _runner
    with _runner_lock:
        _runner = None
