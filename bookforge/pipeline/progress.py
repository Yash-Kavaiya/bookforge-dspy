"""Progress reporting and ETA.

Replaces ADK's ``Event`` stream with a small observer: the orchestrator emits
typed events, and consumers (CLI, MLflow, tests) subscribe. Keeping this
independent of any agent framework is what lets the pipeline run identically
under a CLI, a notebook, or a test harness.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

EventKind = Literal["info", "stage", "chapter_start", "chapter_end", "warning", "error", "done"]


class PipelineCancelled(RuntimeError):
    """Raised when a cooperative cancel is requested mid-pipeline."""


@dataclass
class PipelineEvent:
    kind: EventKind
    message: str
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


Listener = Callable[[PipelineEvent], None]


class Reporter:
    """Fan-out of pipeline events to any number of listeners."""

    def __init__(
        self,
        listeners: list[Listener] | None = None,
        cancel_check: Callable[[], None] | None = None,
    ) -> None:
        self._listeners: list[Listener] = list(listeners or [])
        self._cancel_check = cancel_check

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def checkpoint(self, message: str = "") -> None:
        """Raise if the run was cancelled — call between pipeline stages."""
        if self._cancel_check is not None:
            self._cancel_check()

    def emit(self, kind: EventKind, message: str, **data) -> PipelineEvent:
        # Also check cancel on every event so mid-stage listeners still stop.
        self.checkpoint()
        event = PipelineEvent(kind=kind, message=message, data=data)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as exc:
                # Cancellation must propagate; other listener failures must not.
                if isinstance(exc, PipelineCancelled):
                    raise
                logger.debug("progress listener raised", exc_info=True)
        return event

    # Convenience wrappers ---------------------------------------------------
    def info(self, message: str, **data) -> PipelineEvent:
        logger.info(message)
        return self.emit("info", message, **data)

    def stage(self, message: str, **data) -> PipelineEvent:
        logger.info(message)
        return self.emit("stage", message, **data)

    def warning(self, message: str, **data) -> PipelineEvent:
        logger.warning(message)
        return self.emit("warning", message, **data)

    def error(self, message: str, **data) -> PipelineEvent:
        logger.error(message)
        return self.emit("error", message, **data)


def format_duration(seconds: float) -> str:
    """Format seconds as ``45s`` / ``3m12s`` / ``1h05m``."""
    total = max(0, round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class ProgressTracker:
    """Rolling per-chapter ETA based on observed durations."""

    def __init__(self, total: int, default_chapter_sec: float = 8 * 60) -> None:
        self.total = max(0, total)
        self.done = 0
        self.failed = 0
        self.started_at = time.monotonic()
        self.durations: list[float] = []
        self.scores: list[float] = []
        self.default_chapter_sec = default_chapter_sec
        self._chapter_started_at: float | None = None

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.done)

    def average_chapter_sec(self) -> float:
        if self.durations:
            return sum(self.durations) / len(self.durations)
        return self.default_chapter_sec

    def eta_sec(self, remaining: int | None = None) -> float:
        left = self.remaining if remaining is None else remaining
        return self.average_chapter_sec() * left

    def start_chapter(self, index: int, title: str) -> str:
        self._chapter_started_at = time.monotonic()
        elapsed = time.monotonic() - self.started_at
        left = max(0, self.total - index)
        return (
            f"[{index + 1}/{self.total}] Starting: {title}. "
            f"Elapsed {format_duration(elapsed)}, "
            f"ETA ~{format_duration(self.eta_sec(left))} "
            f"({left} left at ~{format_duration(self.average_chapter_sec())}/chapter)."
        )

    def finish_chapter(
        self, index: int, title: str, status: str, score: float = 0.0
    ) -> str:
        now = time.monotonic()
        if self._chapter_started_at is not None:
            elapsed_chapter = now - self._chapter_started_at
            # Ignore sub-second timings (cache hits, tests) so the ETA stays useful.
            if elapsed_chapter >= 1.0:
                self.durations.append(elapsed_chapter)
        self._chapter_started_at = None
        self.done += 1
        if status in {"failed", "error"}:
            self.failed += 1
        elif score:
            self.scores.append(score)

        elapsed = now - self.started_at
        left = max(0, self.total - self.done)
        score_note = f" score {score:.2f}" if score else ""
        return (
            f"[{index + 1}/{self.total}] Finished: {title} ({status}{score_note}). "
            f"Elapsed {format_duration(elapsed)}, "
            f"ETA ~{format_duration(self.eta_sec(left))} "
            f"({left} left, {self.failed} failed)."
        )

    def summary(self) -> str:
        elapsed = time.monotonic() - self.started_at
        mean_score = sum(self.scores) / len(self.scores) if self.scores else 0.0
        quality = f", mean quality {mean_score:.2f}" if self.scores else ""
        return (
            f"Production complete: {self.done}/{self.total} chapters "
            f"({self.failed} failed) in {format_duration(elapsed)}{quality}."
        )
