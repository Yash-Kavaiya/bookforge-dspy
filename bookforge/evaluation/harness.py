"""Evaluation harness: measure a program, and compare two of them.

An optimizer that reports "score went from 0.61 to 0.78" is only trustworthy if
the same measurement is applied to the baseline and the candidate over the same
examples. This module is that measurement, kept separate from the optimizers so
it cannot drift toward flattering them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import dspy

from bookforge.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSummary:
    label: str
    score: float
    n: int

    def __str__(self) -> str:
        return f"{self.label}: {self.score:.4f} over {self.n} example(s)"


@dataclass
class ComparisonResult:
    baseline: EvaluationSummary
    candidate: EvaluationSummary

    @property
    def delta(self) -> float:
        return round(self.candidate.score - self.baseline.score, 4)

    @property
    def improved(self) -> bool:
        return self.candidate.score > self.baseline.score

    def report(self) -> str:
        arrow = "improved" if self.improved else "did NOT improve"
        return (
            f"{self.baseline}\n{self.candidate}\n"
            f"Optimization {arrow}: {self.delta:+.4f} "
            f"({self.delta / max(self.baseline.score, 1e-9):+.1%} relative)"
        )


def evaluate_program(
    program: dspy.Module,
    devset: list[dspy.Example],
    metric: Callable[..., Any],
    label: str = "program",
    settings: Settings | None = None,
    display_progress: bool = True,
) -> EvaluationSummary:
    """Score a program over a dev set with ``dspy.Evaluate``."""
    settings = settings or get_settings()
    if not devset:
        raise ValueError("cannot evaluate against an empty devset")

    evaluator = dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=settings.num_threads,
        display_progress=display_progress,
        display_table=False,
        provide_traceback=True,
        failure_score=0.0,
    )
    result = evaluator(program)
    score = float(getattr(result, "score", result) or 0.0)
    # dspy.Evaluate reports percentages for float metrics in 0..1.
    if score > 1.0:
        score /= 100.0
    summary = EvaluationSummary(label=label, score=round(score, 4), n=len(devset))
    logger.info("%s", summary)
    return summary


def compare(
    baseline: dspy.Module,
    candidate: dspy.Module,
    devset: list[dspy.Example],
    metric: Callable[..., Any],
    settings: Settings | None = None,
) -> ComparisonResult:
    """Score both programs on the same examples with the same metric."""
    return ComparisonResult(
        baseline=evaluate_program(
            baseline, devset, metric, label="baseline", settings=settings
        ),
        candidate=evaluate_program(
            candidate, devset, metric, label="optimized", settings=settings
        ),
    )
