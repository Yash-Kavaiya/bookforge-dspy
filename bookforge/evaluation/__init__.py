"""Evaluation and optimization.

BookForge's metrics are reference-free, so every video the pipeline processes
becomes a training example. ``datasets`` harvests them, ``harness`` measures a
program, and ``optimize`` compiles better instructions — saving them only when
they demonstrably beat the baseline.
"""

from bookforge.evaluation.datasets import (
    build_analysis_examples,
    build_chapter_examples,
    build_full_analysis_examples,
    collect,
    discover_workspaces,
    split,
)
from bookforge.evaluation.harness import (
    ComparisonResult,
    EvaluationSummary,
    compare,
    evaluate_program,
)
from bookforge.evaluation.optimize import TARGETS, build_optimizer, optimize

__all__ = [
    "TARGETS",
    "ComparisonResult",
    "EvaluationSummary",
    "build_analysis_examples",
    "build_chapter_examples",
    "build_full_analysis_examples",
    "build_optimizer",
    "collect",
    "compare",
    "discover_workspaces",
    "evaluate_program",
    "optimize",
    "split",
]
