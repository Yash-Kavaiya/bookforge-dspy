"""DSPy modules — one per LLM stage of the pipeline.

Each is an independently optimizable ``dspy.Module``; ``ChapterProgram``
composes the writing stages into the verified refinement loop.
"""

from bookforge.dspy_runtime.modules.analyst import TranscriptAnalyst
from bookforge.dspy_runtime.modules.captioner import FrameCaptioner
from bookforge.dspy_runtime.modules.chapter import (
    ChapterAttemptLog,
    ChapterProgram,
    ChapterResult,
)
from bookforge.dspy_runtime.modules.critic import SemanticCritic
from bookforge.dspy_runtime.modules.writer import ChapterRefiner, ChapterWriter

__all__ = [
    "ChapterAttemptLog",
    "ChapterProgram",
    "ChapterRefiner",
    "ChapterResult",
    "ChapterWriter",
    "FrameCaptioner",
    "SemanticCritic",
    "TranscriptAnalyst",
]
