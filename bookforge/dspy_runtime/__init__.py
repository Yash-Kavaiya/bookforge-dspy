"""The DSPy runtime: signatures, modules, LM routing and compiled artifacts.

Everything the ADK implementation expressed as agents and prompt strings lives
here as typed signatures and composable ``dspy.Module`` objects — which is what
makes the pipeline optimizable rather than merely configurable.
"""

from bookforge.dspy_runtime.artifacts import (
    ArtifactMetadata,
    describe_artifacts,
    load_program,
    save_program,
    signature_fingerprint,
)
from bookforge.dspy_runtime.chunking import (
    TranscriptChunk,
    chunk_segments,
    window_around,
)
from bookforge.dspy_runtime.lm import USAGE, LMRouter, build_lm, configure_dspy
from bookforge.dspy_runtime.modules import (
    ChapterProgram,
    ChapterRefiner,
    ChapterResult,
    ChapterWriter,
    FrameCaptioner,
    SemanticCritic,
    TranscriptAnalyst,
)
from bookforge.dspy_runtime.program import BookForgeProgram

__all__ = [
    "USAGE",
    "ArtifactMetadata",
    "BookForgeProgram",
    "ChapterProgram",
    "ChapterRefiner",
    "ChapterResult",
    "ChapterWriter",
    "FrameCaptioner",
    "LMRouter",
    "SemanticCritic",
    "TranscriptAnalyst",
    "TranscriptChunk",
    "build_lm",
    "chunk_segments",
    "configure_dspy",
    "describe_artifacts",
    "load_program",
    "save_program",
    "signature_fingerprint",
    "window_around",
]
