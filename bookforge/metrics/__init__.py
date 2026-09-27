"""Accuracy measurement for BookForge.

Three layers, all deterministic:

``grounding``   Are the claims traceable to the source? (analysis vs transcript,
                chapter vs analysis)
``structure``   Does the chapter satisfy its contract? (sections, assets,
                cross-references, coverage, length)
``verifier``    The composite judgement — compile it, check it, score it — used
                identically by the refinement loop, the runtime guards and the
                optimizers.

``composite`` adapts that judgement into the two shapes DSPy optimizers expect.
"""

from bookforge.metrics.composite import (
    feedback_analysis_metric,
    feedback_chapter_metric,
    get_metric,
    scalar_analysis_metric,
    scalar_chapter_metric,
    score_analysis,
    score_chapter,
)
from bookforge.metrics.grounding import (
    GroundingReport,
    check_chapter_grounding,
    check_grounding,
)
from bookforge.metrics.structure import StructureReport, check_structure
from bookforge.metrics.verifier import (
    VerificationContext,
    merge_semantic_defects,
    verify_chapter,
)

__all__ = [
    "GroundingReport",
    "StructureReport",
    "VerificationContext",
    "check_chapter_grounding",
    "check_grounding",
    "check_structure",
    "feedback_analysis_metric",
    "feedback_chapter_metric",
    "get_metric",
    "merge_semantic_defects",
    "scalar_analysis_metric",
    "scalar_chapter_metric",
    "score_analysis",
    "score_chapter",
    "verify_chapter",
]
