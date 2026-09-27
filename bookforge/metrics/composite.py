"""DSPy-facing metrics: scalar rewards and GEPA textual feedback.

Two shapes of the same judgement:

* ``scalar_*`` metrics return a float in 0..1 — what ``dspy.Evaluate``,
  MIPROv2, SIMBA and BootstrapFewShot consume.
* ``feedback_*`` metrics return ``dspy.Prediction(score=..., feedback=...)`` —
  what GEPA consumes. The feedback string is the whole reason GEPA works: its
  reflection model reads *why* a prediction lost points and rewrites the
  instruction to address that specific failure. Vague feedback produces vague
  instructions, so every message here names the offending value and the fix.

Both shapes share one implementation, so the number an optimizer maximizes is
the same number production ships against. A metric that drifts from the runtime
check is worse than no metric — it optimizes confidently for the wrong thing.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

import dspy

from bookforge.metrics.grounding import check_grounding
from bookforge.metrics.verifier import VerificationContext, verify_chapter
from bookforge.schemas import (
    AssetsManifest,
    ChapterAnalysis,
    coerce_analysis,
    coerce_assets_manifest,
)

logger = logging.getLogger(__name__)

MetricFn = Callable[..., Any]


# ---------------------------------------------------------------------------
# Analyst
# ---------------------------------------------------------------------------


def score_analysis(
    analysis: ChapterAnalysis | None, transcript: str
) -> tuple[float, str]:
    """Score a transcript analysis. Returns ``(score, feedback)``."""
    if analysis is None:
        return 0.0, (
            "No analysis was produced. The output must be a ChapterAnalysis object "
            "with chapter_title, summary, learning_objectives and at least one "
            "concept."
        )

    grounding = check_grounding(analysis, transcript)
    notes: list[str] = []

    # Richness: an analysis that extracts one thin concept from a 40-minute talk
    # is technically valid and practically useless.
    concept_component = min(len(analysis.concepts) / 5.0, 1.0)
    depth = [len(c.explanation.split()) for c in analysis.concepts] or [0]
    depth_component = min((sum(depth) / len(depth)) / 90.0, 1.0)

    evidence_present = sum(1 for c in analysis.concepts if c.evidence_timestamps)
    evidence_component = evidence_present / max(len(analysis.concepts), 1)

    structure_component = min(len(analysis.learning_objectives) / 4.0, 1.0)
    extras = (
        len(analysis.tables)
        + len(analysis.charts)
        + len(analysis.diagrams)
        + len(analysis.derivations)
    )
    extras_component = min(extras / 3.0, 1.0)
    exercises_component = min(len(analysis.exercises) / 5.0, 1.0)
    math_component = min(len(analysis.derivations) / 2.0, 1.0)

    richness = (
        0.28 * concept_component
        + 0.22 * depth_component
        + 0.14 * structure_component
        + 0.14 * extras_component
        + 0.12 * exercises_component
        + 0.10 * math_component
    )

    score = round(
        0.55 * grounding.score + 0.15 * evidence_component + 0.30 * richness, 4
    )

    if grounding.ungrounded or grounding.unsupported_concepts:
        notes.append(grounding.feedback())
    if len(analysis.concepts) < 3:
        notes.append(
            f"Only {len(analysis.concepts)} concept(s) extracted. A lecture window "
            "almost always develops at least three distinct ideas; look for the "
            "points the speaker returns to or spends the most time on."
        )
    # 3-8 sentences is roughly 45-140 words. Complaining above that floor would
    # teach the optimizer to pad, which is the opposite of what we want.
    if sum(depth) / len(depth) < 45:
        notes.append(
            "Concept explanations average "
            f"{sum(depth) // len(depth)} words, which is below the 3-8 sentence "
            "target. Each explanation should carry the speaker's actual reasoning, "
            "not restate the heading."
        )
    if evidence_component < 0.8:
        notes.append(
            f"Only {evidence_present}/{len(analysis.concepts)} concepts carry "
            "evidence_timestamps. Populate them from the [HH:MM:SS] markers so every "
            "claim can be traced back to the moment it was spoken."
        )
    if not analysis.exercises:
        notes.append(
            "No exercises were produced. Generate 4–8 review questions with answers, "
            "each answerable from the concepts alone. Empty exercises are not OK."
        )
    if len(analysis.glossary) < 4:
        notes.append(
            f"Only {len(analysis.glossary)} glossary term(s). Extract 6–15 terms "
            "a newcomer needs, with one-sentence definitions from the talk."
        )
    if extras == 0:
        notes.append(
            "No tables, charts, diagrams or derivations were extracted. Prefer at "
            "least one comparison table, process flowchart, or worked formula "
            "when the talk compares options, walks a pipeline, or uses maths. "
            "Charts still require spoken numbers — do not invent values."
        )
    if not analysis.derivations and re.search(
        r"quantiz|equation|formula|scale factor|loss|gradient|probabilit|"
        r"entropy|matrix|vector|derivative|integral|bit-?width|softmax",
        " ".join(
            [analysis.chapter_title, analysis.summary]
            + [c.heading + " " + c.explanation for c in analysis.concepts]
        ),
        re.I,
    ):
        notes.append(
            "This chapter looks mathematical but has no derivations. Add "
            "MathDerivation entries with full step-by-step algebra ending in "
            "formula_latex so the writer can produce derivationbox blocks."
        )

    feedback = " ".join(notes) if notes else (
        "Grounded, detailed and well-structured analysis. Keep this level of "
        "evidence linking and explanatory depth."
    )
    return score, feedback


def _example_field(example: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(example, name, None)
        if value is None and isinstance(example, dict):
            value = example.get(name)
        if value is not None:
            return value
    return default


def _prediction_analysis(prediction: Any) -> ChapterAnalysis | None:
    raw = getattr(prediction, "analysis", None)
    if raw is None:
        return None
    try:
        return coerce_analysis(raw)
    except Exception as exc:
        logger.debug("could not coerce predicted analysis: %s", exc)
        return None


def scalar_analysis_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Float reward for an analyst prediction. For Evaluate/MIPROv2/SIMBA."""
    transcript = _example_field(example, "transcript", "transcript_chunk", default="")
    score, _ = score_analysis(_prediction_analysis(prediction), str(transcript))
    # In bootstrapping mode (trace is not None) DSPy wants a pass/fail decision.
    if trace is not None:
        return score >= 0.7
    return score


def feedback_analysis_metric(
    example: Any,
    prediction: Any,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> dspy.Prediction:
    """GEPA metric for the analyst: score plus the reason it lost points."""
    transcript = _example_field(example, "transcript", "transcript_chunk", default="")
    score, feedback = score_analysis(_prediction_analysis(prediction), str(transcript))
    return dspy.Prediction(score=score, feedback=feedback)


# ---------------------------------------------------------------------------
# Chapter writer / refiner
# ---------------------------------------------------------------------------


def score_chapter(
    chapter_tex: str,
    analysis: ChapterAnalysis,
    manifest: AssetsManifest,
    context: VerificationContext | None = None,
) -> tuple[float, str]:
    """Score a written chapter with the same verifier production uses."""
    ctx = context or VerificationContext(analysis=analysis, manifest=manifest)
    verdict = verify_chapter(chapter_tex, ctx)

    if verdict.score >= 0.95 and not verdict.defects:
        return verdict.score, (
            "Chapter compiles cleanly, covers the whole analysis, references every "
            "asset correctly and states no unsupported figures."
        )

    lead = (
        "The chapter compiled." if verdict.compiles
        else "The chapter FAILED to compile — nothing else matters until that is fixed."
    )
    return verdict.score, f"{lead}\n{verdict.feedback_text()}"


def _build_context(example: Any) -> VerificationContext | None:
    """Reconstruct a verification context from a dataset example."""
    try:
        analysis = coerce_analysis(_example_field(example, "analysis"))
        manifest = coerce_assets_manifest(_example_field(example, "assets_manifest_obj", "manifest"))
    except Exception as exc:
        logger.debug("example lacks a usable verification context: %s", exc)
        return None
    return VerificationContext(
        analysis=analysis,
        manifest=manifest,
        workspace=_example_field(example, "workspace"),
        build_dir=_example_field(example, "build_dir"),
    )


def scalar_chapter_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Float reward for a chapter prediction."""
    context = _build_context(example)
    if context is None:
        return 0.0
    tex = getattr(prediction, "chapter_tex", "") or ""
    score, _ = score_chapter(tex, context.analysis, context.manifest, context)
    if trace is not None:
        return score >= 0.8
    return score


def feedback_chapter_metric(
    example: Any,
    prediction: Any,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> dspy.Prediction:
    """GEPA metric for the writer/refiner: score plus the exact defect list.

    The feedback is the verifier's own defect list, which means the reflection
    model sees precisely what a failing chapter did wrong — a compiler error at
    a named line, an invented filename, a number absent from the analysis — and
    can write an instruction that prevents that specific class of mistake.
    """
    context = _build_context(example)
    if context is None:
        return dspy.Prediction(
            score=0.0,
            feedback=(
                "The example is missing its analysis or assets manifest, so the "
                "chapter could not be verified."
            ),
        )
    tex = getattr(prediction, "chapter_tex", "") or ""
    score, feedback = score_chapter(tex, context.analysis, context.manifest, context)
    return dspy.Prediction(score=score, feedback=feedback)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCALAR_METRICS: dict[str, MetricFn] = {
    "analysis": scalar_analysis_metric,
    "chapter": scalar_chapter_metric,
}

FEEDBACK_METRICS: dict[str, MetricFn] = {
    "analysis": feedback_analysis_metric,
    "chapter": feedback_chapter_metric,
}


def get_metric(target: str, feedback: bool = False) -> MetricFn:
    """Look up the metric for an optimization target."""
    registry = FEEDBACK_METRICS if feedback else SCALAR_METRICS
    if target not in registry:
        raise KeyError(f"unknown metric target {target!r}; expected {sorted(registry)}")
    return registry[target]
