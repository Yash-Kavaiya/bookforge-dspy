"""Prompt optimization: compile BookForge's modules against its own metrics.

Why this exists
---------------
In the ADK implementation, prompt quality was a matter of opinion — someone
edited ``prompts.py``, ran a book, and eyeballed the PDF. There was no way to
tell whether an edit helped, and no way to recover the reasoning behind a
prompt six months later.

Here the instruction text is a parameter fitted against a measurable objective.
``optimize`` runs an optimizer, scores the result against the *same* verifier
production uses, and — this is the part that matters — **refuses to save an
artifact that did not beat the baseline.** An optimizer run that makes things
worse should leave the system exactly as it was.

Optimizer selection
-------------------
``gepa``      Reflective prompt evolution. Reads the *textual feedback* from our
              metrics (which compiler error, which invented filename) and
              rewrites instructions to prevent that failure. Best fit here,
              because our metrics produce exactly that kind of diagnostic.
``miprov2``   Bayesian search over instructions and few-shot demonstrations.
              Strong when you have enough examples to bootstrap demos.
``bootstrap`` Few-shot demonstrations via random search. Cheapest, no
              instruction rewriting.
``simba``     Stochastic introspective mini-batch ascent; good middle ground.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dspy

from bookforge.config import Settings, get_settings
from bookforge.dspy_runtime.artifacts import save_program
from bookforge.dspy_runtime.lm import USAGE, LMRouter, build_lm
from bookforge.dspy_runtime.modules import ChapterProgram, TranscriptAnalyst
from bookforge.evaluation.datasets import collect, split
from bookforge.evaluation.harness import ComparisonResult, compare
from bookforge.metrics.composite import get_metric

logger = logging.getLogger(__name__)

#: Which module, dataset and metric make up each optimization target.
TARGETS: dict[str, dict[str, str]] = {
    "analyst": {"dataset": "analysis", "metric": "analysis", "artifact": "analyst"},
    "chapter": {"dataset": "chapter", "metric": "chapter", "artifact": "chapter"},
}


def build_target_program(
    target: str, settings: Settings, router: LMRouter
) -> dspy.Module:
    """Construct the module to optimize, with its production LMs attached."""
    if target == "analyst":
        program = TranscriptAnalyst(settings)
        program.set_lm(router.get("analyst"))
        return program
    if target == "chapter":
        # The semantic critic is disabled during optimization: it is a
        # judgement call that adds variance to the reward and cost to every
        # rollout. We optimize against the deterministic verifier, which is
        # reproducible, and keep the critic for production runs.
        program = ChapterProgram(settings, use_semantic_critic=False)
        program.writer.set_lm(router.get("writer"))
        program.refiner.set_lm(router.get("writer"))
        return program
    raise KeyError(f"unknown optimization target {target!r}; expected {sorted(TARGETS)}")


def build_optimizer(
    name: str,
    metric: Callable[..., Any],
    settings: Settings,
    router: LMRouter,
) -> Any:
    """Instantiate the requested optimizer, configured for this project."""
    name = name.lower()

    if name == "gepa":
        return dspy.GEPA(
            metric=metric,
            auto=settings.optimizer_budget,
            reflection_lm=build_lm("reflection", settings, temperature=1.0, max_tokens=32000),
            num_threads=settings.num_threads,
            track_stats=True,
            # Our metrics score in 0..1 and a perfect chapter is genuinely
            # reachable, so let GEPA stop refining examples that reach it.
            failure_score=0.0,
            perfect_score=1.0,
            skip_perfect_score=True,
            use_mlflow=settings.enable_mlflow,
        )

    if name == "miprov2":
        return dspy.MIPROv2(
            metric=metric,
            auto=settings.optimizer_budget,
            prompt_model=build_lm("reflection", settings, temperature=1.0),
            task_model=router.get("writer"),
            num_threads=settings.num_threads,
            verbose=False,
        )

    if name == "simba":
        return dspy.SIMBA(
            metric=metric,
            num_threads=settings.num_threads,
            max_steps=6,
            max_demos=4,
        )

    if name == "bootstrap":
        return dspy.BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=3,
            max_labeled_demos=3,
            num_candidate_programs=6,
            num_threads=settings.num_threads,
        )

    raise KeyError(
        f"unknown optimizer {name!r}; expected gepa, miprov2, simba or bootstrap"
    )


def _compile(
    optimizer: Any,
    name: str,
    student: dspy.Module,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
) -> dspy.Module:
    """Call ``compile`` with the arguments each optimizer actually accepts."""
    if name in {"gepa", "miprov2"}:
        return optimizer.compile(student, trainset=trainset, valset=valset)
    # SIMBA and BootstrapFewShotWithRandomSearch take a trainset only.
    return optimizer.compile(student, trainset=trainset + valset)


def optimize(
    target: str,
    optimizer_name: str | None = None,
    settings: Settings | None = None,
    workspace_root: Path | None = None,
    max_examples: int | None = None,
    save: bool = True,
    force_save: bool = False,
) -> ComparisonResult:
    """Optimize one target and persist the result if it beat the baseline."""
    settings = settings or get_settings()
    optimizer_name = (optimizer_name or settings.optimizer).lower()
    if optimizer_name == "none":
        raise ValueError("optimizer is set to 'none'; choose gepa, miprov2, simba or bootstrap")
    if target not in TARGETS:
        raise KeyError(f"unknown target {target!r}; expected {sorted(TARGETS)}")

    spec = TARGETS[target]
    router = LMRouter(settings)
    USAGE.reset()

    examples = collect(spec["dataset"], workspace_root, settings)
    if max_examples:
        examples = examples[:max_examples]
    if len(examples) < 4:
        raise RuntimeError(
            f"only {len(examples)} example(s) available for '{target}'. Optimization "
            "needs at least 4 — process a few more videos first, since the training "
            "data is the pipeline's own output."
        )
    trainset, valset = split(examples)
    logger.info(
        "optimizing '%s' with %s: %d train / %d val examples",
        target, optimizer_name, len(trainset), len(valset),
    )

    scalar_metric = get_metric(spec["metric"], feedback=False)
    optimizer_metric = get_metric(
        spec["metric"], feedback=(optimizer_name == "gepa")
    )

    baseline_program = build_target_program(target, settings, router)
    student = build_target_program(target, settings, router)

    optimizer = build_optimizer(optimizer_name, optimizer_metric, settings, router)
    compiled = _compile(optimizer, optimizer_name, student, trainset, valset)

    # Both programs, same examples, same scalar metric — the only honest test.
    result = compare(baseline_program, compiled, valset, scalar_metric, settings)
    print(result.report())
    print(USAGE.summary())

    if not save:
        return result

    if not result.improved and not force_save:
        logger.warning(
            "not saving: optimized score %.4f did not beat baseline %.4f. "
            "Re-run with a larger budget or more examples, or pass force_save=True.",
            result.candidate.score, result.baseline.score,
        )
        return result

    save_program(
        compiled,
        settings.artifacts_dir_abs,
        spec["artifact"],
        optimizer=optimizer_name,
        train_score=result.candidate.score,
        val_score=result.candidate.score,
        baseline_score=result.baseline.score,
        trainset_size=len(trainset),
        valset_size=len(valset),
        notes=f"budget={settings.optimizer_budget}",
    )
    return result
