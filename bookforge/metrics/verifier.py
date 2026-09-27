"""The chapter verifier — one deterministic function that scores a candidate.

This is the centre of gravity for accuracy in BookForge. It answers a single
question about a chapter draft: *how good is this, concretely?* — by compiling
it, checking its structure, and grounding its claims. Everything else is built
on the answer:

* the refinement loop uses it to keep the best candidate and to stop early;
* the optimizers use it as the reward signal;
* the runtime guards use it to refuse to ship a chapter below threshold;
* the critic LM receives its findings so it does not waste effort re-deriving
  what a regex already knows.

Because it is deterministic, the same draft always scores the same. That is
what makes the score comparable across refinement iterations — the property the
original LoopAgent lacked, which is why it could return a *worse* chapter than
the one it started with and call it done.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from bookforge.config import Settings, get_settings
from bookforge.metrics.grounding import check_chapter_grounding
from bookforge.metrics.structure import check_structure
from bookforge.schemas import (
    AssetsManifest,
    ChapterAnalysis,
    ChapterVerdict,
    Defect,
)
from bookforge.tools import latex as latex_tools
from bookforge.workspace import Workspace

logger = logging.getLogger(__name__)

# Weights sum to 1.0. Compilation dominates because an uncompilable chapter has
# no value at all, whatever else is right about it.
WEIGHT_COMPILE = 0.40
WEIGHT_STRUCTURE = 0.30
WEIGHT_GROUNDING = 0.25
WEIGHT_POLISH = 0.05


@dataclass
class VerificationContext:
    """Everything the verifier needs that is not the draft itself."""

    analysis: ChapterAnalysis
    manifest: AssetsManifest
    workspace: Workspace | None = None
    build_dir: Path | None = None
    settings: Settings | None = None

    def resolved_settings(self) -> Settings:
        return self.settings or get_settings()

    def existing_paths(self) -> set[str] | None:
        """Workspace-relative paths that actually exist on disk."""
        if self.workspace is None:
            return None
        root = self.workspace.root
        return {
            path for path in self.manifest.allowed_paths if (root / path).exists()
        }


def verify_chapter(chapter_tex: str, context: VerificationContext) -> ChapterVerdict:
    """Score one chapter draft. Never raises — a broken draft scores zero."""
    settings = context.resolved_settings()
    verdict = ChapterVerdict()
    tex = latex_tools.sanitize_chapter_tex(chapter_tex or "")

    if not tex.strip():
        verdict.defects.append(
            Defect(
                severity="critical",
                message="The chapter body is empty.",
                fix="Emit the full LaTeX chapter body, starting with \\chapter{...}.",
            )
        )
        return verdict

    # --- structure -------------------------------------------------------
    structure = check_structure(
        tex, context.analysis, context.manifest, context.existing_paths()
    )
    verdict.missing_assets = structure.missing_assets
    verdict.invented_assets = structure.invented_assets
    verdict.missing_sections = structure.missing_sections
    verdict.defects.extend(structure.defects())

    # --- grounding -------------------------------------------------------
    # Manifest captions are content BookForge generated and handed to the
    # writer, so numbers they contain (frame timestamps, for instance) are not
    # inventions by the writer.
    manifest_text = " ".join(
        [figure.caption for figure in context.manifest.figures]
        + [table.caption for table in context.manifest.tables]
    )
    grounding = check_chapter_grounding(tex, context.analysis, manifest_text)
    verdict.ungrounded_numbers = grounding.ungrounded
    if grounding.ungrounded:
        verdict.defects.append(
            Defect(
                severity="critical",
                message=(
                    "The chapter states numbers that do not appear in the analysis: "
                    f"{', '.join(grounding.ungrounded[:10])}."
                ),
                fix=(
                    "Delete each unsupported figure or replace it with the value the "
                    "analysis records. Never estimate a number to complete a sentence."
                ),
            )
        )

    # --- compilation -----------------------------------------------------
    compile_result = _compile(tex, context, settings)
    verdict.compiles = compile_result.ok
    verdict.compile_skipped = compile_result.skipped
    verdict.compile_log = compile_result.errors or compile_result.log_tail[-800:]

    if not compile_result.ok and not compile_result.skipped:
        verdict.defects.insert(
            0,
            Defect(
                severity="critical",
                message=f"pdflatex failed:\n{compile_result.errors[:1200]}",
                fix=(
                    "Fix the LaTeX error at the reported line. Common causes: an "
                    "unescaped & % # _ or $ in prose, an unclosed environment, or a "
                    "TikZ node using an undefined style."
                ),
            ),
        )
    elif compile_result.skipped:
        # No pdflatex available. Do not claim verification we did not perform.
        logger.debug("compile skipped: %s", compile_result.errors)

    if compile_result.overfull_boxes:
        verdict.defects.append(
            Defect(
                severity="minor",
                message=f"{compile_result.overfull_boxes} overfull hbox warnings — text runs into the margin.",
                fix=(
                    "Break long identifiers and URLs with \\texttt{} plus explicit "
                    "hyphenation points, or shorten the offending lines."
                ),
            )
        )
    if compile_result.undefined_references:
        verdict.defects.append(
            Defect(
                severity="major",
                message=(
                    "Undefined cross-references: "
                    f"{', '.join(compile_result.undefined_references[:6])}."
                ),
                fix="Define the missing \\label or remove the \\ref.",
            )
        )

    # --- score -----------------------------------------------------------
    compile_component = 1.0 if compile_result.ok else 0.0
    if compile_result.skipped:
        # Cannot verify by execution; fall back to structural confidence rather
        # than rewarding or punishing something we did not measure.
        compile_component = 0.5 * structure.score

    polish = 1.0
    if compile_result.overfull_boxes:
        polish -= min(compile_result.overfull_boxes / 10.0, 0.6)
    if structure.unreferenced_floats:
        polish -= min(len(structure.unreferenced_floats) / 8.0, 0.4)
    polish = max(0.0, polish)

    verdict.score = round(
        WEIGHT_COMPILE * compile_component
        + WEIGHT_STRUCTURE * structure.score
        + WEIGHT_GROUNDING * grounding.score
        + WEIGHT_POLISH * polish,
        4,
    )
    return verdict


def _compile(
    tex: str, context: VerificationContext, settings: Settings
) -> latex_tools.CompileResult:
    """Compile the draft standalone with the shared preamble."""
    if not settings.compile_latex:
        return latex_tools.CompileResult(
            ok=False, errors="LaTeX compilation disabled by configuration", skipped=True
        )
    if not latex_tools.pdflatex_available():
        return latex_tools.CompileResult(
            ok=False, errors="pdflatex not found on PATH", skipped=True
        )
    if context.build_dir is None or context.workspace is None:
        return latex_tools.CompileResult(
            ok=False, errors="no build directory configured for verification", skipped=True
        )

    build_dir = Path(context.build_dir)
    wrapper = build_dir / "verify.tex"
    # Verify against the very preamble the book will use, or a chapter can
    # pass QA and then fail in the assembled book.
    from bookforge.design import build_preamble, get_theme

    preamble = build_preamble(get_theme(getattr(settings, "book_theme", "textbook")))
    Workspace.write_text(wrapper, latex_tools.chapter_wrapper(preamble, tex))
    # cwd = workspace root so relative asset paths resolve exactly as they will
    # in the assembled book.
    return latex_tools.compile_tex(
        wrapper,
        build_dir,
        passes=1,
        timeout_sec=settings.latex_timeout_sec,
        cwd=context.workspace.root,
    )


def merge_semantic_defects(
    verdict: ChapterVerdict, semantic: list[Defect]
) -> ChapterVerdict:
    """Fold an LM critic's findings into a mechanical verdict.

    Semantic defects adjust the score modestly — they are judgement, not
    measurement, so they refine the ranking rather than dominate it. A critical
    semantic finding still blocks acceptance via ``is_acceptable``.
    """
    if not semantic:
        return verdict
    verdict.defects.extend(semantic)
    penalty = sum(
        {"critical": 0.12, "major": 0.05, "minor": 0.01}.get(d.severity, 0.01)
        for d in semantic
    )
    verdict.score = round(max(0.0, verdict.score - min(penalty, 0.35)), 4)
    return verdict
