"""Stage 4 — one video becomes one verified chapter.

Sequence: media -> analysis -> assets -> write/verify/refine -> persist.

Every stage records its status in the manifest before moving on, so an
interrupted run resumes at the boundary it reached rather than starting the
video over.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from bookforge.config import Settings
from bookforge.design.layout import declaration_block
from bookforge.dspy_runtime.program import BookForgeProgram
from bookforge.metrics.grounding import check_grounding
from bookforge.pipeline.assets import build_assets
from bookforge.pipeline.media import acquire_media
from bookforge.pipeline.progress import Reporter
from bookforge.schemas import ChapterAnalysis, VideoRecord
from bookforge.tools import latex as latex_tools
from bookforge.workspace import Workspace, slugify

logger = logging.getLogger(__name__)


@dataclass
class ChapterOutcome:
    video_id: str
    chapter_number: int
    title: str
    status: str
    score: float = 0.0
    grounding_score: float = 0.0
    iterations: int = 0
    compiled: bool = False
    error: str = ""


def produce_chapter(
    record: VideoRecord,
    workspace: Workspace,
    program: BookForgeProgram,
    settings: Settings,
    reporter: Reporter,
) -> ChapterOutcome:
    """Run the full per-video pipeline. Raises only on unrecoverable errors."""
    chapter_dir = workspace.chapter_dir(record)

    # --- media -------------------------------------------------------------
    reporter.checkpoint()
    reporter.stage(f"  [{record.chapter_number:02d}] acquiring media ...")
    bundle, segments = acquire_media(record, workspace, settings)
    workspace.update_video(record.video_id, status="media", error="")
    reporter.info(
        f"  [{record.chapter_number:02d}] transcript via {bundle.transcript_source} "
        f"({len(segments)} segments), {len(bundle.frames)} distinct frames."
    )

    # --- analysis ----------------------------------------------------------
    reporter.checkpoint()
    reporter.stage(f"  [{record.chapter_number:02d}] analysing transcript ...")
    analysis, grounding_score = _analyze(
        record,
        segments,
        program,
        settings,
        reporter,
        visual_insights=bundle.visual_insights,
    )
    Workspace.write_text(
        chapter_dir / "analysis.json", analysis.model_dump_json(indent=2)
    )
    workspace.update_video(
        record.video_id, status="analyzed", grounding_score=grounding_score
    )
    reporter.info(
        f"  [{record.chapter_number:02d}] '{analysis.chapter_title}': "
        f"{len(analysis.concepts)} concepts, {len(analysis.tables)} tables, "
        f"{len(analysis.charts)} charts, {len(analysis.diagrams)} diagrams "
        f"(grounding {grounding_score:.2f})."
    )

    # --- assets ------------------------------------------------------------
    reporter.checkpoint()
    reporter.stage(f"  [{record.chapter_number:02d}] rendering assets ...")
    manifest = build_assets(
        record,
        analysis,
        bundle.frames,
        segments,
        workspace,
        settings,
        captioner=program.captioner,
        visual_insights=bundle.visual_insights,
    )
    workspace.update_video(record.video_id, status="assets")

    # --- write, verify, refine --------------------------------------------
    reporter.checkpoint()
    reporter.stage(f"  [{record.chapter_number:02d}] writing and verifying ...")
    prediction = program.chapter(
        video_title=record.title,
        chapter_number=record.chapter_number,
        chapter_label_slug=slugify(record.title, max_len=32),
        analysis=analysis,
        manifest=manifest,
        workspace=workspace,
        build_dir=workspace.build_dir / f"qa_{record.video_id}",
    )
    result = prediction.result

    tex = latex_tools.sanitize_chapter_tex(prediction.chapter_tex)
    if not tex.strip():
        raise RuntimeError("the chapter pipeline produced an empty body")

    # Prepend the asset declarations. These are generated from the manifest,
    # never written by the model, so paths and captions cannot drift. Strip
    # first so a re-persist (or LM that echoed declarations) cannot double up.
    tex = declaration_block(manifest) + "\n" + latex_tools.strip_asset_declarations(tex)
    Workspace.write_text(chapter_dir / "chapter.tex", tex)
    _persist_verdict(chapter_dir, result)

    status = _status_for(result, settings)
    workspace.update_video(
        record.video_id,
        status=status,
        error="" if status != "failed" else "chapter quality below threshold",
        qa_score=result.score,
        qa_iterations=result.iterations,
        compiled=result.verdict.compiles,
    )
    reporter.info(f"  [{record.chapter_number:02d}] {result.summary()}")

    return ChapterOutcome(
        video_id=record.video_id,
        chapter_number=record.chapter_number,
        title=record.title,
        status=status,
        score=result.score,
        grounding_score=grounding_score,
        iterations=result.iterations,
        compiled=result.verdict.compiles,
    )


def _analyze(
    record: VideoRecord,
    segments,
    program: BookForgeProgram,
    settings: Settings,
    reporter: Reporter,
    visual_insights: str = "",
) -> tuple[ChapterAnalysis, float]:
    """Analyse the transcript, resampling once if grounding is poor.

    The guard is deliberately single-shot: one retry catches the common case
    where a model free-associated on a thin transcript, and refusing to spend
    more than that keeps a pathological video from consuming the run's budget.
    """
    spoken = "\n".join(seg.text for seg in segments)
    prediction = program.analyst(
        video_title=record.title,
        segments=segments,
        visual_insights=visual_insights,
    )
    analysis: ChapterAnalysis = prediction.analysis
    score: float = prediction.grounding_score

    if score >= settings.min_grounding_score:
        return analysis, score

    reporter.warning(
        f"  [{record.chapter_number:02d}] analysis grounding {score:.2f} is below "
        f"{settings.min_grounding_score:.2f} "
        f"({len(prediction.grounding_report.ungrounded)} unsupported numbers) — "
        "resampling once."
    )
    # A fresh rollout_id defeats the LM cache, so the retry is a real second
    # opinion rather than the same cached completion replayed. The module's own
    # LM must be swapped rather than dspy.context used, because a predictor with
    # an explicit lm ignores the context default.
    base_lm = program.analyst.get_lm()
    retry_lm = base_lm.copy(rollout_id=1, temperature=0.7)
    try:
        program.analyst.set_lm(retry_lm)
        retry = program.analyst(
            video_title=record.title,
            segments=segments,
            visual_insights=visual_insights,
        )
    except Exception as exc:
        logger.warning("analysis resample failed: %s", exc)
        return analysis, score
    finally:
        program.analyst.set_lm(base_lm)

    # Ground only against spoken text — never the visual briefing.
    retry_score = check_grounding(retry.analysis, spoken).score
    if retry_score > score:
        reporter.info(
            f"  [{record.chapter_number:02d}] resample improved grounding "
            f"{score:.2f} -> {retry_score:.2f}."
        )
        return retry.analysis, retry_score
    return analysis, score


def _status_for(result, settings: Settings) -> str:
    """Map a chapter result onto a manifest status.

    'verified' means the loop accepted it. 'written' means it is usable but
    imperfect. 'failed' means it should not appear in the book — shipping a
    chapter that does not compile would break the whole build.
    """
    # 'verified' requires a real compile. In draft mode the loop can accept a
    # chapter so it stops refining, but we never record verification we did not
    # actually perform.
    if result.accepted and result.verdict.compiles:
        return "verified"
    if result.score >= settings.qa_min_acceptable_score and not result.verdict.critical_defects:
        return "written"
    return "failed"


def _persist_verdict(chapter_dir: Path, result) -> None:
    """Write the QA audit trail beside the chapter."""
    payload = {
        "score": result.score,
        "accepted": result.accepted,
        "iterations": result.iterations,
        "compiles": result.verdict.compiles,
        "missing_assets": result.verdict.missing_assets,
        "invented_assets": result.verdict.invented_assets,
        "missing_sections": result.verdict.missing_sections,
        "ungrounded_numbers": result.verdict.ungrounded_numbers,
        "defects": [
            {"severity": d.severity, "message": d.message, "fix": d.fix}
            for d in result.verdict.defects
        ],
        "history": [asdict(entry) for entry in result.history],
    }
    Workspace.write_text(
        chapter_dir / "verdict.json", json.dumps(payload, indent=2, ensure_ascii=False)
    )
