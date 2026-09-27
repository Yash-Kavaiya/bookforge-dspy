"""Training and validation sets built from real pipeline output.

BookForge has no labelled corpus and does not need one: its metrics are
*reference-free*. Grounding compares an analysis against its own transcript;
structural checks compare a chapter against its own analysis and assets;
compilation is checked by running pdflatex. So an example needs only inputs and
the context to verify against — which means every video the pipeline has already
processed is a usable training example, and the dataset grows as the system runs.

That property is what makes optimization practical here. The alternative — a
human writing gold-standard LaTeX chapters — would cost more than the pipeline
saves.
"""

from __future__ import annotations

import logging
from pathlib import Path

import dspy

from bookforge.config import Settings, get_settings
from bookforge.dspy_runtime.chunking import chunk_segments
from bookforge.schemas import (
    AssetsManifest,
    ChapterAnalysis,
    MediaBundle,
)
from bookforge.tools import youtube as yt
from bookforge.workspace import Workspace

logger = logging.getLogger(__name__)


def discover_workspaces(root: Path) -> list[Workspace]:
    """Every channel workspace under ``root`` that has a manifest."""
    root = Path(root)
    if not root.exists():
        return []
    found: list[Workspace] = []
    for candidate in sorted(root.iterdir()):
        if candidate.is_dir() and (candidate / "manifest.json").exists():
            found.append(Workspace(root, candidate.name))
    return found


# ---------------------------------------------------------------------------
# Analyst dataset
# ---------------------------------------------------------------------------


def build_analysis_examples(
    workspace: Workspace,
    settings: Settings | None = None,
    per_video_chunks: int = 2,
) -> list[dspy.Example]:
    """One example per transcript window, for optimizing the analyst.

    Windows are sampled rather than exhausted: taking every window of every
    video would let a single long lecture dominate the objective, and the
    optimizer would learn that lecture's quirks instead of the general task.
    """
    settings = settings or get_settings()
    examples: list[dspy.Example] = []

    for record in workspace.load_manifest().ordered():
        media_path = workspace.video_dir(record.video_id) / "media.json"
        if not media_path.exists():
            continue
        try:
            bundle = MediaBundle.model_validate_json(
                media_path.read_text(encoding="utf-8")
            )
            segments = yt.parse_transcript(Path(bundle.transcript_path))
        except Exception as exc:
            logger.debug("skipping %s: %s", record.video_id, exc)
            continue
        if not segments:
            continue

        chunks = chunk_segments(
            segments,
            chunk_chars=settings.transcript_chunk_chars,
            overlap_chars=settings.transcript_chunk_overlap_chars,
            max_chunks=settings.max_transcript_chunks,
        )
        if not chunks:
            continue

        # Spread the sample across the talk: the opening and the middle exercise
        # different behaviour (framing vs technical depth).
        step = max(1, len(chunks) // max(1, per_video_chunks))
        for chunk in chunks[::step][:per_video_chunks]:
            examples.append(
                dspy.Example(
                    video_title=record.title,
                    chunk_index=chunk.index,
                    chunk_total=chunk.total,
                    transcript_chunk=chunk.text,
                    transcript=chunk.text,  # what the metric grounds against
                    video_id=record.video_id,
                ).with_inputs(
                    "video_title", "chunk_index", "chunk_total", "transcript_chunk"
                )
            )
    return examples


def build_full_analysis_examples(workspace: Workspace) -> list[dspy.Example]:
    """One example per video, for optimizing the end-to-end analyst module."""
    examples: list[dspy.Example] = []
    for record in workspace.load_manifest().ordered():
        media_path = workspace.video_dir(record.video_id) / "media.json"
        if not media_path.exists():
            continue
        try:
            bundle = MediaBundle.model_validate_json(
                media_path.read_text(encoding="utf-8")
            )
            segments = yt.parse_transcript(Path(bundle.transcript_path))
        except Exception:
            continue
        if not segments:
            continue
        examples.append(
            dspy.Example(
                video_title=record.title,
                segments=segments,
                transcript=yt.segments_to_text(segments),
                video_id=record.video_id,
            ).with_inputs("video_title", "segments")
        )
    return examples


# ---------------------------------------------------------------------------
# Chapter dataset
# ---------------------------------------------------------------------------


def build_chapter_examples(workspace: Workspace) -> list[dspy.Example]:
    """One example per analysed chapter, for optimizing the writer and refiner.

    Requires ``analysis.json`` and ``assets.json`` on disk — i.e. a video that
    reached the asset stage at least once. The workspace and build directory
    ride along on the example so the metric can compile candidates for real.
    """
    from bookforge.workspace import slugify

    examples: list[dspy.Example] = []
    for record in workspace.load_manifest().ordered():
        chapter_dir = workspace.chapter_dir(record)
        analysis_path = chapter_dir / "analysis.json"
        assets_path = chapter_dir / "assets.json"
        if not (analysis_path.exists() and assets_path.exists()):
            continue
        try:
            analysis = ChapterAnalysis.model_validate_json(
                analysis_path.read_text(encoding="utf-8")
            )
            manifest = AssetsManifest.model_validate_json(
                assets_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            logger.debug("skipping chapter %s: %s", record.video_id, exc)
            continue

        examples.append(
            dspy.Example(
                video_title=record.title,
                chapter_number=record.chapter_number,
                chapter_label_slug=slugify(record.title, max_len=32),
                analysis=analysis,
                assets_manifest=manifest.describe_for_prompt(),
                # Verification context for the metric:
                assets_manifest_obj=manifest,
                workspace=workspace,
                build_dir=workspace.build_dir / f"opt_{record.video_id}",
                video_id=record.video_id,
            ).with_inputs(
                "video_title",
                "chapter_number",
                "chapter_label_slug",
                "analysis",
                "assets_manifest",
            )
        )
    return examples


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def split(
    examples: list[dspy.Example], val_fraction: float = 0.4, seed: int = 7
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Deterministic train/validation split.

    Deterministic on purpose: comparing two optimizer runs is only meaningful
    when they were scored against the same validation set.
    """
    import random

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * (1 - val_fraction))) if len(shuffled) > 1 else 1
    return shuffled[:cut], shuffled[cut:]


def collect(
    target: str,
    workspace_root: Path | None = None,
    settings: Settings | None = None,
) -> list[dspy.Example]:
    """Gather every example of one kind across all workspaces."""
    settings = settings or get_settings()
    root = Path(workspace_root or settings.workspace_root_abs)
    workspaces = discover_workspaces(root)
    if not workspaces:
        raise RuntimeError(
            f"No processed channels found under {root}. Run the pipeline on at "
            "least one channel before optimizing — the training data is the "
            "output of previous runs."
        )

    builders = {
        "analysis": lambda ws: build_analysis_examples(ws, settings),
        "analysis_full": build_full_analysis_examples,
        "chapter": build_chapter_examples,
    }
    if target not in builders:
        raise KeyError(f"unknown dataset target {target!r}; expected {sorted(builders)}")

    examples: list[dspy.Example] = []
    for workspace in workspaces:
        examples.extend(builders[target](workspace))
    logger.info("collected %d '%s' examples from %d workspace(s)",
                len(examples), target, len(workspaces))
    return examples
