"""Stage 3 — turn the analysis into real files on disk.

Deterministic by design: charts are rendered by matplotlib from the analyst's
validated numbers, never drawn by an LLM, and tables become booktabs fragments.
The one LLM touch is frame captioning, which happens *here* rather than during
analysis — see ``modules/captioner.py`` for why that ordering is the whole
point.

The manifest this stage publishes is a contract: the writer may reference these
paths and no others, and the verifier enforces that.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from bookforge.config import Settings
from bookforge.dspy_runtime.modules.captioner import FrameCaptioner
from bookforge.schemas import (
    AssetsManifest,
    ChapterAnalysis,
    FigureAsset,
    FrameAsset,
    TableAsset,
    TranscriptSegment,
    VideoRecord,
)
from bookforge.tools import frames as frame_tools
from bookforge.tools import latex as latex_tools
from bookforge.workspace import Workspace

logger = logging.getLogger(__name__)


def build_assets(
    record: VideoRecord,
    analysis: ChapterAnalysis,
    frames: Sequence[FrameAsset],
    segments: Sequence[TranscriptSegment],
    workspace: Workspace,
    settings: Settings,
    captioner: FrameCaptioner | None = None,
    visual_insights: str = "",
) -> AssetsManifest:
    """Render tables, charts and captioned figures for one chapter."""
    chapter_dir = workspace.chapter_dir(record)
    number = record.chapter_number

    tables = _render_tables(analysis, chapter_dir, workspace, number)
    figures = _render_charts(analysis, chapter_dir, workspace, number)
    figures += _render_frames(
        record,
        analysis,
        frames,
        segments,
        chapter_dir,
        workspace,
        settings,
        captioner,
        visual_insights=visual_insights,
    )

    manifest = AssetsManifest(
        chapter_slug=chapter_dir.name,
        chapter_dir=workspace.rel(chapter_dir),
        figures=figures,
        tables=tables,
    )
    Workspace.write_text(chapter_dir / "assets.json", manifest.model_dump_json(indent=2))
    logger.info(
        "chapter %02d assets: %d figures, %d tables",
        number, len(figures), len(tables),
    )
    return manifest


def _render_tables(
    analysis: ChapterAnalysis,
    chapter_dir,
    workspace: Workspace,
    number: int,
) -> list[TableAsset]:
    tables: list[TableAsset] = []
    for index, spec in enumerate(analysis.tables, start=1):
        label = f"tab:ch{number:02d}-{index}"
        try:
            fragment = latex_tools.render_table_fragment(
                spec, chapter_dir / "tables" / f"table_{index}.tex", label=label
            )
        except Exception as exc:
            logger.warning("table %d failed to render for chapter %d: %s", index, number, exc)
            continue
        tables.append(
            TableAsset(
                tex_file=workspace.rel(fragment), caption=spec.caption, label=label
            )
        )
    return tables


def _render_charts(
    analysis: ChapterAnalysis,
    chapter_dir,
    workspace: Workspace,
    number: int,
) -> list[FigureAsset]:
    figures: list[FigureAsset] = []
    for index, spec in enumerate(analysis.charts, start=1):
        try:
            pdf = latex_tools.render_chart(
                spec, chapter_dir / "figures" / f"chart_{index}.pdf"
            )
        except Exception as exc:
            logger.warning("chart %d failed to render for chapter %d: %s", index, number, exc)
            continue
        figures.append(
            FigureAsset(
                filename=workspace.rel(pdf),
                kind="chart",
                caption=spec.caption,
                label=f"fig:ch{number:02d}-chart{index}",
            )
        )
    return figures


def _render_frames(
    record: VideoRecord,
    analysis: ChapterAnalysis,
    frames: Sequence[FrameAsset],
    segments: Sequence[TranscriptSegment],
    chapter_dir,
    workspace: Workspace,
    settings: Settings,
    captioner: FrameCaptioner | None,
    visual_insights: str = "",
) -> list[FigureAsset]:
    """Curate the best frames, caption them from the transcript, drop the rest."""
    curated = frame_tools.curate_frames(
        list(frames), chapter_dir / "figures", settings.max_frames_per_chapter
    )
    if not curated:
        return []

    captions: dict[str, str] = {}
    dropped: set[str] = set()

    if captioner is not None:
        prediction = captioner(
            video_title=record.title,
            analysis=analysis,
            frames=curated,
            segments=list(segments),
            visual_insights=visual_insights,
        )
        captions = {item.frame_file: item.caption for item in prediction.captions}
        dropped = set(prediction.dropped)

    figures: list[FigureAsset] = []
    position = 0
    for frame in curated:
        if frame.file in dropped:
            # Uninformative frame: remove the copy so the chapter directory
            # mirrors the manifest exactly.
            stale = chapter_dir / "figures" / frame.file
            stale.unlink(missing_ok=True)
            continue
        position += 1
        caption = captions.get(frame.file) or (
            f"Slide shown at {_mm_ss(frame.timestamp_sec)} in '{record.title}'."
        )
        if settings.figure_time_footnotes:
            caption = frame_tools.with_time_footnote(caption, frame.timestamp_sec)
        figures.append(
            FigureAsset(
                filename=workspace.rel(chapter_dir / "figures" / frame.file),
                kind="frame",
                caption=caption,
                label=f"fig:ch{record.chapter_number:02d}-frame{position}",
                timestamp_sec=frame.timestamp_sec,
            )
        )
    return figures


def _mm_ss(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"
