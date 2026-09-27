"""Stage 5 — assemble every completed chapter into book.pdf."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from bookforge.config import Settings
from bookforge.design import build_preamble, get_theme
from bookforge.pipeline.progress import Reporter
from bookforge.schemas import INCLUDABLE_STATUSES
from bookforge.tools import latex as latex_tools
from bookforge.workspace import Workspace

logger = logging.getLogger(__name__)


def compile_book(
    workspace: Workspace, settings: Settings, reporter: Reporter
) -> dict:
    """Build main.tex from the verified chapters and compile it."""
    manifest = workspace.load_manifest()
    completed = [v for v in manifest.ordered() if v.status in INCLUDABLE_STATUSES]
    failed = [v for v in manifest.videos if v.status == "failed"]

    if not completed:
        raise RuntimeError(
            "No chapter reached a shippable state, so there is nothing to compile. "
            "Check chapters/*/verdict.json for the defects that blocked each one."
        )

    reporter.stage(
        f"Compiling the book from {len(completed)} chapter(s); "
        f"{len(failed)} excluded as failed."
    )

    chapter_dirs: list[str] = []
    seen_labels: set[str] = set()

    for record in completed:
        chapter_dir = workspace.chapter_dir(record)
        chapter_tex = chapter_dir / "chapter.tex"
        if not chapter_tex.exists():
            reporter.warning(
                f"chapter.tex missing for '{record.title}' — excluded from the book."
            )
            continue

        body = latex_tools.sanitize_chapter_tex(Workspace.read_text(chapter_tex))
        # Labels are chapter-local until the book assembles them; a collision
        # would make every \ref resolve to the last definition, silently
        # pointing readers at the wrong figure.
        body, renamed = latex_tools.strip_duplicate_labels(body, seen_labels)
        if renamed:
            reporter.warning(
                f"renamed {len(renamed)} duplicate label(s) in chapter "
                f"{record.chapter_number:02d}: {', '.join(renamed[:5])}"
            )
        Workspace.write_text(chapter_tex, body)
        chapter_dirs.append(workspace.rel(chapter_dir))

    if not chapter_dirs:
        raise RuntimeError("No chapter.tex files were found on disk.")

    book_title = f"{manifest.channel_title or 'YouTube Channel'} — The Complete Book"
    author_line = "Compiled by BookForge (DSPy multi-stage pipeline)"

    theme_name = getattr(settings, "book_theme", "textbook")
    Workspace.write_text(
        workspace.root / "preamble.tex", build_preamble(get_theme(theme_name))
    )
    verified = sum(1 for v in completed if v.status == "verified")
    main_tex = Workspace.write_text(
        workspace.root / "main.tex",
        latex_tools.assemble_main_tex(
            book_title,
            author_line,
            chapter_dirs,
            subtitle=settings.book_subtitle or "Compiled from recorded talks",
            source_line=manifest.channel_url or manifest.channel_title,
            stats_line=(
                f"{len(chapter_dirs)} chapters, {verified} verified by compilation "
                f"and checked against their transcripts."
            ),
            theme_name=theme_name,
        ),
    )

    result = latex_tools.CompileResult(ok=False, errors="compilation disabled", skipped=True)
    if settings.compile_latex:
        result = latex_tools.compile_tex(
            main_tex,
            workspace.book_dir,
            passes=2,  # two passes so the table of contents resolves
            timeout_sec=max(settings.latex_timeout_sec, 300),
            cwd=workspace.root,
        )
        produced = workspace.book_dir / "main.pdf"
        if result.ok and produced.exists():
            produced.replace(workspace.book_dir / "book.pdf")

    pdf_exists = (workspace.book_dir / "book.pdf").exists()
    meta = {
        "title": book_title,
        "channel_url": manifest.channel_url,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chapters_included": len(chapter_dirs),
        "chapters_failed": [v.video_id for v in failed],
        "mean_quality_score": round(
            sum(v.qa_score for v in completed) / len(completed), 4
        ),
        "verified_chapters": sum(1 for v in completed if v.status == "verified"),
        "pdf": "book/book.pdf" if pdf_exists else None,
        "compile_ok": result.ok,
        "overfull_boxes": result.overfull_boxes,
        "undefined_references": result.undefined_references,
    }
    Workspace.write_text(
        workspace.book_dir / "book_manifest.json",
        json.dumps(meta, indent=2, ensure_ascii=False),
    )

    if settings.compile_latex and not result.ok:
        logger.error("book compilation failed:\n%s", result.errors)
        raise RuntimeError(f"pdflatex failed on the assembled book: {result.errors[:900]}")

    if result.undefined_references:
        reporter.warning(
            f"{len(result.undefined_references)} undefined cross-reference(s) in the book."
        )
    reporter.stage(
        f"Book compiled: {len(chapter_dirs)} chapters, "
        f"{meta['verified_chapters']} fully verified, "
        f"mean quality {meta['mean_quality_score']:.2f}. "
        f"PDF: {meta['pdf'] or '(compilation disabled)'}"
    )
    return meta


def book_pdf_path(workspace: Workspace) -> Path | None:
    candidate = workspace.book_dir / "book.pdf"
    return candidate if candidate.exists() else None
