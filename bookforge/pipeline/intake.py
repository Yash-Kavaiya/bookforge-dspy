"""Stage 1 — whatever the user pointed at becomes a resumable manifest.

Deterministic: no LLM decides what the material contains. Which *kind* of source
this is gets decided in ``bookforge.ingest``; intake just runs it and records
the result.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from bookforge.config import Settings
from bookforge.ingest import resolve_source
from bookforge.ingest.base import Discovery
from bookforge.pipeline.progress import Reporter
from bookforge.schemas import ChannelManifest
from bookforge.workspace import Workspace, slugify

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s\"']*(?:youtube\.com|youtu\.be)[^\s\"']*")


def extract_channel_url(text: str) -> str | None:
    """Pull a YouTube URL out of free-form text, trimming chat punctuation."""
    match = _URL_RE.search(text or "")
    if not match:
        return None
    return re.sub(r"[.,;/\s]+$", "", match.group(0))


def run_intake(
    spec: str | Path | list[str | Path],
    settings: Settings,
    reporter: Reporter,
    title: str = "",
) -> tuple[Workspace, ChannelManifest]:
    """Resolve the input and build (or resume) its manifest."""
    if isinstance(spec, str):
        # A pasted chat message may wrap the URL in prose.
        spec = extract_channel_url(spec) or spec

    source = resolve_source(spec, settings, title=title)
    reporter.stage(f"Resolving {_describe(spec)} ...")

    discovery: Discovery = source.discover()
    for warning in discovery.warnings:
        reporter.warning(warning)

    if not discovery.items:
        raise RuntimeError(
            f"No usable recordings found in {_describe(spec)}. "
            + (discovery.warnings[0] if discovery.warnings else "")
        )

    slug = slugify(discovery.title)
    workspace = Workspace(settings.workspace_root_abs, slug)

    if workspace.manifest_exists():
        manifest = workspace.load_manifest()
        reporter.info(
            f"Resuming existing manifest: {len(manifest.videos)} recording(s) known."
        )
        added = _merge_new_items(workspace, manifest, discovery)
        if added:
            manifest = workspace.load_manifest()
            reporter.info(f"Added {added} new recording(s) since the last run.")
    else:
        manifest = workspace.init_manifest(discovery.source_line, discovery.title, slug)
        ordered = sorted(discovery.items, key=lambda item: item.order_key)
        manifest.videos = [
            item.to_record(chapter_number=index)
            for index, item in enumerate(ordered, start=1)
        ]
        workspace.save_manifest(manifest)

    pending = workspace.pending_videos()
    reporter.stage(
        f"Intake complete: {len(manifest.videos)} recording(s) "
        f"({len(pending)} pending) for '{manifest.channel_title or discovery.title}'."
    )
    return workspace, manifest


def _merge_new_items(
    workspace: Workspace, manifest: ChannelManifest, discovery: Discovery
) -> int:
    """Append recordings that appeared since the manifest was created.

    Existing chapter numbers are never renumbered — that would orphan the
    chapter directories already on disk and silently break resume.
    """
    known = {video.video_id for video in manifest.videos}
    fresh = [item for item in discovery.items if item.video_id not in known]
    if not fresh:
        return 0
    fresh.sort(key=lambda item: item.order_key)
    next_number = max((v.chapter_number for v in manifest.videos), default=0) + 1
    for offset, item in enumerate(fresh):
        manifest.videos.append(item.to_record(chapter_number=next_number + offset))
    workspace.save_manifest(manifest)
    return len(fresh)


def _describe(spec) -> str:
    if isinstance(spec, list):
        return ", ".join(str(item) for item in spec[:3]) + (
            f" (+{len(spec) - 3} more)" if len(spec) > 3 else ""
        )
    return str(spec)
