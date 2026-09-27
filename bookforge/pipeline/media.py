"""Stage 2 — per-recording media acquisition, dispatched by source.

Idempotent by construction: every step checks for its output file first, so this
can be prefetched for upcoming chapters while the current one is in the LLM
stages, and a killed run resumes for free.

Three paths, chosen by ``VideoRecord.source``:

``youtube``       download, captions or Whisper, frames
``local_video``   no download; a sidecar transcript if one exists, else Whisper
``transcript``    text only — no ffmpeg, no Whisper, no frames
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from bookforge.config import Settings
from bookforge.schemas import MediaBundle, TranscriptSegment, VideoRecord
from bookforge.tools import frames as frame_tools
from bookforge.tools import gemini_video
from bookforge.tools import transcripts as transcript_tools
from bookforge.tools import youtube as yt
from bookforge.workspace import Workspace

logger = logging.getLogger(__name__)

_ACQUIRE_LOCKS: dict[str, threading.Lock] = {}
_ACQUIRE_GUARD = threading.Lock()


def _acquire_lock(video_id: str) -> threading.Lock:
    """Serialize prefetch vs main-thread acquire for the same video."""
    with _ACQUIRE_GUARD:
        lock = _ACQUIRE_LOCKS.get(video_id)
        if lock is None:
            lock = threading.Lock()
            _ACQUIRE_LOCKS[video_id] = lock
        return lock


def media_settings_fingerprint(settings: Settings) -> str:
    """Stable key for media cache invalidation when extraction settings change."""
    return (
        f"fi={int(settings.frame_interval_sec)};"
        f"sc={float(settings.frame_scene_threshold):.3f};"
        f"ph={int(settings.frame_phash_threshold)};"
        f"res={int(settings.download_resolution)};"
        f"wh={settings.whisper_model}"
    )


def acquire_media(
    record: VideoRecord, workspace: Workspace, settings: Settings
) -> tuple[MediaBundle, list[TranscriptSegment]]:
    """Obtain transcript and frames for one recording.

    Returns the bundle and parsed segments. Segments rather than a text blob:
    chunking, frame captioning and grounding all need timestamps.
    """
    video_dir = workspace.video_dir(record.video_id)
    cached = video_dir / "media.json"
    fingerprint = media_settings_fingerprint(settings)

    # Prefetch and the main chapter thread can both call this for the same
    # video; serialize so transcript/video writes are not interleaved.
    with _acquire_lock(record.video_id):
        return _acquire_media_locked(
            record, workspace, settings, video_dir, cached, fingerprint
        )


def _acquire_media_locked(
    record: VideoRecord,
    workspace: Workspace,
    settings: Settings,
    video_dir: Path,
    cached: Path,
    fingerprint: str,
) -> tuple[MediaBundle, list[TranscriptSegment]]:
    if cached.exists():
        try:
            bundle = MediaBundle.model_validate_json(cached.read_text(encoding="utf-8"))
            if _cached_bundle_usable(bundle, record, fingerprint):
                segments = transcript_tools.load_segments(Path(bundle.transcript_path))
                if segments:
                    if not bundle.visual_insights:
                        insights_path = video_dir / "visual_insights.md"
                        if insights_path.exists():
                            bundle.visual_insights = insights_path.read_text(
                                encoding="utf-8"
                            ).strip()
                    logger.debug("reusing cached media for %s", record.video_id)
                    return bundle, segments
                logger.warning(
                    "cached media for %s has an empty transcript; re-acquiring",
                    record.video_id,
                )
            else:
                logger.info(
                    "media cache for %s is stale or incomplete; re-acquiring",
                    record.video_id,
                )
        except Exception as exc:  # a corrupt cache must not be fatal
            logger.warning("unreadable media cache for %s: %s", record.video_id, exc)

    handlers = {
        "youtube": _acquire_youtube,
        "local_video": _acquire_local_video,
        "transcript": _acquire_transcript,
    }
    handler = handlers.get(record.source)
    if handler is None:
        raise ValueError(f"unknown source {record.source!r} for {record.video_id}")

    bundle, segments = handler(record, video_dir, settings)
    bundle.settings_fingerprint = fingerprint

    if not segments:
        raise RuntimeError(
            f"no transcript could be obtained for '{record.title}'. The recording "
            "may have no speech, or captions may have failed to download."
        )
    Workspace.write_text(cached, bundle.model_dump_json(indent=2))
    return bundle, segments


def _cached_bundle_usable(
    bundle: MediaBundle, record: VideoRecord, fingerprint: str
) -> bool:
    """True when media.json can safely skip re-download / re-extract."""
    if bundle.settings_fingerprint != fingerprint:
        return False

    transcript = Path(bundle.transcript_path) if bundle.transcript_path else None
    if transcript is None or not yt.transcript_file_usable(transcript):
        return False

    if record.source in {"youtube", "local_video"}:
        if not bundle.video_path or not Path(bundle.video_path).is_file():
            return False
        if Path(bundle.video_path).stat().st_size < 1024:
            return False

    for frame in bundle.frames:
        path = Path(frame.file)
        if not path.is_file():
            return False
    return True


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------


def _acquire_youtube(
    record: VideoRecord, video_dir: Path, settings: Settings
) -> tuple[MediaBundle, list[TranscriptSegment]]:
    auth = yt.auth_options(
        cookies_file=settings.youtube_cookies_file,
        cookies_from_browser=settings.youtube_cookies_from_browser,
        proxy=settings.youtube_proxy,
    )
    video_path = yt.download_video(
        record.url, video_dir, max_height=settings.download_resolution, auth=auth
    )

    captions = yt.fetch_captions(record.url, video_dir, auth=auth)
    if captions is not None:
        transcript_path = yt.vtt_to_text(captions, video_dir)
        source = "captions"
    else:
        logger.info("no captions for %s; falling back to whisper", record.video_id)
        audio_path = yt.extract_audio(video_path, video_dir)
        transcript_path = yt.whisper_transcribe(
            audio_path, video_dir, model_size=settings.whisper_model
        )
        source = "whisper"

    segments = transcript_tools.load_segments(transcript_path)
    frames = _extract_frames(video_path, video_dir, settings)
    insights = _visual_insights(
        settings,
        video_dir=video_dir,
        video_path=video_path,
        youtube_url=record.url,
        title=record.title,
    )

    return (
        MediaBundle(
            video_id=record.video_id,
            video_path=str(video_path),
            transcript_path=str(transcript_path),
            transcript_source=source,  # type: ignore[arg-type]
            duration_sec=yt.probe_duration(video_path),
            frames=frames,
            visual_insights=insights,
        ),
        segments,
    )


# ---------------------------------------------------------------------------
# Local video
# ---------------------------------------------------------------------------

#: Suffixes checked for a transcript sitting beside a video file.
_SIDECAR_SUFFIXES = (".srt", ".vtt", ".txt", ".en.srt", ".en.vtt")


def find_sidecar_transcript(video_path: Path) -> Path | None:
    """A transcript named after the video, e.g. ``talk.mp4`` -> ``talk.srt``.

    Worth checking before Whisper: an existing subtitle file is exact, free and
    instant, where transcription is slow, approximate and burns CPU.
    """
    for suffix in _SIDECAR_SUFFIXES:
        candidate = video_path.with_suffix("").with_name(video_path.stem + suffix)
        if candidate.exists() and candidate != video_path:
            return candidate
    return None


def _acquire_local_video(
    record: VideoRecord, video_dir: Path, settings: Settings
) -> tuple[MediaBundle, list[TranscriptSegment]]:
    source_path = Path(record.source_path)
    if not source_path.exists():
        raise RuntimeError(
            f"the source file for '{record.title}' is gone: {source_path}"
        )

    sidecar = find_sidecar_transcript(source_path)
    if sidecar is not None:
        logger.info("using sidecar transcript %s", sidecar.name)
        segments = transcript_tools.load_segments(sidecar)
        transcript_path = transcript_tools.write_normalized(
            segments, video_dir / "transcript.txt"
        )
        transcript_source = "captions"
    else:
        audio_path = yt.extract_audio(source_path, video_dir)
        transcript_path = yt.whisper_transcribe(
            audio_path, video_dir, model_size=settings.whisper_model
        )
        segments = transcript_tools.load_segments(transcript_path)
        transcript_source = "whisper"

    frames = _extract_frames(source_path, video_dir, settings)
    insights = _visual_insights(
        settings,
        video_dir=video_dir,
        video_path=source_path,
        youtube_url=None,
        title=record.title,
    )

    return (
        MediaBundle(
            video_id=record.video_id,
            # The user's file is read in place and never copied: these can be
            # gigabytes, and duplicating them into the workspace would waste
            # disk for no benefit.
            video_path=str(source_path),
            transcript_path=str(transcript_path),
            transcript_source=transcript_source,  # type: ignore[arg-type]
            duration_sec=yt.probe_duration(source_path),
            frames=frames,
            visual_insights=insights,
        ),
        segments,
    )


# ---------------------------------------------------------------------------
# Transcript only
# ---------------------------------------------------------------------------


def _acquire_transcript(
    record: VideoRecord, video_dir: Path, settings: Settings
) -> tuple[MediaBundle, list[TranscriptSegment]]:
    source_path = Path(record.source_path)
    if not source_path.exists():
        raise RuntimeError(
            f"the transcript for '{record.title}' is gone: {source_path}"
        )

    segments = transcript_tools.load_segments(source_path)
    transcript_path = transcript_tools.write_normalized(
        segments, video_dir / "transcript.txt"
    )
    duration = segments[-1].start_sec if segments else 0.0

    return (
        MediaBundle(
            video_id=record.video_id,
            transcript_path=str(transcript_path),
            transcript_source="captions",
            duration_sec=duration,
            frames=[],  # no recording, so no screenshots
        ),
        segments,
    )


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


def _extract_frames(
    video_path: Path, video_dir: Path, settings: Settings
) -> list:
    """Extract and deduplicate frames, tolerating a missing ffmpeg.

    Frames are an enhancement: a book without screenshots is still a book, but
    a run that dies at frame extraction produces nothing at all.
    """
    frames_dir = video_dir / "frames_raw"
    try:
        raw_frames = frame_tools.extract_frames(
            video_path,
            frames_dir,
            interval_sec=settings.frame_interval_sec,
            scene_threshold=settings.frame_scene_threshold,
        )
    except Exception as exc:
        logger.warning(
            "frame extraction failed for %s (%s); the chapter will have no "
            "screenshots", video_path.name, exc,
        )
        return []
    return frame_tools.dedupe_frames(
        raw_frames,
        interval_sec=settings.frame_interval_sec,
        hash_threshold=settings.frame_phash_threshold,
        timestamps=frame_tools.load_frame_timestamps(frames_dir),
    )


def _visual_insights(
    settings: Settings,
    *,
    video_dir: Path,
    video_path: Path | None,
    youtube_url: str | None,
    title: str,
) -> str:
    """Optional Gemini video understanding; cached beside media.json."""
    cache = video_dir / "visual_insights.md"
    if cache.exists():
        text = cache.read_text(encoding="utf-8").strip()
        if text:
            return text
    if not settings.gemini_video_enrich:
        return ""
    text = gemini_video.enrich_video_visuals(
        video_path=video_path,
        youtube_url=youtube_url,
        title=title,
        model=settings.gemini_video_model,
        max_file_mb=settings.gemini_video_max_upload_mb,
    ).strip()
    if text:
        Workspace.write_text(cache, text + "\n")
        logger.info("stored Gemini visual briefing (%d chars) for %s", len(text), title)
    return text
