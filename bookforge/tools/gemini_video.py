"""Optional Gemini video understanding for lecture visuals.

Uses the Google GenAI File API (or a public YouTube URL) when
``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` is set and enrichment is enabled.
Failures are soft: frame-based captioning still works without this pass.

See https://ai.google.dev/gemini-api/docs/video-understanding
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = """\
You are helping turn a lecture / talk video into a technical textbook chapter.

Watch the video and produce a compact visual briefing for downstream writing:

1. **Slide / screen inventory** — list the distinct diagrams, tables, equations,
   UI screenshots, or architecture drawings that appear, with approximate
   timestamps (MM:SS). Skip talking-head-only shots.
2. **Key visual claims** — numbers, labels, axis titles, and equation forms
   readable on screen (quote them exactly when clear).
3. **Suggested figures** — which 5–12 moments are worth printing as textbook
   figures, and one sentence each on what a caption should emphasize.
4. **Pedagogy cues** — where the speaker walks through a derivation or
   comparison on screen (so the chapter can mirror that structure).

Keep the whole briefing under 900 words. Prefer bullet lists. If the video is
mostly a face with no slides, say so in one sentence.
"""


def enrich_video_visuals(
    *,
    video_path: Path | None = None,
    youtube_url: str | None = None,
    title: str = "",
    model: str = "gemini-2.0-flash",
    max_file_mb: int = 100,
    timeout_sec: int = 300,
) -> str:
    """Return a visual briefing string, or empty when unavailable / disabled.

    Prefer a local file upload for reliability; fall back to a YouTube URL when
    no file is present (or the file is too large for a cheap upload path).
    """
    api_key = (
        os.getenv("GOOGLE_API_KEY", "").strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
    )
    if not api_key:
        logger.debug("gemini video enrich skipped: no GOOGLE_API_KEY")
        return ""

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning(
            "google-genai is not installed; set BOOKFORGE_GEMINI_VIDEO_ENRICH=false "
            "or pip install 'bookforge[gemini]'"
        )
        return ""

    client = genai.Client(api_key=api_key)
    prompt = f"Video title: {title or '(untitled)'}\n\n{_DEFAULT_PROMPT}"
    model_name = model.removeprefix("gemini/")

    try:
        if video_path and Path(video_path).is_file():
            size_mb = Path(video_path).stat().st_size / (1024 * 1024)
            if size_mb <= max_file_mb:
                return _generate_from_file(
                    client, types, Path(video_path), prompt, model_name, timeout_sec
                )
            logger.info(
                "video %.0fMB exceeds upload cap %dMB; trying YouTube URL if any",
                size_mb,
                max_file_mb,
            )
        if youtube_url and "youtu" in youtube_url:
            return _generate_from_youtube(
                client, types, youtube_url, prompt, model_name
            )
    except Exception as exc:  # enrichment must never fail the book
        logger.warning("gemini video enrich failed: %s", exc)
        return ""

    logger.debug("gemini video enrich skipped: no usable video input")
    return ""


def _generate_from_file(
    client,
    types,
    video_path: Path,
    prompt: str,
    model: str,
    timeout_sec: int,
) -> str:
    uploaded = client.files.upload(file=str(video_path))
    deadline = time.time() + timeout_sec
    name = getattr(uploaded, "name", None)
    while time.time() < deadline:
        state = getattr(getattr(uploaded, "state", None), "name", None) or str(
            getattr(uploaded, "state", "")
        )
        if state in {"ACTIVE", "FileState.ACTIVE"}:
            break
        if state in {"FAILED", "FileState.FAILED"}:
            raise RuntimeError(f"Gemini file processing failed for {video_path.name}")
        time.sleep(3)
        if name:
            uploaded = client.files.get(name=name)
    else:
        raise TimeoutError(f"Gemini file still processing after {timeout_sec}s")

    response = client.models.generate_content(
        model=model,
        contents=[uploaded, prompt],
    )
    text = (getattr(response, "text", None) or "").strip()
    try:
        if name:
            client.files.delete(name=name)
    except Exception:
        pass
    return text


def _generate_from_youtube(client, types, url: str, prompt: str, model: str) -> str:
    # YouTube URLs are a first-class Gemini video input for public videos.
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_uri(file_uri=url, mime_type="video/*"),
            prompt,
        ],
    )
    return (getattr(response, "text", None) or "").strip()
