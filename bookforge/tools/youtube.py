"""YouTube acquisition: listing, download, captions, whisper fallback.

Every function is synchronous, deterministic and idempotent — if the target
file exists the work is skipped, which makes resume free.

The transcript is treated as a list of timestamped ``TranscriptSegment`` values
rather than a blob of text. That matters downstream: chunking can split on
segment boundaries instead of mid-sentence, the frame captioner can look up the
words spoken around a screenshot's timestamp, and the grounding metric can cite
the moment a claim came from.
"""

from __future__ import annotations

import functools
import logging
import re
import subprocess
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.cookies import CookieLoadError, extract_cookies_from_browser

from bookforge.schemas import TranscriptSegment

logger = logging.getLogger(__name__)

_YT_EXTRACTOR_ARGS = {"youtube": {"player_client": ["android", "web", "ios"]}}


def auth_options(
    cookies_file: Path | None = None,
    cookies_from_browser: str = "",
    proxy: str = "",
) -> dict:
    """yt-dlp options that get past YouTube's datacentre block.

    From a server IP, YouTube answers media and caption requests with
    "Sign in to confirm you\u2019re not a bot" while still serving flat channel
    listings. The result is a pipeline that discovers every video and then fails
    to download any of them -- a confusing failure, because intake looks healthy.

    A cookies.txt export from a signed-in browser is the supported way through.
    ``cookies_from_browser`` reads a local browser profile directly and only
    works on a desktop. A proxy helps when the block is purely IP-based.
    """
    options: dict = {}
    if cookies_file:
        path = Path(cookies_file).expanduser()
        if path.exists():
            options["cookiefile"] = str(path)
        else:
            logger.warning("cookies file %s does not exist; ignoring", path)
    if cookies_from_browser and browser_cookies_readable(cookies_from_browser):
        options["cookiesfrombrowser"] = (cookies_from_browser,)
    if proxy:
        options["proxy"] = proxy
    return options


@functools.cache
def browser_cookies_readable(browser: str) -> bool:
    """Can yt-dlp actually load this browser's cookies? Probed once per process.

    Unreadable browser cookies are dropped rather than fatal: every yt-dlp call
    would otherwise fail on them, while from a residential IP the same request
    succeeds with no cookies at all. Chrome 127+ on Windows is never readable
    (app-bound encryption), and any Chromium browser is locked while open.
    """
    try:
        extract_cookies_from_browser(browser)
    except Exception as exc:
        logger.warning(
            "ignoring BOOKFORGE_YOUTUBE_COOKIES_FROM_BROWSER=%s and continuing "
            "without cookies: %s (original error: %s)",
            browser, COOKIE_LOAD_HELP, exc,
        )
        return False
    return True


def is_bot_block(error: BaseException | str) -> bool:
    """Recognise YouTube's bot challenge so callers can say something useful."""
    text = str(error).lower()
    return "confirm you" in text and "not a bot" in text


BOT_BLOCK_HELP = (
    "YouTube refused the download with its bot check. This is normal from a "
    "server or datacentre IP. Export cookies.txt from a signed-in browser and "
    "set BOOKFORGE_YOUTUBE_COOKIES_FILE, or set "
    "BOOKFORGE_YOUTUBE_COOKIES_FROM_BROWSER=chrome on a desktop. Local video "
    "files and transcript folders are unaffected."
)

#: yt-dlp raises this when it cannot read the configured browser's cookie
#: store — on Windows this is almost always the browser being open, which
#: locks the cookie database (https://github.com/yt-dlp/yt-dlp/issues/7271).
#: Without this, the real cause was swallowed into a generic "check the URL"
#: message that sent people looking in exactly the wrong place.
COOKIE_LOAD_HELP = (
    "Could not read cookies from the configured browser "
    "(BOOKFORGE_YOUTUBE_COOKIES_FROM_BROWSER). On Windows this usually means "
    "the browser is currently running and has locked its cookie database — "
    "close it and retry, point at a different installed browser, or export "
    "a cookies.txt file instead and set BOOKFORGE_YOUTUBE_COOKIES_FILE. "
    "Cookies are only needed to get past YouTube's bot check from a "
    "datacentre IP; running locally, you can often just clear "
    "BOOKFORGE_YOUTUBE_COOKIES_FROM_BROWSER and skip them entirely."
)


# ---------------------------------------------------------------------------
# Channel listing
# ---------------------------------------------------------------------------


def list_channel_videos(
    channel_url: str,
    max_videos: int | None = None,
    min_duration_sec: int = 0,
    max_duration_sec: int = 10**9,
    auth: dict | None = None,
) -> tuple[str, list[dict]]:
    """Return ``(channel_title, video dicts)`` for a channel or playlist URL.

    Flat extraction: metadata only, no downloads. When ``max_videos`` is set we
    over-fetch and then filter, because shorts and trailers are excluded by
    duration and would otherwise eat into the cap.
    """
    cleaned = re.sub(r"[.,;/\s]+$", "", str(channel_url).strip())
    normalized = cleaned
    if "/@" in normalized and not normalized.rstrip("/").endswith(
        ("/videos", "/streams", "/playlists")
    ):
        normalized = normalized.rstrip("/") + "/videos"

    options: dict = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,  # private/deleted videos must not kill intake
        "extractor_args": _YT_EXTRACTOR_ARGS,
        **(auth or {}),
    }
    if max_videos:
        # Over-fetch generously: duration filtering happens after extraction.
        options["playlistend"] = max(max_videos * 5, max_videos + 20)

    info = None
    with YoutubeDL(options) as ydl:
        for candidate in (normalized, cleaned):
            try:
                info = ydl.extract_info(candidate, download=False)
                if info:
                    break
            except CookieLoadError as exc:
                # Cookie loading fails identically for every candidate URL —
                # retrying the second one would just print the same error
                # again and mask it behind the generic message below.
                raise RuntimeError(f"{COOKIE_LOAD_HELP} (original error: {exc})") from exc
            except Exception as exc:
                logger.debug("extraction failed for %s: %s", candidate, exc)

    if not info:
        raise RuntimeError(
            f"Could not read {channel_url}. Check the URL is a public channel "
            "or playlist and that yt-dlp is up to date."
        )

    channel_title = info.get("title") or info.get("channel") or "Unknown channel"
    videos: list[dict] = []
    skipped_by_duration = 0

    for entry in info.get("entries") or []:
        if not entry:
            continue
        video_id = entry.get("id") or ""
        if not video_id:
            continue
        duration = int(entry.get("duration") or 0)
        if duration and not (min_duration_sec <= duration <= max_duration_sec):
            skipped_by_duration += 1
            continue

        url = entry.get("url") or ""
        if "youtube.com" not in url and "youtu.be" not in url:
            url = f"https://www.youtube.com/watch?v={video_id}"

        videos.append(
            {
                "video_id": video_id,
                "title": entry.get("title") or video_id,
                "url": url,
                "duration_sec": duration,
                "upload_date": entry.get("upload_date") or "",
            }
        )
        if max_videos and len(videos) >= max_videos:
            break

    if skipped_by_duration:
        logger.info(
            "skipped %d videos outside the %d-%ds duration window",
            skipped_by_duration,
            min_duration_sec,
            max_duration_sec,
        )
    if max_videos and len(videos) < max_videos:
        logger.warning(
            "requested %d videos but only %d matched the duration filter",
            max_videos,
            len(videos),
        )
    return channel_title, videos


# ---------------------------------------------------------------------------
# Download + audio
# ---------------------------------------------------------------------------


def download_video(
    url: str, dest_dir: Path, max_height: int = 480, auth: dict | None = None
) -> Path:
    """Download a resolution-capped mp4; returns the existing file on re-run."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest_dir.glob("video.*"))
    usable = [path for path in existing if path.is_file() and path.stat().st_size >= 1024]
    if usable:
        return usable[0]
    for stub in existing:
        try:
            if stub.is_file() and stub.stat().st_size < 1024:
                stub.unlink(missing_ok=True)
                logger.warning("removed empty download stub %s", stub.name)
        except OSError:
            pass

    options = {
        "format": (
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]"
            f"/best[height<={max_height}][ext=mp4]"
            f"/best[height<={max_height}]/best"
        ),
        "outtmpl": str(dest_dir / "video.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "overwrites": False,
        "extractor_args": _YT_EXTRACTOR_ARGS,
        **(auth or {}),
    }
    try:
        with YoutubeDL(options) as ydl:
            ydl.download([url])
    except CookieLoadError as exc:
        raise RuntimeError(f"{COOKIE_LOAD_HELP} (original error: {exc})") from exc
    except Exception as exc:
        if is_bot_block(exc):
            raise RuntimeError(f"{BOT_BLOCK_HELP} (original error: {exc})") from exc
        raise

    downloaded = sorted(dest_dir.glob("video.*"))
    usable = [path for path in downloaded if path.is_file() and path.stat().st_size >= 1024]
    if not usable:
        raise RuntimeError(f"yt-dlp produced no file for {url}")
    return usable[0]


def extract_audio(video_path: Path, dest_dir: Path) -> Path:
    """16 kHz mono wav for whisper; idempotent."""
    dest = Path(dest_dir) / "audio.wav"
    if dest.exists() and dest.stat().st_size >= 256:
        return dest
    if dest.exists():
        dest.unlink(missing_ok=True)
        logger.warning("removed empty audio cache at %s", dest)
    _run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(dest),
        ],
        f"audio extraction for {video_path}",
    )
    return dest


def probe_duration(video_path: Path) -> float:
    """Video duration in seconds via ffprobe; 0.0 when unavailable."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
            ],
            capture_output=True, text=True, timeout=60, check=False,
        )
        return float((proc.stdout or "0").strip() or 0.0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------


def fetch_captions(
    url: str,
    dest_dir: Path,
    langs: tuple[str, ...] = ("en.*", "en"),
    auth: dict | None = None,
) -> Path | None:
    """Fetch human or automatic captions as WebVTT. None when unavailable."""
    dest_dir = Path(dest_dir)
    existing = _find_vtt(dest_dir)
    if existing:
        return existing

    options = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": list(langs),
        "subtitlesformat": "vtt",
        "outtmpl": str(dest_dir / "captions.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "extractor_args": _YT_EXTRACTOR_ARGS,
        **(auth or {}),
    }
    try:
        with YoutubeDL(options) as ydl:
            ydl.download([url])
    except CookieLoadError:
        # Not fatal here — falling through to Whisper still produces a
        # transcript — but silently eating this wastes minutes of Whisper
        # transcription for a problem that is usually a one-line env fix.
        logger.warning(
            "caption fetch skipped for %s: %s Falling back to Whisper.",
            url, COOKIE_LOAD_HELP,
        )
        return None
    except Exception as exc:
        if is_bot_block(exc):
            logger.warning("caption fetch blocked for %s. %s", url, BOT_BLOCK_HELP)
        else:
            logger.warning("caption fetch failed for %s: %s", url, exc)
        return None
    return _find_vtt(dest_dir)


def _find_vtt(dest_dir: Path) -> Path | None:
    found = sorted(dest_dir.glob("captions.*.vtt")) + sorted(dest_dir.glob("captions.vtt"))
    return found[0] if found else None


def transcript_file_usable(path: Path, min_chars: int = 20) -> bool:
    """False for missing or near-empty transcript caches that must be regenerated."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return False
    return len(text) >= min_chars


def whisper_transcribe(
    audio_path: Path, dest_dir: Path, model_size: str = "base"
) -> Path:
    """Local faster-whisper transcription with timestamps; idempotent."""
    dest = Path(dest_dir) / "transcript.txt"
    if transcript_file_usable(dest):
        return dest
    if dest.exists():
        dest.unlink(missing_ok=True)
        logger.warning("removed empty transcript cache at %s", dest)

    from faster_whisper import WhisperModel  # heavy import, deferred

    logger.info("transcribing %s with faster-whisper (%s)", audio_path.name, model_size)
    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, _info = model.transcribe(str(audio_path), vad_filter=True)
    lines = [f"[{format_timestamp(seg.start)}] {seg.text.strip()}" for seg in segments]
    dest.write_text("\n".join(lines), encoding="utf-8")
    if not transcript_file_usable(dest):
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"whisper produced an empty transcript for {audio_path}")
    return dest


_VTT_TS_RE = re.compile(r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})[.,]\d+\s*-->")
_VTT_TAG_RE = re.compile(r"<[^>]+>")


def vtt_to_text(vtt_path: Path, dest_dir: Path) -> Path:
    """Convert WebVTT to timestamped plain text; idempotent.

    Output lines look like ``[00:01:12] the spoken words``. Consecutive
    duplicate lines — the scrolling artifact of auto-captions — are dropped.
    """
    dest = Path(dest_dir) / "transcript.txt"
    if transcript_file_usable(dest):
        return dest
    if dest.exists():
        dest.unlink(missing_ok=True)
        logger.warning("removed empty transcript cache at %s", dest)

    lines_out: list[str] = []
    last_text = ""
    timestamp = "00:00:00"
    raw_text = Path(vtt_path).read_text(encoding="utf-8", errors="ignore")

    for raw in raw_text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        match = _VTT_TS_RE.search(line)
        if match:
            timestamp = f"{match.group('h').zfill(2)}:{match.group('m')}:{match.group('s')}"
            continue
        if line.isdigit():
            continue  # cue index
        text = _VTT_TAG_RE.sub("", line).strip()
        if not text or text == last_text:
            continue
        lines_out.append(f"[{timestamp}] {text}")
        last_text = text

    dest.write_text("\n".join(lines_out), encoding="utf-8")
    if not transcript_file_usable(dest):
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"VTT conversion produced an empty transcript for {vtt_path}")
    return dest


_SEGMENT_RE = re.compile(r"^\[(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})\]\s*(?P<text>.*)$")


def parse_transcript(path: Path) -> list[TranscriptSegment]:
    """Read ``transcript.txt`` into timestamped segments.

    Lines without a timestamp marker are attached to the previous segment so no
    spoken content is ever dropped.
    """
    segments: list[TranscriptSegment] = []
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _SEGMENT_RE.match(line)
        if match:
            start = (
                int(match.group("h")) * 3600
                + int(match.group("m")) * 60
                + int(match.group("s"))
            )
            segments.append(
                TranscriptSegment(start_sec=float(start), text=match.group("text").strip())
            )
        elif segments:
            segments[-1].text = f"{segments[-1].text} {line}".strip()
        else:
            segments.append(TranscriptSegment(start_sec=0.0, text=line))
    return segments


def segments_to_text(segments: list[TranscriptSegment]) -> str:
    return "\n".join(f"[{seg.timestamp}] {seg.text}" for seg in segments)


def load_transcript_text(path: Path) -> str:
    """Full transcript text.

    Note there is deliberately no truncation here. The old implementation kept
    the head and tail and dropped the middle of long talks, so a three-hour
    lecture silently lost its entire core. Long transcripts are handled by
    chunked map-reduce analysis instead — see ``bookforge.dspy_runtime.chunking``.
    """
    return Path(path).read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _run(cmd: list[str], what: str, timeout_sec: int = 1800) -> None:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace",
            timeout=timeout_sec, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{what} timed out after {timeout_sec}s") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed: {(proc.stderr or '')[-2000:]}")
