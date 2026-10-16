"""Frame extraction, deduplication and quality-ranked curation.

Per video::

    ffmpeg interval samples  +  optional scene-change samples
      -> perceptual-hash dedupe + junk rejection, scoring each survivor
      -> bucketed, quality-ranked curation (<= K frames)  ->  chapters/../figures/

Interval sampling alone misses slide flips that fall between ticks; scene
detection alone misses slow builds on a single slide. The hybrid keeps both,
inspired by lecture-note pipelines and everyday ffmpeg scene filters, without
copying any third-party code.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

import imagehash
from PIL import Image, ImageFilter, ImageStat

from bookforge.schemas import FrameAsset

logger = logging.getLogger(__name__)

_FRAME_GLOB = "frame_*.jpg"
_PTS_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")


def extract_frames(
    video_path: Path,
    out_dir: Path,
    interval_sec: int = 5,
    scene_threshold: float = 0.25,
    timeout_sec: int = 1800,
) -> list[Path]:
    """Hybrid screenshots: fixed interval plus optional scene cuts.

    Writes ``timestamps.json`` beside the frames so dedupe can use real times
    instead of guessing from frame index. Idempotent per directory *and*
    extraction settings — changing interval/scene forces a re-extract.
    """
    out_dir = Path(out_dir)
    existing = sorted(out_dir.glob(_FRAME_GLOB))
    stamp_path = out_dir / "timestamps.json"
    meta_path = out_dir / "extract_meta.json"
    meta = {
        "interval_sec": int(interval_sec),
        "scene_threshold": float(scene_threshold),
    }
    if existing and stamp_path.exists() and _extract_meta_matches(meta_path, meta):
        logger.debug("reusing %d extracted frames in %s", len(existing), out_dir)
        return existing

    # Stale cache from a previous settings combination — start clean.
    if existing or stamp_path.exists() or meta_path.exists():
        for stale in out_dir.glob(_FRAME_GLOB):
            stale.unlink(missing_ok=True)
        stamp_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    timed: list[tuple[float, Path]] = []
    timed.extend(
        _extract_interval_samples(
            video_path, work / "interval", interval_sec, timeout_sec
        )
    )
    if scene_threshold and scene_threshold > 0:
        timed.extend(
            _extract_scene_samples(
                video_path, work / "scene", scene_threshold, timeout_sec
            )
        )

    if not timed:
        raise RuntimeError(f"ffmpeg produced no frames for {video_path}")

    timed.sort(key=lambda item: item[0])
    # Drop near-simultaneous captures from interval+scene overlap before
    # writing the final numbered set (phash dedupe still runs later).
    merged: list[tuple[float, Path]] = []
    for ts, path in timed:
        if merged and abs(ts - merged[-1][0]) < 0.75:
            continue
        merged.append((ts, path))

    timestamps: dict[str, float] = {}
    final_paths: list[Path] = []
    for index, (ts, source) in enumerate(merged, start=1):
        name = f"frame_{index:05d}.jpg"
        dest = out_dir / name
        dest.write_bytes(source.read_bytes())
        timestamps[name] = round(ts, 2)
        final_paths.append(dest)

    import json

    stamp_path.write_text(json.dumps(timestamps, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    logger.info(
        "extracted %d frames (%d after near-time merge) from %s",
        len(timed),
        len(final_paths),
        video_path.name,
    )
    return final_paths


def _extract_meta_matches(meta_path: Path, expected: dict) -> bool:
    if not meta_path.exists():
        return False
    try:
        import json

        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        return (
            int(raw.get("interval_sec", -1)) == int(expected["interval_sec"])
            and abs(float(raw.get("scene_threshold", -1)) - float(expected["scene_threshold"]))
            < 1e-6
        )
    except (OSError, ValueError, TypeError):
        return False


def load_frame_timestamps(out_dir: Path) -> dict[str, float]:
    """Return basename -> seconds mapping written by ``extract_frames``."""
    path = Path(out_dir) / "timestamps.json"
    if not path.exists():
        return {}
    try:
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in raw.items()}
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("unreadable frame timestamps in %s: %s", path, exc)
        return {}


def _extract_interval_samples(
    video_path: Path, out_dir: Path, interval_sec: int, timeout_sec: int
) -> list[tuple[float, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    interval = max(1, int(interval_sec))
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval}",
        "-q:v",
        "3",
        str(out_dir / "interval_%05d.jpg"),
    ]
    _run_ffmpeg(cmd, timeout_sec, video_path, "interval frame extraction")
    frames = sorted(out_dir.glob("interval_*.jpg"))
    # fps filter samples near the midpoint of each interval window.
    return [
        (round((index + 0.5) * interval, 2), path)
        for index, path in enumerate(frames)
    ]


def _extract_scene_samples(
    video_path: Path, out_dir: Path, threshold: float, timeout_sec: int
) -> list[tuple[float, Path]]:
    """Scene-change screenshots with timestamps parsed from showinfo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # showinfo logs pts_time on stderr; keep loglevel info for that stream.
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "info",
        "-i",
        str(video_path),
        "-vf",
        f"select='gt(scene\\,{threshold:.3f})',showinfo",
        "-vsync",
        "vfr",
        "-q:v",
        "3",
        str(out_dir / "scene_%05d.jpg"),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg scene extraction timed out after {timeout_sec}s for {video_path}"
        ) from exc
    if proc.returncode != 0:
        logger.warning(
            "scene extraction failed for %s; continuing with interval frames only: %s",
            video_path.name,
            (proc.stderr or "")[-800:],
        )
        return []

    frames = sorted(out_dir.glob("scene_*.jpg"))
    times = [float(m) for m in _PTS_RE.findall(proc.stderr or "")]
    if len(times) < len(frames):
        # Fallback: space unknown times evenly — better than discarding scenes.
        duration = _probe_duration(video_path)
        times = [
            round((i + 1) * duration / (len(frames) + 1), 2)
            for i in range(len(frames))
        ]
    return list(zip(times[: len(frames)], frames, strict=False))


def _probe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", timeout=60, check=False
        )
        return max(1.0, float((proc.stdout or "0").strip() or "0"))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 600.0


def _run_ffmpeg(
    cmd: list[str], timeout_sec: int, video_path: Path, label: str
) -> None:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg {label} timed out after {timeout_sec}s for {video_path}"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg {label} failed: {(proc.stderr or '')[-2000:]}")


# ---------------------------------------------------------------------------
# Quality signals
# ---------------------------------------------------------------------------


def _sharpness(img: Image.Image) -> float:
    """Edge energy as a focus proxy. Higher is crisper."""
    edges = img.convert("L").resize((96, 96)).filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(edges).stddev[0])


def _brightness(img: Image.Image) -> float:
    return float(ImageStat.Stat(img.convert("L").resize((32, 32))).mean[0])


def _contrast(img: Image.Image) -> float:
    """Luminance stddev — a robust detector of flat, empty or faded frames."""
    return float(ImageStat.Stat(img.convert("L").resize((64, 64))).stddev[0])


def _skin_fraction(img: Image.Image) -> float:
    """Rough YCbCr skin share in the center crop — proxy for talking-head shots.

    Lecture-note pipelines demote presenter close-ups when slides matter more.
    This stays dependency-free (no face model): high center skin → demote.
    """
    rgb = img.convert("RGB")
    w, h = rgb.size
    left, top = int(w * 0.3), int(h * 0.2)
    right, bottom = int(w * 0.7), int(h * 0.85)
    crop = rgb.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
    small = crop.resize((48, 48))
    skin = 0
    total = 0
    for r, g, b in small.getdata():
        total += 1
        # Classic loose skin gate in RGB (works well enough for demotion).
        if (
            r > 95
            and g > 40
            and b > 20
            and r > g
            and r > b
            and abs(r - g) > 15
            and (max(r, g, b) - min(r, g, b)) > 15
        ):
            skin += 1
    return skin / max(1, total)


def _quality_score(
    sharpness: float,
    contrast: float,
    brightness: float,
    skin_fraction: float = 0.0,
) -> float:
    """Blend the signals into a 0..1 desirability used for ranking.

    Slides — the frames worth printing — are high-contrast and sharp-edged.
    Mid-brightness is preferred because both blown-out and near-black frames
    reproduce badly in print. Heavy center skin demotes talking-head frames.
    """
    sharp_component = min(sharpness / 40.0, 1.0)
    contrast_component = min(contrast / 70.0, 1.0)
    brightness_penalty = abs(brightness - 128.0) / 128.0
    score = (
        0.5 * sharp_component
        + 0.4 * contrast_component
        + 0.1 * (1 - brightness_penalty)
    )
    if skin_fraction > 0.28:
        score *= max(0.35, 1.0 - (skin_fraction - 0.28) * 1.4)
    return round(max(0.0, min(1.0, score)), 4)


def dedupe_frames(
    frame_paths: list[Path],
    interval_sec: int,
    hash_threshold: int = 6,
    min_sharpness: float = 4.0,
    min_contrast: float = 6.0,
    brightness_range: tuple[float, float] = (12.0, 245.0),
    timestamps: dict[str, float] | None = None,
) -> list[FrameAsset]:
    """Drop near-duplicates and junk, scoring every survivor.

    Frames are compared against the last *kept* frame, which collapses long
    static runs (a slide held for two minutes) into a single representative
    while preserving genuine visual changes.
    """
    timestamps = timestamps or {}
    kept: list[FrameAsset] = []
    last_hash: imagehash.ImageHash | None = None

    for index, path in enumerate(frame_paths):
        try:
            with Image.open(path) as img:
                img.load()
                sharpness = _sharpness(img)
                brightness = _brightness(img)
                contrast = _contrast(img)
                skin = _skin_fraction(img)
                if (
                    sharpness < min_sharpness
                    or contrast < min_contrast
                    or not (brightness_range[0] <= brightness <= brightness_range[1])
                ):
                    continue
                current = imagehash.phash(img)
        except Exception as exc:
            logger.warning("skipping unreadable frame %s: %s", path, exc)
            continue

        if last_hash is not None and abs(current - last_hash) <= hash_threshold:
            continue
        last_hash = current
        ts = timestamps.get(path.name)
        if ts is None:
            ts = round((index + 0.5) * max(1, interval_sec), 1)
        kept.append(
            FrameAsset(
                file=str(path),
                timestamp_sec=float(ts),
                phash=str(current),
                sharpness=round(sharpness, 3),
                contrast=round(contrast, 3),
                quality=_quality_score(sharpness, contrast, brightness, skin),
            )
        )
    logger.info("kept %d/%d frames after dedupe", len(kept), len(frame_paths))
    return kept


def curate_frames(
    assets: list[FrameAsset],
    dest_dir: Path,
    max_frames: int,
) -> list[FrameAsset]:
    """Select at most ``max_frames``, best-in-bucket across the timeline.

    Returns assets whose ``file`` is the *basename* inside ``dest_dir``.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if max_frames <= 0 or not assets:
        return []

    ordered = sorted(assets, key=lambda a: a.timestamp_sec)
    if len(ordered) <= max_frames:
        picked = ordered
    else:
        picked = []
        count = len(ordered)
        for bucket in range(max_frames):
            start = (bucket * count) // max_frames
            end = ((bucket + 1) * count) // max_frames
            window = ordered[start:max(end, start + 1)]
            # Best quality wins; ties break toward the earlier frame.
            picked.append(max(window, key=lambda a: (a.quality, -a.timestamp_sec)))

    curated: list[FrameAsset] = []
    for position, asset in enumerate(picked, start=1):
        source = Path(asset.file)
        dest = dest_dir / f"frame_{position:02d}_{int(asset.timestamp_sec)}s.jpg"
        try:
            dest.write_bytes(source.read_bytes())
        except OSError as exc:
            logger.warning("could not copy frame %s: %s", source, exc)
            continue
        curated.append(
            FrameAsset(
                file=dest.name,
                timestamp_sec=asset.timestamp_sec,
                phash=asset.phash,
                sharpness=asset.sharpness,
                contrast=asset.contrast,
                quality=asset.quality,
            )
        )
    return curated


def format_video_timestamp(seconds: float) -> str:
    """HH:MM:SS for figure provenance footnotes."""
    total = max(0, round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def with_time_footnote(caption: str, timestamp_sec: float) -> str:
    """Append a compact video-time provenance marker to a caption."""
    stamp = format_video_timestamp(timestamp_sec)
    bare = (caption or "").rstrip()
    if not bare:
        return f"Video frame at {stamp}."
    if stamp in bare or re.search(r"\[video\s+\d", bare, re.I):
        return bare
    if bare.endswith("."):
        return f"{bare[:-1]} [video {stamp}]."
    return f"{bare} [video {stamp}]."
