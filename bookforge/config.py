"""Env-driven configuration for BookForge.

Every field maps to a ``BOOKFORGE_<NAME>`` environment variable and can also be
set in a ``.env`` file. Provider credentials for Gemini/Vertex are read by
LiteLLM straight from the environment and are intentionally not mirrored here.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

OptimizerName = Literal["gepa", "miprov2", "bootstrap", "simba", "none"]


class Settings(BaseSettings):
    """Runtime configuration for the whole pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BOOKFORGE_",
        extra="ignore",
        validate_assignment=True,
    )

    # --- LLM routing -------------------------------------------------------
    # Model names are LiteLLM identifiers, e.g.
    #   openai/nvidia/nemotron-3-ultra-550b-a55b   (with api_base set)
    #   gemini/gemini-2.5-pro
    #   anthropic/claude-sonnet-4-5
    analyst_model: str = "openai/nvidia/nemotron-3-ultra-550b-a55b"
    writer_model: str = "openai/nvidia/nemotron-3-ultra-550b-a55b"
    critic_model: str = "openai/nvidia/nemotron-3-ultra-550b-a55b"
    caption_model: str = "openai/nvidia/nemotron-3-ultra-550b-a55b"
    #: Larger model used by GEPA to reflect on failures and rewrite instructions.
    reflection_model: str = "openai/nvidia/nemotron-3-ultra-550b-a55b"

    openai_api_base: str = "https://integrate.api.nvidia.com/v1"
    openai_api_key: str = ""  # BOOKFORGE_OPENAI_API_KEY — never hardcode

    max_tokens: int = 16384
    analyst_temperature: float = 0.2
    writer_temperature: float = 0.4
    critic_temperature: float = 0.0
    lm_num_retries: int = 4
    lm_timeout_sec: int = 300
    lm_cache: bool = True

    # --- Book design -------------------------------------------------------
    #: Named theme from bookforge.design.THEMES.
    book_theme: str = "textbook"
    book_subtitle: str = ""
    book_author: str = "Compiled by BookForge"

    # --- Media pipeline ----------------------------------------------------
    frame_interval_sec: int = 5
    #: ffmpeg scene score (0..1). 0 disables scene-change frames; ~0.25 is a
    #: good default for slide decks. Interval sampling still always runs.
    frame_scene_threshold: float = 0.25
    frame_phash_threshold: int = 6
    max_frames_per_chapter: int = 12
    download_resolution: int = 480
    whisper_model: str = "base"
    #: Append ``[video MM:SS]`` provenance to every frame caption (lecture-notes style).
    figure_time_footnotes: bool = True
    #: Optional Gemini File API / YouTube video understanding pass when a Google
    #: API key is present. Soft-fail; never blocks the book.
    gemini_video_enrich: bool = False
    gemini_video_model: str = "gemini-2.0-flash"
    #: Skip File API upload above this size; YouTube URL may still be tried.
    gemini_video_max_upload_mb: int = 100

    #: YouTube blocks media and caption downloads from datacentre IPs with
    #: "Sign in to confirm you're not a bot". A cookies.txt export (Netscape
    #: format) from a signed-in browser is the supported way through; without
    #: one, server-side YouTube ingest fails on every video while channel
    #: listing still works, which is a confusing failure to debug.
    youtube_cookies_file: Path | None = None
    #: Alternative: read cookies straight from a local browser profile, e.g.
    #: "chrome", "firefox", "edge". Only useful on a desktop.
    youtube_cookies_from_browser: str = ""
    #: Optional proxy for yt-dlp, for the same blocking problem.
    youtube_proxy: str = ""

    # --- Local media ingest -------------------------------------------------
    #: Extensions treated as video when scanning a folder. NoDecode keeps
    #: pydantic-settings from JSON-parsing the env value before our validator
    #: sees it, so a comma-separated list works.
    video_extensions: Annotated[tuple[str, ...], NoDecode] = (
        ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv",
    )
    #: Extensions treated as a ready-made transcript.
    transcript_extensions: Annotated[tuple[str, ...], NoDecode] = (
        ".srt", ".vtt", ".txt", ".md",
    )
    #: How local files are ordered into chapters.
    local_order: Literal["name", "modified", "created"] = "name"

    # --- Transcript handling ----------------------------------------------
    #: Characters per analysis chunk. The old pipeline truncated the middle of
    #: long transcripts; we now map-reduce over chunks so coverage is complete.
    transcript_chunk_chars: int = 24_000
    transcript_chunk_overlap_chars: int = 1_200
    #: Hard ceiling on chunks per video (cost guard). 0 = unlimited.
    max_transcript_chunks: int = 12

    # --- Scope / safety rails ---------------------------------------------
    max_videos: int | None = None
    min_video_duration_sec: int = 30
    max_video_duration_sec: int = 3 * 3600

    # --- Quality loop ------------------------------------------------------
    qa_max_iterations: int = 3
    #: Composite score at or above which a chapter is accepted immediately.
    qa_accept_score: float = 0.92
    #: Below this the chapter is marked failed rather than shipped.
    qa_min_acceptable_score: float = 0.55
    compile_latex: bool = True
    latex_timeout_sec: int = 180

    # --- Runtime accuracy guards -------------------------------------------
    #: Number of analyst samples to reconcile (1 = single pass, no self-consistency).
    analyst_consistency_samples: int = 1
    #: dspy.Refine attempts for the writer when the first draft scores low.
    writer_refine_attempts: int = 2
    #: Reject analysis whose grounding score falls below this and resample.
    min_grounding_score: float = 0.6

    # --- Concurrency -------------------------------------------------------
    media_prefetch_concurrency: int = 3
    chapter_concurrency: int = 1
    num_threads: int = 4
    #: Transcript windows analysed concurrently. Each window is an independent
    #: LLM call (map step, before the merge), so a long talk with many windows
    #: no longer pays for them one round-trip at a time.
    analyst_concurrency: int = 4

    # --- Optimization ------------------------------------------------------
    optimizer: OptimizerName = "gepa"
    optimizer_budget: Literal["light", "medium", "heavy"] = "light"
    #: Directory of compiled DSPy programs loaded at runtime when present.
    artifacts_dir: Path = Path("artifacts")
    #: Set false to run with raw (unoptimized) instructions.
    load_optimized: bool = True

    # --- Web UI ------------------------------------------------------------
    #: Ceiling on a single upload batch. Without one, a browser tab can fill
    #: the server's disk, and the failure lands on whatever writes next --
    #: usually a chapter compile, far from the cause.
    upload_max_mb: int = 2048

    # --- Observability -----------------------------------------------------
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "bookforge-dspy"
    enable_mlflow: bool = False

    # --- Housekeeping ------------------------------------------------------
    workspace_root: Path = Path("data")
    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("video_extensions", "transcript_extensions", mode="before")
    @classmethod
    def _extensions_from_env(cls, value: object) -> object:
        """Accept ".mp4,.mkv" as well as a real sequence.

        pydantic-settings parses a tuple field from the environment as JSON, so
        the obvious ``BOOKFORGE_VIDEO_EXTENSIONS=.mp4,.mkv`` raised an opaque
        SettingsError naming the field but not the format. Anyone adding one
        extension hit it.
        """
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                # NoDecode turned off JSON parsing for the comma case, but a
                # JSON list was the only spelling that worked before, so it has
                # to keep working -- otherwise it degrades to one nonsense
                # extension that matches nothing and the scan silently finds
                # no videos.
                try:
                    text = ",".join(str(part) for part in json.loads(text))
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"could not read an extension list from {value!r}; use "
                        'either ".mp4,.mkv" or \'[".mp4", ".mkv"]\''
                    ) from exc
            parts = [part.strip() for part in text.replace(" ", ",").split(",")]
            value = tuple(
                part if part.startswith(".") else f".{part}" for part in parts if part
            )
        if isinstance(value, (list, tuple)):
            return tuple(str(part).lower() for part in value)
        return value

    @field_validator("qa_max_iterations")
    @classmethod
    def _at_least_one_pass(cls, value: int) -> int:
        return max(1, value)

    @field_validator("transcript_chunk_overlap_chars")
    @classmethod
    def _overlap_non_negative(cls, value: int) -> int:
        return max(0, value)

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> Settings:
        if self.transcript_chunk_overlap_chars >= self.transcript_chunk_chars:
            raise ValueError(
                "transcript_chunk_overlap_chars must be smaller than "
                "transcript_chunk_chars, otherwise chunking never advances"
            )
        if self.qa_min_acceptable_score > self.qa_accept_score:
            raise ValueError("qa_min_acceptable_score cannot exceed qa_accept_score")
        return self

    @property
    def workspace_root_abs(self) -> Path:
        return self.workspace_root.expanduser().resolve()

    @property
    def artifacts_dir_abs(self) -> Path:
        return self.artifacts_dir.expanduser().resolve()

    def has_llm_credentials(self) -> bool:
        """True when at least one provider is configured."""
        return bool(
            self.openai_api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE"
        )

    def resolved_api_key(self) -> str:
        return (
            self.openai_api_key
            or os.getenv("BOOKFORGE_OPENAI_API_KEY", "")
            or os.getenv("NVIDIA_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached Settings — used by tests and by CLI overrides."""
    get_settings.cache_clear()


_JSON_FMT = (
    '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
    '"message":"%(message)s"}'
)
_TEXT_FMT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"

_NOISY_LOGGERS = (
    "urllib3",
    "httpcore",
    "httpx",
    "yt_dlp",
    "LiteLLM",
    "litellm",
    "matplotlib",
    "matplotlib.font_manager",
    "PIL",
    "filelock",
)


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure structured logging once, quieting third-party noise."""
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format=_JSON_FMT if json_format else _TEXT_FMT,
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def preflight(settings: Settings) -> list[str]:
    """Validate the environment. Raises on fatal problems, returns warnings.

    Unlike the original implementation this does not mutate cached settings as a
    side effect; the caller decides what to do with each warning.
    """
    import shutil

    if not settings.has_llm_credentials():
        raise RuntimeError(
            "No LLM credentials found. Set BOOKFORGE_OPENAI_API_KEY (NVIDIA NIM / "
            "OpenAI), GOOGLE_API_KEY, or ANTHROPIC_API_KEY."
        )

    warnings: list[str] = []
    if not shutil.which("ffmpeg"):
        warnings.append(
            "ffmpeg not found on PATH — video download and frame extraction "
            "will fail. Install it from https://ffmpeg.org/download.html"
        )
    if settings.compile_latex and not shutil.which("pdflatex"):
        warnings.append(
            "pdflatex not found on PATH while BOOKFORGE_COMPILE_LATEX is true — "
            "chapters cannot be verified by compilation. Install TeX Live/MiKTeX "
            "or set BOOKFORGE_COMPILE_LATEX=false to run in draft mode."
        )
    return warnings
