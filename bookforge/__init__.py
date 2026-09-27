"""BookForge — YouTube channels into publication-grade LaTeX textbooks.

Built on DSPy: every LLM stage is a typed, optimizable module rather than a
prompt string, and every chapter is verified by execution before it ships.

Quick start::

    from bookforge import run_pipeline
    result = run_pipeline("https://www.youtube.com/@channel")
"""

from bookforge.config import Settings, configure_logging, get_settings, preflight
from bookforge.schemas import (
    AssetsManifest,
    ChannelManifest,
    ChapterAnalysis,
    ChapterVerdict,
    VideoRecord,
)

__version__ = "1.0.0"

__all__ = [
    "AssetsManifest",
    "ChannelManifest",
    "ChapterAnalysis",
    "ChapterVerdict",
    "Settings",
    "VideoRecord",
    "__version__",
    "configure_logging",
    "get_settings",
    "preflight",
    "run_pipeline",
]


def __getattr__(name: str):
    """Lazily expose the pipeline so `import bookforge` stays cheap.

    ``run_pipeline`` pulls in DSPy, matplotlib and yt-dlp; importing the package
    to read a schema should not pay that cost.
    """
    if name == "run_pipeline":
        from bookforge.pipeline.orchestrator import run_pipeline

        return run_pipeline
    raise AttributeError(f"module 'bookforge' has no attribute {name!r}")
