"""Pipeline orchestration — plain Python, no agent framework.

``run_pipeline`` is the entry point: intake, per-chapter production with media
prefetch, and book compilation, with per-video error isolation throughout.
"""

# Force numpy to fully resolve before dspy's lazy-loader can stub
# sys.modules["numpy"]. ``assets`` below imports the captioner (dspy) before
# ``frames`` imports imagehash, which does `import numpy.typing` — against
# dspy's proxy, which crashes with "TypeError: data type 'bool' not
# understood" (the proxy never runs numpy's real __init__, so numpy._core's
# submodules initialize out of order).
#
# A bare `import numpy` is not enough: if something else already did
# `import dspy` first (e.g. another test module, collected earlier by
# pytest), sys.modules["numpy"] already holds dspy's *unresolved* proxy, and
# a plain import is then a no-op — only touching an attribute forces the
# proxy to load the real module and replace itself in sys.modules. This is
# the actual choke point: every consumer of bookforge.pipeline.* (the CLI,
# the web API, pytest collection, ...) comes through this __init__ first, so
# fixing it here covers all of them — see the matching guard in cli.py for
# the one entrypoint that used to need its own copy.
import numpy

_ = numpy.__version__

from bookforge.pipeline.assets import build_assets
from bookforge.pipeline.chapter import ChapterOutcome, produce_chapter
from bookforge.pipeline.compile import book_pdf_path, compile_book
from bookforge.pipeline.intake import extract_channel_url, run_intake
from bookforge.pipeline.media import acquire_media
from bookforge.pipeline.orchestrator import RunResult, run_pipeline
from bookforge.pipeline.progress import (
    PipelineEvent,
    ProgressTracker,
    Reporter,
    format_duration,
)

__all__ = [
    "ChapterOutcome",
    "PipelineEvent",
    "ProgressTracker",
    "Reporter",
    "RunResult",
    "acquire_media",
    "book_pdf_path",
    "build_assets",
    "compile_book",
    "extract_channel_url",
    "format_duration",
    "produce_chapter",
    "run_intake",
    "run_pipeline",
]
