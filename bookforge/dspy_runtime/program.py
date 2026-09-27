"""BookForgeProgram — every LLM stage, wired and optimization-aware.

This is the single object the pipeline talks to. It owns the four DSPy modules,
assigns each its role-specific LM, and loads any compiled artifacts on startup
so production runs use optimized instructions without a code change.
"""

from __future__ import annotations

import logging
from pathlib import Path

import dspy

from bookforge.config import Settings, get_settings
from bookforge.dspy_runtime.artifacts import load_program, save_program
from bookforge.dspy_runtime.lm import LMRouter
from bookforge.dspy_runtime.modules import (
    ChapterProgram,
    FrameCaptioner,
    TranscriptAnalyst,
)

logger = logging.getLogger(__name__)

#: Artifact name per optimizable component.
ARTIFACT_NAMES = {
    "analyst": "analyst",
    "chapter": "chapter",
    "captioner": "captioner",
}


class BookForgeProgram:
    """Container for the pipeline's DSPy modules with per-role LMs."""

    def __init__(
        self,
        settings: Settings | None = None,
        router: LMRouter | None = None,
        load_optimized: bool | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.router = router or LMRouter(self.settings)

        self.analyst = TranscriptAnalyst(self.settings)
        self.captioner = FrameCaptioner(self.settings)
        self.chapter = ChapterProgram(self.settings)

        self._assign_lms()

        should_load = (
            self.settings.load_optimized if load_optimized is None else load_optimized
        )
        self.loaded_artifacts: list[str] = []
        if should_load:
            self.load_artifacts(self.settings.artifacts_dir_abs)

    # -- LM assignment ------------------------------------------------------

    def _assign_lms(self) -> None:
        """Give each stage its own model and temperature.

        ``set_lm`` applies to every predictor inside a module, so the analyst's
        chunk, merge and single-shot predictors all run cold together.
        """
        self.analyst.set_lm(self.router.get("analyst"))
        self.captioner.set_lm(self.router.get("caption"))
        # The chapter loop mixes roles: writing wants variety, judging does not.
        self.chapter.writer.set_lm(self.router.get("writer"))
        self.chapter.refiner.set_lm(self.router.get("writer"))
        if self.chapter.critic is not None:
            self.chapter.critic.set_lm(self.router.get("critic"))

    # -- artifacts ----------------------------------------------------------

    def components(self) -> dict[str, dspy.Module]:
        return {
            "analyst": self.analyst,
            "captioner": self.captioner,
            "chapter": self.chapter,
        }

    def load_artifacts(self, directory: Path, strict: bool = False) -> list[str]:
        """Load every available compiled artifact. Returns the names applied."""
        directory = Path(directory)
        loaded: list[str] = []
        for key, module in self.components().items():
            if load_program(module, directory, ARTIFACT_NAMES[key], strict=strict):
                loaded.append(key)
        # Re-assign LMs: loading state can replace predictor objects.
        self._assign_lms()
        self.loaded_artifacts = loaded
        if loaded:
            logger.info("running with optimized: %s", ", ".join(loaded))
        else:
            logger.info(
                "running with authored instructions (no artifacts in %s)", directory
            )
        return loaded

    def save_artifact(self, key: str, directory: Path, **metadata) -> Path:
        module = self.components()[key]
        return save_program(module, directory, ARTIFACT_NAMES[key], **metadata)

    def describe(self) -> str:
        rows = [
            f"  analyst    {self.settings.analyst_model}",
            f"  writer     {self.settings.writer_model}",
            f"  critic     {self.settings.critic_model}",
            f"  captioner  {self.settings.caption_model}",
        ]
        optimized = ", ".join(self.loaded_artifacts) or "none"
        return "BookForge program:\n" + "\n".join(rows) + f"\n  optimized: {optimized}"
