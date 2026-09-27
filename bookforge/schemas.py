"""Typed contracts shared across the BookForge DSPy pipeline.

Design rule: **a schema that can silently corrupt data is a bug, not a
convenience.** The original ADK implementation padded ragged table rows and
truncated mismatched chart series inside the renderer, which meant a chart
could display three bars under five labels and nobody would ever know. Every
such case is now a hard validation error at the boundary, so a bad LLM output
is rejected and retried instead of being quietly typeset into the book.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Manifest (manifest.json)
# ---------------------------------------------------------------------------

#: Where a chapter's source material comes from. The pipeline dispatches media
#: acquisition on this, so a book can mix a YouTube playlist with local files.
SourceKind = Literal["youtube", "local_video", "transcript"]

VideoStatus = Literal[
    "pending",
    "media",
    "analyzed",
    "assets",
    "written",
    "verified",
    "failed",
]

#: Statuses that mean "this video already produced a chapter we trust".
TERMINAL_OK_STATUSES: frozenset[str] = frozenset({"verified"})

#: Statuses whose chapter.tex is good enough to include in the book.
INCLUDABLE_STATUSES: frozenset[str] = frozenset({"verified", "written"})


class VideoRecord(BaseModel):
    """One source recording and the state of its chapter."""

    video_id: str
    title: str
    url: str = ""
    #: How to acquire this record's media.
    source: SourceKind = "youtube"
    #: Absolute path for local sources; empty for YouTube.
    source_path: str = ""
    duration_sec: int = 0
    upload_date: str = ""
    chapter_number: int = 0
    status: VideoStatus = "pending"
    error: str = ""

    # Quality telemetry, filled in as the chapter progresses.
    qa_score: float = 0.0
    qa_iterations: int = 0
    grounding_score: float = 0.0
    compiled: bool = False

    @field_validator("video_id")
    @classmethod
    def _video_id_is_safe(cls, value: str) -> str:
        """video_id becomes a directory name; reject path traversal."""
        if not value or not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", value):
            raise ValueError(f"unsafe video_id: {value!r}")
        return value


class ChannelManifest(BaseModel):
    channel_url: str
    channel_title: str = ""
    channel_slug: str
    created_at: str = ""
    videos: list[VideoRecord] = Field(default_factory=list)

    def by_id(self, video_id: str) -> VideoRecord | None:
        return next((v for v in self.videos if v.video_id == video_id), None)

    def ordered(self) -> list[VideoRecord]:
        return sorted(self.videos, key=lambda v: v.chapter_number)


# ---------------------------------------------------------------------------
# Media assets
# ---------------------------------------------------------------------------


class FrameAsset(BaseModel):
    """A single extracted video frame plus the quality signals used to rank it."""

    file: str
    timestamp_sec: float
    phash: str
    sharpness: float = 0.0
    contrast: float = 0.0
    #: Composite 0..1 desirability used by frame curation.
    quality: float = 0.0


class TranscriptSegment(BaseModel):
    """One timestamped line of transcript. The unit of grounding evidence."""

    start_sec: float
    text: str

    @property
    def timestamp(self) -> str:
        total = int(self.start_sec)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


class MediaBundle(BaseModel):
    video_id: str
    video_path: str = ""
    audio_path: str = ""
    transcript_path: str = ""
    transcript_source: Literal["captions", "whisper", "none"] = "none"
    duration_sec: float = 0.0
    frames: list[FrameAsset] = Field(default_factory=list)
    #: Optional Gemini (or similar) visual briefing used by captioning/analysis.
    visual_insights: str = ""
    #: Fingerprint of media settings at acquisition time; cache miss when stale.
    settings_fingerprint: str = ""


# ---------------------------------------------------------------------------
# Chapter analysis (structured LLM output)
# ---------------------------------------------------------------------------


class ConceptNote(BaseModel):
    heading: str
    explanation: str = Field(
        description=(
            "3-8 sentences of real technical substance from the transcript. "
            "When the idea is mathematical, state the governing quantities and "
            "point at the matching derivation rather than waving at 'the formula'."
        )
    )
    key_points: list[str] = Field(default_factory=list)
    #: Transcript timestamps (HH:MM:SS) this concept was drawn from. Used by the
    #: grounding metric and by the frame captioner to align figures to concepts.
    evidence_timestamps: list[str] = Field(default_factory=list)

    @field_validator("explanation")
    @classmethod
    def _non_trivial(cls, value: str) -> str:
        if len(value.strip()) < 40:
            raise ValueError("explanation too short to be textbook-grade")
        return value


class TableSpec(BaseModel):
    title: str
    caption: str
    headers: list[str] = Field(min_length=1, max_length=8)
    rows: list[list[str]] = Field(min_length=1)
    note: str = ""

    @model_validator(mode="after")
    def _rectangular(self) -> TableSpec:
        """Ragged rows used to be silently padded. That corrupts the table."""
        width = len(self.headers)
        bad = [i for i, row in enumerate(self.rows) if len(row) != width]
        if bad:
            raise ValueError(
                f"table '{self.title}': rows {bad} have the wrong number of cells "
                f"(expected {width} to match headers {self.headers})"
            )
        return self


class ChartSpec(BaseModel):
    title: str
    caption: str
    chart_type: Literal["bar", "line", "pie"] = "bar"
    x_label: str = ""
    y_label: str = ""
    categories: list[str] = Field(min_length=1, max_length=24)
    series_name: str = "value"
    values: list[float] = Field(min_length=1, max_length=24)
    #: Where the numbers came from, so grounding can verify them.
    evidence_timestamps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _series_aligned(self) -> ChartSpec:
        """Mismatched lengths used to be truncated, silently mislabelling bars."""
        if len(self.categories) != len(self.values):
            raise ValueError(
                f"chart '{self.title}': {len(self.categories)} categories but "
                f"{len(self.values)} values — every value needs its own label"
            )
        if self.chart_type == "pie" and any(v < 0 for v in self.values):
            raise ValueError(f"chart '{self.title}': pie slices cannot be negative")
        return self


class DiagramSpec(BaseModel):
    title: str
    caption: str
    diagram_type: Literal["flowchart", "sequence", "hierarchy", "mindmap"] = "flowchart"
    elements: list[str] = Field(
        min_length=2,
        max_length=12,
        description="Ordered nodes/steps the diagram must show",
    )
    relationships: list[str] = Field(
        default_factory=list, description='Edges as "A -> B : label"'
    )

    @model_validator(mode="after")
    def _edges_reference_known_nodes(self) -> DiagramSpec:
        """An edge naming a node that does not exist renders as a dangling arrow."""
        known = {e.strip().lower() for e in self.elements}
        for edge in self.relationships:
            head = edge.split(":", 1)[0]
            parts = [p.strip().lower() for p in head.split("->")]
            unknown = [p for p in parts if p and p not in known]
            if unknown:
                raise ValueError(
                    f"diagram '{self.title}': edge {edge!r} references unknown "
                    f"node(s) {unknown}; elements are {self.elements}"
                )
        return self


class Exercise(BaseModel):
    question: str
    answer: str

    @model_validator(mode="after")
    def _answer_present(self) -> Exercise:
        if not self.answer.strip():
            raise ValueError(f"exercise {self.question!r} has no answer")
        return self


class GlossaryItem(BaseModel):
    term: str
    definition: str


class MathDerivation(BaseModel):
    """A formula the chapter must derive in full, not merely mention.

    Used when the talk involves equations, scales, losses, probabilities,
    bit-widths, complexity, etc. Empty ``derivations`` is fine for purely
    narrative product demos; it is not fine for quantization, ML theory,
    algorithms, or any topic whose claims rest on maths.
    """

    title: str = Field(description="Short name, e.g. 'Symmetric quantization scale'")
    motivation: str = Field(
        description="1-3 sentences: what this derives and why the reader needs it"
    )
    formula_latex: str = Field(
        description=(
            "Final result as LaTeX math body only (no $$ or equation env), "
            "e.g. s = (x_{max} - x_{min}) / (2^{b} - 1)"
        )
    )
    steps: list[str] = Field(
        min_length=2,
        max_length=12,
        description=(
            "Ordered derivation steps. Each item is prose that may embed inline "
            "math with $...$. Together they must reach formula_latex without gaps."
        ),
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Conditions required for the result (e.g. symmetric range)",
    )
    related_concept: str = Field(
        default="",
        description="Concept heading this derivation belongs under, if any",
    )


class ChapterAnalysis(BaseModel):
    """Validated structured analysis of one video. The writer's only source."""

    chapter_title: str
    summary: str = Field(description="One-paragraph executive summary")
    learning_objectives: list[str] = Field(min_length=2, max_length=8)
    concepts: list[ConceptNote] = Field(min_length=1, max_length=12)
    tables: list[TableSpec] = Field(default_factory=list, max_length=6)
    charts: list[ChartSpec] = Field(default_factory=list, max_length=6)
    diagrams: list[DiagramSpec] = Field(default_factory=list, max_length=4)
    derivations: list[MathDerivation] = Field(default_factory=list, max_length=8)
    glossary: list[GlossaryItem] = Field(default_factory=list, max_length=30)
    exercises: list[Exercise] = Field(default_factory=list, max_length=10)

    @property
    def glossary_dict(self) -> dict[str, str]:
        return {item.term: item.definition for item in self.glossary}

    def all_numbers(self) -> list[str]:
        """Every numeric literal asserted anywhere in the analysis.

        Captions and titles are included deliberately: a caption reading
        "costs roughly 17 percent of throughput" asserts a statistic just as
        much as a chart value does, and a derived figure the speaker never
        stated is exactly the kind of plausible invention worth catching.

        Returned in normalized form so ``940.0`` (a float chart value) and
        ``940`` (the same number in prose) compare as one claim.
        """
        blobs: list[str] = [self.summary, self.chapter_title]
        blobs += self.learning_objectives
        for concept in self.concepts:
            blobs.append(concept.heading)
            blobs.append(concept.explanation)
            blobs.extend(concept.key_points)
        for table in self.tables:
            blobs += [table.title, table.caption, table.note]
            blobs.extend(table.headers)
            for row in table.rows:
                blobs.extend(row)
        for chart in self.charts:
            blobs += [chart.title, chart.caption, chart.x_label, chart.y_label]
            blobs.extend(normalize_number(str(value)) for value in chart.values)
        for diagram in self.diagrams:
            blobs += [diagram.title, diagram.caption]
            blobs.extend(diagram.elements)
        for derivation in self.derivations:
            blobs += [derivation.title, derivation.motivation, derivation.formula_latex]
            blobs.extend(derivation.steps)
            blobs.extend(derivation.assumptions)
        for item in self.glossary:
            blobs.append(item.definition)
        return [
            normalize_number(n) for blob in blobs for n in extract_numbers(blob)
        ]


# ---------------------------------------------------------------------------
# Visual asset manifest
# ---------------------------------------------------------------------------


class FigureAsset(BaseModel):
    filename: str  # path relative to the workspace root
    kind: Literal["frame", "chart"]
    caption: str
    label: str
    timestamp_sec: float = 0.0


class TableAsset(BaseModel):
    tex_file: str
    caption: str
    label: str


class AssetsManifest(BaseModel):
    chapter_slug: str
    chapter_dir: str
    figures: list[FigureAsset] = Field(default_factory=list)
    tables: list[TableAsset] = Field(default_factory=list)

    @property
    def allowed_paths(self) -> set[str]:
        """Every path the writer is permitted to reference. Anything else is a defect."""
        return {f.filename for f in self.figures} | {t.tex_file for t in self.tables}

    def describe_for_prompt(self) -> str:
        """Compact, unambiguous listing handed to the writer LM."""
        lines: list[str] = []
        for fig in self.figures:
            lines.append(
                f"FIGURE path={fig.filename} label={fig.label} "
                f"kind={fig.kind} caption={fig.caption}"
            )
        for tab in self.tables:
            lines.append(
                f"TABLE  path={tab.tex_file} label={tab.label} caption={tab.caption}"
            )
        return "\n".join(lines) or "(no assets available for this chapter)"


# ---------------------------------------------------------------------------
# Chapter QA
# ---------------------------------------------------------------------------


class Defect(BaseModel):
    severity: Literal["critical", "major", "minor"]
    message: str
    fix: str = ""

    def __str__(self) -> str:
        tail = f" FIX: {self.fix}" if self.fix else ""
        return f"[{self.severity.upper()}] {self.message}{tail}"


class ChapterVerdict(BaseModel):
    """Result of verifying one chapter candidate. Produced deterministically."""

    #: True only when pdflatex actually ran and succeeded.
    compiles: bool = False
    #: True when compilation was not attempted (disabled, or pdflatex absent).
    #: Kept distinct from a failure so the pipeline never claims verification it
    #: did not perform, while still being able to make progress in draft mode.
    compile_skipped: bool = False
    compile_log: str = ""
    missing_assets: list[str] = Field(default_factory=list)
    invented_assets: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    ungrounded_numbers: list[str] = Field(default_factory=list)
    defects: list[Defect] = Field(default_factory=list)

    #: 0..1 composite. See metrics.composite for the weighting.
    score: float = 0.0

    @property
    def critical_defects(self) -> list[Defect]:
        return [d for d in self.defects if d.severity == "critical"]

    @property
    def is_acceptable(self) -> bool:
        """Good enough to stop refining.

        In draft mode (no pdflatex) compilation is skipped rather than failed,
        so a structurally sound chapter can still be accepted — otherwise every
        chapter would burn the full refinement budget for nothing. It is still
        never recorded as *verified*; see ``pipeline.chapter._status_for``.
        """
        return (self.compiles or self.compile_skipped) and not self.critical_defects

    def feedback_text(self) -> str:
        """Human/LM-readable critique used by the refiner and by GEPA."""
        if not self.defects:
            return "No defects found."
        order = {"critical": 0, "major": 1, "minor": 2}
        ranked = sorted(self.defects, key=lambda d: order[d.severity])
        return "\n".join(f"{i}. {d}" for i, d in enumerate(ranked, start=1))


class ChapterCandidate(BaseModel):
    """One attempt at a chapter, with its verdict. The loop keeps the best."""

    tex: str
    iteration: int
    verdict: ChapterVerdict

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.tex.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

#: Numbers worth verifying: integers, decimals, percentages, and figures with
#: thousands separators. Deliberately ignores 1-digit numbers ("3 steps") and
#: years, which are rarely the hallucination risk and create metric noise.
_NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{2,})(?![\w])")


def extract_numbers(text: str) -> list[str]:
    """Pull verifiable numeric literals out of prose."""
    return [m.group(1) for m in _NUMBER_RE.finditer(str(text or ""))]


def normalize_number(raw: str) -> str:
    """'1,234.50' -> '1234.5' so transcript and analysis compare fairly."""
    cleaned = str(raw).replace(",", "").strip()
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def coerce_analysis(value: object) -> ChapterAnalysis:
    if isinstance(value, ChapterAnalysis):
        return value
    if isinstance(value, str):
        return ChapterAnalysis.model_validate_json(value)
    return ChapterAnalysis.model_validate(value)


def coerce_assets_manifest(value: object) -> AssetsManifest:
    if isinstance(value, AssetsManifest):
        return value
    if isinstance(value, str):
        return AssetsManifest.model_validate_json(value)
    return AssetsManifest.model_validate(value)
