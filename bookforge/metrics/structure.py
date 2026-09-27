"""Structural verification of a written chapter.

Everything here is deterministic. That is the point: the original critic was an
LLM asked to check section presence, asset references and numeric fidelity by
reading — three jobs a regex does perfectly and a language model does
probabilistically. Moving them into code frees the critic LM to do the one job
only it can do (semantic judgement) and makes the verdict reproducible enough
to optimize against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bookforge.schemas import AssetsManifest, ChapterAnalysis, Defect
from bookforge.tools import latex as latex_tools

#: Sections the chapter contract requires, matched case-insensitively.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "Overview",
    "Learning Objectives",
    "Key Takeaways",
    "Glossary",
    "Exercises",
    "Solutions",
)

MIN_WORDS = 700
TARGET_WORDS = 1200
MIN_GLOSSARY_ITEMS = 4
MIN_EXERCISES = 4

_GLOSSARY_ITEM_RE = re.compile(
    r"\\bfsection\{Glossary\}(?P<body>.*?)(?=\\bfsection\{|\\section\{|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_EXERCISES_ITEM_RE = re.compile(
    r"\\bfsection\{Exercises\}(?P<body>.*?)(?=\\bfsection\{|\\section\{|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_SOLUTIONS_ITEM_RE = re.compile(
    r"\\bfsection\{Solutions\}(?P<body>.*?)(?=\\bfsection\{|\\section\{|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_DESC_ITEM_RE = re.compile(r"\\item\s*\[")
_ENUM_ITEM_RE = re.compile(r"\\item\b")
_EMPTY_PEDAGOGY_RE = re.compile(
    r"(?:no glossary|no exercises|no solutions|were (?:not )?provided|not provided in the analysis)",
    re.IGNORECASE,
)


@dataclass
class StructureReport:
    missing_sections: list[str] = field(default_factory=list)
    missing_assets: list[str] = field(default_factory=list)
    invented_assets: list[str] = field(default_factory=list)
    #: Paths the chapter referenced directly instead of via a declared label.
    raw_asset_references: list[str] = field(default_factory=list)
    unused_assets: list[str] = field(default_factory=list)
    unreferenced_floats: list[str] = field(default_factory=list)
    dangling_refs: list[str] = field(default_factory=list)
    duplicate_labels: list[str] = field(default_factory=list)
    missing_concepts: list[str] = field(default_factory=list)
    missing_diagrams: list[str] = field(default_factory=list)
    missing_derivations: list[str] = field(default_factory=list)
    glossary_items: int = 0
    exercise_items: int = 0
    solution_items: int = 0
    empty_pedagogy: list[str] = field(default_factory=list)
    word_count: int = 0
    has_chapter_command: bool = False

    @property
    def score(self) -> float:
        """0..1 structural completeness."""
        penalties = (
            0.30 * bool(not self.has_chapter_command)
            + 0.10 * min(len(self.missing_sections), 3) / 3
            + 0.25 * bool(self.invented_assets)
            + 0.15 * bool(self.missing_assets)
            + 0.10 * min(len(self.missing_concepts), 3) / 3
            + 0.05 * min(len(self.unreferenced_floats), 4) / 4
            + 0.05 * min(len(self.dangling_refs), 4) / 4
            + 0.05 * bool(self.duplicate_labels)
            + 0.08 * bool(self.raw_asset_references)
            + 0.10 * bool(self.word_count < MIN_WORDS)
            + 0.12 * bool(self.missing_diagrams)
            + 0.12 * bool(self.missing_derivations)
            + 0.15 * min(len(self.empty_pedagogy), 3) / 3
        )
        return round(max(0.0, 1.0 - penalties), 4)

    def defects(self) -> list[Defect]:
        """Every finding as an actionable defect, ordered by severity."""
        found: list[Defect] = []
        if not self.has_chapter_command:
            found.append(
                Defect(
                    severity="critical",
                    message="The body has no \\chapter command.",
                    fix="Begin the body with \\chapter{<title>}\\label{ch:...}.",
                )
            )
        if self.invented_assets:
            found.append(
                Defect(
                    severity="critical",
                    message=(
                        "These asset labels were placed but never declared: "
                        f"{', '.join(self.invented_assets)}. LaTeX will fail on them."
                    ),
                    fix=(
                        "Place only labels listed in the assets manifest. Every "
                        "declared asset is available; nothing else exists."
                    ),
                )
            )
        if self.missing_assets:
            found.append(
                Defect(
                    severity="critical",
                    message=(
                        "These referenced files do not exist on disk: "
                        f"{', '.join(self.missing_assets)}."
                    ),
                    fix="Reference only paths listed in the assets manifest.",
                )
            )
        if self.missing_sections:
            found.append(
                Defect(
                    severity="major",
                    message=f"Required sections are missing: {', '.join(self.missing_sections)}.",
                    fix=(
                        "Add each missing section in the contract's order, using "
                        "\\bfsection{<name>} for the unnumbered ones so they reach "
                        "the table of contents and the running head."
                    ),
                )
            )
        if self.missing_concepts:
            found.append(
                Defect(
                    severity="major",
                    message=(
                        "The analysis contains concepts the chapter never covers: "
                        f"{', '.join(self.missing_concepts)}."
                    ),
                    fix="Add one numbered \\section per missing concept.",
                )
            )
        if self.missing_diagrams:
            found.append(
                Defect(
                    severity="major",
                    message=(
                        f"{len(self.missing_diagrams)} diagram spec(s) from the "
                        f"analysis have no TikZ figure: {', '.join(self.missing_diagrams)}."
                    ),
                    fix=(
                        "Render each missing diagram in 'Conceptual Models' with "
                        "\\bfDiagram{caption}{label}{tikz body}, using the "
                        "modernbox/accentbox/neutralbox styles and flowarrow."
                    ),
                )
            )
        if self.missing_derivations:
            found.append(
                Defect(
                    severity="critical",
                    message=(
                        "These mathematical derivations are missing or too thin "
                        "(need a derivationbox with multi-line align expanding "
                        f"every analysis step): {', '.join(self.missing_derivations)}."
                    ),
                    fix=(
                        "Under Worked Derivations, write \\begin{derivationbox}{Title} "
                        "with assumptions, one prose sentence per analysis step, and "
                        "\\begin{align}...\\\\ ...\\end{align} with ≥2 equation lines "
                        "ending at formula_latex. A single S = ... line is not enough."
                    ),
                )
            )
        if self.empty_pedagogy:
            found.append(
                Defect(
                    severity="major",
                    message=(
                        "Pedagogical sections are empty or placeholder-only: "
                        + ", ".join(self.empty_pedagogy)
                        + "."
                    ),
                    fix=(
                        f"Fill Glossary with at least {MIN_GLOSSARY_ITEMS} "
                        f"\\item[Term] entries, and Exercises/Solutions with "
                        f"at least {MIN_EXERCISES} real questions and matching "
                        "answers. Never leave '% No glossary/exercises' comments."
                    ),
                )
            )
        if self.raw_asset_references:
            found.append(
                Defect(
                    severity="major",
                    message=(
                        "These assets were included with a raw path instead of a "
                        f"declared label: {', '.join(self.raw_asset_references)}."
                    ),
                    fix=(
                        "Replace each \\includegraphics/\\input with the marker for "
                        "its label, e.g. \\bfFigure{fig:ch01-frame2}. The marker "
                        "supplies the path, caption, width and placement."
                    ),
                )
            )
        if self.unused_assets:
            found.append(
                Defect(
                    severity="major",
                    message=(
                        "These generated assets were never included: "
                        f"{', '.join(self.unused_assets)}."
                    ),
                    fix=(
                        "Include every asset from the manifest — they were rendered "
                        "specifically for this chapter."
                    ),
                )
            )
        if self.dangling_refs:
            found.append(
                Defect(
                    severity="major",
                    message=f"\\ref points at undefined labels: {', '.join(self.dangling_refs)}.",
                    fix="Reference only labels defined in this chapter or the manifest.",
                )
            )
        if self.duplicate_labels:
            found.append(
                Defect(
                    severity="minor",
                    message=f"Labels defined more than once: {', '.join(self.duplicate_labels)}.",
                    fix="Give every float a unique label.",
                )
            )
        if self.unreferenced_floats:
            found.append(
                Defect(
                    severity="minor",
                    message=(
                        "These floats are never mentioned in the prose: "
                        f"{', '.join(self.unreferenced_floats)}."
                    ),
                    fix="Refer to each float in the running text (Figure~\\ref{...}).",
                )
            )
        if self.word_count < MIN_WORDS:
            found.append(
                Defect(
                    severity="major",
                    message=(
                        f"The chapter is {self.word_count} words; a chapter needs at "
                        f"least {MIN_WORDS} and targets {TARGET_WORDS}."
                    ),
                    fix=(
                        "Expand the concept sections using the analysis explanations "
                        "and key points. Do not pad with filler."
                    ),
                )
            )
        return found


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def check_structure(
    chapter_tex: str,
    analysis: ChapterAnalysis,
    manifest: AssetsManifest,
    existing_paths: set[str] | None = None,
) -> StructureReport:
    """Verify a chapter body against its contract, analysis and assets.

    ``existing_paths`` are workspace-relative paths known to exist on disk; when
    omitted, only manifest membership is checked.
    """
    report = StructureReport()
    tex = chapter_tex or ""

    report.has_chapter_command = bool(re.search(r"\\chapter\*?\{", tex))
    report.word_count = latex_tools.count_words(tex)

    sections = {_normalize(s) for s in latex_tools.find_sections(tex)}
    report.missing_sections = [
        name for name in REQUIRED_SECTIONS if _normalize(name) not in sections
    ]

    # --- assets ----------------------------------------------------------
    # Assets are placed by label, not by path: BookForge declares each one from
    # the manifest and the chapter places it with \bfFigure / \bfTable. So
    # "invented" now means a label that was never declared — which LaTeX itself
    # turns into a compile error, making this check a faster diagnosis of the
    # same failure rather than the only line of defence it used to be.
    manifest_labels = {f.label for f in manifest.figures} | {t.label for t in manifest.tables}
    placed = set(latex_tools.find_placed_assets(tex))
    report.invented_assets = sorted(placed - manifest_labels)
    report.unused_assets = sorted(manifest_labels - placed)

    # A writer that hand-rolls \includegraphics has bypassed the declaration
    # system, losing the shared width, the grounded caption and the placement
    # tuning. Treat any raw path reference as an invented asset unless it is a
    # declaration BookForge itself emitted.
    declared_paths = set(latex_tools.find_declared_assets(tex).values())
    raw_paths = set(latex_tools.find_referenced_files(tex)) - declared_paths
    report.raw_asset_references = sorted(raw_paths)

    if existing_paths is not None:
        # Verify the declared files are really on disk.
        placed_paths = {
            path
            for label, path in latex_tools.find_declared_assets(tex).items()
            if label in placed
        }
        report.missing_assets = sorted(
            path for path in placed_paths if path not in existing_paths
        )

    # Cross-references.
    labels = latex_tools.find_labels(tex)
    seen: set[str] = set()
    duplicates: set[str] = set()
    for label in labels:
        if label in seen:
            duplicates.add(label)
        seen.add(label)
    report.duplicate_labels = sorted(duplicates)

    # Table fragments carry their own \label, and the chapter only \input's
    # them — so a manifest label is defined even though it never appears in the
    # chapter body. Counting those as dangling would penalise the writer for
    # doing exactly what the contract asks.
    # \bfFigure / \bfTable emit the \label themselves, and \bfDiagram takes
    # its label as an argument, so a reference to any of those is defined even
    # though no literal \label appears in the body.
    defined = seen | manifest_labels | placed | set(
        latex_tools.find_diagram_labels(tex)
    )

    refs = set(latex_tools.find_refs(tex))
    report.dangling_refs = sorted(refs - defined)
    report.unreferenced_floats = sorted(
        label for label in manifest_labels if label in defined and label not in refs
    )

    # Coverage of the analysis.
    body_normalized = _normalize(tex)
    report.missing_concepts = [
        concept.heading
        for concept in analysis.concepts
        if _normalize(concept.heading) and _normalize(concept.heading) not in body_normalized
    ]

    tikz_count = len(latex_tools.find_diagram_labels(tex)) or len(
        re.findall(r"\\begin\{tikzpicture\}", tex)
    )
    if len(analysis.diagrams) > tikz_count:
        report.missing_diagrams = [
            d.title for d in analysis.diagrams[tikz_count:]
        ]

    report.missing_derivations = _missing_derivations(tex, analysis)

    report.glossary_items, report.exercise_items, report.solution_items = (
        _pedagogy_counts(tex)
    )
    report.empty_pedagogy = _empty_pedagogy_sections(
        tex,
        report.glossary_items,
        report.exercise_items,
        report.solution_items,
    )
    return report


def _section_body(tex: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(tex or "")
    return match.group("body") if match else ""


def _pedagogy_counts(tex: str) -> tuple[int, int, int]:
    glossary_body = _section_body(tex, _GLOSSARY_ITEM_RE)
    exercises_body = _section_body(tex, _EXERCISES_ITEM_RE)
    solutions_body = _section_body(tex, _SOLUTIONS_ITEM_RE)
    return (
        len(_DESC_ITEM_RE.findall(glossary_body)),
        len(_ENUM_ITEM_RE.findall(exercises_body)),
        len(_ENUM_ITEM_RE.findall(solutions_body)),
    )


_MATH_DISPLAY_RE = re.compile(
    r"\\begin\{(?P<env>equation|align|gather|multline)\*?\}\s*(?P<body>.*?)\\end\{(?P=env)\*?\}",
    re.DOTALL,
)


def _missing_derivations(tex: str, analysis: ChapterAnalysis) -> list[str]:
    """Titles of analysis derivations that never appear as real maths in the body."""
    if not analysis.derivations:
        return []
    body = latex_tools.strip_comments(tex or "")
    # Multi-step = at least one \\ line break inside align/gather, or ≥2
    # equality signs across display maths (a lone S = ... does not count).
    substantive = False
    for match in _MATH_DISPLAY_RE.finditer(body):
        env = match.group("env")
        math_body = match.group("body")
        linebreaks = len(re.findall(r"\\\\", math_body))
        equals = math_body.count("=")
        if env in {"align", "gather", "multline"} and linebreaks >= 1:
            substantive = True
            break
        if equals >= 2 and linebreaks >= 1:
            substantive = True
            break
    has_box = bool(re.search(r"\\begin\{derivationbox\}", body, re.IGNORECASE))
    missing: list[str] = []
    for derivation in analysis.derivations:
        title_key = _normalize(derivation.title)
        titled = bool(title_key) and title_key in _normalize(body)
        if not titled or not has_box:
            missing.append(derivation.title)
            continue
        if len(derivation.steps) >= 2 and not substantive:
            missing.append(derivation.title)
    return missing


def _empty_pedagogy_sections(
    tex: str,
    glossary_items: int,
    exercise_items: int,
    solution_items: int,
) -> list[str]:
    """Names of required pedagogy sections that have no real content."""
    empty: list[str] = []
    checks = (
        ("Glossary", glossary_items, MIN_GLOSSARY_ITEMS, _GLOSSARY_ITEM_RE),
        ("Exercises", exercise_items, MIN_EXERCISES, _EXERCISES_ITEM_RE),
        ("Solutions", solution_items, MIN_EXERCISES, _SOLUTIONS_ITEM_RE),
    )
    for name, count, minimum, pattern in checks:
        body = latex_tools.strip_comments(_section_body(tex, pattern)).strip()
        placeholder = bool(_EMPTY_PEDAGOGY_RE.search(body)) or not body
        if count < minimum or placeholder:
            empty.append(name)
    return empty
