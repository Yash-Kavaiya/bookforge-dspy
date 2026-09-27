"""LaTeX: deterministic asset rendering, sanitization and verification.

The book owns exactly one preamble (defined here). Chapters are preamble-free
bodies, which lets any chapter be compiled standalone by wrapping it in that
same preamble — that is what makes verify-by-execution cheap enough to run on
every refinement candidate.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from bookforge.schemas import ChartSpec, TableSpec
from bookforge.workspace import Workspace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------

#: Characters that are syntactically special in LaTeX text mode.
_TEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

#: Unicode the LM emits that pdflatex cannot typeset with the shared preamble.
UNICODE_REPLACEMENTS = {
    "≥": r"$\ge$",
    "≤": r"$\le$",
    "≈": r"$\approx$",
    "≠": r"$\neq$",
    "×": r"$\times$",
    "±": r"$\pm$",
    "÷": r"$\div$",
    "∞": r"$\infty$",
    "→": r"$\to$",
    "←": r"$\leftarrow$",
    "↔": r"$\leftrightarrow$",
    "⇒": r"$\Rightarrow$",
    "⇐": r"$\Leftarrow$",
    "⇔": r"$\Leftrightarrow$",
    "∑": r"$\sum$",
    "∏": r"$\prod$",
    "√": r"$\sqrt{}$",
    "∈": r"$\in$",
    "µ": r"$\mu$",
    "α": r"$\alpha$",
    "β": r"$\beta$",
    "γ": r"$\gamma$",
    "δ": r"$\delta$",
    "λ": r"$\lambda$",
    "π": r"$\pi$",
    "σ": r"$\sigma$",
    "θ": r"$\theta$",
    "Ω": r"$\Omega$",
    "•": r"\textbullet{}",
    "—": "---",
    "–": "--",
    "“": "``",
    "”": "''",
    "‘": "`",
    "’": "'",
    "…": r"\dots{}",
    "™": r"\texttrademark{}",
    "®": r"\textregistered{}",
    "©": r"\textcopyright{}",
    "°": r"$^\circ$",
    "€": r"\texteuro{}",
    "£": r"\pounds{}",
    "\u00a0": " ",  # non-breaking space
    "\u200b": "",  # zero-width space
    "\ufeff": "",  # BOM
}


#: One lookup, applied in a single pass. Merging matters: escaping first and
#: replacing second would mangle the replacements' own backslashes and dollars,
#: and doing it the other way round would leave real specials unescaped.
_ESCAPE_MAP = {**UNICODE_REPLACEMENTS, **_TEX_SPECIALS}

#: Codepoints pdflatex + T1 fontenc can actually render: ASCII plus the Latin-1
#: Supplement / Latin Extended-A/B blocks (accented Western names and the
#: like). Everything above this — CJK, other scripts, and above all emoji,
#: which sit at U+1F000+ — has no glyph in the book's fonts. A title or a
#: chapter carrying one used to reach pdflatex raw and fail the entire build
#: on a single character (e.g. a flag emoji copied from a source video's
#: title: "LaTeX Error: Unicode character not set up for use with LaTeX").
_MAX_SAFE_CODEPOINT = 0x024F


def _drop_unsupported(text: str, *, context: str = "") -> str:
    """Strip characters with no glyph in the book's fonts.

    Known symbols are already handled by ``UNICODE_REPLACEMENTS``/``_ESCAPE_MAP``;
    this is the catch-all for whatever that table doesn't cover, so an
    unexpected character degrades the text instead of crashing pdflatex.
    """
    dropped: set[str] = set()

    def _keep(ch: str) -> bool:
        if ord(ch) <= _MAX_SAFE_CODEPOINT:
            return True
        dropped.add(ch)
        return False

    result = "".join(ch for ch in text if _keep(ch))
    if dropped:
        logger.warning(
            "dropped %d unsupported unicode character(s) %s from %s",
            len(dropped),
            " ".join(f"U+{ord(ch):04X}" for ch in sorted(dropped, key=ord)),
            context or "LaTeX output",
        )
    return result


def tex_escape(text: str) -> str:
    """Escape a plain string for LaTeX text mode.

    Applied to data we control (table cells, titles), never to LM-authored
    LaTeX — escaping that would destroy the markup. Characters outside the
    book's fonts (emoji, CJK, ...) are dropped rather than passed through raw.
    """
    escaped = "".join(_ESCAPE_MAP.get(ch, ch) for ch in str(text))
    return _drop_unsupported(escaped, context="a title or table")


# ---------------------------------------------------------------------------
# Tables (booktabs fragments, \input by the chapter)
# ---------------------------------------------------------------------------


def _column_spec(ncols: int) -> str:
    """Column layout that wraps rather than overflowing the margin."""
    if ncols <= 2:
        return "ll"
    usable = 0.92
    width = usable / ncols
    return "".join(f"p{{{width:.2f}\\textwidth}}" for _ in range(ncols))


def render_table_fragment(spec: TableSpec, dest: Path, label: str = "") -> Path:
    """Write a booktabs *tabular* fragment — no float, no caption, no label.

    The float is supplied by ``\\bfTable{<label>}`` at placement time, which is
    what lets BookForge control where tables land and keeps the caption in one
    place (the manifest) rather than two. A fragment that carried its own
    ``table`` environment would nest inside the one ``\\bfTable`` opens, and
    LaTeX aborts with "Not in outer par mode".

    ``TableSpec`` guarantees rectangular rows, so no padding or truncation
    happens here — a malformed table is rejected upstream instead of being
    silently repaired into a wrong one. ``label`` is accepted and ignored for
    call-site compatibility.
    """
    ncols = len(spec.headers)
    lines = [
        "% Generated by BookForge. Placed by \\bfTable{<label>}; do not add a float.",
        f"\\begin{{tabular}}{{{_column_spec(ncols)}}}",
        "  \\toprule",
        "  " + " & ".join(f"\\textbf{{{tex_escape(h)}}}" for h in spec.headers) + " \\\\",
        "  \\midrule",
    ]
    for row in spec.rows:
        lines.append("  " + " & ".join(tex_escape(cell) for cell in row) + " \\\\")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    if spec.note:
        # After \\end{tabular} we are in vertical mode. `\\\\` ("there's no
        # line here to end") aborts pdflatex and was failing every chapter
        # whose table carried a source note.
        lines.append(
            r"{\footnotesize\itshape " + tex_escape(spec.note) + "}"
        )
    return Workspace.write_text(Path(dest), "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Charts (matplotlib -> vector PDF)
# ---------------------------------------------------------------------------

_CHART_PALETTE = [
    "#2563eb",
    "#10b981",
    "#f59e0b",
    "#ef4444",
    "#8b5cf6",
    "#0891b2",
    "#db2777",
    "#65a30d",
]


def render_chart(spec: ChartSpec, dest: Path, overwrite: bool = False) -> Path:
    """Render a validated chart spec to a vector PDF.

    ``ChartSpec`` guarantees ``len(categories) == len(values)``, so unlike the
    original renderer this never truncates one series to fit the other — which
    silently mislabelled bars.
    """
    import matplotlib

    matplotlib.use("Agg")

    dest = Path(dest)
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    categories = list(spec.categories)
    values = list(spec.values)

    try:
        _render_to(dest, spec, spec.chart_type, categories, values)
    except Exception as exc:
        # A backend or library-version quirk in one chart type must not cost
        # the figure. The chapter already references this path, so failing here
        # would leave a dangling \includegraphics and break the whole build;
        # a bar chart carries the same data and always renders.
        if spec.chart_type == "bar":
            raise
        logger.warning(
            "could not render '%s' as a %s chart (%s); falling back to a bar chart",
            spec.title, spec.chart_type, exc,
        )
        _render_to(dest, spec, "bar", categories, values)
    return dest


def _render_to(
    dest: Path, spec: ChartSpec, kind: str, categories: list[str], values: list[float]
) -> None:
    """Draw one chart kind onto a fresh figure and save it.

    A fresh figure per attempt: a partially-constructed artist from a failed
    draw can survive ``Axes.clear`` and blow up again at save time.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    try:
        if kind == "pie":
            ax.pie(
                values,
                labels=categories,
                autopct="%1.1f%%",
                startangle=90,
                colors=_CHART_PALETTE[: len(values)],
                textprops={"fontsize": 8},
            )
            ax.axis("equal")
        else:
            if kind == "line":
                ax.plot(
                    categories, values, marker="o",
                    color=_CHART_PALETTE[0], linewidth=2,
                )
                ax.grid(True, linestyle=":", alpha=0.4)
            else:
                ax.bar(categories, values, color=_CHART_PALETTE[0], width=0.62)
                ax.grid(True, axis="y", linestyle=":", alpha=0.4)
                ax.set_axisbelow(True)
            _rotate_ticks(ax, categories)
            if spec.x_label:
                ax.set_xlabel(spec.x_label)
            ax.set_ylabel(spec.y_label or spec.series_name)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

        ax.set_title(spec.title, fontsize=11, pad=10)
        fig.tight_layout()
        fig.savefig(dest, format="pdf", bbox_inches="tight")
    finally:
        plt.close(fig)


def _rotate_ticks(ax, categories: list[str]) -> None:
    """Rotate long category labels so they never collide."""
    if any(len(c) > 8 for c in categories) or len(categories) > 6:
        ax.tick_params(axis="x", rotation=30)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")


# ---------------------------------------------------------------------------
# Book assembly
# ---------------------------------------------------------------------------

def default_preamble() -> str:
    """The preamble for the default theme.

    Design lives in ``bookforge.design``; this is the convenience accessor for
    code that does not care which theme is in force. Imported lazily to keep
    ``tools.latex`` free of a design dependency (design imports tex_escape from
    here).
    """
    from bookforge.design import build_preamble

    return build_preamble()


class _PreambleProxy(str):
    """Backwards-compatible ``latex.PREAMBLE``.

    Older call sites treat the preamble as a module constant. It is now theme
    data, so this evaluates the default theme once on first use rather than at
    import time, which would create a circular import.
    """

    _cached: str | None = None

    def __new__(cls):
        return super().__new__(cls, "")

    def _resolve(self) -> str:
        if _PreambleProxy._cached is None:
            _PreambleProxy._cached = default_preamble()
        return _PreambleProxy._cached

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self._resolve()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<PREAMBLE {len(self._resolve())} chars>"

    def __len__(self) -> int:
        return len(self._resolve())

    def __contains__(self, item) -> bool:
        return item in self._resolve()

    def __add__(self, other):
        return self._resolve() + other

    def __radd__(self, other):
        return other + self._resolve()

    def __iter__(self):
        return iter(self._resolve())

    def format(self, *args, **kwargs):  # pragma: no cover - trivial
        return self._resolve().format(*args, **kwargs)


PREAMBLE = _PreambleProxy()


def chapter_wrapper(preamble: str, body: str) -> str:
    """Standalone-compilable document used by the QA verifier."""
    return f"{preamble}\n\\begin{{document}}\n{body}\n\\end{{document}}\n"


def assemble_main_tex(
    book_title: str,
    author_line: str,
    chapter_rel_dirs: list[str],
    subtitle: str = "",
    source_line: str = "",
    stats_line: str = "",
    theme_name: str = "textbook",
) -> str:
    """Assemble the complete book: front matter, chapters, back matter.

    Front matter is a designed title page, a colophon recording provenance, the
    table of contents and (per theme) a list of figures. The old version called
    ``\\maketitle`` and stopped there, which produced a centred stack of text on
    an otherwise blank page and no way to tell where the book came from.
    """
    from bookforge.design import colophon, get_theme, title_page

    theme = get_theme(theme_name)
    inputs = "\n\n".join(f"\\input{{{rel}/chapter.tex}}" for rel in chapter_rel_dirs)

    front = [
        "\\frontmatter",
        "\\pagestyle{empty}",
        title_page(
            book_title,
            subtitle or "Compiled from recorded talks",
            author_line,
            source_line,
            theme,
        ),
        colophon(
            "Generated by BookForge, a DSPy pipeline for turning recorded talks "
            "into typeset books.",
            source_line or "Source: recorded video.",
            stats_line or "",
        ),
        "\\pagestyle{bookforge}",
        "\\tableofcontents",
    ]
    if theme.include_list_of_figures:
        # Only worth a page when the book actually has figures to list.
        front.append("\\cleardoublepage\n\\listoffigures")

    back = ["\\backmatter"]
    if theme.include_index:
        # printindex is a no-op without a .ind file, so a book built in one pass
        # (no makeindex run) still compiles cleanly.
        back.append("\\IfFileExists{\\jobname.ind}{\\printindex}{}")

    return (
        "\\input{preamble.tex}\n"
        f"\\setbookforgetitle{{{tex_escape(book_title)}}}\n"
        "\\begin{document}\n"
        + "\n".join(front)
        + "\n\n\\mainmatter\n\n"
        + inputs
        + "\n\n"
        + "\n".join(back)
        + "\n\\end{document}\n"
    )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


@dataclass
class CompileResult:
    """Structured outcome of a pdflatex run."""

    ok: bool
    errors: str = ""
    log_tail: str = ""
    overfull_boxes: int = 0
    undefined_references: list[str] = field(default_factory=list)
    skipped: bool = False

    @property
    def clean(self) -> bool:
        """Compiles with no overfull boxes and no dangling \\ref."""
        return self.ok and not self.overfull_boxes and not self.undefined_references


def pdflatex_available() -> bool:
    return shutil.which("pdflatex") is not None


# pdflatex reports errors two ways. Without -file-line-error they start with
# "!"; with it (which we pass, because the file and line are worth having) they
# come as "path/file.tex:42: LaTeX Error: ...". Matching only the first form
# meant every error since we added the flag fell through to the log tail — so
# the critic and the refiner were handed font-loading noise instead of the
# actual error, on every failure.
_ERROR_RE = re.compile(
    r"^(?:!|[^\s:]+\.\w+:\d+:).*(?:\n(?!\s*$).*){0,4}", re.MULTILINE
)
_OVERFULL_RE = re.compile(r"^Overfull \\hbox", re.MULTILINE)
_UNDEFINED_REF_RE = re.compile(r"Reference [`'](?P<ref>[^']+)' on page \d+ undefined")


def extract_latex_errors(log: str, max_errors: int = 6) -> str:
    """Pull the '!' error blocks out of a pdflatex log, dropping the noise."""
    errors = _ERROR_RE.findall(log or "")
    if not errors:
        return (log or "")[-1500:]
    return "\n---\n".join(err.strip() for err in errors[:max_errors])


def makeindex_available() -> bool:
    return shutil.which("makeindex") is not None


#: The writer calls \term{} sparingly — most chapters mark only a couple of
#: concepts — so a book-wide index frequently ends up with two or three
#: entries. That reads as unfinished rather than useful, especially as the
#: very last page of the book, so an index thinner than this is dropped
#: rather than printed.
_MIN_INDEX_ENTRIES = 8


def _build_index(build_dir: Path, stem: str, timeout_sec: int) -> bool:
    """Run makeindex over ``stem.idx`` if there is one. Never fatal.

    An index is a nicety; a book without one is still a book. A failure here
    must not cost the compile, so this reports and moves on. A sparse index
    is worse than no index — closing the book on a page with two entries
    looks broken, not helpful — so a thin result is discarded rather than
    printed; ``\\printindex`` in the preamble already no-ops when ``.ind``
    doesn't exist, so removing the file is enough to suppress the page.
    """
    idx = build_dir / f"{stem}.idx"
    if not idx.exists() or idx.stat().st_size == 0 or not makeindex_available():
        return False
    try:
        proc = subprocess.run(
            # Relative name, run from build_dir: TeX's paranoid openout_any
            # setting refuses to write an output file named by absolute path,
            # so passing the full path fails with "Not writing to ...".
            ["makeindex", "-q", idx.name],
            cwd=build_dir,
            capture_output=True, text=True, errors="replace",
            timeout=timeout_sec, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.info("makeindex failed for %s: %s", idx.name, exc)
        return False
    if proc.returncode != 0:
        logger.info("makeindex returned %s: %s", proc.returncode, proc.stderr[:400])
        return False
    ind = build_dir / f"{stem}.ind"
    if not ind.exists():
        return False
    entries = ind.read_text(encoding="utf-8", errors="replace").count(r"\item ")
    if entries < _MIN_INDEX_ENTRIES:
        logger.info(
            "index has only %d entr%s; dropping it instead of printing a near-empty page",
            entries, "y" if entries == 1 else "ies",
        )
        ind.unlink(missing_ok=True)
        return False
    return True


def compile_tex(
    main_tex: Path,
    build_dir: Path,
    passes: int = 2,
    timeout_sec: int = 180,
    cwd: Path | None = None,
) -> CompileResult:
    """Compile ``main_tex``, returning a structured result.

    ``cwd`` controls how relative paths inside the document resolve. BookForge
    always compiles with cwd = workspace root so a chapter's asset paths behave
    identically whether it is built standalone during QA or as part of the book.
    """
    main_tex = Path(main_tex)
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    if not pdflatex_available():
        return CompileResult(ok=False, errors="pdflatex not found on PATH", skipped=True)

    work_cwd = Path(cwd) if cwd else main_tex.parent
    main_ref = str(main_tex) if main_tex.is_absolute() else main_tex.name

    last_log = ""
    total_passes = max(1, passes)
    for _pass in range(total_passes):
        try:
            proc = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-file-line-error",
                    f"-output-directory={build_dir}",
                    main_ref,
                ],
                cwd=work_cwd,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CompileResult(
                ok=False,
                errors=(
                    f"pdflatex timed out after {timeout_sec}s — usually an "
                    "unterminated environment or a runaway TikZ definition"
                ),
            )
        last_log = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        if proc.returncode != 0:
            return CompileResult(
                ok=False,
                errors=extract_latex_errors(last_log),
                log_tail=last_log[-2000:],
            )

        # The index has to be built between passes: \index writes to .idx, and
        # \printindex reads the .ind that makeindex produces from it. Without
        # this the whole index chain is inert -- \term{} dutifully records
        # every entry, \printindex finds no .ind and prints nothing, and the
        # book ships with no index and no error anywhere.
        if _pass < total_passes - 1:
            _build_index(build_dir, main_tex.stem, timeout_sec)

    # LaTeX resolves \ref through the .aux file, which is only read back on the
    # *next* pass. After a single pass every forward reference is "undefined",
    # so reporting them would be a guaranteed false positive — the QA check
    # compiles once for speed and must not invent defects because of it.
    undefined: list[str] = []
    if passes > 1:
        undefined = sorted(
            {m.group("ref") for m in _UNDEFINED_REF_RE.finditer(last_log)}
        )

    return CompileResult(
        ok=True,
        log_tail=last_log[-2000:],
        overfull_boxes=len(_OVERFULL_RE.findall(last_log)),
        undefined_references=undefined,
    )


# ---------------------------------------------------------------------------
# Static inspection
# ---------------------------------------------------------------------------

_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
_REF_RE = re.compile(r"\\(?:ref|autoref|cref|eqref)\{([^}]+)\}")
# \bfsection is BookForge's unnumbered-section macro (adds to TOC and running
# head); it must count as a section or every required unnumbered heading reads
# as missing.
_SECTION_RE = re.compile(r"\\(?:bf|sub)?section\*?\{([^}]+)\}")
_VERBATIM_RE = re.compile(
    r"\\begin\{(verbatim|lstlisting|minted)\}.*?\\end\{\1\}", re.DOTALL
)


#: A LaTeX comment runs from an unescaped % to end of line. Static inspection
#: must ignore comments, or a comment that *documents* a macro — such as the
#: header BookForge writes above its own asset declarations — is parsed as a
#: use of it.
_COMMENT_RE = re.compile(r"(?<!\\)%.*?$", re.MULTILINE)


def strip_comments(tex_body: str) -> str:
    """Remove LaTeX comments, preserving escaped percent signs."""
    return _COMMENT_RE.sub("", tex_body or "")


def find_referenced_files(tex_body: str) -> list[str]:
    """Every asset path the chapter references via includegraphics or input."""
    body = strip_comments(tex_body)
    return _INCLUDEGRAPHICS_RE.findall(body) + _INPUT_RE.findall(body)


#: Asset placement markers. The chapter references declared assets by label
#: rather than by path, so these are what "did the writer place the figure?"
#: actually means now.
_BFFIGURE_RE = re.compile(r"\\bfFigure(?:Here)?(?:\[[^\]]*\])?\{([^}]+)\}")
_BFPAIR_RE = re.compile(r"\\bfFigurePair(?:\[[^\]]*\])?\{([^}]+)\}\{([^}]+)\}")
_BFTABLE_RE = re.compile(r"\\bfTable(?:\[[^\]]*\])?\{([^}]+)\}")
_BFDIAGRAM_RE = re.compile(r"\\bfDiagram\{[^}]*\}\{([^}]+)\}")
_BFDECLARE_FIG_RE = re.compile(r"\\bfDeclareFigure\{([^}]+)\}\{([^}]+)\}")
_BFDECLARE_TAB_RE = re.compile(r"\\bfDeclareTable\{([^}]+)\}\{([^}]+)\}")
#: Full declaration lines, including optional caption/width args. Models
#: occasionally invent these; BookForge owns them and must be the only source.
_BFDECLARE_LINE_RE = re.compile(
    r"^[ \t]*\\bfDeclare(?:Figure|Table)\{.*?\}[^\n]*\n?",
    re.MULTILINE,
)
_ASSET_DECL_HEADER_RE = re.compile(
    r"^[ \t]*% Asset declarations for this chapter[^\n]*\n"
    r"(?:^[ \t]*%[^\n]*\n)*",
    re.MULTILINE,
)


def find_placed_assets(tex_body: str) -> list[str]:
    """Labels the chapter places with a marker, in document order."""
    body = strip_comments(tex_body)
    placed = _BFFIGURE_RE.findall(body) + _BFTABLE_RE.findall(body)
    for left, right in _BFPAIR_RE.findall(body):
        placed.extend((left, right))
    return placed


def find_declared_assets(tex_body: str) -> dict[str, str]:
    """label -> path for every asset declared in the body."""
    body = strip_comments(tex_body)
    declared = dict(_BFDECLARE_FIG_RE.findall(body))
    declared.update(dict(_BFDECLARE_TAB_RE.findall(body)))
    return declared


def strip_asset_declarations(tex_body: str) -> str:
    """Remove BookForge/LM asset declaration lines from a chapter body.

    Declarations are always regenerated from ``assets.json`` before verify and
    before persist, so anything already in the body is either a stale copy or
    an LM hallucination.
    """
    body = tex_body or ""
    body = _ASSET_DECL_HEADER_RE.sub("", body)
    body = _BFDECLARE_LINE_RE.sub("", body)
    return body.lstrip("\n")


def find_diagram_labels(tex_body: str) -> list[str]:
    """Labels of inline TikZ diagrams wrapped in \\bfDiagram."""
    return _BFDIAGRAM_RE.findall(strip_comments(tex_body))


def find_labels(tex_body: str) -> list[str]:
    return _LABEL_RE.findall(strip_comments(tex_body))


def find_refs(tex_body: str) -> list[str]:
    return _REF_RE.findall(strip_comments(tex_body))


def find_sections(tex_body: str) -> list[str]:
    return [s.strip() for s in _SECTION_RE.findall(strip_comments(tex_body))]


def count_words(tex_body: str) -> int:
    """Approximate prose word count, excluding markup and verbatim blocks."""
    text = _VERBATIM_RE.sub(" ", tex_body)
    text = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}$&%#_~^\\]", " ", text)
    return len([w for w in text.split() if any(ch.isalpha() for ch in w)])


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:latex|tex)?\s*\n(?P<body>.*?)\n?```\s*$", re.DOTALL)
_PREAMBLE_LINE_RE = re.compile(
    r"^[ \t]*\\(?:documentclass|usepackage|usetikzlibrary|tikzset|definecolor)"
    r"(?:\[[^\]]*\])?(?:\{[^}]*\})*[^\n]*\n?",
    re.MULTILINE,
)
_DOC_ENV_RE = re.compile(r"\\(?:begin|end)\{document\}")
_REDUNDANT_TABLE_WRAP_RE = re.compile(
    r"\\begin\{table\}(?:\[[^\]]*\])?\s*"
    r"(?:\\centering\s*)?"
    r"(?:\\caption\{[^}]*\}\s*)?"
    r"(?:\\label\{[^}]*\}\s*)?"
    r"(?P<inner>\\input\{[^}]*tables/[^}]*\.tex\})\s*"
    r"(?:\\caption\{[^}]*\}\s*)?"
    r"(?:\\label\{[^}]*\}\s*)?"
    r"\\end\{table\}"
)
_HTML_TAG_RE = re.compile(
    r"<\s*/?\s*(?:div|span|p|section|article|br|ul|ol|li|h[1-6])[^>]*>", re.IGNORECASE
)
_MALFORMED_ENV_RE = re.compile(r"\\(begin|end)\{([A-Za-z*]+)>")


def _protect_verbatim(tex: str) -> tuple[str, list[str]]:
    """Swap verbatim blocks for placeholders so sanitizers cannot corrupt them.

    A chapter about LaTeX legitimately contains ``\\usepackage`` inside a
    verbatim block; the original sanitizer stripped those lines and destroyed
    the example it was teaching.
    """
    blocks: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        blocks.append(match.group(0))
        return f"@@BOOKFORGE_VERBATIM_{len(blocks) - 1}@@"

    return _VERBATIM_RE.sub(_stash, tex), blocks


def _restore_verbatim(tex: str, blocks: list[str]) -> str:
    for index, block in enumerate(blocks):
        tex = tex.replace(f"@@BOOKFORGE_VERBATIM_{index}@@", block)
    return tex


def sanitize_chapter_tex(tex: str) -> str:
    """Normalize LM-authored LaTeX into a safe, preamble-free chapter body.

    Every transformation here repairs a failure mode observed in production.
    Verbatim blocks are protected throughout so code examples survive intact.
    """
    body = (tex or "").strip()
    if not body:
        return ""

    fence = _FENCE_RE.match(body)
    if fence:
        body = fence.group("body").strip()

    body, verbatim_blocks = _protect_verbatim(body)

    # The book owns the preamble; never trust LM-supplied preamble material.
    body = _PREAMBLE_LINE_RE.sub("", body)
    body = _DOC_ENV_RE.sub("", body)

    # HTML leakage: models occasionally emit markup from their pretraining.
    body = re.sub(r"<\s*/\s*figure\s*>", r"\\end{figure}", body, flags=re.IGNORECASE)
    body = re.sub(r"<\s*figure\s*>", r"\\begin{figure}[htbp]", body, flags=re.IGNORECASE)
    body = re.sub(r"<\s*/\s*table\s*>", r"\\end{table}", body, flags=re.IGNORECASE)
    body = re.sub(r"<\s*table\s*>", r"\\begin{table}[htbp]", body, flags=re.IGNORECASE)
    body = _HTML_TAG_RE.sub("", body)
    body = _MALFORMED_ENV_RE.sub(r"\\\1{\2}", body)

    # Table fragments already carry their own table environment.
    body = _REDUNDANT_TABLE_WRAP_RE.sub(lambda m: m.group("inner"), body)

    # TikZ node labels containing \\ need align=center or LaTeX aborts.
    body = _ensure_tikz_alignment(body)

    for char, repl in UNICODE_REPLACEMENTS.items():
        body = body.replace(char, repl)

    # Catch-all for whatever that table doesn't cover — every character left
    # is either plain ASCII (all LaTeX syntax is) or genuinely unsupported.
    body = _drop_unsupported(body, context="a chapter body")

    body = _restore_verbatim(body, verbatim_blocks)
    return body.strip() + "\n"


_TIKZ_BEGIN_RE = re.compile(r"\\begin\{tikzpicture\}(?:\[(?P<opts>[^\]]*)\])?")


def _ensure_tikz_alignment(body: str) -> str:
    def _fix(match: re.Match[str]) -> str:
        opts = match.group("opts") or ""
        if "align=center" in opts:
            return match.group(0)
        prefix = f"{opts}, " if opts.strip() else ""
        return (
            "\\begin{tikzpicture}["
            + prefix
            + "every node/.append style={align=center}]"
        )

    return _TIKZ_BEGIN_RE.sub(_fix, body)


def strip_duplicate_labels(tex: str, seen: set[str]) -> tuple[str, list[str]]:
    """Rename labels already used elsewhere in the book.

    Duplicate ``\\label`` keys make pdflatex resolve every ``\\ref`` to the last
    definition, so a cross-reference silently points at the wrong figure. This
    renames collisions and rewrites the matching refs within the same body.
    """
    renamed: list[str] = []
    mapping: dict[str, str] = {}
    for label in find_labels(tex):
        if label in seen and label not in mapping:
            suffix = 2
            while f"{label}-{suffix}" in seen:
                suffix += 1
            mapping[label] = f"{label}-{suffix}"
            seen.add(mapping[label])
            renamed.append(label)
        else:
            seen.add(label)
    for old, new in mapping.items():
        tex = tex.replace(f"\\label{{{old}}}", f"\\label{{{new}}}")
        for cmd in ("ref", "autoref", "cref", "eqref"):
            tex = tex.replace(f"\\{cmd}{{{old}}}", f"\\{cmd}{{{new}}}")
    return tex, renamed
