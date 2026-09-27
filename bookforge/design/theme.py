"""Book design: page furniture, typography, callouts and float behaviour.

The ADK preamble was a working preamble, not a designed one: `article`-grade
margins on a `book` class, no running heads, no footers, default chapter
openers, and float parameters left at LaTeX's 1980s defaults — which is why
figures drifted to the end of chapters and landed in clumps.

Everything here is parameterized by ``BookTheme`` so the look is data rather
than a hand-edited string, and every knob has a reason attached.

The chapter body stays preamble-free. This module owns the preamble; chapters
own their content. That separation is what lets a single chapter be compiled
standalone during QA using exactly the preamble the book will use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PaperSize = Literal["a4paper", "letterpaper"]


@dataclass(frozen=True)
class Palette:
    """Colours, as RGB triples. Print-safe: nothing relies on hue alone."""

    primary: tuple[int, int, int] = (30, 64, 175)  # deep blue — chapter furniture
    accent: tuple[int, int, int] = (13, 148, 136)  # teal — links, highlights
    ink: tuple[int, int, int] = (23, 30, 43)  # body text
    muted: tuple[int, int, int] = (100, 116, 139)  # captions, running heads
    rule: tuple[int, int, int] = (203, 213, 225)  # hairlines
    wash: tuple[int, int, int] = (243, 246, 252)  # callout backgrounds
    wash_accent: tuple[int, int, int] = (236, 253, 250)
    wash_warn: tuple[int, int, int] = (255, 247, 237)
    warn: tuple[int, int, int] = (194, 65, 12)

    def definitions(self) -> str:
        rows = [
            ("bfPrimary", self.primary),
            ("bfAccent", self.accent),
            ("bfInk", self.ink),
            ("bfMuted", self.muted),
            ("bfRule", self.rule),
            ("bfWash", self.wash),
            ("bfWashAccent", self.wash_accent),
            ("bfWashWarn", self.wash_warn),
            ("bfWarn", self.warn),
        ]
        return "\n".join(
            f"\\definecolor{{{name}}}{{RGB}}{{{r}, {g}, {b}}}" for name, (r, g, b) in rows
        )


@dataclass(frozen=True)
class BookTheme:
    """A complete book design."""

    name: str = "textbook"
    palette: Palette = field(default_factory=Palette)
    paper: PaperSize = "a4paper"
    base_font_pt: int = 11

    #: Two-sided printing mirrors the margins and alternates running heads.
    #: Chapters open on the next page (``openany``), not the next recto, so a
    #: short chapter does not burn a blank verso.
    two_sided: bool = True
    #: Margin notes still need an outer column, but it is only as wide as the
    #: note itself. The rest of the page belongs to the text.
    margin_notes: bool = True

    chapter_opener: Literal["banded", "plain", "minimal"] = "banded"
    running_heads: bool = True
    footer_rule: bool = True
    show_book_title_in_footer: bool = True

    include_list_of_figures: bool = True
    include_index: bool = True

    def geometry(self) -> str:
        """Page geometry.

        Margins stay above a typical printer's unprintable edge and leave a
        small binding allowance, but they are tight on purpose: a textbook that
        sits in a 28–45 mm frame wastes the page. The outer margin is only as
        wide as the margin note plus a few millimetres of trim. ``marginparwidth``
        must be declared to geometry or notes overflow that trim.
        """
        # headheight stays at 15pt so a footnotesize running head does not
        # trigger fancyhdr's "headheight is too small" warning.
        furniture = "headheight=15pt, headsep=4.5mm, footskip=10mm"
        if self.margin_notes:
            spec = (
                "inner=15mm, outer=24mm, top=16mm, bottom=18mm, "
                "marginparwidth=16mm, marginparsep=2.5mm, "
                f"{furniture}"
            )
        else:
            spec = f"inner=15mm, outer=15mm, top=16mm, bottom=18mm, {furniture}"
        return f"\\usepackage[{self.paper}, {spec}]{{geometry}}"

    def class_options(self) -> str:
        options = [f"{self.base_font_pt}pt"]
        # openany: do not insert a blank verso just so the next chapter starts
        # on a recto. Running heads still alternate when two_sided is set.
        options.append("twoside,openany" if self.two_sided else "oneside,openany")
        return ",".join(options)


# ---------------------------------------------------------------------------
# Preamble sections
# ---------------------------------------------------------------------------

_FONTS = r"""
% --- Fonts -----------------------------------------------------------------
% lmodern must precede fontenc: T1 otherwise selects the bitmap EC fonts, and
% microtype's font expansion is a FATAL error on non-scalable fonts.
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{textcomp}
\usepackage[final]{microtype}
"""

_MATH_AND_GRAPHICS = r"""
% --- Maths and graphics -----------------------------------------------------
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{adjustbox}
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
\usepackage{longtable}
% Number equations per chapter so references stay local.
\numberwithin{equation}{chapter}
"""

_TIKZ = r"""
% --- Diagrams ---------------------------------------------------------------
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,shapes,shapes.geometric,shadows,calc,fit,backgrounds}

\tikzset{
    every picture/.style={font=\small},
    every node/.style={align=center},
    modernbox/.style={
        rectangle, draw=bfPrimary, fill=bfWash, thick,
        rounded corners=3pt, minimum height=1.0cm, minimum width=2.6cm,
        inner sep=6pt, align=center, text width=3.2cm,
        font=\small\bfseries\color{bfInk},
        drop shadow={opacity=0.12, shadow xshift=1.2pt, shadow yshift=-1.2pt}
    },
    accentbox/.style={
        rectangle, draw=bfAccent, fill=bfWashAccent, thick,
        rounded corners=3pt, minimum height=1.0cm, minimum width=2.6cm,
        inner sep=6pt, align=center, text width=3.2cm,
        font=\small\bfseries\color{bfInk},
        drop shadow={opacity=0.12, shadow xshift=1.2pt, shadow yshift=-1.2pt}
    },
    neutralbox/.style={
        rectangle, draw=bfRule, fill=white, thick,
        rounded corners=3pt, minimum height=1.0cm, minimum width=2.6cm,
        inner sep=6pt, align=center, text width=3.2cm,
        font=\small\color{bfInk},
        drop shadow={opacity=0.08, shadow xshift=1pt, shadow yshift=-1pt}
    },
    flowarrow/.style={
        ->, >={Stealth[length=6pt, width=5pt]}, thick,
        color=bfPrimary, font=\footnotesize\color{bfMuted}
    },
    backarrow/.style={
        ->, >={Stealth[length=6pt, width=5pt]}, thick, dashed,
        color=bfMuted, font=\footnotesize\color{bfMuted}
    }
}
"""


def _float_placement() -> str:
    """Float parameters, and why each one is loosened.

    LaTeX's defaults were tuned for documents with a handful of floats. A
    BookForge chapter carries a dozen figures plus tables, and with the stock
    parameters LaTeX refuses most in-text placements and defers them — which is
    exactly how the old books ended up with figure clumps at chapter ends,
    pages away from the prose discussing them.
    """
    return r"""
% --- Float placement --------------------------------------------------------
\usepackage{float}
\usepackage{placeins}   % \FloatBarrier
\usepackage{afterpage}

% Allow a page to be mostly float (default 0.7) and require less text on a
% page that carries one (default 0.2). Without this a 12-figure chapter defers
% almost everything to the end.
\renewcommand{\topfraction}{0.88}
\renewcommand{\bottomfraction}{0.72}
\renewcommand{\textfraction}{0.12}
% A float-only page must be at least this full, so we never get a page with one
% small figure marooned in white space.
\renewcommand{\floatpagefraction}{0.72}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
% How hard LaTeX tries to place a float where it was written.
\setlength{\intextsep}{8pt plus 2pt minus 2pt}
\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}

% Sections act as float barriers: a figure introduced under one heading must
% not drift under the next, where its caption would contradict its context.
\let\bfOldSection\section
\renewcommand{\section}{\FloatBarrier\bfOldSection}
"""


def _typography() -> str:
    return r"""
% --- Typography -------------------------------------------------------------
% Never break a paragraph across pages leaving one line behind. 10000 is
% LaTeX's "infinitely bad" — widows and orphans read as typesetting errors.
\widowpenalty=10000
\clubpenalty=10000
\displaywidowpenalty=10000
\brokenpenalty=10000
% Discourage hyphenating across a page break, and never hyphenate a heading.
\raggedbottom
\setlength{\parskip}{0pt}
\setlength{\parindent}{1.2em}
\setlength{\emergencystretch}{3em}
\hyphenpenalty=200
\tolerance=1200
"""


def _callouts() -> str:
    """Callout boxes.

    A textbook needs somewhere to put a definition, a caution, or "the one
    thing to remember" that is visibly not body prose. Without these the writer
    can only bold a sentence, which readers skim past.
    """
    return r"""
% --- Callout boxes ----------------------------------------------------------
\usepackage[most]{tcolorbox}
\tcbuselibrary{skins,breakable}
% For the "g" argument type below: an optional *braced* argument. The LaTeX
% kernel's own \NewDocumentEnvironment dropped it; xparse still provides it,
% and tests/test_design.py compiles every argument form so a future removal
% fails the suite rather than the book.
\usepackage{xparse}

% Each box takes its title, and only its title, as an argument. The obvious
% definition makes the optional argument a raw tcolorbox key list, which turns
% the most natural thing a writer can type -- \begin{keyidea}[A custom heading]
% -- into a FATAL error reading "I do not know the key '/tcb/A custom heading'".
% The chapter is discarded and the critic is handed an error naming pgfkeys
% rather than the box. Here every spelling below reaches the same place:
%
%   \begin{keyidea}                 \begin{definitionbox}{Backpressure}
%   \begin{keyidea}[Custom title]   \begin{definitionbox}[Backpressure]
%
% Raw key passthrough is deliberately gone. Nothing needed it, and it cost more
% than it was worth.

\newtcolorbox{bfKeyIdeaBox}[1]{%
  enhanced, breakable, colback=bfWash, colframe=bfPrimary,
  boxrule=0pt, leftrule=3pt, arc=2pt, left=8pt, right=8pt,
  top=6pt, bottom=6pt, fontupper=\small,
  title={\bfseries\sffamily #1}, coltitle=bfPrimary,
  attach boxed title to top left={xshift=10pt, yshift=-8pt},
  boxed title style={colback=white, colframe=white, boxrule=0pt, size=small}}

\newtcolorbox{bfDefinitionBox}[1]{%
  enhanced, breakable, colback=white, colframe=bfAccent,
  boxrule=0.8pt, arc=2pt, left=8pt, right=8pt, top=6pt, bottom=6pt,
  fontupper=\small, title={\bfseries\sffamily #1}, coltitle=white,
  colbacktitle=bfAccent}

\newtcolorbox{bfCautionBox}[1]{%
  enhanced, breakable, colback=bfWashWarn, colframe=bfWarn,
  boxrule=0pt, leftrule=3pt, arc=2pt, left=8pt, right=8pt,
  top=6pt, bottom=6pt, fontupper=\small,
  title={\bfseries\sffamily #1}, coltitle=bfWarn,
  attach boxed title to top left={xshift=10pt, yshift=-8pt},
  boxed title style={colback=bfWashWarn, colframe=bfWashWarn, boxrule=0pt, size=small}}

\newtcolorbox{bfExampleBox}[1]{%
  enhanced, breakable, colback=white, colframe=bfRule,
  boxrule=0.6pt, arc=2pt, left=8pt, right=8pt, top=6pt, bottom=6pt,
  fontupper=\small, title={\bfseries\sffamily #1}, coltitle=bfInk,
  colbacktitle=bfWash}

\newtcolorbox{bfDerivationBox}[1]{%
  enhanced, breakable, colback=bfWash, colframe=bfPrimary,
  boxrule=0.7pt, arc=2pt, left=8pt, right=8pt, top=6pt, bottom=6pt,
  fontupper=\small, title={\bfseries\sffamily #1}, coltitle=white,
  colbacktitle=bfPrimary}

% "o g" = an optional bracketed title and an optional braced one. A braced
% title wins, then a bracketed one, then the default.
\NewDocumentEnvironment{keyidea}{o g}
  {\begin{bfKeyIdeaBox}{\IfValueTF{#2}{#2}{\IfValueTF{#1}{#1}{Key idea}}}}
  {\end{bfKeyIdeaBox}}

\NewDocumentEnvironment{definitionbox}{o g}
  {\begin{bfDefinitionBox}{\IfValueTF{#2}{#2}{\IfValueTF{#1}{#1}{Definition}}}}
  {\end{bfDefinitionBox}}

\NewDocumentEnvironment{cautionbox}{o g}
  {\begin{bfCautionBox}{\IfValueTF{#2}{#2}{\IfValueTF{#1}{#1}{Caution}}}}
  {\end{bfCautionBox}}

\NewDocumentEnvironment{exampleblock}{o g}
  {\begin{bfExampleBox}{\IfValueTF{#2}{#2}{\IfValueTF{#1}{#1}{Example}}}}
  {\end{bfExampleBox}}

\NewDocumentEnvironment{derivationbox}{o g}
  {\begin{bfDerivationBox}{\IfValueTF{#2}{#2}{\IfValueTF{#1}{#1}{Derivation}}}}
  {\end{bfDerivationBox}}
"""


def _margin_notes(theme: BookTheme) -> str:
    if not theme.margin_notes:
        return (
            "\n% Margin notes disabled by theme; keep the macro so chapters"
            "\n% written for either theme still compile.\n"
            "\\newcommand{\\marginkey}[1]{}\n"
        )
    return r"""
% --- Margin notes -----------------------------------------------------------
\usepackage{marginnote}
\renewcommand*{\marginfont}{\footnotesize\sffamily\color{bfMuted}}
% A short pointer in the margin: a term being introduced, or a cross-reference.
% Deliberately narrow in scope — margin notes that carry argument compete with
% the body text instead of supporting it.
\newcommand{\marginkey}[1]{\marginnote{\raggedright #1}}
"""


def _headers_and_footers(theme: BookTheme) -> str:
    """Running heads and footers.

    The convention this follows is the standard scholarly one: verso carries
    the chapter, recto carries the section, so an open spread tells you where
    you are at both levels without turning back.
    """
    if not theme.running_heads:
        return "\n\\pagestyle{plain}\n"

    palette_rule = "0.4pt" if theme.footer_rule else "0pt"
    # The page number sits on the OUTER edge (LE/RO) where a thumbing reader
    # finds it; the book title sits on the INNER edge (RE/LO). Putting both in
    # the same slot silently drops one — fancyhdr keeps only the last
    # assignment, and the page numbers vanish from the whole book.
    footer_title = (
        r"  \fancyfoot[RE,LO]{\footnotesize\sffamily\color{bfMuted}\bfBookTitle}"
        if theme.show_book_title_in_footer
        else ""
    )

    return f"""
% --- Running heads and footers ----------------------------------------------
\\usepackage{{fancyhdr}}
\\usepackage{{emptypage}}   % blank versos really are blank

% Verso: chapter. Recto: section. An open spread then answers "where am I?"
% at both levels without turning back.
\\renewcommand{{\\chaptermark}}[1]{{\\markboth{{\\thechapter.\\ #1}}{{}}}}
\\renewcommand{{\\sectionmark}}[1]{{\\markright{{\\thesection.\\ #1}}}}

\\fancypagestyle{{bookforge}}{{%
  \\fancyhf{{}}
  \\fancyhead[LE]{{\\footnotesize\\sffamily\\color{{bfMuted}}\\nouppercase{{\\leftmark}}}}
  \\fancyhead[RO]{{\\footnotesize\\sffamily\\color{{bfMuted}}\\nouppercase{{\\rightmark}}}}
  \\fancyfoot[LE,RO]{{\\normalsize\\sffamily\\color{{bfPrimary}}\\thepage}}
{footer_title}
  \\renewcommand{{\\headrulewidth}}{{0.4pt}}
  \\renewcommand{{\\footrulewidth}}{{{palette_rule}}}
  \\renewcommand{{\\headrule}}{{\\color{{bfRule}}\\hrule height \\headrulewidth}}
  \\renewcommand{{\\footrule}}{{\\color{{bfRule}}\\hrule height \\footrulewidth}}
}}

% Chapter-opening pages use `plain`; give it the same footer so page numbers
% never change position mid-book.
\\fancypagestyle{{plain}}{{%
  \\fancyhf{{}}
  \\fancyfoot[LE,RO]{{\\normalsize\\sffamily\\color{{bfPrimary}}\\thepage}}
{footer_title}
  \\renewcommand{{\\headrulewidth}}{{0pt}}
  \\renewcommand{{\\footrulewidth}}{{{palette_rule}}}
  \\renewcommand{{\\footrule}}{{\\color{{bfRule}}\\hrule height \\footrulewidth}}
}}

\\pagestyle{{bookforge}}
"""


def _headings(theme: BookTheme) -> str:
    """Chapter openers and section styling."""
    common = r"""
% --- Headings ---------------------------------------------------------------
\usepackage{titlesec}
\usepackage{titletoc}
"""
    if theme.chapter_opener == "plain":
        opener = r"""
\titleformat{\chapter}[display]
  {\normalfont\huge\bfseries\sffamily\color{bfInk}}
  {\normalsize\sffamily\color{bfMuted}\MakeUppercase{\chaptertitlename}\ \thechapter}
  {8pt}{\Huge}
\titlespacing*{\chapter}{0pt}{8pt}{16pt}
"""
    elif theme.chapter_opener == "minimal":
        opener = r"""
\titleformat{\chapter}[hang]
  {\normalfont\Huge\bfseries\sffamily\color{bfInk}}
  {\color{bfPrimary}\thechapter\hspace{14pt}}{0pt}{\Huge}
\titlespacing*{\chapter}{0pt}{4pt}{12pt}
"""
    else:  # banded — a numeral in the colour band, then the title beneath
        opener = r"""
% A large ghosted numeral behind a coloured rule: the reader can find a chapter
% start by flipping, which is the whole job of a chapter opener.
\titleformat{\chapter}[display]
  {\normalfont\sffamily}
  {\raggedleft
     \begin{tikzpicture}
       \node[anchor=east, inner sep=0pt] (num)
         {\fontsize{42}{42}\selectfont\bfseries\color{bfPrimary!22}\thechapter};
     \end{tikzpicture}}
  {-8pt}
  {\raggedright\color{bfInk}\Huge\bfseries}
  [\vspace{4pt}{\color{bfPrimary}\rule{\textwidth}{1.6pt}}]
\titlespacing*{\chapter}{0pt}{0pt}{14pt}
"""

    sections = r"""
\titleformat{\section}[hang]
  {\normalfont\Large\bfseries\sffamily\color{bfInk}}
  {\color{bfPrimary}\thesection}{12pt}{}
\titlespacing*{\section}{0pt}{12pt plus 2pt minus 2pt}{6pt}

\titleformat{\subsection}[hang]
  {\normalfont\large\bfseries\sffamily\color{bfInk}}
  {\color{bfPrimary}\thesubsection}{10pt}{}
\titlespacing*{\subsection}{0pt}{8pt plus 2pt minus 1pt}{4pt}

% Unnumbered sections (Overview, Glossary, ...) still need to reach the TOC and
% the running head, or the reader loses them entirely.
\newcommand{\bfsection}[1]{%
  \section*{#1}\addcontentsline{toc}{section}{#1}\markright{#1}}
"""

    toc = r"""
% --- Table of contents ------------------------------------------------------
\titlecontents{chapter}[0pt]
  {\addvspace{6pt}\sffamily\bfseries\color{bfInk}}
  {\color{bfPrimary}\thecontentslabel\enspace}{}
  {\hfill\color{bfMuted}\contentspage}
\titlecontents{section}[16pt]
  {\sffamily\small\color{bfInk}}
  {\color{bfMuted}\thecontentslabel\enspace}{}
  % The leading \hspace matters: \titlerule* aligns its dots to a grid, so
  % without it an entry ending near a grid point gets its first dot flush
  % against the last letter and "Overview" reads as "Overview."
  {\hspace{0.6em}\titlerule*[0.6pc]{\color{bfRule}.}\color{bfMuted}\contentspage}
\setcounter{tocdepth}{1}
\setcounter{secnumdepth}{2}
"""
    return common + opener + sections + toc


def _captions() -> str:
    return r"""
% --- Captions ---------------------------------------------------------------
\usepackage[
  font={small},
  labelfont={bf,sf,color=bfPrimary},
  textfont={color=bfInk},
  margin=4pt,
  skip=6pt,
  justification=centering
]{caption}
\usepackage{subcaption}
% Figures are numbered per chapter so a reference tells you where to look.
\counterwithin{figure}{chapter}
\counterwithin{table}{chapter}
"""


def _lists_and_links(theme: BookTheme) -> str:
    index = "\\usepackage{makeidx}\n\\makeindex\n" if theme.include_index else ""
    return (
        r"""
% --- Lists and links --------------------------------------------------------
\usepackage{enumitem}
\setlist{leftmargin=*, itemsep=2pt, parsep=0pt, topsep=4pt}
\setlist[itemize]{label=\textcolor{bfPrimary}{\textbullet}}
\setlist[enumerate]{label=\textcolor{bfPrimary}{\arabic*.}}
\setlist[description]{font=\sffamily\bfseries\color{bfInk}}

\usepackage[hidelinks]{hyperref}
\hypersetup{
  colorlinks=true,
  linkcolor=bfPrimary,
  citecolor=bfAccent,
  urlcolor=bfAccent,
  linktoc=all,
  bookmarksnumbered=true,
  pdfborder={0 0 0}
}
\usepackage{url}
\urlstyle{sf}
"""
        + index
    )


_HELPER_MACROS = r"""
% --- Chapter-facing macros --------------------------------------------------
% These exist so a chapter body never needs a \usepackage or a raw dimension.
\newcommand{\bfBookTitle}{}
\newcommand{\setbookforgetitle}[1]{\renewcommand{\bfBookTitle}{#1}}

% A term being defined: bold, coloured, and indexed in one call.
\newcommand{\term}[1]{\textbf{\color{bfPrimary}#1}\index{#1}}

% Fixed-width text that may break across lines — long identifiers and paths
% are the single commonest cause of overfull boxes in generated prose.
\newcommand{\code}[1]{\texttt{\small\seqsplit{#1}}}

% Standard figure width, so figures share one measure down the book.
\newcommand{\figwidth}{0.96\textwidth}
\newcommand{\figwidthwide}{\textwidth}
\newcommand{\figwidthnarrow}{0.72\textwidth}
"""

_SEQSPLIT = r"""
\usepackage{seqsplit}
"""


def build_preamble(theme: BookTheme | None = None) -> str:
    """Assemble the complete document preamble for a theme."""
    theme = theme or BookTheme()
    parts = [
        f"% BookForge preamble — theme '{theme.name}'. Generated; do not hand-edit.",
        f"\\documentclass[{theme.class_options()}]{{book}}",
        _FONTS,
        theme.geometry(),
        "\n% --- Palette ---------------------------------------------------------------",
        "\\usepackage{xcolor}",
        theme.palette.definitions(),
        "\\color{bfInk}",
        _MATH_AND_GRAPHICS,
        _SEQSPLIT,
        _TIKZ,
        _typography(),
        _float_placement(),
        _captions(),
        _callouts(),
        _margin_notes(theme),
        _headings(theme),
        _lists_and_links(theme),
        _headers_and_footers(theme),
        _HELPER_MACROS,
        _float_macros(),
    ]
    return "\n".join(parts) + "\n"


def _float_macros() -> str:
    # Imported lazily: layout imports tex_escape from tools.latex, which imports
    # workspace — keeping it out of module scope avoids an import cycle.
    from bookforge.design.layout import FLOAT_MACROS

    return FLOAT_MACROS


# ---------------------------------------------------------------------------
# Front and back matter
# ---------------------------------------------------------------------------


def short_source(source_line: str) -> str:
    """The title-page form of a provenance line.

    A local source line carries the absolute path it was built from, which is
    what the colophon should record and the last thing a title page should
    show -- a reader does not need a build machine's directory layout, and a
    long one wraps across three lines under the author. URLs are left alone:
    they identify the source and are the natural thing to print.
    """
    label, _, rest = source_line.partition(":")
    rest = rest.strip()
    if not rest or rest.startswith("//") or "://" in source_line:
        return source_line
    names = [Path(part.strip()).name for part in rest.split(",") if part.strip()]
    names = [name for name in names if name]
    if not names:
        return label.strip() or source_line
    shown = ", ".join(names[:3]) + (" and others" if len(names) > 3 else "")
    return f"{label.strip()}: {shown}" if label.strip() else shown


def title_page(
    title: str,
    subtitle: str,
    author: str,
    source_line: str,
    theme: BookTheme | None = None,
) -> str:
    """A designed half-title page rather than \\maketitle's centred stack."""
    theme = theme or BookTheme()
    from bookforge.tools.latex import tex_escape

    source_line = short_source(source_line)

    return rf"""
\begin{{titlepage}}
\thispagestyle{{empty}}
\begin{{center}}
\vspace*{{1.6cm}}
{{\color{{bfPrimary}}\rule{{0.34\textwidth}}{{2.4pt}}}}\\[1.0cm]
{{\sffamily\bfseries\fontsize{{30}}{{34}}\selectfont\color{{bfInk}} {tex_escape(title)}}}\\[0.55cm]
{{\sffamily\large\color{{bfMuted}} {tex_escape(subtitle)}}}\\[1.4cm]
{{\color{{bfRule}}\rule{{0.6\textwidth}}{{0.6pt}}}}\\[0.55cm]
{{\sffamily\normalsize\color{{bfInk}} {tex_escape(author)}}}\\[0.35cm]
{{\sffamily\small\color{{bfMuted}} {tex_escape(source_line)}}}
\vfill
{{\sffamily\small\color{{bfMuted}} \today}}
\end{{center}}
\end{{titlepage}}
"""


def colophon(generator_line: str, source_line: str, stats_line: str) -> str:
    """Provenance page.

    A generated book should say so, and say what it was generated from. Anyone
    holding this later needs to know the chapters came from a pipeline and
    which recording each one covers.
    """
    from bookforge.tools.latex import tex_escape

    return rf"""
\thispagestyle{{empty}}
\vspace*{{\fill}}
{{\footnotesize\sffamily\color{{bfMuted}}
\noindent {tex_escape(generator_line)}\\[4pt]
\noindent {tex_escape(source_line)}\\[4pt]
\noindent {tex_escape(stats_line)}\\[10pt]
\noindent Chapters were drafted from recorded talks, verified by compilation
and checked against their source transcripts before inclusion. Figures are
frames captured from the recordings; charts are rendered from figures stated
by the speaker.
}}
\clearpage
"""
