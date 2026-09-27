"""Float layout: the chapter declares *where*, BookForge decides *how*.

The placement problem
---------------------
In the ADK design the writer LM emitted complete ``figure`` environments —
path, width, caption, label, placement specifier, all of it. That put four
decisions in the hands of the component least able to make them:

* **the path**, which it could invent (a critical defect the verifier had to
  catch after the fact);
* **the caption**, which it could paraphrase away from the grounded caption
  the captioner produced;
* **the width**, which it varied per figure, so no two figures in a chapter
  shared a measure;
* **the placement specifier**, which it usually wrote as ``[htbp]`` — and with
  LaTeX's stock float parameters that resolves to "defer", which is how a
  twelve-figure chapter ends with a clump of figures pages away from the prose
  that discusses them.

The fix
-------
Invert it. Every asset is *declared* once, deterministically, at the top of the
chapter by BookForge — path, caption and label straight from the manifest. The
writer then places it with a single marker in the prose:

    Figure~\\ref{fig:ch01-frame3} shows the scheduler queue. \\bfFigure{fig:ch01-frame3}

``\\bfFigure`` expands to a properly-typeset float using the declared values. The
writer chooses *where*, which is the judgement it is good at; BookForge chooses
*how*, which is the mechanical part it was getting wrong. An undeclared label
fails loudly at compile time instead of silently rendering a broken reference.
"""

from __future__ import annotations

from dataclasses import dataclass

from bookforge.schemas import AssetsManifest, FigureAsset, TableAsset

#: LaTeX side of the mechanism. Lives in the preamble; chapters only call it.
FLOAT_MACROS = r"""
% --- Declared floats --------------------------------------------------------
% BookForge declares each asset once with \bfDeclareFigure / \bfDeclareTable,
% then the chapter places it with \bfFigure{<label>} wherever it belongs in the
% argument. Placement, sizing and captioning stay consistent across the book,
% and a reference to an undeclared asset is a compile error rather than a
% silently broken float.

\makeatletter
\newcommand{\bfDeclareFigure}[4]{%
  \expandafter\gdef\csname bffig@#1@path\endcsname{#2}%
  \expandafter\gdef\csname bffig@#1@caption\endcsname{#3}%
  \expandafter\gdef\csname bffig@#1@width\endcsname{#4}%
}
\newcommand{\bfDeclareTable}[3]{%
  \expandafter\gdef\csname bftab@#1@path\endcsname{#2}%
  \expandafter\gdef\csname bftab@#1@caption\endcsname{#3}%
}

\newcommand{\bf@figureerror}[1]{%
  \PackageError{bookforge}{Figure '#1' was never declared}%
    {The chapter referenced a figure that is not in the assets manifest.}%
}

% \bfFigure[placement]{label}
\newcommand{\bfFigure}[2][htbp]{%
  \@ifundefined{bffig@#2@path}{\bf@figureerror{#2}}{%
    \begin{figure}[#1]
      \centering
      \includegraphics[width=\csname bffig@#2@width\endcsname,
                       height=0.48\textheight, keepaspectratio]%
                      {\csname bffig@#2@path\endcsname}
      \caption{\csname bffig@#2@caption\endcsname}
      \label{#2}
    \end{figure}%
  }%
}

% Compact video-time provenance under a figure caption (lecture-notes style).
% Captions already embed ``[video MM:SS]`` when footnotes are enabled; this
% macro is available for chapter prose that wants an explicit source line.
\newcommand{\srcnote}[1]{%
  \par\nobreak\smallskip
  {\footnotesize\sffamily\color{bfMuted}Source: #1}\par
}

% A figure that must appear exactly here (H), for step-by-step sequences where
% drifting one page would break the argument.
\newcommand{\bfFigureHere}[1]{\bfFigure[H]{#1}}

% Two figures side by side: for before/after and comparison pairs, where
% separating them onto different pages destroys the comparison.
\newcommand{\bfFigurePair}[3][htbp]{%
  \begin{figure}[#1]
    \centering
    \begin{subfigure}[t]{0.47\textwidth}
      \centering
      \includegraphics[width=\linewidth, height=0.30\textheight, keepaspectratio]%
                      {\csname bffig@#2@path\endcsname}
      \caption{\csname bffig@#2@caption\endcsname}
      \label{#2}
    \end{subfigure}\hfill
    \begin{subfigure}[t]{0.47\textwidth}
      \centering
      \includegraphics[width=\linewidth, height=0.30\textheight, keepaspectratio]%
                      {\csname bffig@#3@path\endcsname}
      \caption{\csname bffig@#3@caption\endcsname}
      \label{#3}
    \end{subfigure}
  \end{figure}%
}

% \bfTable{label} — the fragment carries its own tabular; we supply the float.
\newcommand{\bfTable}[2][htbp]{%
  \@ifundefined{bftab@#2@path}{%
    \PackageError{bookforge}{Table '#2' was never declared}{}%
  }{%
    \begin{table}[#1]
      \centering
      \caption{\csname bftab@#2@caption\endcsname}
      \label{#2}
      \input{\csname bftab@#2@path\endcsname}
    \end{table}%
  }%
}

% A diagram authored inline as TikZ still gets the house float treatment.
% \bfDiagram{caption}{label}{tikz body}
\newcommand{\bfDiagram}[3]{%
  \begin{figure}[htbp]
    \centering
    \begin{adjustbox}{max width=\textwidth, max totalheight=0.50\textheight}
      #3
    \end{adjustbox}
    \caption{#1}
    \label{#2}
  \end{figure}%
}
\makeatother
"""


@dataclass(frozen=True)
class ChapterLayout:
    """Widths chosen per asset kind, so a chapter reads as one system."""

    #: Screenshots stay just inside the measure so a caption still has air,
    #: but they use the page rather than floating in a wide empty frame.
    frame_width: str = "0.94\\textwidth"
    #: Charts are vector and information-dense — give them the full measure.
    chart_width: str = "\\textwidth"
    #: Anything unclassified.
    default_width: str = "0.96\\textwidth"

    def width_for(self, asset: FigureAsset) -> str:
        if asset.kind == "frame":
            return self.frame_width
        if asset.kind == "chart":
            return self.chart_width
        return self.default_width


def _escape_caption(caption: str) -> str:
    """Escape a caption for use inside a macro definition.

    Captions come from the captioner (an LM) and from chart specs, so they can
    contain any character. They are stored in a macro body, which makes an
    unescaped ``%`` or ``#`` a syntax error at declaration time rather than a
    visible defect later.
    """
    from bookforge.tools.latex import tex_escape

    return tex_escape(caption)


def declaration_block(
    manifest: AssetsManifest, layout: ChapterLayout | None = None
) -> str:
    """Every asset in the chapter, declared once, from the manifest.

    Emitted by BookForge and prepended to the chapter body, so the writer never
    types a path and cannot invent one.
    """
    layout = layout or ChapterLayout()
    lines = [
        "% Asset declarations for this chapter — generated from assets.json.",
        "% The chapter places these with \\bfFigure{<label>} / \\bfTable{<label>}.",
    ]
    for figure in manifest.figures:
        lines.append(
            f"\\bfDeclareFigure{{{figure.label}}}{{{figure.filename}}}"
            f"{{{_escape_caption(figure.caption)}}}{{{layout.width_for(figure)}}}"
        )
    for table in manifest.tables:
        lines.append(
            f"\\bfDeclareTable{{{table.label}}}{{{table.tex_file}}}"
            f"{{{_escape_caption(table.caption)}}}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Fallback emitters
# ---------------------------------------------------------------------------


def figure_block(asset: FigureAsset, layout: ChapterLayout | None = None) -> str:
    """A complete float for one figure.

    Used when placing assets the writer omitted — see ``ensure_all_assets_placed``.
    """
    layout = layout or ChapterLayout()
    return (
        "\\begin{figure}[htbp]\n"
        "  \\centering\n"
        f"  \\includegraphics[width={layout.width_for(asset)}, "
        "height=0.48\\textheight, keepaspectratio]"
        f"{{{asset.filename}}}\n"
        f"  \\caption{{{_escape_caption(asset.caption)}}}\n"
        f"  \\label{{{asset.label}}}\n"
        "\\end{figure}\n"
    )


def table_block(asset: TableAsset) -> str:
    return (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        f"  \\caption{{{_escape_caption(asset.caption)}}}\n"
        f"  \\label{{{asset.label}}}\n"
        f"  \\input{{{asset.tex_file}}}\n"
        "\\end{table}\n"
    )


def side_by_side_block(
    left: FigureAsset, right: FigureAsset
) -> str:
    """Two figures on one row, for comparisons that lose meaning when split."""
    return (
        "\\begin{figure}[htbp]\n  \\centering\n"
        "  \\begin{subfigure}[t]{0.47\\textwidth}\n    \\centering\n"
        f"    \\includegraphics[width=\\linewidth, height=0.30\\textheight, keepaspectratio]{{{left.filename}}}\n"
        f"    \\caption{{{_escape_caption(left.caption)}}}\n    \\label{{{left.label}}}\n"
        "  \\end{subfigure}\\hfill\n"
        "  \\begin{subfigure}[t]{0.47\\textwidth}\n    \\centering\n"
        f"    \\includegraphics[width=\\linewidth, height=0.30\\textheight, keepaspectratio]{{{right.filename}}}\n"
        f"    \\caption{{{_escape_caption(right.caption)}}}\n    \\label{{{right.label}}}\n"
        "  \\end{subfigure}\n"
        "\\end{figure}\n"
    )
