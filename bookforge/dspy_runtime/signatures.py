"""DSPy signatures — the typed I/O contracts for every LLM stage.

This file replaces the hand-written prompt strings of the ADK implementation.
The difference is not cosmetic:

* **Instructions are data, not code.** Each signature's docstring is its initial
  instruction. GEPA/MIPROv2 rewrite these at optimization time and the improved
  text is persisted as an artifact, so prompt quality becomes something we
  measure and improve rather than something we argue about.
* **Outputs are typed.** ``ChapterAnalysis`` is a Pydantic model, so DSPy's
  adapter handles schema-conformant decoding and retries. The old pipeline
  needed a bespoke ``extract_clean_json`` that scanned every pair of braces in
  the response (quadratic, and still guessy); that whole class of failure is
  handled by the adapter now.
* **No brace escaping hazard.** ADK injected session state into instruction
  strings with ``{placeholder}`` formatting, which is why the original prompts
  could not contain a single literal LaTeX brace and had to *describe* markup in
  English. DSPy passes inputs as arguments, so instructions here can show real
  LaTeX.
"""

from __future__ import annotations

import dspy
from pydantic import BaseModel, Field, model_validator

from bookforge.schemas import (
    ChapterAnalysis,
    ChartSpec,
    ConceptNote,
    DiagramSpec,
    GlossaryItem,
    MathDerivation,
    TableSpec,
    trim_lists,
)

# ---------------------------------------------------------------------------
# Intermediate types
# ---------------------------------------------------------------------------


class ChunkAnalysis(BaseModel):
    """What one transcript window yields. Merged into a ChapterAnalysis later."""

    concepts: list[ConceptNote] = Field(default_factory=list, max_length=6)
    tables: list[TableSpec] = Field(default_factory=list, max_length=3)
    charts: list[ChartSpec] = Field(default_factory=list, max_length=3)
    diagrams: list[DiagramSpec] = Field(default_factory=list, max_length=2)
    derivations: list[MathDerivation] = Field(default_factory=list, max_length=4)
    glossary: list[GlossaryItem] = Field(default_factory=list, max_length=12)
    #: A one-line summary of what this window covered, used by the merge step.
    segment_summary: str = ""

    @model_validator(mode="before")
    @classmethod
    def _trim_surplus(cls, data: object) -> object:
        return trim_lists(
            data,
            {"concepts": 6, "tables": 3, "charts": 3, "diagrams": 2,
             "derivations": 4, "glossary": 12},
            "ChunkAnalysis",
        )


class FrameCaption(BaseModel):
    """A caption for one curated video frame, tied to what was being said."""

    frame_file: str
    caption: str = Field(description="One or two sentences, textbook voice")
    #: Which concept heading this frame illustrates, or "" if none.
    concept_heading: str = ""
    #: False when the frame shows nothing worth printing (title card, talking
    #: head, transition). Such frames are dropped rather than padded with a
    #: meaningless "Video frame at 03:15" caption.
    is_useful: bool = True


class FrameCaptionSet(BaseModel):
    captions: list[FrameCaption] = Field(default_factory=list)


class SemanticDefect(BaseModel):
    """A defect only a reader can find — deterministic checks catch the rest."""

    severity: str = Field(description="critical, major, or minor")
    message: str
    fix: str = Field(description="The exact change the refiner must make")


class SemanticCritique(BaseModel):
    defects: list[SemanticDefect] = Field(default_factory=list, max_length=12)


# ---------------------------------------------------------------------------
# Stage 1 — transcript analysis (map)
# ---------------------------------------------------------------------------


class AnalyzeTranscriptChunk(dspy.Signature):
    r"""Extract textbook-grade structure from one window of a lecture transcript.

    You are reading a contiguous window of a longer talk, so cover THIS window
    only; another pass handles the rest and a merge step reconciles them.

    Fidelity still rules for facts and numbers: every claim must be traceable
    to words actually spoken. But a textbook chapter that never visualizes a
    process or defines its terms is unfinished — extract structure the reader
    needs, not only what was written on a slide.

    - concepts: the ideas genuinely developed here. Each explanation is 3-8
      sentences of real technical substance, in the speaker's own logic, not a
      restatement of the heading. Populate evidence_timestamps with the
      [HH:MM:SS] markers the material came from.
    - tables: emit when the speaker compares options, attributes, phases,
      pros/cons, or roles — even if they never say "table". Qualitative
      comparison rows are fine (no invented numbers). Every row must have
      exactly as many cells as headers. Only return [] when there is truly
      nothing comparable in this window.
    - charts: only for numbers the speaker actually states. categories and
      values must be the same length and pair up positionally. Never invent,
      round, or interpolate a figure. No spoken numbers means charts: [].
    - diagrams: if the speaker describes a pipeline, architecture, agent
      handoff, lifecycle, hierarchy, or ordered steps (A then B then C), emit
      a flowchart/sequence/hierarchy DiagramSpec. Empty diagrams is only
      correct when the window has no process to draw. Every relationship must
      reference nodes that appear in elements.
    - glossary: every technical term a newcomer would need (APIs, product
      names, jargon the speaker uses without defining). Prefer 4–12 terms
      when the window is technical; [] only for purely narrative asides.
    - derivations: when the window involves formulae, scales, bit-widths,
      losses, probabilities, complexity, or any quantitative relationship,
      emit MathDerivation entries. Each needs:
        * motivation (why derive this),
        * ≥3 steps that a student can follow without guessing,
        * formula_latex for the final result using proper LaTeX
          (\frac, \max, 2^{b}, subscripts),
        * assumptions when relevant (symmetric range, zero-point, etc.).
      Expand compressed speech ("scale is max over 2^b-1") into complete
      algebra. Prefer separate derivations for distinct formulae (e.g. scale
      AND zero-point AND dequantization), not one vague mega-formula.
      [] only when the window has no mathematical content at all.
    """

    video_title: str = dspy.InputField(desc="Title of the source video")
    chunk_index: int = dspy.InputField(desc="1-based index of this window")
    chunk_total: int = dspy.InputField(desc="Total number of windows")
    transcript_chunk: str = dspy.InputField(
        desc="Timestamped transcript window; [HH:MM:SS] markers begin each line"
    )
    analysis: ChunkAnalysis = dspy.OutputField(
        desc="Structured content extracted from this window only"
    )


class MergeChunkAnalyses(dspy.Signature):
    """Reconcile per-window analyses into one coherent chapter analysis.

    The windows were read independently, so they overlap and repeat. Your job is
    editorial for facts and numbers: you may merge, deduplicate, reorder and
    retitle, but you must not invent a numeric value, table cell, or concept
    absent from the inputs. Pedagogical structure is different — you SHOULD
    author glossary entries and review exercises from the merged concepts.

    - chapter_title: precise and academic. Not the video's clickbait title.
    - summary: one dense paragraph an expert could skim.
    - learning_objectives: 2-8, each stated as a capability ("The reader can
      distinguish ...", "The reader can configure ...").
    - concepts: merge duplicates that describe the same idea, keeping the
      fullest explanation and the union of their evidence_timestamps. Order them
      the way the talk developed them, not by importance.
    - tables / charts / diagrams: carry through the strongest, dropping exact
      duplicates. Preserve every numeric value verbatim. If chunks left
      diagrams empty but concepts describe a process/architecture, synthesize
      ONE grounded flowchart from those concepts (nodes and edges only from
      named ideas in the inputs — no new components). Prefer at least one
      table or diagram when the talk compares options or walks a pipeline.
    - derivations: union of chunk derivations, dropping duplicates of the same
      final formula. If the topic is mathematical (quantization, ML loss,
      algorithms, scaling laws, …) and chunks left this empty, reconstruct
      1–4 standard derivations that the concepts require — steps must be
      complete, and every numeric constant must already appear in the inputs.
      Never invent empirical measurements.
    - glossary: union, deduplicated by term. Target 6–15 entries covering
      every non-obvious term in the concepts; write crisp one-sentence
      definitions from the material (not dictionary fluff).
    - exercises: REQUIRED — 4 to 8 review questions with full answers,
      ordered easy → hard. Each must be answerable from the concepts alone.
      Include at least one calculation/derivation question when derivations
      exist. Empty exercises is never acceptable for a production chapter.
    """

    video_title: str = dspy.InputField()
    chunk_analyses: list[ChunkAnalysis] = dspy.InputField(
        desc="Per-window analyses in chronological order"
    )
    analysis: ChapterAnalysis = dspy.OutputField(
        desc="One merged, deduplicated, chronologically coherent analysis"
    )


class SingleShotAnalysis(dspy.Signature):
    """Extract a complete textbook chapter analysis from a full transcript.

    Used when the transcript fits in one window, so no merge step is needed.

    Facts and numbers must be traceable to the talk. Production chapters also
    need pedagogy and visuals when the topic supports them:

    - concepts: 3+ ideas with 3–8 sentence explanations and evidence_timestamps.
    - tables: qualitative or numeric comparisons the speaker makes (options,
      phases, roles). [] only when nothing is comparable.
    - charts: only with spoken numbers; equal-length categories/values; never
      invent figures.
    - diagrams: emit a flowchart/sequence when the talk describes a process,
      architecture, or agent/system flow.
    - derivations: REQUIRED whenever the talk uses formulae or quantitative
      relationships. Each MathDerivation needs motivation, ≥2 steps of full
      reasoning, and formula_latex for the result. Expand "scale is max over
      range" style speech into student-ready algebra. [] only if there is
      genuinely no maths.
    - glossary: 6–15 terms a newcomer needs, defined from the talk.
    - exercises: 4–8 questions with answers, easy → hard, grounded in concepts.
      Prefer at least one worked numerical/derivation question when maths
      appears. Never leave exercises empty.

    Charts must carry equal-length categories and values that pair up
    positionally, and tables must have exactly as many cells per row as headers.
    """

    video_title: str = dspy.InputField()
    transcript: str = dspy.InputField(desc="Full timestamped transcript")
    analysis: ChapterAnalysis = dspy.OutputField()


# ---------------------------------------------------------------------------
# Stage 2 — frame captioning
# ---------------------------------------------------------------------------


class CaptionFrames(dspy.Signature):
    """Write grounded captions for screenshots taken from the lecture.

    Each frame is identified by its filename and the timestamp it was captured
    at. You are given the transcript lines spoken around each of those moments,
    and optionally a visual briefing from a video-understanding model. Caption
    the frame by what was on screen and what the speaker was explaining — a
    caption that could apply to any frame in any video is a failure.

    Set is_useful to false for frames that carry no instructional content: title
    cards, a presenter's face with no visible material, transitions, blank or
    duplicated slides. Those frames are dropped from the chapter rather than
    printed under filler text, so being decisive here directly improves the
    book.

    Where a frame illustrates one of the listed concepts, set concept_heading to
    that concept's exact heading so the figure can be placed beside its
    discussion. Leave it empty when no concept fits. Do not invent timestamps;
    provenance is added separately.
    """

    video_title: str = dspy.InputField()
    concept_headings: list[str] = dspy.InputField(
        desc="Headings of the concepts in this chapter"
    )
    frames_with_context: str = dspy.InputField(
        desc=(
            "Optional visual briefing, then one block per frame: "
            "'FRAME <filename> @ <HH:MM:SS>' followed by the transcript lines "
            "spoken around that timestamp"
        )
    )
    captions: FrameCaptionSet = dspy.OutputField(
        desc="Exactly one entry per frame given, in the same order"
    )


# ---------------------------------------------------------------------------
# Stage 3 — chapter composition
# ---------------------------------------------------------------------------

_LATEX_CONTRACT = r"""
Structure, in this exact order:

  \chapter{<analysis chapter_title>}\label{ch:<NN>-<slug>}
  \bfsection{Overview}               -- the summary paragraph
  \bfsection{Learning Objectives}    -- itemize of the objectives
  \section{<concept heading>}        -- one numbered section per concept
  \section{Key Data and Comparisons} -- tables/charts when present; else a short
                                       comparison prose block (never a blank section)
  \section{Worked Derivations}       -- full maths for every analysis.derivations
                                       entry (omit the section only if none)
  \section{Conceptual Models}        -- one \bfDiagram per diagram spec; if none,
                                       one process diagram distilled from the concepts
  \bfsection{Key Takeaways}          -- itemize across all concepts
  \bfsection{Glossary}               -- description list with REAL entries
  \bfsection{Exercises}              -- enumerate of questions
  \bfsection{Solutions}              -- enumerate of answers, same order

Use \bfsection for the unnumbered sections, never \section*. It adds them to
the table of contents and the running head; \section* silently drops both.

PEDAGOGY — empty shells are a defect, not a placeholder.

Never write comments like "% No glossary terms were provided" or prose that
says exercises were missing. Production notes always fill these:

  \bfsection{Glossary}
  \begin{description}
    \item[Term] Definition grounded in this chapter's concepts.
    ...
  \end{description}

  \bfsection{Exercises}
  \begin{enumerate}
    \item Question ...
  \end{enumerate}

  \bfsection{Solutions}
  \begin{enumerate}
    \item Answer ...
  \end{enumerate}

Rules:
- If the analysis supplies glossary/exercises, use them (you may lightly edit
  for clarity). Target at least 6 glossary terms and 4 exercises.
- If the analysis lists are thin or empty, synthesize glossary terms from
  \term{} usages and concept headings, and write 4–6 review questions whose
  answers are fully supported by the chapter concepts. Do not invent numeric
  facts that are not in the analysis.
- Solutions must answer the Exercises in the same order, with substance (not
  "see above").
- Prefer depth over page-wasting whitespace: expand concept sections with
  keyidea / definitionbox / exampleblock / derivationbox rather than leaving
  large empty regions. Aim for roughly 1200–2500 words.

MATHEMATICS — when the analysis has derivations, or the topic is quantitative.

Do not hand-wave ("the scale is just max over 2^b-1"). Derive it.

For each MathDerivation in the analysis, write a full worked block:

  \begin{derivationbox}{Symmetric quantization scale}
  Quantization maps a real value $x$ into an integer grid of bit-width $b$.
  Assume a symmetric range $[-x_{\max}, x_{\max}]$.
  \begin{align}
    \Delta &= \frac{2 x_{\max}}{2^{b}-1}, \\
    q &= \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{x}{\Delta}\right),
         -2^{b-1},\, 2^{b-1}-1\right), \\
    \hat{x} &= q \cdot \Delta.
  \end{align}
  Each line follows from the previous: $\Delta$ is the step size that tiles
  the range into $2^{b}-1$ intervals; rounding projects $x$ onto that grid;
  dequantization recovers an approximation $\hat{x}$.
  \end{derivationbox}

Rules for maths:
- Use amsmath environments: prefer align for multi-step derivations (never a
  single unexplained equation unless the result is definitional and trivial).
- For EVERY MathDerivation, structure the derivationbox as:
    1. Motivation + assumptions (from the analysis).
    2. One short prose sentence per analysis step.
    3. An align block with ≥2 lines that mirrors those steps, ending at
       formula_latex (or an equivalent form).
    4. A closing sentence interpreting what the result means in the talk.
- Inline maths with $...$; never Unicode operators (× ÷ ≤) — use \times,
  \div, \le, \approx, \sum, \frac{}{}, \mathrm{}, \operatorname{}, \max, \min.
- Write \max, \min, \mathrm{clip}, \mathrm{round} — not bare max_{real}.
- Explain symbols the first time they appear.
- If analysis.derivations is non-empty, every entry MUST appear. If it is empty
  but a concept is clearly mathematical (loss, scale factor, complexity,
  probability, bit-width), still add one grounded derivationbox under Worked
  Derivations using only quantities named in the analysis — do not invent
  empirical constants.
- Worked numerical examples belong in exampleblock when the speaker gave
  numbers; keep them consistent with the derivation.
- Include at least one calculation exercise in Exercises when derivations
  exist, with the algebra shown in Solutions.

FIGURES AND TABLES -- read this carefully, it is the part most often done wrong.

Every asset has already been declared for you at the top of the chapter. You
never write a file path, a width, or a caption. You place an asset with one
marker, on its own line, immediately after the paragraph that discusses it:

  \bfFigure{fig:ch01-frame3}
  \bfTable{tab:ch01-1}

Refer to it in the prose by label, in the sentence before the marker:

  The scheduler drains the queue in three passes (Figure~\ref{fig:ch01-frame3}).
  \bfFigure{fig:ch01-frame3}

Rules:
- Use ONLY labels listed in the assets manifest. A label that was not declared
  is a compile error, not a warning.
- Place every declared asset exactly once. An asset you never place is a figure
  that was rendered and then wasted.
- Put the marker next to the discussion, not in a block at the end of the
  section. Placement is what decides where the figure lands on the page.
- Never write \begin{figure}, \includegraphics, \input, \caption or \label for
  a declared asset. The marker supplies all of them.
- \bfFigurePair{labelA}{labelB} sets two figures side by side, for before/after
  or comparison pairs that lose their meaning when split across pages.
- \bfFigureHere{label} forces a figure to appear exactly where written. Use it
  only for step-by-step sequences where drifting a page breaks the argument.
- Screenshots (frame assets) illustrate UI/context; charts and tables carry
  comparisons. Do not dump every frame into Conceptual Models — that section
  is for TikZ process diagrams.

DIAGRAMS

Render each diagram spec with \bfDiagram{caption}{label}{tikz body}. If the
analysis has diagram specs, every one MUST appear. If it has none but concepts
describe a pipeline, architecture, or agent flow, still draw ONE grounded
flowchart whose nodes are concept names from the analysis:

  \bfDiagram{How a request reaches a pod}{fig:ch01-flow}{%
    \begin{tikzpicture}[node distance=1.2cm]
      \node[modernbox] (ingress) {Ingress};
      \node[neutralbox, below=1.2cm of ingress] (svc) {Service};
      \node[accentbox, below=1.2cm of svc] (pod) {Pod};
      \draw[flowarrow] (ingress) -- node[right, font=\footnotesize]{by hostname} (svc);
      \draw[flowarrow] (svc) -- (pod);
    \end{tikzpicture}}

Use only the preamble styles: modernbox (steps/components), accentbox
(targets/outputs), neutralbox (data/secondary), flowarrow (transitions),
backarrow (feedback edges). Lay out strictly top-to-bottom (below=1.2cm of X)
or left-to-right (right=1.8cm of X) -- never diagonal chains. Node text is
4-7 words; the styles already set text width and centring. \bfDiagram scales
the picture to fit, so you never set scale= yourself.

EMPHASIS AND STRUCTURE -- use these; a wall of undifferentiated prose is the
most common failure of generated chapters.

  \term{backpressure}          first use of a term: bolds, colours and indexes it
  \marginkey{Short gloss.}    four words or fewer; the margin column is only 16mm
  \code{some_long_identifier}  inline code that may break across lines

  \begin{keyidea}
  The single sentence a reader must retain from this section.
  \end{keyidea}

  \begin{definitionbox}{Backpressure}
  A precise definition of a term the speaker introduced.
  \end{definitionbox}

  \begin{cautionbox}
  A pitfall the speaker warned about.
  \end{cautionbox}

  \begin{exampleblock}{Worked example}
  A concrete case the speaker walked through.
  \end{exampleblock}

Every callout takes an optional title, in braces or brackets, and every one has
a sensible default: \begin{keyidea}, \begin{keyidea}{Custom title} and
\begin{keyidea}[Custom title] all work. Nothing else goes in that argument.

Aim for at least one keyidea per chapter and a definitionbox for each glossary
term the speaker actually defined. Use derivationbox for every mathematical
result. Do not manufacture callouts the material does not support -- a
cautionbox with no warning in it is noise.

PREAMBLE AND ESCAPING

The book owns one shared preamble. Never emit \documentclass, \usepackage,
\usetikzlibrary or \begin{document}.

In prose escape & % # _ $ and use \textbackslash{} for a literal backslash.
Wrap long identifiers, paths and URLs in \code{} so they break instead of
overflowing the margin.

LENGTH

Roughly 1200-2500 words of prose. Depth comes from the analysis, never padding.
Fill the page with substance (diagrams, tables, worked callouts), not blank
vertical gaps.

Output only the LaTeX body. No markdown fences, no commentary before or after.
"""


class WriteChapter(dspy.Signature):
    __doc__ = (
        "Compose one complete LaTeX chapter body from a validated analysis and "
        "the assets that exist on disk.\n\n"
        "You are writing production textbook notes, not a transcript summary. "
        "Explain in your own prose, use every declared asset, render process "
        "diagrams, derive mathematical results with full align/equation detail "
        "when the topic is quantitative, and always fill Glossary, Exercises, "
        "and Solutions with real content. Never leave those sections as empty "
        "comments. Never invent numeric facts absent from the analysis.\n"
        + _LATEX_CONTRACT
    )

    video_title: str = dspy.InputField()
    chapter_number: int = dspy.InputField()
    chapter_label_slug: str = dspy.InputField(desc="Slug for the \\label{ch:...}")
    analysis: ChapterAnalysis = dspy.InputField(desc="The only permitted source")
    assets_manifest: str = dspy.InputField(
        desc="Every figure/table path, label and caption available"
    )
    chapter_tex: str = dspy.OutputField(desc="LaTeX chapter body only")


class RefineChapter(dspy.Signature):
    __doc__ = (
        "Repair a LaTeX chapter so every listed defect is fixed.\n\n"
        "This is surgery, not a rewrite. Preserve everything the critique did "
        "not flag — prose the critic accepted is prose that already works, and "
        "rewriting it risks new defects while losing verified content.\n\n"
        "Work in this order: LaTeX compile errors first (they block everything), "
        "then missing or invented asset references, then missing sections, then "
        "unsupported claims, then wording. Every fix must be traceable to a "
        "numbered defect.\n" + _LATEX_CONTRACT
    )

    video_title: str = dspy.InputField()
    chapter_number: int = dspy.InputField()
    chapter_label_slug: str = dspy.InputField()
    analysis: ChapterAnalysis = dspy.InputField()
    assets_manifest: str = dspy.InputField()
    current_chapter_tex: str = dspy.InputField(desc="The chapter as it stands")
    critique: str = dspy.InputField(desc="Numbered defects, most severe first")
    chapter_tex: str = dspy.OutputField(desc="Corrected LaTeX chapter body only")


# ---------------------------------------------------------------------------
# Stage 4 — semantic critique
# ---------------------------------------------------------------------------


class CritiqueChapter(dspy.Signature):
    """Find the defects that only a careful reader can find.

    A deterministic verifier has already run: it compiled the chapter with
    pdflatex, checked every referenced file exists, confirmed the required
    sections are present and flagged numbers that do not appear in the analysis.
    Its findings are given to you. Do not repeat them and do not contradict
    them — report only what a machine check cannot see:

    - Claims the chapter makes that the analysis does not support. Quote the
      sentence.
    - Concepts, tables, charts, diagrams or mathematical derivations from the
      analysis that the chapter silently dropped.
    - Quantitative topics explained without a full derivation (hand-wavy
      "the formula is …" with no align/equation steps, missing assumptions,
      or undefined symbols).
    - Figures or tables that are included but never discussed in the prose, or
      discussed in a way that misdescribes what they show.
    - Explanations that are circular, contradictory, or so vague they teach
      nothing.
    - Exercises whose stated answer is wrong or unanswerable from the chapter.

    Severity: critical if it makes the chapter factually wrong or structurally
    broken, major if a reader would be misled or a required element is missing,
    minor for wording. Return an empty list if the chapter is sound; a clean
    chapter with no defects is a valid and expected outcome. Do not manufacture
    minor complaints to look thorough.

    Every defect needs a fix that names the exact change to make.
    """

    video_title: str = dspy.InputField()
    analysis: ChapterAnalysis = dspy.InputField(desc="Ground truth for this chapter")
    chapter_tex: str = dspy.InputField()
    mechanical_findings: str = dspy.InputField(
        desc="What the deterministic verifier already found; do not repeat it"
    )
    critique: SemanticCritique = dspy.OutputField()
