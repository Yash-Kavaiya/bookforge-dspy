# Architecture

## Layout

```
bookforge/
├── config.py            Settings (env-driven), logging, preflight
├── schemas.py           Typed contracts + validators that reject bad LLM output
├── workspace.py         Durable, concurrency-safe project directory
├── telemetry.py         Optional MLflow tracing
├── cli.py               build · optimize · evaluate · serve · artifacts · verify
│
├── ingest/              Where recordings come from
│   ├── base.py          Source protocol, stable ids, natural sort, titles
│   └── sources.py       YouTube · local video · transcripts · mixed folder
│
├── design/              How the book looks
│   ├── theme.py         Themes, palettes, the generated preamble
│   └── layout.py        Float declaration and placement macros
│
├── api/                 The web UI and its HTTP surface
│   ├── app.py           Routes: preview · upload · build · stream · download
│   ├── jobs.py          Job runner, event buffer, cooperative cancellation
│   └── static/          index.html · app.css · app.js (no build step)
│
├── tools/               Deterministic. No LLM touches these.
│   ├── latex.py         Rendering, sanitization, compilation, static inspection
│   ├── frames.py        Extraction, pHash dedupe, quality-ranked curation
│   ├── transcripts.py   SRT · VTT · timestamped · plain text → timed segments
│   └── youtube.py       Listing, download, captions, Whisper, cookie auth
│
├── dspy_runtime/        Everything that calls a model
│   ├── signatures.py    Typed I/O contracts — replaces prompts.py
│   ├── chunking.py      Transcript windowing
│   ├── lm.py            Per-role LMs, caching, retries, usage tracking
│   ├── artifacts.py     Save/load optimized programs, fingerprint-checked
│   ├── program.py       BookForgeProgram: the modules, wired
│   └── modules/
│       ├── analyst.py     map-reduce transcript analysis
│       ├── captioner.py   grounded frame captions
│       ├── writer.py      chapter composition and repair
│       ├── critic.py      semantic critique only
│       └── chapter.py     the verified refinement loop
│
├── metrics/             How good is it, concretely?
│   ├── grounding.py     Claims vs source
│   ├── structure.py     Chapter vs its contract
│   ├── verifier.py      The composite judgement
│   └── composite.py     DSPy metric adapters (scalar + GEPA feedback)
│
├── evaluation/
│   ├── datasets.py      Harvest examples from previous runs
│   ├── harness.py       Measure, and compare two programs
│   └── optimize.py      GEPA · MIPROv2 · SIMBA · Bootstrap
│
└── pipeline/            Orchestration, plain Python
    ├── intake.py · media.py · assets.py · chapter.py · compile.py
    ├── orchestrator.py  Sequential run + media prefetch + error isolation
    └── progress.py      Events and ETA
```

## Design decisions

### Deterministic by default; a model only where judgement is needed

Four stages call a model: analysis, frame captioning, chapter writing/refining,
and semantic critique. Everything else — asset rendering, compilation,
structural checking, grounding, book assembly — is code.

This is not conservatism. Charts are drawn by matplotlib *from the analyst's
validated numbers* rather than by an LLM emitting TikZ, because a rendering
mistake in a chart is invisible: it produces a chart. The same reasoning removed
three of the critic's four jobs.

### One verifier, used everywhere

`metrics/verifier.py:verify_chapter` is the only definition of chapter quality.
The refinement loop, the runtime guards, the optimizers and the `verify` CLI all
call it.

A metric that drifts from the runtime check is worse than no metric — it
optimizes confidently for the wrong thing. Sharing one implementation makes the
drift impossible rather than merely unlikely.

### The workspace is the state; memory is a cache

Every stage writes its output to disk and records a status in `manifest.json`
before moving on. There is no session object to lose. `pending_videos()` reads
the manifest, so resume is not a feature to maintain — it is what happens when
you re-run.

This also killed a whole class of ADK bug: state that lives only in a session
dict is state that can be clobbered, and `frame_captions` failed precisely in
the gap between two stages' views of that dict.

### Instructions are parameters

Every prompt lives in a `dspy.Signature` docstring. GEPA and MIPROv2 rewrite
them; artifacts persist the result; the runtime loads it. Prompt quality becomes
something measured rather than argued about.

A side benefit worth naming: ADK injected session state into instruction strings
via `{placeholder}` formatting, so a prompt could not contain a literal brace.
That is why the original writer prompt had to *describe* LaTeX in English
("include a figure environment with caption and label") instead of showing it.
DSPy passes inputs as arguments, so the writer's contract shows real markup.

### The writer places; BookForge typesets

The ADK writer emitted complete figure environments, choosing the file path, the
width, the caption and the float placement. Three of those four it had no basis
for, and the first it could get wrong in a way nothing caught — a path to a file
that did not exist compiled to a missing-image box.

Assets are now declared once from the manifest, in a generated block at the top
of each chapter, and the writer places one by label:

```latex
\bfDeclareFigure{fig:ch01-frame3}{figures/frame_03_145s.jpg}{0.82}{The speaker's
  slide at 2:25, showing the flat pod network model}   % generated
\bfFigure{fig:ch01-frame3}                              % written
```

The writer decides *where in the argument* an asset belongs, which is the one
judgement it is actually qualified to make. Everything else — path, sizing,
caption, placement policy — comes from the manifest, and an undeclared label is
a compile error rather than a silently broken float.

### Sources are an interface, not a special case

`ingest.resolve_source` maps a URL, a file, a folder or a list to one of four
sources, all satisfying the same protocol: `discover()` enumerates cheaply, with
no downloads and no transcription. `VideoRecord.source` records which kind each
chapter came from, so a single workspace can mix YouTube videos, local files and
bare transcripts — and media acquisition dispatches per record rather than
assuming.

Discovery being cheap is what makes the UI's preview possible: the folder scan
that shows you the chapter list costs a directory walk and an ffprobe, not an
hour and a model bill.

### Concurrency is IO-only

Media acquisition — download, transcribe, extract frames — runs in a thread pool
for upcoming videos while the current one is in its LLM stages. Those threads
only write per-video cache files. **Every manifest mutation stays on the main
thread**, and the manifest itself is guarded by a thread lock plus a file lock
for the multi-process case.

The rule is simple: parallelize the IO, serialize the state.

## Data flow

```
manifest.json           source → videos, statuses, quality scores
uploads/batch-<hex>/    files staged by the UI, ingested like any local folder
videos/<id>/            video.mp4 · transcript.txt · media.json · frames_raw/
chapters/<NN>_<slug>/   chapter.tex · analysis.json · assets.json · verdict.json
                        figures/ · tables/
build/qa_<id>/          standalone compile of each candidate
book/                   main.tex · book.pdf · book_manifest.json
artifacts/              analyst.json · chapter.json + .meta.json
```

`verdict.json` is the audit trail: the score, every defect, and the full
iteration history with per-candidate scores. When a chapter looks wrong, that
file says what the pipeline thought of it and why.

## Extension points

**A new LLM stage.** Add a `dspy.Signature`, wrap it in a `dspy.Module`, attach
it in `BookForgeProgram`, and register a metric in `metrics/composite.py` if it
should be optimizable.

**A new quality check.** Add it to `metrics/structure.py` (returning `Defect`s)
or `metrics/verifier.py` (contributing to the score). Every consumer — loop,
guards, optimizers — picks it up automatically.

**A new input source.** Implement `discover()` returning a `Discovery`, add it
to `resolve_source`, and teach `pipeline/media.py` how to acquire that kind. The
CLI, the API and the UI pick it up with no further changes.

**A new book design.** Add a `BookTheme` to `design/theme.py`. Themes are data —
palette, geometry, running-head content, callout styling — so a new one is a
literal, not a template fork.

**A new output format.** `tools/latex.py` and `design/` are the only modules
that know about LaTeX. The analysis is format-neutral.

**A different model.** Any LiteLLM identifier in `BOOKFORGE_*_MODEL`. Roles are
independent, so you can run a large analyst and a small captioner.
