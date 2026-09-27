# BookForge (DSPy)

**Turn a channel, a folder of recordings, or a pile of transcripts into a
typeset, verified LaTeX textbook.**

A rewrite of [adk-demo](https://github.com/Yash-Kavaiya/adk-demo) that replaces
Google ADK with [DSPy](https://dspy.ai). The point of the rewrite is not the
framework swap — it is what the swap makes possible: **every prompt becomes a
parameter you can optimize against a measurable objective, and every chapter is
verified by execution before it ships.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![DSPy 3.3](https://img.shields.io/badge/DSPy-3.3-orange.svg)](https://dspy.ai)
[![Tests](https://img.shields.io/badge/tests-439_passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)

---

## What changed, and why it matters

The original pipeline worked. It also had a class of bug that working pipelines
specialize in: failures that produce plausible output. A caption that fits any
frame. A statistic nobody said. A chapter that got worse on its last revision
and shipped anyway. None of these throw an exception, and none are visible in
a PDF unless you go looking.

| | ADK version | This version |
|---|---|---|
| **Prompts** | Hand-edited strings in `prompts.py` | Typed signatures, optimized by GEPA/MIPROv2 against a metric |
| **Long videos** | Middle of the transcript deleted past a char cap | Windowed map-reduce; every second is analysed |
| **Hallucination defence** | A sentence in the prompt | `check_grounding` verifies every number against the transcript |
| **Chapter QA** | An LLM asked to compile, diff assets, and check numbers by reading | Deterministic verifier does all three; the LM judges only what needs reading |
| **Refinement loop** | Stopped on the refiner; the last draft shipped unverified | Every candidate scored; **the best one wins, always** |
| **Figure captions** | Feature was wired but structurally dead (see below) | Captioned after curation, from the transcript at that timestamp |
| **Manifest writes** | `asyncio.Lock` guarding OS threads (no mutual exclusion) | Thread lock + file lock; safe across threads *and* processes |
| **Bad LLM output** | Ragged tables padded, mismatched charts truncated | Rejected at the schema boundary |
| **Input** | A YouTube URL, and nothing else | YouTube, local videos, a folder, or transcripts alone |
| **Float placement** | The writer emitted whole figure environments | BookForge declares; the writer places by label |
| **Page design** | LaTeX `book` defaults | Themed: running heads, callouts, margin notes, chapter openers |
| **Interface** | CLI only | CLI, Python, and a web UI with live progress and per-chapter verdicts |

The full catalogue, with evidence and fixes, is in
**[docs/ACCURACY.md](docs/ACCURACY.md)**.

### The bug that explains the rest

`ChapterAnalysis` had a `frame_captions` field. The analyst prompt asked for
"caption overrides keyed by frame filename". The asset stage looked them up:

```python
overrides = analysis.frame_captions_dict
caption = overrides.get(frame.file) or f"Video frame at {mm_ss} illustrating…"
```

But the pipeline ran `analyst → assets`, and the filenames (`frame_03_145s.jpg`)
were invented by the *asset* stage — after the analyst had already answered. The
analyst had never seen a filename, so the lookup never matched, and every figure
in every book fell through to the fallback.

Fully wired, plausible-looking, and inert. It is fixed here by ordering plus
context: captioning happens after curation, with the transcript spoken around
each frame's timestamp — and frames showing nothing worth printing are dropped
rather than padded with filler.

---

## Install

```bash
git clone https://github.com/Yash-Kavaiya/bookforge-dspy.git
cd bookforge-dspy
pip install -e ".[ui,dev,observability]"
```

System dependencies: **ffmpeg** (media) and **pdflatex** (TeX Live or MiKTeX).
Without pdflatex the pipeline runs in draft mode and says so — it never records
verification it did not perform. Neither is needed for transcript-only input.

Or take both, plus a full TeX distribution, from the image:

```bash
docker build -t bookforge .
docker run --rm -p 8000:8000 -v "$PWD/data:/app/data" --env-file .env bookforge
```

```bash
cp .env.example .env   # then add your API key
```

Any [LiteLLM](https://docs.litellm.ai/docs/providers) model works:

```env
# NVIDIA NIM (or any OpenAI-compatible gateway: vLLM, Ollama, Together…)
BOOKFORGE_OPENAI_API_KEY=nvapi-...
BOOKFORGE_OPENAI_API_BASE=https://integrate.api.nvidia.com/v1
BOOKFORGE_ANALYST_MODEL=openai/nvidia/nemotron-3-ultra-550b-a55b

# or Gemini
GOOGLE_API_KEY=...
BOOKFORGE_ANALYST_MODEL=gemini/gemini-2.5-pro
```

## Use

### The web UI

The UI is an Angular app (`frontend/`) that FastAPI serves once built — build
it once, then just run `bookforge serve`:

```bash
pip install -e ".[ui]"

cd frontend && npm install && npm run build && cd ..   # compiles to bookforge/api/webapp/
bookforge serve                       # http://127.0.0.1:8000
```

`npm run build` needs to be re-run after pulling changes to `frontend/`; the
Python side picks up the compiled output automatically (`bookforge serve`
doesn't rebuild it for you). The Docker image builds it as part of `docker
build`, so nothing extra is needed there — see [Install](#install) above.

Pick a source, preview the chapters it found *before* spending a model call,
start the build, and watch it: live progress over a WebSocket, per-chapter
verdicts with the defects the verifier found and the full refinement history,
and the PDF when it lands. Drag files in, or point it at a folder on the
server. A separate Config page lets you review and change every
hyperparameter and provider API key live, with a one-click connection test,
and writes changes straight back to `.env`.

There is no authentication — it binds to localhost on purpose. See
[docs/INPUTS_AND_UI.md](docs/INPUTS_AND_UI.md) before exposing it.

### The CLI

```bash
# Any source: a URL, files, or a folder of videos and/or transcripts
bookforge build "https://www.youtube.com/@channel" --max-videos 3
bookforge build ./talks --title "Decoding AI"
bookforge build ./transcripts                  # no download, no ffmpeg, no Whisper
bookforge build talk1.mp4 talk2.mkv --theme academic

# Optimize the prompts against your own metrics
bookforge optimize analyst --optimizer gepa --budget medium
bookforge optimize chapter --optimizer miprov2

# Score without optimizing
bookforge evaluate chapter --optimized

# Inspect compiled artifacts
bookforge artifacts

# Verify one chapter already on disk
bookforge verify data/my-channel/chapters/01_intro/chapter.tex

# Serve the UI
bookforge serve --port 9000 --theme compact
```

Runs are checkpointed per stage. Interrupt at any point and re-run — verified
chapters are skipped, failed ones are retried.

```python
from bookforge import run_pipeline
result = run_pipeline("https://www.youtube.com/@channel")
print(result.summary())
```

---

## How it works

```
URL · files · folder · transcripts
    │
    ▼
ingest ──────────────► resolve_source → the right source, one interface
    │
    ▼
intake ──────────────► manifest.json (resumable; one chapter per video)
    │
    ▼  for each pending video, media for the next N prefetched in parallel
    │
    ├─ media ────────► download · captions or Whisper · pHash-deduped frames
    │
    ├─ analysis ─────► ┌ window the transcript on segment boundaries
    │                  │ analyse each window          (DSPy: AnalyzeTranscriptChunk)
    │                  └ merge and deduplicate        (DSPy: MergeChunkAnalyses)
    │                    → grounding check; resample once if it fails
    │
    ├─ assets ───────► matplotlib charts · booktabs tables · curated frames
    │                    → captions from the transcript  (DSPy: CaptionFrames)
    │                    → each declared once, by label
    │
    ├─ chapter ──────► ┌ write                          (DSPy: WriteChapter)
    │                  │ VERIFY: compile · structure · grounding
    │                  │ critique what needs reading    (DSPy: CritiqueChapter)
    │                  └ refine ──┐                     (DSPy: RefineChapter)
    │                      ▲      │  bounded; every candidate scored
    │                      └──────┘  BEST CANDIDATE WINS
    │
    ▼
book ────────────────► theme · front matter · label de-duplication · book.pdf
```

Deterministic stages are plain Python. Only the four boxes marked `DSPy:` call a
model, and each is independently optimizable.

### The verifier is the centre of gravity

One function answers *how good is this chapter, concretely?* — by compiling it,
checking its structure, and grounding its claims:

```python
verdict = verify_chapter(tex, context)
verdict.score            # 0..1, deterministic
verdict.compiles         # pdflatex actually ran and succeeded
verdict.invented_assets  # paths not in the manifest
verdict.ungrounded_numbers
verdict.feedback_text()  # numbered defects, each with its fix
```

Because it is deterministic, the same draft always scores the same. That single
property is what makes everything else work:

- the **loop** can compare iteration 3 against iteration 1 and keep the better one;
- the **optimizers** get a stable reward signal;
- the **runtime** can refuse to ship a chapter below threshold;
- the **critic LM** is told what the regex already found, so it spends its
  attention on judgement instead of re-deriving string comparisons.

The scoring weights compilation at 40%, structure at 30%, grounding at 25% and
polish at 5% — an uncompilable chapter has no value whatever else is right
about it.

### Monotonic refinement

This is the property the ADK loop lacked, asserted directly in
[`tests/test_chapter_loop.py`](tests/test_chapter_loop.py):

```python
assert result.chapter_tex == good, "the loop shipped a worse revision"
assert result.score == max(entry.score for entry in result.result.history)
```

Refinement can never make the shipped chapter worse than the draft it started
from. It stops early when the score clears the bar, when the refiner stops
changing anything, or after two attempts that fail to improve.

---

## The book

The output is a designed book, not a LaTeX dump: running heads (chapter on the
verso, section on the recto), page numbers on the outer edge where a thumbing
reader finds them, chapter openers with a ghosted numeral, coloured callouts for
key ideas and definitions, margin notes, a generated index, and three themes
(`textbook`, `academic`, `compact`).

The change that mattered most is not cosmetic. The writer used to emit whole
figure environments — path, width, caption *and* float placement — which put
four decisions in the hands of the component least equipped to make any of them,
and let it reference files that did not exist. BookForge now declares every
asset once from the manifest, and the writer places it with a single marker
where it belongs in the argument:

```latex
Kubernetes assigns every pod a routable address, so the network is flat.

\bfFigure{fig:ch01-frame3}
```

**The writer chooses where; BookForge chooses how.** An undeclared label is a
compile error rather than a silently broken float, and float parameters are
loosened from LaTeX's defaults — which were tuned for documents with a handful
of floats, not a chapter carrying twelve, and otherwise defer almost everything
to the end of the chapter. Sections act as float barriers, so a figure cannot
drift under the next heading where its caption would contradict its context.

---

## Optimization

BookForge's metrics are **reference-free** — grounding compares an analysis
against its own transcript, structure compares a chapter against its own
analysis, and compilation is checked by running pdflatex. No gold-standard
chapters are needed, so **every video the pipeline processes becomes a training
example**, and the dataset grows as the system runs.

```bash
bookforge build "https://www.youtube.com/@channel" --max-videos 6  # produces data
bookforge optimize analyst --optimizer gepa --budget medium        # learns from it
bookforge build "https://www.youtube.com/@other-channel"           # uses the result
```

GEPA fits best because our metrics emit *textual* feedback — not "score 0.62"
but "the chapter references `figures/hallucinated.pdf`, which is not in the
manifest". Its reflection model reads that and rewrites the instruction to
prevent that specific failure.

Compiled artifacts are versioned by signature fingerprint and refused if the
code has moved on, so stale instructions can never half-apply. **An optimizer
run that fails to beat the baseline is not saved.**

See **[docs/OPTIMIZATION.md](docs/OPTIMIZATION.md)** for optimizer selection,
budgets and cost.

---

## Development

```bash
pytest                      # 439 tests, no network, no API key
pytest --cov=bookforge      # 82% coverage
ruff check bookforge tests
```

Every test runs offline. Model calls are driven by scripted `dspy.Module`
stubs, so the control flow — windowing, the refinement loop, error isolation,
resume — is exercised without a live LM.

## Documentation

| | |
|---|---|
| [docs/ACCURACY.md](docs/ACCURACY.md) | Every defect found in the ADK version, with evidence and fix |
| [docs/INPUTS_AND_UI.md](docs/INPUTS_AND_UI.md) | The four input sources, YouTube cookies, the web UI and its API |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module layout and design decisions |
| [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) | Running GEPA/MIPROv2, budgets, cost |
| [docs/MIGRATION.md](docs/MIGRATION.md) | ADK → DSPy, component by component |

## License

Apache 2.0. See [LICENSE](LICENSE).
