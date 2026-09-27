# Accuracy review

An audit of [`Yash-Kavaiya/adk-demo`](https://github.com/Yash-Kavaiya/adk-demo)
at the commit this project was rewritten from, plus the defects found while
building the replacement.

The original repository's own `CODEBASE-REVIEW.md` graded the Python
implementation **A+ (Outstanding)** on architecture and **A (Very Good)** on
code quality. That grade is defensible on the axes it measured — separation of
concerns, type safety, error isolation, test coverage. It measured the wrong
axes. Every finding below is a *behavioural* defect that leaves the structure
untouched: nothing raises, nothing is untyped, no test fails, and the output is
quietly wrong.

That is the characteristic failure mode of LLM pipelines. A traditional bug
crashes; an LLM-pipeline bug produces a plausible artifact. A caption that fits
any frame, a statistic nobody said, a chapter that got worse on its final
revision — these look exactly like success in a PDF.

**Severity:** *critical* = wrong content reaches the reader or the build breaks ·
*high* = silent data corruption · *medium* = degraded quality or lost work ·
*low* = robustness and operability.

---

## 1. `frame_captions` was structurally impossible to satisfy — *critical*

**Where:** `schemas.py`, `prompts.py:ANALYST_INSTRUCTION`, `agents/assets.py`,
and the stage ordering in `agents/orchestrator.py`.

`ChapterAnalysis` carried the field:

```python
frame_captions: list[FrameCaptionItem] = Field(
    default_factory=list,
    description="Optional per-frame caption overrides",
)
```

The analyst was asked for it:

> `10. frame_captions — optional caption overrides keyed by frame filename.`

And the asset stage consumed it:

```python
overrides = analysis.frame_captions_dict
for j, frame in enumerate(curated, start=1):
    caption = overrides.get(frame.file) or (
        f"Video frame at {_mm_ss(frame.timestamp_sec)} illustrating "
        f"the segment discussed in this chapter."
    )
```

But the pipeline order was `media → analyst → assets`, and `frame.file` is
generated *by the asset stage*:

```python
dest = dest_dir / f"frame_{i:02d}_{int(asset.timestamp_sec)}s.jpg"
```

The analyst had never seen a filename and could not have guessed one. Every
`overrides.get(...)` missed. Every figure in every book therefore carried:

> *Video frame at 02:25 illustrating the segment discussed in this chapter.*

A textbook whose every figure has the same non-caption. The feature was fully
wired, described in the prompt, typed in the schema, consumed by the renderer —
and inert.

**Why nothing caught it.** `.get()` with a fallback is total: no exception, no
log line, no failed assertion. The tests checked that captions *existed*, which
they did. Reviewing the analyst and the asset stage separately, each is correct;
the bug lives in the seam between them.

**Fix.** [`dspy_runtime/modules/captioner.py`](../bookforge/dspy_runtime/modules/captioner.py)
captions frames *after* curation, when the filenames exist, and supplies the
transcript spoken around each frame's timestamp so the caption describes what
was on screen. Frames with no instructional content are dropped rather than
padded. Covered by `tests/test_frames_and_captions.py::TestCaptioner`.

---

## 2. The refinement loop shipped its one unverified draft — *critical*

**Where:** `agents/critic.py:make_qa_loop`, `agents/orchestrator.py`.

```python
LoopAgent(
    name="chapter_qa_loop",
    sub_agents=[make_critic_agent(settings), make_refiner_agent(settings)],
    max_iterations=settings.qa_max_iterations,
)
```

The critic could approve and escalate out. Otherwise the refiner rewrote
`chapter_tex` and the loop went round. When the iteration budget ran out, the
loop stopped — **on the refiner**. That final rewrite went straight to disk:

```python
tex = latex_tools.sanitize_chapter_tex(state.get("chapter_tex", ""))
Workspace.write_text(ws.chapter_dir(record) / "chapter.tex", tex)
```

It was never compiled, never asset-checked, never read by anything.

So the worst case was not "we ran out of attempts and shipped our best effort".
It was **"we ran out of attempts and shipped the one draft nobody looked at"** —
which could be worse than the first draft and might not compile. And because the
critic's verdict existed only as prose in session state, nothing *could* compare
iteration 3 against iteration 1 to notice.

The status then recorded was `"written"` rather than `"verified"`, and
`BookCompilerAgent` included both.

**Fix.** [`dspy_runtime/modules/chapter.py`](../bookforge/dspy_runtime/modules/chapter.py)
scores every candidate with the same deterministic verifier and returns the
best one ever seen. The invariant is asserted directly:

```python
assert result.score == max(entry.score for entry in result.result.history)
```

Refinement now stops early on success, on an unchanged revision, or after two
non-improving attempts. `tests/test_chapter_loop.py::TestMonotonicity`.

---

## 3. Long transcripts were analysed from their ends — *critical*

**Where:** `tools/youtube.py:load_transcript_text`.

```python
def load_transcript_text(path: Path, max_chars: int = 120_000) -> str:
    text = Path(path).read_text(...)
    if len(text) > max_chars:
        half = max_chars // 2
        text = text[:half] + "\n...[middle truncated]...\n" + text[-half:]
    return text
```

For a three-hour lecture this deletes the entire middle of the talk. The
analyst was simultaneously instructed:

> *Be faithful: prefer omission over invention. The book will be judged on
> accuracy against the transcript.*

So the model faithfully described a video it had seen only the ends of, and
produced a confident chapter with an hour-shaped hole in it. Nothing downstream
could tell the difference between "the speaker did not cover that" and "we
deleted the part where they did".

**Fix.** [`dspy_runtime/chunking.py`](../bookforge/dspy_runtime/chunking.py)
windows the transcript on segment boundaries with overlap; the analyst runs
map-reduce over every window and a merge step reconciles them. When a cost
budget is hit, windows are *widened* to keep full coverage rather than dropping
the tail. `tests/test_chunking.py::TestChunking::test_every_segment_is_covered`.

---

## 4. Hallucination defence was a sentence in a prompt — *critical*

**Where:** `prompts.py`, `agents/critic.py`.

The only barrier between an invented benchmark number and the printed book was
instruction text ("Never fabricate numbers"), plus a critic asked to check
numeric fidelity by reading. A fabricated figure that survived both would be
rendered by matplotlib into a vector chart and typeset with a caption — arriving
in the PDF looking more authoritative than the real numbers, because it had been
through more machinery.

**Fix.** [`metrics/grounding.py`](../bookforge/metrics/grounding.py) makes it a
measurement, applied at two boundaries: analysis-against-transcript and
chapter-against-analysis. It reconciles the forms a number legitimately takes —
`twenty five` ↔ `25`, `0.17` ↔ `17 percent`, `1.5k` ↔ `1500` — because a
digit-only check under-reports so badly on auto-captions that it would train the
optimizer to strip numbers entirely.

It runs in three places: as a runtime guard that resamples a poorly-grounded
analysis, as a verifier component, and as an optimizer metric.

---

## 5. Ragged table rows were silently padded — *high*

**Where:** `tools/latex.py:render_table_fragment`.

```python
padded = list(row)[:ncols] + [""] * max(0, ncols - len(row))
```

A row with a missing cell was repaired into a *different, wrong* row. If the LLM
dropped the second of four cells, every value after it shifted one column left
and the last column was blanked — producing a table that is well-formed,
booktabs-styled, and misattributes each figure to the wrong heading.

**Fix.** `TableSpec` rejects non-rectangular rows at the schema boundary, so
malformed output is retried rather than typeset. `tests/test_schemas.py`.

---

## 6. Mismatched chart series were silently truncated — *high*

**Where:** `tools/latex.py:render_chart`.

```python
elif len(categories) > len(values):
    categories = categories[: len(values)]
elif len(values) > len(categories):
    values = values[: len(categories)]
```

Five categories and three values produced three bars carrying the *first three*
labels. If the model emitted values in a different order from categories — or
dropped one — the chart is confidently mislabelled, and the caption asserts it.

**Fix.** `ChartSpec` requires `len(categories) == len(values)`. Same principle:
a schema that can silently corrupt data is a bug, not a convenience.

---

## 7. `asyncio.Lock` guarding OS threads — *high*

**Where:** `tools/workspace_lock.py`.

```python
_workspace_locks: dict[str, asyncio.Lock] = {}

async def safe_update_video(ws, video_id, **changes):
    lock = get_workspace_lock(ws.root)
    async with lock:
        return ws.update_video(video_id, **changes)
```

The racing writers were OS threads:

```python
await asyncio.to_thread(acquire_media_bundle, record, ws, settings)
```

An `asyncio.Lock` serializes coroutines on one event loop. It provides **no
mutual exclusion between threads at all**. The lock read as thread-safe, was
documented as thread-safe ("prevent race conditions when multiple videos are
processed in parallel"), and was not. `update_video` is a read-modify-write of
one JSON file, so a lost update makes a finished chapter look pending forever
and a resumed run redoes work it already paid for.

The window was narrow in practice because prefetch threads only wrote
`media.json`. It would have opened the moment anyone raised
`chapter_concurrency`, which the config already anticipated.

**Fix.** [`workspace.py`](../bookforge/workspace.py) uses a process-wide
`threading.RLock` *and* an on-disk lock file, so concurrent threads and
concurrent processes sharing a workspace are both safe.
`tests/test_workspace.py::TestConcurrency` runs 24 threads through a barrier
and asserts no update is lost.

---

## 8. One writer bypassed the lock entirely — *medium*

**Where:** `agents/assets.py`.

```python
ws.update_video(record.video_id, status="assets", error="")
```

Every other stage used `safe_update_video`; this one called the raw method. Even
had the lock worked, this path was outside it.

**Fix.** There is now exactly one supported mutation path (`Workspace.update_video`),
and it is internally locked — you cannot call the unsafe version because it does
not exist.

---

## 9. The sanitizer destroyed the examples it was cleaning — *medium*

**Where:** `tools/latex.py:sanitize_chapter_tex`.

```python
body = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\}\n?", "", body)
```

Applied blindly to the whole body. A chapter *about* LaTeX — entirely plausible
for a programming channel — has `\usepackage` inside a `verbatim` block as the
thing being taught. The sanitizer deleted it, leaving a code example missing its
key line, still compiling, teaching the wrong thing.

**Fix.** Verbatim, lstlisting and minted blocks are swapped for placeholders
before sanitizing and restored afterwards.
`tests/test_latex.py::TestSanitizer::test_verbatim_blocks_are_protected`.

---

## 10. Frame curation discarded the quality signal it computed — *medium*

**Where:** `tools/frames.py`.

`dedupe_frames` measured sharpness, brightness and contrast for every frame.
`curate_frames` then selected purely by position:

```python
step = len(assets) / max_frames
picked = [assets[int(i * step)] for i in range(max_frames)]
```

A blurry mid-transition frame beat a crisp slide sitting one index away. The
measurements were used only as a pass/fail junk filter and then thrown away.

**Fix.** The timeline is split into K buckets and the highest-quality frame
*within each bucket* wins — chronological coverage preserved, quality signal
actually used. `tests/test_frames_and_captions.py::TestCuration::test_quality_wins_within_a_bucket`.

---

## 11. Quadratic JSON salvage — *medium*

**Where:** `agents/model_resolver.py:extract_clean_json`.

```python
starts = [i for i, c in enumerate(text) if c == "{"]
ends = [i for i, c in enumerate(text) if c == "}"]
for start in starts:
    for end in reversed(ends):
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
```

A `json.loads` per brace pair. A 16k-token response with 500 braces is up to
250,000 parse attempts on strings averaging tens of kilobytes — and it is still
guessing, because the largest parseable object is not necessarily the intended
one.

**Fix.** Deleted. DSPy's adapters own schema-conformant decoding and retry, so
the whole failure class is handled upstream by typed output fields.

---

## 12. Duplicate labels across chapters — *medium*

**Where:** `agents/compiler.py`.

Chapter bodies were concatenated with no label reconciliation. Two chapters
generated from the same template share `\label{ch:01-...}` and figure labels.
pdflatex resolves every `\ref` to the *last* definition and emits a warning
among hundreds — so a cross-reference silently points the reader at a figure in
a different chapter.

**Fix.** `strip_duplicate_labels` renames collisions at assembly time and
rewrites the matching refs within the same body.
`tests/test_integration.py::TestHappyPath::test_duplicate_labels_across_chapters_are_renamed`.

---

## 13. `preflight` mutated cached global settings — *low*

**Where:** `main.py`.

```python
def preflight_check(settings: Settings) -> list[str]:
    if settings.compile_latex and not shutil.which("pdflatex"):
        warnings.append(...)
        settings.compile_latex = False   # side effect
```

`get_settings()` is `@lru_cache`d, so a "check" silently reconfigured the whole
process. Correct behaviour, invisible mechanism — the kind of thing that makes a
later bug take an afternoon.

**Fix.** `preflight` returns warnings and mutates nothing; the CLI decides what
to do with each. `tests/test_config_and_cli.py::TestPreflight`.

---

## 14. `max_videos` could silently under-deliver — *low*

**Where:** `tools/youtube.py:list_channel_videos`.

```python
"playlistend": max_videos * 3 if max_videos else None,
```

Duration filtering happened *after* extraction. A channel whose recent uploads
are mostly Shorts returns fewer videos than requested, with no explanation.

**Fix.** Over-fetch more generously and log explicitly when the duration filter
prevented the cap from being met.

---

## 15. The critic did four jobs; three were string comparison — *design*

**Where:** `prompts.py:CRITIC_INSTRUCTION`.

The critic LM was asked to: call a compile tool, diff every asset reference
against a manifest, check that required sections were present, and verify
numbers against the analysis. Three of those are exact string work — a regex
does them perfectly; a language model does them *probabilistically*. When the
model skipped a step, nothing noticed, because its verdict was free prose.

That is also why refinement could not be evaluated: there was no score to
compare across iterations, only opinions.

**Fix.** [`metrics/verifier.py`](../bookforge/metrics/verifier.py) does the
mechanical work deterministically and always. The critic LM is given those
findings and asked only for what needs reading comprehension: silently dropped
content, prose contradicting its own figures, wrong exercise answers. This is
cheaper, reproducible, and lets the LM spend attention where it has an edge.

---

## 16. No cost or usage visibility — *low*

Nothing recorded tokens or calls. A book's cost was unknowable, and an optimizer
run — which makes far more calls than production — could not be budgeted.

**Fix.** `dspy_runtime/lm.py` tracks per-role usage via an LM callback and
reports it at the end of every run and optimization.

---

## Defects found in the new code, by its own tests

Worth recording: the tests earned their keep during the rewrite.

**`tex_escape` double-escaped its own replacements.** Unicode substitution ran
first, producing `$\to$`, and the special-character pass then escaped the `$`
and `\` it had just inserted — yielding `\$\textbackslash{}to\$`. Caught by
`test_unicode_is_converted`. Fixed with a single merged lookup table.

**A chapter could never be accepted without pdflatex.** `is_acceptable` required
`compiles`, which is false when compilation is *skipped* as well as when it
fails. In draft mode every chapter therefore burned the full refinement budget
and could never terminate early. Caught by
`test_high_scoring_draft_skips_refinement`. Fixed by distinguishing
`compile_skipped` from `compiles` — the loop can accept a structurally sound
draft, but the pipeline still never records it as *verified*, because that
verification did not happen.

**The grounding metric punished good writing.** Lexical overlap between a
concept and the transcript entered the score as a continuous term, so an
explanation that introduced its own vocabulary — which a textbook chapter is
supposed to do — scored below one that parroted the speaker. As an optimizer
objective this would have actively taught the model to paraphrase less. Fixed:
overlap now contributes only as a penalty for concepts that are *both* lexically
disjoint *and* whose heading never appears in the talk — the signature of an
off-topic invention rather than an expansion.

**A chart type failure cost the whole figure.** A backend quirk in one chart
kind raised, the asset stage dropped the figure, and the chapter was left
referencing a path that did not exist — breaking the build. Fixed: rendering
falls back to a bar chart, which carries the same data.

### Four more found only by running the pipeline for real

The unit suite was green when an end-to-end run — stubbed network and model,
but **real matplotlib and real pdflatex** — produced no book at all. Every
chapter scored 0.83 and was rejected. Three of the four defects were the
verifier *inventing* problems, which is the more dangerous direction: a lax
verifier ships a mediocre chapter, but a paranoid one burns the refinement
budget chasing phantoms and teaches the optimizer to avoid correct behaviour.

**The shared preamble did not compile.** `\usepackage[T1]{fontenc}` without a
scalable font family selects the bitmap EC fonts, and `microtype`'s font
expansion is a *fatal* error on non-scalable fonts:

```
! pdfTeX error (font expansion): auto expansion is only possible with scalable fonts.
==> Fatal error occurred, no output PDF file produced!
```

Every book, on every machine, at the last step. Fixed by loading `lmodern`
before `fontenc`. No unit test could have caught this — it requires running TeX.

**Manifest captions were treated as writer hallucinations.** The contract
requires the writer to copy figure captions from the manifest verbatim. Those
captions are generated by BookForge and contain timestamps ("the slide shown at
90 seconds"). The chapter-vs-analysis grounding check found `15, 30, 45, 60, 75,
90` in the chapter and nowhere in the analysis, and raised a *critical* defect
accusing the writer of inventing numbers the pipeline had handed it. Fixed:
`check_chapter_grounding` takes the manifest captions as an additional allowed
source.

**Table labels were reported as dangling.** `render_table_fragment` writes
`\label{tab:ch01-1}` *inside* the fragment; the chapter only `\input`s it and
refers to `Table~\ref{tab:ch01-1}`. The structural check scanned the chapter
body for label definitions, did not find it, and flagged a correct reference as
dangling — penalising the writer for following the contract exactly. Fixed:
manifest labels count as defined.

**Every cross-reference was "undefined".** LaTeX resolves `\ref` through the
`.aux` file, which is read back on the *next* pass. The QA check compiles once
for speed, so every forward reference is undefined by construction — and the
contract requires every float to be referenced, so this fired on every chapter.
Fixed: undefined-reference detection is suppressed at `passes=1`, where it
cannot mean anything.

After the four fixes the same run produced a 365 KB `book.pdf`, two chapters,
both scoring **1.00 with zero defects**, with duplicate labels correctly renamed
across chapters. The lesson generalises: a test suite that never executes the
real toolchain will certify a pipeline that cannot produce its own output.

### Six more found while building the design and the UI

**The book had no page numbers.** `fancyhdr` assigns a footer per slot, and both
the page number and the book title were assigned to the same one. The second
assignment silently replaced the first, so every page in the book carried the
title and no page number at all. It typeset, it looked deliberate, and only
opening the PDF and looking for a number revealed it. Fixed by putting the page
number on the outer edge (`LE,RO`) and the title on the inner (`RE,LO`), which
is where a thumbing reader expects each.

**Table fragments nested a float inside a float.** `render_table_fragment`
emitted a complete `table` environment; the placement macro opens one around
whatever it is given. The result was `! LaTeX Error: Not in outer par mode` —
fatal, on every chapter containing a table. Fixed: fragments are bare
`tabular`s, and the caption lives only in the manifest, which is also the only
place it can be checked against.

**Error extraction had stopped matching errors.** `_ERROR_RE` matched lines
beginning with `!`, the classic TeX format. We pass `-file-line-error`, which
emits `file.tex:42: Undefined control sequence` instead. Since the day that flag
was added, every compile failure handed the critic font-loading noise instead of
the actual error, so the refinement loop was being asked to fix problems it
could not see. Fixed by matching both formats.

**Static inspection read LaTeX comments as markup.** The generated declaration
block carries a comment documenting the placement macro; the structural check
counted it as a use. BookForge's own documentation of a feature was being parsed
as an instance of it. Fixed with `strip_comments()` before any inspection —
which also means a commented-out figure no longer counts as a placed figure.

**The unnumbered-section macro was not recognised as a section.** `\bfsection`
produces a section; the structure check looked for `\section`. Every chapter
using an unnumbered heading — which the contract encourages — was told its
sections were missing. Fixed in the section pattern.

**Extension lists could only be set as JSON.** `video_extensions` is a tuple, and
pydantic-settings parses tuples from the environment as JSON, so
`BOOKFORGE_VIDEO_EXTENSIONS=.mp4,.mkv` — the spelling anyone would try — raised
a `SettingsError` naming the field but not the expected format. Fixed with
`NoDecode` plus a validator that takes either spelling; the JSON form still
works, because breaking the one syntax that *did* work would have degraded it to
a single nonsense extension matching nothing, and a folder scan that silently
finds no videos is worse than an error.

### Four more found by typesetting a real book from a real folder

Building an actual book from a folder of transcripts — real matplotlib, real
pdflatex, real makeindex, the model stubbed — then rendering the pages and
reading them.

**The index was never built.** `\term{}` writes `\index{...}` entries, the
preamble loads `makeidx` and calls `\makeindex`, and the back matter calls
`\printindex`. Nothing ever ran `makeindex`, which is the step that turns the
`.idx` file into the `.ind` file `\printindex` reads. So every entry was
dutifully recorded, `\printindex` found no `.ind` and printed nothing, and every
book shipped without an index and without a single error anywhere. This is the
`frame_captions` defect exactly: a feature wired end to end with one missing
link in the middle, invisible unless you go and look at the output. Fixed by
running `makeindex` between compile passes — from the build directory, since
TeX's `openout_any` setting refuses an output file named by absolute path. The
image build now asserts the binary is present, because a missing `makeindex`
would restore the silent version of this bug.

**A callout title was parsed as a key list.** The natural definition,
`\newtcolorbox{keyidea}[1][]`, makes the optional argument raw tcolorbox keys.
So `\begin{keyidea}[A custom heading]` — the most obvious thing a writer can
type, and the standard LaTeX spelling for an optional title — is a *fatal* error
reading `I do not know the key '/tcb/A custom heading'`. The chapter is
discarded, and the critic is handed an error naming pgfkeys rather than the box
it came from. Fixed: every callout takes a title, in braces or brackets, with a
sensible default; raw key passthrough is gone, because nothing needed it and it
cost more than it was worth.

**Unnumbered entries read as sentences in the contents.** `\titlerule*` aligns
its dots to a grid, so an entry whose text ends near a grid point gets its first
dot flush against the last letter: the contents line read `Overview.` — a
heading with a full stop. Fixed with an explicit space before the leader.

**The title page printed a build machine's directory layout.** The provenance
line for a local source is the absolute path the run started from, which is
exactly right for the colophon and exactly wrong under the author on a title
page, where a long one wraps across three lines. Fixed: the title page shows the
folder or file names, the colophon keeps the full path.

Three of the six above and two of these four are the same shape as the ADK
defects catalogued earlier: the code ran, produced output, and was wrong in a way
nothing threw an exception about. The footer collision and the missing index are
the purest examples — neither is detectable by any test that does not build the
artefact and look at it. `tests/test_design.py` now does exactly that: it
compiles every callout spelling, every theme, and a document with indexed terms,
then reads the resulting PDF back as text and checks what is on the page.

---

## What is deliberately *not* claimed

- The metrics are heuristics, not proofs. Grounding checks that a number was
  *plausibly spoken*, not that the claim around it is true. It is a substantial
  improvement over a prompt sentence, and it is not a fact-checker.
- The semantic critic is a language model judging prose. Its findings adjust
  ranking modestly rather than dominating the score, because it is judgement
  rather than measurement.
- Without pdflatex the pipeline cannot verify by execution, and it says so —
  chapters are marked `written`, never `verified`.
