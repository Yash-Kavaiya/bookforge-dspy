# Optimization

## Why this is cheap here

Prompt optimization usually stalls on data: you need examples with known-good
outputs, and for a task like "write a textbook chapter" that means a human
writing gold-standard LaTeX. That costs more than the pipeline saves.

BookForge's metrics are **reference-free**. Grounding compares an analysis
against *its own transcript*. Structural checks compare a chapter against *its
own analysis and assets*. Compilation is checked by running pdflatex. None of
these need a reference answer — they need the inputs, which you already have.

So every video the pipeline processes becomes a training example, automatically,
and the dataset grows as the system runs.

```bash
bookforge build "https://www.youtube.com/@channel" --max-videos 6
bookforge optimize analyst --optimizer gepa --budget medium
bookforge build "https://www.youtube.com/@another-channel"   # uses the result
```

## Targets

| Target | Optimizes | Dataset | Metric |
|---|---|---|---|
| `analyst` | `AnalyzeTranscriptChunk`, `MergeChunkAnalyses`, `SingleShotAnalysis` | one example per transcript window | grounding + evidence linking + richness |
| `chapter` | `WriteChapter`, `RefineChapter` | one example per analysed chapter | compile + structure + grounding + polish |

The `chapter` target runs with the semantic critic **disabled**. Optimizing
against a stochastic LLM judge adds variance to the reward and cost to every
rollout; the deterministic verifier is reproducible, which is what an optimizer
needs. The critic is re-enabled for production runs.

## Choosing an optimizer

**`gepa` (default).** Reflective prompt evolution. It reads the *textual
feedback* our metrics emit and rewrites instructions to prevent that specific
failure. Our feedback is unusually good raw material — not "score 0.62" but:

> `[CRITICAL] These paths are referenced but are not in the assets manifest:`
> `chapters/03_x/figures/hallucinated.pdf. FIX: Remove every reference to a path`
> `that is not listed in the manifest.`

A reflection model can act on that. It cannot act on a number.

**`miprov2`.** Bayesian search over instructions *and* few-shot demonstrations.
Stronger when you have enough examples to bootstrap good demos (roughly 20+).
Also more expensive.

**`simba`.** Stochastic introspective mini-batch ascent. A reasonable middle
ground when GEPA's budget is too high.

**`bootstrap`.** Few-shot demonstrations via random search, no instruction
rewriting. Cheapest; use it as a floor to check whether optimization helps at
all before spending on GEPA.

```bash
bookforge optimize chapter --optimizer gepa --budget light
bookforge optimize chapter --optimizer miprov2 --budget medium
bookforge optimize analyst --optimizer bootstrap
```

## Budgets and cost

`--budget light|medium|heavy` maps to the optimizer's `auto` setting. Rough
guidance, and it depends heavily on your models:

| Budget | Metric calls | Wall clock | When |
|---|---|---|---|
| `light` | tens | minutes | first run, checking the plumbing |
| `medium` | low hundreds | tens of minutes | the usual choice |
| `heavy` | many hundreds | hours | a stable task worth the spend |

The `chapter` target compiles LaTeX on every rollout, so it is materially slower
than `analyst` and its cost is dominated by pdflatex rather than tokens. Start
with `--max-examples` to size the run:

```bash
bookforge optimize chapter --budget light --max-examples 8
```

Token usage is reported per role at the end of every run.

## The saving rule

**An optimizer run that does not beat the baseline is not saved.**

```python
result = compare(baseline_program, compiled, valset, scalar_metric, settings)
if not result.improved and not force_save:
    logger.warning("not saving: optimized %.4f did not beat baseline %.4f", ...)
    return result
```

Both programs are scored on the same validation examples with the same scalar
metric — the one production ships against, not the feedback variant used to
drive GEPA. An optimizer that makes things worse should leave the system exactly
as it was.

`--dry-run` measures without saving. `--force-save` overrides, for when you have
a reason.

## Artifacts

```bash
bookforge artifacts
```

```
Optimized programs in /work/artifacts:
  analyst          gepa      val=0.812 baseline=0.634 (+0.178)  sig=a3f9c21e08b74d55  2026-09-09T11:04:22
  chapter          miprov2   val=0.771 baseline=0.702 (+0.069)  sig=6b2e40d7c918aa31  2026-09-09T12:47:10
```

Each artifact records which optimizer produced it, its scores, and a
**signature fingerprint** — a hash of the program's predictor signatures. If you
add, rename or retype a signature field, the fingerprint changes and the stale
artifact is refused with an explanation rather than half-applied. Instructions
optimized for a field that no longer exists produce prompts referencing nothing.

```bash
bookforge build <url> --no-optimized   # ignore artifacts, use authored instructions
```

## Reading the result

Optimized instructions are stored as plain JSON:

```bash
python -c "import json; d=json.load(open('artifacts/analyst.json')); print(json.dumps(d, indent=2)[:2000])"
```

Reading what GEPA changed is worth the minute it takes. It usually reveals
something about the task you did not know — a failure mode you had not
considered, or an instruction that was ambiguous in a way you could not see.

## Measuring without optimizing

```bash
bookforge evaluate analyst                # baseline
bookforge evaluate analyst --optimized    # with the saved artifact
```

Useful for tracking whether an artifact still helps after the codebase or the
underlying model has moved.
