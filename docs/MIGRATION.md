# Migrating from the ADK version

Component-by-component mapping from
[`Yash-Kavaiya/adk-demo`](https://github.com/Yash-Kavaiya/adk-demo).

## Component map

| ADK | Here | Note |
|---|---|---|
| `agents/intake.py` `ChannelIntakeAgent` | `pipeline/intake.py:run_intake` | Plain function; also merges videos published since the last run |
| `agents/media.py` `MediaAcquisitionAgent` | `pipeline/media.py:acquire_media` | Returns parsed segments, not a text blob |
| `agents/analyst.py` `TranscriptAnalystAgent` | `dspy_runtime/modules/analyst.py` | Map-reduce over windows; grounding guard |
| — | `dspy_runtime/modules/captioner.py` | **New.** Fixes the dead `frame_captions` feature |
| `agents/assets.py` `VisualAssetAgent` | `pipeline/assets.py:build_assets` | Captions frames after curation |
| `agents/writer.py` `ChapterWriterAgent` | `dspy_runtime/modules/writer.py` | |
| `agents/critic.py` `ChapterCriticAgent` | `metrics/verifier.py` + `modules/critic.py` | Mechanical checks split from judgement |
| `agents/critic.py` `make_qa_loop` | `dspy_runtime/modules/chapter.py` | Best-candidate-wins, not last-candidate-wins |
| `agents/compiler.py` `BookCompilerAgent` | `pipeline/compile.py:compile_book` | Adds label de-duplication |
| `agents/orchestrator.py` `BookProductionAgent` | `pipeline/orchestrator.py:run_pipeline` | |
| `prompts.py` | `dspy_runtime/signatures.py` | Instructions are now optimizable parameters |
| `agents/model_resolver.py` | `dspy_runtime/lm.py` | `RobustNvidiaLlm` and `extract_clean_json` deleted — adapters handle it |
| `tools/workspace_lock.py` | folded into `workspace.py` | The `asyncio.Lock` was not thread-safe (ACCURACY §7) |
| `integrations/mlflow_integration.py` | `telemetry.py` | `mlflow.dspy.autolog()` replaces manual OTLP spans |
| `main.py` | `cli.py` | Subcommands: build, optimize, evaluate, artifacts, verify |

## Configuration

Environment variables keep the `BOOKFORGE_` prefix. Model names are now
**LiteLLM identifiers**, which is the one breaking change you must make:

```diff
- BOOKFORGE_ANALYST_MODEL=nvidia/nemotron-3-ultra-550b-a55b
+ BOOKFORGE_ANALYST_MODEL=openai/nvidia/nemotron-3-ultra-550b-a55b

- BOOKFORGE_ANALYST_MODEL=gemini-2.5-pro
+ BOOKFORGE_ANALYST_MODEL=gemini/gemini-2.5-pro
```

The `openai/` prefix plus `BOOKFORGE_OPENAI_API_BASE` is how any
OpenAI-compatible gateway is addressed — NVIDIA NIM, vLLM, Ollama, Together.

New settings worth knowing:

```env
BOOKFORGE_TRANSCRIPT_CHUNK_CHARS=24000     # windowing (replaces truncation)
BOOKFORGE_MAX_TRANSCRIPT_CHUNKS=12         # cost guard; widens rather than drops
BOOKFORGE_QA_ACCEPT_SCORE=0.92             # accept and stop refining
BOOKFORGE_QA_MIN_ACCEPTABLE_SCORE=0.55     # below this, the chapter fails
BOOKFORGE_MIN_GROUNDING_SCORE=0.6          # resample the analysis below this
BOOKFORGE_OPTIMIZER=gepa
BOOKFORGE_LOAD_OPTIMIZED=true
```

`BOOKFORGE_MAX_CONCURRENT_VIDEOS` and `BOOKFORGE_ENABLE_PARALLEL_VIDEOS` are
gone — they were documented as "kept for compatibility; the ADK graph is
sequential" and did nothing. `BOOKFORGE_CHAPTER_CONCURRENCY` replaces them.

## Workspace compatibility

The layout is unchanged, so **an existing `data/` directory resumes**:

```
data/<slug>/manifest.json
            videos/<id>/…
            chapters/<NN>_<slug>/…
            book/…
```

Two additions: `chapters/*/verdict.json` (the QA audit trail) and
`VideoRecord` gains `qa_score`, `qa_iterations`, `grounding_score` and
`compiled`. Pydantic defaults these, so old manifests load unchanged.

One caveat: chapters your old runs marked `verified` were verified by the old
critic, which could approve a chapter it had not really checked. To re-verify
them under the new rules:

```bash
# check one without changing anything
bookforge verify data/my-channel/chapters/01_intro/chapter.tex

# or force a full rebuild by resetting statuses
python - <<'PY'
from bookforge.workspace import Workspace
ws = Workspace("data", "my-channel")
for video in ws.load_manifest().videos:
    ws.update_video(video.video_id, status="analyzed")
PY
```

Media and analyses on disk are reused, so re-running costs only the writing
stages.

## What you lose

**`adk web` and `adk run`.** There is no ADK web UI. The CLI streams the same
progress, and MLflow tracing gives a richer view of prompts and completions
(`BOOKFORGE_ENABLE_MLFLOW=true`).

**`adk eval` evalsets.** `eval/*.evalset.json` were substring assertions against
agent chatter ("Response contains `Intake complete`"). They are replaced by 439
pytest tests that assert behaviour rather than log text, plus a real evaluation
harness (`bookforge evaluate`) that scores output quality numerically.

## What you gain

- Prompts you can optimize against a metric, and artifacts that persist the result
- A deterministic chapter score, and the guarantee that refinement never regresses
- Full transcript coverage on long videos
- Figure captions that describe the figure
- Programmatic hallucination detection at two boundaries
- Manifest writes that are actually thread-safe
- Per-role token accounting
- An offline test suite that runs in six seconds

## API changes

```python
# Before
from bookforge.agents.orchestrator import build_root_agent
runner = Runner(agent=build_root_agent(settings), ...)
async for event in runner.run_async(...): ...

# After
from bookforge import run_pipeline
result = run_pipeline("https://www.youtube.com/@channel")
print(result.summary())
```

For progress events, pass a `Reporter`:

```python
from bookforge.pipeline import Reporter, run_pipeline

reporter = Reporter([lambda event: print(event.kind, event.message)])
run_pipeline(url, settings, reporter)
```
