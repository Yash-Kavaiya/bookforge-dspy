"""BookForge command line.

    bookforge build "https://www.youtube.com/@channel" --max-videos 2
    bookforge optimize analyst --optimizer gepa --budget medium
    bookforge evaluate chapter
    bookforge artifacts
    bookforge verify data/<slug>/chapters/01_.../chapter.tex
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# The dspy/numpy lazy-import guard lives in bookforge/pipeline/__init__.py —
# importing bookforge.pipeline.progress below already goes through it.
from bookforge import telemetry
from bookforge.config import configure_logging, get_settings, preflight, reset_settings_cache
from bookforge.pipeline.progress import PipelineEvent, Reporter

logger = logging.getLogger("bookforge")

_PREFIX = {
    "warning": "!",
    "error": "x",
    "stage": ">",
    "done": "*",
}


def _console_listener(event: PipelineEvent) -> None:
    print(f"{_PREFIX.get(event.kind, ' ')} {event.message}", flush=True)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    from bookforge.dspy_runtime.program import BookForgeProgram
    from bookforge.pipeline.orchestrator import run_pipeline

    settings = get_settings()
    if args.max_videos is not None:
        settings.max_videos = args.max_videos
    if args.theme:
        settings.book_theme = args.theme
    if args.no_latex:
        settings.compile_latex = False
    if args.no_optimized:
        settings.load_optimized = False
    if args.qa_iterations is not None:
        settings.qa_max_iterations = args.qa_iterations

    for warning in preflight(settings):
        logger.warning(warning)
        if "pdflatex not found" in warning:
            # Verification by compilation is impossible; say so once, loudly,
            # rather than silently scoring chapters as if they had been checked.
            settings.compile_latex = False

    telemetry.setup_tracing(settings)
    reporter = Reporter([_console_listener])
    program = BookForgeProgram(settings)

    sources = args.source if len(args.source) > 1 else args.source[0]
    with telemetry.run("build", source=str(sources), max_videos=settings.max_videos):
        result = run_pipeline(sources, settings, reporter, program, title=args.title)
        telemetry.log_metrics(
            {
                "chapters_total": len(result.outcomes),
                "chapters_verified": result.verified,
                "chapters_failed": result.failed,
                "mean_quality": (result.book or {}).get("mean_quality_score", 0.0),
            }
        )

    print()
    print(result.summary())
    if result.error:
        print(f"ERROR: {result.error}", file=sys.stderr)
        return 1
    if result.book and result.book.get("pdf"):
        print(f"BOOK READY: {Path(result.workspace_root) / result.book['pdf']}")
        return 0
    print(
        "Run finished without a PDF (compilation disabled or every chapter failed).",
        file=sys.stderr,
    )
    return 2


def cmd_optimize(args: argparse.Namespace) -> int:
    from bookforge.evaluation.optimize import optimize

    settings = get_settings()
    if args.optimizer:
        settings.optimizer = args.optimizer
    if args.budget:
        settings.optimizer_budget = args.budget

    preflight(settings)
    telemetry.setup_tracing(settings)

    with telemetry.run(f"optimize-{args.target}", optimizer=settings.optimizer):
        result = optimize(
            target=args.target,
            optimizer_name=settings.optimizer,
            settings=settings,
            workspace_root=Path(args.workspace) if args.workspace else None,
            max_examples=args.max_examples,
            save=not args.dry_run,
            force_save=args.force_save,
        )
        telemetry.log_metrics(
            {
                "baseline_score": result.baseline.score,
                "optimized_score": result.candidate.score,
                "delta": result.delta,
            }
        )
    return 0 if result.improved else 3


def cmd_evaluate(args: argparse.Namespace) -> int:
    from bookforge.dspy_runtime.lm import LMRouter
    from bookforge.evaluation.datasets import collect, split
    from bookforge.evaluation.harness import evaluate_program
    from bookforge.evaluation.optimize import TARGETS, build_target_program
    from bookforge.metrics.composite import get_metric

    settings = get_settings()
    preflight(settings)
    spec = TARGETS[args.target]

    examples = collect(spec["dataset"], Path(args.workspace) if args.workspace else None, settings)
    _, valset = split(examples)
    if args.max_examples:
        valset = valset[: args.max_examples]

    router = LMRouter(settings)
    program = build_target_program(args.target, settings, router)
    if args.optimized:
        program.load(str(settings.artifacts_dir_abs / f"{spec['artifact']}.json"))

    summary = evaluate_program(
        program, valset, get_metric(spec["metric"]), label=args.target, settings=settings
    )
    print(summary)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the web UI and API."""
    try:
        from bookforge.api import serve
    except ImportError as exc:
        print(
            "The web UI needs FastAPI and uvicorn. Install them with:\n"
            "  pip install 'bookforge[ui]'",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    settings = get_settings()
    if args.theme:
        settings.book_theme = args.theme
    # Warn rather than refuse: the UI is useful for inspecting existing books
    # even with no credentials configured, and it reports its own readiness.
    if not settings.has_llm_credentials():
        logger.warning(
            "No LLM credentials configured — the UI will load but builds will "
            "be refused until you set BOOKFORGE_OPENAI_API_KEY or similar."
        )
    print(f"BookForge UI on http://{args.host}:{args.port}")
    serve(host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_artifacts(args: argparse.Namespace) -> int:
    from bookforge.dspy_runtime.artifacts import describe_artifacts

    print(describe_artifacts(get_settings().artifacts_dir_abs))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a chapter.tex already on disk — the same check the loop applies."""
    import json

    from bookforge.metrics.verifier import VerificationContext, verify_chapter
    from bookforge.schemas import AssetsManifest, ChapterAnalysis
    from bookforge.workspace import Workspace

    chapter_tex = Path(args.chapter).resolve()
    chapter_dir = chapter_tex.parent
    analysis_path = chapter_dir / "analysis.json"
    assets_path = chapter_dir / "assets.json"

    for path in (chapter_tex, analysis_path, assets_path):
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 1

    # chapters/<NN>_<slug>/ sits two levels below the workspace root.
    workspace_root = chapter_dir.parent.parent
    workspace = Workspace(workspace_root.parent, workspace_root.name)

    verdict = verify_chapter(
        chapter_tex.read_text(encoding="utf-8"),
        VerificationContext(
            analysis=ChapterAnalysis.model_validate_json(
                analysis_path.read_text(encoding="utf-8")
            ),
            manifest=AssetsManifest.model_validate_json(
                assets_path.read_text(encoding="utf-8")
            ),
            workspace=workspace,
            build_dir=workspace.build_dir / "verify_cli",
            settings=get_settings(),
        ),
    )
    if args.json:
        print(json.dumps(verdict.model_dump(), indent=2))
    else:
        print(f"score      : {verdict.score:.3f}")
        print(f"compiles   : {verdict.compiles}")
        print(f"acceptable : {verdict.is_acceptable}")
        print(f"defects    : {len(verdict.defects)}")
        if verdict.defects:
            print()
            print(verdict.feedback_text())
    return 0 if verdict.is_acceptable else 4


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bookforge",
        description="Turn a YouTube channel into a compiled LaTeX textbook (DSPy).",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    parser.add_argument("--log-json", action="store_true", help="structured JSON logs")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="produce a book from any source")
    build.add_argument(
        "source",
        nargs="+",
        help=(
            "YouTube channel/playlist/video URL, a video file, a transcript "
            "file, or a folder of either"
        ),
    )
    build.add_argument("--title", default="", help="book title (default: from the source)")
    build.add_argument(
        "--theme", choices=["textbook", "academic", "compact"], default=None
    )
    build.add_argument("--max-videos", type=int, default=None)
    build.add_argument("--qa-iterations", type=int, default=None,
                       help="max refinement iterations per chapter")
    build.add_argument("--no-latex", action="store_true",
                       help="skip pdflatex (draft mode; chapters are not compile-verified)")
    build.add_argument("--no-optimized", action="store_true",
                       help="ignore compiled artifacts and use authored instructions")
    build.set_defaults(func=cmd_build)

    optimize = sub.add_parser("optimize", help="compile better instructions")
    optimize.add_argument("target", choices=["analyst", "chapter"])
    optimize.add_argument("--optimizer", choices=["gepa", "miprov2", "simba", "bootstrap"])
    optimize.add_argument("--budget", choices=["light", "medium", "heavy"])
    optimize.add_argument("--workspace", default=None, help="workspace root to harvest")
    optimize.add_argument("--max-examples", type=int, default=None)
    optimize.add_argument("--dry-run", action="store_true", help="do not save the artifact")
    optimize.add_argument("--force-save", action="store_true",
                          help="save even if it did not beat the baseline")
    optimize.set_defaults(func=cmd_optimize)

    evaluate = sub.add_parser("evaluate", help="score a program without optimizing")
    evaluate.add_argument("target", choices=["analyst", "chapter"])
    evaluate.add_argument("--optimized", action="store_true", help="load the saved artifact")
    evaluate.add_argument("--workspace", default=None)
    evaluate.add_argument("--max-examples", type=int, default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    serve_cmd = sub.add_parser("serve", help="run the web UI and API")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    # Cloud Run (and most PaaS targets) inject $PORT and require the container
    # to listen on it; falling back to 8000 keeps local `bookforge serve` runs
    # unchanged.
    serve_cmd.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    serve_cmd.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    serve_cmd.add_argument("--theme", choices=["textbook", "academic", "compact"], default=None)
    serve_cmd.set_defaults(func=cmd_serve)

    artifacts = sub.add_parser("artifacts", help="list compiled programs")
    artifacts.set_defaults(func=cmd_artifacts)

    verify = sub.add_parser("verify", help="verify a chapter.tex on disk")
    verify.add_argument("chapter", help="path to chapter.tex")
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    reset_settings_cache()

    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(
        "DEBUG" if args.verbose else settings.log_level,
        json_format=args.log_json or settings.log_json,
    )
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is checkpointed — re-run to resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.exception("command failed")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cli_entry() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli_entry()
