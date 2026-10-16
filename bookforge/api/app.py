"""FastAPI application: build books from the browser.

Endpoints are deliberately few. The UI needs to start a build, watch it, inspect
what the verifier thought of each chapter, and download the PDF — everything
else the CLI already does better.

Security note: this serves files out of the workspace, so every path the client
supplies is resolved and checked to be inside the workspace root before it is
opened. A book pipeline that will happily read ``../../etc/passwd`` because a
filename came from a URL is a real hazard, not a theoretical one.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from contextlib import suppress
from pathlib import Path
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from bookforge import __version__
from bookforge.api import env_file
from bookforge.api.config_schema import FIELD_SCHEMA, GROUPS, PROVIDER_ENV_KEYS, TEST_ROLES
from bookforge.api.jobs import Job, get_runner
from bookforge.config import Settings, get_settings, preflight, reset_settings_cache
from bookforge.design import THEMES
from bookforge.dspy_runtime.lm import build_lm
from bookforge.ingest import resolve_source
from bookforge.workspace import Workspace

logger = logging.getLogger(__name__)

#: Angular's compiled output — see frontend/, built by `npm run build`
#: (frontend/angular.json points outputPath here).
WEBAPP_DIR = Path(__file__).parent / "webapp" / "browser"

#: Uploads land here, one directory per batch, and are then ingested like any
#: other local folder.
UPLOAD_SUBDIR = "uploads"


def _remove_upload_batch(batch: Path) -> None:
    """Delete a partial upload batch, including on Windows with sticky files."""
    if not batch.exists():
        return
    # Unlink files first so a lingering handle is less likely to strand the tree.
    for path in sorted(batch.rglob("*"), reverse=True):
        try:
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        except OSError:
            pass
    shutil.rmtree(batch, ignore_errors=True)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class BuildRequest(BaseModel):
    source: list[str] = Field(
        min_length=1,
        description="YouTube URL(s), file paths, or folder paths on the server",
    )
    title: str = ""
    theme: str = "textbook"
    max_videos: int | None = None
    compile_latex: bool | None = None
    qa_iterations: int | None = None


class PreviewRequest(BaseModel):
    source: list[str] = Field(min_length=1)
    title: str = ""


class ConfigUpdateRequest(BaseModel):
    #: BOOKFORGE_-prefixed Settings fields, by their Python name (e.g. "max_tokens").
    settings: dict[str, object] = Field(default_factory=dict)
    #: Raw provider env vars (GOOGLE_API_KEY, ...) — not Settings fields.
    env: dict[str, str] = Field(default_factory=dict)


class ConfigTestRequest(BaseModel):
    role: str = "analyst"
    model: str | None = None
    api_key: str | None = None
    api_base: str | None = None


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "•" * len(value)
    return "•" * (len(value) - 4) + value[-4:]


def _env_str(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in error['loc'])}: {error['msg']}" for error in exc.errors()
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_app(settings=None, env_path: Path | None = None) -> FastAPI:
    settings = settings or get_settings()
    #: Where PUT /api/config persists changes. Defaults to the same ".env" the
    #: process itself was started with (relative to the working directory).
    env_path = env_path or Path(".env")

    app = FastAPI(
        title="BookForge",
        version=__version__,
        description="Turn recorded talks into a compiled, verified LaTeX textbook.",
    )
    # Permissive CORS: this is a local single-tenant tool. Put it behind a proxy
    # with real auth before exposing it to a network you do not control.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # The Angular bundle and JSON API responses (event/chapter lists can run to
    # hundreds of KB on a long build) compress well; this cuts UI load time and
    # polling payload size for free.
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    upload_root = settings.workspace_root_abs / UPLOAD_SUBDIR
    upload_root.mkdir(parents=True, exist_ok=True)

    # -- helpers ------------------------------------------------------------

    def _job_or_404(job_id: str) -> Job:
        job = get_runner(settings).get(job_id)
        if job is None:
            raise HTTPException(404, f"No build with id {job_id}")
        return job

    def _safe_workspace_path(slug: str, relative: str) -> Path:
        """Resolve a workspace-relative path, refusing anything outside it."""
        root = (settings.workspace_root_abs / slug).resolve()
        if not root.is_dir():
            raise HTTPException(404, f"No workspace for {slug}")
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise HTTPException(403, "Path escapes the workspace")
        if not target.exists():
            raise HTTPException(404, f"{relative} not found")
        return target

    # -- meta ---------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:
        """What the server can actually do right now.

        The UI shows this on load: a missing pdflatex or ffmpeg changes what a
        build will produce, and finding that out at the end of an hour-long run
        is the worst time to learn it.
        """
        warnings: list[str] = []
        credentials_ok = settings.has_llm_credentials()
        if credentials_ok:
            try:
                warnings = preflight(settings)
            except RuntimeError as exc:
                credentials_ok = False
                warnings = [str(exc)]
        else:
            warnings = [
                "No LLM credentials configured. Set BOOKFORGE_OPENAI_API_KEY, "
                "GOOGLE_API_KEY or ANTHROPIC_API_KEY before building."
            ]

        return {
            "version": __version__,
            "ready": credentials_ok,
            "warnings": warnings,
            "tools": {
                "pdflatex": bool(shutil.which("pdflatex")),
                "ffmpeg": bool(shutil.which("ffmpeg")),
            },
            "themes": sorted(THEMES),
            "settings": {
                "theme": settings.book_theme,
                "workspace_root": str(settings.workspace_root_abs),
                "compile_latex": settings.compile_latex,
                "qa_max_iterations": settings.qa_max_iterations,
                "video_extensions": list(settings.video_extensions),
                "transcript_extensions": list(settings.transcript_extensions),
                "youtube_cookies_configured": bool(
                    settings.youtube_cookies_file or settings.youtube_cookies_from_browser
                ),
            },
            "models": {
                "analyst": settings.analyst_model,
                "writer": settings.writer_model,
                "critic": settings.critic_model,
                "caption": settings.caption_model,
                "reflection": settings.reflection_model,
            },
            "concurrency": {
                # Chapters are produced one at a time (LLM stages are strictly
                # sequential); only media download/transcription for upcoming
                # chapters overlaps with the current chapter's LLM work.
                "chapters_parallel": 1,
                "media_prefetch_workers": settings.media_prefetch_concurrency,
            },
        }

    # -- config ---------------------------------------------------------------

    @app.get("/api/config")
    def get_config() -> dict:
        """Every editable hyperparameter and credential, with live values.

        One schema drives both this response and the browser form — see
        ``bookforge.api.config_schema`` for why.
        """
        fields = []
        for spec in FIELD_SCHEMA:
            value = getattr(settings, spec["key"])
            item = dict(spec)
            if item.get("secret"):
                item["value"] = _mask_secret(value)
                item["is_set"] = bool(value)
            else:
                item["value"] = value
            fields.append(item)
        return {
            "fields": fields,
            "groups": GROUPS,
            "provider_env": {name: bool(os.environ.get(name)) for name in PROVIDER_ENV_KEYS},
            "env_file": str(env_path.resolve()) if env_path.exists() else None,
        }

    @app.put("/api/config")
    def update_config(request: ConfigUpdateRequest) -> dict:
        """Validate, apply to the live settings, and persist to .env.

        Applied in place on the *same* Settings object every closure in this
        app already holds — Settings.model_config sets validate_assignment for
        exactly this, so a saved change takes effect immediately, no restart.
        """
        known = {spec["key"] for spec in FIELD_SCHEMA}
        unknown = set(request.settings) - known
        if unknown:
            raise HTTPException(400, f"Unknown setting(s): {', '.join(sorted(unknown))}")
        unknown_env = set(request.env) - set(PROVIDER_ENV_KEYS)
        if unknown_env:
            raise HTTPException(400, f"Unknown env var(s): {', '.join(sorted(unknown_env))}")

        updates = request.settings
        if not updates and not request.env:
            return {"applied": [], "env_updated": []}

        # Validate the whole merged state at once so a change that only makes
        # sense together with another (e.g. the chunk/overlap pair) can't be
        # applied half-finished.
        merged = settings.model_dump()
        merged.update(updates)
        try:
            validated = Settings.model_validate(merged)
        except ValidationError as exc:
            raise HTTPException(422, _format_validation_error(exc)) from exc

        env_updates: dict[str, str] = {}
        for key in updates:
            new_value = getattr(validated, key)
            object.__setattr__(settings, key, new_value)
            env_key = f"BOOKFORGE_{key.upper()}"
            rendered = _env_str(new_value)
            os.environ[env_key] = rendered
            env_updates[env_key] = rendered

        for name, value in request.env.items():
            if value:
                os.environ[name] = value
                env_updates[name] = value

        env_file.upsert(env_path, env_updates)
        reset_settings_cache()

        return {"applied": sorted(updates), "env_updated": sorted(env_updates)}

    @app.post("/api/config/test")
    def test_config(request: ConfigTestRequest) -> dict:
        """Fire one cheap real call so a bad key/model/base fails now, not an hour in."""
        if request.role not in TEST_ROLES:
            raise HTTPException(400, f"Unknown role {request.role!r}; expected one of {TEST_ROLES}")

        trial = settings.model_copy()
        if request.model:
            setattr(trial, f"{request.role}_model", request.model)
        if request.api_key:
            trial.openai_api_key = request.api_key
        if request.api_base:
            trial.openai_api_base = request.api_base

        model_name = getattr(trial, f"{request.role}_model")
        started = time.monotonic()
        try:
            lm = build_lm(request.role, trial, cache=False, num_retries=0, timeout=30, max_tokens=16)
            reply = lm(prompt="Reply with exactly one word: OK")
            latency_ms = round((time.monotonic() - started) * 1000)
            text = reply[0] if isinstance(reply, list) and reply else str(reply)
            return {"ok": True, "model": model_name, "latency_ms": latency_ms, "reply": str(text)[:200]}
        except Exception as exc:
            latency_ms = round((time.monotonic() - started) * 1000)
            return {"ok": False, "model": model_name, "latency_ms": latency_ms, "error": str(exc)[:500]}

    # -- ingest preview -----------------------------------------------------

    @app.post("/api/preview")
    def preview(request: PreviewRequest) -> dict:
        """What would this source produce? Cheap: metadata only, no downloads.

        Worth its own endpoint because the alternative is committing to a long
        run and finding out afterwards that a folder held one usable file.
        """
        try:
            source = resolve_source(list(request.source), settings, title=request.title)
            discovery = source.discover()
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

        ordered = sorted(discovery.items, key=lambda item: item.order_key)
        return {
            "title": discovery.title,
            "source_line": discovery.source_line,
            "warnings": discovery.warnings,
            "count": len(ordered),
            "items": [
                {
                    "video_id": item.video_id,
                    "title": item.title,
                    "source": item.source,
                    "duration_sec": item.duration_sec,
                    "path": item.source_path,
                    "url": item.url,
                }
                for item in ordered
            ],
        }

    # -- uploads ------------------------------------------------------------

    @app.post("/api/upload")
    async def upload(files: Annotated[list[UploadFile], File()]) -> dict:
        """Stage uploaded videos/transcripts into a batch folder on the server.

        Returns the folder path, which the client then submits as a build
        source — so an upload and a server-side folder take the same code path
        from here on.
        """
        import uuid

        if not files:
            raise HTTPException(400, "No files were uploaded")

        batch = upload_root / f"batch-{uuid.uuid4().hex[:8]}"
        batch.mkdir(parents=True, exist_ok=True)

        allowed = {
            ext.lower()
            for ext in (*settings.video_extensions, *settings.transcript_extensions)
        }
        saved: list[dict] = []
        rejected: list[str] = []
        budget = settings.upload_max_mb * 1024 * 1024
        written = 0

        for upload_file in files:
            name = Path(upload_file.filename or "unnamed").name  # strip any path
            if Path(name).suffix.lower() not in allowed:
                rejected.append(name)
                continue
            destination = batch / name
            size = 0
            over_budget = False
            # Close the file *before* deleting the batch. On Windows an open
            # handle makes shutil.rmtree leave the partial upload behind even
            # with ignore_errors=True — exactly the disk-fill failure mode this
            # ceiling exists to prevent.
            with destination.open("wb") as handle:
                while chunk := await upload_file.read(1024 * 1024):
                    handle.write(chunk)
                    size += len(chunk)
                    written += len(chunk)
                    if written > budget:
                        over_budget = True
                        break
            if over_budget:
                _remove_upload_batch(batch)
                raise HTTPException(
                    413,
                    f"Upload exceeds the {settings.upload_max_mb} MB limit "
                    "for one batch. Raise BOOKFORGE_UPLOAD_MAX_MB, or "
                    "point the build at a folder on the server instead "
                    "of uploading it.",
                )
            saved.append({"name": name, "bytes": size})

        if not saved:
            _remove_upload_batch(batch)
            raise HTTPException(
                400,
                "None of the uploaded files is a supported video or transcript. "
                f"Accepted: {', '.join(sorted(allowed))}",
            )

        return {
            "path": str(batch),
            "saved": saved,
            "rejected": rejected,
            "total_bytes": sum(item["bytes"] for item in saved),
        }

    # -- builds -------------------------------------------------------------

    @app.post("/api/builds")
    def start_build(request: BuildRequest) -> dict:
        if not settings.has_llm_credentials():
            raise HTTPException(
                400,
                "No LLM credentials configured. Set BOOKFORGE_OPENAI_API_KEY, "
                "GOOGLE_API_KEY or ANTHROPIC_API_KEY and restart the server.",
            )
        if request.theme not in THEMES:
            raise HTTPException(400, f"Unknown theme {request.theme!r}")

        # Fail fast on an unusable source rather than inside a background thread
        # where the user only learns about it from the event log.
        try:
            resolve_source(list(request.source), settings, title=request.title)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

        try:
            job = get_runner(settings).submit(
                source=request.source,
                title=request.title,
                theme=request.theme,
                max_videos=request.max_videos,
                compile_latex=request.compile_latex,
                qa_iterations=request.qa_iterations,
            )
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return job.to_dict()

    @app.get("/api/builds")
    def list_builds() -> dict:
        return {"builds": [job.to_dict() for job in get_runner(settings).list()]}

    @app.get("/api/builds/{job_id}")
    def get_build(job_id: str, events: bool = False) -> dict:
        return _job_or_404(job_id).to_dict(include_events=events)

    @app.get("/api/builds/{job_id}/events")
    def get_events(job_id: str, after: int = Query(0, ge=0)) -> dict:
        """Polling fallback for clients without WebSockets."""
        job = _job_or_404(job_id)
        return {
            "status": job.status,
            "events": [event.to_dict() for event in job.events_since(after)],
            "last_seq": job.to_dict()["last_seq"],
        }

    @app.post("/api/builds/{job_id}/cancel")
    def cancel_build(job_id: str) -> dict:
        job = _job_or_404(job_id)
        if not job.cancel():
            raise HTTPException(409, f"Build is already {job.status}")
        return job.to_dict()

    @app.websocket("/api/builds/{job_id}/stream")
    async def stream(websocket: WebSocket, job_id: str) -> None:
        """Live event stream, replayed from the start on connect.

        Replay matters: a page refresh or a dropped connection must not lose
        the run's history, and the buffer already holds it.
        """
        await websocket.accept()
        job = get_runner(settings).get(job_id)
        if job is None:
            await websocket.send_json({"type": "error", "message": "unknown build"})
            await websocket.close()
            return

        cursor = 0
        try:
            while True:
                events = job.events_since(cursor)
                for event in events:
                    await websocket.send_json({"type": "event", **event.to_dict()})
                    cursor = event.seq
                if job.is_finished and not job.events_since(cursor):
                    await websocket.send_json({"type": "status", **job.to_dict()})
                    break
                # Block off the event loop in a worker thread; the job's
                # condition wakes it the moment an event arrives.
                await asyncio.to_thread(job.wait_for_event, cursor, 15.0)
        except WebSocketDisconnect:
            return
        except Exception:
            logger.debug("stream for %s ended", job_id, exc_info=True)
        finally:
            # Closing an already-closed socket raises; the client disconnecting
            # first is the normal case, not an error.
            with suppress(RuntimeError):
                await websocket.close()

    # -- workspace inspection ----------------------------------------------

    @app.get("/api/workspaces")
    def list_workspaces() -> dict:
        root = settings.workspace_root_abs
        found = []
        if root.exists():
            for candidate in sorted(root.iterdir()):
                if not (candidate / "manifest.json").exists():
                    continue
                try:
                    workspace = Workspace(root, candidate.name)
                    manifest = workspace.load_manifest()
                except Exception:
                    continue
                found.append(
                    {
                        "slug": candidate.name,
                        "title": manifest.channel_title,
                        "source": manifest.channel_url,
                        "created_at": manifest.created_at,
                        "chapters": len(manifest.videos),
                        "verified": sum(
                            1 for v in manifest.videos if v.status == "verified"
                        ),
                        "has_pdf": (workspace.book_dir / "book.pdf").exists(),
                    }
                )
        return {"workspaces": found}

    @app.get("/api/workspaces/{slug}/chapters")
    def workspace_chapters(slug: str) -> dict:
        """Per-chapter quality, straight from the manifest and verdicts."""
        import json

        workspace = Workspace(settings.workspace_root_abs, slug)
        if not workspace.manifest_exists():
            raise HTTPException(404, f"No workspace for {slug}")
        manifest = workspace.load_manifest()

        chapters = []
        for record in manifest.ordered():
            chapter_dir = workspace.chapter_dir(record)
            verdict = None
            verdict_path = chapter_dir / "verdict.json"
            if verdict_path.exists():
                try:
                    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
                except Exception:
                    verdict = None
            chapters.append(
                {
                    "video_id": record.video_id,
                    "chapter_number": record.chapter_number,
                    "title": record.title,
                    "status": record.status,
                    "source": record.source,
                    "qa_score": record.qa_score,
                    "grounding_score": record.grounding_score,
                    "iterations": record.qa_iterations,
                    "compiled": record.compiled,
                    "error": record.error,
                    "verdict": verdict,
                }
            )
        return {
            "slug": slug,
            "title": manifest.channel_title,
            "chapters": chapters,
        }

    @app.get("/api/workspaces/{slug}/book")
    def download_book(slug: str, inline: bool = True):
        path = _safe_workspace_path(slug, "book/book.pdf")
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=f"{slug}.pdf",
            content_disposition_type="inline" if inline else "attachment",
        )

    @app.get("/api/workspaces/{slug}/file")
    def workspace_file(slug: str, path: str = Query(..., min_length=1)):
        target = _safe_workspace_path(slug, path)
        if target.is_dir():
            raise HTTPException(400, "That path is a directory")
        return FileResponse(target)

    # -- UI -------------------------------------------------------------------
    # The UI is an Angular SPA (see frontend/) built into WEBAPP_DIR by
    # `npm run build`. Angular Router handles navigation client-side, so any
    # path that isn't an API route or a real asset file falls back to
    # index.html and the router takes it from there.

    @app.get("/{full_path:path}", response_model=None)
    def spa(full_path: str) -> FileResponse | HTMLResponse:
        index = WEBAPP_DIR / "index.html"
        if not index.exists():
            return HTMLResponse(
                "<h1>BookForge</h1><p>The UI is not built. Run <code>npm run build</code> "
                "in frontend/. The API is at <a href='/docs'>/docs</a>.</p>",
                status_code=200,
            )
        if full_path:
            candidate = (WEBAPP_DIR / full_path).resolve()
            if candidate.is_relative_to(WEBAPP_DIR.resolve()) and candidate.is_file():
                # Angular's production build content-hashes its JS/CSS
                # filenames, so any file that isn't index.html itself can be
                # cached indefinitely — a rebuild changes the filename, not
                # the content behind an old one.
                return FileResponse(
                    candidate,
                    headers={"Cache-Control": "public, max-age=31536000, immutable"},
                )
        # index.html references the current build's hashed filenames, so it
        # must always be revalidated rather than cached.
        return HTMLResponse(
            index.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-cache"},
        )

    @app.exception_handler(ValueError)
    def value_error_handler(request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


app = None


def get_app() -> FastAPI:
    """Lazily built ASGI app, so importing this module stays cheap."""
    global app
    if app is None:
        app = create_app()
    return app


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    import uvicorn

    uvicorn.run(
        "bookforge.api.app:get_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        log_level="info",
    )
