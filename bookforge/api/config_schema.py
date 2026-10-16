"""Declarative schema for the web config page.

One list is the single source of truth for both the API (``GET /api/config``
folds live values into it) and the browser (which renders a form from it) —
the alternative is duplicating the field list in Python and JavaScript, which
drifts the first time someone adds a setting to :mod:`bookforge.config`.
"""

from __future__ import annotations

GROUPS: list[dict] = [
    {"key": "credentials", "title": "Provider & credentials"},
    {"key": "sampling", "title": "Sampling"},
    {"key": "quality", "title": "Quality loop"},
    {"key": "media", "title": "Media & transcript"},
    {"key": "concurrency", "title": "Concurrency"},
    {"key": "optimization", "title": "Optimization"},
]

FIELD_SCHEMA: list[dict] = [
    # --- credentials --------------------------------------------------------
    {
        "key": "openai_api_key", "group": "credentials", "type": "password",
        "label": "API key", "secret": True,
        "hint": "BOOKFORGE_OPENAI_API_KEY — an NVIDIA NIM / OpenAI-compatible key. "
                "For Gemini or Anthropic, use the provider keys below instead.",
    },
    {
        "key": "openai_api_base", "group": "credentials", "type": "text",
        "label": "API base URL",
        "hint": "Only used for openai/-prefixed models. Leave empty for real OpenAI.",
    },
    {
        "key": "analyst_model", "group": "credentials", "type": "text",
        "label": "Analyst model",
        "hint": "LiteLLM id, e.g. openai/gpt-4o, gemini/gemini-2.5-pro, anthropic/claude-sonnet-4-5",
    },
    {"key": "writer_model", "group": "credentials", "type": "text", "label": "Writer model"},
    {"key": "critic_model", "group": "credentials", "type": "text", "label": "Critic model"},
    {"key": "caption_model", "group": "credentials", "type": "text", "label": "Caption model"},
    {
        "key": "reflection_model", "group": "credentials", "type": "text",
        "label": "Reflection model",
        "hint": "Used by GEPA to reflect on failures and rewrite instructions.",
    },
    # --- sampling ------------------------------------------------------------
    {
        "key": "analyst_temperature", "group": "sampling", "type": "number",
        "label": "Analyst temperature", "min": 0, "max": 2, "step": 0.05,
    },
    {
        "key": "writer_temperature", "group": "sampling", "type": "number",
        "label": "Writer temperature", "min": 0, "max": 2, "step": 0.05,
    },
    {
        "key": "critic_temperature", "group": "sampling", "type": "number",
        "label": "Critic temperature", "min": 0, "max": 2, "step": 0.05,
    },
    {
        "key": "max_tokens", "group": "sampling", "type": "number",
        "label": "Max tokens", "min": 256, "max": 131072, "step": 256,
    },
    {
        "key": "lm_num_retries", "group": "sampling", "type": "number",
        "label": "Retries", "min": 0, "max": 10, "step": 1,
    },
    {
        "key": "lm_timeout_sec", "group": "sampling", "type": "number",
        "label": "Timeout (sec)", "min": 10, "max": 1800, "step": 5,
    },
    {"key": "lm_cache", "group": "sampling", "type": "checkbox", "label": "Cache identical calls"},
    # --- quality loop --------------------------------------------------------
    {
        "key": "qa_max_iterations", "group": "quality", "type": "number",
        "label": "Max QA iterations", "min": 1, "max": 8, "step": 1,
    },
    {
        "key": "qa_accept_score", "group": "quality", "type": "number",
        "label": "Accept score", "min": 0, "max": 1, "step": 0.01,
        "hint": "Composite score at or above which a chapter ships immediately.",
    },
    {
        "key": "qa_min_acceptable_score", "group": "quality", "type": "number",
        "label": "Min acceptable score", "min": 0, "max": 1, "step": 0.01,
        "hint": "Below this the chapter is marked failed rather than shipped.",
    },
    {
        "key": "min_grounding_score", "group": "quality", "type": "number",
        "label": "Min grounding score", "min": 0, "max": 1, "step": 0.01,
        "hint": "Resample the analysis once if grounding falls below this.",
    },
    {
        "key": "writer_refine_attempts", "group": "quality", "type": "number",
        "label": "Writer refine attempts", "min": 0, "max": 5, "step": 1,
    },
    {
        "key": "analyst_consistency_samples", "group": "quality", "type": "number",
        "label": "Analyst consistency samples", "min": 1, "max": 5, "step": 1,
    },
    {"key": "compile_latex", "group": "quality", "type": "checkbox", "label": "Compile with pdflatex"},
    {
        "key": "latex_timeout_sec", "group": "quality", "type": "number",
        "label": "LaTeX timeout (sec)", "min": 10, "max": 900, "step": 5,
    },
    # --- media & transcript ---------------------------------------------------
    {
        "key": "frame_interval_sec", "group": "media", "type": "number",
        "label": "Frame interval (sec)", "min": 1, "max": 60, "step": 1,
    },
    {
        "key": "frame_scene_threshold", "group": "media", "type": "number",
        "label": "Scene-change threshold", "min": 0, "max": 1, "step": 0.01,
    },
    {
        "key": "frame_phash_threshold", "group": "media", "type": "number",
        "label": "pHash de-dupe threshold", "min": 0, "max": 20, "step": 1,
    },
    {
        "key": "max_frames_per_chapter", "group": "media", "type": "number",
        "label": "Max frames / chapter", "min": 1, "max": 50, "step": 1,
    },
    {
        "key": "download_resolution", "group": "media", "type": "select",
        "label": "Download resolution", "options": [240, 360, 480, 720, 1080],
    },
    {
        "key": "whisper_model", "group": "media", "type": "select",
        "label": "Whisper model", "options": ["tiny", "base", "small", "medium", "large"],
    },
    {
        "key": "transcript_chunk_chars", "group": "media", "type": "number",
        "label": "Transcript chunk size (chars)", "min": 1000, "max": 100000, "step": 500,
    },
    {
        "key": "transcript_chunk_overlap_chars", "group": "media", "type": "number",
        "label": "Chunk overlap (chars)", "min": 0, "max": 10000, "step": 100,
    },
    {
        "key": "max_transcript_chunks", "group": "media", "type": "number",
        "label": "Max chunks / video", "min": 0, "max": 50, "step": 1,
        "hint": "0 = unlimited.",
    },
    # --- concurrency -----------------------------------------------------------
    {
        "key": "media_prefetch_concurrency", "group": "concurrency", "type": "number",
        "label": "Media prefetch workers", "min": 1, "max": 16, "step": 1,
    },
    {
        "key": "chapter_concurrency", "group": "concurrency", "type": "number",
        "label": "Chapters in parallel", "min": 1, "max": 4, "step": 1,
    },
    {
        "key": "num_threads", "group": "concurrency", "type": "number",
        "label": "DSPy worker threads", "min": 1, "max": 32, "step": 1,
    },
    {
        "key": "analyst_concurrency", "group": "concurrency", "type": "number",
        "label": "Transcript windows analysed in parallel", "min": 1, "max": 16, "step": 1,
    },
    # --- optimization ------------------------------------------------------
    {
        "key": "optimizer", "group": "optimization", "type": "select",
        "label": "Optimizer", "options": ["gepa", "miprov2", "bootstrap", "simba", "none"],
    },
    {
        "key": "optimizer_budget", "group": "optimization", "type": "select",
        "label": "Budget", "options": ["light", "medium", "heavy"],
    },
    {
        "key": "load_optimized", "group": "optimization", "type": "checkbox",
        "label": "Load optimized programs when present",
    },
]

#: Plain process env vars LiteLLM reads directly — alternative providers, per
#: the README — rather than BOOKFORGE_-prefixed Settings fields.
PROVIDER_ENV_KEYS = ["GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY"]

#: Roles the "test connection" button can exercise, mirroring
#: bookforge.dspy_runtime.lm._ROLE_MODEL.
TEST_ROLES = ["analyst", "writer", "critic", "caption", "reflection"]
