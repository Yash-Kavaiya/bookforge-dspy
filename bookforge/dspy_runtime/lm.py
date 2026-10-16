"""Language-model wiring: per-role LMs, caching, retries and cost accounting.

Every stage gets its own LM instance so temperature and model can be tuned per
role — the analyst runs cold because extraction should be reproducible, the
writer runs warmer because prose benefits from variety, and the critic runs at
zero because a judge that changes its mind is not a judge.

The ADK implementation needed a bespoke ``RobustNvidiaLlm`` subclass to
sanitize malformed JSON coming back from NIM endpoints. DSPy's adapters own
that responsibility, so provider quirks are a configuration concern here rather
than a class hierarchy.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import dspy
from dspy.utils.callback import BaseCallback

from bookforge.config import Settings, get_settings

logger = logging.getLogger(__name__)

Role = str  # "analyst" | "writer" | "critic" | "caption" | "reflection"


# ---------------------------------------------------------------------------
# Cost / usage accounting
# ---------------------------------------------------------------------------


@dataclass
class UsageRecord:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failures: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class UsageTracker:
    """Thread-safe token accounting, reported per role at the end of a run.

    The original pipeline had no visibility into what a book cost. Optimizer
    runs make many more calls than production runs, so knowing which stage
    dominates is what makes a budget decision possible.
    """

    by_role: dict[str, UsageRecord] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        role: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        failed: bool = False,
        count_call: bool = True,
    ) -> None:
        with self._lock:
            entry = self.by_role.setdefault(role, UsageRecord())
            if count_call:
                entry.calls += 1
            entry.prompt_tokens += prompt_tokens
            entry.completion_tokens += completion_tokens
            if failed:
                entry.failures += 1

    def totals(self) -> UsageRecord:
        with self._lock:
            total = UsageRecord()
            for entry in self.by_role.values():
                total.calls += entry.calls
                total.prompt_tokens += entry.prompt_tokens
                total.completion_tokens += entry.completion_tokens
                total.failures += entry.failures
            return total

    def summary(self) -> str:
        with self._lock:
            if not self.by_role:
                return "No LLM calls recorded."
            rows = [
                f"  {role:<12} {rec.calls:>4} calls  "
                f"{rec.prompt_tokens:>9,} in  {rec.completion_tokens:>9,} out"
                + (f"  ({rec.failures} failed)" if rec.failures else "")
                for role, rec in sorted(self.by_role.items())
            ]
        total = self.totals()
        rows.append(
            f"  {'TOTAL':<12} {total.calls:>4} calls  "
            f"{total.prompt_tokens:>9,} in  {total.completion_tokens:>9,} out"
        )
        return "LLM usage:\n" + "\n".join(rows)

    def reset(self) -> None:
        with self._lock:
            self.by_role.clear()

    def to_dict(self) -> dict:
        """JSON-friendly snapshot for the web UI's live usage panel."""
        with self._lock:
            by_role = {
                role: {
                    "calls": rec.calls,
                    "prompt_tokens": rec.prompt_tokens,
                    "completion_tokens": rec.completion_tokens,
                    "failures": rec.failures,
                }
                for role, rec in sorted(self.by_role.items())
            }
        total = self.totals()
        return {
            "by_role": by_role,
            "total": {
                "calls": total.calls,
                "prompt_tokens": total.prompt_tokens,
                "completion_tokens": total.completion_tokens,
                "failures": total.failures,
            },
        }


USAGE = UsageTracker()


class _UsageCallback(BaseCallback):
    """Attaches token counts from each LM response to the tracker."""

    def __init__(self, role: str) -> None:
        self.role = role

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: BaseException | None = None,
    ) -> None:
        if exception is not None:
            USAGE.record(self.role, failed=True)
            return
        prompt_tokens, completion_tokens = _read_usage(outputs)
        USAGE.record(self.role, prompt_tokens, completion_tokens)


def _field(source: Any, name: str) -> int:
    """Read a usage counter from either a dict or an SDK object."""
    value = source.get(name) if isinstance(source, dict) else getattr(source, name, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _read_usage(outputs: dict[str, Any] | None) -> tuple[int, int]:
    """Best-effort token counts. Accounting must never break a run."""
    try:
        usage: Any = (outputs or {}).get("usage") or {}
        if not usage:
            # LiteLLM sometimes tucks usage inside the raw response objects.
            for item in (outputs or {}).get("response", []) or []:
                usage = getattr(item, "usage", None) or {}
                if usage:
                    break
        if not usage:
            return 0, 0
        return _field(usage, "prompt_tokens"), _field(usage, "completion_tokens")
    except Exception:
        logger.debug("could not read LM usage", exc_info=True)
        return 0, 0


# ---------------------------------------------------------------------------
# LM construction
# ---------------------------------------------------------------------------


_REASONING_FAMILY_RE = re.compile(r"^(?:o\d+(?:-.*)?|gpt-(?:[5-9]|\d{2,})(?:[.\-].*)?)$")

#: Reasoning tokens are billed against the completion budget, so a chapter that
#: fits in 16k tokens of prose can still be cut off once the model has thought.
_REASONING_MIN_COMPLETION_TOKENS = 32_000


def is_reasoning_model(model: str) -> bool:
    """OpenAI reasoning models: o-series and gpt-5 onward, excluding chat aliases.

    They reject ``max_tokens`` (``max_completion_tokens`` only) and any
    temperature other than the default. DSPy's own check stops at gpt-5, so
    gpt-6 would otherwise be sent both and every call would 400.
    """
    family = model.split("/")[-1].lower()
    return "-chat" not in family and _REASONING_FAMILY_RE.match(family) is not None


class BookForgeLM(dspy.LM):
    """``dspy.LM`` with two BookForge concerns attached.

    * It never sends a temperature to a reasoning model. Per-call and
      ``copy()`` overrides (the resample paths use ``temperature=0.7`` to
      escape the cache) would otherwise reach the provider and be rejected.
      ``rollout_id`` still separates cache entries.
    * It records token usage from the raw provider response. DSPy's
      ``on_lm_end`` callback only receives the completion text, so usage read
      there was always zero.
    """

    #: Usage role; set by ``build_lm``. Not an LM kwarg, so never sent upstream.
    bookforge_role: str = ""

    def _sanitize(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if is_reasoning_model(self.model):
            kwargs = {**kwargs, "temperature": None}
        return kwargs

    def _record_usage(self, results: Any) -> Any:
        if self.bookforge_role:
            prompt_tokens, completion_tokens = _read_usage({"usage": _response_usage(results)})
            if prompt_tokens or completion_tokens:
                USAGE.record(
                    self.bookforge_role, prompt_tokens, completion_tokens, count_call=False
                )
        return results

    def forward(self, prompt=None, messages=None, **kwargs):
        return self._record_usage(
            super().forward(prompt=prompt, messages=messages, **self._sanitize(kwargs))
        )

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return self._record_usage(
            await super().aforward(prompt=prompt, messages=messages, **self._sanitize(kwargs))
        )


def _response_usage(results: Any) -> Any:
    if isinstance(results, dict):
        return results.get("usage") or {}
    return getattr(results, "usage", None) or {}


def _provider_kwargs(settings: Settings, model: str) -> dict[str, Any]:
    """Credentials and endpoint for a LiteLLM model id.

    An ``openai/`` prefixed model with a configured base URL is how NVIDIA NIM,
    vLLM, Ollama and any other OpenAI-compatible gateway are addressed.
    """
    kwargs: dict[str, Any] = {}
    if model.startswith("openai/"):
        if settings.openai_api_base:
            kwargs["api_base"] = settings.openai_api_base
        key = settings.resolved_api_key()
        if key:
            kwargs["api_key"] = key
    return kwargs


_ROLE_TEMPERATURE = {
    "analyst": "analyst_temperature",
    "writer": "writer_temperature",
    "critic": "critic_temperature",
    "caption": "analyst_temperature",
    "reflection": "writer_temperature",
}

_ROLE_MODEL = {
    "analyst": "analyst_model",
    "writer": "writer_model",
    "critic": "critic_model",
    "caption": "caption_model",
    "reflection": "reflection_model",
}


def build_lm(role: Role, settings: Settings | None = None, **overrides: Any) -> dspy.LM:
    """Construct the LM for one pipeline role."""
    settings = settings or get_settings()
    if role not in _ROLE_MODEL:
        raise ValueError(f"unknown LM role {role!r}; expected one of {sorted(_ROLE_MODEL)}")

    model = overrides.pop("model", None) or getattr(settings, _ROLE_MODEL[role])
    temperature = overrides.pop(
        "temperature", getattr(settings, _ROLE_TEMPERATURE[role])
    )

    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": settings.max_tokens,
        "cache": settings.lm_cache,
        "num_retries": settings.lm_num_retries,
        "timeout": settings.lm_timeout_sec,
        "callbacks": [_UsageCallback(role)],
    }
    kwargs.update(_provider_kwargs(settings, model))
    kwargs.update(overrides)
    if is_reasoning_model(model):
        budget = max(kwargs.get("max_tokens") or 0, _REASONING_MIN_COMPLETION_TOKENS)
        kwargs["temperature"] = None
        if _dspy_knows_reasoning_model(model):
            # DSPy renames max_tokens itself and rejects anything else.
            kwargs["max_tokens"] = budget
        else:
            kwargs["max_tokens"] = None
            kwargs["max_completion_tokens"] = budget
    logger.debug("LM[%s] model=%s temperature=%s", role, model, kwargs["temperature"])
    lm = BookForgeLM(model, **kwargs)
    lm.bookforge_role = role
    return lm


def _dspy_knows_reasoning_model(model: str) -> bool:
    try:
        from dspy.clients.lm import _is_openai_reasoning_model
    except ImportError:
        return False
    return _is_openai_reasoning_model(model)


class LMRouter:
    """Lazily builds and caches one LM per role for the life of a run."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._cache: dict[str, dspy.LM] = {}
        self._lock = threading.Lock()

    def get(self, role: Role) -> dspy.LM:
        with self._lock:
            lm = self._cache.get(role)
            if lm is None:
                lm = build_lm(role, self.settings)
                self._cache[role] = lm
            return lm

    def __getitem__(self, role: Role) -> dspy.LM:
        return self.get(role)


def configure_dspy(settings: Settings | None = None) -> LMRouter:
    """Install the default LM and global execution settings.

    Modules override the default per predictor via ``set_lm``; the global
    default here is what optimizers and ad-hoc calls use.
    """
    settings = settings or get_settings()
    router = LMRouter(settings)
    dspy.configure(lm=router.get("writer"))
    dspy.settings.configure(track_usage=True)
    return router


@contextmanager
def using_lm(lm: dspy.LM) -> Iterator[None]:
    """Scoped default-LM override. Thread-safe via dspy.context."""
    with dspy.context(lm=lm):
        yield
