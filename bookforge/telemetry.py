"""MLflow tracing and run tracking.

DSPy ships a native MLflow autologger, so a single call captures every module
call, prompt, completion and token count — replacing the manual OpenTelemetry
span plumbing the ADK version needed.

Telemetry is strictly optional and never fatal: an unreachable tracking server
degrades to local logging rather than taking the run down with it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from bookforge.config import Settings, get_settings

logger = logging.getLogger(__name__)

_ENABLED = False


def setup_tracing(settings: Settings | None = None) -> bool:
    """Enable MLflow autologging for DSPy. Returns True when active."""
    global _ENABLED
    settings = settings or get_settings()
    if not settings.enable_mlflow:
        return False

    try:
        import mlflow
        import mlflow.dspy
    except ImportError:
        logger.warning(
            "MLflow tracing requested but mlflow is not installed. "
            "Install it with: pip install 'bookforge[observability]'"
        )
        return False

    try:
        if settings.mlflow_tracking_uri:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        mlflow.dspy.autolog(
            log_compiles=True,
            log_evals=True,
            log_traces_from_compile=True,
        )
    except Exception as exc:
        logger.warning("could not initialise MLflow tracing: %s", exc)
        return False

    _ENABLED = True
    logger.info(
        "MLflow tracing active: experiment '%s' at %s",
        settings.mlflow_experiment,
        settings.mlflow_tracking_uri or "(default local store)",
    )
    return True


@contextmanager
def run(name: str, **params: Any) -> Iterator[None]:
    """Wrap a unit of work in an MLflow run when tracing is enabled."""
    if not _ENABLED:
        yield
        return
    try:
        import mlflow
    except ImportError:
        yield
        return

    with mlflow.start_run(run_name=name):
        try:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        except Exception:
            logger.debug("could not log MLflow params", exc_info=True)
        yield


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Record metrics on the active run; a no-op when tracing is off."""
    if not _ENABLED:
        return
    try:
        import mlflow

        mlflow.log_metrics(
            {k: float(v) for k, v in metrics.items() if v is not None}, step=step
        )
    except Exception:
        logger.debug("could not log MLflow metrics", exc_info=True)


def log_artifact(path: str) -> None:
    if not _ENABLED:
        return
    try:
        import mlflow

        mlflow.log_artifact(path)
    except Exception:
        logger.debug("could not log MLflow artifact", exc_info=True)
