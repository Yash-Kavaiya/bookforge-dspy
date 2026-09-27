"""HTTP API and web UI.

``create_app`` builds the FastAPI application; ``serve`` runs it. Jobs live in
``bookforge.api.jobs`` — a book build takes minutes to hours, so it runs in the
background and the browser watches it.
"""

from bookforge.api.app import create_app, get_app, serve
from bookforge.api.jobs import Job, JobRunner, get_runner

__all__ = ["Job", "JobRunner", "create_app", "get_app", "get_runner", "serve"]
