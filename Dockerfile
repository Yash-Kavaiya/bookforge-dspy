# BookForge: DSPy pipeline turning YouTube channels into LaTeX textbooks.
#
# The image carries a real TeX distribution on purpose. Verification by
# execution is the core of the design — without pdflatex the pipeline runs in
# draft mode and cannot confirm a chapter compiles.

# --- UI build stage ----------------------------------------------------------
# The web UI is an Angular SPA (frontend/); it compiles to bookforge/api/webapp/,
# which the FastAPI app serves. Built here so the runtime image never needs Node.
FROM node:22-slim AS webapp
WORKDIR /webapp
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend ./
RUN npm run build

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-pictures \
        texlive-fonts-recommended \
        texlive-extra-utils \
        lmodern \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY bookforge ./bookforge
COPY --from=webapp /bookforge/api/webapp ./bookforge/api/webapp
RUN pip install --upgrade pip && pip install -e ".[ui,observability]"

# Workspace and compiled artifacts are mounted, not baked in.
VOLUME ["/app/data", "/app/artifacts"]

# Fail the build if the shared preamble cannot be typeset — the failure mode
# that otherwise surfaces only at the very last step of a long run.
COPY scripts/check_preamble.py /tmp/check_preamble.py
RUN python /tmp/check_preamble.py && rm /tmp/check_preamble.py

# The UI binds to every interface inside the container; the container itself
# is what should be kept private. There is no authentication in BookForge --
# publish this port only behind a proxy that authenticates.
EXPOSE 8000
ENV BOOKFORGE_WORKSPACE_ROOT=/app/data \
    BOOKFORGE_ARTIFACTS_DIR=/app/artifacts

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request as u; p=os.environ.get('PORT','8000'); u.urlopen('http://127.0.0.1:'+p+'/api/health', timeout=4)"

# Default to the UI, listening on $PORT (Cloud Run injects this; falls back to
# 8000 for a plain `docker run`). For a one-shot build instead:
#   docker run --rm -v "$PWD/data:/app/data" bookforge \
#     build /app/data/talks --max-videos 3
ENTRYPOINT ["bookforge"]
CMD ["serve", "--host", "0.0.0.0"]
