"""Docker build-time check: fail the image build if the shared LaTeX preamble
cannot be typeset. Otherwise this failure mode surfaces only at the very last
step of an hour-long pipeline run.

A plain script rather than an inline `RUN python - <<'PY'` heredoc, because
heredocs need a BuildKit-aware Dockerfile parser — Cloud Build's docker
builder step doesn't reliably honor `# syntax=docker/dockerfile:1` and
misparses the heredoc body as new Dockerfile instructions.
"""

from pathlib import Path

from bookforge.tools import latex as lx
from bookforge.workspace import Workspace

body = (
    r"\chapter{Build Check}\label{ch:check}" "\n"
    r"\begin{figure}[htbp]\centering\begin{tikzpicture}"
    r"\node[modernbox] (a) {Input};\node[accentbox, below=1.2cm of a] (b) {Output};"
    r"\draw[flowarrow] (a) -- (b);\end{tikzpicture}\caption{c}\label{fig:c}\end{figure}"
)
out = Path("/tmp/preamble-check")
main = Workspace.write_text(out / "check.tex", lx.chapter_wrapper(lx.PREAMBLE, body))
result = lx.compile_tex(main, out, passes=1, cwd=out)
assert result.ok, f"the shared preamble does not compile:\n{result.errors[:2000]}"
# Without makeindex every \term{} entry is recorded and no index is ever
# printed -- inert, and silent about it.
assert lx.makeindex_available(), "makeindex is missing; every book would ship without an index"
print("preamble and index tooling OK")
