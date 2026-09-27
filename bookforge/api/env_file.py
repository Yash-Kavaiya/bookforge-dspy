"""Read/write helpers for the ``.env`` file the config page edits.

Line-oriented on purpose: a ``.env`` file is meant to be hand-edited too, so a
round trip through this module must preserve comments, blank lines and the
order of everything it doesn't touch.
"""

from __future__ import annotations

from pathlib import Path


def _quote(value: str) -> str:
    if value == "" or any(c in value for c in " #\"'"):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def upsert(path: Path, updates: dict[str, str]) -> None:
    """Set ``KEY=VALUE`` for each entry, replacing an existing line or appending."""
    if not updates:
        return
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(updates)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in remaining:
            lines[i] = f"{key}={_quote(remaining.pop(key))}"
    for key, value in remaining.items():
        lines.append(f"{key}={_quote(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
