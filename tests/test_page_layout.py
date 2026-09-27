"""Page geometry stays tight enough to use the sheet.

The previous frame (inner 28mm, outer 45mm, top 27mm, bottom 32mm) left a
printed page looking empty. These checks lock the tighter spec so a later
edit cannot quietly restore that frame.
"""

from __future__ import annotations

import re

from bookforge.design import THEMES, BookTheme


def _mm(spec: str, key: str) -> float:
    match = re.search(rf"{key}=(\d+(?:\.\d+)?)mm", spec)
    assert match, f"{key} missing from {spec}"
    return float(match.group(1))


def test_textbook_margins_are_tighter_than_the_old_frame():
    spec = BookTheme(margin_notes=True).geometry()
    assert _mm(spec, "inner") <= 16
    assert _mm(spec, "outer") <= 26
    assert _mm(spec, "top") <= 16
    assert _mm(spec, "bottom") <= 18
    # The note column must fit inside the outer margin, with a little trim.
    assert _mm(spec, "marginparwidth") + _mm(spec, "marginparsep") <= _mm(spec, "outer") - 3


def test_themes_without_a_margin_column_use_the_page():
    for name in ("academic", "compact"):
        spec = THEMES[name].geometry()
        assert _mm(spec, "inner") <= 16
        assert _mm(spec, "outer") <= 16
        assert "marginparwidth" not in spec


def test_chapters_do_not_force_a_blank_verso():
    assert "openany" in BookTheme().class_options()
    assert "openright" not in BookTheme().class_options()
