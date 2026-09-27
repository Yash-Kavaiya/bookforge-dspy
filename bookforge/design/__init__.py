"""Book design: themes, page furniture and float layout.

Separating design from content is what lets the same chapter body be typeset
in a different theme, and lets a chapter be compiled standalone during QA with
exactly the preamble the finished book will use.
"""

from bookforge.design.layout import (
    ChapterLayout,
    figure_block,
    side_by_side_block,
    table_block,
)
from bookforge.design.theme import (
    BookTheme,
    Palette,
    build_preamble,
    colophon,
    short_source,
    title_page,
)

#: Named themes the CLI and API expose.
THEMES: dict[str, BookTheme] = {
    "textbook": BookTheme(name="textbook"),
    "academic": BookTheme(
        name="academic",
        chapter_opener="plain",
        margin_notes=False,
        show_book_title_in_footer=False,
    ),
    "compact": BookTheme(
        name="compact",
        chapter_opener="minimal",
        margin_notes=False,
        base_font_pt=10,
        include_list_of_figures=False,
    ),
}


def get_theme(name: str) -> BookTheme:
    if name not in THEMES:
        raise KeyError(f"unknown theme {name!r}; expected one of {sorted(THEMES)}")
    return THEMES[name]


__all__ = [
    "THEMES",
    "BookTheme",
    "ChapterLayout",
    "Palette",
    "build_preamble",
    "colophon",
    "figure_block",
    "get_theme",
    "short_source",
    "side_by_side_block",
    "table_block",
    "title_page",
]
