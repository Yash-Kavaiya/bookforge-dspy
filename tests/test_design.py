"""The book design, checked by typesetting it.

Everything here runs real pdflatex. That is the point: the defects this file
guards against are invisible to any test that only reads strings. A footer
that silently drops the page number, a callout whose title argument is parsed
as a key list, a theme that does not load its fonts -- all of them produce
perfectly well-formed LaTeX source.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from bookforge.design import THEMES, build_preamble
from bookforge.tools import latex as lx
from bookforge.workspace import Workspace

pytestmark = pytest.mark.skipif(
    not lx.pdflatex_available(), reason="pdflatex not installed"
)


def compile_body(tmp_path, body: str, preamble: str | None = None, passes: int = 1):
    main = Workspace.write_text(
        tmp_path / "probe.tex",
        lx.chapter_wrapper(preamble or lx.PREAMBLE, body),
    )
    return lx.compile_tex(main, tmp_path, passes=passes, cwd=tmp_path)


CHAPTER = r"\chapter{Probe}\label{ch:probe}" "\n\n"

needs_pdftotext = pytest.mark.skipif(
    shutil.which("pdftotext") is None, reason="poppler-utils not installed"
)


def pdf_to_text(pdf: Path) -> str:
    """What the reader actually sees. Source inspection cannot answer that."""
    completed = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return completed.stdout


class TestCalloutArguments:
    """A title must never be mistaken for a tcolorbox key list.

    The obvious definition -- ``\newtcolorbox{keyidea}[1][]`` -- makes the
    optional argument raw pgfkeys options, so ``\begin{keyidea}[A heading]``
    is a *fatal* error reading "I do not know the key '/tcb/A heading'". The
    chapter is discarded and the critic is handed an error that names pgfkeys
    rather than the box it came from. Every spelling below must typeset.
    """

    @pytest.mark.parametrize(
        "snippet",
        [
            r"\begin{keyidea}Body.\end{keyidea}",
            r"\begin{keyidea}[Custom heading]Body.\end{keyidea}",
            r"\begin{keyidea}{Custom heading}Body.\end{keyidea}",
            r"\begin{definitionbox}Body.\end{definitionbox}",
            r"\begin{definitionbox}[Backpressure]Body.\end{definitionbox}",
            r"\begin{definitionbox}{Backpressure}Body.\end{definitionbox}",
            r"\begin{cautionbox}Body.\end{cautionbox}",
            r"\begin{cautionbox}[Watch out]Body.\end{cautionbox}",
            r"\begin{exampleblock}Body.\end{exampleblock}",
            r"\begin{exampleblock}{Worked example}Body.\end{exampleblock}",
            r"\begin{exampleblock}[Worked example]Body.\end{exampleblock}",
            r"\begin{derivationbox}Body $x=1$.\end{derivationbox}",
            r"\begin{derivationbox}{Scale factor}$s=\frac{x}{2^{b}-1}$.\end{derivationbox}",
            r"\begin{derivationbox}[Scale factor]\begin{align}s&=\frac{x_{\max}}{2^{b-1}}.\end{align}\end{derivationbox}",
        ],
    )
    def test_every_argument_form_typesets(self, tmp_path, snippet):
        result = compile_body(tmp_path, CHAPTER + snippet)
        assert result.ok, result.errors[:600]

    def test_all_four_in_one_chapter(self, tmp_path):
        body = CHAPTER + "\n\n".join(
            [
                r"\begin{keyidea}One.\end{keyidea}",
                r"\begin{definitionbox}{Term}Two.\end{definitionbox}",
                r"\begin{cautionbox}[Careful]Three.\end{cautionbox}",
                r"\begin{exampleblock}[Example]Four.\end{exampleblock}",
            ]
        )
        assert compile_body(tmp_path, body).ok

    @needs_pdftotext
    def test_the_title_reaches_the_page(self, tmp_path):
        """A silently ignored title is the failure this guards against."""
        result = compile_body(
            tmp_path,
            CHAPTER + r"\begin{definitionbox}[Backpressure]Body.\end{definitionbox}",
        )
        assert result.ok
        text = pdf_to_text(tmp_path / "probe.pdf")
        assert "Backpressure" in text, "the bracketed title never rendered"


class TestThemes:
    @pytest.mark.parametrize("name", sorted(THEMES))
    def test_every_theme_compiles(self, tmp_path, name):
        """A theme nobody has typeset is a theme that does not work."""
        preamble = build_preamble(THEMES[name])
        result = compile_body(tmp_path, CHAPTER + "Body text.", preamble=preamble)
        assert result.ok, f"theme {name}: {result.errors[:600]}"


class TestPageFurniture:
    @needs_pdftotext
    def test_page_numbers_survive_the_footer(self, tmp_path):
        """fancyhdr keeps only the last assignment to a slot.

        Putting the page number and the book title in the same slot silently
        drops one of them -- and the book then has no page numbers at all,
        which looks entirely deliberate on the page.
        """
        body = CHAPTER + ("Body text that runs on. " * 400)
        result = compile_body(tmp_path, body, passes=2)
        assert result.ok, result.errors[:600]
        text = pdf_to_text(tmp_path / "probe.pdf")
        # Page 2 onwards carry the running foot; the number must be in there.
        assert "2" in text
        pages = text.split("\f")
        assert len(pages) >= 2, "the probe did not span two pages"
        assert any(line.strip() == "2" for line in pages[1].splitlines()), (
            "page 2 carries no page number; the footer slots collided"
        )


class TestTitlePageProvenance:
    """The title page names the source; the colophon records the path.

    A local source line carries the absolute path the build ran from. That is
    exactly what provenance needs and exactly what a title page must not show:
    it is a build machine's directory layout, and a long one wraps across three
    lines under the author.
    """

    def test_a_folder_becomes_its_name(self):
        from bookforge.design import short_source

        assert (
            short_source("Local folder: /home/yash/Talks/Decoding AI")
            == "Local folder: Decoding AI"
        )

    def test_files_become_their_basenames(self):
        from bookforge.design import short_source

        assert (
            short_source("Local recordings: /a/b/one.mp4, /a/b/two.mp4")
            == "Local recordings: one.mp4, two.mp4"
        )

    def test_a_long_list_is_summarised(self):
        from bookforge.design import short_source

        shortened = short_source("Local recordings: /a/1.mp4, /a/2.mp4, /a/3.mp4, /a/4.mp4")
        assert shortened.endswith("and others")
        assert "/a/" not in shortened

    def test_urls_are_left_alone(self):
        """A URL identifies the source and is the natural thing to print."""
        from bookforge.design import short_source

        url = "https://www.youtube.com/@itsdecodingai"
        assert short_source(url) == url

    def test_a_plain_channel_name_is_left_alone(self):
        from bookforge.design import short_source

        assert short_source("Test Channel") == "Test Channel"

    def test_the_colophon_still_carries_the_full_path(self):
        from bookforge.design import colophon

        path = "/home/yash/Talks/Decoding AI"
        assert path in colophon("BookForge", f"Local folder: {path}", "2 chapters")


class TestIndex:
    """\\term{} records index entries; something has to turn them into an index.

    \\index writes to .idx and \\printindex reads the .ind that makeindex builds
    from it. Miss the makeindex step and the whole chain is inert: every entry
    is recorded, \\printindex finds no .ind and prints nothing, and the book
    ships without an index and without an error anywhere -- the same shape as
    the frame_captions defect this project was rewritten to fix.
    """

    needs_makeindex = pytest.mark.skipif(
        not lx.makeindex_available(), reason="makeindex not installed"
    )

    @needs_makeindex
    @needs_pdftotext
    def test_indexed_terms_reach_the_printed_index(self, tmp_path):
        # latex.py's _MIN_INDEX_ENTRIES (8) drops a thinner index outright
        # rather than print a near-empty page, so this needs enough distinct
        # \term{} calls to clear that bar and actually exercise makeindex.
        terms = [
            "harness", "stopping rule", "refiner", "verifier",
            "grounding score", "chunk analysis", "manifest", "chapter loop",
        ]
        body = (
            CHAPTER
            + " ".join(rf"The \term{{{term}}} matters here." for term in terms)
            + " "
            + ("Body text that runs on. " * 200)
            # chapter_wrapper omits the back matter that assemble_main_tex adds.
            + "\n\\clearpage\n\\printindex\n"
        )
        result = compile_body(tmp_path, body, passes=3)
        assert result.ok, result.errors[:600]

        assert (tmp_path / "probe.idx").exists(), "no index entries were recorded"
        assert (tmp_path / "probe.ind").exists(), "makeindex never ran"

        text = pdf_to_text(tmp_path / "probe.pdf")
        tail = text.split("\f")[-3:]
        printed = "\n".join(tail)
        assert "Index" in printed, "the index was built but never printed"
        assert "harness" in printed
        assert "stopping rule" in printed

    def test_a_document_with_no_terms_still_compiles(self, tmp_path):
        """No .idx means nothing to do, not a failure."""
        assert compile_body(tmp_path, CHAPTER + "Plain body.", passes=2).ok
