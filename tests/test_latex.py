"""LaTeX tooling: rendering, sanitization and static inspection."""

from __future__ import annotations

import pytest

from bookforge.schemas import ChartSpec, TableSpec
from bookforge.tools import latex as lx


class TestEscaping:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("50% off", r"50\% off"),
            ("A & B", r"A \& B"),
            ("snake_case", r"snake\_case"),
            ("#hashtag", r"\#hashtag"),
            ("$100", r"\$100"),
        ],
    )
    def test_specials_escaped(self, raw, expected):
        assert lx.tex_escape(raw) == expected

    def test_unicode_is_converted(self):
        assert lx.tex_escape("a → b") == r"a $\to$ b"
        assert lx.tex_escape("temp 20°") == r"temp 20$^\circ$"

    def test_backslash_becomes_textbackslash(self):
        assert r"\textbackslash{}" in lx.tex_escape("C:\\path")


class TestTableRendering:
    def test_fragment_is_a_bare_tabular(self, tmp_path):
        """The float comes from \\bfTable; a fragment that opened its own would
        nest inside it and abort with "Not in outer par mode"."""
        spec = TableSpec(
            title="t", caption="A caption",
            headers=["Mode", "Mbps"], rows=[["Host", "940"], ["Overlay", "780"]],
        )
        out = lx.render_table_fragment(spec, tmp_path / "table_1.tex", label="tab:x")
        body = out.read_text()
        assert r"\begin{table}" not in body
        assert r"\caption" not in body
        assert body.count(r"\begin{tabular}") == 1
        assert r"\toprule" in body and r"\bottomrule" in body
        assert "940" in body

    def test_cells_are_escaped(self, tmp_path):
        spec = TableSpec(
            title="t", caption="c", headers=["Metric"], rows=[["99% uptime"]],
        )
        body = lx.render_table_fragment(spec, tmp_path / "t.tex").read_text()
        assert r"99\% uptime" in body

    def test_wide_tables_use_wrapping_columns(self, tmp_path):
        spec = TableSpec(
            title="t", caption="c",
            headers=list("ABCDE"), rows=[["1", "2", "3", "4", "5"]],
        )
        body = lx.render_table_fragment(spec, tmp_path / "t.tex").read_text()
        assert r"p{0.18\textwidth}" in body

    def test_note_does_not_use_linebreak_after_tabular(self, tmp_path):
        """A trailing `\\\\` after \\end{tabular} is 'There's no line here to end'."""
        spec = TableSpec(
            title="t", caption="c",
            headers=["A", "B"], rows=[["1", "2"]],
            note="Based on the speaker at 00:02:52.",
        )
        body = lx.render_table_fragment(spec, tmp_path / "t.tex").read_text()
        assert r"\end{tabular}" in body
        assert r"\footnotesize" in body
        assert "00:02:52" in body
        after = body.split(r"\end{tabular}", 1)[1]
        assert r"\\" not in after


class TestChartRendering:
    @pytest.mark.parametrize("kind", ["bar", "line", "pie"])
    def test_renders_pdf(self, tmp_path, kind):
        spec = ChartSpec(
            title="Throughput", caption="c", chart_type=kind,
            categories=["Host", "Overlay"], values=[940.0, 780.0],
        )
        out = lx.render_chart(spec, tmp_path / f"{kind}.pdf")
        assert out.exists() and out.stat().st_size > 500

    def test_existing_file_is_reused(self, tmp_path):
        spec = ChartSpec(title="t", caption="c", categories=["a"], values=[1.0])
        dest = tmp_path / "c.pdf"
        dest.write_bytes(b"sentinel")
        assert lx.render_chart(spec, dest).read_bytes() == b"sentinel"


class TestSanitizer:
    def test_strips_markdown_fences(self):
        assert lx.sanitize_chapter_tex("```latex\n\\chapter{X}\n```").strip() == r"\chapter{X}"

    def test_strips_preamble_material(self):
        raw = "\\documentclass{book}\n\\usepackage{tikz}\n\\begin{document}\n\\chapter{X}\n\\end{document}"
        out = lx.sanitize_chapter_tex(raw)
        for banned in ("documentclass", "usepackage", "begin{document}", "end{document}"):
            assert banned not in out
        assert r"\chapter{X}" in out

    def test_verbatim_blocks_are_protected(self):
        """A chapter *about* LaTeX must keep \\usepackage inside its examples.

        The original sanitizer stripped those lines wholesale and destroyed the
        very example the chapter was teaching.
        """
        raw = (
            "\\chapter{TeX}\n"
            "\\begin{verbatim}\n\\usepackage{amsmath}\n\\documentclass{book}\n\\end{verbatim}\n"
            "\\usepackage{shouldgo}\n"
        )
        out = lx.sanitize_chapter_tex(raw)
        assert r"\usepackage{amsmath}" in out, "verbatim example was destroyed"
        assert r"\documentclass{book}" in out
        assert r"\usepackage{shouldgo}" not in out, "real preamble line survived"

    def test_html_leakage_is_converted(self):
        out = lx.sanitize_chapter_tex("<figure>\\includegraphics{a.pdf}</figure><div>x</div>")
        assert r"\begin{figure}" in out and r"\end{figure}" in out
        assert "<div>" not in out

    def test_malformed_environment_close_is_repaired(self):
        assert r"\end{figure}" in lx.sanitize_chapter_tex(r"\begin{figure} x \end{figure>")

    def test_redundant_table_wrapper_is_unwrapped(self):
        raw = (
            "\\begin{table}[htbp]\\centering\\caption{c}"
            "\\input{chapters/01_x/tables/table_1.tex}\\end{table}"
        )
        out = lx.sanitize_chapter_tex(raw)
        assert out.count(r"\begin{table}") == 0
        assert r"\input{chapters/01_x/tables/table_1.tex}" in out

    def test_tikz_gets_alignment(self):
        out = lx.sanitize_chapter_tex(r"\begin{tikzpicture}[node distance=1cm]\end{tikzpicture}")
        assert "align=center" in out
        assert "node distance=1cm" in out

    def test_tikz_alignment_not_duplicated(self):
        raw = r"\begin{tikzpicture}[every node/.style={align=center}]\end{tikzpicture}"
        assert lx.sanitize_chapter_tex(raw).count("align=center") == 1

    def test_empty_input(self):
        assert lx.sanitize_chapter_tex("") == ""
        assert lx.sanitize_chapter_tex(None) == ""

    def test_strip_asset_declarations(self):
        raw = (
            "% Asset declarations for this chapter — generated from assets.json.\n"
            "% The chapter places these with \\bfFigure{<label>} / \\bfTable{<label>}.\n"
            "\\bfDeclareFigure{fig:ch01-frame1}{chapters/01_x/figures/a.jpg}"
            "{A caption}{0.82\\textwidth}\n"
            "\\bfDeclareTable{tab:ch01-t1}{chapters/01_x/tables/t.tex}{T}\n"
            "\\chapter{X}\n"
            "\\bfFigure{fig:ch01-frame1}\n"
        )
        out = lx.strip_asset_declarations(raw)
        assert "bfDeclareFigure" not in out
        assert "bfDeclareTable" not in out
        assert "Asset declarations" not in out
        assert r"\chapter{X}" in out
        assert r"\bfFigure{fig:ch01-frame1}" in out


class TestStaticInspection:
    def test_finds_referenced_files(self):
        tex = r"\includegraphics[width=0.8\textwidth]{f/a.pdf} \input{t/b.tex}"
        assert lx.find_referenced_files(tex) == ["f/a.pdf", "t/b.tex"]

    def test_finds_labels_and_refs(self):
        tex = r"\label{fig:a}\label{tab:b} see Figure~\ref{fig:a} and Table~\ref{tab:c}"
        assert set(lx.find_labels(tex)) == {"fig:a", "tab:b"}
        assert set(lx.find_refs(tex)) == {"fig:a", "tab:c"}

    def test_finds_sections(self):
        tex = r"\section*{Overview}\section{Concept One}\subsection{Detail}"
        assert lx.find_sections(tex) == ["Overview", "Concept One", "Detail"]

    def test_word_count_excludes_markup(self):
        tex = r"\chapter{Title} one two three \begin{tikzpicture}\node{a b c d e};\end{tikzpicture}"
        assert lx.count_words(tex) < 8

    def test_word_count_excludes_verbatim(self):
        prose = r"\section{X} " + "word " * 50
        code = "\\begin{verbatim}\n" + "codeword " * 200 + "\n\\end{verbatim}"
        assert lx.count_words(prose + code) < 80


class TestDuplicateLabels:
    def test_collision_is_renamed_with_refs_updated(self):
        seen = {"fig:ch01-1"}
        tex = r"\label{fig:ch01-1} see Figure~\ref{fig:ch01-1}"
        out, renamed = lx.strip_duplicate_labels(tex, seen)
        assert renamed == ["fig:ch01-1"]
        assert r"\label{fig:ch01-1-2}" in out
        assert r"\ref{fig:ch01-1-2}" in out, "the ref must follow its label"

    def test_unique_labels_untouched(self):
        seen: set[str] = set()
        tex = r"\label{fig:a}\label{fig:b}"
        out, renamed = lx.strip_duplicate_labels(tex, seen)
        assert out == tex and renamed == []
        assert seen == {"fig:a", "fig:b"}


class TestCompileErrorExtraction:
    def test_pulls_bang_lines(self):
        log = "junk\n! Undefined control sequence.\nl.42 \\badcmd\nmore junk"
        errors = lx.extract_latex_errors(log)
        assert "Undefined control sequence" in errors

    def test_falls_back_to_tail(self):
        assert "no errors here" in lx.extract_latex_errors("no errors here")


class TestAssembly:
    def test_main_tex_inputs_every_chapter(self):
        out = lx.assemble_main_tex("T", "A", ["chapters/01_a", "chapters/02_b"])
        assert r"\input{chapters/01_a/chapter.tex}" in out
        assert r"\input{chapters/02_b/chapter.tex}" in out
        assert r"\tableofcontents" in out

    def test_wrapper_is_a_full_document(self):
        out = lx.chapter_wrapper(lx.PREAMBLE, r"\chapter{X}")
        assert out.startswith("%") or out.startswith(r"\documentclass")
        assert r"\begin{document}" in out and r"\end{document}" in out
