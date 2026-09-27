"""Schema validation — the boundary that stops silent data corruption.

Each test here corresponds to a real failure mode of the ADK implementation,
where the renderer quietly "repaired" malformed LLM output into something that
looked fine and was wrong.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bookforge.schemas import (
    ChartSpec,
    ConceptNote,
    DiagramSpec,
    Exercise,
    TableSpec,
    VideoRecord,
    extract_numbers,
    normalize_number,
)


class TestTableSpec:
    def test_rectangular_table_is_accepted(self):
        spec = TableSpec(
            title="t", caption="c",
            headers=["A", "B"], rows=[["1", "2"], ["3", "4"]],
        )
        assert len(spec.rows) == 2

    def test_ragged_rows_are_rejected(self):
        """The old renderer padded short rows, shifting every cell after the gap."""
        with pytest.raises(ValidationError, match="wrong number of cells"):
            TableSpec(
                title="t", caption="c",
                headers=["A", "B", "C"], rows=[["1", "2", "3"], ["4", "5"]],
            )

    def test_overlong_rows_are_rejected(self):
        with pytest.raises(ValidationError, match="wrong number of cells"):
            TableSpec(
                title="t", caption="c",
                headers=["A", "B"], rows=[["1", "2", "3"]],
            )


class TestChartSpec:
    def test_aligned_series_is_accepted(self):
        spec = ChartSpec(
            title="t", caption="c",
            categories=["a", "b"], values=[1.0, 2.0],
        )
        assert len(spec.values) == 2

    def test_mismatched_series_is_rejected(self):
        """Truncating to the shorter list mislabels every remaining bar."""
        with pytest.raises(ValidationError, match="every value needs its own label"):
            ChartSpec(
                title="t", caption="c",
                categories=["a", "b", "c"], values=[1.0, 2.0],
            )

    def test_negative_pie_slice_is_rejected(self):
        with pytest.raises(ValidationError, match="pie slices cannot be negative"):
            ChartSpec(
                title="t", caption="c", chart_type="pie",
                categories=["a", "b"], values=[1.0, -2.0],
            )

    def test_empty_series_is_rejected(self):
        with pytest.raises(ValidationError):
            ChartSpec(title="t", caption="c", categories=[], values=[])


class TestDiagramSpec:
    def test_edges_between_known_nodes_are_accepted(self):
        spec = DiagramSpec(
            title="t", caption="c",
            elements=["Ingress", "Service", "Pod"],
            relationships=["Ingress -> Service : routes", "Service -> Pod"],
        )
        assert len(spec.relationships) == 2

    def test_edge_to_unknown_node_is_rejected(self):
        """A dangling edge renders as an arrow pointing at nothing."""
        with pytest.raises(ValidationError, match="unknown"):
            DiagramSpec(
                title="t", caption="c",
                elements=["Ingress", "Service"],
                relationships=["Ingress -> Gateway"],
            )

    def test_single_node_diagram_is_rejected(self):
        with pytest.raises(ValidationError):
            DiagramSpec(title="t", caption="c", elements=["Only"])


class TestVideoRecord:
    def test_normal_id_accepted(self):
        record = VideoRecord(video_id="dQw4w9WgXcQ", title="t", url="u")
        assert record.status == "pending"

    @pytest.mark.parametrize("bad_id", ["../../etc/passwd", "a/b", "", "id with space", "x" * 100])
    def test_unsafe_ids_rejected(self, bad_id):
        """video_id becomes a directory name, so traversal must be impossible."""
        with pytest.raises(ValidationError):
            VideoRecord(video_id=bad_id, title="t", url="u")


class TestConceptAndExercise:
    def test_trivial_explanation_rejected(self):
        with pytest.raises(ValidationError, match="too short"):
            ConceptNote(heading="H", explanation="Too short.")

    def test_exercise_without_answer_rejected(self):
        with pytest.raises(ValidationError, match="no answer"):
            Exercise(question="Why?", answer="   ")


class TestNumberHelpers:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("throughput was 940 Mbps", ["940"]),
            ("1,234 requests", ["1,234"]),
            ("it fell by 17.5 percent", ["17.5"]),
            ("only 3 nodes", []),  # single digits are too noisy to verify
            ("no numbers here", []),
        ],
    )
    def test_extract_numbers(self, text, expected):
        assert extract_numbers(text) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [("1,234.50", "1234.5"), ("940", "940"), ("940.0", "940"), ("abc", "abc")],
    )
    def test_normalize_number(self, raw, expected):
        assert normalize_number(raw) == expected


class TestAnalysisAggregation:
    def test_all_numbers_spans_every_field(self, analysis):
        numbers = set(analysis.all_numbers())
        assert "940" in numbers  # from a table cell and a chart value
        assert "780" in numbers
        assert "1450" in numbers  # from a concept explanation
