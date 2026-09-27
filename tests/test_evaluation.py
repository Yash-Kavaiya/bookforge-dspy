"""Dataset harvesting, the evaluation harness and optimizer wiring.

The design claim being tested: BookForge's metrics are reference-free, so every
video the pipeline processes becomes a training example with no labelling. If
dataset collection needed hand-written gold chapters, optimization would cost
more than it saves.
"""

from __future__ import annotations

import dspy
import pytest

from bookforge.dspy_runtime.lm import LMRouter
from bookforge.evaluation.datasets import (
    build_analysis_examples,
    build_chapter_examples,
    build_full_analysis_examples,
    collect,
    discover_workspaces,
    split,
)
from bookforge.evaluation.harness import compare, evaluate_program
from bookforge.evaluation.optimize import TARGETS, build_optimizer, build_target_program
from bookforge.schemas import MediaBundle, VideoRecord


@pytest.fixture
def populated_workspace(workspace, analysis, assets_manifest, transcript_segments):
    """A workspace that looks like the pipeline already ran over three videos."""
    manifest = workspace.init_manifest("https://yt/c", "Test Channel", "test-channel")
    for index in range(1, 4):
        record = VideoRecord(
            video_id=f"vid{index:03d}",
            title=f"Lecture {index}: Kubernetes Networking",
            url=f"https://yt/watch?v=vid{index:03d}",
            chapter_number=index,
            status="verified",
        )
        manifest.videos.append(record)

        video_dir = workspace.video_dir(record.video_id)
        transcript_path = video_dir / "transcript.txt"
        transcript_path.write_text(
            "\n".join(f"[{s.timestamp}] {s.text}" for s in transcript_segments) * 30,
            encoding="utf-8",
        )
        (video_dir / "media.json").write_text(
            MediaBundle(
                video_id=record.video_id,
                transcript_path=str(transcript_path),
                transcript_source="captions",
            ).model_dump_json(),
            encoding="utf-8",
        )

        chapter_dir = workspace.chapter_dir(record)
        (chapter_dir / "analysis.json").write_text(
            analysis.model_dump_json(), encoding="utf-8"
        )
        (chapter_dir / "assets.json").write_text(
            assets_manifest.model_dump_json(), encoding="utf-8"
        )
    workspace.save_manifest(manifest)
    return workspace


class TestDiscovery:
    def test_finds_workspaces_with_manifests(self, populated_workspace, settings):
        found = discover_workspaces(settings.workspace_root_abs)
        assert [w.channel_slug for w in found] == ["test-channel"]

    def test_ignores_directories_without_a_manifest(self, populated_workspace, settings):
        (settings.workspace_root_abs / "not-a-channel").mkdir()
        assert len(discover_workspaces(settings.workspace_root_abs)) == 1

    def test_missing_root_returns_empty(self, tmp_path):
        assert discover_workspaces(tmp_path / "nope") == []


class TestAnalysisExamples:
    def test_builds_examples_with_inputs(self, populated_workspace, settings):
        examples = build_analysis_examples(populated_workspace, settings)
        assert examples
        first = examples[0]
        assert set(first.inputs().keys()) == {
            "video_title", "chunk_index", "chunk_total", "transcript_chunk"
        }

    def test_carries_the_transcript_for_the_metric(self, populated_workspace, settings):
        """Reference-free scoring: the example's own transcript is the ground truth."""
        example = build_analysis_examples(populated_workspace, settings)[0]
        assert example.transcript
        assert "Kubernetes" in example.transcript

    def test_samples_rather_than_exhausts(self, populated_workspace, settings):
        settings.transcript_chunk_chars = 2_000
        examples = build_analysis_examples(populated_workspace, settings, per_video_chunks=2)
        assert len(examples) <= 6  # 3 videos x 2 windows

    def test_full_examples_carry_segments(self, populated_workspace):
        examples = build_full_analysis_examples(populated_workspace)
        assert len(examples) == 3
        assert examples[0].segments
        assert set(examples[0].inputs().keys()) == {"video_title", "segments"}


class TestChapterExamples:
    def test_builds_from_analysis_and_assets_on_disk(self, populated_workspace):
        examples = build_chapter_examples(populated_workspace)
        assert len(examples) == 3
        assert set(examples[0].inputs().keys()) == {
            "video_title", "chapter_number", "chapter_label_slug",
            "analysis", "assets_manifest",
        }

    def test_carries_verification_context(self, populated_workspace):
        """The metric must be able to compile candidates for real."""
        example = build_chapter_examples(populated_workspace)[0]
        assert example.assets_manifest_obj is not None
        assert example.workspace is not None
        assert example.build_dir is not None

    def test_skips_chapters_without_analysis(self, populated_workspace):
        for chapter_dir in (populated_workspace.chapters_dir).iterdir():
            (chapter_dir / "analysis.json").unlink()
        assert build_chapter_examples(populated_workspace) == []


class TestCollectAndSplit:
    def test_collect_across_workspaces(self, populated_workspace, settings):
        assert len(collect("chapter", settings.workspace_root_abs, settings)) == 3

    def test_collect_unknown_target(self, populated_workspace, settings):
        with pytest.raises(KeyError):
            collect("nonsense", settings.workspace_root_abs, settings)

    def test_collect_with_no_data_explains_why(self, settings, tmp_path):
        with pytest.raises(RuntimeError, match="Run the pipeline"):
            collect("chapter", tmp_path / "empty", settings)

    def test_split_is_deterministic(self):
        examples = [dspy.Example(i=i) for i in range(20)]
        first = split(examples)
        second = split(examples)
        assert [e.i for e in first[0]] == [e.i for e in second[0]]
        assert [e.i for e in first[1]] == [e.i for e in second[1]]

    def test_split_partitions_without_overlap(self):
        examples = [dspy.Example(i=i) for i in range(20)]
        train, val = split(examples, val_fraction=0.4)
        assert len(train) + len(val) == 20
        assert not ({e.i for e in train} & {e.i for e in val})

    def test_split_of_a_single_example(self):
        train, val = split([dspy.Example(i=0)])
        assert len(train) == 1 and val == []


class TestHarness:
    def test_scores_a_program(self, populated_workspace, analysis):
        class Stub(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(analysis=analysis)

        devset = build_analysis_examples(populated_workspace)[:2]
        summary = evaluate_program(
            Stub(), devset,
            __import__("bookforge.metrics.composite", fromlist=["x"]).scalar_analysis_metric,
            label="stub", display_progress=False,
        )
        assert 0.0 <= summary.score <= 1.0
        assert summary.n == len(devset)

    def test_comparison_reports_a_delta(self, populated_workspace, analysis):
        from bookforge.metrics.composite import scalar_analysis_metric

        class Good(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(analysis=analysis)

        class Bad(dspy.Module):
            def forward(self, **kwargs):
                return dspy.Prediction(analysis=None)

        devset = build_analysis_examples(populated_workspace)[:2]
        result = compare(Bad(), Good(), devset, scalar_analysis_metric)
        assert result.improved
        assert result.delta > 0
        assert "improved" in result.report()

    def test_empty_devset_rejected(self):
        with pytest.raises(ValueError, match="empty devset"):
            evaluate_program(dspy.Module(), [], lambda *a, **k: 1.0)


class TestOptimizerWiring:
    @pytest.mark.parametrize("name", ["gepa", "miprov2", "simba", "bootstrap"])
    def test_every_optimizer_constructs(self, name, settings):
        optimizer = build_optimizer(name, lambda *a, **k: 1.0, settings, LMRouter(settings))
        assert hasattr(optimizer, "compile")

    def test_unknown_optimizer_rejected(self, settings):
        with pytest.raises(KeyError, match="unknown optimizer"):
            build_optimizer("magic", lambda *a, **k: 1.0, settings, LMRouter(settings))

    @pytest.mark.parametrize("target", sorted(TARGETS))
    def test_target_programs_build(self, target, settings):
        program = build_target_program(target, settings, LMRouter(settings))
        assert list(program.named_predictors())

    def test_chapter_target_disables_the_semantic_critic(self, settings):
        """Optimize against the reproducible verifier, not a stochastic judge."""
        program = build_target_program("chapter", settings, LMRouter(settings))
        assert program.critic is None

    def test_unknown_target_rejected(self, settings):
        with pytest.raises(KeyError):
            build_target_program("nonsense", settings, LMRouter(settings))
