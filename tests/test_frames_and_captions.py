"""Frame curation and captioning.

Two ADK defects are covered here:

1. ``curate_frames`` computed sharpness and contrast for every frame and then
   selected purely by position (``assets[int(i * step)]``), so a blurry
   transition frame beat a crisp slide sitting one index away.

2. ``frame_captions`` was asked of the analyst, which ran *before* the asset
   stage invented the filenames — so the lookup never matched and every figure
   in every book got the same fallback caption.
"""

from __future__ import annotations

import dspy

from bookforge.dspy_runtime.modules.captioner import FrameCaptioner
from bookforge.dspy_runtime.signatures import FrameCaption, FrameCaptionSet
from bookforge.schemas import FrameAsset
from bookforge.tools import frames as frame_tools

# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------


def _asset(index: int, quality: float, tmp_path) -> FrameAsset:
    source = tmp_path / f"src_{index:03d}.jpg"
    source.write_bytes(b"jpeg-bytes-" + str(index).encode())
    return FrameAsset(
        file=str(source), timestamp_sec=index * 5.0, phash=f"{index:016x}",
        sharpness=quality * 40, contrast=quality * 70, quality=quality,
    )


class TestCuration:
    def test_returns_all_when_under_the_cap(self, tmp_path):
        assets = [_asset(i, 0.5, tmp_path) for i in range(5)]
        curated = frame_tools.curate_frames(assets, tmp_path / "out", max_frames=12)
        assert len(curated) == 5

    def test_respects_the_cap(self, tmp_path):
        assets = [_asset(i, 0.5, tmp_path) for i in range(40)]
        curated = frame_tools.curate_frames(assets, tmp_path / "out", max_frames=8)
        assert len(curated) == 8

    def test_quality_wins_within_a_bucket(self, tmp_path):
        """The regression: a sharp frame must beat its blurry neighbour.

        Twenty frames, one cap of four buckets; the best frame in each bucket
        carries a distinctive timestamp so we can prove it was chosen.
        """
        assets = []
        for i in range(20):
            # The 3rd frame of each 5-frame bucket is the crisp one.
            quality = 0.95 if i % 5 == 2 else 0.15
            assets.append(_asset(i, quality, tmp_path))

        curated = frame_tools.curate_frames(assets, tmp_path / "out", max_frames=4)
        chosen_indices = [int(a.timestamp_sec / 5) for a in curated]
        assert chosen_indices == [2, 7, 12, 17], "curation ignored the quality signal"

    def test_coverage_is_spread_across_the_timeline(self, tmp_path):
        assets = [_asset(i, 0.5 + (i % 7) / 20, tmp_path) for i in range(60)]
        curated = frame_tools.curate_frames(assets, tmp_path / "out", max_frames=6)
        stamps = [a.timestamp_sec for a in curated]
        assert stamps == sorted(stamps)
        assert stamps[-1] - stamps[0] > 200, "all picks clustered in one part of the video"

    def test_files_are_copied_with_stable_names(self, tmp_path):
        assets = [_asset(i, 0.5, tmp_path) for i in range(3)]
        out = tmp_path / "figures"
        curated = frame_tools.curate_frames(assets, out, max_frames=3)
        for asset in curated:
            assert (out / asset.file).exists()
            assert asset.file.startswith("frame_")
            assert "/" not in asset.file, "curated file must be a bare basename"

    def test_empty_and_zero_cap(self, tmp_path):
        assert frame_tools.curate_frames([], tmp_path / "o", 5) == []
        assets = [_asset(0, 0.5, tmp_path)]
        assert frame_tools.curate_frames(assets, tmp_path / "o", 0) == []

    def test_missing_source_is_skipped_not_fatal(self, tmp_path):
        assets = [_asset(0, 0.5, tmp_path)]
        assets.append(
            FrameAsset(file=str(tmp_path / "gone.jpg"), timestamp_sec=99.0, phash="x")
        )
        curated = frame_tools.curate_frames(assets, tmp_path / "out", max_frames=5)
        assert len(curated) == 1


class TestQualityScore:
    def test_sharp_high_contrast_beats_flat_dark(self):
        good = frame_tools._quality_score(sharpness=35, contrast=60, brightness=130)
        bad = frame_tools._quality_score(sharpness=3, contrast=5, brightness=8)
        assert good > bad
        assert 0.0 <= bad <= good <= 1.0

    def test_high_skin_fraction_demotes_talking_head(self):
        slide = frame_tools._quality_score(35, 60, 130, skin_fraction=0.05)
        face = frame_tools._quality_score(35, 60, 130, skin_fraction=0.55)
        assert face < slide


class TestTimeFootnotes:
    def test_appends_video_stamp(self):
        assert frame_tools.with_time_footnote("A crisp slide.", 75) == (
            "A crisp slide [video 01:15]."
        )

    def test_idempotent_when_stamp_present(self):
        once = frame_tools.with_time_footnote("Diagram of the queue.", 90)
        assert frame_tools.with_time_footnote(once, 90) == once

    def test_format_omits_hours_when_short(self):
        assert frame_tools.format_video_timestamp(65) == "01:05"
        assert frame_tools.format_video_timestamp(3661) == "01:01:01"


# ---------------------------------------------------------------------------
# Captioning
# ---------------------------------------------------------------------------


class ScriptedCaption(dspy.Module):
    def __init__(self, response: FrameCaptionSet | None = None, fail: bool = False) -> None:
        super().__init__()
        self.response = response
        self.fail = fail
        self.seen_context = ""

    def forward(self, frames_with_context: str = "", **kwargs) -> dspy.Prediction:
        self.seen_context = frames_with_context
        if self.fail:
            raise RuntimeError("captioning backend unavailable")
        return dspy.Prediction(captions=self.response or FrameCaptionSet())


def _curated() -> list[FrameAsset]:
    return [
        FrameAsset(file="frame_01_30s.jpg", timestamp_sec=30.0, phash="a", quality=0.8),
        FrameAsset(file="frame_02_48s.jpg", timestamp_sec=48.0, phash="b", quality=0.7),
    ]


class TestCaptioner:
    def test_captions_are_keyed_by_the_real_filenames(
        self, settings, analysis, transcript_segments
    ):
        """The whole bug: the model must see the filenames it is captioning.

        In the ADK pipeline the analyst was asked for captions keyed by
        filename before any filename existed, so the lookup silently missed.
        """
        captioner = FrameCaptioner(settings)
        captioner.caption = ScriptedCaption(
            FrameCaptionSet(
                captions=[
                    FrameCaption(frame_file="frame_01_30s.jpg", caption="The MTU slide."),
                    FrameCaption(frame_file="frame_02_48s.jpg", caption="Throughput results."),
                ]
            )
        )
        result = captioner(
            video_title="K8s", analysis=analysis,
            frames=_curated(), segments=transcript_segments,
        )
        by_file = {c.frame_file: c.caption for c in result.captions}
        assert by_file["frame_01_30s.jpg"] == "The MTU slide."
        assert by_file["frame_02_48s.jpg"] == "Throughput results."

    def test_model_receives_the_spoken_context(
        self, settings, analysis, transcript_segments
    ):
        """A caption is only grounded if the model saw what was being said."""
        captioner = FrameCaptioner(settings)
        scripted = ScriptedCaption(FrameCaptionSet())
        captioner.caption = scripted
        captioner(
            video_title="K8s", analysis=analysis,
            frames=_curated(), segments=transcript_segments,
        )
        context = scripted.seen_context
        assert "frame_01_30s.jpg" in context
        assert "00:00:30" in context
        assert "MTU" in context, "the transcript window was not supplied"
        assert "940 megabits" in context

    def test_visual_briefing_is_prepended(
        self, settings, analysis, transcript_segments
    ):
        captioner = FrameCaptioner(settings)
        scripted = ScriptedCaption(FrameCaptionSet())
        captioner.caption = scripted
        captioner(
            video_title="K8s",
            analysis=analysis,
            frames=_curated(),
            segments=transcript_segments,
            visual_insights="At 00:48 a bar chart shows 940 Mbps throughput.",
        )
        assert "VISUAL BRIEFING" in scripted.seen_context
        assert "940 Mbps" in scripted.seen_context

    def test_useless_frames_are_dropped(self, settings, analysis, transcript_segments):
        captioner = FrameCaptioner(settings)
        captioner.caption = ScriptedCaption(
            FrameCaptionSet(
                captions=[
                    FrameCaption(frame_file="frame_01_30s.jpg", caption="Title card.", is_useful=False),
                    FrameCaption(frame_file="frame_02_48s.jpg", caption="Throughput results."),
                ]
            )
        )
        result = captioner(
            video_title="K8s", analysis=analysis,
            frames=_curated(), segments=transcript_segments,
        )
        assert result.dropped == ["frame_01_30s.jpg"]
        assert [c.frame_file for c in result.captions] == ["frame_02_48s.jpg"]

    def test_missing_caption_falls_back_rather_than_dropping_the_figure(
        self, settings, analysis, transcript_segments
    ):
        captioner = FrameCaptioner(settings)
        captioner.caption = ScriptedCaption(FrameCaptionSet())  # model returned nothing
        result = captioner(
            video_title="K8s", analysis=analysis,
            frames=_curated(), segments=transcript_segments,
        )
        assert len(result.captions) == 2
        assert all(c.caption for c in result.captions)

    def test_backend_failure_still_returns_captions(
        self, settings, analysis, transcript_segments
    ):
        """Captioning is an enhancement; its failure must not lose the figures."""
        captioner = FrameCaptioner(settings)
        captioner.caption = ScriptedCaption(fail=True)
        result = captioner(
            video_title="K8s", analysis=analysis,
            frames=_curated(), segments=transcript_segments,
        )
        assert len(result.captions) == 2

    def test_no_frames(self, settings, analysis, transcript_segments):
        captioner = FrameCaptioner(settings)
        captioner.caption = ScriptedCaption(FrameCaptionSet())
        result = captioner(
            video_title="K8s", analysis=analysis, frames=[], segments=transcript_segments
        )
        assert result.captions == [] and result.dropped == []

    def test_batches_large_frame_sets(self, settings, analysis, transcript_segments):
        captioner = FrameCaptioner(settings)
        calls = {"n": 0}

        class Counting(dspy.Module):
            def forward(self, **kwargs):
                calls["n"] += 1
                return dspy.Prediction(captions=FrameCaptionSet())

        captioner.caption = Counting()
        frames = [
            FrameAsset(file=f"frame_{i:02d}.jpg", timestamp_sec=i * 5.0, phash=str(i))
            for i in range(30)
        ]
        captioner(
            video_title="K8s", analysis=analysis, frames=frames, segments=transcript_segments
        )
        assert calls["n"] == 3  # 30 frames / 12 per call
