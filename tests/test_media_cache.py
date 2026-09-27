"""Media cache fingerprint and resume behaviour."""

from __future__ import annotations

from bookforge.config import Settings
from bookforge.pipeline.media import (
    _cached_bundle_usable,
    media_settings_fingerprint,
)
from bookforge.schemas import FrameAsset, MediaBundle, VideoRecord


def _settings(**overrides) -> Settings:
    base = Settings(
        openai_api_key="test",
        frame_interval_sec=5,
        frame_scene_threshold=0.25,
        frame_phash_threshold=6,
        download_resolution=480,
        whisper_model="base",
    )
    return base.model_copy(update=overrides)


class TestMediaFingerprint:
    def test_fingerprint_changes_with_frame_interval(self):
        a = media_settings_fingerprint(_settings(frame_interval_sec=5))
        b = media_settings_fingerprint(_settings(frame_interval_sec=10))
        assert a != b

    def test_stale_fingerprint_rejects_cache(self, tmp_path):
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("[00:00:00] hello there everyone watching\n", encoding="utf-8")
        video = tmp_path / "video.mp4"
        video.write_bytes(b"x" * 2048)
        bundle = MediaBundle(
            video_id="v1",
            video_path=str(video),
            transcript_path=str(transcript),
            settings_fingerprint="old-fingerprint",
        )
        record = VideoRecord(
            video_id="v1",
            title="Talk",
            url="",
            source="local_video",
            source_path=str(video),
            chapter_number=1,
        )
        assert not _cached_bundle_usable(
            bundle, record, media_settings_fingerprint(_settings())
        )

    def test_matching_fingerprint_with_intact_files_is_usable(self, tmp_path):
        settings = _settings()
        fingerprint = media_settings_fingerprint(settings)
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("[00:00:00] hello there everyone watching\n", encoding="utf-8")
        video = tmp_path / "video.mp4"
        video.write_bytes(b"x" * 2048)
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"jpeg")
        bundle = MediaBundle(
            video_id="v1",
            video_path=str(video),
            transcript_path=str(transcript),
            settings_fingerprint=fingerprint,
            frames=[FrameAsset(file=str(frame), timestamp_sec=1.0, phash="a")],
        )
        record = VideoRecord(
            video_id="v1",
            title="Talk",
            url="",
            source="local_video",
            source_path=str(video),
            chapter_number=1,
        )
        assert _cached_bundle_usable(bundle, record, fingerprint)

    def test_missing_video_file_rejects_cache(self, tmp_path):
        settings = _settings()
        fingerprint = media_settings_fingerprint(settings)
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("[00:00:00] hello there everyone watching\n", encoding="utf-8")
        bundle = MediaBundle(
            video_id="v1",
            video_path=str(tmp_path / "gone.mp4"),
            transcript_path=str(transcript),
            settings_fingerprint=fingerprint,
        )
        record = VideoRecord(
            video_id="v1",
            title="Talk",
            url="",
            source="youtube",
            chapter_number=1,
        )
        assert not _cached_bundle_usable(bundle, record, fingerprint)
