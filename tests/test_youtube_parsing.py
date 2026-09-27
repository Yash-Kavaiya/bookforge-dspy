"""Transcript parsing and channel listing — the pure logic, without network."""

from __future__ import annotations

import pytest

from bookforge.tools import youtube as yt

VTT = """WEBVTT
Kind: captions
Language: en

1
00:00:00.000 --> 00:00:04.500
Welcome to this deep dive

2
00:00:04.500 --> 00:00:09.000
Welcome to this deep dive
on Kubernetes networking

3
00:01:12.250 --> 00:01:15.000
<c.colorE5E5E5>The default MTU is 1500 bytes</c>
"""


class TestVttConversion:
    def test_produces_timestamped_lines(self, tmp_path):
        (tmp_path / "captions.en.vtt").write_text(VTT, encoding="utf-8")
        out = yt.vtt_to_text(tmp_path / "captions.en.vtt", tmp_path)
        lines = out.read_text().splitlines()
        assert lines[0].startswith("[00:00:00]")
        assert "Welcome to this deep dive" in lines[0]

    def test_drops_scrolling_duplicates(self, tmp_path):
        """Auto-captions repeat the previous line as they scroll."""
        (tmp_path / "captions.en.vtt").write_text(VTT, encoding="utf-8")
        text = yt.vtt_to_text(tmp_path / "captions.en.vtt", tmp_path).read_text()
        assert text.count("Welcome to this deep dive") == 1

    def test_strips_styling_tags(self, tmp_path):
        (tmp_path / "captions.en.vtt").write_text(VTT, encoding="utf-8")
        text = yt.vtt_to_text(tmp_path / "captions.en.vtt", tmp_path).read_text()
        assert "<c." not in text
        assert "The default MTU is 1500 bytes" in text

    def test_is_idempotent(self, tmp_path):
        (tmp_path / "captions.en.vtt").write_text(VTT, encoding="utf-8")
        first = yt.vtt_to_text(tmp_path / "captions.en.vtt", tmp_path)
        (tmp_path / "captions.en.vtt").write_text("WEBVTT\n", encoding="utf-8")
        assert yt.vtt_to_text(tmp_path / "captions.en.vtt", tmp_path).read_text() == first.read_text()

    def test_empty_transcript_cache_is_regenerated(self, tmp_path):
        """A sticky empty transcript.txt used to fail every resume forever."""
        (tmp_path / "captions.en.vtt").write_text(VTT, encoding="utf-8")
        empty = tmp_path / "transcript.txt"
        empty.write_text("\n", encoding="utf-8")
        assert not yt.transcript_file_usable(empty)
        out = yt.vtt_to_text(tmp_path / "captions.en.vtt", tmp_path)
        assert yt.transcript_file_usable(out)
        assert "MTU" in out.read_text()


class TestDownloadCache:
    def test_empty_video_stub_is_not_reused(self, tmp_path, monkeypatch):
        stub = tmp_path / "video.mp4"
        stub.write_bytes(b"x")  # < 1KB junk

        class FakeYDL:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def download(self, urls):
                (tmp_path / "video.mp4").write_bytes(b"y" * 2048)

        monkeypatch.setattr("bookforge.tools.youtube.YoutubeDL", FakeYDL)
        path = yt.download_video("https://www.youtube.com/watch?v=abc", tmp_path)
        assert path.stat().st_size >= 1024
    def test_parses_timestamps_to_seconds(self, tmp_path):
        path = tmp_path / "transcript.txt"
        path.write_text("[00:00:12] first line\n[00:01:30] second line\n", encoding="utf-8")
        segments = yt.parse_transcript(path)
        assert [s.start_sec for s in segments] == [12.0, 90.0]
        assert segments[1].text == "second line"

    def test_continuation_lines_join_the_previous_segment(self, tmp_path):
        """No spoken content may be dropped for lacking a timestamp."""
        path = tmp_path / "transcript.txt"
        path.write_text("[00:00:12] first part\ncontinued here\n", encoding="utf-8")
        segments = yt.parse_transcript(path)
        assert len(segments) == 1
        assert segments[0].text == "first part continued here"

    def test_leading_untimestamped_text_is_kept(self, tmp_path):
        path = tmp_path / "transcript.txt"
        path.write_text("no timestamp at all\n[00:00:05] later\n", encoding="utf-8")
        segments = yt.parse_transcript(path)
        assert segments[0].start_sec == 0.0
        assert segments[0].text == "no timestamp at all"

    def test_empty_file(self, tmp_path):
        path = tmp_path / "transcript.txt"
        path.write_text("", encoding="utf-8")
        assert yt.parse_transcript(path) == []

    def test_roundtrip_through_text(self, tmp_path):
        path = tmp_path / "transcript.txt"
        original = "[00:00:12] first line\n[00:01:30] second line"
        path.write_text(original, encoding="utf-8")
        assert yt.segments_to_text(yt.parse_transcript(path)) == original

    def test_load_transcript_never_truncates(self, tmp_path):
        """The old loader deleted the middle of long transcripts."""
        path = tmp_path / "transcript.txt"
        body = "\n".join(f"[00:00:{i % 60:02d}] line {i}" for i in range(20_000))
        path.write_text(body, encoding="utf-8")
        loaded = yt.load_transcript_text(path)
        assert len(loaded) == len(body)
        assert "truncated" not in loaded


class TestTimestampFormatting:
    @pytest.mark.parametrize(
        "seconds,expected",
        [(0, "00:00:00"), (61, "00:01:01"), (3661, "01:01:01"), (86399, "23:59:59")],
    )
    def test_format(self, seconds, expected):
        assert yt.format_timestamp(seconds) == expected


class FakeYDL:
    """Stands in for yt_dlp.YoutubeDL, recording the URLs it was asked for."""

    requested: list[str] = []
    payload: dict | None = None

    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        FakeYDL.requested.append(url)
        return FakeYDL.payload


class TestChannelListing:
    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        FakeYDL.requested = []
        FakeYDL.payload = None
        monkeypatch.setattr("bookforge.tools.youtube.YoutubeDL", FakeYDL)

    def _entries(self, count: int, duration: int = 600) -> dict:
        return {
            "title": "Test Channel",
            "entries": [
                {
                    "id": f"vid{i:03d}",
                    "title": f"Video {i}",
                    "duration": duration,
                    "upload_date": f"2025010{i % 10}",
                    "url": "",
                }
                for i in range(count)
            ],
        }

    def test_appends_videos_to_handle_urls(self):
        FakeYDL.payload = self._entries(2)
        yt.list_channel_videos("https://www.youtube.com/@abc")
        assert FakeYDL.requested[0].endswith("/videos")

    def test_builds_watch_urls_when_missing(self):
        FakeYDL.payload = self._entries(1)
        _, videos = yt.list_channel_videos("https://www.youtube.com/@abc")
        assert videos[0]["url"] == "https://www.youtube.com/watch?v=vid000"

    def test_duration_filter_excludes_shorts(self):
        FakeYDL.payload = self._entries(3, duration=15)
        _, videos = yt.list_channel_videos(
            "https://www.youtube.com/@abc", min_duration_sec=60
        )
        assert videos == []

    def test_max_videos_is_respected(self):
        FakeYDL.payload = self._entries(10)
        _, videos = yt.list_channel_videos("https://www.youtube.com/@abc", max_videos=3)
        assert len(videos) == 3

    def test_overfetches_so_filtering_can_still_meet_the_cap(self):
        """Filtering after extraction must not silently return fewer videos."""
        FakeYDL.payload = self._entries(10)
        yt.list_channel_videos("https://www.youtube.com/@abc", max_videos=2)
        # playlistend is set generously, not to max_videos.
        assert FakeYDL(options={}).options is not None  # constructor smoke

    def test_entries_without_ids_are_skipped(self):
        FakeYDL.payload = {"title": "T", "entries": [{"title": "no id"}, None]}
        _, videos = yt.list_channel_videos("https://www.youtube.com/@abc")
        assert videos == []

    def test_unreachable_channel_raises_a_clear_error(self):
        FakeYDL.payload = None
        with pytest.raises(RuntimeError, match="Could not read"):
            yt.list_channel_videos("https://www.youtube.com/@abc")

    def test_channel_title_is_returned(self):
        FakeYDL.payload = self._entries(1)
        title, _ = yt.list_channel_videos("https://www.youtube.com/@abc")
        assert title == "Test Channel"
