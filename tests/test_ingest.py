"""Ingest: resolving inputs, discovering recordings, importing transcripts."""

from __future__ import annotations

import pytest

from bookforge.ingest import (
    LocalVideoSource,
    MixedFolderSource,
    TranscriptSource,
    YouTubeSource,
    humanize_filename,
    natural_key,
    resolve_source,
    stable_id,
)
from bookforge.tools import transcripts as tt
from bookforge.tools.youtube import BOT_BLOCK_HELP, auth_options, is_bot_block

SRT = """1
00:00:01,000 --> 00:00:04,500
Welcome to the talk.

2
00:00:04,500 --> 00:00:09,000
Today we are building an agent harness
across two lines.

3
00:01:12,250 --> 00:01:15,000
<i>The default budget is 1500 tokens.</i>
"""

VTT = """WEBVTT
Kind: captions

00:00:02.000 --> 00:00:05.000
First line here

00:00:05.000 --> 00:00:08.000
First line here
Second line here
"""


@pytest.fixture
def video_folder(tmp_path):
    folder = tmp_path / "Decoding AI Talks"
    folder.mkdir()
    for name in ("01_intro-to-harnesses.mp4", "02_evaluation-driven.mkv", "10_wrap-up.mp4"):
        (folder / name).write_bytes(b"not really a video")
    (folder / "notes.pdf").write_bytes(b"ignore me")
    return folder


class TestIdentifiers:
    def test_stable_id_is_reproducible(self):
        assert stable_id("/a/b/talk.mp4") == stable_id("/a/b/talk.mp4")

    def test_stable_id_distinguishes_paths(self):
        assert stable_id("/a/talk.mp4") != stable_id("/b/talk.mp4")

    def test_stable_id_is_filesystem_safe(self):
        from bookforge.schemas import VideoRecord

        generated = stable_id("/a/weird name (final)!.mp4")
        VideoRecord(video_id=generated, title="t")  # validator would reject unsafe ids

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("01_intro-to-harnesses.mp4", "Intro To Harnesses"),
            ("03 - The Harness Boundary.mp4", "The Harness Boundary"),
            ("plain.mp4", "Plain"),
            ("Already Capitalised Name.mp4", "Already Capitalised Name"),
        ],
    )
    def test_humanize_filename(self, name, expected):
        from pathlib import Path

        assert humanize_filename(Path(name)) == expected

    def test_natural_sort_orders_episodes_like_a_human(self):
        names = ["ep10.mp4", "ep2.mp4", "ep1.mp4"]
        assert sorted(names, key=natural_key) == ["ep1.mp4", "ep2.mp4", "ep10.mp4"]


class TestResolveSource:
    def test_youtube_url(self, settings):
        source = resolve_source("https://youtube.com/@channel", settings)
        assert isinstance(source, YouTubeSource)

    def test_folder_becomes_mixed_source(self, settings, video_folder):
        assert isinstance(resolve_source(video_folder, settings), MixedFolderSource)

    def test_video_files(self, settings, video_folder):
        files = sorted(video_folder.glob("*.mp4"))
        assert isinstance(resolve_source(files, settings), LocalVideoSource)

    def test_transcript_files(self, settings, tmp_path):
        path = tmp_path / "talk.srt"
        path.write_text(SRT)
        assert isinstance(resolve_source([path], settings), TranscriptSource)

    def test_non_youtube_url_is_rejected(self, settings):
        with pytest.raises(ValueError, match="not a YouTube URL"):
            resolve_source("https://vimeo.com/12345", settings)

    def test_mixing_urls_and_paths_is_rejected(self, settings, video_folder):
        with pytest.raises(ValueError, match="Mixing URLs and local paths"):
            resolve_source(["https://youtube.com/@c", str(video_folder)], settings)

    def test_missing_path_is_named(self, settings, tmp_path):
        with pytest.raises(ValueError, match="do not exist"):
            resolve_source(tmp_path / "nope.mp4", settings)

    def test_unrecognised_file_type_explains_what_is_supported(self, settings, tmp_path):
        path = tmp_path / "thing.pdf"
        path.write_bytes(b"x")
        with pytest.raises(ValueError, match="recognised video or transcript"):
            resolve_source([path], settings)

    def test_empty_input(self, settings):
        with pytest.raises(ValueError, match="no input"):
            resolve_source([], settings)


class TestLocalVideoDiscovery:
    def test_finds_videos_and_ignores_other_files(self, settings, video_folder):
        discovery = LocalVideoSource([video_folder], settings).discover()
        assert len(discovery.items) == 3
        assert all(item.source == "local_video" for item in discovery.items)

    def test_titles_come_from_filenames(self, settings, video_folder):
        discovery = LocalVideoSource([video_folder], settings).discover()
        titles = {item.title for item in discovery.items}
        assert "Intro To Harnesses" in titles
        assert "Evaluation Driven" in titles

    def test_natural_ordering(self, settings, video_folder):
        discovery = LocalVideoSource([video_folder], settings).discover()
        ordered = sorted(discovery.items, key=lambda i: i.order_key)
        assert [i.title for i in ordered][-1] == "Wrap Up", "10 must sort after 02"

    def test_duplicate_paths_produce_one_chapter(self, settings, video_folder):
        one = video_folder / "01_intro-to-harnesses.mp4"
        discovery = LocalVideoSource([one, one, video_folder], settings).discover()
        ids = [item.video_id for item in discovery.items]
        assert len(ids) == len(set(ids))

    def test_max_videos_is_respected(self, settings, video_folder):
        settings.max_videos = 2
        assert len(LocalVideoSource([video_folder], settings).discover().items) == 2

    def test_empty_folder_warns(self, settings, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        discovery = LocalVideoSource([empty], settings).discover()
        assert discovery.items == []
        assert any("no files with a known video extension" in w for w in discovery.warnings)

    def test_source_line_records_provenance(self, settings, video_folder):
        assert "Local recordings" in LocalVideoSource([video_folder], settings).discover().source_line


class TestTranscriptDiscovery:
    def test_finds_transcripts(self, settings, tmp_path):
        folder = tmp_path / "transcripts"
        folder.mkdir()
        (folder / "01_first.srt").write_text(SRT)
        (folder / "02_second.vtt").write_text(VTT)
        discovery = TranscriptSource([folder], settings).discover()
        assert len(discovery.items) == 2
        assert all(item.source == "transcript" for item in discovery.items)


class TestMixedFolder:
    def test_sidecar_transcript_is_paired_not_duplicated(self, settings, tmp_path):
        """A .srt beside its .mp4 is that video's transcript, not a chapter."""
        folder = tmp_path / "mixed"
        folder.mkdir()
        (folder / "talk.mp4").write_bytes(b"video")
        (folder / "talk.srt").write_text(SRT)
        (folder / "orphan.srt").write_text(SRT)

        discovery = MixedFolderSource(folder, settings).discover()
        kinds = sorted(item.source for item in discovery.items)
        assert kinds == ["local_video", "transcript"], "the paired srt became a chapter"
        assert any("matched to a video" in w for w in discovery.warnings)

    def test_a_transcripts_only_folder_is_not_warned_about(self, settings, tmp_path):
        """It is a supported input, and one of the fastest. Not a mistake.

        The video scan legitimately reports finding no videos; a mixed folder
        that then found transcripts must not pass that on, or the preview
        renders a warning as the loudest thing on the screen for an input that
        worked exactly as documented.
        """
        folder = tmp_path / "transcripts-only"
        folder.mkdir()
        (folder / "01_first.srt").write_text(SRT)
        (folder / "02_second.vtt").write_text(VTT)

        discovery = MixedFolderSource(folder, settings).discover()
        assert len(discovery.items) == 2
        assert not any("known video extension" in w for w in discovery.warnings), (
            f"warned about a valid input: {discovery.warnings}"
        )

    def test_a_truly_empty_folder_still_warns(self, settings, tmp_path):
        """Silence there would leave the user with no explanation at all."""
        folder = tmp_path / "empty"
        folder.mkdir()
        discovery = MixedFolderSource(folder, settings).discover()
        assert discovery.items == []
        assert discovery.warnings

    def test_empty_folder_explains_supported_types(self, settings, tmp_path):
        folder = tmp_path / "nothing"
        folder.mkdir()
        discovery = MixedFolderSource(folder, settings).discover()
        assert discovery.items == []
        assert any(".mp4" in w for w in discovery.warnings)


class TestTranscriptParsing:
    def test_srt(self, tmp_path):
        path = tmp_path / "a.srt"
        path.write_text(SRT)
        segments = tt.load_segments(path)
        assert [s.start_sec for s in segments] == [1.0, 4.0, 72.0]
        assert "across two lines" in segments[1].text, "multi-line cue was truncated"
        assert "<i>" not in segments[2].text

    def test_vtt_drops_scrolling_duplicates(self, tmp_path):
        path = tmp_path / "a.vtt"
        path.write_text(VTT)
        segments = tt.load_segments(path)
        assert [s.text for s in segments] == ["First line here", "Second line here"]

    def test_timestamped_text(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text("[00:00:12] first\n[00:01:30] second\n")
        segments = tt.load_segments(path)
        assert [s.start_sec for s in segments] == [12.0, 90.0]

    def test_plain_text_gets_estimated_timings(self, tmp_path):
        """Everything downstream is keyed on time; untimed text still needs it."""
        path = tmp_path / "a.txt"
        path.write_text(
            "This is the first sentence of the talk. " * 20
        )
        segments = tt.load_segments(path)
        assert len(segments) > 1
        stamps = [s.start_sec for s in segments]
        assert stamps == sorted(stamps), "estimated timings must increase"
        assert stamps[0] >= 0.0

    def test_format_detection(self, tmp_path):
        for name, body, expected in [
            ("a.srt", SRT, "srt"),
            ("a.vtt", VTT, "vtt"),
            ("a.txt", "[00:00:01] hi", "timestamped"),
            ("a.txt", "just words here", "plain"),
        ]:
            path = tmp_path / name
            path.write_text(body)
            assert tt.detect_format(path) == expected

    def test_empty_file(self, tmp_path):
        path = tmp_path / "a.srt"
        path.write_text("")
        assert tt.load_segments(path) == []

    def test_normalized_roundtrip(self, tmp_path):
        path = tmp_path / "a.srt"
        path.write_text(SRT)
        segments = tt.load_segments(path)
        out = tt.write_normalized(segments, tmp_path / "transcript.txt")
        assert tt.load_segments(out)[0].text == segments[0].text


class TestYouTubeAuth:
    def test_cookies_file_is_used_when_it_exists(self, tmp_path):
        cookies = tmp_path / "cookies.txt"
        cookies.write_text("# Netscape HTTP Cookie File\n")
        assert auth_options(cookies_file=cookies)["cookiefile"] == str(cookies)

    def test_missing_cookies_file_is_ignored_not_fatal(self, tmp_path):
        assert auth_options(cookies_file=tmp_path / "absent.txt") == {}

    def test_browser_and_proxy(self, monkeypatch):
        from bookforge.tools import youtube

        monkeypatch.setattr(youtube, "browser_cookies_readable", lambda browser: True)
        options = auth_options(cookies_from_browser="firefox", proxy="http://p:1")
        assert options["cookiesfrombrowser"] == ("firefox",)
        assert options["proxy"] == "http://p:1"

    def test_unreadable_browser_cookies_are_dropped_not_fatal(self, monkeypatch):
        """Seen on Windows: Chrome's cookie store could not be decrypted, and
        every playlist failed even though it downloads fine without cookies."""
        from yt_dlp.cookies import CookieLoadError

        from bookforge.tools import youtube

        def locked(browser):
            raise CookieLoadError("failed to load cookies")

        monkeypatch.setattr(youtube, "extract_cookies_from_browser", locked)
        youtube.browser_cookies_readable.cache_clear()
        try:
            options = auth_options(cookies_from_browser="chrome", proxy="http://p:1")
        finally:
            youtube.browser_cookies_readable.cache_clear()
        assert "cookiesfrombrowser" not in options
        assert options["proxy"] == "http://p:1"

    def test_no_auth_configured(self):
        assert auth_options() == {}

    def test_bot_block_is_recognised(self):
        """The exact message YouTube returns from a datacentre IP."""
        message = (
            "ERROR: [youtube] abc: Sign in to confirm you're not a bot. "
            "Use --cookies-from-browser or --cookies for the authentication."
        )
        assert is_bot_block(message)
        assert not is_bot_block("HTTP Error 404: Not Found")

    def test_help_text_names_the_way_out(self):
        assert "cookies.txt" in BOT_BLOCK_HELP
        assert "BOOKFORGE_YOUTUBE_COOKIES_FILE" in BOT_BLOCK_HELP


class TestSidecarDetection:
    def test_finds_matching_transcript(self, tmp_path):
        from bookforge.pipeline.media import find_sidecar_transcript

        video = tmp_path / "talk.mp4"
        video.write_bytes(b"v")
        (tmp_path / "talk.srt").write_text(SRT)
        found = find_sidecar_transcript(video)
        assert found is not None and found.name == "talk.srt"

    def test_returns_none_without_one(self, tmp_path):
        from bookforge.pipeline.media import find_sidecar_transcript

        video = tmp_path / "talk.mp4"
        video.write_bytes(b"v")
        assert find_sidecar_transcript(video) is None
