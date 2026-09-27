"""Workspace durability and concurrency.

The ADK version guarded manifest writes with an ``asyncio.Lock`` while the
actual racing writers were OS threads, which provides no mutual exclusion at
all — a lost status update makes a finished chapter look pending forever, and
a resumed run redoes work it already paid for.
"""

from __future__ import annotations

import json
import threading

import pytest

from bookforge.schemas import ChannelManifest, VideoRecord
from bookforge.workspace import Workspace, slugify


class TestSlugify:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Hello World!", "hello-world"),
            ("  Multiple   Spaces  ", "multiple-spaces"),
            ("Ünïcödé", "n-c-d"),
            ("", "untitled"),
            ("!!!", "untitled"),
        ],
    )
    def test_slugify(self, raw, expected):
        assert slugify(raw) == expected

    def test_length_capped(self):
        assert len(slugify("x" * 200)) <= 48


class TestManifest:
    def test_init_and_load_roundtrip(self, workspace):
        workspace.init_manifest("https://yt/c", "My Channel", "my-channel")
        assert workspace.manifest_exists()
        assert workspace.load_manifest().channel_title == "My Channel"

    def test_update_video(self, workspace, video_record):
        manifest = workspace.init_manifest("u", "t", "s")
        manifest.videos.append(video_record)
        workspace.save_manifest(manifest)

        updated = workspace.update_video(video_record.video_id, status="verified", qa_score=0.9)
        assert updated.status == "verified"
        assert workspace.load_manifest().videos[0].qa_score == 0.9

    def test_update_unknown_video_raises(self, workspace):
        workspace.init_manifest("u", "t", "s")
        with pytest.raises(KeyError):
            workspace.update_video("nope", status="verified")

    def test_update_unknown_field_raises(self, workspace, video_record):
        manifest = workspace.init_manifest("u", "t", "s")
        manifest.videos.append(video_record)
        workspace.save_manifest(manifest)
        with pytest.raises(AttributeError):
            workspace.update_video(video_record.video_id, not_a_field=1)

    def test_pending_excludes_verified(self, workspace):
        manifest = workspace.init_manifest("u", "t", "s")
        manifest.videos = [
            VideoRecord(video_id="a", title="A", url="u", chapter_number=1, status="verified"),
            VideoRecord(video_id="b", title="B", url="u", chapter_number=2, status="written"),
            VideoRecord(video_id="c", title="C", url="u", chapter_number=3),
        ]
        workspace.save_manifest(manifest)
        assert [v.video_id for v in workspace.pending_videos()] == ["b", "c"]

    def test_pending_is_chapter_ordered(self, workspace):
        manifest = workspace.init_manifest("u", "t", "s")
        manifest.videos = [
            VideoRecord(video_id="c", title="C", url="u", chapter_number=3),
            VideoRecord(video_id="a", title="A", url="u", chapter_number=1),
            VideoRecord(video_id="b", title="B", url="u", chapter_number=2),
        ]
        workspace.save_manifest(manifest)
        assert [v.chapter_number for v in workspace.pending_videos()] == [1, 2, 3]


class TestConcurrency:
    def test_concurrent_updates_do_not_lose_writes(self, workspace):
        """The core regression: N threads, N distinct updates, none lost."""
        manifest = workspace.init_manifest("u", "t", "s")
        ids = [f"vid{i:03d}" for i in range(24)]
        manifest.videos = [
            VideoRecord(video_id=vid, title=vid, url="u", chapter_number=i)
            for i, vid in enumerate(ids, start=1)
        ]
        workspace.save_manifest(manifest)

        errors: list[Exception] = []
        barrier = threading.Barrier(len(ids))

        def worker(video_id: str) -> None:
            try:
                barrier.wait(timeout=10)  # maximise contention
                workspace.update_video(video_id, status="verified", qa_score=0.77)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(vid,)) for vid in ids]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert errors == []
        final = workspace.load_manifest()
        assert all(v.status == "verified" for v in final.videos), "an update was lost"
        assert len(final.videos) == len(ids)

    def test_manifest_stays_valid_json_under_contention(self, workspace):
        manifest = workspace.init_manifest("u", "t", "s")
        manifest.videos = [
            VideoRecord(video_id=f"v{i}", title="t", url="u", chapter_number=i)
            for i in range(1, 9)
        ]
        workspace.save_manifest(manifest)

        def churn(video_id: str) -> None:
            for _ in range(6):
                workspace.update_video(video_id, status="media")
                workspace.update_video(video_id, status="written")

        threads = [
            threading.Thread(target=churn, args=(f"v{i}",)) for i in range(1, 9)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        # A torn write would leave unparseable JSON behind.
        payload = json.loads(workspace.manifest_path.read_text())
        ChannelManifest.model_validate(payload)

    def test_lock_file_is_released(self, workspace, video_record):
        manifest = workspace.init_manifest("u", "t", "s")
        manifest.videos.append(video_record)
        workspace.save_manifest(manifest)
        workspace.update_video(video_record.video_id, status="media")
        assert not workspace.lock_path.exists()


class TestAtomicWrites:
    def test_write_text_is_atomic(self, workspace, tmp_path):
        target = tmp_path / "nested" / "file.tex"
        Workspace.write_text(target, "content")
        assert target.read_text() == "content"
        assert not list(tmp_path.rglob("*.tmp")), "a temp file leaked"

    def test_overwrite(self, tmp_path):
        target = tmp_path / "f.txt"
        Workspace.write_text(target, "one")
        Workspace.write_text(target, "two")
        assert target.read_text() == "two"

    def test_read_text_tolerates_bad_encoding(self, tmp_path):
        target = tmp_path / "f.txt"
        target.write_bytes(b"valid \xff\xfe bytes")
        assert "valid" in Workspace.read_text(target)


class TestPaths:
    def test_chapter_dir_created_with_subdirs(self, workspace, video_record):
        directory = workspace.chapter_dir(video_record)
        assert directory.name.startswith("01_")
        assert (directory / "figures").is_dir()
        assert (directory / "tables").is_dir()

    def test_rel_is_workspace_relative_posix(self, workspace, video_record):
        rel = workspace.rel(workspace.chapter_dir(video_record) / "figures" / "a.pdf")
        assert rel.startswith("chapters/01_")
        assert "\\" not in rel
