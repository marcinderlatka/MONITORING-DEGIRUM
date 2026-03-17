from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace()

sys.path.append(str(Path(__file__).resolve().parents[2]))

from monitoring.recordings import RecordingMetadata, thumbnail_candidates_for_entry

rb = pytest.importorskip("monitoring.widgets.recordings_browser", reason="PyQt runtime unavailable in test environment", exc_type=ImportError)


def _entry(tmp_path: Path) -> RecordingMetadata:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    return RecordingMetadata(
        filepath=str(video.resolve()),
        camera="Cam1",
        label="person",
        confidence=0.8,
        timestamp=1.0,
        display_time="2024-01-01 00:00:00",
        thumb_path="",
        extra={},
    )


def test_thumbnail_candidate_order_prefers_explicit_jpg(tmp_path):
    meta = _entry(tmp_path)
    explicit = tmp_path / "explicit.jpg"
    explicit.write_bytes(b"x")
    meta.thumb_path = str(explicit)
    candidates = thumbnail_candidates_for_entry(meta)
    assert candidates[0] == str(explicit.resolve())


def test_thumbnail_request_resolves_success_or_failure(monkeypatch, tmp_path):
    meta = _entry(tmp_path)

    class Signal:
        def __init__(self):
            self._calls = []

        def emit(self, *args):
            self._calls.append(args)

    task = rb.ThumbnailTask(meta)
    task.signals.ready = Signal()
    task.signals.failed = Signal()
    monkeypatch.setattr(task, "_load_image", lambda: (None, "failure"))
    task.run()
    assert len(task.signals.failed._calls) == 1

    class FakeImage:
        def isNull(self):
            return False

    task_ok = rb.ThumbnailTask(meta)
    task_ok.signals.ready = Signal()
    task_ok.signals.failed = Signal()
    monkeypatch.setattr(task_ok, "_load_image", lambda: (FakeImage(), "jpg"))
    task_ok.run()
    assert len(task_ok.signals.ready._calls) == 1


def test_thumbnail_pipeline_logs_failure_reason_if_extracted_helpers_exist(monkeypatch):
    logs = []
    monkeypatch.setattr(rb, "app_log", lambda group, message, **kwargs: logs.append((group, message, kwargs)))

    fake = types.SimpleNamespace(
        thumbnail_cache={},
        _failed_thumbnails=set(),
        _thumb_cache_key=rb.RecordingsBrowserDialog._thumb_cache_key,
        _failure_pixmap=lambda: object(),
        _apply_thumbnail_failure_to_card=lambda *_a, **_k: None,
        _apply_thumbnail_to_table=lambda *_a, **_k: None,
    )

    rb.RecordingsBrowserDialog._apply_thumbnail_failed(fake, "/tmp/x.mp4", "decode-error")

    assert logs
    assert "decode-error" in logs[-1][2].get("details", "")


def test_thumbnail_failure_path_always_marks_final_state(monkeypatch):
    fake = types.SimpleNamespace(
        _pending_thumbnails={"/tmp/x.mp4"},
        _thumbnail_tasks={"/tmp/x.mp4": object()},
        _thumb_cache_key=lambda fp: fp,
        _apply_thumbnail_failed=lambda *_a, **_k: None,
        _is_tile_visible=lambda *_a, **_k: False,
        _mp4_fallback_requested=set(),
        _thumbnail_entries={},
        _start_thumbnail_request=lambda *_a, **_k: None,
    )

    rb.RecordingsBrowserDialog._on_thumbnail_failed(fake, "/tmp/x.mp4", "mp4-read-failed")

    assert "/tmp/x.mp4" not in fake._pending_thumbnails
    assert "/tmp/x.mp4" not in fake._thumbnail_tasks
