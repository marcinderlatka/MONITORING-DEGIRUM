from __future__ import annotations

import os
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


def test_thumbnail_request_always_resolves_success_or_failure(monkeypatch, tmp_path):
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


def test_tile_thumbnail_cache_reused_on_rebuild(tmp_path):
    meta = _entry(tmp_path)
    key = rb.RecordingsBrowserDialog._thumb_cache_key(meta.filepath)

    called = {"ok": 0, "fail": 0}
    fake = types.SimpleNamespace(
        thumbnail_cache={key: object()},
        _pending_thumbnails=set(),
        _failed_thumbnails=set(),
        _thumb_cache_key=rb.RecordingsBrowserDialog._thumb_cache_key,
        _apply_thumbnail_success=lambda *_a, **_k: called.__setitem__("ok", called["ok"] + 1),
        _apply_thumbnail_failure=lambda *_a, **_k: called.__setitem__("fail", called["fail"] + 1),
    )

    rb.RecordingsBrowserDialog._request_thumbnail(fake, meta)

    assert called["ok"] == 1
    assert called["fail"] == 0


def test_failed_thumbnail_can_retry_on_refresh(monkeypatch):
    class _Btn:
        def setEnabled(self, *_a):
            return None

    fake = types.SimpleNamespace(
        refresh_btn=_Btn(),
        _history_items=[],
        _history_path="",
        _camera_dirs=[],
        _failed_thumbnails={"a", "b"},
        thumbnail_cache={"a": object(), "b": object(), "ok": object()},
        _entries=[],
        _load_diagnostics={},
        _ensure_class_filter_entries=lambda *_a: None,
        _set_default_date_bounds=lambda *_a: None,
        _apply_filters=lambda: None,
    )

    monkeypatch.setattr(rb, "load_recording_entries", lambda *_a, **_k: ([], {}))
    rb.RecordingsBrowserDialog.refresh(fake, retry_failed=True)

    assert fake._failed_thumbnails == set()
    assert "a" not in fake.thumbnail_cache and "b" not in fake.thumbnail_cache
    assert "ok" in fake.thumbnail_cache


def test_thumbnail_pipeline_logs_failure_reason_if_extracted_helpers_exist(monkeypatch):
    logs = []
    monkeypatch.setattr(rb, "app_log", lambda group, message, **kwargs: logs.append((group, message, kwargs)))

    fake = types.SimpleNamespace(
        thumbnail_cache={},
        _failed_thumbnails=set(),
        _thumb_cache_key=rb.RecordingsBrowserDialog._thumb_cache_key,
        _failure_pixmap=lambda: object(),
        _apply_thumbnail_to_card=lambda *_a, **_k: None,
        _apply_thumbnail_to_table=lambda *_a, **_k: None,
    )

    rb.RecordingsBrowserDialog._apply_thumbnail_failure(fake, "/tmp/x.mp4", "decode-error")

    assert logs
    assert "decode-error" in logs[-1][2].get("details", "")
