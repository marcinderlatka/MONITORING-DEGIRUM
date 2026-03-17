from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace()
cv2_stub = sys.modules["cv2"]
if not hasattr(cv2_stub, "resize"):
    cv2_stub.INTER_AREA = 0
    cv2_stub.FONT_HERSHEY_SIMPLEX = 0
    cv2_stub.resize = lambda frame, size, interpolation=0: np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2_stub.rectangle = lambda *args, **kwargs: None
    cv2_stub.putText = lambda *args, **kwargs: None

if "degirum_tools" not in sys.modules:
    class _DummyVideoWriter:
        def __init__(self, *_args, **_kwargs):
            pass
        def write(self, _frame):
            return None
        def release(self):
            return None

    class _DummyOpenStream:
        def __init__(self, *_args, **_kwargs):
            pass
        def __enter__(self):
            return types.SimpleNamespace(get=lambda *_a, **_k: 25.0)
        def __exit__(self, *_args):
            return False

    sys.modules["degirum_tools"] = types.SimpleNamespace(
        VideoWriter=_DummyVideoWriter,
        open_video_stream=lambda *_a, **_k: _DummyOpenStream(),
        video_source=lambda *_a, **_k: iter(()),
    )

if "PyQt5" not in sys.modules:
    qtcore = types.ModuleType("PyQt5.QtCore")

    class _QThread:
        def __init__(self, *_args, **_kwargs):
            pass
        def wait(self, *_args, **_kwargs):
            return True
        def isRunning(self):
            return False
        @staticmethod
        def msleep(_ms):
            return None

    class _Signal:
        def __init__(self, *_args, **_kwargs):
            pass
        def emit(self, *_args, **_kwargs):
            return None

    qtcore.QThread = _QThread
    qtcore.pyqtSignal = lambda *_args, **_kwargs: _Signal()
    pyqt5 = types.ModuleType("PyQt5")
    pyqt5.QtCore = qtcore
    sys.modules["PyQt5"] = pyqt5
    sys.modules["PyQt5.QtCore"] = qtcore

sys.path.append(str(Path(__file__).resolve().parents[2]))

from monitoring.workers import (
    CameraWorker,
    _advance_next_due,
    _aggregate_fps,
    _build_metrics_payload,
    _dropped_frames_delta,
    _preview_interval_for_role,
)


class _DummyModel:
    def predict(self, _frame):
        return types.SimpleNamespace(results=[])


def _worker() -> CameraWorker:
    return CameraWorker(camera={"name": "Cam", "rtsp": "rtsp://x"}, model=_DummyModel(), index=0)


def test_thumbnail_generation():
    worker = _worker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[120:320, 200:420] = (255, 255, 255)
    thumb = worker._build_thumbnail_frame(frame, (200, 120, 420, 320), "person", 0.8)
    assert thumb.shape[:2] == (180, 320)


def test_writer_fps_computation():
    worker = _worker()
    worker.rtsp_fps = 8
    worker.fps = 20
    assert worker._compute_effective_writer_fps(25.0) == 8.0


def test_stream_fps_estimation():
    worker = _worker()
    worker._stream_fps_window.extend([1.0, 1.5, 2.0, 2.5, 3.0])
    worker._stream_fps_last_calc_ts = 0.0
    est = worker._get_effective_stream_fps()
    assert est == 2.0


def test_detection_event_summary():
    worker = _worker()
    worker.current_event_detection_count = 4
    worker.current_event_confidence_sum = 2.0
    worker.current_event_max_confidence = 0.8
    avg = worker.current_event_confidence_sum / worker.current_event_detection_count
    assert avg == 0.5
    assert worker.current_event_max_confidence == 0.8


def test_effective_preview_emit_policy_for_roles():
    main_i = _preview_interval_for_role("main", 15, 3, True)
    thumb_i = _preview_interval_for_role("thumb", 15, 3, True)
    hidden_i = _preview_interval_for_role("hidden", 15, 3, True)
    assert main_i < thumb_i
    assert hidden_i == float("inf")


def test_inference_schedule_next_due_logic():
    next_due, skipped = _advance_next_due(now_ts=10.0, next_due_ts=8.0, interval=0.5)
    assert next_due >= 9.5
    assert skipped >= 2


def test_prerecord_buffer_basis_helper():
    worker = _worker()
    worker.rtsp_fps = 0
    worker.stream_fps = 7.5
    assert worker._get_prerecord_buffer_fps_basis() == 7.5


def test_hidden_preview_role_does_not_break_recording_state():
    worker = _worker()
    worker.set_preview_role("hidden")
    worker.recording = True
    worker.is_recording_active = True
    assert worker.preview_role == "hidden"
    assert worker.is_recording_active is True


def test_record_start_mode_semantics_for_include_prerecord_first(monkeypatch):
    worker = _worker()
    worker.enable_recording = True
    worker.record_start_mode = "include_prerecord_first"
    worker.prerecord_buffer.clear()
    prerecord_a = np.ones((2, 2, 3), dtype=np.uint8) * 10
    prerecord_b = np.ones((2, 2, 3), dtype=np.uint8) * 20
    detection = np.ones((2, 2, 3), dtype=np.uint8) * 30
    worker.prerecord_buffer.extend([prerecord_a, prerecord_b])

    writes: list[np.ndarray] = []

    class DummyThread:
        def __init__(self, *_a, **_k):
            self.frames_written = 0
            self.dropped_frames = 0
            self.queue_peak = 0
        def start(self):
            return None
        def write(self, frame):
            writes.append(frame.copy())
        def stop(self):
            return None

    monkeypatch.setattr("monitoring.workers.RecordingThread", DummyThread)
    monkeypatch.setattr(worker, "_save_recording_metadata", lambda *_a, **_k: None)
    monkeypatch.setattr(worker, "_update_event_thumbnail", lambda *_a, **_k: None)

    ok = worker._start_recording_session(detection, detection, "person", 0.9, None, 25.0, 3.0)
    assert ok is True
    assert len(writes) == 3
    assert np.array_equal(writes[0], prerecord_a)
    assert np.array_equal(writes[1], prerecord_b)
    assert np.array_equal(writes[2], detection)


def test_runtime_settings_apply_without_restart_for_live_fields():
    worker = _worker()
    worker.visible_classes = ["person"]
    cfg = {
        "name": "Cam",
        "fps": 12,
        "rtsp_fps": 6,
        "confidence_threshold": 0.4,
        "confidence_threshold_draw": 0.35,
        "confidence_threshold_record": 0.5,
        "draw_overlays": False,
        "enable_detection": True,
        "enable_recording": True,
        "visible_classes": ["person", "car"],
        "record_classes": ["car"],
        "detection_hours": "08:00-20:00",
        "record_path": "./recordings",
        "pre_seconds": 2,
        "lost_seconds": 1,
        "post_seconds": 3,
        "required_hits_to_start_recording": 2,
        "required_misses_to_end_detection": 2,
        "min_record_seconds": 4,
        "thumbnail_mode": "best_detection",
        "record_start_mode": "include_prerecord_first",
        "preview_fps_main": 10,
        "preview_fps_thumb": 2,
        "preview_pause_when_hidden": True,
    }
    worker.apply_runtime_settings(cfg)

    assert worker.fps == 12
    assert worker.rtsp_fps == 6
    assert worker.confidence_threshold_draw == 0.35
    assert worker.confidence_threshold_record == 0.5
    assert "car" in worker.visible_classes_lower


def test_apply_runtime_settings_refreshes_class_filters():
    worker = _worker()
    worker.apply_runtime_settings({
        "name": "Cam",
        "visible_classes": ["Person", "DOG"],
        "record_classes": ["Car"],
    })
    assert worker.visible_classes_lower == {"person", "dog"}
    assert worker.record_classes_lower == {"car"}


def test_worker_stop_returns_bool_and_sets_signal():
    worker = _worker()
    result = worker.stop(timeout_ms=1)
    assert worker.stop_signal is True
    assert result is True


def test_aggregate_fps_handles_zero_window():
    assert _aggregate_fps(10, 0.0) == 0.0


def test_aggregate_fps_computes_expected_rate():
    assert _aggregate_fps(25, 5.0) == 5.0


def test_dropped_frames_delta_never_negative():
    assert _dropped_frames_delta(3, 5) == 0
    assert _dropped_frames_delta(12, 5) == 7


def test_metrics_payload_uses_consistent_keys():
    payload = _build_metrics_payload(capture_fps=4.0, queue_size=3)
    assert sorted(payload.keys()) == sorted([
        "capture_fps",
        "infer_fps",
        "preview_emit_fps",
        "ui_render_ms",
        "queue_size",
        "dropped_frames",
        "cpu_percent",
        "rss_mb",
    ])
    assert payload["capture_fps"] == 4.0
    assert payload["queue_size"] == 3
    assert payload["infer_fps"] == 0.0
