from __future__ import annotations

import sys
import time
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


class _ModelWithResults:
    def __init__(self, results):
        self._results = list(results)

    def predict(self, _frame):
        return types.SimpleNamespace(results=list(self._results))


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
    assert worker._compute_effective_writer_fps(25.0) == 25.0
    assert worker._compute_effective_writer_fps(0.0) == 8.0


def test_stream_fps_estimation():
    worker = _worker()
    worker._stream_fps_window.extend([1.0, 1.5, 2.0, 2.5, 3.0])
    worker._stream_fps_last_calc_ts = 0.0
    est = worker._get_effective_stream_fps()
    assert est == 2.0


def test_start_recording_prefers_measured_stream_over_detect_fps(monkeypatch):
    worker = _worker()
    worker.enable_recording = True
    worker.fps = 3
    worker.rtsp_fps = 0
    worker._stream_fps_window.extend([float(i) * 0.1 for i in range(30)])
    worker.stream_fps = 10.0
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    meta_box: dict[str, object] = {}

    class DummyThread:
        def __init__(self, *_a, **_k):
            self.frames_written = 0
            self.dropped_frames = 0
            self.queue_peak = 0
        def start(self):
            return None
        def write(self, _frame):
            return None
        def stop(self):
            return True

    monkeypatch.setattr("monitoring.workers.RecordingThread", DummyThread)
    monkeypatch.setattr(worker, "_update_event_thumbnail", lambda *_a, **_k: None)
    monkeypatch.setattr(worker, "_save_recording_metadata", lambda meta: meta_box.update(meta))

    ok = worker._start_recording_session(frame, frame, "person", 0.8, None, 30.0, 3.0)
    assert ok is True
    assert worker.current_writer_fps > 9.0
    assert meta_box["writer_fps_reason"] == "measured_stream"
    assert float(meta_box["writer_fps_selected"]) > 9.0


def test_start_recording_fallback_when_stream_measurement_missing(monkeypatch):
    worker = _worker()
    worker.enable_recording = True
    worker.rtsp_fps = 6
    worker.fps = 3
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    meta_box: dict[str, object] = {}

    class DummyThread:
        def __init__(self, *_a, **_k):
            self.frames_written = 0
            self.dropped_frames = 0
            self.queue_peak = 0
        def start(self):
            return None
        def write(self, _frame):
            return None
        def stop(self):
            return True

    monkeypatch.setattr("monitoring.workers.RecordingThread", DummyThread)
    monkeypatch.setattr(worker, "_update_event_thumbnail", lambda *_a, **_k: None)
    monkeypatch.setattr(worker, "_save_recording_metadata", lambda meta: meta_box.update(meta))

    ok = worker._start_recording_session(frame, frame, "person", 0.8, None, 30.0, 3.0)
    assert ok is True
    assert worker.current_writer_fps == 6.0
    assert meta_box["stream_fps_measured"] == 0.0
    assert meta_box["writer_fps_reason"] == "rtsp_limit"


def test_start_recording_uses_ffmpeg_backend_when_configured(monkeypatch):
    worker = CameraWorker(camera={"name": "Cam", "rtsp": "rtsp://x", "recording_backend": "ffmpeg"}, model=_DummyModel(), index=0)
    worker.enable_recording = True
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    meta_box: dict[str, object] = {}

    class DummyThread:
        def __init__(self, *_a, **kwargs):
            self.frames_written = 0
            self.dropped_frames = 0
            self.queue_peak = 0
            backend = kwargs["backend"]
            self.ffmpeg_exit_code = getattr(backend, "ffmpeg_exit_code", None)
        def start(self):
            return None
        def write(self, _frame):
            return None
        def stop(self):
            return True

    monkeypatch.setattr("monitoring.workers.RecordingThread", DummyThread)
    monkeypatch.setattr(worker, "_update_event_thumbnail", lambda *_a, **_k: None)
    monkeypatch.setattr(worker, "_save_recording_metadata", lambda meta: meta_box.update(meta))

    ok = worker._start_recording_session(frame, frame, "person", 0.8, None, 30.0, 3.0)
    assert ok is True
    assert worker.current_recording_backend == "ffmpeg"
    assert meta_box["writer_backend"] == "ffmpeg"


def test_writer_fps_session_hysteresis_and_clamp():
    worker = _worker()
    worker._last_session_writer_fps = 10.0

    held_fps, held_reason = worker._select_session_writer_fps(measured_stream_fps=10.8, detect_fps=3.0)
    assert held_fps == 10.0
    assert held_reason.endswith("hysteresis_hold")

    clamped_fps, clamped_reason = worker._select_session_writer_fps(measured_stream_fps=25.0, detect_fps=3.0)
    assert clamped_fps == 13.0
    assert clamped_reason.endswith("clamped")


def test_detection_event_summary():
    worker = _worker()
    worker.current_event_detection_count = 4
    worker.current_event_confidence_sum = 2.0
    worker.current_event_max_confidence = 0.8
    avg = worker.current_event_confidence_sum / worker.current_event_detection_count
    assert avg == 0.5
    assert worker.current_event_max_confidence == 0.8


def test_sensitivity_profile_applies_thresholds():
    worker = CameraWorker(
        camera={"name": "Cam", "rtsp": "rtsp://x", "sensitivity_profile": "high_precision"},
        model=_DummyModel(),
        index=0,
    )
    assert worker.confidence_threshold_record == 0.7
    assert worker.required_hits_to_start_recording == 3


def test_telemetry_payload_contains_calibration_stats():
    worker = _worker()
    now = worker.telemetry_started_ts + 25 * 3600
    worker.telemetry_trigger_count = 10
    worker.telemetry_false_positive_proxy_count = 2
    worker.telemetry_confidence_sum = 6.2
    worker.telemetry_confidence_count = 10

    payload = worker._telemetry_payload(now)

    assert payload["false_positive_proxy_rate"] == 0.2
    assert payload["avg_confidence"] == 0.62
    assert payload["trigger_frequency_per_hour"] > 0
    assert payload["suggested_record_threshold"] is not None


def test_effective_preview_emit_policy_for_roles():
    main_i = _preview_interval_for_role("main", 15, 8, 3, True)
    grid_i = _preview_interval_for_role("grid", 15, 8, 3, True)
    thumb_i = _preview_interval_for_role("thumb", 15, 8, 3, True)
    hidden_i = _preview_interval_for_role("hidden", 15, 8, 3, True)
    assert main_i < grid_i < thumb_i
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



def test_grid_preview_role_is_accepted_and_has_target_fps():
    worker = _worker()
    worker.set_preview_role("grid")
    worker.preview_fps_grid = 7.0

    assert worker.preview_role == "grid"
    assert worker._effective_preview_target_fps() == 7.0


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




def test_preview_path_without_detection_or_overlays_avoids_copy():
    worker = _worker()
    worker.set_enable_detection(False)
    worker.set_draw_overlays(False)
    worker.set_preview_role("main")

    class _FrameProxy:
        def __init__(self, frame):
            self._frame = frame
            self.copy_calls = 0
        def copy(self):
            self.copy_calls += 1
            return self._frame.copy()
        def __getattr__(self, name):
            return getattr(self._frame, name)

    emitted: list[object] = []

    class _SignalRecorder:
        def emit(self, frame, _index):
            emitted.append(frame)

    worker.main_preview_signal = _SignalRecorder()
    worker.thumb_preview_signal = _SignalRecorder()
    frame = _FrameProxy(np.zeros((8, 8, 3), dtype=np.uint8))

    raw_frame, preview_frame = worker._capture_next_frame(frame, now_mono=1.0)
    _result, _detected, _best_label, _best_score, _best_bbox, overlays = worker._maybe_run_inference(raw_frame, now_mono=1.0)
    worker._maybe_emit_preview(preview_frame, overlays, now_mono=1.0)

    assert overlays == []
    assert frame.copy_calls == 0
    assert emitted == [frame, frame]
    assert worker.prerecord_buffer[-1] is frame

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
        "thumbnail_overlay_enabled": False,
        "thumbnail_box_thickness": 3,
        "thumbnail_font_scale": 0.9,
        "thumbnail_font_thickness": 2,
        "record_start_mode": "include_prerecord_first",
        "preview_fps_main": 10,
        "preview_fps_thumb": 2,
        "preview_pause_when_hidden": True,
        "preview_main_max_width": 1024,
        "preview_main_max_height": 576,
        "preview_thumb_max_width": 256,
        "preview_thumb_max_height": 144,
    }
    worker.apply_runtime_settings(cfg)

    assert worker.fps == 12
    assert worker.rtsp_fps == 6
    assert worker.confidence_threshold_draw == 0.35
    assert worker.confidence_threshold_record == 0.5
    assert "car" in worker.visible_classes_lower
    assert worker.thumbnail_overlay_enabled is False
    assert worker.thumbnail_box_thickness == 3
    assert worker.thumbnail_font_scale == 0.9
    assert worker.thumbnail_font_thickness == 2


def test_thumbnail_overlay_style_is_applied(monkeypatch):
    worker = _worker()
    worker.thumbnail_overlay_enabled = True
    worker.thumbnail_box_thickness = 4
    worker.thumbnail_font_scale = 1.1
    worker.thumbnail_font_thickness = 3

    calls: dict[str, object] = {}

    def _rectangle(_canvas, _p1, _p2, _color, thickness):
        calls["rectangle_thickness"] = thickness
        return None

    def _put_text(_canvas, _text, _org, _font, font_scale, _color, thickness):
        calls["font_scale"] = font_scale
        calls["font_thickness"] = thickness
        return None

    monkeypatch.setattr("monitoring.workers.cv2.rectangle", _rectangle)
    monkeypatch.setattr("monitoring.workers.cv2.putText", _put_text)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    worker._make_detection_overlay_frame(frame, (10, 10, 80, 80), "person", 0.99)

    assert calls["rectangle_thickness"] == 4
    assert calls["font_scale"] == 1.1
    assert calls["font_thickness"] == 3


def test_thumbnail_overlay_can_be_disabled(monkeypatch):
    worker = _worker()
    worker.thumbnail_overlay_enabled = False

    draw_calls = {"rectangle": 0, "put_text": 0}

    def _rectangle(*_args, **_kwargs):
        draw_calls["rectangle"] += 1
        return None

    def _put_text(*_args, **_kwargs):
        draw_calls["put_text"] += 1
        return None

    monkeypatch.setattr("monitoring.workers.cv2.rectangle", _rectangle)
    monkeypatch.setattr("monitoring.workers.cv2.putText", _put_text)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    worker._make_detection_overlay_frame(frame, (10, 10, 80, 80), "person", 0.99)

    assert draw_calls["rectangle"] == 0
    assert draw_calls["put_text"] == 0


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


def test_heartbeat_reports_nonzero_cpu_after_metrics_window_with_diagnostics_disabled(monkeypatch):
    worker = _worker()
    worker.performance_diagnostics_enabled = False
    worker.state.last_metrics_log_ts = 0.0
    worker.state.metrics_window_started_ts = 100.0
    worker.state.metrics_last_cpu_wall_ts = 100.0
    worker.state.metrics_last_cpu_process_ts = 10.0
    worker.state.metrics_last_sample_ts = 100.0

    monotonic_values = iter([110.0, 110.0, 120.0, 120.0])
    monkeypatch.setattr("monitoring.workers.time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr("monitoring.workers.time.process_time", lambda: 15.0)

    captured_status: dict[str, object] = {}

    class _SignalRecorder:
        def emit(self, camera_name, status):
            captured_status["camera_name"] = camera_name
            captured_status["status"] = status

    worker.worker_status_signal = _SignalRecorder()

    worker._maybe_log_metrics(detection_interval=1.0)
    worker._maybe_emit_heartbeat()

    assert worker.state.last_cpu_percent > 0.0
    status = dict(captured_status.get("status") or {})
    assert float(status.get("cpu_percent", 0.0)) > 0.0
    assert float(status.get("cpu_sample_age_ms", -1.0)) >= 0.0
    assert status.get("metrics_fresh") is True


def test_thumb_hidden_role_downscales_payload_before_emit():
    worker = _worker()
    worker.set_preview_role("thumb")
    worker.preview_main_max_width = 1920
    worker.preview_main_max_height = 1080
    worker.preview_thumb_max_width = 320
    worker.preview_thumb_max_height = 180

    emitted_main: list[np.ndarray] = []
    emitted_thumb: list[np.ndarray] = []

    class _SignalRecorderMain:
        def emit(self, frame, _index):
            emitted_main.append(frame)

    class _SignalRecorderThumb:
        def emit(self, frame, _index):
            emitted_thumb.append(frame)

    worker.main_preview_signal = _SignalRecorderMain()
    worker.thumb_preview_signal = _SignalRecorderThumb()

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    worker._maybe_emit_preview(frame, overlays=[], now_mono=10.0)
    worker._maybe_emit_preview(frame, overlays=[], now_mono=11.0)

    assert emitted_main
    assert emitted_thumb
    assert emitted_main[-1].shape[1] <= 320
    assert emitted_main[-1].shape[0] <= 180
    assert emitted_thumb[-1].shape[1] <= 320
    assert emitted_thumb[-1].shape[0] <= 180

def test_duplicate_worker_slot_is_blocked_for_same_camera_source():
    first = CameraWorker(camera={"name": "CamA", "rtsp": "rtsp://same"}, model=_DummyModel(), index=1)
    second = CameraWorker(camera={"name": "CamB", "rtsp": "rtsp://same"}, model=_DummyModel(), index=2)

    assert first._acquire_worker_slot() is True
    try:
        assert second._acquire_worker_slot() is False
    finally:
        first._release_worker_slot()
        second._release_worker_slot()


def test_worker_slot_release_allows_new_worker_after_cleanup():
    first = CameraWorker(camera={"name": "CamA", "rtsp": "rtsp://again"}, model=_DummyModel(), index=10)
    second = CameraWorker(camera={"name": "CamB", "rtsp": "rtsp://again"}, model=_DummyModel(), index=11)

    assert first._acquire_worker_slot() is True
    first._release_worker_slot()
    assert second._acquire_worker_slot() is True
    second._release_worker_slot()


def test_recorder_degradation_level_responds_to_queue_and_drops():
    worker = _worker()
    worker.recorder_queue_warn_threshold = 10
    worker.recorder_queue_critical_threshold = 20
    worker.recorder_dropped_warn_threshold = 2
    worker.recorder_dropped_critical_threshold = 4
    worker.recorder_queue_peak_warn_threshold = 12
    worker.recorder_queue_peak_critical_threshold = 24

    assert worker._compute_recorder_degradation_level(queue_size=5, dropped_frames=0, queue_peak=5) == 0
    assert worker._compute_recorder_degradation_level(queue_size=11, dropped_frames=0, queue_peak=5) >= 1
    assert worker._compute_recorder_degradation_level(queue_size=21, dropped_frames=0, queue_peak=5) >= 2


def test_dynamic_recorder_degradation_updates_stride_and_writer_fps():
    worker = _worker()
    worker.recording = True
    worker.current_writer_fps = 8.0
    worker._current_writer_fps_base = 8.0
    worker.recorder_min_dynamic_writer_fps = 1.0
    worker.recorder_queue_warn_threshold = 2
    worker.recorder_queue_critical_threshold = 4
    worker.recorder_dropped_warn_threshold = 1
    worker.recorder_dropped_critical_threshold = 3
    worker.recorder_queue_peak_warn_threshold = 3
    worker.recorder_queue_peak_critical_threshold = 5

    class _Queue:
        def qsize(self):
            return 5

    worker.record_thread = types.SimpleNamespace(queue=_Queue(), dropped_frames=3, queue_peak=6)
    worker._apply_dynamic_recorder_degradation(now_mono=100.0)

    assert worker._recorder_degradation_level >= 2
    assert worker.recorder_enqueue_stride >= 3
    assert worker.current_writer_fps < 8.0


def test_thumb_hidden_fast_path_uses_single_resize_call(monkeypatch):
    worker = _worker()
    worker.set_draw_overlays(False)
    worker.set_preview_role("thumb")

    calls: list[tuple[int, int]] = []

    def _resize_cached(frame, max_width, max_height):
        calls.append((int(max_width), int(max_height)))
        return frame

    monkeypatch.setattr(worker, "_resize_for_preview_cached", _resize_cached)

    class _SignalRecorder:
        def __init__(self):
            self.frames: list[object] = []
        def emit(self, frame, _index):
            self.frames.append(frame)

    main_signal = _SignalRecorder()
    thumb_signal = _SignalRecorder()
    worker.main_preview_signal = main_signal
    worker.thumb_preview_signal = thumb_signal
    frame = np.zeros((20, 40, 3), dtype=np.uint8)

    worker._maybe_emit_preview(frame, overlays=[], now_mono=1.0)
    worker._maybe_emit_preview(frame, overlays=[], now_mono=2.0)

    expected_w = int(max(1, round(worker.preview_thumb_max_width * worker.preview_resolution_factor)))
    expected_h = int(max(1, round(worker.preview_thumb_max_height * worker.preview_resolution_factor)))
    assert calls == [(expected_w, expected_h)]


def test_rejection_counter_below_record_threshold():
    model = _ModelWithResults([{"label": "person", "confidence": 0.2, "bbox": [0.1, 0.1, 0.2, 0.2]}])
    worker = CameraWorker(camera={"name": "Cam", "rtsp": "rtsp://x"}, model=model, index=0)
    worker.enable_detection = True
    worker.record_classes = ["person"]
    worker.refresh_class_filters()
    worker.confidence_threshold_record = 0.5

    _result, detected, *_ = worker._maybe_run_inference(np.zeros((10, 10, 3), dtype=np.uint8), now_mono=1.0)

    assert detected is False
    assert worker.rejection_counters["below_record_threshold"] == 1


def test_rejection_counter_class_not_in_record_classes():
    model = _ModelWithResults([{"label": "car", "confidence": 0.95, "bbox": [1, 1, 5, 5]}])
    worker = CameraWorker(camera={"name": "Cam", "rtsp": "rtsp://x"}, model=model, index=0)
    worker.enable_detection = True
    worker.record_classes = ["person"]
    worker.refresh_class_filters()

    _result, detected, *_ = worker._maybe_run_inference(np.zeros((10, 10, 3), dtype=np.uint8), now_mono=1.0)

    assert detected is False
    assert worker.rejection_counters["class_not_in_record_classes"] == 1


def test_rejection_counter_outside_schedule(monkeypatch):
    model = _ModelWithResults([{"label": "person", "confidence": 0.95, "bbox": [1, 1, 5, 5]}])
    worker = CameraWorker(camera={"name": "Cam", "rtsp": "rtsp://x"}, model=model, index=0)
    worker.enable_detection = True
    worker.record_classes = ["person"]
    worker.refresh_class_filters()
    monkeypatch.setattr(worker, "_is_within_schedule", lambda: False)

    _result, detected, *_ = worker._maybe_run_inference(np.zeros((10, 10, 3), dtype=np.uint8), now_mono=1.0)

    assert detected is False
    assert worker.rejection_counters["outside_schedule"] == 1


def test_rejection_counter_detection_disabled():
    model = _ModelWithResults([{"label": "person", "confidence": 0.95, "bbox": [1, 1, 5, 5]}])
    worker = CameraWorker(camera={"name": "Cam", "rtsp": "rtsp://x"}, model=model, index=0)
    worker.enable_detection = False
    worker.draw_overlays = True

    _result, detected, *_ = worker._maybe_run_inference(np.zeros((10, 10, 3), dtype=np.uint8), now_mono=1.0)

    assert detected is False
    assert worker.rejection_counters["detection_disabled"] == 1


def test_heartbeat_includes_rejection_counters():
    worker = _worker()
    worker.rejection_counters["below_record_threshold"] = 3
    worker.rejection_counters["class_not_in_record_classes"] = 2
    worker.rejection_counters["outside_schedule"] = 1
    worker.rejection_counters["detection_disabled"] = 4

    captured_status: dict[str, object] = {}

    class _SignalRecorder:
        def emit(self, camera_name, status):
            captured_status["camera_name"] = camera_name
            captured_status["status"] = status

    worker.worker_status_signal = _SignalRecorder()
    worker._maybe_emit_heartbeat()

    status = dict(captured_status.get("status") or {})
    counters = dict(status.get("rejection_counters") or {})
    assert counters == {
        "below_record_threshold": 3,
        "class_not_in_record_classes": 2,
        "outside_schedule": 1,
        "detection_disabled": 4,
    }


def test_resize_cache_reuses_last_scaled_frame(monkeypatch):
    worker = _worker()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, :, 0] = 5

    resize_calls = {"count": 0}
    real_resize = cv2_stub.resize

    def _counting_resize(*args, **kwargs):
        resize_calls["count"] += 1
        return real_resize(*args, **kwargs)

    monkeypatch.setattr("monitoring.workers.cv2.resize", _counting_resize)
    resized_a = worker._resize_for_preview_cached(frame, 80, 60)
    resized_b = worker._resize_for_preview_cached(frame.copy(), 80, 60)

    assert resize_calls["count"] == 1
    assert resized_a is resized_b


def test_synthetic_preview_helpers_benchmark():
    worker = _worker()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    loops = 250

    start_plain = time.perf_counter()
    for _ in range(loops):
        worker._resize_for_preview(frame, 320, 180)
    plain_elapsed = time.perf_counter() - start_plain

    worker._preview_resize_cache_key = None
    worker._preview_resize_cache_frame = None
    start_cached = time.perf_counter()
    for _ in range(loops):
        worker._resize_for_preview_cached(frame, 320, 180)
    cached_elapsed = time.perf_counter() - start_cached

    assert plain_elapsed >= 0.0
    assert cached_elapsed >= 0.0
