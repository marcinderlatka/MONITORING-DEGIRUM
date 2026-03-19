from __future__ import annotations

import json
import sys
import types
from pathlib import Path

if "PyQt5" not in sys.modules:
    qtcore = types.ModuleType("PyQt5.QtCore")
    qtwidgets = types.ModuleType("PyQt5.QtWidgets")

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, _name):
            return lambda *a, **k: None

    class _QThread(_Dummy):
        @staticmethod
        def msleep(_ms):
            return None
        def wait(self, *_a, **_k):
            return True
        def isRunning(self):
            return False

    class _Signal:
        def __init__(self, *_args, **_kwargs):
            pass
        def emit(self, *_args, **_kwargs):
            return None
        def connect(self, *_args, **_kwargs):
            return None

    class _ListWidget(_Dummy):
        def __init__(self, *args, **kwargs):
            self._items = []
        def clear(self):
            self._items.clear()
        def addItem(self, item):
            self._items.append(item)
        def count(self):
            return len(self._items)
        def item(self, i):
            return self._items[i]
        def setItemWidget(self, *_a, **_k):
            return None
        def selectedIndexes(self):
            return []

    qtcore.Qt = types.SimpleNamespace(AlignLeft=0, ScrollBarAlwaysOff=0, MouseButtonPress=1, MouseMove=2, MouseButtonRelease=3)
    qtcore.QEvent = object
    qtcore.QPoint = object
    qtcore.QTimer = _Dummy
    qtcore.QThread = _QThread
    qtcore.pyqtSignal = lambda *_a, **_k: _Signal()
    qtwidgets.QApplication = _Dummy
    qtwidgets.QCheckBox = _Dummy
    qtwidgets.QDialog = _Dummy
    qtwidgets.QFrame = _Dummy
    qtwidgets.QHBoxLayout = _Dummy
    qtwidgets.QLabel = _Dummy
    qtwidgets.QListWidget = _ListWidget
    qtwidgets.QListWidgetItem = _Dummy
    qtwidgets.QMenu = _Dummy
    qtwidgets.QMessageBox = _Dummy
    qtwidgets.QPushButton = _Dummy
    qtwidgets.QSizePolicy = types.SimpleNamespace(Expanding=0, Preferred=0)
    qtwidgets.QSlider = _Dummy
    qtwidgets.QVBoxLayout = _Dummy
    qtwidgets.QWidget = _Dummy
    pyqt5 = types.ModuleType("PyQt5")
    pyqt5.QtCore = qtcore
    pyqt5.QtWidgets = qtwidgets
    sys.modules["PyQt5"] = pyqt5
    sys.modules["PyQt5.QtCore"] = qtcore
    sys.modules["PyQt5.QtWidgets"] = qtwidgets

sys.path.append(str(Path(__file__).resolve().parents[2]))

from monitoring.runtime_helpers import (
    classify_camera_setting_changes,
    evaluate_heartbeat_health,
    evaluate_overload_transition,
    worker_stop_timeout_details,
)
from monitoring.log_messages import (
    PERFORMANCE_PARAM_LABELS,
    format_dict_multiline,
    summarize_performance_details,
)
from monitoring.widgets.logs import LogWindow, format_log_entry_for_clipboard


class _FakeHistoryWriter:
    def __init__(self) -> None:
        self.scheduled = 0
        self.flushed = 0
        self.last_payload = None

    def schedule(self, payload):
        self.scheduled += 1
        self.last_payload = list(payload)

    def flush(self):
        self.flushed += 1
        if self.last_payload is None:
            return
        Path(self.path).write_text(json.dumps(self.last_payload, indent=2), encoding="utf-8")


def _build_headless_log_window(tmp_path):
    window = LogWindow.__new__(LogWindow)
    window.log_path = str(tmp_path / "logs.json")
    window.retention_hours = 24 * 365
    window.history = []
    window._selected_rows = set()
    window.VISIBLE_HISTORY_LIMIT = 200
    window._io_writer = _FakeHistoryWriter()
    window._io_writer.path = window.log_path
    window._history_writer = window._io_writer
    window._refresh_count = 0
    window._append_count = 0
    window._refresh_widget = lambda: setattr(window, "_refresh_count", window._refresh_count + 1)
    window._append_incremental = lambda _entry: setattr(window, "_append_count", window._append_count + 1)
    return window


def test_log_window_accepts_warning_group(tmp_path):
    window = _build_headless_log_window(tmp_path)
    entry_id = window.add_structured_entry({"group": "warning", "action": "camera stop timeout"})
    assert entry_id
    assert window.history[-1]["group"] == "warning"


def test_log_window_preserves_backward_compat_entries(tmp_path):
    window = _build_headless_log_window(tmp_path)
    import datetime
    legacy_ts = datetime.datetime.now().strftime("%A %H:%M:%S %Y-%m-%d")
    legacy = [{"group": "application", "timestamp": legacy_ts, "action": "legacy"}]
    window.history = window._retention_filtered(legacy)
    assert len(window.history) == 1
    assert window.history[0]["level"] == "INFO"


def test_structured_log_entry_serialization(tmp_path):
    window = _build_headless_log_window(tmp_path)
    window.add_structured_entry({"group": "error", "action": "boom", "details": "x", "traceback": "tb", "source": "test", "level": "CRITICAL"})
    window.flush_history_persist()
    data = json.loads(Path(window.log_path).read_text(encoding="utf-8"))
    assert data[0]["source"] == "test"
    assert data[0]["traceback"] == "tb"


def test_log_entry_formatter_builds_full_copy_text():
    text = format_log_entry_for_clipboard({
        "timestamp": "Thursday 12:34:56 2026-03-19",
        "group": "error",
        "level": "ERROR",
        "source": "worker",
        "camera": "Camera 1",
        "action": "Błąd połączenia",
        "details": "Nie udało się otworzyć strumienia",
        "traceback": "Traceback line 1\nline 2",
    })
    assert "Czas: Thursday 12:34:56 2026-03-19" in text
    assert "Kamera: Camera 1" in text
    assert "Akcja: Błąd połączenia" in text
    assert "Szczegóły: Nie udało się otworzyć strumienia" in text
    assert "Traceback:\nTraceback line 1\nline 2" in text


def test_format_dict_multiline_builds_one_line_per_param():
    text = format_dict_multiline(
        {"capture_fps": "12.50", "queue_size": 3},
        PERFORMANCE_PARAM_LABELS,
    )
    lines = text.splitlines()
    assert lines == ["FPS przechwytywania: 12.50", "Rozmiar kolejki: 3"]


def test_summarize_performance_details_supports_legacy_single_line_format():
    legacy = "capture_fps=12.50 queue_size=3 cpu_percent=17.0"
    summary = summarize_performance_details(legacy)
    assert "Podsumowanie wydajności" in summary
    assert "FPS przechwytywania: 12.50" in summary
    assert "Rozmiar kolejki: 3" in summary
    assert "CPU [%]: 17.0" in summary


def test_camera_setting_change_logging_helpers_if_extracted():
    changed, restart = classify_camera_setting_changes({"name": "A", "rtsp": "x"}, {"name": "A", "rtsp": "y"}, {"rtsp"})
    assert "rtsp" in changed
    assert restart == ["rtsp"]


def test_worker_heartbeat_timeout_helper_if_extracted():
    stale = evaluate_heartbeat_health({"Cam1": True, "Cam2": False}, {"Cam1": 1.0}, now_ts=20.0, timeout_seconds=10.0)
    assert stale == ["Cam1"]


def test_overload_does_not_activate_below_min_camera_threshold():
    level, _ts, reason = evaluate_overload_transition(
        now_ts=10.0,
        active_camera_count=1,
        gui_load_fps=100.0,
        recording_count=0,
        currently_level=0,
        last_change_ts=0.0,
        protection_enabled=True,
        min_camera_count=2,
        camera_threshold=1,
        load_per_camera_threshold=10.0,
        enter_debounce_seconds=1.0,
        exit_debounce_seconds=1.0,
        ui_render_ms=1.0,
        max_ui_render_ms=10.0,
        queue_size=0,
        max_queue_size=20,
        preview_bandwidth_mbps=1.0,
        max_preview_bandwidth_mbps=10.0,
    )
    assert level == 0
    assert reason == "below-min-camera-threshold"


def test_overload_enter_exit_debounce_helper():
    level, ts, reason = evaluate_overload_transition(
        now_ts=1.0, active_camera_count=4, gui_load_fps=60.0, recording_count=0,
        currently_level=0, last_change_ts=0.0, protection_enabled=True,
        min_camera_count=2, camera_threshold=3, load_per_camera_threshold=10.0,
        enter_debounce_seconds=3.0, exit_debounce_seconds=4.0,
        ui_render_ms=18.0, max_ui_render_ms=14.0,
        queue_size=10, max_queue_size=24,
        preview_bandwidth_mbps=8.0, max_preview_bandwidth_mbps=12.0,
    )
    assert level == 0 and reason == "enter-debounce-pending"

    level, ts, reason = evaluate_overload_transition(
        now_ts=3.5, active_camera_count=4, gui_load_fps=60.0, recording_count=0,
        currently_level=level, last_change_ts=ts, protection_enabled=True,
        min_camera_count=2, camera_threshold=3, load_per_camera_threshold=10.0,
        enter_debounce_seconds=3.0, exit_debounce_seconds=4.0,
        ui_render_ms=18.0, max_ui_render_ms=14.0,
        queue_size=10, max_queue_size=24,
        preview_bandwidth_mbps=8.0, max_preview_bandwidth_mbps=12.0,
    )
    assert level == 2 and reason == "condition-stable-enter-L2"

    level, _ts, reason = evaluate_overload_transition(
        now_ts=5.0, active_camera_count=2, gui_load_fps=0.0, recording_count=0,
        currently_level=level, last_change_ts=ts, protection_enabled=True,
        min_camera_count=2, camera_threshold=3, load_per_camera_threshold=10.0,
        enter_debounce_seconds=3.0, exit_debounce_seconds=4.0,
        ui_render_ms=4.0, max_ui_render_ms=14.0,
        queue_size=0, max_queue_size=24,
        preview_bandwidth_mbps=1.0, max_preview_bandwidth_mbps=12.0,
    )
    assert level == 2 and reason == "exit-debounce-pending"


def test_worker_stop_timeout_log_helper():
    details = worker_stop_timeout_details("Cam-A", 3500)
    assert "Cam-A" in details
    assert "3500" in details


def test_log_window_batched_io_not_per_entry(tmp_path):
    window = _build_headless_log_window(tmp_path)
    for idx in range(50):
        window.add_structured_entry({"group": "application", "action": f"entry-{idx}"})
    assert window._io_writer.scheduled == 50
    assert not Path(window.log_path).exists()
    window.flush_history_persist()
    assert window._io_writer.flushed == 1
    data = json.loads(Path(window.log_path).read_text(encoding="utf-8"))
    assert len(data) == 50


def test_log_window_avoids_full_refresh_for_each_entry(tmp_path):
    window = _build_headless_log_window(tmp_path)
    for idx in range(120):
        window.add_structured_entry({"group": "application", "action": f"entry-{idx}"})
    assert window._append_count == 120
    assert window._refresh_count == 0


def test_overload_levels_hysteresis_and_stabilization():
    level, ts, _ = evaluate_overload_transition(
        now_ts=0.0, active_camera_count=6, gui_load_fps=120.0, recording_count=0,
        currently_level=0, last_change_ts=0.0, protection_enabled=True,
        min_camera_count=2, camera_threshold=3, load_per_camera_threshold=10.0,
        enter_debounce_seconds=2.0, exit_debounce_seconds=3.0,
        ui_render_ms=32.0, max_ui_render_ms=14.0,
        queue_size=40, max_queue_size=24,
        preview_bandwidth_mbps=22.0, max_preview_bandwidth_mbps=12.0,
    )
    assert level == 0

    level, ts, reason = evaluate_overload_transition(
        now_ts=2.2, active_camera_count=6, gui_load_fps=120.0, recording_count=0,
        currently_level=level, last_change_ts=ts, protection_enabled=True,
        min_camera_count=2, camera_threshold=3, load_per_camera_threshold=10.0,
        enter_debounce_seconds=2.0, exit_debounce_seconds=3.0,
        ui_render_ms=32.0, max_ui_render_ms=14.0,
        queue_size=40, max_queue_size=24,
        preview_bandwidth_mbps=22.0, max_preview_bandwidth_mbps=12.0,
    )
    assert level == 3 and reason == "condition-stable-enter-L3"

    level2, ts2, reason2 = evaluate_overload_transition(
        now_ts=3.0, active_camera_count=6, gui_load_fps=30.0, recording_count=0,
        currently_level=level, last_change_ts=ts, protection_enabled=True,
        min_camera_count=2, camera_threshold=3, load_per_camera_threshold=10.0,
        enter_debounce_seconds=2.0, exit_debounce_seconds=3.0,
        ui_render_ms=8.0, max_ui_render_ms=14.0,
        queue_size=4, max_queue_size=24,
        preview_bandwidth_mbps=2.0, max_preview_bandwidth_mbps=12.0,
    )
    assert level2 == 3 and reason2 == "exit-debounce-pending"

    level3, _ts3, reason3 = evaluate_overload_transition(
        now_ts=6.5, active_camera_count=6, gui_load_fps=30.0, recording_count=0,
        currently_level=level2, last_change_ts=ts2, protection_enabled=True,
        min_camera_count=2, camera_threshold=3, load_per_camera_threshold=10.0,
        enter_debounce_seconds=2.0, exit_debounce_seconds=3.0,
        ui_render_ms=8.0, max_ui_render_ms=14.0,
        queue_size=4, max_queue_size=24,
        preview_bandwidth_mbps=2.0, max_preview_bandwidth_mbps=12.0,
    )
    assert level3 == 2 and reason3 == "condition-stable-exit-L2"
