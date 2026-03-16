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
from monitoring.widgets.logs import LogWindow


def _build_headless_log_window(tmp_path):
    window = LogWindow.__new__(LogWindow)
    window.log_path = str(tmp_path / "logs.json")
    window.retention_hours = 24 * 365
    window.history = []
    window._refresh_widget = lambda: None
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
    data = json.loads(Path(window.log_path).read_text(encoding="utf-8"))
    assert data[0]["source"] == "test"
    assert data[0]["traceback"] == "tb"


def test_camera_setting_change_logging_helpers_if_extracted():
    changed, restart = classify_camera_setting_changes({"name": "A", "rtsp": "x"}, {"name": "A", "rtsp": "y"}, {"rtsp"})
    assert "rtsp" in changed
    assert restart == ["rtsp"]


def test_worker_heartbeat_timeout_helper_if_extracted():
    stale = evaluate_heartbeat_health({"Cam1": True, "Cam2": False}, {"Cam1": 1.0}, now_ts=20.0, timeout_seconds=10.0)
    assert stale == ["Cam1"]


def test_overload_does_not_activate_below_min_camera_threshold():
    active, _ts, reason = evaluate_overload_transition(
        now_ts=10.0,
        active_camera_count=1,
        gui_load_fps=100.0,
        recording_count=0,
        currently_active=False,
        last_change_ts=0.0,
        protection_enabled=True,
        min_camera_count=2,
        camera_threshold=1,
        load_per_camera_threshold=10.0,
        enter_debounce_seconds=1.0,
        exit_debounce_seconds=1.0,
    )
    assert active is False
    assert reason == "below-min-camera-threshold"


def test_overload_enter_exit_debounce_helper():
    active, ts, reason = evaluate_overload_transition(
        now_ts=1.0, active_camera_count=4, gui_load_fps=90.0, recording_count=0,
        currently_active=False, last_change_ts=0.0, protection_enabled=True,
        min_camera_count=2, camera_threshold=3, load_per_camera_threshold=10.0,
        enter_debounce_seconds=3.0, exit_debounce_seconds=4.0,
    )
    assert active is False and reason == "enter-debounce-pending"

    active, ts, reason = evaluate_overload_transition(
        now_ts=3.5, active_camera_count=4, gui_load_fps=90.0, recording_count=0,
        currently_active=active, last_change_ts=ts, protection_enabled=True,
        min_camera_count=2, camera_threshold=3, load_per_camera_threshold=10.0,
        enter_debounce_seconds=3.0, exit_debounce_seconds=4.0,
    )
    assert active is True and reason == "condition-stable-enter"

    active, _ts, reason = evaluate_overload_transition(
        now_ts=5.0, active_camera_count=2, gui_load_fps=0.0, recording_count=0,
        currently_active=active, last_change_ts=ts, protection_enabled=True,
        min_camera_count=2, camera_threshold=3, load_per_camera_threshold=10.0,
        enter_debounce_seconds=3.0, exit_debounce_seconds=4.0,
    )
    assert active is True and reason == "exit-debounce-pending"


def test_worker_stop_timeout_log_helper():
    details = worker_stop_timeout_details("Cam-A", 3500)
    assert "Cam-A" in details
    assert "3500" in details
