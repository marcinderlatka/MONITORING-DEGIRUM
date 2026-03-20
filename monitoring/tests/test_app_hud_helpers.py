from __future__ import annotations

import sys
from pathlib import Path
import types

import pytest

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace(FONT_HERSHEY_PLAIN=0)
if "degirum_tools" not in sys.modules:
    sys.modules["degirum_tools"] = types.SimpleNamespace()

sys.path.append(str(Path(__file__).resolve().parents[2]))

app_mod = pytest.importorskip("monitoring.app", reason="PyQt runtime unavailable in test environment", exc_type=ImportError)


def test_camera_hud_lines_are_polish_labels_if_helper_is_extractable():
    fake = types.SimpleNamespace(
        cameras=[{"name": "Brama"}],
        _last_status={0: "Połączono"},
        _last_error={},
        _last_fps_text={0: "12.4 fps"},
        worker_status={"Brama": {"stream_fps": 15.0, "detect_fps": 3.0, "writer_fps": 5.0, "queue_size": 0, "dropped_frames": 0}},
    )
    lines = app_mod.MainWindow._build_camera_hud_lines(fake, 0)
    joined = "\n".join(lines)
    assert "Kamera:" in joined
    assert "Status:" in joined
    assert "Strumień FPS:" in joined
    assert "Pominięte klatki:" in joined


def test_settings_change_triggers_hud_refresh_if_helper_is_extractable():
    refreshed = {"count": 0}

    fake_worker = types.SimpleNamespace(isRunning=lambda: True, apply_runtime_settings=lambda *_a, **_k: None)
    fake = types.SimpleNamespace(
        workers=[fake_worker],
        cameras=[{"name": "Brama"}],
        _requires_worker_restart=lambda changed, *_a: (False, []),
        _refresh_camera_hud=lambda *_a, **_k: refreshed.__setitem__("count", refreshed["count"] + 1),
        _maybe_restart_camera_after_settings=lambda *_a, **_k: False,
        _log_info=lambda *_a, **_k: None,
    )

    result = app_mod.MainWindow._apply_camera_settings_change(fake, 0, {"name": "Brama", "fps": 10}, {"name": "Brama", "fps": 12})
    assert result["applied_live"] is True
    assert refreshed["count"] == 1


def test_device_override_change_is_classified_as_restart_not_live():
    refreshed = {"count": 0}
    restart_calls: list[tuple[int, bool, bool, list[str]]] = []
    restart_all = {"count": 0}
    runtime_apply = {"count": 0}

    fake_worker = types.SimpleNamespace(
        isRunning=lambda: True,
        apply_runtime_settings=lambda *_a, **_k: runtime_apply.__setitem__("count", runtime_apply["count"] + 1),
    )
    fake = types.SimpleNamespace(
        workers=[fake_worker],
        cameras=[{"name": "Brama"}],
        _requires_worker_restart=lambda changed, old, new: app_mod.MainWindow._requires_worker_restart(fake, changed, old, new),
        _refresh_camera_hud=lambda *_a, **_k: refreshed.__setitem__("count", refreshed["count"] + 1),
        _maybe_restart_camera_after_settings=lambda idx, was_running, requires_restart, restart_reason_keys: (
            restart_calls.append((idx, was_running, requires_restart, restart_reason_keys.copy())) or True
        ),
        _log_info=lambda *_a, **_k: None,
        restart_workers_and_ui=lambda: restart_all.__setitem__("count", restart_all["count"] + 1),
    )

    old = {"name": "Brama", "degirum_device_override_enabled": False, "degirum_device_override": "inherit"}
    new = {"name": "Brama", "degirum_device_override_enabled": True, "degirum_device_override": "gpu"}
    result = app_mod.MainWindow._apply_camera_settings_change(fake, 0, old, new)

    assert result["applied_live"] is False
    assert result["restarted"] is True
    assert set(result["restart_reason_keys"]) == {"degirum_device_override_enabled", "degirum_device_override"}
    assert len(restart_calls) == 1
    assert restart_calls[0][0] == 0
    assert restart_calls[0][1] is True
    assert restart_calls[0][2] is True
    assert set(restart_calls[0][3]) == {"degirum_device_override_enabled", "degirum_device_override"}
    assert runtime_apply["count"] == 0
    assert restart_all["count"] == 0
    assert refreshed["count"] == 1


def test_requires_worker_restart_marks_degirum_override_fields():
    fake = types.SimpleNamespace()
    changed_keys = ["fps", "degirum_device_override_enabled", "degirum_device_override"]

    requires_restart, restart_keys = app_mod.MainWindow._requires_worker_restart(fake, changed_keys, {}, {})

    assert requires_restart is True
    assert set(restart_keys) == {"degirum_device_override_enabled", "degirum_device_override"}
