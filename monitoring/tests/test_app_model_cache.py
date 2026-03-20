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


def _fake_window() -> types.SimpleNamespace:
    entries: list[tuple[str, str]] = []
    fake = types.SimpleNamespace(
        model_cache={},
        config={},
        log_window=types.SimpleNamespace(add_entry=lambda group, message: entries.append((group, message))),
        _entries=entries,
    )
    fake._build_model_cache_key = lambda model_name, device_config=None: app_mod.MainWindow._build_model_cache_key(
        fake, model_name, device_config
    )
    return fake


def test_build_model_cache_key_is_deterministic_for_dict_config() -> None:
    fake = _fake_window()
    cfg_a = {"device_type": "GPU", "degirum_device_override_enabled": False}
    cfg_b = {"degirum_device_override_enabled": False, "device_type": "gpu"}

    key_a = app_mod.MainWindow._build_model_cache_key(fake, " model-x ", cfg_a)
    key_b = app_mod.MainWindow._build_model_cache_key(fake, "model-x", cfg_b)

    assert key_a == ("model-x", "gpu")
    assert key_a == key_b


def test_get_model_keeps_separate_cpu_gpu_cache_entries_on_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_window()
    calls: list[dict[str, object]] = []

    def _load_model(**kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("device_type") == "gpu":
            raise RuntimeError("gpu unavailable")
        return {"loaded_on": kwargs.get("device_type", "auto")}

    monkeypatch.setattr(app_mod.dg, "load_model", _load_model)

    model = app_mod.MainWindow._get_model(fake, "demo-model", {"device_type": "gpu"})

    gpu_key = app_mod.MainWindow._build_model_cache_key(fake, "demo-model", "gpu")
    cpu_key = app_mod.MainWindow._build_model_cache_key(fake, "demo-model", "cpu")

    assert model["loaded_on"] == "cpu"
    assert gpu_key not in fake.model_cache
    assert cpu_key in fake.model_cache

    cached = app_mod.MainWindow._get_model(fake, "demo-model", {"device_type": "cpu"})
    assert cached is fake.model_cache[cpu_key]
    assert len(calls) == 2


def test_resolve_effective_degirum_device_prefers_camera_override() -> None:
    fake = _fake_window()
    fake.config = {
        "degirum_device_mode": "cpu",
        "degirum_preferred_device": "gpu",
        "degirum_available_devices": ["cpu", "gpu"],
    }
    cam = {"name": "Cam-1", "degirum_device_override_enabled": True, "degirum_device_override": "gpu"}

    selected = app_mod.MainWindow._resolve_effective_degirum_device(fake, cam)

    assert selected == "gpu"
    assert any("per-camera override -> gpu" in msg for _, msg in fake._entries)


def test_resolve_effective_degirum_device_uses_global_manual_mode() -> None:
    fake = _fake_window()
    fake.config = {
        "degirum_device_mode": "gpu",
        "degirum_preferred_device": "cpu",
        "degirum_available_devices": ["cpu", "gpu"],
    }

    selected = app_mod.MainWindow._resolve_effective_degirum_device(fake, {"name": "Cam-2"})

    assert selected == "gpu"
    assert any("global manual mode -> gpu" in msg for _, msg in fake._entries)


def test_resolve_effective_degirum_device_uses_auto_recommendation() -> None:
    fake = _fake_window()
    fake.config = {
        "degirum_device_mode": "auto",
        "degirum_preferred_device": "gpu",
        "degirum_available_devices": ["cpu", "gpu"],
    }

    selected = app_mod.MainWindow._resolve_effective_degirum_device(fake, {"name": "Cam-3"})

    assert selected == "gpu"
    assert any("global auto + recommendation -> gpu" in msg for _, msg in fake._entries)


def test_resolve_effective_degirum_device_forces_cpu_when_selected_device_missing() -> None:
    fake = _fake_window()
    fake.config = {
        "degirum_device_mode": "gpu",
        "degirum_preferred_device": "gpu",
        "degirum_available_devices": ["cpu"],
    }

    selected = app_mod.MainWindow._resolve_effective_degirum_device(fake, {"name": "Cam-4"})

    assert selected == "cpu"
    assert any(group == "warning" and "forcing cpu" in msg for group, msg in fake._entries)
