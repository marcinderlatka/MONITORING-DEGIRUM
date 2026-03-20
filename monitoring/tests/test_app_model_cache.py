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
