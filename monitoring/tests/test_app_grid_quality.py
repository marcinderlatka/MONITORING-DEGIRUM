from __future__ import annotations

import sys
from pathlib import Path
import types

import numpy as np
import pytest

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace(FONT_HERSHEY_PLAIN=0)
if "degirum_tools" not in sys.modules:
    sys.modules["degirum_tools"] = types.SimpleNamespace()

sys.path.append(str(Path(__file__).resolve().parents[2]))

app_mod = pytest.importorskip("monitoring.app", reason="PyQt runtime unavailable in test environment", exc_type=ImportError)


class _Size:
    def __init__(self, w: int, h: int):
        self._w = w
        self._h = h

    def width(self) -> int:
        return self._w

    def height(self) -> int:
        return self._h


class _Label:
    def __init__(self, w: int, h: int, dpr: float = 1.0):
        self._size = _Size(w, h)
        self._dpr = dpr

    def size(self) -> _Size:
        return self._size

    def devicePixelRatioF(self) -> float:
        return self._dpr


def _fake_window(*, main_source: np.ndarray | None, thumb_source: np.ndarray | None, hq: str = "normal", overload_level: int = 0):
    grid_item = types.SimpleNamespace(frame_label=_Label(960, 540, 1.25))
    return types.SimpleNamespace(
        _last_main_frame={0: main_source} if main_source is not None else {},
        _last_thumb_frame={0: thumb_source} if thumb_source is not None else {},
        camera_grid=types.SimpleNamespace(isVisible=lambda: True, items=[grid_item]),
        grid_preview_quality=hq,
        preview_grid_max_width=1280,
        preview_grid_max_height=720,
        overload_mode_active=overload_level > 0,
        overload_level=overload_level,
        worker_status={"Cam 1": {}},
        cameras=[{"name": "Cam 1"}],
    )


def test_resolve_grid_render_params_prefers_main_for_hq_without_critical_overload():
    main = np.zeros((720, 1280, 3), dtype=np.uint8)
    thumb = np.zeros((180, 320, 3), dtype=np.uint8)
    fake = _fake_window(main_source=main, thumb_source=thumb, hq="high-quality", overload_level=1)

    source, width, height, dpr, source_tag = app_mod.MainWindow._resolve_grid_render_params(fake, 0)

    assert source is main
    assert source_tag == "main-overload"
    assert width <= 1280
    assert height <= 720
    assert dpr == 1.25


def test_resolve_grid_render_params_falls_back_to_thumb_on_critical_overload():
    main = np.zeros((720, 1280, 3), dtype=np.uint8)
    thumb = np.zeros((180, 320, 3), dtype=np.uint8)
    fake = _fake_window(main_source=main, thumb_source=thumb, hq="high-quality", overload_level=2)

    source, _width, _height, _dpr, source_tag = app_mod.MainWindow._resolve_grid_render_params(fake, 0)

    assert source is thumb
    assert source_tag == "thumb-overload"


def test_resolve_grid_render_params_hq_uses_thumb_tag_when_main_missing():
    thumb = np.zeros((180, 320, 3), dtype=np.uint8)
    fake = _fake_window(main_source=None, thumb_source=thumb, hq="high-quality", overload_level=0)

    source, width, height, _dpr, source_tag = app_mod.MainWindow._resolve_grid_render_params(fake, 0)

    assert source is thumb
    assert source_tag == "thumb-hq"
    assert width == 1280
    assert height == 720


def test_grid_target_fps_high_quality_and_overload_modes():
    fake = types.SimpleNamespace(
        preview_fps_grid=8.0,
        preview_fps_main=20.0,
        camera_grid=types.SimpleNamespace(isVisible=lambda: True),
        grid_preview_quality="high-quality",
        overload_mode_active=False,
        overload_level=0,
    )

    assert app_mod.MainWindow._grid_target_fps(fake) == 20.0

    fake.overload_mode_active = True
    fake.overload_level = 1
    assert app_mod.MainWindow._grid_target_fps(fake) == 20.0

    fake.overload_level = 3
    assert app_mod.MainWindow._grid_target_fps(fake) < 20.0
