"""Application configuration utilities."""

from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, MutableMapping

from . import PROJECT_ROOT

BASE_DIR = PROJECT_ROOT
ICON_DIR = BASE_DIR / "icons"
CONFIG_PATH = BASE_DIR / "config.json"
MODELS_PATH = BASE_DIR / "models"
ALERTS_HISTORY_PATH = BASE_DIR / "alerts_history.json"
RECORDINGS_CATALOG_PATH = BASE_DIR / "recordings_catalog.json"
LOG_HISTORY_PATH = BASE_DIR / "log_history.json"
LOG_RETENTION_HOURS = 48

VISIBLE_CLASSES = ["person", "car", "cat", "dog", "bird"]
RECORD_CLASSES = ["person", "car", "cat", "dog", "bird"]

DEFAULT_MODEL = "yolov5nu_silu_coco--640x640_float_tflite_multidevice_1"
DEFAULT_FPS = 3
DEFAULT_RTSP_FPS = 0
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_CONFIDENCE_THRESHOLD_DRAW = DEFAULT_CONFIDENCE_THRESHOLD
DEFAULT_CONFIDENCE_THRESHOLD_RECORD = DEFAULT_CONFIDENCE_THRESHOLD
DEFAULT_DRAW_OVERLAYS = True
DEFAULT_ENABLE_DETECTION = True
DEFAULT_ENABLE_RECORDING = True
DEFAULT_DETECTION_HOURS = "00:00-23:59"
DEFAULT_RECORD_PATH = BASE_DIR / "nagrania"
DEFAULT_PRE_SECONDS = 5
DEFAULT_POST_SECONDS = 5
DEFAULT_LOST_SECONDS = 10
DEFAULT_THUMBNAIL_MODE = "first_detection"
DEFAULT_RECORD_START_MODE = "detection_first"
DEFAULT_REQUIRED_HITS_TO_START_RECORDING = 1
DEFAULT_REQUIRED_MISSES_TO_END_DETECTION = 1
DEFAULT_MIN_RECORD_SECONDS = 3
DEFAULT_PREVIEW_FPS_MAIN = 15
DEFAULT_PREVIEW_FPS_THUMB = 3
DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN = True
DEFAULT_PREVIEW_MAIN_MAX_WIDTH = 1280
DEFAULT_PREVIEW_MAIN_MAX_HEIGHT = 720
DEFAULT_PREVIEW_THUMB_MAX_WIDTH = 320
DEFAULT_PREVIEW_THUMB_MAX_HEIGHT = 180
DEFAULT_SHOW_CAMERA_INFO_OVERLAY = True
DEFAULT_THUMBNAIL_OVERLAY_ENABLED = True
DEFAULT_THUMBNAIL_BOX_THICKNESS = 1
DEFAULT_THUMBNAIL_FONT_SCALE = 0.5
DEFAULT_THUMBNAIL_FONT_THICKNESS = 1

DEFAULT_OVERLOAD_PROTECTION_ENABLED = True
DEFAULT_OVERLOAD_MIN_CAMERA_COUNT = 2
DEFAULT_OVERLOAD_CAMERA_COUNT_THRESHOLD = 6
DEFAULT_OVERLOAD_REDUCE_THUMB_PREVIEW_FPS = 1
DEFAULT_OVERLOAD_REDUCE_DETECT_FPS_FACTOR = 0.75
DEFAULT_OVERLOAD_DISABLE_NONESSENTIAL_OVERLAYS = True
DEFAULT_OVERLOAD_ENTER_DEBOUNCE_SECONDS = 3.0
DEFAULT_OVERLOAD_EXIT_DEBOUNCE_SECONDS = 5.0
DEFAULT_OVERLOAD_MAX_UI_RENDER_MS = 14.0
DEFAULT_OVERLOAD_MAX_QUEUE_SIZE = 24
DEFAULT_OVERLOAD_MAX_PREVIEW_BANDWIDTH_MBPS = 12.0


def _resolve_path(value: str | os.PathLike[str] | None, *, default: Path) -> Path:
    """Resolve a path coming from configuration."""
    if not value:
        return default
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate


def fill_camera_defaults(camera: MutableMapping[str, object]) -> MutableMapping[str, object]:
    """Fill missing camera parameters with default values."""
    defaults: Dict[str, object] = {
        "model": DEFAULT_MODEL,
        "fps": DEFAULT_FPS,
        "rtsp_fps": DEFAULT_RTSP_FPS,
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "confidence_threshold_draw": DEFAULT_CONFIDENCE_THRESHOLD_DRAW,
        "confidence_threshold_record": DEFAULT_CONFIDENCE_THRESHOLD_RECORD,
        "draw_overlays": DEFAULT_DRAW_OVERLAYS,
        "enable_detection": DEFAULT_ENABLE_DETECTION,
        "enable_recording": DEFAULT_ENABLE_RECORDING,
        "detection_hours": DEFAULT_DETECTION_HOURS,
        "visible_classes": list(VISIBLE_CLASSES),
        "record_classes": list(RECORD_CLASSES),
        "record_path": str(DEFAULT_RECORD_PATH),
        "pre_seconds": DEFAULT_PRE_SECONDS,
        "post_seconds": DEFAULT_POST_SECONDS,
        "lost_seconds": DEFAULT_LOST_SECONDS,
        "thumbnail_mode": DEFAULT_THUMBNAIL_MODE,
        "record_start_mode": DEFAULT_RECORD_START_MODE,
        "required_hits_to_start_recording": DEFAULT_REQUIRED_HITS_TO_START_RECORDING,
        "required_misses_to_end_detection": DEFAULT_REQUIRED_MISSES_TO_END_DETECTION,
        "min_record_seconds": DEFAULT_MIN_RECORD_SECONDS,
        "preview_fps_main": DEFAULT_PREVIEW_FPS_MAIN,
        "preview_fps_thumb": DEFAULT_PREVIEW_FPS_THUMB,
        "preview_pause_when_hidden": DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN,
        "preview_main_max_width": DEFAULT_PREVIEW_MAIN_MAX_WIDTH,
        "preview_main_max_height": DEFAULT_PREVIEW_MAIN_MAX_HEIGHT,
        "preview_thumb_max_width": DEFAULT_PREVIEW_THUMB_MAX_WIDTH,
        "preview_thumb_max_height": DEFAULT_PREVIEW_THUMB_MAX_HEIGHT,
        "show_camera_info_overlay": DEFAULT_SHOW_CAMERA_INFO_OVERLAY,
        "thumbnail_overlay_enabled": DEFAULT_THUMBNAIL_OVERLAY_ENABLED,
        "thumbnail_box_thickness": DEFAULT_THUMBNAIL_BOX_THICKNESS,
        "thumbnail_font_scale": DEFAULT_THUMBNAIL_FONT_SCALE,
        "thumbnail_font_thickness": DEFAULT_THUMBNAIL_FONT_THICKNESS,
        "type": "rtsp",
    }

    legacy_confidence = camera.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    if "confidence_threshold_draw" not in camera:
        camera["confidence_threshold_draw"] = legacy_confidence
    if "confidence_threshold_record" not in camera:
        camera["confidence_threshold_record"] = legacy_confidence

    for key, value in defaults.items():
        camera.setdefault(key, value)

    # ``record_path`` can be provided as a relative path in the configuration
    # file.  Normalise it so the rest of the application always works with an
    # absolute location rooted at :data:`BASE_DIR`.
    record_path = _resolve_path(camera.get("record_path"), default=DEFAULT_RECORD_PATH)
    camera["record_path"] = str(record_path)
    return camera


def list_usb_cameras() -> List[tuple[int, str]]:
    """Return a list of available USB cameras as ``(index, name)`` tuples."""
    devices: List[tuple[int, str]] = []
    for dev in sorted(glob("/dev/video*")):
        try:
            idx = int(Path(dev).name.replace("video", ""))
        except ValueError:
            continue
        name_path = Path(f"/sys/class/video4linux/video{idx}/name")
        try:
            name = name_path.read_text().strip()
        except OSError:
            name = f"Kamera {idx}"
        devices.append((idx, name))
    return devices


def load_config(path: Path | None = None) -> Dict[str, object]:
    """Load the application configuration."""
    global LOG_HISTORY_PATH, LOG_RETENTION_HOURS

    cfg_path = path or CONFIG_PATH
    if not cfg_path.exists():
        cfg: Dict[str, object] = {
            "log_history_path": str(LOG_HISTORY_PATH),
            "log_retention_hours": LOG_RETENTION_HOURS,
            "cameras": [
                {
                    "name": "kamera1",
                    "rtsp": "rtsp://admin:IBLTSQ@192.168.8.165:554",
                }
            ],
        }
    else:
        with cfg_path.open("r", encoding="utf-8") as handle:
            cfg = json.load(handle)

    LOG_HISTORY_PATH = _resolve_path(cfg.get("log_history_path"), default=LOG_HISTORY_PATH)
    LOG_RETENTION_HOURS = int(cfg.get("log_retention_hours", LOG_RETENTION_HOURS))
    cfg.setdefault("overload_protection_enabled", DEFAULT_OVERLOAD_PROTECTION_ENABLED)
    cfg.setdefault("overload_min_camera_count", DEFAULT_OVERLOAD_MIN_CAMERA_COUNT)
    cfg.setdefault("overload_camera_count_threshold", DEFAULT_OVERLOAD_CAMERA_COUNT_THRESHOLD)
    cfg.setdefault("overload_reduce_thumb_preview_fps", DEFAULT_OVERLOAD_REDUCE_THUMB_PREVIEW_FPS)
    cfg.setdefault("overload_reduce_detect_fps_factor", DEFAULT_OVERLOAD_REDUCE_DETECT_FPS_FACTOR)
    cfg.setdefault("overload_disable_nonessential_overlays", DEFAULT_OVERLOAD_DISABLE_NONESSENTIAL_OVERLAYS)
    cfg.setdefault("overload_enter_debounce_seconds", DEFAULT_OVERLOAD_ENTER_DEBOUNCE_SECONDS)
    cfg.setdefault("overload_exit_debounce_seconds", DEFAULT_OVERLOAD_EXIT_DEBOUNCE_SECONDS)
    cfg.setdefault("overload_max_ui_render_ms", DEFAULT_OVERLOAD_MAX_UI_RENDER_MS)
    cfg.setdefault("overload_max_queue_size", DEFAULT_OVERLOAD_MAX_QUEUE_SIZE)
    cfg.setdefault("overload_max_preview_bandwidth_mbps", DEFAULT_OVERLOAD_MAX_PREVIEW_BANDWIDTH_MBPS)

    for camera in cfg.get("cameras", []):
        if isinstance(camera, MutableMapping):
            fill_camera_defaults(camera)
    return cfg


def save_config(config: MutableMapping[str, object], path: Path | None = None) -> None:
    """Persist configuration to disk."""
    global LOG_HISTORY_PATH, LOG_RETENTION_HOURS

    for camera in config.get("cameras", []):
        if isinstance(camera, MutableMapping):
            fill_camera_defaults(camera)
    config.setdefault("log_history_path", str(LOG_HISTORY_PATH))
    config.setdefault("log_retention_hours", LOG_RETENTION_HOURS)

    cfg_path = path or CONFIG_PATH
    cfg_path.write_text(json.dumps(config, indent=4), encoding="utf-8")


__all__ = [
    "ALERTS_HISTORY_PATH",
    "CONFIG_PATH",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_CONFIDENCE_THRESHOLD_DRAW",
    "DEFAULT_CONFIDENCE_THRESHOLD_RECORD",
    "DEFAULT_DETECTION_HOURS",
    "DEFAULT_DRAW_OVERLAYS",
    "DEFAULT_ENABLE_DETECTION",
    "DEFAULT_ENABLE_RECORDING",
    "DEFAULT_FPS",
    "DEFAULT_RTSP_FPS",
    "DEFAULT_LOST_SECONDS",
    "DEFAULT_MODEL",
    "DEFAULT_POST_SECONDS",
    "DEFAULT_PRE_SECONDS",
    "DEFAULT_RECORD_PATH",
    "DEFAULT_RECORD_START_MODE",
    "DEFAULT_REQUIRED_HITS_TO_START_RECORDING",
    "DEFAULT_REQUIRED_MISSES_TO_END_DETECTION",
    "DEFAULT_MIN_RECORD_SECONDS",
    "DEFAULT_PREVIEW_FPS_MAIN",
    "DEFAULT_PREVIEW_FPS_THUMB",
    "DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN",
    "DEFAULT_PREVIEW_MAIN_MAX_WIDTH",
    "DEFAULT_PREVIEW_MAIN_MAX_HEIGHT",
    "DEFAULT_PREVIEW_THUMB_MAX_WIDTH",
    "DEFAULT_PREVIEW_THUMB_MAX_HEIGHT",
    "DEFAULT_SHOW_CAMERA_INFO_OVERLAY",
    "DEFAULT_OVERLOAD_PROTECTION_ENABLED",
    "DEFAULT_OVERLOAD_MIN_CAMERA_COUNT",
    "DEFAULT_OVERLOAD_CAMERA_COUNT_THRESHOLD",
    "DEFAULT_OVERLOAD_REDUCE_THUMB_PREVIEW_FPS",
    "DEFAULT_OVERLOAD_REDUCE_DETECT_FPS_FACTOR",
    "DEFAULT_OVERLOAD_DISABLE_NONESSENTIAL_OVERLAYS",
    "DEFAULT_OVERLOAD_ENTER_DEBOUNCE_SECONDS",
    "DEFAULT_OVERLOAD_EXIT_DEBOUNCE_SECONDS",
    "DEFAULT_OVERLOAD_MAX_UI_RENDER_MS",
    "DEFAULT_OVERLOAD_MAX_QUEUE_SIZE",
    "DEFAULT_OVERLOAD_MAX_PREVIEW_BANDWIDTH_MBPS",
    "DEFAULT_THUMBNAIL_OVERLAY_ENABLED",
    "DEFAULT_THUMBNAIL_BOX_THICKNESS",
    "DEFAULT_THUMBNAIL_FONT_SCALE",
    "DEFAULT_THUMBNAIL_FONT_THICKNESS",
    "DEFAULT_THUMBNAIL_MODE",
    "ICON_DIR",
    "LOG_HISTORY_PATH",
    "LOG_RETENTION_HOURS",
    "MODELS_PATH",
    "RECORDINGS_CATALOG_PATH",
    "RECORD_CLASSES",
    "VISIBLE_CLASSES",
    "fill_camera_defaults",
    "list_usb_cameras",
    "load_config",
    "save_config",
]
