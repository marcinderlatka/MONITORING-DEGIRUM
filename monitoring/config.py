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
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_DRAW_OVERLAYS = True
DEFAULT_ENABLE_DETECTION = True
DEFAULT_ENABLE_RECORDING = True
DEFAULT_DETECTION_HOURS = "00:00-23:59"
DEFAULT_RECORD_PATH = BASE_DIR / "nagrania"
DEFAULT_PRE_SECONDS = 5
DEFAULT_POST_SECONDS = 5
DEFAULT_LOST_SECONDS = 10


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
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
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
        "type": "rtsp",
    }
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
    "DEFAULT_DETECTION_HOURS",
    "DEFAULT_DRAW_OVERLAYS",
    "DEFAULT_ENABLE_DETECTION",
    "DEFAULT_ENABLE_RECORDING",
    "DEFAULT_FPS",
    "DEFAULT_LOST_SECONDS",
    "DEFAULT_MODEL",
    "DEFAULT_POST_SECONDS",
    "DEFAULT_PRE_SECONDS",
    "DEFAULT_RECORD_PATH",
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
