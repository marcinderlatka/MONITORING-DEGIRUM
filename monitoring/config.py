"""Application configuration utilities."""

from __future__ import annotations

import json
import os
import re
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
DEFAULT_DETECTION_FPS_LIMIT = 8
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
DEFAULT_RECORDING_BACKEND = "current"
DEFAULT_REQUIRED_HITS_TO_START_RECORDING = 1
DEFAULT_REQUIRED_MISSES_TO_END_DETECTION = 3
DEFAULT_MIN_RECORD_SECONDS = 3
DEFAULT_SENSITIVITY_PROFILE = "balanced"
DEFAULT_DEGIRUM_DEVICE_MODE = "auto"
DEFAULT_DEGIRUM_PREFERRED_DEVICE = "auto"
DEFAULT_DEGIRUM_AUTO_SELECT_BEST = True
DEFAULT_DEGIRUM_AVAILABLE_DEVICES: list[str] = []
DEFAULT_DEGIRUM_LAST_BENCHMARK: dict[str, object] = {}
DEFAULT_DEGIRUM_DEVICE_OVERRIDE_ENABLED = False
DEFAULT_DEGIRUM_DEVICE_OVERRIDE = "inherit"

SENSITIVITY_PROFILES = {
    "high_recall": {
        "confidence_threshold_draw": 0.35,
        "confidence_threshold_record": 0.40,
        "required_hits_to_start_recording": 1,
        "required_misses_to_end_detection": 4,
        "min_record_seconds": 6,
    },
    "balanced": {
        "confidence_threshold_draw": DEFAULT_CONFIDENCE_THRESHOLD_DRAW,
        "confidence_threshold_record": DEFAULT_CONFIDENCE_THRESHOLD_RECORD,
        "required_hits_to_start_recording": DEFAULT_REQUIRED_HITS_TO_START_RECORDING,
        "required_misses_to_end_detection": DEFAULT_REQUIRED_MISSES_TO_END_DETECTION,
        "min_record_seconds": DEFAULT_MIN_RECORD_SECONDS,
    },
    "high_precision": {
        "confidence_threshold_draw": 0.60,
        "confidence_threshold_record": 0.70,
        "required_hits_to_start_recording": 3,
        "required_misses_to_end_detection": 1,
        "min_record_seconds": 2,
    },
}
DEFAULT_PREVIEW_FPS_MAIN = 12
DEFAULT_PREVIEW_FPS_THUMB = 2
DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN = True
DEFAULT_PREVIEW_MAIN_MAX_WIDTH = 768
DEFAULT_PREVIEW_MAIN_MAX_HEIGHT = 432
DEFAULT_PREVIEW_THUMB_MAX_WIDTH = 256
DEFAULT_PREVIEW_THUMB_MAX_HEIGHT = 144
DEFAULT_SHOW_CAMERA_INFO_OVERLAY = True
DEFAULT_CAMERA_INFO_OVERLAY_ALPHA = 153
DEFAULT_THUMBNAIL_OVERLAY_ENABLED = True
DEFAULT_THUMBNAIL_BOX_THICKNESS = 1
DEFAULT_THUMBNAIL_FONT_SCALE = 0.5
DEFAULT_THUMBNAIL_FONT_THICKNESS = 1
DEFAULT_CAMERA_PRIORITY = "normal"
CAMERA_PRIORITIES = ("high", "normal", "low")
DEFAULT_PREVIEW_FPS_GRID = 3
DEFAULT_PREVIEW_GRID_MAX_WIDTH = 512
DEFAULT_PREVIEW_GRID_MAX_HEIGHT = 288
DEFAULT_GRID_PREVIEW_QUALITY = "normal"

DEFAULT_OVERLOAD_PROTECTION_ENABLED = True
DEFAULT_OVERLOAD_MIN_CAMERA_COUNT = 2
DEFAULT_OVERLOAD_CAMERA_COUNT_THRESHOLD = 6
DEFAULT_OVERLOAD_REDUCE_THUMB_PREVIEW_FPS = 1
DEFAULT_OVERLOAD_REDUCE_DETECT_FPS_FACTOR = 0.75
DEFAULT_OVERLOAD_DISABLE_NONESSENTIAL_OVERLAYS = True
DEFAULT_OVERLAY_TEXT_ENABLED = True
DEFAULT_OVERLAY_DRAW_EVERY_N = 2
DEFAULT_OVERLOAD_ENTER_DEBOUNCE_SECONDS = 3.0
DEFAULT_OVERLOAD_EXIT_DEBOUNCE_SECONDS = 5.0
DEFAULT_OVERLOAD_MAX_UI_RENDER_MS = 14.0
DEFAULT_OVERLOAD_MAX_QUEUE_SIZE = 24
DEFAULT_OVERLOAD_MAX_PREVIEW_BANDWIDTH_MBPS = 12.0
DEFAULT_QUALITY_PERFORMANCE_PRESET = "economy_monitoring"
QUALITY_PERFORMANCE_PRESETS = {
    "quality_monitoring": {
        "label": "Monitoring jakościowy",
        "preview_fps_main": 20.0,
        "preview_fps_grid": 10.0,
        "preview_fps_thumb": 4.0,
        "preview_main_max_width": 1600,
        "preview_main_max_height": 900,
        "preview_grid_max_width": 1280,
        "preview_grid_max_height": 720,
        "preview_thumb_max_width": 384,
        "preview_thumb_max_height": 216,
    },
    "balanced": {
        "label": "Zbalansowany",
        "preview_fps_main": float(DEFAULT_PREVIEW_FPS_MAIN),
        "preview_fps_grid": float(DEFAULT_PREVIEW_FPS_GRID),
        "preview_fps_thumb": float(DEFAULT_PREVIEW_FPS_THUMB),
        "preview_main_max_width": int(DEFAULT_PREVIEW_MAIN_MAX_WIDTH),
        "preview_main_max_height": int(DEFAULT_PREVIEW_MAIN_MAX_HEIGHT),
        "preview_grid_max_width": int(DEFAULT_PREVIEW_GRID_MAX_WIDTH),
        "preview_grid_max_height": int(DEFAULT_PREVIEW_GRID_MAX_HEIGHT),
        "preview_thumb_max_width": int(DEFAULT_PREVIEW_THUMB_MAX_WIDTH),
        "preview_thumb_max_height": int(DEFAULT_PREVIEW_THUMB_MAX_HEIGHT),
    },
    "economy_monitoring": {
        "label": "Monitoring oszczędny",
        "preview_fps_main": 12.0,
        "preview_fps_grid": 4.0,
        "preview_fps_thumb": 2.0,
        "preview_main_max_width": 768,
        "preview_main_max_height": 432,
        "preview_grid_max_width": 512,
        "preview_grid_max_height": 288,
        "preview_thumb_max_width": 256,
        "preview_thumb_max_height": 144,
    },
}
DEFAULT_CONFIG_WATCHDOG_ENABLED = True
DEFAULT_CONFIG_WATCHDOG_EVAL_SECONDS = 20.0
DEFAULT_CONFIG_WATCHDOG_DROP_DELTA_THRESHOLD = 5
DEFAULT_CONFIG_WATCHDOG_QUEUE_DELTA_THRESHOLD = 4
DEFAULT_PERFORMANCE_LOG_INTERVAL_S = 45.0
DEFAULT_PERFORMANCE_DIAGNOSTICS_ENABLED = False
DEFAULT_DETECTION_DEBUG_ENABLED = False
DEFAULT_RECORDER_QUEUE_WARN_THRESHOLD = 28
DEFAULT_RECORDER_QUEUE_CRITICAL_THRESHOLD = 52
DEFAULT_RECORDER_DROPPED_WARN_THRESHOLD = 2
DEFAULT_RECORDER_DROPPED_CRITICAL_THRESHOLD = 8
DEFAULT_RECORDER_QUEUE_PEAK_WARN_THRESHOLD = 40
DEFAULT_RECORDER_QUEUE_PEAK_CRITICAL_THRESHOLD = 72
DEFAULT_RECORDER_DEGRADE_WARN_WINDOW_S = 30.0
DEFAULT_RECORDER_MIN_DYNAMIC_WRITER_FPS = 1.0
DEFAULT_RECORDER_DEGRADE_ENTER_HYSTERESIS_S = 3.0
DEFAULT_RECORDER_DEGRADE_EXIT_HYSTERESIS_S = 5.0
DEFAULT_LOG_FILTER_GROUPS = [
    "application",
    "browser",
    "detection",
    "error",
    "performance",
    "recording",
    "settings",
    "ui",
    "warning",
    "worker",
]
DEFAULT_LOG_FILTER_LEVELS = ["INFO", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_LOG_FILTER_SOURCES = ["worker", "ui", "app"]
LOG_FILTERS: Dict[str, List[str]] = {
    "groups": list(DEFAULT_LOG_FILTER_GROUPS),
    "levels": list(DEFAULT_LOG_FILTER_LEVELS),
    "sources": list(DEFAULT_LOG_FILTER_SOURCES),
}




def apply_sensitivity_profile(camera: MutableMapping[str, object], profile_name: str, *, force: bool = False) -> None:
    """Apply sensitivity profile values into ``camera`` in-place."""
    profile = SENSITIVITY_PROFILES.get(str(profile_name))
    if not profile:
        return
    for key, value in profile.items():
        if force or key not in camera:
            camera[key] = value


def infer_sensitivity_profile(camera: MutableMapping[str, object]) -> str:
    """Infer profile name based on camera thresholds or return ``custom``."""
    for profile_name, values in SENSITIVITY_PROFILES.items():
        if all(camera.get(k) == v for k, v in values.items()):
            return profile_name
    return "custom"

def _resolve_path(value: str | os.PathLike[str] | None, *, default: Path) -> Path:
    """Resolve a path coming from configuration."""
    if not value:
        return default
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate


def _normalize_degirum_device_value(value: object, *, allow_inherit: bool = False) -> str:
    """Normalize DeGirum device selector values from configuration."""
    if value is None:
        return "cpu"
    normalized = str(value).strip()
    if not normalized:
        return "cpu"
    lowered = normalized.lower()
    if allow_inherit and lowered == "inherit":
        return "inherit"
    if lowered in {"auto", "cpu", "gpu"}:
        return lowered
    if re.match(r"^[A-Za-z0-9_.:-]+$", normalized):
        return normalized
    return "cpu"


def fill_camera_defaults(camera: MutableMapping[str, object]) -> MutableMapping[str, object]:
    """Fill missing camera parameters with default values."""
    has_detection_fps_limit = "detection_fps_limit" in camera

    defaults: Dict[str, object] = {
        "model": DEFAULT_MODEL,
        "fps": DEFAULT_FPS,
        "detection_fps_limit": DEFAULT_DETECTION_FPS_LIMIT,
        "rtsp_fps": DEFAULT_RTSP_FPS,
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "confidence_threshold_draw": DEFAULT_CONFIDENCE_THRESHOLD_DRAW,
        "confidence_threshold_record": DEFAULT_CONFIDENCE_THRESHOLD_RECORD,
        "draw_overlays": DEFAULT_DRAW_OVERLAYS,
        "overlay_text_enabled": DEFAULT_OVERLAY_TEXT_ENABLED,
        "overlay_draw_every_n": DEFAULT_OVERLAY_DRAW_EVERY_N,
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
        "recording_backend": DEFAULT_RECORDING_BACKEND,
        "required_hits_to_start_recording": DEFAULT_REQUIRED_HITS_TO_START_RECORDING,
        "required_misses_to_end_detection": DEFAULT_REQUIRED_MISSES_TO_END_DETECTION,
        "min_record_seconds": DEFAULT_MIN_RECORD_SECONDS,
        "sensitivity_profile": DEFAULT_SENSITIVITY_PROFILE,
        "preview_fps_main": DEFAULT_PREVIEW_FPS_MAIN,
        "preview_fps_grid": DEFAULT_PREVIEW_FPS_GRID,
        "preview_fps_thumb": DEFAULT_PREVIEW_FPS_THUMB,
        "preview_pause_when_hidden": DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN,
        "preview_main_max_width": DEFAULT_PREVIEW_MAIN_MAX_WIDTH,
        "preview_main_max_height": DEFAULT_PREVIEW_MAIN_MAX_HEIGHT,
        "preview_grid_max_width": DEFAULT_PREVIEW_GRID_MAX_WIDTH,
        "preview_grid_max_height": DEFAULT_PREVIEW_GRID_MAX_HEIGHT,
        "preview_thumb_max_width": DEFAULT_PREVIEW_THUMB_MAX_WIDTH,
        "preview_thumb_max_height": DEFAULT_PREVIEW_THUMB_MAX_HEIGHT,
        "camera_priority": DEFAULT_CAMERA_PRIORITY,
        "show_camera_info_overlay": DEFAULT_SHOW_CAMERA_INFO_OVERLAY,
        "camera_info_overlay_alpha": DEFAULT_CAMERA_INFO_OVERLAY_ALPHA,
        "thumbnail_overlay_enabled": DEFAULT_THUMBNAIL_OVERLAY_ENABLED,
        "thumbnail_box_thickness": DEFAULT_THUMBNAIL_BOX_THICKNESS,
        "thumbnail_font_scale": DEFAULT_THUMBNAIL_FONT_SCALE,
        "thumbnail_font_thickness": DEFAULT_THUMBNAIL_FONT_THICKNESS,
        "type": "rtsp",
        "recorder_queue_warn_threshold": DEFAULT_RECORDER_QUEUE_WARN_THRESHOLD,
        "recorder_queue_critical_threshold": DEFAULT_RECORDER_QUEUE_CRITICAL_THRESHOLD,
        "recorder_dropped_warn_threshold": DEFAULT_RECORDER_DROPPED_WARN_THRESHOLD,
        "recorder_dropped_critical_threshold": DEFAULT_RECORDER_DROPPED_CRITICAL_THRESHOLD,
        "recorder_queue_peak_warn_threshold": DEFAULT_RECORDER_QUEUE_PEAK_WARN_THRESHOLD,
        "recorder_queue_peak_critical_threshold": DEFAULT_RECORDER_QUEUE_PEAK_CRITICAL_THRESHOLD,
        "recorder_degrade_warn_window_s": DEFAULT_RECORDER_DEGRADE_WARN_WINDOW_S,
        "recorder_min_dynamic_writer_fps": DEFAULT_RECORDER_MIN_DYNAMIC_WRITER_FPS,
        "recorder_degrade_enter_hysteresis_s": DEFAULT_RECORDER_DEGRADE_ENTER_HYSTERESIS_S,
        "recorder_degrade_exit_hysteresis_s": DEFAULT_RECORDER_DEGRADE_EXIT_HYSTERESIS_S,
        "detection_debug_enabled": DEFAULT_DETECTION_DEBUG_ENABLED,
        "degirum_device_override_enabled": DEFAULT_DEGIRUM_DEVICE_OVERRIDE_ENABLED,
        "degirum_device_override": DEFAULT_DEGIRUM_DEVICE_OVERRIDE,
    }

    had_profile_inputs = any(
        key in camera
        for key in (
            "confidence_threshold",
            "confidence_threshold_draw",
            "confidence_threshold_record",
            "required_hits_to_start_recording",
            "required_misses_to_end_detection",
            "min_record_seconds",
        )
    )

    legacy_confidence = camera.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    if "confidence_threshold_draw" not in camera:
        camera["confidence_threshold_draw"] = legacy_confidence
    if "confidence_threshold_record" not in camera:
        camera["confidence_threshold_record"] = legacy_confidence
    if "confidence_threshold" not in camera:
        camera["confidence_threshold"] = camera.get("confidence_threshold_record", legacy_confidence)

    explicit_profile = camera.get("sensitivity_profile")
    for key, value in defaults.items():
        camera.setdefault(key, value)

    if not has_detection_fps_limit:
        camera["detection_fps_limit"] = camera.get("fps", DEFAULT_FPS)

    if explicit_profile is None and had_profile_inputs:
        profile_name = "custom"
    else:
        profile_name = str(explicit_profile or DEFAULT_SENSITIVITY_PROFILE)
    camera["sensitivity_profile"] = profile_name
    if profile_name != "custom":
        apply_sensitivity_profile(camera, profile_name, force=True)

    # ``record_path`` can be provided as a relative path in the configuration
    # file.  Normalise it so the rest of the application always works with an
    # absolute location rooted at :data:`BASE_DIR`.
    record_path = _resolve_path(camera.get("record_path"), default=DEFAULT_RECORD_PATH)
    camera["record_path"] = str(record_path)
    camera_priority = str(camera.get("camera_priority", DEFAULT_CAMERA_PRIORITY)).lower()
    camera["camera_priority"] = camera_priority if camera_priority in CAMERA_PRIORITIES else DEFAULT_CAMERA_PRIORITY
    recording_backend = str(camera.get("recording_backend", DEFAULT_RECORDING_BACKEND)).lower()
    camera["recording_backend"] = recording_backend if recording_backend in {"current", "ffmpeg"} else DEFAULT_RECORDING_BACKEND
    camera["degirum_device_override_enabled"] = bool(camera.get("degirum_device_override_enabled", DEFAULT_DEGIRUM_DEVICE_OVERRIDE_ENABLED))
    camera["degirum_device_override"] = _normalize_degirum_device_value(
        camera.get("degirum_device_override", DEFAULT_DEGIRUM_DEVICE_OVERRIDE), allow_inherit=True
    )
    camera.setdefault(
        "preview_channel_policies",
        {
            "main": {
                "fps": float(camera.get("preview_fps_main", DEFAULT_PREVIEW_FPS_MAIN)),
                "max_width": int(camera.get("preview_main_max_width", DEFAULT_PREVIEW_MAIN_MAX_WIDTH)),
                "max_height": int(camera.get("preview_main_max_height", DEFAULT_PREVIEW_MAIN_MAX_HEIGHT)),
            },
            "grid": {
                "fps": float(camera.get("preview_fps_grid", DEFAULT_PREVIEW_FPS_GRID)),
                "max_width": int(camera.get("preview_grid_max_width", DEFAULT_PREVIEW_GRID_MAX_WIDTH)),
                "max_height": int(camera.get("preview_grid_max_height", DEFAULT_PREVIEW_GRID_MAX_HEIGHT)),
            },
            "thumb": {
                "fps": float(camera.get("preview_fps_thumb", DEFAULT_PREVIEW_FPS_THUMB)),
                "max_width": int(camera.get("preview_thumb_max_width", DEFAULT_PREVIEW_THUMB_MAX_WIDTH)),
                "max_height": int(camera.get("preview_thumb_max_height", DEFAULT_PREVIEW_THUMB_MAX_HEIGHT)),
            },
        },
    )
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


def normalize_log_filters(raw_filters: object) -> Dict[str, List[str]]:
    """Normalize log filter structure from configuration data."""
    normalized: Dict[str, List[str]] = {
        "groups": list(DEFAULT_LOG_FILTER_GROUPS),
        "levels": list(DEFAULT_LOG_FILTER_LEVELS),
        "sources": list(DEFAULT_LOG_FILTER_SOURCES),
    }
    if not isinstance(raw_filters, MutableMapping):
        return normalized
    for key in ("groups", "levels", "sources"):
        values = raw_filters.get(key, normalized[key])
        if isinstance(values, list):
            cleaned = [str(item).strip() for item in values if str(item).strip()]
            if key == "levels":
                cleaned = [item.upper() for item in cleaned]
            unique: List[str] = []
            for item in cleaned:
                if item not in unique:
                    unique.append(item)
            normalized[key] = unique
    return normalized


def is_log_entry_enabled(group: str, level: str, source: str) -> bool:
    """Return ``True`` if a log entry matches active filter configuration."""
    group_name = str(group or "application").strip()
    level_name = str(level or "INFO").strip().upper()
    source_name = str(source or "").strip()
    source_category = source_name if source_name in {"worker", "ui"} else "app"
    enabled_groups = set(LOG_FILTERS.get("groups", []))
    enabled_levels = set(LOG_FILTERS.get("levels", []))
    enabled_sources = set(LOG_FILTERS.get("sources", []))
    if enabled_groups and group_name not in enabled_groups:
        return False
    if enabled_levels and level_name not in enabled_levels:
        return False
    if enabled_sources and source_category not in enabled_sources:
        return False
    return True


def load_config(path: Path | None = None) -> Dict[str, object]:
    """Load the application configuration."""
    global LOG_HISTORY_PATH, LOG_RETENTION_HOURS, LOG_FILTERS

    cfg_path = path or CONFIG_PATH
    if not cfg_path.exists():
        cfg: Dict[str, object] = {
            "log_history_path": str(LOG_HISTORY_PATH),
            "log_retention_hours": LOG_RETENTION_HOURS,
            "log_filters": normalize_log_filters(None),
            "cameras": [
                {
                    "name": "kamera1",
                    "rtsp": "rtsp://admin:IBLTSQ@192.168.8.165:554",
                }
            ],
        }
    else:
        raw_content = cfg_path.read_text(encoding="utf-8")
        try:
            cfg = json.loads(raw_content)
        except json.JSONDecodeError:
            sanitized = re.sub(r",(\s*[}\]])", r"\1", raw_content)
            cfg = json.loads(sanitized)
            cfg_path.write_text(json.dumps(cfg, indent=4), encoding="utf-8")

    LOG_HISTORY_PATH = _resolve_path(cfg.get("log_history_path"), default=LOG_HISTORY_PATH)
    LOG_RETENTION_HOURS = int(cfg.get("log_retention_hours", LOG_RETENTION_HOURS))
    LOG_FILTERS = normalize_log_filters(cfg.get("log_filters"))
    cfg["log_filters"] = dict(LOG_FILTERS)
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
    cfg.setdefault("quality_performance_preset", DEFAULT_QUALITY_PERFORMANCE_PRESET)
    cfg.setdefault("grid_preview_quality", DEFAULT_GRID_PREVIEW_QUALITY)
    cfg.setdefault("preview_fps_grid", DEFAULT_PREVIEW_FPS_GRID)
    cfg.setdefault("preview_grid_max_width", DEFAULT_PREVIEW_GRID_MAX_WIDTH)
    cfg.setdefault("preview_grid_max_height", DEFAULT_PREVIEW_GRID_MAX_HEIGHT)
    cfg.setdefault("config_watchdog_enabled", DEFAULT_CONFIG_WATCHDOG_ENABLED)
    cfg.setdefault("config_watchdog_eval_seconds", DEFAULT_CONFIG_WATCHDOG_EVAL_SECONDS)
    cfg.setdefault("config_watchdog_drop_delta_threshold", DEFAULT_CONFIG_WATCHDOG_DROP_DELTA_THRESHOLD)
    cfg.setdefault("config_watchdog_queue_delta_threshold", DEFAULT_CONFIG_WATCHDOG_QUEUE_DELTA_THRESHOLD)
    cfg.setdefault("performance_log_interval_s", DEFAULT_PERFORMANCE_LOG_INTERVAL_S)
    cfg.setdefault("performance_diagnostics_enabled", DEFAULT_PERFORMANCE_DIAGNOSTICS_ENABLED)
    cfg.setdefault("degirum_device_mode", DEFAULT_DEGIRUM_DEVICE_MODE)
    cfg.setdefault("degirum_preferred_device", DEFAULT_DEGIRUM_PREFERRED_DEVICE)
    cfg.setdefault("degirum_auto_select_best", DEFAULT_DEGIRUM_AUTO_SELECT_BEST)
    cfg.setdefault("degirum_available_devices", list(DEFAULT_DEGIRUM_AVAILABLE_DEVICES))
    cfg.setdefault("degirum_last_benchmark", dict(DEFAULT_DEGIRUM_LAST_BENCHMARK))

    cfg["degirum_device_mode"] = _normalize_degirum_device_value(cfg.get("degirum_device_mode"))
    cfg["degirum_preferred_device"] = _normalize_degirum_device_value(cfg.get("degirum_preferred_device"))

    for camera in cfg.get("cameras", []):
        if isinstance(camera, MutableMapping):
            fill_camera_defaults(camera)
    return cfg


def save_config(config: MutableMapping[str, object], path: Path | None = None) -> None:
    """Persist configuration to disk."""
    global LOG_HISTORY_PATH, LOG_RETENTION_HOURS, LOG_FILTERS

    for camera in config.get("cameras", []):
        if isinstance(camera, MutableMapping):
            fill_camera_defaults(camera)
    config.setdefault("log_history_path", str(LOG_HISTORY_PATH))
    config.setdefault("log_retention_hours", LOG_RETENTION_HOURS)
    LOG_FILTERS = normalize_log_filters(config.get("log_filters"))
    config["log_filters"] = dict(LOG_FILTERS)
    config.setdefault("degirum_device_mode", DEFAULT_DEGIRUM_DEVICE_MODE)
    config.setdefault("degirum_preferred_device", DEFAULT_DEGIRUM_PREFERRED_DEVICE)
    config.setdefault("degirum_auto_select_best", DEFAULT_DEGIRUM_AUTO_SELECT_BEST)
    config.setdefault("degirum_available_devices", list(DEFAULT_DEGIRUM_AVAILABLE_DEVICES))
    config.setdefault("degirum_last_benchmark", dict(DEFAULT_DEGIRUM_LAST_BENCHMARK))
    config["degirum_device_mode"] = _normalize_degirum_device_value(config.get("degirum_device_mode"))
    config["degirum_preferred_device"] = _normalize_degirum_device_value(config.get("degirum_preferred_device"))

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
    "DEFAULT_DETECTION_FPS_LIMIT",
    "DEFAULT_RTSP_FPS",
    "DEFAULT_SENSITIVITY_PROFILE",
    "DEFAULT_DEGIRUM_DEVICE_MODE",
    "DEFAULT_DEGIRUM_PREFERRED_DEVICE",
    "DEFAULT_DEGIRUM_AUTO_SELECT_BEST",
    "DEFAULT_DEGIRUM_AVAILABLE_DEVICES",
    "DEFAULT_DEGIRUM_LAST_BENCHMARK",
    "DEFAULT_DEGIRUM_DEVICE_OVERRIDE_ENABLED",
    "DEFAULT_DEGIRUM_DEVICE_OVERRIDE",
    "DEFAULT_LOST_SECONDS",
    "DEFAULT_MODEL",
    "DEFAULT_POST_SECONDS",
    "DEFAULT_PRE_SECONDS",
    "DEFAULT_RECORD_PATH",
    "DEFAULT_RECORD_START_MODE",
    "DEFAULT_RECORDING_BACKEND",
    "DEFAULT_REQUIRED_HITS_TO_START_RECORDING",
    "DEFAULT_REQUIRED_MISSES_TO_END_DETECTION",
    "DEFAULT_MIN_RECORD_SECONDS",
    "DEFAULT_PREVIEW_FPS_MAIN",
    "DEFAULT_PREVIEW_FPS_GRID",
    "DEFAULT_PREVIEW_FPS_THUMB",
    "DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN",
    "DEFAULT_PREVIEW_MAIN_MAX_WIDTH",
    "DEFAULT_PREVIEW_MAIN_MAX_HEIGHT",
    "DEFAULT_PREVIEW_GRID_MAX_WIDTH",
    "DEFAULT_PREVIEW_GRID_MAX_HEIGHT",
    "DEFAULT_GRID_PREVIEW_QUALITY",
    "DEFAULT_PREVIEW_THUMB_MAX_WIDTH",
    "DEFAULT_PREVIEW_THUMB_MAX_HEIGHT",
    "DEFAULT_CAMERA_PRIORITY",
    "CAMERA_PRIORITIES",
    "DEFAULT_SHOW_CAMERA_INFO_OVERLAY",
    "DEFAULT_CAMERA_INFO_OVERLAY_ALPHA",
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
    "DEFAULT_QUALITY_PERFORMANCE_PRESET",
    "QUALITY_PERFORMANCE_PRESETS",
    "DEFAULT_CONFIG_WATCHDOG_ENABLED",
    "DEFAULT_CONFIG_WATCHDOG_EVAL_SECONDS",
    "DEFAULT_CONFIG_WATCHDOG_DROP_DELTA_THRESHOLD",
    "DEFAULT_CONFIG_WATCHDOG_QUEUE_DELTA_THRESHOLD",
    "DEFAULT_PERFORMANCE_LOG_INTERVAL_S",
    "DEFAULT_PERFORMANCE_DIAGNOSTICS_ENABLED",
    "DEFAULT_THUMBNAIL_OVERLAY_ENABLED",
    "DEFAULT_THUMBNAIL_BOX_THICKNESS",
    "DEFAULT_THUMBNAIL_FONT_SCALE",
    "DEFAULT_THUMBNAIL_FONT_THICKNESS",
    "DEFAULT_THUMBNAIL_MODE",
    "ICON_DIR",
    "LOG_HISTORY_PATH",
    "LOG_FILTERS",
    "LOG_RETENTION_HOURS",
    "DEFAULT_LOG_FILTER_GROUPS",
    "DEFAULT_LOG_FILTER_LEVELS",
    "DEFAULT_LOG_FILTER_SOURCES",
    "is_log_entry_enabled",
    "MODELS_PATH",
    "RECORDINGS_CATALOG_PATH",
    "SENSITIVITY_PROFILES",
    "RECORD_CLASSES",
    "VISIBLE_CLASSES",
    "apply_sensitivity_profile",
    "fill_camera_defaults",
    "infer_sensitivity_profile",
    "list_usb_cameras",
    "load_config",
    "normalize_log_filters",
    "save_config",
]
