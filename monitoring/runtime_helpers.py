"""Runtime helper utilities that are test-friendly and dependency-light."""

from __future__ import annotations

import time
from threading import Lock
from typing import Callable


_APP_LOGGER: Callable[..., object] | None = None
_LOGGER_LOCK = Lock()


def register_app_logger(logger_callable: Callable[..., object] | None) -> None:
    """Register global callable used by non-UI modules to push structured logs."""
    global _APP_LOGGER
    with _LOGGER_LOCK:
        _APP_LOGGER = logger_callable


def app_log(group: str, message: str, **kwargs) -> bool:
    """Forward a structured log entry to UI bridge if available."""
    with _LOGGER_LOCK:
        logger_callable = _APP_LOGGER
    if logger_callable is None:
        return False
    logger_callable(group=group, message=message, **kwargs)
    return True


def classify_camera_setting_changes(old_camera: dict, new_camera: dict, restart_required_fields: set[str]) -> tuple[list[str], list[str]]:
    """Return changed camera keys and subset that require worker restart."""
    changed_keys = sorted(key for key in (set(old_camera) | set(new_camera)) if old_camera.get(key) != new_camera.get(key))
    restart_keys = [key for key in changed_keys if key in restart_required_fields]
    return changed_keys, restart_keys


def compute_effective_writer_fps(rtsp_fps: int, detect_fps: float, stream_fps: float) -> float:
    """Compute MP4 writer FPS so playback matches processed frame cadence."""
    if rtsp_fps > 0:
        return float(max(1.0, rtsp_fps))
    if detect_fps > 0:
        if stream_fps > 0:
            return float(max(1.0, min(stream_fps, detect_fps)))
        return float(max(1.0, detect_fps))
    if stream_fps > 0:
        return float(max(1.0, stream_fps))
    return 1.0


def compute_letterboxed_rect(frame_width: int, frame_height: int, canvas_width: int, canvas_height: int) -> tuple[int, int, int, int]:
    """Compute visible image rectangle for a frame letterboxed into a canvas."""
    if frame_width <= 0 or frame_height <= 0 or canvas_width <= 0 or canvas_height <= 0:
        return 0, 0, max(1, canvas_width), max(1, canvas_height)
    scale = min(canvas_width / frame_width, canvas_height / frame_height)
    new_w = max(1, int(frame_width * scale))
    new_h = max(1, int(frame_height * scale))
    x0 = (canvas_width - new_w) // 2
    y0 = (canvas_height - new_h) // 2
    return x0, y0, new_w, new_h


def camera_overlay_anchor(image_rect: tuple[int, int, int, int], box_size: tuple[int, int], padding: int = 10) -> tuple[int, int]:
    """Anchor bottom-left HUD box inside the visible image rectangle."""
    x0, y0, _iw, ih = image_rect
    bw, bh = box_size
    x = x0 + padding
    y = y0 + ih - bh - padding
    return max(x0, x), max(y0, y)


def thumbnail_load_outcome(image: object) -> str:
    """Classify thumbnail result to avoid indefinite loading state."""
    if image is None:
        return "fallback"
    is_null = getattr(image, "isNull", None)
    if callable(is_null):
        try:
            return "fallback" if bool(is_null()) else "success"
        except Exception:
            return "fallback"
    return "success"


def evaluate_heartbeat_health(
    active_workers: dict[str, bool],
    last_heartbeat_ts: dict[str, float],
    now_ts: float | None = None,
    timeout_seconds: float = 15.0,
) -> list[str]:
    """Return camera names that are active but heartbeat timed out."""
    now = float(now_ts if now_ts is not None else time.monotonic())
    stale: list[str] = []
    for camera_name, is_active in active_workers.items():
        if not is_active:
            continue
        last_ts = float(last_heartbeat_ts.get(camera_name, 0.0) or 0.0)
        if last_ts <= 0:
            stale.append(camera_name)
            continue
        if now - last_ts > timeout_seconds:
            stale.append(camera_name)
    return stale


def evaluate_overload_transition(
    *,
    now_ts: float,
    active_camera_count: int,
    gui_load_fps: float,
    recording_count: int,
    currently_active: bool,
    last_change_ts: float,
    protection_enabled: bool,
    min_camera_count: int,
    camera_threshold: int,
    load_per_camera_threshold: float,
    enter_debounce_seconds: float,
    exit_debounce_seconds: float,
) -> tuple[bool, float, str]:
    """Decide overload mode transition with camera floor + debounce hysteresis."""

    if not protection_enabled:
        return False, (now_ts if currently_active else last_change_ts), "disabled"

    if active_camera_count < max(1, int(min_camera_count)):
        if currently_active:
            if now_ts - last_change_ts >= max(0.0, exit_debounce_seconds):
                return False, now_ts, "below-min-camera-threshold"
            return True, last_change_ts, "exit-debounce-pending"
        return False, last_change_ts, "below-min-camera-threshold"

    overload_condition = (
        active_camera_count >= max(1, int(camera_threshold))
        or (active_camera_count > 0 and gui_load_fps > active_camera_count * float(load_per_camera_threshold))
    )
    if recording_count > 0 and active_camera_count <= camera_threshold:
        overload_condition = False

    if overload_condition:
        if currently_active:
            return True, last_change_ts, "already-active"
        if now_ts - last_change_ts >= max(0.0, enter_debounce_seconds):
            return True, now_ts, "condition-stable-enter"
        return False, last_change_ts, "enter-debounce-pending"

    if not currently_active:
        return False, last_change_ts, "already-inactive"
    if now_ts - last_change_ts >= max(0.0, exit_debounce_seconds):
        return False, now_ts, "condition-stable-exit"
    return True, last_change_ts, "exit-debounce-pending"


def worker_stop_timeout_details(camera_name: str, timeout_ms: int) -> str:
    """Consistent worker stop timeout detail string for logs/tests."""
    return f"camera={camera_name} timeout_ms={int(timeout_ms)}"
