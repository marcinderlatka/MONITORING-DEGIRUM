"""Runtime helper utilities that are test-friendly and dependency-light."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from threading import Lock
from typing import Callable


_APP_LOGGER: Callable[..., object] | None = None
_LOGGER_LOCK = Lock()
_FALLBACK_LOGGER = logging.getLogger(__name__)


def register_app_logger(logger_callable: Callable[..., object] | None) -> None:
    """Register global callable used by non-UI modules to push structured logs."""
    global _APP_LOGGER
    with _LOGGER_LOCK:
        _APP_LOGGER = logger_callable


def app_log(group: str, message: str, **kwargs) -> bool:
    """Forward a structured log entry to UI bridge if available."""
    if "traceback" in kwargs and "traceback_text" not in kwargs:
        kwargs["traceback_text"] = kwargs.pop("traceback")
    details = str(kwargs.get("details", "") or "")
    traceback_text = str(kwargs.get("traceback_text", "") or "")
    if traceback_text and traceback_text not in details:
        kwargs["details"] = f"{details}\n\n{traceback_text}".strip() if details else traceback_text

    with _LOGGER_LOCK:
        logger_callable = _APP_LOGGER
    if logger_callable is None:
        return False
    try:
        logger_callable(group=group, message=message, **kwargs)
        return True
    except Exception:
        _FALLBACK_LOGGER.exception("app_log forwarding failed: group=%s message=%s", group, message)
        return False


def classify_camera_setting_changes(old_camera: dict, new_camera: dict, restart_required_fields: set[str]) -> tuple[list[str], list[str]]:
    """Return changed camera keys and subset that require worker restart."""
    changed_keys = sorted(key for key in (set(old_camera) | set(new_camera)) if old_camera.get(key) != new_camera.get(key))
    restart_keys = [key for key in changed_keys if key in restart_required_fields]
    return changed_keys, restart_keys


def compute_effective_writer_fps(rtsp_fps: int, detect_fps: float, stream_fps: float) -> float:
    """Compute MP4 writer FPS so playback matches incoming frame cadence."""
    fps, _reason = compute_effective_writer_fps_details(rtsp_fps, detect_fps, stream_fps)
    return fps


def compute_effective_writer_fps_details(rtsp_fps: int, detect_fps: float, stream_fps: float) -> tuple[float, str]:
    """Compute MP4 writer FPS with explicit priority and explainable reason.

    Priority:
      1) stable measured stream FPS (``stream_fps`` from runtime window),
      2) configured RTSP throttle,
      3) detection FPS fallback,
      4) safe minimum.
    """
    if stream_fps > 0:
        return float(max(1.0, stream_fps)), "measured_stream"
    if rtsp_fps > 0:
        return float(max(1.0, rtsp_fps)), "rtsp_limit"
    if detect_fps > 0:
        return float(max(1.0, detect_fps)), "fallback_detect"
    return 1.0, "fallback_min"


def stabilized_stream_fps(samples: list[float] | tuple[float, ...], fallback: float = 0.0) -> float:
    """Return a robust stream FPS estimate based on trimmed median."""
    values = [float(v) for v in samples if float(v) > 0.0]
    if len(values) < 5:
        return float(max(0.0, fallback))
    values.sort()
    trim = max(0, int(len(values) * 0.1))
    core = values[trim: len(values) - trim] if (len(values) - (2 * trim)) >= 3 else values
    mid = len(core) // 2
    if len(core) % 2:
        median = core[mid]
    else:
        median = (core[mid - 1] + core[mid]) / 2.0
    return float(max(0.0, median))


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
    """Anchor bottom-right HUD box inside the visible image rectangle."""
    x0, y0, iw, ih = image_rect
    bw, bh = box_size
    x = x0 + iw - bw - padding
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
    currently_level: int,
    last_change_ts: float,
    protection_enabled: bool,
    min_camera_count: int,
    camera_threshold: int,
    load_per_camera_threshold: float,
    enter_debounce_seconds: float,
    exit_debounce_seconds: float,
    ui_render_ms: float,
    max_ui_render_ms: float,
    queue_size: int,
    max_queue_size: int,
    preview_bandwidth_mbps: float,
    max_preview_bandwidth_mbps: float,
) -> tuple[int, float, str]:
    """Decide overload level transition (L0-L3) with hysteresis and debounce."""

    level_now = max(0, min(3, int(currently_level)))

    if not protection_enabled:
        return 0, (now_ts if level_now > 0 else last_change_ts), "disabled"

    if active_camera_count < max(1, int(min_camera_count)):
        if level_now > 0:
            if now_ts - last_change_ts >= max(0.0, exit_debounce_seconds):
                return 0, now_ts, "below-min-camera-threshold"
            return level_now, last_change_ts, "exit-debounce-pending"
        return 0, last_change_ts, "below-min-camera-threshold"

    load_ratio = (gui_load_fps / max(1e-6, active_camera_count * float(load_per_camera_threshold))) if active_camera_count > 0 else 0.0
    ui_ratio = float(ui_render_ms) / max(1e-6, float(max_ui_render_ms))
    queue_ratio = float(max(0, queue_size)) / max(1e-6, float(max_queue_size))
    bandwidth_ratio = float(max(0.0, preview_bandwidth_mbps)) / max(1e-6, float(max_preview_bandwidth_mbps))
    peak_ratio = max(load_ratio, ui_ratio, queue_ratio, bandwidth_ratio)

    def _ratio_level(ratio: float, *, for_enter: bool) -> int:
        # Hysteresis: entering overload requires higher pressure than exiting it.
        if for_enter:
            if ratio >= 1.65:
                return 3
            if ratio >= 1.25:
                return 2
            if ratio >= 1.0:
                return 1
            return 0
        if ratio >= 1.45:
            return 3
        if ratio >= 1.10:
            return 2
        if ratio >= 0.85:
            return 1
        return 0

    def _camera_level(count: int, *, for_enter: bool) -> int:
        threshold = max(1, int(camera_threshold))
        if for_enter:
            if count >= threshold + 4:
                return 3
            if count >= threshold + 2:
                return 2
            if count >= threshold:
                return 1
            return 0
        if count >= threshold + 4:
            return 3
        if count >= threshold + 2:
            return 2
        if count >= threshold:
            return 1
        return 0

    desired_level_up = max(_ratio_level(peak_ratio, for_enter=True), _camera_level(active_camera_count, for_enter=True))
    desired_level_down = max(_ratio_level(peak_ratio, for_enter=False), _camera_level(active_camera_count, for_enter=False))
    desired_level = level_now
    if desired_level_up > level_now:
        desired_level = desired_level_up
    elif desired_level_down < level_now:
        desired_level = desired_level_down

    if recording_count > 0 and active_camera_count <= camera_threshold and peak_ratio < 1.1:
        desired_level = 0

    if desired_level == level_now:
        return level_now, last_change_ts, f"stable-L{level_now}"
    if desired_level > level_now:
        if now_ts - last_change_ts >= max(0.0, enter_debounce_seconds):
            return desired_level, now_ts, f"condition-stable-enter-L{desired_level}"
        return level_now, last_change_ts, "enter-debounce-pending"
    if now_ts - last_change_ts >= max(0.0, exit_debounce_seconds):
        return desired_level, now_ts, f"condition-stable-exit-L{desired_level}"
    return level_now, last_change_ts, "exit-debounce-pending"


@dataclass(frozen=True)
class OverloadLevelProfile:
    detect_fps_factor: float
    main_preview_fps_factor: float
    grid_preview_fps_factor: float
    thumb_preview_fps_factor: float
    overlay_stride: int
    performance_log_interval_s: float
    preview_resolution_factor: float
    disable_nonessential_overlays: bool


def overload_level_profile(level: int) -> OverloadLevelProfile:
    level_n = max(0, min(3, int(level)))
    profiles = {
        0: OverloadLevelProfile(1.0, 1.0, 1.0, 1.0, 1, 10.0, 1.0, False),
        # L1: first reduce grid/thumb FPS while keeping main/detection untouched.
        1: OverloadLevelProfile(1.0, 1.0, 0.82, 0.82, 1, 14.0, 1.0, False),
        # L2: additionally reduce main preview FPS and overlay frequency.
        2: OverloadLevelProfile(1.0, 0.82, 0.62, 0.62, 2, 18.0, 0.75, True),
        # L3: aggressive mode -> detection cuts for non-priority cameras + extra UI reductions.
        3: OverloadLevelProfile(0.6, 0.70, 0.42, 0.42, 3, 24.0, 0.55, True),
    }
    return profiles[level_n]


def worker_stop_timeout_details(camera_name: str, timeout_ms: int) -> str:
    """Consistent worker stop timeout detail string for logs/tests."""
    return f"camera={camera_name} timeout_ms={int(timeout_ms)}"


def build_root_cause_summary(
    *,
    ui_render_ms: float,
    ui_render_limit_ms: float,
    queue_size: int,
    queue_limit: int,
    infer_fps: float,
    detect_fps_target: float,
    stream_fps: float,
    writer_fps: float,
) -> str:
    """Summarize likely bottleneck category for faster diagnostics."""
    reasons: list[str] = []
    if ui_render_ms > max(1.0, ui_render_limit_ms):
        reasons.append("gui_render_overload")
    if queue_size > max(1, queue_limit):
        reasons.append("recording_pipeline_overload")
    if detect_fps_target > 0 and infer_fps < (0.7 * detect_fps_target):
        reasons.append("inference_overload")
    if stream_fps > 0 and writer_fps > 0 and stream_fps < (0.75 * writer_fps):
        reasons.append("stream_bottleneck")
    return ",".join(reasons) if reasons else "healthy"
