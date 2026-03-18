"""Thread workers handling RTSP streams and recordings."""

from __future__ import annotations

import datetime
import json
import logging
import os
import resource
import time
import traceback
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Lock
from typing import Any

import cv2
import degirum_tools  # type: ignore
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from .config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DETECTION_HOURS,
    DEFAULT_DRAW_OVERLAYS,
    DEFAULT_ENABLE_DETECTION,
    DEFAULT_ENABLE_RECORDING,
    DEFAULT_FPS,
    DEFAULT_LOST_SECONDS,
    DEFAULT_MIN_RECORD_SECONDS,
    DEFAULT_OVERLOAD_DISABLE_NONESSENTIAL_OVERLAYS,
    DEFAULT_POST_SECONDS,
    DEFAULT_PREVIEW_FPS_MAIN,
    DEFAULT_PREVIEW_FPS_THUMB,
    DEFAULT_PREVIEW_MAIN_MAX_HEIGHT,
    DEFAULT_PREVIEW_MAIN_MAX_WIDTH,
    DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN,
    DEFAULT_PREVIEW_THUMB_MAX_HEIGHT,
    DEFAULT_PREVIEW_THUMB_MAX_WIDTH,
    DEFAULT_PRE_SECONDS,
    DEFAULT_RECORD_PATH,
    DEFAULT_RECORD_START_MODE,
    DEFAULT_REQUIRED_HITS_TO_START_RECORDING,
    DEFAULT_REQUIRED_MISSES_TO_END_DETECTION,
    DEFAULT_RTSP_FPS,
    DEFAULT_THUMBNAIL_BOX_THICKNESS,
    DEFAULT_THUMBNAIL_FONT_SCALE,
    DEFAULT_THUMBNAIL_FONT_THICKNESS,
    DEFAULT_THUMBNAIL_MODE,
    DEFAULT_THUMBNAIL_OVERLAY_ENABLED,
    RECORD_CLASSES,
    VISIBLE_CLASSES,
)
from .recordings import build_recording_sidecar_metadata
from .runtime_helpers import app_log, compute_effective_writer_fps, worker_stop_timeout_details
from .storage import update_recordings_catalog

LABEL_COLORS = {
    "person": (0, 0, 255),
    "car": (255, 0, 0),
    "cat": (0, 255, 255),
    "dog": (255, 255, 0),
    "bird": (0, 255, 0),
}
PALETTE = [(255, 0, 255), (0, 165, 255), (255, 255, 0), (0, 255, 255), (255, 0, 0), (0, 255, 0)]

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    last_capture_ts: float = 0.0
    last_preview_emit_ts: float = 0.0
    last_inference_ts: float = 0.0
    last_metrics_log_ts: float = 0.0
    last_heartbeat_ts: float = 0.0
    last_detection_ts: float = 0.0
    recording_started_ts: float = 0.0
    stream_start_ts: float = 0.0
    frames_captured: int = 0
    frames_emitted: int = 0
    inferences_run: int = 0
    positive_detections: int = 0
    dropped_preview_frames: int = 0
    skipped_inference_cycles: int = 0
    next_inference_due_ts: float = 0.0
    preview_frame_skip_counter: int = 0
    preview_frames_dropped_total: int = 0
    metrics_window_started_ts: float = 0.0
    metrics_frames_captured: int = 0
    metrics_inferences_run: int = 0
    metrics_frames_emitted: int = 0
    metrics_dropped_frames: int = 0
    metrics_last_cpu_process_ts: float = 0.0
    metrics_last_cpu_wall_ts: float = 0.0


METRIC_KEYS = (
    "capture_fps",
    "infer_fps",
    "preview_emit_fps",
    "ui_render_ms",
    "queue_size",
    "dropped_frames",
    "cpu_percent",
    "rss_mb",
)


def _label_color(label: str) -> tuple[int, int, int]:
    key = (label or "").lower()
    if key in LABEL_COLORS:
        return LABEL_COLORS[key]
    if not key:
        return (255, 255, 255)
    return PALETTE[hash(key) % len(PALETTE)]


def _normalize_size(value: Any) -> tuple[int, int] | None:
    if isinstance(value, dict):
        w = value.get("width") or value.get("w")
        h = value.get("height") or value.get("h")
        if w and h:
            return int(w), int(h)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        h, w = value[:2]
        return int(w), int(h)
    return None


def _extract_image_size(result: Any) -> tuple[int, int] | None:
    for attr in ("image_size", "input_image_size", "image_shape", "input_shape"):
        size = _normalize_size(getattr(result, attr, None))
        if size:
            return size
    if isinstance(result, dict):
        return _normalize_size(result.get("image_size") or result.get("input_image_size"))
    return None


def _scale_bbox(bbox: list[float] | tuple[float, ...], frame_shape: tuple[int, ...], source_size: tuple[int, int] | None) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, bbox)
    h, w = frame_shape[:2]
    if source_size:
        src_w, src_h = source_size
        if src_w and src_h and (src_w != w or src_h != h):
            x1 *= w / src_w
            x2 *= w / src_w
            y1 *= h / src_h
            y2 *= h / src_h
    if 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0:
        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return int(max(0, x1)), int(max(0, y1)), int(min(w - 1, x2)), int(min(h - 1, y2))


def _preview_interval_for_role(role: str, main_fps: float, thumb_fps: float, pause_hidden: bool) -> float:
    role_l = (role or "thumb").lower()
    if role_l == "main":
        return 1.0 / max(1e-3, main_fps)
    if role_l == "hidden" and pause_hidden:
        return float("inf")
    basis = max(0.5, thumb_fps)
    if role_l == "hidden":
        basis = min(basis, 1.0)
    return 1.0 / max(1e-3, basis)


def _advance_next_due(now_ts: float, next_due_ts: float, interval: float) -> tuple[float, int]:
    if interval <= 0:
        return now_ts, 0
    if next_due_ts <= 0:
        return now_ts, 0
    skipped = 0
    while now_ts - next_due_ts > interval:
        next_due_ts += interval
        skipped += 1
    if now_ts - next_due_ts > interval * 4:
        next_due_ts = now_ts + interval
    return next_due_ts, skipped


def _aggregate_fps(count_delta: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return float(max(0, count_delta) / elapsed_s)


def _dropped_frames_delta(total_dropped: int, baseline_dropped: int) -> int:
    return int(max(0, int(total_dropped) - int(baseline_dropped)))


def _rss_mb() -> float:
    rss_raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss_raw / 1024.0


def _build_metrics_payload(**kwargs: float | int) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}
    for key in METRIC_KEYS:
        payload[key] = kwargs.get(key, 0.0 if key.endswith(("fps", "ms", "percent", "mb")) else 0)
    return payload


class RecordingThread(QThread):
    def __init__(self, filepath: str, width: int, height: int, fps: float) -> None:
        super().__init__()
        self.filepath = filepath
        self.width = width
        self.height = height
        self.fps = float(max(1.0, fps))
        self.queue: "Queue[np.ndarray]" = Queue(maxsize=120)
        self.running = True
        self.writer = None
        self.dropped_frames = 0
        self.frames_written = 0
        self.started_ts = 0.0
        self.last_write_ts = 0.0
        self.queue_peak = 0

    def run(self) -> None:
        self.writer = degirum_tools.VideoWriter(self.filepath, self.width, self.height, self.fps)
        self.started_ts = time.monotonic()
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                self.writer.write(frame)
                self.frames_written += 1
                self.last_write_ts = time.monotonic()
            except Empty:
                pass
        if self.writer:
            with suppress(AttributeError):
                self.writer.release()
            self.writer = None

    def write(self, frame: np.ndarray) -> None:
        if self.running:
            try:
                self.queue.put_nowait(frame)
                self.queue_peak = max(self.queue_peak, self.queue.qsize())
            except Full:
                self.dropped_frames += 1

    def stop(self, timeout_ms: int = 2000) -> bool:
        self.running = False
        stopped = self.wait(timeout_ms)
        if not stopped:
            app_log("warning", "recording thread stop timeout", source="worker", level="WARNING", details=f"file={self.filepath} timeout_ms={timeout_ms}")
        return bool(stopped)


class CameraWorker(QThread):
    main_preview_signal = pyqtSignal(object, int)
    thumb_preview_signal = pyqtSignal(object, int)
    alert_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str, int)
    status_signal = pyqtSignal(str, int)
    record_signal = pyqtSignal(str, str)
    worker_status_signal = pyqtSignal(str, object)
    _active_workers_lock = Lock()
    _active_workers_by_camera: dict[str, int] = {}

    def __init__(self, camera: dict, model: Any, index: int = 0) -> None:
        super().__init__()
        self.camera = dict(camera)
        self.model = model
        self.index = index
        self.state = PipelineState()

        self.fps = int(self.camera.get("fps", DEFAULT_FPS))
        self.rtsp_fps = int(self.camera.get("rtsp_fps", DEFAULT_RTSP_FPS))
        legacy_conf = float(self.camera.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
        self.confidence_threshold_draw = float(self.camera.get("confidence_threshold_draw", legacy_conf))
        self.confidence_threshold_record = float(self.camera.get("confidence_threshold_record", legacy_conf))
        self.confidence_threshold = self.confidence_threshold_record
        self.draw_overlays = bool(self.camera.get("draw_overlays", DEFAULT_DRAW_OVERLAYS))
        self.enable_detection = bool(self.camera.get("enable_detection", DEFAULT_ENABLE_DETECTION))
        self.enable_recording = bool(self.camera.get("enable_recording", DEFAULT_ENABLE_RECORDING))
        self.detection_hours = str(self.camera.get("detection_hours", DEFAULT_DETECTION_HOURS))
        self.visible_classes = list(self.camera.get("visible_classes", VISIBLE_CLASSES))
        self.record_classes = list(self.camera.get("record_classes", RECORD_CLASSES))
        self.visible_classes_lower = {c.lower() for c in self.visible_classes}
        self.record_classes_lower = {c.lower() for c in self.record_classes}
        self.pre_seconds = int(self.camera.get("pre_seconds", DEFAULT_PRE_SECONDS))
        self.post_seconds = int(self.camera.get("post_seconds", DEFAULT_POST_SECONDS))
        self.lost_seconds = int(self.camera.get("lost_seconds", DEFAULT_LOST_SECONDS))
        self.thumbnail_mode = str(self.camera.get("thumbnail_mode", DEFAULT_THUMBNAIL_MODE))
        self.thumbnail_overlay_enabled = bool(self.camera.get("thumbnail_overlay_enabled", DEFAULT_THUMBNAIL_OVERLAY_ENABLED))
        self.thumbnail_box_thickness = int(self.camera.get("thumbnail_box_thickness", DEFAULT_THUMBNAIL_BOX_THICKNESS))
        self.thumbnail_font_scale = float(self.camera.get("thumbnail_font_scale", DEFAULT_THUMBNAIL_FONT_SCALE))
        self.thumbnail_font_thickness = int(self.camera.get("thumbnail_font_thickness", DEFAULT_THUMBNAIL_FONT_THICKNESS))
        self.record_start_mode = str(self.camera.get("record_start_mode", DEFAULT_RECORD_START_MODE))
        self.required_hits_to_start_recording = int(self.camera.get("required_hits_to_start_recording", DEFAULT_REQUIRED_HITS_TO_START_RECORDING))
        self.required_misses_to_end_detection = int(self.camera.get("required_misses_to_end_detection", DEFAULT_REQUIRED_MISSES_TO_END_DETECTION))
        self.min_record_seconds = int(self.camera.get("min_record_seconds", DEFAULT_MIN_RECORD_SECONDS))

        self.preview_fps_main = float(self.camera.get("preview_fps_main", DEFAULT_PREVIEW_FPS_MAIN))
        self.preview_fps_thumb = float(self.camera.get("preview_fps_thumb", DEFAULT_PREVIEW_FPS_THUMB))
        self.preview_pause_when_hidden = bool(self.camera.get("preview_pause_when_hidden", DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN))
        self.preview_main_max_width = int(self.camera.get("preview_main_max_width", DEFAULT_PREVIEW_MAIN_MAX_WIDTH))
        self.preview_main_max_height = int(self.camera.get("preview_main_max_height", DEFAULT_PREVIEW_MAIN_MAX_HEIGHT))
        self.preview_thumb_max_width = int(self.camera.get("preview_thumb_max_width", DEFAULT_PREVIEW_THUMB_MAX_WIDTH))
        self.preview_thumb_max_height = int(self.camera.get("preview_thumb_max_height", DEFAULT_PREVIEW_THUMB_MAX_HEIGHT))
        self.preview_role = "thumb"
        self.is_overload_degraded = False
        self.app_overload_mode = False
        self.overload_disable_nonessential_overlays = bool(self.camera.get("overload_disable_nonessential_overlays", DEFAULT_OVERLOAD_DISABLE_NONESSENTIAL_OVERLAYS))
        self.detect_fps_factor = 1.0
        self.overload_level = 0
        self.overlay_stride = 1
        self.preview_resolution_factor = 1.0
        self.performance_log_interval_s = 10.0
        self.is_recording_active = False

        rec_path = str(self.camera.get("record_path", DEFAULT_RECORD_PATH))
        self.output_dir = os.path.join(rec_path, self.camera.get("name", "camera"))
        os.makedirs(self.output_dir, exist_ok=True)

        self.recording = False
        self.record_thread: RecordingThread | None = None
        self.output_file: str | None = None
        self.stop_signal = False
        self._current_stream = None
        self.record_lock = Lock()
        self.prerecord_buffer = deque(maxlen=max(1, int(self.pre_seconds * max(1, self.fps))))

        self.stream_fps = 0.0
        self.source_fps = 0.0
        self.writer_fps = 0.0
        self.last_frame_ts = 0.0
        self.last_stream_reset_ts = 0.0
        self.stream_stall_seconds = 5.0
        self.stream_start_ts = 0.0
        self.frame_counter = 0
        self.loop_fps = 0.0
        self.restart_requested = False
        self.error_counter = 0
        self._stream_fps_window = deque(maxlen=300)
        self._stream_fps_last_calc_ts = 0.0

        self.detection_active = False
        self.detection_last_seen_ts = 0.0
        self.recording_started_ts = 0.0
        self.pending_positive_hits = 0
        self.pending_miss_count = 0
        self.current_event_best_confidence = 0.0
        self.current_event_best_frame: np.ndarray | None = None
        self.current_event_scene_thumbnail_path = ""
        self.current_event_thumbnail_path = ""
        self.current_event_metadata_path = ""
        self.current_event_label = ""
        self.current_event_confidence = 0.0
        self.current_event_start_ts = 0.0
        self.current_writer_fps = 0.0
        self.current_detection_frame_saved = False
        self.current_thumbnail_ts = 0.0
        self.current_event_detection_count = 0
        self.current_event_confidence_sum = 0.0
        self.current_event_max_confidence = 0.0

        self.inference_count = 0
        self.positive_detection_count = 0
        self.last_metrics_log_ts = 0.0
        self._runtime_limit_logged = False
        self._record_queue_full_warned = False
        self._worker_slot_key: str | None = None
        now_mono = time.monotonic()
        self.state.metrics_window_started_ts = now_mono
        self.state.metrics_last_cpu_wall_ts = now_mono
        self.state.metrics_last_cpu_process_ts = time.process_time()

    def _camera_worker_key(self) -> str:
        camera_type = str(self.camera.get("type", "rtsp")).lower()
        src = str(self.camera.get("rtsp", ""))
        if camera_type == "usb":
            with suppress(Exception):
                src = str(int(src))
        return f"{camera_type}:{src}"

    def _acquire_worker_slot(self) -> bool:
        key = self._camera_worker_key()
        with self._active_workers_lock:
            owner = self._active_workers_by_camera.get(key)
            if owner is not None and owner != self.index:
                app_log(
                    "warning",
                    "duplicate worker blocked",
                    camera=str(self.camera.get("name", self.index)),
                    source="worker",
                    level="WARNING",
                    details=f"camera_key={key} owner_index={owner} blocked_index={self.index}",
                )
                return False
            self._active_workers_by_camera[key] = self.index
            self._worker_slot_key = key
            return True

    def _release_worker_slot(self) -> None:
        key = self._worker_slot_key
        if not key:
            return
        with self._active_workers_lock:
            owner = self._active_workers_by_camera.get(key)
            if owner == self.index:
                self._active_workers_by_camera.pop(key, None)
        self._worker_slot_key = None

    def set_preview_role(self, role: str) -> None:
        self.preview_role = role if role in {"main", "thumb", "hidden"} else "thumb"

    def set_overload_state(
        self,
        overload_level: int,
        detect_fps_factor: float | None = None,
        thumb_preview_fps: float | None = None,
        disable_overlays: bool | None = None,
        overlay_stride: int | None = None,
        preview_resolution_factor: float | None = None,
    ) -> None:
        previous_level = self.overload_level
        level = max(0, min(3, int(overload_level)))
        self.overload_level = level
        self.app_overload_mode = level > 0
        self.is_overload_degraded = bool(self.app_overload_mode and self.preview_role != "main")
        if detect_fps_factor is not None:
            self.detect_fps_factor = float(max(0.2, min(1.0, detect_fps_factor)))
        else:
            self.detect_fps_factor = 1.0
        if thumb_preview_fps is not None and thumb_preview_fps > 0:
            self.preview_fps_thumb = float(thumb_preview_fps)
        if disable_overlays is not None:
            self.overload_disable_nonessential_overlays = bool(disable_overlays)
        if overlay_stride is not None:
            self.overlay_stride = int(max(1, overlay_stride))
        if preview_resolution_factor is not None:
            self.preview_resolution_factor = float(max(0.3, min(1.0, preview_resolution_factor)))
        self.performance_log_interval_s = 10.0 + float(self.overload_level * 4.0)
        if previous_level != self.overload_level:
            app_log("worker", "overload state updated", camera=str(self.camera.get("name", self.index)), source="worker", level="INFO", details=f"level=L{self.overload_level} active={self.app_overload_mode} detect_fps_factor={self.detect_fps_factor:.2f} overlay_stride={self.overlay_stride} preview_res_factor={self.preview_resolution_factor:.2f} preview_role={self.preview_role}")

    def _build_thumbnail_frame(self, preview_frame: np.ndarray, best_bbox: tuple[int, int, int, int] | None, best_label: str, best_score: float) -> np.ndarray:
        scene = self._build_scene_thumbnail_frame(preview_frame)
        overlay_bbox: tuple[int, int, int, int] | None = None
        if best_bbox is not None and preview_frame is not None and preview_frame.size > 0:
            src_h, src_w = preview_frame.shape[:2]
            if src_h > 0 and src_w > 0:
                dst_h, dst_w = scene.shape[:2]
                sx = dst_w / float(src_w)
                sy = dst_h / float(src_h)
                x1, y1, x2, y2 = best_bbox
                overlay_bbox = (
                    int(max(0, min(dst_w - 1, x1 * sx))),
                    int(max(0, min(dst_h - 1, y1 * sy))),
                    int(max(0, min(dst_w - 1, x2 * sx))),
                    int(max(0, min(dst_h - 1, y2 * sy))),
                )
        return self._make_detection_overlay_frame(scene, overlay_bbox, best_label, best_score)

    @staticmethod
    def _build_scene_thumbnail_frame(preview_frame: np.ndarray) -> np.ndarray:
        return cv2.resize(preview_frame, (320, 180), interpolation=cv2.INTER_AREA)

    def _get_effective_stream_fps(self) -> float:
        now = time.monotonic()
        if len(self._stream_fps_window) < 2:
            return self.stream_fps
        if self._stream_fps_last_calc_ts and now - self._stream_fps_last_calc_ts < 5.0:
            return self.stream_fps
        elapsed = self._stream_fps_window[-1] - self._stream_fps_window[0]
        if elapsed <= 0:
            return self.stream_fps
        self.stream_fps = float((len(self._stream_fps_window) - 1) / elapsed)
        self._stream_fps_last_calc_ts = now
        return self.stream_fps

    def refresh_class_filters(self) -> None:
        self.visible_classes_lower = {c.lower() for c in self.visible_classes}
        self.record_classes_lower = {c.lower() for c in self.record_classes}

    def set_confidence(self, threshold: float) -> None:
        value = float(threshold)
        self.confidence_threshold = value
        self.confidence_threshold_draw = value
        self.confidence_threshold_record = value
        self.camera["confidence_threshold"] = value
        self.camera["confidence_threshold_draw"] = value
        self.camera["confidence_threshold_record"] = value

    def set_fps(self, fps: int) -> None:
        self.fps = int(max(1, fps))
        self.camera["fps"] = self.fps

    def set_draw_overlays(self, value: bool) -> None:
        self.draw_overlays = bool(value)
        self.camera["draw_overlays"] = self.draw_overlays

    def set_enable_detection(self, value: bool) -> None:
        self.enable_detection = bool(value)
        self.camera["enable_detection"] = self.enable_detection

    def set_enable_recording(self, value: bool) -> None:
        self.enable_recording = bool(value)
        self.camera["enable_recording"] = self.enable_recording

    def set_detection_schedule(self, hours: str) -> None:
        self.detection_hours = str(hours or "").strip() or "00:00-23:59"
        self.camera["detection_hours"] = self.detection_hours

    def apply_runtime_settings(self, camera_config: dict) -> None:
        """Apply runtime-safe camera settings without restarting worker thread."""
        legacy_conf = float(camera_config.get("confidence_threshold", self.confidence_threshold_record))
        self.set_fps(int(camera_config.get("fps", self.fps)))
        self.rtsp_fps = int(camera_config.get("rtsp_fps", self.rtsp_fps))
        self.camera["rtsp_fps"] = self.rtsp_fps
        self.confidence_threshold = legacy_conf
        self.confidence_threshold_draw = float(camera_config.get("confidence_threshold_draw", legacy_conf))
        self.confidence_threshold_record = float(camera_config.get("confidence_threshold_record", legacy_conf))
        self.camera["confidence_threshold"] = legacy_conf
        self.camera["confidence_threshold_draw"] = self.confidence_threshold_draw
        self.camera["confidence_threshold_record"] = self.confidence_threshold_record

        self.set_draw_overlays(camera_config.get("draw_overlays", self.draw_overlays))
        self.set_enable_detection(camera_config.get("enable_detection", self.enable_detection))
        self.set_enable_recording(camera_config.get("enable_recording", self.enable_recording))
        self.set_detection_schedule(camera_config.get("detection_hours", self.detection_hours))

        self.visible_classes = list(camera_config.get("visible_classes", self.visible_classes))
        self.record_classes = list(camera_config.get("record_classes", self.record_classes))
        self.camera["visible_classes"] = list(self.visible_classes)
        self.camera["record_classes"] = list(self.record_classes)
        self.refresh_class_filters()

        self.pre_seconds = int(camera_config.get("pre_seconds", self.pre_seconds))
        self.lost_seconds = int(camera_config.get("lost_seconds", self.lost_seconds))
        self.post_seconds = int(camera_config.get("post_seconds", self.post_seconds))
        self.required_hits_to_start_recording = int(camera_config.get("required_hits_to_start_recording", self.required_hits_to_start_recording))
        self.required_misses_to_end_detection = int(camera_config.get("required_misses_to_end_detection", self.required_misses_to_end_detection))
        self.min_record_seconds = int(camera_config.get("min_record_seconds", self.min_record_seconds))
        self.thumbnail_mode = str(camera_config.get("thumbnail_mode", self.thumbnail_mode))
        self.thumbnail_overlay_enabled = bool(camera_config.get("thumbnail_overlay_enabled", self.thumbnail_overlay_enabled))
        self.thumbnail_box_thickness = int(camera_config.get("thumbnail_box_thickness", self.thumbnail_box_thickness))
        self.thumbnail_font_scale = float(camera_config.get("thumbnail_font_scale", self.thumbnail_font_scale))
        self.thumbnail_font_thickness = int(camera_config.get("thumbnail_font_thickness", self.thumbnail_font_thickness))
        self.record_start_mode = str(camera_config.get("record_start_mode", self.record_start_mode))

        self.preview_fps_main = float(camera_config.get("preview_fps_main", self.preview_fps_main))
        self.preview_fps_thumb = float(camera_config.get("preview_fps_thumb", self.preview_fps_thumb))
        self.preview_pause_when_hidden = bool(camera_config.get("preview_pause_when_hidden", self.preview_pause_when_hidden))
        self.preview_main_max_width = int(camera_config.get("preview_main_max_width", self.preview_main_max_width))
        self.preview_main_max_height = int(camera_config.get("preview_main_max_height", self.preview_main_max_height))
        self.preview_thumb_max_width = int(camera_config.get("preview_thumb_max_width", self.preview_thumb_max_width))
        self.preview_thumb_max_height = int(camera_config.get("preview_thumb_max_height", self.preview_thumb_max_height))
        self.camera["preview_fps_main"] = self.preview_fps_main
        self.camera["preview_fps_thumb"] = self.preview_fps_thumb
        self.camera["preview_pause_when_hidden"] = self.preview_pause_when_hidden
        self.camera["preview_main_max_width"] = self.preview_main_max_width
        self.camera["preview_main_max_height"] = self.preview_main_max_height
        self.camera["preview_thumb_max_width"] = self.preview_thumb_max_width
        self.camera["preview_thumb_max_height"] = self.preview_thumb_max_height

        self.camera["pre_seconds"] = self.pre_seconds
        self.camera["lost_seconds"] = self.lost_seconds
        self.camera["post_seconds"] = self.post_seconds
        self.camera["required_hits_to_start_recording"] = self.required_hits_to_start_recording
        self.camera["required_misses_to_end_detection"] = self.required_misses_to_end_detection
        self.camera["min_record_seconds"] = self.min_record_seconds
        self.camera["thumbnail_mode"] = self.thumbnail_mode
        self.camera["thumbnail_overlay_enabled"] = self.thumbnail_overlay_enabled
        self.camera["thumbnail_box_thickness"] = self.thumbnail_box_thickness
        self.camera["thumbnail_font_scale"] = self.thumbnail_font_scale
        self.camera["thumbnail_font_thickness"] = self.thumbnail_font_thickness
        self.camera["record_start_mode"] = self.record_start_mode

        record_base = camera_config.get("record_path", self.camera.get("record_path", DEFAULT_RECORD_PATH))
        self.camera["record_path"] = str(record_base)
        self.output_dir = str(os.path.join(str(record_base), camera_config.get("name", self.camera.get("name", "camera"))))
        os.makedirs(self.output_dir, exist_ok=True)

        buffer_fps = self._get_prerecord_buffer_fps_basis()
        self.prerecord_buffer = deque(self.prerecord_buffer, maxlen=max(1, int(self.pre_seconds * max(1.0, buffer_fps))))
        self.camera.update(camera_config)

    def _is_within_schedule(self) -> bool:
        try:
            now = datetime.datetime.now().time()
            for part in self.detection_hours.replace(" ", "").split(";"):
                if not part:
                    continue
                a, b = part.split("-")
                ha, ma = map(int, a.split(":")); hb, mb = map(int, b.split(":"))
                start = datetime.time(ha, ma); end = datetime.time(hb, mb)
                if (start <= end and start <= now <= end) or (start > end and (now >= start or now <= end)):
                    return True
            return False
        except Exception:
            return True

    def _compute_effective_writer_fps(self, stream_fps: float) -> float:
        return float(max(1.0, compute_effective_writer_fps(self.rtsp_fps, float(self.fps), stream_fps)))

    def _get_prerecord_buffer_fps_basis(self) -> float:
        if self.rtsp_fps > 0:
            return float(max(1.0, self.rtsp_fps))
        measured = self.stream_fps or self.loop_fps or self.fps
        return float(min(60.0, max(1.0, measured)))

    def _sync_prerecord_buffer(self) -> None:
        basis = self._get_prerecord_buffer_fps_basis()
        maxlen = max(1, int(self.pre_seconds * basis))
        if self.prerecord_buffer.maxlen != maxlen:
            self.prerecord_buffer = deque(self.prerecord_buffer, maxlen=maxlen)
            logger.info("prerecord buffer updated camera=%s pre_seconds=%s basis=%.2f maxlen=%s", self.camera.get("name", self.index), self.pre_seconds, basis, maxlen)
            app_log("worker", "prerecord buffer updated", camera=str(self.camera.get("name", self.index)), source="worker", level="INFO", details=f"basis={basis:.2f} maxlen={maxlen}")

    def _make_detection_overlay_frame(self, frame: np.ndarray, bbox: tuple[int, int, int, int] | None, label: str, confidence: float) -> np.ndarray:
        canvas = frame.copy()
        if bbox and self.thumbnail_overlay_enabled:
            x1, y1, x2, y2 = bbox
            color = _label_color(label)
            box_thickness = max(0, int(self.thumbnail_box_thickness))
            font_scale = max(0.1, float(self.thumbnail_font_scale))
            font_thickness = max(1, int(self.thumbnail_font_thickness))
            if box_thickness > 0:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(
                canvas,
                f"{label}: {confidence * 100:.1f}%",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                font_thickness,
            )
        return canvas

    def _build_recording_meta(self, **kwargs: Any) -> dict:
        event_time = datetime.datetime.fromtimestamp(float(kwargs["event_start_ts"]))
        scene_thumb = str(kwargs.get("scene_thumb") or kwargs.get("alert_thumb") or kwargs.get("thumb_path") or "")
        return build_recording_sidecar_metadata(
            camera=self.camera.get("name", ""), label=kwargs["label"], confidence=kwargs["confidence"],
            event_time=event_time.strftime("%Y-%m-%d %H:%M:%S"), filepath=kwargs["filepath"], thumb=kwargs["thumb_path"],
            alert_thumb=scene_thumb,
            scene_thumb=scene_thumb,
            source_fps=kwargs["source_fps"], writer_fps=kwargs["writer_fps"], detect_fps=kwargs["detect_fps"],
            event_start_ts=kwargs["event_start_ts"], thumbnail_ts=kwargs["thumbnail_ts"], frames_written=kwargs["frames_written"],
            dropped_frames=kwargs["dropped_frames"], thumbnail_mode=kwargs["thumbnail_mode"], inference_count=self.inference_count,
            positive_detection_count=self.positive_detection_count, record_start_mode=self.record_start_mode,
            min_record_seconds=self.min_record_seconds, required_hits_to_start_recording=self.required_hits_to_start_recording,
            required_misses_to_end_detection=self.required_misses_to_end_detection, event_end_ts=kwargs.get("event_end_ts", 0.0),
            recording_duration=kwargs.get("recording_duration", 0.0), detection_count=kwargs.get("detection_count", 0),
            max_confidence=kwargs.get("max_confidence", 0.0), avg_confidence=kwargs.get("avg_confidence", 0.0), stream_fps=kwargs.get("stream_fps", self.stream_fps),
            preview_role_at_start=kwargs.get("preview_role_at_start", self.preview_role),
            overload_degraded_at_start=kwargs.get("overload_degraded_at_start", self.is_overload_degraded),
            measured_capture_fps=kwargs.get("measured_capture_fps", self.stream_fps),
            effective_detect_fps=kwargs.get("effective_detect_fps", self._effective_detect_fps(max(time.monotonic() - self.state.stream_start_ts, 1.0))),
            preview_frames_dropped=kwargs.get("preview_frames_dropped", self.state.preview_frames_dropped_total),
            skipped_inference_cycles=kwargs.get("skipped_inference_cycles", self.state.skipped_inference_cycles),
            app_overload_mode=kwargs.get("app_overload_mode", self.app_overload_mode),
            recorder_queue_peak=kwargs.get("recorder_queue_peak", self.record_thread.queue_peak if self.record_thread else 0),
        )

    def _save_recording_metadata(self, meta: dict) -> None:
        if not self.output_file:
            return
        try:
            with open(self.output_file + ".json", "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)
        except Exception as exc:
            logger.exception("Nie zapisano metadanych")
            app_log("error", "Nie zapisano metadanych", camera=str(self.camera.get("name", self.index)), source="worker", level="ERROR", details=str(exc))
        update_recordings_catalog(dict(meta))

    def _update_event_thumbnail(self, preview_frame: np.ndarray, best_bbox: tuple[int, int, int, int] | None, best_label: str, best_score: float) -> None:
        if not self.output_file or not best_label:
            return
        if self.thumbnail_mode == "first_detection" and self.current_detection_frame_saved:
            return
        if self.thumbnail_mode == "best_detection" and best_score <= self.current_event_best_confidence:
            return
        if self.thumbnail_mode == "first_frame" and self.current_detection_frame_saved:
            return
        scene_thumb_path = self.output_file + ".alert.jpg"
        thumb_frame = self._build_thumbnail_frame(preview_frame, best_bbox, best_label, best_score)
        if cv2.imwrite(scene_thumb_path, thumb_frame):
            app_log("worker", "event thumbnail created", camera=str(self.camera.get("name", self.index)), source="worker", level="INFO", details=f"path={scene_thumb_path} type=scene")
            self.current_event_scene_thumbnail_path = scene_thumb_path
            self.current_event_thumbnail_path = scene_thumb_path
            self.current_thumbnail_ts = time.time()
            self.current_detection_frame_saved = True
            app_log("worker", "event thumbnail assigned", camera=str(self.camera.get("name", self.index)), source="worker", level="INFO", details=f"path={scene_thumb_path}")
            if best_score >= self.current_event_best_confidence:
                self.current_event_best_confidence = float(best_score)
                self.current_event_best_frame = thumb_frame
        else:
            app_log("warning", "alert thumbnail write failed", camera=str(self.camera.get("name", self.index)), source="worker", level="WARNING", details=scene_thumb_path)
            return

    def _should_start_recording_now(self) -> bool:
        return self.pending_positive_hits >= max(1, self.required_hits_to_start_recording)

    def _should_end_detection_now(self, now_ts: float) -> bool:
        if self.pending_miss_count < max(1, self.required_misses_to_end_detection) or self.detection_last_seen_ts <= 0:
            return False
        if now_ts - self.detection_last_seen_ts < float(self.lost_seconds):
            return False
        if self.recording:
            if now_ts - self.detection_last_seen_ts < float(self.lost_seconds + self.post_seconds):
                return False
            if self.recording_started_ts > 0 and now_ts - self.recording_started_ts < float(self.min_record_seconds):
                return False
        return True

    def _effective_detect_fps(self, elapsed_window: float) -> float:
        return 0.0 if elapsed_window <= 0 else float(self.inference_count / elapsed_window)

    def _start_recording_session(self, raw_frame: np.ndarray, preview_frame: np.ndarray, best_label: str, best_score: float, best_bbox: tuple[int, int, int, int] | None, stream_fps: float, detect_fps: float) -> bool:
        if not self.enable_recording or not self.record_lock.acquire(blocking=False):
            return False
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(self.output_dir, f"nagranie_{self.camera['name']}_{timestamp}.mp4")
        self.current_event_metadata_path = self.output_file + ".json"
        h, w = raw_frame.shape[:2]
        self.current_writer_fps = self._compute_effective_writer_fps(stream_fps)
        self.writer_fps = self.current_writer_fps
        self.record_thread = RecordingThread(self.output_file, w, h, self.current_writer_fps)
        self.record_thread.start()
        self.recording = True
        self.is_recording_active = True
        self.recording_started_ts = time.monotonic()
        self.state.recording_started_ts = self.recording_started_ts
        self.current_event_start_ts = time.time()
        self.current_event_label = best_label or "object"
        self.current_event_confidence = float(best_score)
        self.current_event_best_confidence = float(best_score)
        self.current_event_detection_count = 0
        self.current_event_confidence_sum = 0.0
        self.current_event_max_confidence = 0.0
        if self.record_start_mode == "include_prerecord_first":
            for buffer_frame in list(self.prerecord_buffer):
                self.record_thread.write(buffer_frame)
            self.record_thread.write(raw_frame)
        else:
            self.record_thread.write(raw_frame)
        self._update_event_thumbnail(preview_frame, best_bbox, self.current_event_label, best_score)
        scene_thumb_path = self.current_event_scene_thumbnail_path or self.current_event_thumbnail_path
        self.current_event_scene_thumbnail_path = scene_thumb_path
        self.current_event_thumbnail_path = scene_thumb_path
        meta = self._build_recording_meta(
            filepath=self.output_file, thumb_path=scene_thumb_path, label=self.current_event_label, confidence=self.current_event_confidence,
            alert_thumb=scene_thumb_path, scene_thumb=scene_thumb_path,
            event_start_ts=self.current_event_start_ts, writer_fps=self.current_writer_fps, source_fps=self.source_fps, detect_fps=detect_fps,
            frames_written=0, dropped_frames=0, thumbnail_ts=self.current_thumbnail_ts, thumbnail_mode=self.thumbnail_mode,
            preview_role_at_start=self.preview_role, overload_degraded_at_start=self.is_overload_degraded,
        )
        self._save_recording_metadata(meta)
        self.record_signal.emit("start", self.output_file)
        app_log("recording", "recording session started", camera=str(self.camera.get("name", self.index)), source="worker", level="INFO", details=f"file={self.output_file} writer_fps={self.current_writer_fps:.2f}")
        return True

    def _finalize_recording_session(self) -> None:
        if not self.output_file:
            return
        frames_written = dropped_frames = queue_peak = 0
        if self.record_thread:
            thread_stopped = self.record_thread.stop()
            if not thread_stopped:
                app_log("warning", "recording thread did not stop cleanly", camera=str(self.camera.get("name", self.index)), source="worker", level="WARNING", details=f"file={self.output_file}")
            frames_written = self.record_thread.frames_written
            dropped_frames = self.record_thread.dropped_frames
            queue_peak = self.record_thread.queue_peak
            self.record_thread = None
        event_end_ts = time.time(); event_start_ts = self.current_event_start_ts or event_end_ts
        duration = max(0.0, event_end_ts - event_start_ts)
        detection_count = int(self.current_event_detection_count)
        avg_conf = float(self.current_event_confidence_sum / detection_count) if detection_count > 0 else 0.0
        max_conf = float(self.current_event_max_confidence)
        event_elapsed = max(0.001, event_end_ts - (self.current_event_start_ts or event_end_ts))
        scene_thumb_path = self.current_event_scene_thumbnail_path or self.current_event_thumbnail_path
        self.current_event_scene_thumbnail_path = scene_thumb_path
        self.current_event_thumbnail_path = scene_thumb_path
        meta = self._build_recording_meta(
            filepath=self.output_file, thumb_path=scene_thumb_path, label=self.current_event_label or "object",
            alert_thumb=scene_thumb_path, scene_thumb=scene_thumb_path,
            confidence=self.current_event_confidence, event_start_ts=event_start_ts, writer_fps=self.current_writer_fps,
            source_fps=self.source_fps, detect_fps=self._effective_detect_fps(event_elapsed), frames_written=frames_written,
            dropped_frames=dropped_frames, thumbnail_ts=self.current_thumbnail_ts, thumbnail_mode=self.thumbnail_mode,
            event_end_ts=event_end_ts, recording_duration=duration, detection_count=detection_count, max_confidence=max_conf,
            avg_confidence=avg_conf, stream_fps=self.stream_fps, preview_role_at_start=self.preview_role,
            overload_degraded_at_start=self.is_overload_degraded, measured_capture_fps=self.stream_fps,
            effective_detect_fps=self._effective_detect_fps(max(time.monotonic() - self.state.stream_start_ts, 1.0)),
            preview_frames_dropped=self.state.preview_frames_dropped_total, skipped_inference_cycles=self.state.skipped_inference_cycles,
            app_overload_mode=self.app_overload_mode, recorder_queue_peak=queue_peak,
        )
        self._save_recording_metadata(meta)
        app_log("worker", "event finalized", camera=str(self.camera.get("name", self.index)), source="worker", level="INFO", details=f"scene_thumb={scene_thumb_path}")
        self.record_signal.emit("stop", self.output_file)
        app_log("recording", "recording session finalized", camera=str(self.camera.get("name", self.index)), source="worker", level="INFO", details=f"frames_written={frames_written} dropped_frames={dropped_frames} queue_peak={queue_peak}")
        if dropped_frames > 0:
            app_log("warning", "recorder dropped frames", camera=str(self.camera.get("name", self.index)), source="worker", level="WARNING", details=f"dropped_frames={dropped_frames}")
        self.recording = False
        self.is_recording_active = False
        self.output_file = None
        self.recording_started_ts = 0.0
        self.pending_positive_hits = 0
        self.pending_miss_count = 0
        self.current_event_best_confidence = 0.0
        self.current_event_best_frame = None
        self.current_event_scene_thumbnail_path = ""
        self.current_event_thumbnail_path = ""
        self.current_event_metadata_path = ""
        self.current_event_label = ""
        self.current_event_confidence = 0.0
        self.current_event_start_ts = 0.0
        self.current_writer_fps = 0.0
        self.current_detection_frame_saved = False
        self.current_thumbnail_ts = 0.0
        self.current_event_detection_count = 0
        self.current_event_confidence_sum = 0.0
        self.current_event_max_confidence = 0.0
        with suppress(RuntimeError):
            self.record_lock.release()

    def _capture_next_frame(self, frame: np.ndarray, now_mono: float) -> tuple[np.ndarray, np.ndarray]:
        self.state.last_capture_ts = now_mono
        self.state.frames_captured += 1
        self.last_frame_ts = now_mono
        self.frame_counter += 1
        self._stream_fps_window.append(now_mono)
        self._get_effective_stream_fps()
        raw_frame = frame
        preview_frame = frame
        self.prerecord_buffer.append(raw_frame)
        return raw_frame, preview_frame

    def _maybe_run_inference(self, raw_frame: np.ndarray, now_mono: float) -> tuple[Any | None, bool, str, float, tuple[int, int, int, int] | None, list[tuple[int, int, int, int, str, float, tuple[int, int, int]]]]:
        detected = False
        best_label = ""
        best_score = 0.0
        best_bbox = None
        overlays: list[tuple[int, int, int, int, str, float, tuple[int, int, int]]] = []

        run_inference = self.enable_detection or self.draw_overlays
        if not run_inference:
            return None, detected, best_label, best_score, best_bbox, overlays

        detect_fps = max(1e-3, self.fps * (1.0 if self.recording else self.detect_fps_factor))
        interval = 1.0 / detect_fps
        if self.state.next_inference_due_ts <= 0:
            self.state.next_inference_due_ts = now_mono
        if now_mono < self.state.next_inference_due_ts:
            return None, detected, best_label, best_score, best_bbox, overlays

        next_due, skipped = _advance_next_due(now_mono, self.state.next_inference_due_ts, interval)
        self.state.skipped_inference_cycles += skipped

        try:
            result = self.model.predict(raw_frame)
        except Exception as exc:
            app_log("error", "model prediction failure", camera=str(self.camera.get("name", self.index)), source="worker", level="ERROR", details=f"{exc}\n\n{traceback.format_exc()}")
            self.error_signal.emit("Błąd predykcji modelu", self.index)
            return None, detected, best_label, best_score, best_bbox, overlays
        self.state.last_inference_ts = now_mono
        self.state.inferences_run += 1
        self.inference_count += 1
        self.state.next_inference_due_ts = max(next_due + interval, now_mono + interval * 0.1)

        source_size = _extract_image_size(result)
        for obj in result.results:
            label = obj.get("label", "").lower(); confidence = float(obj.get("confidence", obj.get("score", 1.0))); bbox = obj.get("bbox")
            if not label or bbox is None:
                continue
            scaled = _scale_bbox(bbox, raw_frame.shape, source_size)
            if self.draw_overlays and confidence >= self.confidence_threshold_draw and label in self.visible_classes_lower:
                overlays.append((*scaled, label, confidence, _label_color(label)))
            if self.enable_detection and self._is_within_schedule() and label in self.record_classes_lower and confidence >= self.confidence_threshold_record:
                detected = True
                if confidence > best_score:
                    best_score, best_label, best_bbox = confidence, label, scaled
        return result, detected, best_label, best_score, best_bbox, overlays

    @staticmethod
    def _resize_for_preview(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
        if frame is None or getattr(frame, "size", 0) == 0:
            return frame
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return frame
        max_w = max(1, int(max_width))
        max_h = max(1, int(max_height))
        scale = min(1.0, max_w / float(w), max_h / float(h))
        if scale >= 0.999:
            return frame
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _maybe_emit_preview(self, preview_frame: np.ndarray, overlays: list[tuple[int, int, int, int, str, float, tuple[int, int, int]]], now_mono: float) -> None:
        interval = _preview_interval_for_role(self.preview_role, self.preview_fps_main, self.preview_fps_thumb, self.preview_pause_when_hidden)
        if interval == float("inf"):
            self.state.preview_frames_dropped_total += 1
            return
        if self.state.last_preview_emit_ts and now_mono - self.state.last_preview_emit_ts < interval:
            self.state.dropped_preview_frames += 1
            self.state.preview_frame_skip_counter += 1
            self.state.preview_frames_dropped_total += 1
            return
        if self.preview_role in {"thumb", "hidden"} and self.state.preview_frame_skip_counter < (2 if self.app_overload_mode else 1):
            self.state.preview_frame_skip_counter += 1
            self.state.preview_frames_dropped_total += 1
            return
        self.state.preview_frame_skip_counter = 0
        main_emit_frame = preview_frame
        should_draw = (
            bool(overlays)
            and self.draw_overlays
            and not (self.preview_role == "hidden")
            and not (self.app_overload_mode and self.overload_disable_nonessential_overlays and not self.recording)
            and ((self.state.frames_emitted % max(1, self.overlay_stride)) == 0)
        )
        if should_draw:
            main_emit_frame = preview_frame.copy()
            for x1, y1, x2, y2, label, confidence, color in overlays:
                cv2.rectangle(main_emit_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(main_emit_frame, f"{label}: {confidence * 100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        target_main_w = int(max(1, round(self.preview_main_max_width * self.preview_resolution_factor)))
        target_main_h = int(max(1, round(self.preview_main_max_height * self.preview_resolution_factor)))
        target_thumb_w = int(max(1, round(self.preview_thumb_max_width * self.preview_resolution_factor)))
        target_thumb_h = int(max(1, round(self.preview_thumb_max_height * self.preview_resolution_factor)))
        main_emit_frame = self._resize_for_preview(main_emit_frame, target_main_w, target_main_h)
        thumb_emit_frame = self._resize_for_preview(main_emit_frame, target_thumb_w, target_thumb_h)
        if self.preview_role in {"thumb", "hidden"}:
            main_emit_frame = thumb_emit_frame

        self.main_preview_signal.emit(main_emit_frame, self.index)
        self.thumb_preview_signal.emit(thumb_emit_frame, self.index)
        self.state.last_preview_emit_ts = now_mono
        self.state.frames_emitted += 1

    def _maybe_enqueue_record_frame(self, raw_frame: np.ndarray, now_mono: float) -> None:
        if self.recording and self.record_thread:
            prev_drop = int(self.record_thread.dropped_frames)
            self.record_thread.write(raw_frame)
            if self.record_thread.dropped_frames > prev_drop and not self._record_queue_full_warned:
                self._record_queue_full_warned = True
                app_log("warning", "recorder queue full", camera=str(self.camera.get("name", self.index)), source="worker", level="WARNING", details=f"queue_peak={self.record_thread.queue_peak}")
            if self._should_end_detection_now(now_mono):
                self._finalize_recording_session()
        elif self.detection_active and self._should_end_detection_now(now_mono):
            self.detection_active = False
            self.pending_positive_hits = 0
            app_log("worker", "detection ended", camera=str(self.camera.get("name", self.index)), source="worker", level="INFO")

    def _maybe_log_metrics(self, detection_interval: float) -> None:
        now = time.monotonic()
        if self.state.last_metrics_log_ts and now - self.state.last_metrics_log_ts < max(4.0, float(self.performance_log_interval_s)):
            return

        elapsed = max(1e-6, now - (self.state.metrics_window_started_ts or now))
        queue_size = self.record_thread.queue.qsize() if self.record_thread else 0
        dropped_total = self.record_thread.dropped_frames if self.record_thread else 0
        dropped_delta = _dropped_frames_delta(dropped_total, self.state.metrics_dropped_frames)

        cpu_wall = max(1e-6, now - (self.state.metrics_last_cpu_wall_ts or now))
        cpu_proc_now = time.process_time()
        cpu_proc_delta = max(0.0, cpu_proc_now - self.state.metrics_last_cpu_process_ts)
        cpu_percent = float(max(0.0, min(100.0, (cpu_proc_delta / cpu_wall) * 100.0)))

        metrics = _build_metrics_payload(
            capture_fps=_aggregate_fps(self.state.frames_captured - self.state.metrics_frames_captured, elapsed),
            infer_fps=_aggregate_fps(self.state.inferences_run - self.state.metrics_inferences_run, elapsed),
            preview_emit_fps=_aggregate_fps(self.state.frames_emitted - self.state.metrics_frames_emitted, elapsed),
            ui_render_ms=0.0,
            queue_size=int(queue_size),
            dropped_frames=int(dropped_total),
            cpu_percent=cpu_percent,
            rss_mb=_rss_mb(),
        )

        logger.info(
            "performance camera=%s mode=%s overload=%s capture_fps=%.2f infer_fps=%.2f preview_emit_fps=%.2f ui_render_ms=%.2f queue_size=%s dropped_frames=%s cpu_percent=%.1f rss_mb=%.1f detection_interval=%.3f dropped_delta=%s",
            self.camera.get("name", self.index),
            self.preview_role,
            "on" if self.app_overload_mode else "off",
            metrics["capture_fps"],
            metrics["infer_fps"],
            metrics["preview_emit_fps"],
            metrics["ui_render_ms"],
            metrics["queue_size"],
            metrics["dropped_frames"],
            metrics["cpu_percent"],
            metrics["rss_mb"],
            detection_interval,
            dropped_delta,
        )
        app_log(
            "performance",
            "worker metrics summary",
            camera=str(self.camera.get("name", self.index)),
            source="worker",
            level="INFO",
            details=(
                f"mode={self.preview_role} overload={'on' if self.app_overload_mode else 'off'} "
                f"capture_fps={float(metrics['capture_fps']):.2f} infer_fps={float(metrics['infer_fps']):.2f} "
                f"preview_emit_fps={float(metrics['preview_emit_fps']):.2f} ui_render_ms={float(metrics['ui_render_ms']):.2f} "
                f"queue_size={int(metrics['queue_size'])} dropped_frames={int(metrics['dropped_frames'])} "
                f"cpu_percent={float(metrics['cpu_percent']):.1f} rss_mb={float(metrics['rss_mb']):.1f}"
            ),
        )

        self.state.metrics_window_started_ts = now
        self.state.metrics_frames_captured = self.state.frames_captured
        self.state.metrics_inferences_run = self.state.inferences_run
        self.state.metrics_frames_emitted = self.state.frames_emitted
        self.state.metrics_dropped_frames = int(dropped_total)
        self.state.metrics_last_cpu_wall_ts = now
        self.state.metrics_last_cpu_process_ts = cpu_proc_now
        self.state.last_metrics_log_ts = now

    def _maybe_emit_heartbeat(self) -> None:
        now = time.monotonic()
        if self.state.last_heartbeat_ts and now - self.state.last_heartbeat_ts < 10.0:
            return
        queue_size = self.record_thread.queue.qsize() if self.record_thread else 0
        dropped = self.record_thread.dropped_frames if self.record_thread else 0
        since_detect = (now - self.detection_last_seen_ts) if self.detection_last_seen_ts > 0 else -1.0
        elapsed = max(1e-6, now - (self.state.metrics_window_started_ts or now))
        status = {
            "stream_fps": float(self.stream_fps),
            "detect_fps": float(max(0.0, self.fps * (1.0 if self.recording else self.detect_fps_factor))),
            "writer_fps": float(self.current_writer_fps or self.writer_fps),
            "capture_fps": _aggregate_fps(self.state.frames_captured - self.state.metrics_frames_captured, elapsed),
            "infer_fps": _aggregate_fps(self.state.inferences_run - self.state.metrics_inferences_run, elapsed),
            "preview_emit_fps": _aggregate_fps(self.state.frames_emitted - self.state.metrics_frames_emitted, elapsed),
            "ui_render_ms": 0.0,
            "queue_size": int(queue_size),
            "dropped_frames": int(dropped),
            "cpu_percent": 0.0,
            "rss_mb": _rss_mb(),
            "recording_active": bool(self.recording),
            "preview_role": self.preview_role,
            "overload_degraded": bool(self.is_overload_degraded),
            "last_detection_seconds": float(since_detect),
            "preview_frames_dropped": int(self.state.preview_frames_dropped_total),
            "skipped_inference_cycles": int(self.state.skipped_inference_cycles),
            "stream_error_active": bool(self.error_counter > 0),
        }
        status["heartbeat_ts"] = float(time.monotonic())
        self.worker_status_signal.emit(str(self.camera.get("name", self.index)), status)
        self.state.last_heartbeat_ts = now

    def _handle_stream_failure(self, exc: Exception) -> None:
        self.error_counter += 1
        message = str(exc).lower()
        if "401" in message or "unauthorized" in message or "auth" in message:
            msg = "Auth/401"
        elif "timed out" in message or "timeout" in message:
            msg = "Timeout"
        elif "name or service not known" in message or "getaddrinfo" in message or "dns" in message:
            msg = "DNS"
        elif "connection refused" in message:
            msg = "Connection refused"
        elif "no route to host" in message:
            msg = "No route to host"
        else:
            msg = str(exc)
        app_log("error", f"stream failure: {msg}", camera=str(self.camera.get("name", self.index)), source="worker", level="ERROR", details=f"{exc}\n\n{traceback.format_exc()}")
        self.error_signal.emit(msg, self.index)

    def run(self) -> None:
        if not self._acquire_worker_slot():
            self.status_signal.emit("Zduplikowany worker zablokowany", self.index)
            return
        try:
            app_log("worker", "CameraWorker run v2 active", camera=str(self.camera.get("name", self.index)), source="worker", level="INFO", details=f"worker_index={self.index} camera_key={self._camera_worker_key()}")
            while not self.stop_signal:
                connected = False
                src = self.camera.get("rtsp", "")
                if self.camera.get("type") == "usb":
                    with suppress(Exception):
                        src = int(src)
                try:
                    self.status_signal.emit("Łączenie…", self.index)
                    app_log("worker", "stream connect attempt", camera=str(self.camera.get("name", self.index)), source="worker", level="INFO", details=str(src))
                    self.state.stream_start_ts = time.monotonic()
                    with degirum_tools.open_video_stream(src) as stream:
                        self._current_stream = stream
                        stream_fps = float(stream.get(cv2.CAP_PROP_FPS) or 0.0)
                        if stream_fps <= 1e-2:
                            stream_fps = 30.0
                        self.stream_fps = stream_fps
                        self.source_fps = float(self.rtsp_fps if self.rtsp_fps > 0 else stream_fps)
                        self._sync_prerecord_buffer()
                        source_fps = self.rtsp_fps if self.rtsp_fps > 0 else None
                        self.state.next_inference_due_ts = 0.0
                        if not self._runtime_limit_logged and (self.rtsp_fps > 0 or self.fps <= 2):
                            self._runtime_limit_logged = True
                            app_log("performance", "camera runtime limited by config", camera=str(self.camera.get("name", self.index)), source="worker", level="INFO", details=f"rtsp_fps={self.rtsp_fps} detect_fps={self.fps}")

                        video_iter = iter(degirum_tools.video_source(stream, fps=source_fps))
                        frame_retry_count = 0
                        iterator_restart_count = 0
                        while not self.stop_signal:
                            try:
                                frame = next(video_iter)
                            except StopIteration:
                                iterator_restart_count += 1
                                app_log("warning", "restart video iterator", camera=str(self.camera.get("name", self.index)), source="worker", level="WARNING", details=f"reason=stop_iteration restart_count={iterator_restart_count}")
                                if iterator_restart_count >= 3:
                                    app_log("warning", "reconnect stream", camera=str(self.camera.get("name", self.index)), source="worker", level="WARNING", details="reason=iterator_exhausted")
                                    break
                                video_iter = iter(degirum_tools.video_source(stream, fps=source_fps))
                                QThread.msleep(20)
                                continue
                            except Exception as frame_exc:
                                frame_retry_count += 1
                                self.error_counter += 1
                                app_log("warning", "frame read failed", camera=str(self.camera.get("name", self.index)), source="worker", level="WARNING", details=f"retry_count={frame_retry_count} error={frame_exc}")
                                if frame_retry_count <= 3:
                                    app_log("warning", "retry frame", camera=str(self.camera.get("name", self.index)), source="worker", level="WARNING", details=f"attempt={frame_retry_count}")
                                    QThread.msleep(40)
                                    continue
                                iterator_restart_count += 1
                                app_log("warning", "restart video iterator", camera=str(self.camera.get("name", self.index)), source="worker", level="WARNING", details=f"reason=frame_exception restart_count={iterator_restart_count}")
                                frame_retry_count = 0
                                if iterator_restart_count >= 3:
                                    app_log("warning", "reconnect stream", camera=str(self.camera.get("name", self.index)), source="worker", level="WARNING", details=f"reason=frame_exception error={frame_exc}")
                                    break
                                video_iter = iter(degirum_tools.video_source(stream, fps=source_fps))
                                QThread.msleep(80)
                                continue

                            frame_retry_count = 0
                            iterator_restart_count = 0
                            now_mono = time.monotonic()
                            if now_mono - self.last_frame_ts > self.stream_stall_seconds and self.last_frame_ts > 0:
                                self.last_stream_reset_ts = now_mono
                                self.error_counter += 1
                                app_log("warning", "stream stall detected", camera=str(self.camera.get("name", self.index)), source="worker", level="WARNING")
                                break
                            if self.stop_signal:
                                break
                            if frame is None:
                                self.error_signal.emit("Brak sygnału: pusta klatka", self.index)
                                self.error_counter += 1
                                app_log("warning", "empty frame received", camera=str(self.camera.get("name", self.index)), source="worker", level="WARNING", details=f"consecutive_errors={self.error_counter}")
                                if self.error_counter > 10:
                                    self.last_stream_reset_ts = time.monotonic()
                                    break
                                continue
                            if self.restart_requested:
                                self.restart_requested = False
                                break

                            self.error_counter = 0
                            if not connected:
                                self.status_signal.emit("Połączono", self.index)
                                app_log("worker", "stream connected", camera=str(self.camera.get("name", self.index)), source="worker", level="INFO")
                                connected = True

                            raw_frame, preview_frame = self._capture_next_frame(frame, now_mono)
                            inference_result, detected, best_label, best_score, best_bbox, overlays = self._maybe_run_inference(raw_frame, now_mono)

                            if detected:
                                self.positive_detection_count += 1
                                self.state.positive_detections += 1
                                self.detection_last_seen_ts = now_mono
                                self.state.last_detection_ts = now_mono
                                self.pending_miss_count = 0
                                self.pending_positive_hits += 1
                                if not self.detection_active:
                                    app_log("worker", "detection became active", camera=str(self.camera.get("name", self.index)), source="worker", level="INFO")
                                self.detection_active = True
                                if not self.recording and self._should_start_recording_now():
                                    started = self._start_recording_session(raw_frame, preview_frame, best_label or "object", best_score, best_bbox, stream_fps, float(self.fps))
                                    if started:
                                        self.alert_signal.emit({
                                            "camera": self.camera["name"], "label": best_label or "object", "confidence": float(best_score),
                                            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            "frame": self._make_detection_overlay_frame(preview_frame, best_bbox, best_label or "object", best_score),
                                            "filepath": self.output_file or "", "thumb": self.current_event_thumbnail_path,
                                            "alert_thumb": self.current_event_scene_thumbnail_path,
                                        })
                                if self.recording:
                                    self.current_event_confidence = max(self.current_event_confidence, best_score)
                                    self.current_event_detection_count += 1
                                    self.current_event_confidence_sum += float(best_score)
                                    self.current_event_max_confidence = max(self.current_event_max_confidence, float(best_score))
                                    self._update_event_thumbnail(preview_frame, best_bbox, best_label or "object", best_score)
                            else:
                                if inference_result is not None:
                                    self.pending_miss_count += 1
                                    if not self.recording:
                                        self.pending_positive_hits = 0

                            self._maybe_enqueue_record_frame(raw_frame, now_mono)
                            self._maybe_emit_preview(preview_frame, overlays, now_mono)
                            detection_interval = 1.0 / max(1e-3, self.fps * (1.0 if self.recording else self.detect_fps_factor))
                            self._maybe_log_metrics(detection_interval)
                            self._maybe_emit_heartbeat()

                        with suppress(Exception):
                            stream.release()

                except Exception as exc:  # pragma: no cover
                    self._current_stream = None
                    logger.exception("Worker stream failure")
                    app_log("error", "worker stream exception", camera=str(self.camera.get("name", self.index)), source="worker", level="ERROR", details=f"{exc}\n\n{traceback.format_exc()}")
                    self._handle_stream_failure(exc)
                    if self.error_counter > 10:
                        QThread.msleep(2000)
                        self.error_counter = 0

                if self.recording:
                    self._finalize_recording_session()
                self._current_stream = None
                if self.stop_signal:
                    break
                QThread.msleep(300)
        finally:
            self._release_worker_slot()

    def stop(self, timeout_ms: int = 3500) -> bool:
        app_log("worker", "worker stop begin", camera=str(self.camera.get("name", self.index)), source="worker", level="INFO", details=f"timeout_ms={timeout_ms}")
        self.stop_signal = True
        if self.recording:
            self._finalize_recording_session()
        stream = self._current_stream
        if stream is not None:
            with suppress(Exception):
                stream.release()
        self.wait(timeout_ms)
        stopped = not self.isRunning()
        details = worker_stop_timeout_details(str(self.camera.get("name", self.index)), timeout_ms)
        app_log("worker" if stopped else "warning", "worker stop success" if stopped else "worker stop timeout", camera=str(self.camera.get("name", self.index)), source="worker", level="INFO" if stopped else "WARNING", details=details)
        return stopped


__all__ = [
    "CameraWorker",
    "RecordingThread",
    "_advance_next_due",
    "_preview_interval_for_role",
]
