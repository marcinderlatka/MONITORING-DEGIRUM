"""Thread workers handling RTSP streams and recordings."""

from __future__ import annotations

import datetime
import json
import os
import time
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
    DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN,
    DEFAULT_PRE_SECONDS,
    DEFAULT_RECORD_PATH,
    DEFAULT_RECORD_START_MODE,
    DEFAULT_REQUIRED_HITS_TO_START_RECORDING,
    DEFAULT_REQUIRED_MISSES_TO_END_DETECTION,
    DEFAULT_RTSP_FPS,
    DEFAULT_THUMBNAIL_MODE,
    RECORD_CLASSES,
    VISIBLE_CLASSES,
)
from .recordings import build_recording_sidecar_metadata
from .runtime_helpers import compute_effective_writer_fps
from .storage import update_recordings_catalog

LABEL_COLORS = {
    "person": (0, 0, 255),
    "car": (255, 0, 0),
    "cat": (0, 255, 255),
    "dog": (255, 255, 0),
    "bird": (0, 255, 0),
}
PALETTE = [(255, 0, 255), (0, 165, 255), (255, 255, 0), (0, 255, 255), (255, 0, 0), (0, 255, 0)]


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

    def stop(self) -> None:
        self.running = False
        self.wait()


class CameraWorker(QThread):
    frame_signal = pyqtSignal(object, int)
    alert_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str, int)
    status_signal = pyqtSignal(str, int)
    record_signal = pyqtSignal(str, str)
    worker_status_signal = pyqtSignal(str, object)

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
        self.record_start_mode = str(self.camera.get("record_start_mode", DEFAULT_RECORD_START_MODE))
        self.required_hits_to_start_recording = int(self.camera.get("required_hits_to_start_recording", DEFAULT_REQUIRED_HITS_TO_START_RECORDING))
        self.required_misses_to_end_detection = int(self.camera.get("required_misses_to_end_detection", DEFAULT_REQUIRED_MISSES_TO_END_DETECTION))
        self.min_record_seconds = int(self.camera.get("min_record_seconds", DEFAULT_MIN_RECORD_SECONDS))

        self.preview_fps_main = float(self.camera.get("preview_fps_main", DEFAULT_PREVIEW_FPS_MAIN))
        self.preview_fps_thumb = float(self.camera.get("preview_fps_thumb", DEFAULT_PREVIEW_FPS_THUMB))
        self.preview_pause_when_hidden = bool(self.camera.get("preview_pause_when_hidden", DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN))
        self.preview_role = "thumb"
        self.is_overload_degraded = False
        self.app_overload_mode = False
        self.overload_disable_nonessential_overlays = bool(self.camera.get("overload_disable_nonessential_overlays", DEFAULT_OVERLOAD_DISABLE_NONESSENTIAL_OVERLAYS))
        self.detect_fps_factor = 1.0
        self.is_recording_active = False

        rec_path = str(self.camera.get("record_path", DEFAULT_RECORD_PATH))
        self.output_dir = os.path.join(rec_path, self.camera.get("name", "camera"))
        os.makedirs(self.output_dir, exist_ok=True)

        self.recording = False
        self.record_thread: RecordingThread | None = None
        self.output_file: str | None = None
        self.stop_signal = False
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

    def set_preview_role(self, role: str) -> None:
        self.preview_role = role if role in {"main", "thumb", "hidden"} else "thumb"

    def set_overload_state(self, overload_active: bool, detect_fps_factor: float | None = None, thumb_preview_fps: float | None = None, disable_overlays: bool | None = None) -> None:
        self.app_overload_mode = bool(overload_active)
        self.is_overload_degraded = bool(overload_active and self.preview_role != "main")
        if detect_fps_factor is not None:
            self.detect_fps_factor = float(max(0.2, min(1.0, detect_fps_factor)))
        else:
            self.detect_fps_factor = 1.0
        if thumb_preview_fps is not None and thumb_preview_fps > 0:
            self.preview_fps_thumb = float(thumb_preview_fps)
        if disable_overlays is not None:
            self.overload_disable_nonessential_overlays = bool(disable_overlays)

    @staticmethod
    def _crop_with_margin(frame: np.ndarray, bbox: tuple[int, int, int, int] | None, margin_ratio: float = 0.15, min_size: int = 20) -> np.ndarray:
        if bbox is None:
            return frame
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1)); y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
        bw, bh = max(0, x2 - x1), max(0, y2 - y1)
        if bw < min_size or bh < min_size:
            return frame
        mx, my = int(bw * margin_ratio), int(bh * margin_ratio)
        crop = frame[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
        return frame if crop.size == 0 else crop

    def _build_thumbnail_frame(self, preview_frame: np.ndarray, best_bbox: tuple[int, int, int, int] | None, best_label: str, best_score: float) -> np.ndarray:
        cropped = self._crop_with_margin(preview_frame, best_bbox)
        resized = cv2.resize(cropped, (320, 240), interpolation=cv2.INTER_AREA)
        return self._make_detection_overlay_frame(resized, None, best_label, best_score)

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
            print(f"[prerecord] camera={self.camera.get('name', self.index)} pre_seconds={self.pre_seconds} buffer_fps_basis={basis:.2f} buffer_maxlen={maxlen}")

    def _make_detection_overlay_frame(self, frame: np.ndarray, bbox: tuple[int, int, int, int] | None, label: str, confidence: float) -> np.ndarray:
        canvas = frame.copy()
        if bbox:
            x1, y1, x2, y2 = bbox
            color = _label_color(label)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"{label}: {confidence * 100:.1f}%", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return canvas

    def _build_recording_meta(self, **kwargs: Any) -> dict:
        event_time = datetime.datetime.fromtimestamp(float(kwargs["event_start_ts"]))
        return build_recording_sidecar_metadata(
            camera=self.camera.get("name", ""), label=kwargs["label"], confidence=kwargs["confidence"],
            event_time=event_time.strftime("%Y-%m-%d %H:%M:%S"), filepath=kwargs["filepath"], thumb=kwargs["thumb_path"],
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
            print("Nie zapisano metadanych:", exc)
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
        thumb_path = self.output_file + ".jpg"
        thumb_frame = self._build_thumbnail_frame(preview_frame, best_bbox, best_label, best_score)
        if cv2.imwrite(thumb_path, thumb_frame):
            self.current_event_thumbnail_path = thumb_path
            self.current_thumbnail_ts = time.time()
            self.current_detection_frame_saved = True
            if best_score >= self.current_event_best_confidence:
                self.current_event_best_confidence = float(best_score)
                self.current_event_best_frame = thumb_frame

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
        self.record_thread.write(raw_frame)
        if self.record_start_mode == "include_prerecord_first":
            for buffer_frame in list(self.prerecord_buffer):
                self.record_thread.write(buffer_frame)
        self._update_event_thumbnail(preview_frame, best_bbox, self.current_event_label, best_score)
        meta = self._build_recording_meta(
            filepath=self.output_file, thumb_path=self.current_event_thumbnail_path, label=self.current_event_label, confidence=self.current_event_confidence,
            event_start_ts=self.current_event_start_ts, writer_fps=self.current_writer_fps, source_fps=self.source_fps, detect_fps=detect_fps,
            frames_written=0, dropped_frames=0, thumbnail_ts=self.current_thumbnail_ts, thumbnail_mode=self.thumbnail_mode,
            preview_role_at_start=self.preview_role, overload_degraded_at_start=self.is_overload_degraded,
        )
        self._save_recording_metadata(meta)
        self.record_signal.emit("start", self.output_file)
        return True

    def _finalize_recording_session(self) -> None:
        if not self.output_file:
            return
        frames_written = dropped_frames = queue_peak = 0
        if self.record_thread:
            self.record_thread.stop()
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
        meta = self._build_recording_meta(
            filepath=self.output_file, thumb_path=self.current_event_thumbnail_path, label=self.current_event_label or "object",
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
        self.record_signal.emit("stop", self.output_file)
        self.recording = False
        self.is_recording_active = False
        self.output_file = None
        self.recording_started_ts = 0.0
        self.pending_positive_hits = 0
        self.pending_miss_count = 0
        self.current_event_best_confidence = 0.0
        self.current_event_best_frame = None
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
        preview_frame = frame.copy()
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

        result = self.model.predict(raw_frame)
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
        emit_frame = preview_frame
        should_draw = bool(overlays) and self.draw_overlays and not (self.preview_role == "hidden") and not (self.app_overload_mode and self.overload_disable_nonessential_overlays and not self.recording)
        if should_draw:
            emit_frame = preview_frame.copy()
            for x1, y1, x2, y2, label, confidence, color in overlays:
                cv2.rectangle(emit_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(emit_frame, f"{label}: {confidence * 100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        self.frame_signal.emit(emit_frame, self.index)
        self.state.last_preview_emit_ts = now_mono
        self.state.frames_emitted += 1

    def _maybe_enqueue_record_frame(self, raw_frame: np.ndarray, now_mono: float) -> None:
        if self.recording and self.record_thread:
            self.record_thread.write(raw_frame)
            if self._should_end_detection_now(now_mono):
                self._finalize_recording_session()
        elif self.detection_active and self._should_end_detection_now(now_mono):
            self.detection_active = False
            self.pending_positive_hits = 0

    def _maybe_log_metrics(self, detection_interval: float) -> None:
        now = time.monotonic()
        if self.state.last_metrics_log_ts and now - self.state.last_metrics_log_ts < 10.0:
            return
        detect_fps = 1.0 / max(1e-6, detection_interval)
        print(f"[metrics] camera={self.camera.get('name', self.index)} stream_fps={self.stream_fps:.2f} infer_count={self.inference_count} detect_fps_target={detect_fps:.2f} preview_emitted={self.state.frames_emitted} preview_dropped={self.state.preview_frames_dropped_total} skipped_inference={self.state.skipped_inference_cycles} role={self.preview_role}")
        self.state.last_metrics_log_ts = now

    def _maybe_emit_heartbeat(self) -> None:
        now = time.monotonic()
        if self.state.last_heartbeat_ts and now - self.state.last_heartbeat_ts < 10.0:
            return
        queue_size = self.record_thread.queue.qsize() if self.record_thread else 0
        dropped = self.record_thread.dropped_frames if self.record_thread else 0
        since_detect = (now - self.detection_last_seen_ts) if self.detection_last_seen_ts > 0 else -1.0
        status = {
            "stream_fps": float(self.stream_fps),
            "detect_fps": float(max(0.0, self.fps * (1.0 if self.recording else self.detect_fps_factor))),
            "writer_fps": float(self.current_writer_fps or self.writer_fps),
            "queue_size": int(queue_size),
            "dropped_frames": int(dropped),
            "recording_active": bool(self.recording),
            "preview_role": self.preview_role,
            "overload_degraded": bool(self.is_overload_degraded),
            "last_detection_seconds": float(since_detect),
            "preview_frames_dropped": int(self.state.preview_frames_dropped_total),
            "skipped_inference_cycles": int(self.state.skipped_inference_cycles),
            "stream_error_active": bool(self.error_counter > 0),
        }
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
        self.error_signal.emit(msg, self.index)

    def run(self) -> None:
        while not self.stop_signal:
            connected = False
            src = self.camera.get("rtsp", "")
            if self.camera.get("type") == "usb":
                with suppress(Exception):
                    src = int(src)
            try:
                self.status_signal.emit("Łączenie…", self.index)
                self.state.stream_start_ts = time.monotonic()
                with degirum_tools.open_video_stream(src) as stream:
                    stream_fps = float(stream.get(cv2.CAP_PROP_FPS) or 0.0)
                    if stream_fps <= 1e-2:
                        stream_fps = 30.0
                    self.stream_fps = stream_fps
                    self.source_fps = float(self.rtsp_fps if self.rtsp_fps > 0 else stream_fps)
                    self._sync_prerecord_buffer()
                    source_fps = self.rtsp_fps if self.rtsp_fps > 0 else None
                    self.state.next_inference_due_ts = 0.0

                    for frame in degirum_tools.video_source(stream, fps=source_fps):
                        now_mono = time.monotonic()
                        if now_mono - self.last_frame_ts > self.stream_stall_seconds and self.last_frame_ts > 0:
                            self.last_stream_reset_ts = now_mono
                            self.error_counter += 1
                            break
                        if self.stop_signal:
                            break
                        if frame is None:
                            self.error_signal.emit("Brak sygnału: pusta klatka", self.index)
                            self.error_counter += 1
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
                            self.detection_active = True
                            if not self.recording and self._should_start_recording_now():
                                started = self._start_recording_session(raw_frame, preview_frame, best_label or "object", best_score, best_bbox, stream_fps, float(self.fps))
                                if started:
                                    self.alert_signal.emit({
                                        "camera": self.camera["name"], "label": best_label or "object", "confidence": float(best_score),
                                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "frame": self._make_detection_overlay_frame(preview_frame, best_bbox, best_label or "object", best_score),
                                        "filepath": self.output_file or "", "thumb": self.current_event_thumbnail_path,
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

            except Exception as exc:  # pragma: no cover
                self._handle_stream_failure(exc)
                if self.error_counter > 10:
                    QThread.msleep(2000)
                    self.error_counter = 0

            if self.recording:
                self._finalize_recording_session()
            if self.stop_signal:
                break
            QThread.msleep(300)

    def stop(self) -> None:
        self.stop_signal = True
        if self.recording:
            self._finalize_recording_session()
        self.wait(3000)
        if self.isRunning():
            print(f"CameraWorker {self.camera.get('name', self.index)} did not stop in time")


__all__ = [
    "CameraWorker",
    "RecordingThread",
    "_advance_next_due",
    "_preview_interval_for_role",
]
