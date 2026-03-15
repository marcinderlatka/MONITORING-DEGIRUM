"""Thread workers handling RTSP streams and recordings."""

from __future__ import annotations

import datetime
import json
import os
import time
from collections import deque
from contextlib import suppress
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
    DEFAULT_POST_SECONDS,
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

PALETTE = [
    (255, 0, 255),
    (0, 165, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
]


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
        value = getattr(result, attr, None)
        size = _normalize_size(value)
        if size:
            return size
    if isinstance(result, dict):
        size = _normalize_size(result.get("image_size") or result.get("input_image_size"))
        if size:
            return size
    return None


def _scale_bbox(
    bbox: list[float] | tuple[float, ...],
    frame_shape: tuple[int, ...],
    source_size: tuple[int, int] | None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, bbox)
    h, w = frame_shape[:2]
    if source_size:
        src_w, src_h = source_size
        if src_w and src_h and (src_w != w or src_h != h):
            x_scale = w / src_w
            y_scale = h / src_h
            x1 *= x_scale
            x2 *= x_scale
            y1 *= y_scale
            y2 *= y_scale
    if 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0:
        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return int(max(0, x1)), int(max(0, y1)), int(min(w - 1, x2)), int(min(h - 1, y2))


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
        self.required_hits_to_start_recording = int(
            self.camera.get("required_hits_to_start_recording", DEFAULT_REQUIRED_HITS_TO_START_RECORDING)
        )
        self.required_misses_to_end_detection = int(
            self.camera.get("required_misses_to_end_detection", DEFAULT_REQUIRED_MISSES_TO_END_DETECTION)
        )
        self.min_record_seconds = int(self.camera.get("min_record_seconds", DEFAULT_MIN_RECORD_SECONDS))

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
        self.last_heartbeat_ts = 0.0
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

    @staticmethod
    def _crop_with_margin(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
        margin_ratio: float = 0.15,
        min_size: int = 20,
    ) -> np.ndarray:
        if bbox is None:
            return frame
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        if bw < min_size or bh < min_size:
            return frame
        mx = int(bw * margin_ratio)
        my = int(bh * margin_ratio)
        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w, x2 + mx)
        cy2 = min(h, y2 + my)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return frame
        return crop

    def _build_thumbnail_frame(
        self,
        preview_frame: np.ndarray,
        best_bbox: tuple[int, int, int, int] | None,
        best_label: str,
        best_score: float,
    ) -> np.ndarray:
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

    def _emit_heartbeat_if_needed(self) -> None:
        now = time.monotonic()
        if self.last_heartbeat_ts and now - self.last_heartbeat_ts < 10.0:
            return
        queue_size = self.record_thread.queue.qsize() if self.record_thread else 0
        dropped = self.record_thread.dropped_frames if self.record_thread else 0
        since_detect = (now - self.detection_last_seen_ts) if self.detection_last_seen_ts > 0 else -1.0
        status = {
            "stream_fps": float(self.stream_fps),
            "detect_fps": float(max(0.0, self.fps)),
            "writer_fps": float(self.current_writer_fps or self.writer_fps),
            "queue_size": int(queue_size),
            "dropped_frames": int(dropped),
            "recording_active": bool(self.recording),
            "last_detection_seconds": float(since_detect),
        }
        self.worker_status_signal.emit(str(self.camera.get("name", self.index)), status)
        self.last_heartbeat_ts = now

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
            spec = self.detection_hours.replace(" ", "")
            for part in spec.split(";"):
                if not part:
                    continue
                a, b = part.split("-")
                ha, ma = map(int, a.split(":"))
                hb, mb = map(int, b.split(":"))
                start = datetime.time(ha, ma)
                end = datetime.time(hb, mb)
                if start <= end:
                    if start <= now <= end:
                        return True
                else:
                    if now >= start or now <= end:
                        return True
            return False
        except Exception:
            return True

    def _compute_effective_writer_fps(self, stream_fps: float) -> float:
        value = compute_effective_writer_fps(self.rtsp_fps, float(self.fps), stream_fps)
        return float(max(1.0, value))

    def _make_detection_overlay_frame(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
        label: str,
        confidence: float,
    ) -> np.ndarray:
        canvas = frame.copy()
        if bbox:
            x1, y1, x2, y2 = bbox
            color = _label_color(label)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas,
                f"{label}: {confidence * 100:.1f}%",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        return canvas

    def _build_recording_meta(self, **kwargs: Any) -> dict:
        event_time = datetime.datetime.fromtimestamp(float(kwargs["event_start_ts"]))
        return build_recording_sidecar_metadata(
            camera=self.camera.get("name", ""),
            label=kwargs["label"],
            confidence=kwargs["confidence"],
            event_time=event_time.strftime("%Y-%m-%d %H:%M:%S"),
            filepath=kwargs["filepath"],
            thumb=kwargs["thumb_path"],
            source_fps=kwargs["source_fps"],
            writer_fps=kwargs["writer_fps"],
            detect_fps=kwargs["detect_fps"],
            event_start_ts=kwargs["event_start_ts"],
            thumbnail_ts=kwargs["thumbnail_ts"],
            frames_written=kwargs["frames_written"],
            dropped_frames=kwargs["dropped_frames"],
            thumbnail_mode=kwargs["thumbnail_mode"],
            inference_count=self.inference_count,
            positive_detection_count=self.positive_detection_count,
            record_start_mode=self.record_start_mode,
            min_record_seconds=self.min_record_seconds,
            required_hits_to_start_recording=self.required_hits_to_start_recording,
            required_misses_to_end_detection=self.required_misses_to_end_detection,
            event_end_ts=kwargs.get("event_end_ts", 0.0),
            recording_duration=kwargs.get("recording_duration", 0.0),
            detection_count=kwargs.get("detection_count", 0),
            max_confidence=kwargs.get("max_confidence", 0.0),
            avg_confidence=kwargs.get("avg_confidence", 0.0),
            stream_fps=kwargs.get("stream_fps", self.stream_fps),
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

    def _update_event_thumbnail(
        self,
        preview_frame: np.ndarray,
        best_bbox: tuple[int, int, int, int] | None,
        best_label: str,
        best_score: float,
    ) -> None:
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
        if self.pending_miss_count < max(1, self.required_misses_to_end_detection):
            return False
        if self.detection_last_seen_ts <= 0:
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
        if elapsed_window <= 0:
            return 0.0
        return float(self.inference_count / elapsed_window)

    def _start_recording_session(
        self,
        raw_frame: np.ndarray,
        preview_frame: np.ndarray,
        best_label: str,
        best_score: float,
        best_bbox: tuple[int, int, int, int] | None,
        stream_fps: float,
        detect_fps: float,
    ) -> bool:
        if not self.enable_recording:
            return False
        if not self.record_lock.acquire(blocking=False):
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
        self.recording_started_ts = time.monotonic()
        self.current_event_start_ts = time.time()
        self.current_event_label = best_label or "object"
        self.current_event_confidence = float(best_score)
        self.current_event_best_confidence = float(best_score)
        self.current_event_detection_count = 0
        self.current_event_confidence_sum = 0.0
        self.current_event_max_confidence = 0.0

        # Detection-first default behavior.
        self.record_thread.write(raw_frame)
        if self.record_start_mode == "include_prerecord_first":
            for buffer_frame in list(self.prerecord_buffer):
                self.record_thread.write(buffer_frame)

        self._update_event_thumbnail(preview_frame, best_bbox, self.current_event_label, best_score)
        meta = self._build_recording_meta(
            filepath=self.output_file,
            thumb_path=self.current_event_thumbnail_path,
            label=self.current_event_label,
            confidence=self.current_event_confidence,
            event_start_ts=self.current_event_start_ts,
            writer_fps=self.current_writer_fps,
            source_fps=self.source_fps,
            detect_fps=detect_fps,
            frames_written=0,
            dropped_frames=0,
            thumbnail_ts=self.current_thumbnail_ts,
            thumbnail_mode=self.thumbnail_mode,
            event_end_ts=0.0,
            recording_duration=0.0,
            detection_count=0,
            max_confidence=0.0,
            avg_confidence=0.0,
            stream_fps=self.stream_fps,
        )
        self._save_recording_metadata(meta)
        self.record_signal.emit("start", self.output_file)
        return True

    def _finalize_recording_session(self) -> None:
        if not self.output_file:
            return
        frames_written = 0
        dropped_frames = 0
        if self.record_thread:
            self.record_thread.stop()
            frames_written = self.record_thread.frames_written
            dropped_frames = self.record_thread.dropped_frames
            self.record_thread = None

        event_end_ts = time.time()
        event_start_ts = self.current_event_start_ts or event_end_ts
        duration = max(0.0, event_end_ts - event_start_ts)
        detection_count = int(self.current_event_detection_count)
        avg_conf = float(self.current_event_confidence_sum / detection_count) if detection_count > 0 else 0.0
        meta = self._build_recording_meta(
            filepath=self.output_file,
            thumb_path=self.current_event_thumbnail_path,
            label=self.current_event_label or "object",
            confidence=self.current_event_confidence,
            event_start_ts=event_start_ts,
            writer_fps=self.current_writer_fps or self.writer_fps,
            source_fps=self.source_fps,
            detect_fps=float(self.fps),
            frames_written=frames_written,
            dropped_frames=dropped_frames,
            thumbnail_ts=self.current_thumbnail_ts or self.current_event_start_ts,
            thumbnail_mode=self.thumbnail_mode,
            event_end_ts=event_end_ts,
            recording_duration=duration,
            detection_count=detection_count,
            max_confidence=float(self.current_event_max_confidence),
            avg_confidence=avg_conf,
            stream_fps=self.stream_fps,
        )
        self._save_recording_metadata(meta)
        self.record_signal.emit("stop", self.output_file)

        self.recording = False
        self.output_file = None
        if self.record_lock.locked():
            self.record_lock.release()
        self.detection_active = False
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
        self.recording_started_ts = 0.0
        self.current_event_detection_count = 0
        self.current_event_confidence_sum = 0.0
        self.current_event_max_confidence = 0.0

    def _log_metrics_if_needed(self, detect_interval: float) -> None:
        now = time.monotonic()
        if self.last_metrics_log_ts == 0.0:
            self.last_metrics_log_ts = now
            return
        elapsed = now - self.last_metrics_log_ts
        if elapsed < 5.0:
            return
        queue_size = self.record_thread.queue.qsize() if self.record_thread else 0
        dropped = self.record_thread.dropped_frames if self.record_thread else 0
        since_detect = (now - self.detection_last_seen_ts) if self.detection_last_seen_ts > 0 else -1.0
        print(
            f"[perf] camera={self.camera.get('name', self.index)} "
            f"stream_fps={self.stream_fps:.2f} detect_fps={self._effective_detect_fps(elapsed):.2f} "
            f"writer_fps={self.current_writer_fps or self.writer_fps:.2f} queue_size={queue_size} "
            f"dropped_frames={dropped} detections_last_window={self.positive_detection_count} "
            f"detect_interval={detect_interval:.3f} since_last_detection={since_detect:.2f}"
        )
        self.last_metrics_log_ts = now
        self.inference_count = 0
        self.positive_detection_count = 0

    def run(self) -> None:
        while not self.stop_signal:
            try:
                self.status_signal.emit("Łączenie…", self.index)
                connected = False
                self.stream_start_ts = time.monotonic()
                self.last_frame_ts = self.stream_start_ts
                self.frame_counter = 0
                self._stream_fps_window.clear()
                src = self.camera.get("rtsp")
                if self.camera.get("type") == "usb":
                    with suppress(Exception):
                        src = int(src)
                with degirum_tools.open_video_stream(src) as stream:
                    stream_fps = float(stream.get(cv2.CAP_PROP_FPS) or 0.0)
                    if stream_fps <= 1e-2:
                        stream_fps = 30.0
                    self.stream_fps = stream_fps
                    self.source_fps = float(self.rtsp_fps if self.rtsp_fps > 0 else stream_fps)
                    self.prerecord_buffer = deque(maxlen=max(1, int(self.pre_seconds * max(1.0, self.source_fps))))
                    last_inference_time = 0.0
                    detection_interval = 1.0 / max(1, self.fps)
                    source_fps = self.rtsp_fps if self.rtsp_fps > 0 else None

                    for frame in degirum_tools.video_source(stream, fps=source_fps):
                        now_mono = time.monotonic()
                        if now_mono - self.last_frame_ts > self.stream_stall_seconds:
                            print(
                                f"[warn] camera={self.camera.get('name', self.index)} stream stalled for "
                                f"{now_mono - self.last_frame_ts:.2f}s; reopening stream"
                            )
                            self.last_stream_reset_ts = now_mono
                            self.error_counter += 1
                            break
                        if self.stop_signal:
                            break
                        if frame is None:
                            self.error_signal.emit("Brak sygnału: pusta klatka", self.index)
                            self.error_counter += 1
                            if self.error_counter > 10:
                                print(f"[recovery] camera={self.camera.get('name', self.index)} restarting after frame read failures")
                                self.last_stream_reset_ts = time.monotonic()
                                break
                            continue
                        if self.restart_requested:
                            self.restart_requested = False
                            break

                        self.error_counter = 0
                        if self.last_frame_ts > 0:
                            delta = now_mono - self.last_frame_ts
                            if delta > 0:
                                self.loop_fps = 1.0 / delta
                        self.last_frame_ts = now_mono
                        self.frame_counter += 1
                        self._stream_fps_window.append(now_mono)
                        self._get_effective_stream_fps()

                        if not connected:
                            self.status_signal.emit("Połączono", self.index)
                            connected = True

                        raw_frame = frame
                        preview_frame = frame.copy()
                        self.prerecord_buffer.append(raw_frame)

                        run_inference = self.enable_detection or self.draw_overlays
                        inference_result = None
                        if run_inference and (last_inference_time == 0.0 or now_mono - last_inference_time >= detection_interval):
                            last_inference_time = now_mono
                            inference_result = self.model.predict(raw_frame)
                            self.inference_count += 1

                        detected = False
                        best_label = ""
                        best_score = 0.0
                        best_bbox = None

                        if inference_result is not None:
                            source_size = _extract_image_size(inference_result)
                            overlays: list[tuple[int, int, int, int, str, float, tuple[int, int, int]]] = []
                            for obj in inference_result.results:
                                label = obj.get("label", "").lower()
                                confidence = float(obj.get("confidence", obj.get("score", 1.0)))
                                bbox = obj.get("bbox")
                                if not label or bbox is None:
                                    continue
                                scaled = _scale_bbox(bbox, raw_frame.shape, source_size)
                                if self.draw_overlays and confidence >= self.confidence_threshold_draw and label in self.visible_classes_lower:
                                    overlays.append((*scaled, label, confidence, _label_color(label)))
                                if (
                                    self.enable_detection
                                    and self._is_within_schedule()
                                    and label in self.record_classes_lower
                                    and confidence >= self.confidence_threshold_record
                                ):
                                    detected = True
                                    if confidence > best_score:
                                        best_score = confidence
                                        best_label = label
                                        best_bbox = scaled

                            if self.draw_overlays and overlays:
                                for x1, y1, x2, y2, label, confidence, color in overlays:
                                    cv2.rectangle(preview_frame, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(preview_frame, f"{label}: {confidence * 100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        if detected:
                            self.positive_detection_count += 1
                            self.detection_last_seen_ts = now_mono
                            self.pending_miss_count = 0
                            self.pending_positive_hits += 1
                            self.detection_active = True

                            if not self.recording and self._should_start_recording_now():
                                started = self._start_recording_session(
                                    raw_frame,
                                    preview_frame,
                                    best_label or "object",
                                    best_score,
                                    best_bbox,
                                    stream_fps,
                                    float(self.fps),
                                )
                                if started:
                                    alert = {
                                        "camera": self.camera["name"],
                                        "label": best_label or "object",
                                        "confidence": float(best_score),
                                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "frame": self._make_detection_overlay_frame(preview_frame, best_bbox, best_label or "object", best_score),
                                        "filepath": self.output_file or "",
                                        "thumb": self.current_event_thumbnail_path,
                                    }
                                    self.alert_signal.emit(alert)

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

                        if self.recording and self.record_thread:
                            self.record_thread.write(raw_frame)
                            if self._should_end_detection_now(now_mono):
                                self._finalize_recording_session()
                        elif self.detection_active and self._should_end_detection_now(now_mono):
                            self.detection_active = False
                            self.pending_positive_hits = 0

                        self.frame_signal.emit(preview_frame, self.index)
                        self._log_metrics_if_needed(detection_interval)
                        self._emit_heartbeat_if_needed()

            except Exception as exc:  # pragma: no cover
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
                if self.error_counter > 10:
                    print(f"[recovery] camera={self.camera.get('name', self.index)} reopening stream after repeated errors")
                    QThread.msleep(2000)
                    self.error_counter = 0

            if self.recording:
                self._finalize_recording_session()
            if self.stop_signal:
                break
            QThread.msleep(300)

        if self.recording:
            self._finalize_recording_session()

    def stop(self) -> None:
        self.stop_signal = True
        if self.recording:
            self._finalize_recording_session()
        self.wait(3000)
        if self.isRunning():
            print(f"CameraWorker {self.camera.get('name', self.index)} did not stop in time")


__all__ = ["CameraWorker", "RecordingThread"]
