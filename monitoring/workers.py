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
    DEFAULT_CONFIDENCE_THRESHOLD_DRAW,
    DEFAULT_CONFIDENCE_THRESHOLD_RECORD,
    DEFAULT_DETECTION_HOURS,
    DEFAULT_DRAW_OVERLAYS,
    DEFAULT_ENABLE_DETECTION,
    DEFAULT_ENABLE_RECORDING,
    DEFAULT_FPS,
    DEFAULT_LOST_SECONDS,
    DEFAULT_POST_SECONDS,
    DEFAULT_PRE_SECONDS,
    DEFAULT_RECORD_PATH,
    DEFAULT_RECORD_START_MODE,
    DEFAULT_REQUIRED_HITS_TO_START_RECORDING,
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
        self.queue: "Queue[np.ndarray]" = Queue(maxsize=240)
        self.running = True
        self.writer = None
        self.dropped_frames = 0
        self.frames_written = 0

    def run(self) -> None:
        self.writer = degirum_tools.VideoWriter(self.filepath, self.width, self.height, self.fps)
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                self.writer.write(frame)
                self.frames_written += 1
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

    def __init__(self, camera: dict, model: Any, index: int = 0) -> None:
        super().__init__()
        self.camera = dict(camera)
        self.model = model
        self.index = index

        self.fps = int(self.camera.get("fps", DEFAULT_FPS))
        self.rtsp_fps = int(self.camera.get("rtsp_fps", DEFAULT_RTSP_FPS))
        legacy_conf = float(self.camera.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
        self.confidence_threshold_draw = float(
            self.camera.get("confidence_threshold_draw", legacy_conf)
        )
        self.confidence_threshold_record = float(
            self.camera.get("confidence_threshold_record", legacy_conf)
        )
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
        self.loop_fps = 0.0

        self.detection_active = False
        self.pending_detection_hits = 0
        self.detection_last_seen_ts = 0.0

        self.current_event_start_ts = 0.0
        self.current_thumbnail_ts = 0.0
        self.current_thumb_path = ""
        self.current_label = "object"
        self.current_confidence = 0.0

        self.inference_count = 0
        self.positive_detection_count = 0
        self.last_metrics_log_ts = 0.0

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

    def _draw_detection_box(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
        label: str,
        score: float,
    ) -> np.ndarray:
        canvas = frame.copy()
        if bbox:
            x1, y1, x2, y2 = bbox
            color = _label_color(label)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas,
                f"{label}: {score * 100:.1f}%",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        return canvas

    def _save_thumbnail(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
        label: str,
        score: float,
    ) -> str:
        if not self.output_file:
            return ""
        thumb_path = self.output_file + ".jpg"
        thumb_frame = self._draw_detection_box(frame, bbox, label, score)
        try:
            cv2.imwrite(thumb_path, thumb_frame)
            return thumb_path
        except Exception as exc:
            print("Nie zapisano miniatury:", exc)
            return ""

    def _finalize_metadata(self) -> None:
        if not self.output_file:
            return
        frames_written = self.record_thread.frames_written if self.record_thread else 0
        dropped = self.record_thread.dropped_frames if self.record_thread else 0
        event_time = datetime.datetime.fromtimestamp(self.current_event_start_ts).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        meta = build_recording_sidecar_metadata(
            camera=self.camera.get("name", ""),
            label=self.current_label,
            confidence=self.current_confidence,
            event_time=event_time,
            filepath=self.output_file,
            thumb=self.current_thumb_path,
            source_fps=self.source_fps,
            writer_fps=self.writer_fps,
            detect_fps=float(self.fps),
            event_start_ts=self.current_event_start_ts,
            thumbnail_ts=self.current_thumbnail_ts,
            frames_written=frames_written,
            dropped_frames=dropped,
            thumbnail_mode=self.thumbnail_mode,
            inference_count=self.inference_count,
            positive_detection_count=self.positive_detection_count,
        )
        try:
            with open(self.output_file + ".json", "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)
        except Exception as exc:
            print("Nie zapisano metadanych:", exc)
        update_recordings_catalog(dict(meta))

    def _start_recording(
        self,
        raw_frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
        label: str,
        score: float,
        now_mono: float,
    ) -> bool:
        if not self.enable_recording:
            return False
        if not self.record_lock.acquire(blocking=False):
            return False

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(
            self.output_dir,
            f"nagranie_{self.camera['name']}_{timestamp}.mp4",
        )
        h, w = raw_frame.shape[:2]
        self.writer_fps = compute_effective_writer_fps(self.rtsp_fps, self.loop_fps, self.stream_fps)
        self.record_thread = RecordingThread(self.output_file, w, h, self.writer_fps)
        self.record_thread.start()

        if self.record_start_mode == "include_prerecord_first":
            for buffer_frame in list(self.prerecord_buffer):
                self.record_thread.write(buffer_frame)
        self.record_thread.write(raw_frame)

        self.recording = True
        self.current_event_start_ts = time.time()
        self.current_thumbnail_ts = self.current_event_start_ts
        self.current_label = label or "object"
        self.current_confidence = float(score)
        self.current_thumb_path = self._save_thumbnail(raw_frame, bbox, self.current_label, score)
        self.detection_last_seen_ts = now_mono
        print(
            f"[record][{self.camera.get('name', self.index)}] start source_fps={self.source_fps:.2f} "
            f"writer_fps={self.writer_fps:.2f} detect_fps={self.fps:.2f}"
        )
        self.record_signal.emit("start", self.output_file)
        return True

    def _stop_recording(self) -> None:
        if self.record_thread:
            self.record_thread.stop()
            self._finalize_metadata()
            self.record_thread = None
        if self.recording:
            self.record_signal.emit("stop", self.output_file or "")
            print(
                f"[record][{self.camera.get('name', self.index)}] stop frames_written="
                f"{self.inference_count} inferences={self.inference_count} positives={self.positive_detection_count}"
            )
        self.recording = False
        self.output_file = None
        if self.record_lock.locked():
            self.record_lock.release()
        self.detection_active = False
        self.pending_detection_hits = 0

    def _log_metrics_if_needed(self) -> None:
        now = time.monotonic()
        if self.last_metrics_log_ts == 0.0:
            self.last_metrics_log_ts = now
            return
        elapsed = now - self.last_metrics_log_ts
        if elapsed < 5.0:
            return
        queue_size = self.record_thread.queue.qsize() if self.record_thread else 0
        dropped = self.record_thread.dropped_frames if self.record_thread else 0
        print(
            f"[diag][{self.camera.get('name', self.index)}] source_fps={self.source_fps:.2f} "
            f"writer_fps={self.writer_fps:.2f} effective_detect_fps={self.fps:.2f} "
            f"inference_count={self.inference_count} positives={self.positive_detection_count} "
            f"queue={queue_size} dropped={dropped} recording={self.recording}"
        )
        self.last_metrics_log_ts = now

    def run(self) -> None:
        while not self.stop_signal:
            try:
                self.status_signal.emit("Łączenie…", self.index)
                connected = False
                src = self.camera.get("rtsp")
                if self.camera.get("type") == "usb":
                    try:
                        src = int(src)
                    except Exception:
                        pass
                with degirum_tools.open_video_stream(src) as stream:
                    stream_fps = float(stream.get(cv2.CAP_PROP_FPS) or 0.0)
                    if stream_fps <= 1e-2:
                        stream_fps = 30.0
                    self.stream_fps = stream_fps
                    self.source_fps = float(self.rtsp_fps if self.rtsp_fps > 0 else stream_fps)
                    self.prerecord_buffer = deque(
                        maxlen=max(1, int(self.pre_seconds * max(1.0, self.source_fps)))
                    )

                    last_inference_time = 0.0
                    detection_interval = 1.0 / max(1, self.fps)

                    source_fps = self.rtsp_fps if self.rtsp_fps > 0 else None
                    for frame in degirum_tools.video_source(stream, fps=source_fps):
                        if self.stop_signal:
                            break
                        if frame is None:
                            self.error_signal.emit("Brak sygnału: pusta klatka", self.index)
                            continue

                        now = time.monotonic()
                        if self.last_frame_ts > 0:
                            delta = now - self.last_frame_ts
                            if delta > 0:
                                self.loop_fps = 1.0 / delta
                        self.last_frame_ts = now

                        if not connected:
                            self.status_signal.emit("Połączono", self.index)
                            connected = True

                        raw_frame = frame
                        preview_frame = frame.copy()
                        self.prerecord_buffer.append(raw_frame)

                        run_inference = self.enable_detection or self.draw_overlays
                        inference_result = None
                        if run_inference and (
                            last_inference_time == 0.0 or now - last_inference_time >= detection_interval
                        ):
                            last_inference_time = now
                            inference_result = self.model.predict(raw_frame)
                            self.inference_count += 1

                        detected = False
                        best_label = ""
                        best_score = 0.0
                        best_bbox = None

                        if inference_result is not None:
                            source_size = _extract_image_size(inference_result)
                            overlays: list[
                                tuple[int, int, int, int, str, float, tuple[int, int, int]]
                            ] = []

                            for obj in inference_result.results:
                                label = obj.get("label", "").lower()
                                confidence = obj.get("confidence", obj.get("score", 1.0))
                                bbox = obj.get("bbox")
                                if not label or bbox is None:
                                    continue

                                scaled = _scale_bbox(bbox, raw_frame.shape, source_size)
                                if (
                                    self.draw_overlays
                                    and confidence >= self.confidence_threshold_draw
                                    and label in self.visible_classes_lower
                                ):
                                    overlays.append((*scaled, label, confidence, _label_color(label)))

                                if (
                                    self.enable_detection
                                    and self._is_within_schedule()
                                    and label in self.record_classes_lower
                                    and confidence >= self.confidence_threshold_record
                                ):
                                    detected = True
                                    if confidence > best_score:
                                        best_score = float(confidence)
                                        best_label = label
                                        best_bbox = scaled

                            if self.draw_overlays and overlays:
                                for x1, y1, x2, y2, label, confidence, color in overlays:
                                    cv2.rectangle(preview_frame, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(
                                        preview_frame,
                                        f"{label}: {confidence * 100:.1f}%",
                                        (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        color,
                                        2,
                                    )

                        if detected:
                            self.positive_detection_count += 1
                            self.detection_last_seen_ts = now
                            self.pending_detection_hits += 1
                            self.detection_active = True

                            if (
                                not self.recording
                                and self.pending_detection_hits >= max(1, self.required_hits_to_start_recording)
                            ):
                                started = self._start_recording(
                                    raw_frame,
                                    best_bbox,
                                    best_label or "object",
                                    best_score,
                                    now,
                                )
                                if started:
                                    alert = {
                                        "camera": self.camera["name"],
                                        "label": best_label or "object",
                                        "confidence": float(best_score),
                                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "frame": self._draw_detection_box(
                                            preview_frame, best_bbox, best_label or "object", best_score
                                        ),
                                        "filepath": self.output_file or "",
                                        "thumb": self.current_thumb_path,
                                    }
                                    self.alert_signal.emit(alert)
                        else:
                            self.pending_detection_hits = 0

                        if self.recording and self.record_thread:
                            self.record_thread.write(raw_frame)
                            end_after = self.detection_last_seen_ts + self.lost_seconds + self.post_seconds
                            if self.detection_last_seen_ts > 0 and now > end_after:
                                self._stop_recording()
                        elif self.detection_active:
                            if self.detection_last_seen_ts > 0 and now > (
                                self.detection_last_seen_ts + self.lost_seconds + self.post_seconds
                            ):
                                self.detection_active = False

                        self.frame_signal.emit(preview_frame, self.index)
                        self._log_metrics_if_needed()

            except Exception as exc:  # pragma: no cover - interacts with hardware
                message = str(exc).lower()
                if "401" in message or "unauthorized" in message or "auth" in message:
                    msg = "Auth/401"
                elif "timed out" in message or "timeout" in message:
                    msg = "Timeout"
                elif (
                    "name or service not known" in message
                    or "getaddrinfo" in message
                    or "dns" in message
                ):
                    msg = "DNS"
                elif "connection refused" in message:
                    msg = "Connection refused"
                elif "no route to host" in message:
                    msg = "No route to host"
                else:
                    msg = str(exc)
                self.error_signal.emit(msg, self.index)

            if self.recording:
                self._stop_recording()

            if self.stop_signal:
                break
            QThread.msleep(300)

        if self.recording:
            self._stop_recording()

    def stop(self) -> None:
        self.stop_signal = True
        self._stop_recording()
        self.wait(3000)
        if self.isRunning():
            print(f"CameraWorker {self.camera.get('name', self.index)} did not stop in time")


__all__ = ["CameraWorker", "RecordingThread"]
