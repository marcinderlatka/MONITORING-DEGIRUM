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
    DEFAULT_POST_SECONDS,
    DEFAULT_PRE_SECONDS,
    DEFAULT_RECORD_PATH,
    DEFAULT_RTSP_FPS,
    RECORD_CLASSES,
    VISIBLE_CLASSES,
)
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
    def __init__(self, filepath: str, width: int, height: int, fps: int) -> None:
        super().__init__()
        self.filepath = filepath
        self.width = width
        self.height = height
        self.fps = fps
        queue_size = max(10, int(max(1, fps) * 3))
        self.queue: "Queue[np.ndarray]" = Queue(maxsize=queue_size)
        self._stop = False
        self.writer = None
        self.dropped_frames = 0

    def run(self) -> None:
        self.writer = degirum_tools.VideoWriter(self.filepath, self.width, self.height, self.fps)
        while not self._stop or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                self.writer.write(frame)
            except Empty:
                pass
        if self.writer:
            with suppress(AttributeError):
                self.writer.release()
            self.writer = None

    def write(self, frame: np.ndarray) -> None:
        if not self._stop:
            try:
                self.queue.put_nowait(frame)
            except Full:
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass
                try:
                    self.queue.put_nowait(frame)
                except Full:
                    pass
                self.dropped_frames += 1

    def stop(self) -> None:
        self._stop = True
        self.wait()


class CameraWorker(QThread):
    frame_signal = pyqtSignal(object, int)  # (np.ndarray BGR, index)
    alert_signal = pyqtSignal(object)  # dict z klatką i metadanymi
    error_signal = pyqtSignal(str, int)  # komunikat, index
    status_signal = pyqtSignal(str, int)  # status tekstowy, index
    record_signal = pyqtSignal(str, str)  # (event, filepath)

    def __init__(self, camera: dict, model: Any, index: int = 0) -> None:
        super().__init__()
        self.camera = dict(camera)
        self.model = model
        self.index = index

        self.fps = int(self.camera.get("fps", DEFAULT_FPS))
        self.rtsp_fps = int(self.camera.get("rtsp_fps", DEFAULT_RTSP_FPS))
        self.confidence_threshold = float(
            self.camera.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
        )
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
        rec_path = str(self.camera.get("record_path", DEFAULT_RECORD_PATH))
        self.output_dir = os.path.join(rec_path, self.camera.get("name", "camera"))
        os.makedirs(self.output_dir, exist_ok=True)

        self.stream_fps = None

        self.recording = False
        self.record_thread: RecordingThread | None = None
        self.output_file: str | None = None
        self.detection_active = False
        self.stop_signal = False
        self.lost_seconds = int(self.camera.get("lost_seconds", DEFAULT_LOST_SECONDS))
        self.record_lock = Lock()
        self.no_detection_frames = 0
        self.post_countdown_frames = 0

        self.prerecord_buffer = deque(maxlen=int(self.pre_seconds * self.fps))
        self.frame: np.ndarray | None = None
        self.last_metrics_log_ts = 0.0
        self.metrics = {
            "capture": 0.0,
            "inference": 0.0,
            "overlay": 0.0,
            "emit": 0.0,
            "record_enqueue": 0.0,
            "frames": 0,
            "recorder_drops": 0,
        }

    def refresh_class_filters(self) -> None:
        self.visible_classes_lower = {c.lower() for c in self.visible_classes}
        self.record_classes_lower = {c.lower() for c in self.record_classes}

    def _log_metrics_if_needed(self) -> None:
        now = time.monotonic()
        if self.last_metrics_log_ts == 0.0:
            self.last_metrics_log_ts = now
            return
        elapsed = now - self.last_metrics_log_ts
        if elapsed < 5.0 or self.metrics["frames"] <= 0:
            return
        frames = max(1, int(self.metrics["frames"]))
        print(
            f"[perf][{self.camera.get('name', self.index)}] "
            f"capture={self.metrics['capture'] / frames * 1000:.1f}ms "
            f"infer={self.metrics['inference'] / frames * 1000:.1f}ms "
            f"overlay={self.metrics['overlay'] / frames * 1000:.1f}ms "
            f"emit={self.metrics['emit'] / frames * 1000:.1f}ms "
            f"enqueue={self.metrics['record_enqueue'] / frames * 1000:.1f}ms "
            f"drops={self.metrics['recorder_drops']}"
        )
        self.last_metrics_log_ts = now
        for key in ("capture", "inference", "overlay", "emit", "record_enqueue"):
            self.metrics[key] = 0.0
        self.metrics["frames"] = 0
        self.metrics["recorder_drops"] = 0

    def _stop_recording(self) -> None:
        if self.record_thread:
            self.record_thread.stop()
            self.record_thread = None
        if self.recording:
            self.record_signal.emit("stop", self.output_file or "")
        self.recording = False
        self.output_file = None
        if self.record_lock.locked():
            self.record_lock.release()
        self.detection_active = False
        self.no_detection_frames = 0
        self.post_countdown_frames = 0

    def set_confidence(self, threshold: float) -> None:
        self.confidence_threshold = float(threshold)
        self.camera["confidence_threshold"] = self.confidence_threshold

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
                    self.prerecord_buffer = deque(
                        maxlen=max(1, int(self.pre_seconds * stream_fps))
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

                        if not connected:
                            self.status_signal.emit("Połączono", self.index)
                            connected = True

                        frame_start = time.perf_counter()
                        raw_frame = frame
                        should_buffer = self.enable_recording or self.detection_active
                        if should_buffer:
                            self.prerecord_buffer.append(raw_frame.copy())
                        preview_frame = raw_frame

                        run_inference = self.enable_detection or self.draw_overlays
                        now = time.monotonic()
                        inference_result = None
                        if run_inference and (
                            last_inference_time == 0.0
                            or now - last_inference_time >= detection_interval
                        ):
                            last_inference_time = now
                            infer_start = time.perf_counter()
                            inference_result = self.model.predict(raw_frame)
                            self.metrics["inference"] += time.perf_counter() - infer_start

                        if inference_result is not None:
                            detected = False
                            best_label = ""
                            best_score = 0.0
                            best_bbox = None
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

                                if (
                                    self.draw_overlays
                                    and confidence >= self.confidence_threshold
                                    and label in self.visible_classes_lower
                                ):
                                    x1, y1, x2, y2 = _scale_bbox(bbox, raw_frame.shape, source_size)
                                    color = _label_color(label)
                                    overlays.append((x1, y1, x2, y2, label, confidence, color))

                                if (
                                    self.enable_detection
                                    and self._is_within_schedule()
                                    and label in self.record_classes_lower
                                    and confidence >= self.confidence_threshold
                                ):
                                    detected = True
                                    if confidence > best_score:
                                        best_score = confidence
                                        best_label = label
                                        best_bbox = _scale_bbox(bbox, raw_frame.shape, source_size)

                            if self.draw_overlays and overlays:
                                overlay_start = time.perf_counter()
                                preview_frame = raw_frame.copy()
                                for x1, y1, x2, y2, label, confidence, color in overlays:
                                    cv2.rectangle(
                                        preview_frame, (x1, y1), (x2, y2), color, 2
                                    )
                                    text = f"{label}: {confidence * 100:.1f}%"
                                    cv2.putText(
                                        preview_frame,
                                        text,
                                        (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        color,
                                        2,
                                    )
                                self.metrics["overlay"] += time.perf_counter() - overlay_start

                            if detected:
                                if not self.detection_active:
                                    self.no_detection_frames = 0
                                    self.post_countdown_frames = 0
                                    alert_frame = frame.copy()
                                    if best_bbox:
                                        x1, y1, x2, y2 = best_bbox
                                        color = _label_color(best_label or "")
                                        cv2.rectangle(
                                            alert_frame, (x1, y1), (x2, y2), color, 2
                                        )
                                        cv2.putText(
                                            alert_frame,
                                            f"{(best_label or 'object')}: {best_score * 100:.1f}%",
                                            (x1, max(20, y1 - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7,
                                            color,
                                            2,
                                        )

                                    alert = {
                                        "camera": self.camera["name"],
                                        "label": best_label or "object",
                                        "confidence": float(best_score),
                                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "frame": alert_frame,
                                        "filepath": "",
                                        "thumb": "",
                                    }
                                    emit_alert = True
                                    if self.enable_recording:
                                        if self.record_lock.acquire(blocking=False):
                                            timestamp = datetime.datetime.now().strftime(
                                                "%Y%m%d_%H%M%S"
                                            )
                                            self.output_file = os.path.join(
                                                self.output_dir,
                                                f"nagranie_{self.camera['name']}_{timestamp}.mp4",
                                            )
                                            h, w = raw_frame.shape[:2]
                                            record_fps = int(max(1, round(stream_fps)))
                                            self.record_thread = RecordingThread(
                                                self.output_file, w, h, record_fps
                                            )
                                            self.record_thread.start()
                                            for buffer_frame in list(self.prerecord_buffer):
                                                self.record_thread.write(buffer_frame)
                                            self.recording = True
                                            self.no_detection_frames = 0
                                            self.post_countdown_frames = 0
                                            thumb_path = self.output_file + ".jpg"
                                            try:
                                                cv2.imwrite(thumb_path, preview_frame)
                                            except Exception as exc:
                                                print("Nie zapisano miniatury:", exc)
                                            alert["filepath"] = self.output_file
                                            alert["thumb"] = thumb_path
                                            meta = {
                                                "camera": alert["camera"],
                                                "label": alert["label"],
                                                "confidence": alert["confidence"],
                                                "time": alert["time"],
                                                "file": self.output_file,
                                                "thumb": thumb_path,
                                            }
                                            try:
                                                with open(
                                                    self.output_file + ".json",
                                                    "w",
                                                    encoding="utf-8",
                                                ) as handle:
                                                    json.dump(meta, handle, indent=2)
                                            except Exception as exc:
                                                print("Nie zapisano metadanych:", exc)
                                            catalog_entry = dict(meta)
                                            catalog_entry.setdefault(
                                                "filepath", self.output_file
                                            )
                                            update_recordings_catalog(catalog_entry)
                                            self.record_signal.emit("start", self.output_file)
                                        else:
                                            emit_alert = False
                                    if emit_alert:
                                        self.alert_signal.emit(alert)
                                    self.detection_active = True
                                else:
                                    self.no_detection_frames = 0
                                    self.post_countdown_frames = 0
                            else:
                                if self.detection_active:
                                    if self.recording:
                                        if self.no_detection_frames < int(
                                            self.lost_seconds * self.fps
                                        ):
                                            self.no_detection_frames += 1
                                        else:
                                            self.post_countdown_frames += 1
                                            if self.post_countdown_frames >= int(
                                                self.post_seconds * self.fps
                                            ):
                                                self._stop_recording()
                                    else:
                                        if self.no_detection_frames < int(
                                            self.lost_seconds * self.fps
                                        ):
                                            self.no_detection_frames += 1
                                        else:
                                            self.post_countdown_frames += 1
                                            if self.post_countdown_frames >= int(
                                                self.post_seconds * self.fps
                                            ):
                                                self.detection_active = False
                                                self.no_detection_frames = 0
                                                self.post_countdown_frames = 0
                                else:
                                    self.no_detection_frames = 0
                                    self.post_countdown_frames = 0
                        self.frame = preview_frame

                        enqueue_start = time.perf_counter()
                        if self.recording and self.record_thread:
                            self.record_thread.write(raw_frame)
                            self.metrics["recorder_drops"] += self.record_thread.dropped_frames
                            self.record_thread.dropped_frames = 0
                        self.metrics["record_enqueue"] += time.perf_counter() - enqueue_start

                        emit_start = time.perf_counter()
                        self.frame_signal.emit(preview_frame, self.index)
                        self.metrics["emit"] += time.perf_counter() - emit_start
                        self.metrics["capture"] += time.perf_counter() - frame_start
                        self.metrics["frames"] += 1
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
