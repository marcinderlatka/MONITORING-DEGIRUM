"""Thread workers handling RTSP streams and recordings."""

from __future__ import annotations

import datetime
import json
import os
from collections import deque
from contextlib import suppress
from queue import Empty, Queue
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
    RECORD_CLASSES,
    VISIBLE_CLASSES,
)
from .storage import update_recordings_catalog


class RecordingThread(QThread):
    def __init__(self, filepath: str, width: int, height: int, fps: int) -> None:
        super().__init__()
        self.filepath = filepath
        self.width = width
        self.height = height
        self.fps = fps
        self.queue: "Queue[np.ndarray]" = Queue()
        self._stop = False
        self.writer = None

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
            self.queue.put(frame)

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
        self.confidence_threshold = float(
            self.camera.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
        )
        self.draw_overlays = bool(self.camera.get("draw_overlays", DEFAULT_DRAW_OVERLAYS))
        self.enable_detection = bool(self.camera.get("enable_detection", DEFAULT_ENABLE_DETECTION))
        self.enable_recording = bool(self.camera.get("enable_recording", DEFAULT_ENABLE_RECORDING))
        self.detection_hours = str(self.camera.get("detection_hours", DEFAULT_DETECTION_HOURS))
        self.visible_classes = list(self.camera.get("visible_classes", VISIBLE_CLASSES))
        self.record_classes = list(self.camera.get("record_classes", RECORD_CLASSES))
        self.pre_seconds = int(self.camera.get("pre_seconds", DEFAULT_PRE_SECONDS))
        self.post_seconds = int(self.camera.get("post_seconds", DEFAULT_POST_SECONDS))
        rec_path = str(self.camera.get("record_path", DEFAULT_RECORD_PATH))
        self.output_dir = os.path.join(rec_path, self.camera.get("name", "camera"))
        os.makedirs(self.output_dir, exist_ok=True)

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

        self.restart_requested = False

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
        self.restart_requested = True

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
                stream = degirum_tools.predict_stream(
                    self.model, src, fps=self.fps, analyzers=False
                )
                for inference_result in stream:
                    if self.stop_signal:
                        break
                    if self.restart_requested:
                        self.restart_requested = False
                        break

                    frame = getattr(inference_result, "image", None)
                    if frame is None:
                        self.error_signal.emit("Brak sygnału: pusta klatka", self.index)
                        continue

                    if not connected:
                        self.status_signal.emit("Połączono", self.index)
                        connected = True

                    self.prerecord_buffer.append(frame.copy())
                    self.frame = frame.copy()

                    detected = False
                    best_label = ""
                    best_score = 0.0

                    best_bbox = None

                    for obj in inference_result.results:
                        label = obj.get("label", "").lower()
                        confidence = obj.get("confidence", obj.get("score", 1.0))
                        bbox = obj.get("bbox")
                        if not label or bbox is None:
                            continue

                        if (
                            self.draw_overlays
                            and confidence >= self.confidence_threshold
                            and label in [c.lower() for c in self.visible_classes]
                        ):
                            x1, y1, x2, y2 = map(int, bbox)
                            color = (
                                (0, 255, 0)
                                if label in [c.lower() for c in self.record_classes]
                                else (255, 0, 0)
                            )
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)
                            text = f"{label}: {confidence * 100:.1f}%"
                            cv2.putText(
                                self.frame,
                                text,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                color,
                                2,
                            )

                        if (
                            self.enable_detection
                            and self._is_within_schedule()
                            and label in [c.lower() for c in self.record_classes]
                            and confidence >= self.confidence_threshold
                        ):
                            detected = True
                            if confidence > best_score:
                                best_score = confidence
                                best_label = label
                                best_bbox = bbox

                    if detected:
                        if not self.detection_active:
                            self.no_detection_frames = 0
                            self.post_countdown_frames = 0
                            alert_frame = self.frame.copy()
                            if best_bbox:
                                x1, y1, x2, y2 = map(int, best_bbox)
                                color = (
                                    (0, 255, 0)
                                    if (best_label or "")
                                    and best_label in [c.lower() for c in self.record_classes]
                                    else (0, 0, 255)
                                )
                                cv2.rectangle(alert_frame, (x1, y1), (x2, y2), color, 2)
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
                                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                    self.output_file = os.path.join(
                                        self.output_dir, f"nagranie_{self.camera['name']}_{timestamp}.mp4"
                                    )
                                    h, w = self.frame.shape[:2]
                                    self.record_thread = RecordingThread(
                                        self.output_file, w, h, self.fps
                                    )
                                    self.record_thread.start()
                                    for buffer_frame in list(self.prerecord_buffer):
                                        self.record_thread.write(buffer_frame)
                                    self.recording = True
                                    self.no_detection_frames = 0
                                    self.post_countdown_frames = 0
                                    thumb_path = self.output_file + ".jpg"
                                    try:
                                        cv2.imwrite(thumb_path, self.frame)
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
                                        with open(self.output_file + ".json", "w", encoding="utf-8") as handle:
                                            json.dump(meta, handle, indent=2)
                                    except Exception as exc:
                                        print("Nie zapisano metadanych:", exc)
                                    catalog_entry = dict(meta)
                                    catalog_entry.setdefault("filepath", self.output_file)
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
                                if self.no_detection_frames < int(self.lost_seconds * self.fps):
                                    self.no_detection_frames += 1
                                else:
                                    self.post_countdown_frames += 1
                                    if self.post_countdown_frames >= int(self.post_seconds * self.fps):
                                        self._stop_recording()
                            else:
                                if self.no_detection_frames < int(self.lost_seconds * self.fps):
                                    self.no_detection_frames += 1
                                else:
                                    self.post_countdown_frames += 1
                                    if self.post_countdown_frames >= int(self.post_seconds * self.fps):
                                        self.detection_active = False
                                        self.no_detection_frames = 0
                                        self.post_countdown_frames = 0
                        else:
                            self.no_detection_frames = 0
                            self.post_countdown_frames = 0

                    if self.recording and self.record_thread:
                        self.record_thread.write(self.frame)

                    self.frame_signal.emit(self.frame, self.index)

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
        self.wait(2000)
        if self.isRunning():
            self.terminate()
            self.wait()


__all__ = ["CameraWorker", "RecordingThread"]
