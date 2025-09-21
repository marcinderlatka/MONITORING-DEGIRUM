
# -*- coding: utf-8 -*-
import os
import sys
import json
import csv
import cv2
import datetime
import numpy as np
import re
from collections import deque
from queue import Queue, Empty
from threading import Lock
from glob import glob
import argparse
from contextlib import suppress
import base64
import io
import wave
import uuid

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QListWidget, QListWidgetItem,
    QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QGridLayout,
    QMenu, QFrame, QFileDialog, QDialog, QFormLayout,
    QComboBox, QMessageBox, QDateEdit, QLineEdit, QCheckBox, QStackedWidget,
    QSpinBox, QDoubleSpinBox, QToolButton, QStyle, QSizePolicy
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QSize, QTimer, QDate, QPoint, QRect, QUrl,
    QRunnable, QThreadPool, QObject
)
from PyQt5.QtGui import QImage, QPixmap, QClipboard, QPainter, QFont, QColor, QIcon
from PyQt5 import QtSvg
from PyQt5.QtMultimedia import QSoundEffect
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ICON_DIR = BASE_DIR / "icons"

# --- DeGirum ---
import degirum as dg
import degirum_tools

# Qt platform plugin path (Linux)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

# --- ŚCIEŻKI I KONFIG ---
CONFIG_PATH = "./config.json"
MODELS_PATH = "./models"
ALERTS_HISTORY_PATH = "./alerts_history.json"   # plik historii alertów
LOG_HISTORY_PATH = "./log_history.json"        # plik historii logów
LOG_RETENTION_HOURS = 48

# Domyślne klasy (na sztywno)
VISIBLE_CLASSES = ["person", "car", "cat", "dog", "bird"]
RECORD_CLASSES  = ["person", "car", "cat", "dog", "bird"]

# Domyślne parametry kamer
DEFAULT_MODEL = "yolov5nu_silu_coco--640x640_float_tflite_multidevice_1"
DEFAULT_FPS = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_DRAW_OVERLAYS = True
DEFAULT_ENABLE_DETECTION = True
DEFAULT_ENABLE_RECORDING = True
DEFAULT_DETECTION_HOURS = "00:00-23:59"
DEFAULT_RECORD_PATH = "./nagrania"
DEFAULT_PRE_SECONDS = 5
DEFAULT_POST_SECONDS = 5
DEFAULT_LOST_SECONDS = 10

# --- UTIL: Konfiguracja ---
def _fill_camera_defaults(cam):
    """Uzupełnij brakujące pola kamery domyślnymi wartościami."""
    defaults = {
        "model": DEFAULT_MODEL,
        "fps": DEFAULT_FPS,
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "draw_overlays": DEFAULT_DRAW_OVERLAYS,
        "enable_detection": DEFAULT_ENABLE_DETECTION,
        "enable_recording": DEFAULT_ENABLE_RECORDING,
        "detection_hours": DEFAULT_DETECTION_HOURS,
        "visible_classes": VISIBLE_CLASSES,
        "record_classes": RECORD_CLASSES,
        "record_path": DEFAULT_RECORD_PATH,
        "pre_seconds": DEFAULT_PRE_SECONDS,
        "post_seconds": DEFAULT_POST_SECONDS,
        "lost_seconds": DEFAULT_LOST_SECONDS,
        "type": "rtsp",
    }
    for k, v in defaults.items():
        cam.setdefault(k, v)
    return cam


def list_usb_cameras():
    """Zwróć listę dostępnych kamer USB jako pary (index, nazwa)."""
    devices = []
    for dev in sorted(glob("/dev/video*")):
        try:
            idx = int(os.path.basename(dev).replace("video", ""))
        except ValueError:
            continue
        name_path = f"/sys/class/video4linux/video{idx}/name"
        try:
            with open(name_path, "r") as f:
                name = f.read().strip()
        except Exception:
            name = f"Kamera {idx}"
        devices.append((idx, name))
    return devices


def load_config():
    global LOG_HISTORY_PATH, LOG_RETENTION_HOURS
    if not os.path.exists(CONFIG_PATH):
        cfg = {
            "log_history_path": LOG_HISTORY_PATH,
            "log_retention_hours": LOG_RETENTION_HOURS,
            "cameras": [
                {
                    "name": "kamera1",
                    "rtsp": "rtsp://admin:IBLTSQ@192.168.8.165:554",
                }
            ],
        }
    else:
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)

    LOG_HISTORY_PATH = cfg.get("log_history_path", LOG_HISTORY_PATH)
    LOG_RETENTION_HOURS = int(cfg.get("log_retention_hours", LOG_RETENTION_HOURS))

    for cam in cfg.get("cameras", []):
        _fill_camera_defaults(cam)
    return cfg


def save_config(cfg):
    global LOG_HISTORY_PATH, LOG_RETENTION_HOURS
    for cam in cfg.get("cameras", []):
        _fill_camera_defaults(cam)
    cfg.setdefault("log_history_path", LOG_HISTORY_PATH)
    cfg.setdefault("log_retention_hours", LOG_RETENTION_HOURS)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)

# --- PAMIĘĆ ALERTÓW ---
class AlertMemory:
    """Trwała pamięć alertów z plikiem JSON."""
    def __init__(self, path=ALERTS_HISTORY_PATH, max_items=5000):
        self.path = path
        self.max_items = max_items
        self.items = []
        self.load()

    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.items = data[-self.max_items:]
                    else:
                        self.items = []
            else:
                self.items = []
        except Exception:
            self.items = []

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.items[-self.max_items:], f, indent=2)
        except Exception as e:
            print("Nie udało się zapisać historii alertów:", e)

    def add(self, alert_meta: dict):
        slim = {
            "camera": alert_meta.get("camera", ""),
            "label": alert_meta.get("label", ""),
            "confidence": float(alert_meta.get("confidence", 0.0)),
            "time": alert_meta.get("time", ""),
            "filepath": alert_meta.get("filepath", ""),
            "thumb": alert_meta.get("thumb", ""),
        }
        self.items.append(slim)
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]
        self.save()

    def clear(self):
        self.items = []
        self.save()

    def export_csv(self, csv_path):
        fields = ["time", "camera", "label", "confidence", "filepath"]
        try:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for it in self.items:
                    row = {k: it.get(k, "") for k in fields}
                    w.writerow(row)
            return True, None
        except Exception as e:
            return False, str(e)


# --- BACKEND: Wątek kamery (AI + pre/post record + alerty) ---
class RecordingThread(QThread):
    def __init__(self, filepath, width, height, fps):
        super().__init__()
        self.filepath = filepath
        self.width = width
        self.height = height
        self.fps = fps
        self.queue = Queue()
        self._stop = False
        self.writer = None

    def run(self):
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

    def write(self, frame):
        if not self._stop:
            self.queue.put(frame)

    def stop(self):
        self._stop = True
        self.wait()


class CameraWorker(QThread):
    frame_signal = pyqtSignal(object, int)  # (np.ndarray BGR, index)
    alert_signal = pyqtSignal(object)       # dict z klatką i metadanymi
    error_signal = pyqtSignal(str, int)     # komunikat, index
    status_signal = pyqtSignal(str, int)    # status tekstowy, index
    record_signal = pyqtSignal(str, str)  # (event, filepath)

    def __init__(self, camera, model, index=0):
        super().__init__()
        # pełny słownik kamery
        self.camera = dict(camera)
        self.model = model
        self.index = index

        # lokalne ustawienia (z nadpisaniem globalnych)
        self.fps = int(self.camera.get("fps", DEFAULT_FPS))
        self.confidence_threshold = float(self.camera.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
        self.draw_overlays = bool(self.camera.get("draw_overlays", DEFAULT_DRAW_OVERLAYS))
        self.enable_detection = bool(self.camera.get("enable_detection", DEFAULT_ENABLE_DETECTION))
        self.enable_recording = bool(self.camera.get("enable_recording", DEFAULT_ENABLE_RECORDING))
        self.detection_hours = str(self.camera.get("detection_hours", DEFAULT_DETECTION_HOURS))
        self.visible_classes = list(self.camera.get("visible_classes", VISIBLE_CLASSES))
        self.record_classes = list(self.camera.get("record_classes", RECORD_CLASSES))
        self.pre_seconds = int(self.camera.get("pre_seconds", DEFAULT_PRE_SECONDS))
        self.post_seconds = int(self.camera.get("post_seconds", DEFAULT_POST_SECONDS))
        rec_path = self.camera.get("record_path", DEFAULT_RECORD_PATH)
        self.output_dir = os.path.join(rec_path, self.camera.get("name", "camera"))
        os.makedirs(self.output_dir, exist_ok=True)

        self.recording = False
        self.record_thread = None
        self.output_file = None
        self.detection_active = False
        self.stop_signal = False
        self.lost_seconds = int(self.camera.get("lost_seconds", DEFAULT_LOST_SECONDS))
        self.record_lock = Lock()
        self.no_detection_frames = 0
        self.post_countdown_frames = 0

        self.prerecord_buffer = deque(maxlen=int(self.pre_seconds * self.fps))
        self.frame = None

        # hot-reload
        self.restart_requested = False

    def _stop_recording(self):
        """Stop the recording thread and release related resources."""
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

    # hot reload API
    def set_confidence(self, thr: float):
        self.confidence_threshold = float(thr)
        self.camera["confidence_threshold"] = self.confidence_threshold

    def set_fps(self, fps: int):
        self.fps = int(max(1, fps))
        self.camera["fps"] = self.fps
        self.restart_requested = True

    # Feature toggles (live)
    def set_draw_overlays(self, v: bool):
        self.draw_overlays = bool(v)
        self.camera["draw_overlays"] = self.draw_overlays

    def set_enable_detection(self, v: bool):
        self.enable_detection = bool(v)
        self.camera["enable_detection"] = self.enable_detection

    def set_enable_recording(self, v: bool):
        self.enable_recording = bool(v)
        self.camera["enable_recording"] = self.enable_recording

    def set_detection_schedule(self, hours: str):
        # format "HH:MM-HH:MM(;HH:MM-HH:MM ...)", local time
        self.detection_hours = str(hours or "").strip() or "00:00-23:59"
        self.camera["detection_hours"] = self.detection_hours

    def _is_within_schedule(self):
        try:
            now = datetime.datetime.now().time()
            spec = self.detection_hours.replace(" ", "")
            for part in spec.split(";"):
                if not part:
                    continue
                a, b = part.split("-")
                ha, ma = map(int, a.split(":"))
                hb, mb = map(int, b.split(":"))
                t1 = datetime.time(ha, ma)
                t2 = datetime.time(hb, mb)
                if t1 <= t2:
                    if t1 <= now <= t2:
                        return True
                else:
                    # wraps past midnight
                    if now >= t1 or now <= t2:
                        return True
            return False
        except Exception:
            return True  # fail-open

    def run(self):
        # pętla autoreconnect + hot-restart
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
                for inference_result in degirum_tools.predict_stream(
                    self.model,
                    src,
                    fps=self.fps,
                    analyzers=False
                ):
                    if self.stop_signal:
                        break
                    if self.restart_requested:
                        self.restart_requested = False
                        # wyjdź z pętli strumienia, wróć na górę (z nowym FPS)
                        break

                    frame = getattr(inference_result, "image", None)
                    if frame is None:
                        self.error_signal.emit("Brak sygnału: pusta klatka", self.index)
                        continue

                    if not connected:
                        self.status_signal.emit("Połączono", self.index)
                        connected = True

                    # pre-record: trzymaj czyste klatki
                    self.prerecord_buffer.append(frame.copy())
                    # rysujemy na kopii do podglądu/nagrywania
                    self.frame = frame.copy()

                    detected = False
                    best_label = ""
                    best_score = 0.0

                    for obj in inference_result.results:
                        label = obj.get("label", "").lower()
                        confidence = obj.get("confidence", obj.get("score", 1.0))
                        bbox = obj.get("bbox", None)
                        if not label or bbox is None:
                            continue

                        if self.draw_overlays and confidence >= self.confidence_threshold and label in [c.lower() for c in self.visible_classes]:
                            x1, y1, x2, y2 = map(int, bbox)
                            color = (0, 255, 0) if label in [c.lower() for c in self.record_classes] else (255, 0, 0)
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)
                            text = f"{label}: {confidence * 100:.1f}%"
                            cv2.putText(self.frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        if self.enable_detection and self._is_within_schedule() and label in [c.lower() for c in self.record_classes] and confidence >= self.confidence_threshold:
                            detected = True
                            if confidence > best_score:
                                best_score = confidence
                                best_label = label

                    # Detection and recording handling
                    if detected:
                        if not self.detection_active:
                            self.no_detection_frames = 0
                            self.post_countdown_frames = 0
                            alert = {
                                "camera": self.camera["name"],
                                "label": best_label or "object",
                                "confidence": float(best_score),
                                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "frame": self.frame.copy(),
                                "filepath": "",
                                "thumb": "",
                            }
                            emit_alert = True
                            if self.enable_recording:
                                if self.record_lock.acquire(blocking=False):
                                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                    self.output_file = os.path.join(self.output_dir, f"nagranie_{self.camera['name']}_{timestamp}.mp4")
                                    h, w = self.frame.shape[:2]
                                    self.record_thread = RecordingThread(self.output_file, w, h, self.fps)
                                    self.record_thread.start()
                                    for bf in list(self.prerecord_buffer):
                                        self.record_thread.write(bf)
                                    self.recording = True
                                    self.no_detection_frames = 0
                                    self.post_countdown_frames = 0
                                    thumb_path = self.output_file + ".jpg"
                                    try:
                                        cv2.imwrite(thumb_path, self.frame)
                                    except Exception as ex:
                                        print("Nie zapisano miniatury:", ex)
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
                                        with open(self.output_file + ".json", "w") as f:
                                            json.dump(meta, f, indent=2)
                                    except Exception as ex:
                                        print("Nie zapisano metadanych:", ex)
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

            except Exception as e:
                # Zmapuj częstsze przyczyny dla czytelnego overlay
                em = str(e).lower()
                if "401" in em or "unauthorized" in em or "auth" in em:
                    msg = "Auth/401"
                elif "timed out" in em or "timeout" in em:
                    msg = "Timeout"
                elif "name or service not known" in em or "getaddrinfo" in em or "dns" in em:
                    msg = "DNS"
                elif "connection refused" in em:
                    msg = "Connection refused"
                elif "no route to host" in em:
                    msg = "No route to host"
                else:
                    msg = str(e)
                self.error_signal.emit(msg, self.index)

            # ensure any ongoing recording is properly finalized before reconnecting
            if self.recording:
                self._stop_recording()

            if self.stop_signal:
                break
            # krótka pauza zanim spróbujemy ponownie (autoreconnect)
            QThread.msleep(300)

        # sprzątanie
        if self.recording:
            self._stop_recording()

    def stop(self):
        """Request the worker thread to stop and wait for completion.

        W poprzedniej wersji metoda `stop` jedynie ustawiała flagę
        `stop_signal` i czekała maksymalnie 2 sekundy na zakończenie wątku.
        Przy problemach z połączeniem wątek mógł jednak nie
        zakończyć się w tym czasie, co prowadziło do zawieszania się
        aplikacji podczas usuwania kamery. Teraz, jeśli wątek nie
        zatrzyma się w ciągu 2 sekund, zostaje brutalnie zakończony
        metodą `terminate`.
        """
        self.stop_signal = True
        # zwolnij zasoby nagrywania jak najszybciej
        self._stop_recording()
        if self.isRunning() and not self.wait(2000):
            # wątek nadal żyje – wymuś zakończenie, aby GUI nie
            # zawiesiło się podczas operacji usuwania
            self.terminate()
            self.wait()


# --- Miniaturka na liście kamer ---
class CameraListWidgetItem(QWidget):
    def __init__(self, camera_name):
        super().__init__()
        self.setStyleSheet("""
            QWidget#CameraCard {
                background: transparent;
                border: none;
            }
            QLabel#CamName {
                font-weight: 600;
                color: #e6e6e6;
                padding: 6px 8px 2px 8px;
            }
        """)
        self.setObjectName("CameraCard")

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self.text_label = QLabel(camera_name)
        self.text_label.setObjectName("CamName")
        self.text_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.text_label)

        self.icon_label = QLabel()
        self.thumb_w, self.thumb_h = 192, 108
        self.icon_label.setFixedSize(self.thumb_w, self.thumb_h)
        self.icon_label.setFrameShape(QFrame.NoFrame)
        self.icon_label.setStyleSheet("background: transparent; border: none;")
        root.addWidget(self.icon_label, alignment=Qt.AlignCenter)

    def set_thumbnail(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(rgb, (self.thumb_w, self.thumb_h), interpolation=cv2.INTER_AREA)
        qimg = QImage(image.data, self.thumb_w, self.thumb_h, image.strides[0], QImage.Format_RGB888)
        self.icon_label.setPixmap(QPixmap.fromImage(qimg))

class CameraListWidget(QListWidget):
    request_context = pyqtSignal(int, QPoint)

    def __init__(self, cameras):
        super().__init__()
        self.setFixedWidth(300)
        self.setSpacing(12)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("QListWidget{ background: transparent; border: none; padding:8px; }")
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.widgets = []
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
        for cam in cameras:
            widget = CameraListWidgetItem(cam["name"])
            item = QListWidgetItem(self)
            item.setSizeHint(widget.sizeHint())
            self.addItem(item)
            self.setItemWidget(item, widget)
            self.widgets.append(widget)

    def update_thumbnail(self, index, frame):
        if 0 <= index < len(self.widgets):
            self.widgets[index].set_thumbnail(frame)

    def rebuild(self, cameras):
        self.clear()
        self.widgets = []
        for cam in cameras:
            widget = CameraListWidgetItem(cam["name"])
            item = QListWidgetItem(self)
            item.setSizeHint(widget.sizeHint())
            self.addItem(item)
            self.setItemWidget(item, widget)
            self.widgets.append(widget)

    def _on_context_menu(self, pos: QPoint):
        row = self.currentRow()
        item = self.itemAt(pos)
        if item is not None:
            row = self.row(item)
        if row < 0:
            return
        self.request_context.emit(row, self.mapToGlobal(pos))

# --- Podgląd wszystkich kamer w siatce ---
class CameraGridItem(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, index, name):
        super().__init__()
        self.index = index
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setStyleSheet("background:#000;")
        self.frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.frame_label, 1)

        self.name_label = QLabel(name)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("color:white; background:rgba(0,0,0,0.5);")
        layout.addWidget(self.name_label)

        self._pixmap = None

    def set_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_pixmap()

    def _update_pixmap(self):
        if self._pixmap is not None:
            self.frame_label.setPixmap(
                self._pixmap.scaled(
                    self.frame_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def mousePressEvent(self, event):
        self.clicked.emit(self.index)


class CameraGridWidget(QWidget):
    camera_clicked = pyqtSignal(int)

    def __init__(self, cameras):
        super().__init__()
        self.cameras = list(cameras)
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        self.items = []
        self._build()

    def _build(self):
        for it in self.items:
            it.setParent(None)
        self.items = []
        for idx, cam in enumerate(self.cameras):
            item = CameraGridItem(idx, cam["name"])
            item.clicked.connect(self.camera_clicked.emit)
            self.items.append(item)
        self._reflow()

    def _reflow(self):
        while self.layout.count():
            self.layout.takeAt(0)
        n = len(self.items)
        if n == 0:
            return
        cols = int(np.ceil(np.sqrt(n)))
        for idx, item in enumerate(self.items):
            r = idx // cols
            c = idx % cols
            self.layout.addWidget(item, r, c)
            self.layout.setRowStretch(r, 1)
            self.layout.setColumnStretch(c, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow()

    def rebuild(self, cameras):
        self.cameras = list(cameras)
        self._build()

    def update_frame(self, index, frame):
        if 0 <= index < len(self.items):
            self.items[index].set_frame(frame)


# --- Alert z miniaturką (karta) ---
class AlertItemWidget(QWidget):
    def __init__(self, alert: dict, thumb_size=(256, 144)):
        super().__init__()
        self.alert = alert
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setAlignment(Qt.AlignTop)

        self.thumb = QLabel()
        self.thumb.setFixedSize(*thumb_size)
        self.thumb.setStyleSheet("border:1px solid #555; background:#111;")
        v.addWidget(self.thumb, alignment=Qt.AlignCenter)

        cam = alert.get('camera', '?')
        lbl = alert.get('label', 'object')
        conf = float(alert.get('confidence', 0.0)) * 100.0
        ts  = alert.get('time', '--:--:--')
        self.meta = QLabel(f"{cam}\n{ts} — {lbl} ({conf:.1f}%)")
        self.meta.setStyleSheet("padding-top:6px; color:#ddd;")
        self.meta.setAlignment(Qt.AlignCenter)
        v.addWidget(self.meta, alignment=Qt.AlignCenter)

        frame = alert.get('frame')
        if frame is not None:
            self.set_frame(frame)
        else:
            jpg = alert.get("thumb")
            if not jpg:
                fp = alert.get("filepath") or alert.get("file") or ""
                jpg = fp + ".jpg" if fp else ""
            if jpg and os.path.exists(jpg):
                img = cv2.imread(jpg)
                if img is not None:
                    self.set_frame(img)

    def set_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.thumb.setPixmap(pix)

class AlertListWidget(QWidget):
    open_video = pyqtSignal(str)

    def __init__(self, alert_memory: AlertMemory):
        super().__init__()
        self.mem = alert_memory
        self.setFixedWidth(300)
        self.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setFixedWidth(300)
        self.list.setFrameShape(QFrame.NoFrame)
        self.list.setStyleSheet("QListWidget{background: transparent; border: none;}")
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.list)

        self.list.itemDoubleClicked.connect(self._open_selected)

        self.load_from_history(self.mem.items)

    def add_alert(self, alert: dict):
        widget = AlertItemWidget(alert)
        item = QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        self.list.insertItem(0, item)
        self.list.setItemWidget(item, widget)
        self.list.scrollToItem(item, hint=QListWidget.PositionAtTop)

    def load_from_history(self, items: list):
        self.list.clear()
        def parse_dt(a):
            t = a.get("time","")
            try:
                return datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
            except Exception:
                fp = a.get("filepath") or a.get("file") or ""
                try:
                    return datetime.datetime.fromtimestamp(os.path.getmtime(fp)) if fp and os.path.exists(fp) else datetime.datetime.min
                except Exception:
                    return datetime.datetime.min
        sorted_items = sorted(items[-300:], key=parse_dt, reverse=True)
        for a in sorted_items:
            widget = AlertItemWidget(a)
            item = QListWidgetItem()
            item.setSizeHint(widget.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, widget)
        if self.list.count():
            self.list.scrollToTop()

    def _open_selected(self, item):
        widget = self.list.itemWidget(item)
        if isinstance(widget, AlertItemWidget):
            fp = widget.alert.get("filepath") or widget.alert.get("file")
            if fp and os.path.exists(fp):
                self.open_video.emit(fp)

    def reload(self):
        self.mem.load()
        self.load_from_history(self.mem.items)

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Eksport alertów do CSV", "alerts.csv", "CSV (*.csv)")
        if not path:
            return
        ok, err = self.mem.export_csv(path)
        if ok:
            QMessageBox.information(self, "Eksport", f"Zapisano: {path}")
        else:
            QMessageBox.warning(self, "Eksport", f"Nie udało się zapisać CSV:\n{err}")

    def clear(self):
        if QMessageBox.question(self, "Wyczyść pamięć alertów",
                                "Czy na pewno wyczyścić całą historię alertów?",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            self.mem.clear()
            self.list.clear()


class AlertDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.setWindowTitle("Alerty")
        self.setPalette(QApplication.palette())

        v = QVBoxLayout(self)

        self.chk_visible = QCheckBox("Pokaż listę alertów")
        self.chk_visible.setChecked(self.mw.alert_list.isVisible())
        self.chk_visible.toggled.connect(self.mw.alert_list.setVisible)
        v.addWidget(self.chk_visible)

        btn_layout = QHBoxLayout()

        btn_reload = QPushButton("Wczytaj ponownie")
        btn_reload.clicked.connect(self.mw.alert_list.reload)
        btn_layout.addWidget(btn_reload)

        btn_export = QPushButton("Eksport do CSV")
        btn_export.clicked.connect(self.mw.alert_list.export_csv)
        btn_layout.addWidget(btn_export)

        btn_clear = QPushButton("Wyczyść pamięć")
        btn_clear.clicked.connect(self.mw.alert_list.clear)
        btn_layout.addWidget(btn_clear)

        v.addLayout(btn_layout)


class LogEntryWidget(QFrame):
    def __init__(
        self,
        entry_id: str,
        group: str,
        ts: str,
        camera: str = "",
        action: str = "",
        detected: str = "",
        recording: str = "",
    ):
        super().__init__()
        self.group = group
        self.entry_id = entry_id
        colors = {
            "application": "#4aa3ff",
            "detection": "#4caf50",
            "error": "#ff4444",
        }
        self.setStyleSheet(
            "QFrame{border:none; background:rgba(0,0,0,0.4);}" 
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setAlignment(Qt.AlignLeft)

        color = colors.get(group, "#fff")

        dt = None
        try:
            dt = datetime.datetime.strptime(ts, "%A %H:%M:%S %Y-%m-%d")
        except Exception:
            pass

        # Nagłówek z tytułem i datą
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setAlignment(Qt.AlignLeft)
        header_widget.setStyleSheet(f"border-bottom:1px solid {color};")

        self.group_label = QLabel(group.upper())
        self.group_label.setAlignment(Qt.AlignLeft)
        self.group_label.setStyleSheet(
            f"color:{color}; font-size:15px; font-weight:600;"
        )
        self.group_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        header_layout.addWidget(self.group_label)

        date_str = dt.strftime("%Y-%m-%d") if dt else ""
        self.date_label = QLabel(date_str)
        self.date_label.setAlignment(Qt.AlignLeft)
        self.date_label.setStyleSheet(f"color:{color}; font-size:15px;")
        header_layout.addWidget(self.date_label)
        header_layout.addStretch()

        layout.addWidget(header_widget)

        def add_line(text: str, color: str):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignLeft)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color:{color}; font-size:15px;")
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            layout.addWidget(lbl)

        # Wiersz z godziną i dniem tygodnia
        time_weekday_layout = QHBoxLayout()
        time_label = QLabel(dt.strftime("%H:%M:%S") if dt else ts)
        time_label.setAlignment(Qt.AlignLeft)
        time_label.setStyleSheet("color:#ff8800; font-size:15px;")
        weekday_str = dt.strftime("%A").capitalize() if dt else ""
        weekday_label = QLabel(weekday_str)
        weekday_label.setAlignment(Qt.AlignRight)
        weekday_label.setStyleSheet("color:#ff8800; font-size:15px;")
        time_weekday_layout.addWidget(time_label)
        time_weekday_layout.addStretch()
        time_weekday_layout.addWidget(weekday_label)
        layout.addLayout(time_weekday_layout)

        if camera:
            add_line(camera, "#4aa3ff")
        if detected:
            add_line(detected.upper(), "#4caf50")
        if group != "detection" and action:
            add_line(action, "#ff8800")

        if group == "detection":
            action_row = QHBoxLayout()
            self.rec_dot = QLabel()
            self.rec_dot.setFixedSize(10, 10)
            self.rec_text = QLabel()
            action_row.addWidget(self.rec_dot)
            action_row.addWidget(self.rec_text)
            action_row.setAlignment(Qt.AlignLeft)
            layout.addLayout(action_row)
            self.rec_dot.hide()
            self.rec_text.hide()

            self._blink_timer = QTimer(self)
            self._blink_timer.timeout.connect(
                lambda: self.rec_dot.setVisible(not self.rec_dot.isVisible())
            )

            if recording == "started":
                self.start_recording()
            elif recording == "finished":
                self.finish_recording()
            elif recording == "det_started":
                self.start_detection()
            elif recording == "det_finished":
                self.finish_detection()
        else:
            self.rec_dot = QLabel()
            self.rec_text = QLabel()
            self._blink_timer = QTimer(self)

    def start_recording(self):
        self.rec_text.setText("Recording started")
        self.rec_text.setStyleSheet("color:red; font-size:15px;")
        self.rec_dot.setStyleSheet("background:red; border-radius:5px;")
        self.rec_dot.show()
        self.rec_text.show()
        self.rec_dot.setVisible(True)
        self._blink_timer.start(500)

    def finish_recording(self):
        self.rec_text.setText("Recording finished")
        self.rec_text.setStyleSheet("color:red; font-size:15px;")
        self.rec_dot.setStyleSheet("background:red; border-radius:5px;")
        self.rec_dot.show()
        self.rec_text.show()
        self._blink_timer.stop()
        self.rec_dot.setVisible(True)

    def start_detection(self):
        self.rec_text.setText("Detection started")
        self.rec_text.setStyleSheet("color:green; font-size:15px;")
        self.rec_dot.setStyleSheet("background:green; border-radius:5px;")
        self.rec_dot.show()
        self.rec_text.show()
        self.rec_dot.setVisible(True)
        self._blink_timer.start(500)

    def finish_detection(self):
        self.rec_text.setText("Detection finished")
        self.rec_text.setStyleSheet("color:green; font-size:15px;")
        self.rec_dot.setStyleSheet("background:green; border-radius:5px;")
        self.rec_dot.show()
        self.rec_text.show()
        self._blink_timer.stop()
        self.rec_dot.setVisible(True)


class LogWindow(QListWidget):
    """Widget prezentujący logi oraz zapisujący je do pliku."""

    LOG_HISTORY_PATH = LOG_HISTORY_PATH

    def __init__(self, log_path: str = LOG_HISTORY_PATH, retention_hours: int = LOG_RETENTION_HOURS):
        super().__init__()
        self.setFixedWidth(300)
        self.setFrameShape(QFrame.NoFrame)
        self.setSpacing(8)
        self.setStyleSheet("QListWidget{background:transparent; border:none;}")
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.log_path = log_path
        self.retention_hours = retention_hours
        self.history = []

    def _add_widget_entry(self, entry: dict):
        widget = LogEntryWidget(
            entry.get("id", ""),
            entry.get("group", ""),
            entry.get("timestamp", ""),
            entry.get("camera", ""),
            entry.get("action", ""),
            entry.get("detected", ""),
            entry.get("recording", ""),
        )
        item = QListWidgetItem(self)
        self.addItem(item)
        self.setItemWidget(item, widget)
        item.setSizeHint(widget.sizeHint())

    def _refresh_widget(self):
        self.clear()
        for entry in self.history[-200:]:
            self._add_widget_entry(entry)
        if self.count():
            self.scrollToItem(self.item(self.count() - 1))

    def load_history(self):
        self.history = []
        try:
            if os.path.exists(self.log_path):
                with open(self.log_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.history = data
        except Exception:
            self.history = []

        cutoff = datetime.datetime.now() - datetime.timedelta(hours=self.retention_hours)
        filtered = []
        allowed = {"detection", "error", "application"}
        for entry in self.history:
            if entry.get("group") not in allowed:
                continue
            ts_str = entry.get("timestamp")
            try:
                ts_dt = datetime.datetime.strptime(ts_str, "%A %H:%M:%S %Y-%m-%d")
            except Exception:
                continue
            if ts_dt >= cutoff:
                filtered.append(entry)
        self.history = filtered
        self._refresh_widget()

    def add_entry(
        self,
        group: str,
        camera: str = "",
        action: str = "",
        detected: str = "",
    ) -> str:
        if group == "detection object":
            group = "detection"
        allowed = {"detection", "error", "application"}
        if group not in allowed:
            return ""

        ts = datetime.datetime.now().strftime("%A %H:%M:%S %Y-%m-%d")
        entry_id = uuid.uuid4().hex
        entry = {
            "id": entry_id,
            "group": group,
            "camera": camera,
            "action": action,
            "detected": detected,
            "timestamp": ts,
            "recording": "",
        }
        self.history.append(entry)

        cutoff = datetime.datetime.now() - datetime.timedelta(hours=self.retention_hours)
        filtered = []
        for e in self.history:
            try:
                ts_dt = datetime.datetime.strptime(e.get("timestamp", ""), "%A %H:%M:%S %Y-%m-%d")
            except Exception:
                continue
            if ts_dt >= cutoff:
                filtered.append(e)
        self.history = filtered

        self._refresh_widget()

        try:
            with open(self.log_path, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print("Nie udało się zapisać historii logów:", e)
        return entry_id

    def update_recording_by_id(self, entry_id: str, status: str):
        for entry in self.history:
            if entry.get("id") == entry_id:
                entry["recording"] = status
                break
        for idx in range(self.count() - 1, -1, -1):
            item = self.item(idx)
            widget = self.itemWidget(item)
            if isinstance(widget, LogEntryWidget) and widget.entry_id == entry_id:
                if status == "started":
                    widget.start_recording()
                elif status == "finished":
                    widget.finish_recording()
                elif status == "det_started":
                    widget.start_detection()
                    QTimer.singleShot(2000, lambda: self.update_recording_by_id(entry_id, "det_finished"))
                elif status == "det_finished":
                    widget.finish_detection()
                break
        try:
            with open(self.log_path, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception:
            pass


# --- ODTWARZACZ WIDEO ---
class VideoPlayerDialog(QDialog):
    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.resize(900, 600)

        # lista plików w katalogu – umożliwia przełączanie
        folder = os.path.dirname(filepath) or "."
        self.file_list = sorted(glob(os.path.join(folder, "*.mp4")))
        self.file_index = self.file_list.index(filepath) if filepath in self.file_list else 0

        v = QVBoxLayout(self)
        self.video_label = QLabel("Wideo")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#000; color:#fff;")
        v.addWidget(self.video_label, stretch=1)

        ctrl = QHBoxLayout()
        self.btn_play = QPushButton("▶")
        self.btn_pause = QPushButton("⏸")
        self.btn_stop = QPushButton("◼")
        self.btn_back = QPushButton("<<")
        self.btn_fwd = QPushButton(">>")
        self.btn_prev = QPushButton("Nagranie ←")
        self.btn_next = QPushButton("Nagranie →")
        self.btn_snap = QPushButton("📷")
        self.slider = QSlider(Qt.Horizontal)
        self.btn_full = QPushButton("Pełny ekran")
        ctrl.addWidget(self.btn_prev)
        ctrl.addWidget(self.btn_next)
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.btn_pause)
        ctrl.addWidget(self.btn_stop)
        ctrl.addWidget(self.btn_back)
        ctrl.addWidget(self.btn_fwd)
        ctrl.addWidget(self.btn_snap)
        ctrl.addWidget(self.slider, stretch=1)
        ctrl.addWidget(self.btn_full)
        v.addLayout(ctrl)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)

        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_back.clicked.connect(self.step_back)
        self.btn_fwd.clicked.connect(self.step_forward)
        self.btn_prev.clicked.connect(self.prev_video)
        self.btn_next.clicked.connect(self.next_video)
        self.btn_snap.clicked.connect(self.save_screenshot)
        self.btn_full.clicked.connect(self.toggle_fullscreen)
        self.slider.sliderPressed.connect(self.pause)
        self.slider.sliderReleased.connect(self.seek_to_slider)

        self.video_label.mouseDoubleClickEvent = lambda e: self.toggle_fullscreen()

        self.cap = None
        self.current_index = 0
        self.current_frame = None
        self._normal_geometry = self.geometry()
        self.load_video(self.file_list[self.file_index])

    def _read_frame(self, idx=None):
        if idx is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(idx, self.frame_count - 1)))
        ret, frame = self.cap.read()
        return ret, frame

    def _show_frame(self, frame):
        if frame is None:
            return
        self.current_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def _show_frame_at(self, idx):
        ret, frame = self._read_frame(idx)
        if ret:
            self.current_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_index)
            self.slider.blockSignals(False)
            self._show_frame(frame)

    def _next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.pause()
            return
        self.current_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_index)
        self.slider.blockSignals(False)
        self._show_frame(frame)

    def play(self):
        if not self.timer.isActive():
            interval_ms = int(1000 / max(self.fps, 0.01))
            self.timer.start(interval_ms)

    def pause(self):
        self.timer.stop()

    def stop(self):
        self.pause()
        self._show_frame_at(0)

    def step_forward(self):
        self.pause()
        self._show_frame_at(self.current_index + 1)

    def step_back(self):
        self.pause()
        self._show_frame_at(self.current_index - 1)

    def seek_to_slider(self):
        self.pause()
        self._show_frame_at(self.slider.value())

    # --- Pliki ---
    def load_video(self, filepath):
        if self.cap:
            self.cap.release()
        self.filepath = filepath
        self.setWindowTitle(os.path.basename(filepath))
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Błąd", f"Nie można otworzyć pliku:\n{filepath}")
            return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.slider.setRange(0, max(self.frame_count - 1, 0))
        self.current_index = 0
        self._show_frame_at(0)

    def next_video(self):
        if self.file_index < len(self.file_list) - 1:
            self.file_index += 1
            self.load_video(self.file_list[self.file_index])

    def prev_video(self):
        if self.file_index > 0:
            self.file_index -= 1
            self.load_video(self.file_list[self.file_index])

    def save_screenshot(self):
        if self.current_frame is None:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(os.path.dirname(self.filepath), f"frame_{ts}.jpg")
        try:
            cv2.imwrite(out, self.current_frame)
            QMessageBox.information(self, "Zapisano", f"Kadr zapisany jako: {os.path.basename(out)}")
            parent = self.parent()
            if parent is not None and hasattr(parent, "log_window"):
                parent.log_window.add_entry("application", f"wyeksportowano kadr {os.path.basename(out)}")
        except Exception as e:
            QMessageBox.warning(self, "Błąd", str(e))
            parent = self.parent()
            if parent is not None and hasattr(parent, "log_window"):
                parent.log_window.add_entry("error", f"kadr: {e}")

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            if getattr(self, "_normal_geometry", None):
                self.setGeometry(self._normal_geometry)
        else:
            self._normal_geometry = self.geometry()
            self.showFullScreen()

    def closeEvent(self, e):
        self.pause()
        if self.cap:
            self.cap.release()
        super().closeEvent(e)


# --- Panel „Nagrania” (przeglądarka z filtrami + usuwanie) ---


class ThumbnailLoader(QObject, QRunnable):
    thumbnailReady = pyqtSignal(QImage)

    def __init__(self, filepath: str):
        QObject.__init__(self)
        QRunnable.__init__(self)
        self._filepath = filepath or ""

    def run(self):
        qimg = self._load_thumbnail()
        if qimg is None:
            qimg = QImage()
        self.thumbnailReady.emit(qimg)

    def _load_thumbnail(self):
        if not self._filepath:
            return None
        jpg = self._filepath + ".jpg"
        try:
            if os.path.exists(jpg):
                img = cv2.imread(jpg)
                if img is not None:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        except Exception:
            pass
        if os.path.exists(self._filepath):
            cap = None
            try:
                cap = cv2.VideoCapture(self._filepath)
                if cap.isOpened():
                    ok, frame = cap.read()
                else:
                    ok, frame = False, None
            except Exception:
                ok, frame = False, None
            finally:
                if cap is not None:
                    cap.release()
            if ok and frame is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
                except Exception:
                    return None
        return None


class RecordingsScanWorker(QObject, QRunnable):
    recordFound = pyqtSignal(dict)
    scanFinished = pyqtSignal()

    def __init__(self, base_dir: str, cameras: list):
        QObject.__init__(self)
        QRunnable.__init__(self)
        self._base_dir = base_dir
        self._cameras = cameras
        self._abort = False

    def stop(self):
        self._abort = True

    def run(self):
        try:
            self._scan()
        finally:
            self.scanFinished.emit()

    def _scan(self):
        if not os.path.isdir(self._base_dir):
            return
        for cam in self._cameras:
            if self._abort:
                break
            cam_dir = os.path.join(self._base_dir, cam["name"])
            if not os.path.isdir(cam_dir):
                continue
            files = sorted(glob(os.path.join(cam_dir, "nagranie_*.mp4")), reverse=True)
            for mp4 in files:
                if self._abort:
                    break
                meta = self._build_meta(cam, mp4)
                self.recordFound.emit(meta)

    def _build_meta(self, cam, mp4):
        meta_path = mp4 + ".json"
        meta = {
            "camera": cam["name"],
            "label": "unknown",
            "confidence": 0.0,
            "time": None,
            "file": mp4,
        }
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    m = json.load(f)
                    meta.update(m)
                meta["file"] = mp4
            except Exception:
                pass
        else:
            base = os.path.basename(mp4)
            m = re.search(r"_(\d{8})_(\d{6})\.mp4$", base)
            if m:
                ds, ts = m.group(1), m.group(2)
                try:
                    dt = datetime.datetime.strptime(ds + ts, "%Y%m%d%H%M%S")
                    meta["time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
        if not meta.get("time"):
            try:
                ts = datetime.datetime.fromtimestamp(os.path.getmtime(mp4))
                meta["time"] = ts.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                meta["time"] = ""
        return meta


class RecordingItemWidget(QWidget):
    selectionToggled = pyqtSignal(bool)

    def __init__(self, meta: dict, thread_pool: QThreadPool = None, thumb_size=(256, 144)):
        super().__init__()
        self.meta = meta
        self._thread_pool = thread_pool or QThreadPool.globalInstance()
        self._thumb_size = thumb_size
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setAlignment(Qt.AlignCenter)

        self.thumb = QLabel()
        self.thumb.setFixedSize(*thumb_size)
        self.thumb.setStyleSheet("border:1px solid #555; background:#111;")
        placeholder = QPixmap(self.thumb.size())
        placeholder.fill(Qt.black)
        self.thumb.setPixmap(placeholder)
        v.addWidget(self.thumb)

        cam = meta.get("camera", "?")
        lbl = meta.get("label", "object")
        conf = float(meta.get("confidence", 0.0)) * 100.0
        ts = meta.get("time", "--:--:--")
        name = os.path.basename(meta.get("file", ""))

        self.meta_label = QLabel(f"{cam} | {ts}\n{lbl} ({conf:.1f}%)\n{name}")
        self.meta_label.setAlignment(Qt.AlignCenter)
        self.meta_label.setStyleSheet("padding-top:6px; color:#000;")
        v.addWidget(self.meta_label)

        self.checkbox = QCheckBox("Zaznacz")
        self.checkbox.setTristate(False)
        v.addWidget(self.checkbox, alignment=Qt.AlignCenter)

        self.checkbox.toggled.connect(self._on_checkbox_toggled)

        self._loader = ThumbnailLoader(meta.get("file", ""))
        self._loader.thumbnailReady.connect(self._apply_thumbnail)
        self._thread_pool.start(self._loader)

    def _on_checkbox_toggled(self, checked: bool):
        self.selectionToggled.emit(checked)

    def set_checked(self, checked: bool):
        block = self.checkbox.blockSignals(True)
        self.checkbox.setChecked(checked)
        self.checkbox.blockSignals(block)

    def is_checked(self):
        return self.checkbox.isChecked()

    def _apply_thumbnail(self, qimg: QImage):
        if qimg is None or qimg.isNull():
            return
        pix = QPixmap.fromImage(qimg)
        self.thumb.setPixmap(pix.scaled(self.thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class RecordingsBrowserDialog(QDialog):
    open_video = pyqtSignal(str)

    def __init__(self, base_dir, cameras, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nagrania – przeglądarka")
        self.resize(1100, 700)
        self.base_dir = base_dir
        self.cameras = cameras

        self.scan_pool = QThreadPool()
        self.thumbnail_pool = QThreadPool()
        self._scan_worker = None
        self._visible_paths = set()

        layout = QVBoxLayout(self)

        # FILTRY
        filters = QHBoxLayout()
        self.camera_filter = QComboBox()
        self.camera_filter.addItem("Wszystkie kamery")
        for cam in cameras:
            self.camera_filter.addItem(cam["name"])

        self.class_filter = QComboBox()
        self.class_filter.addItem("Wszystkie klasy")
        for c in VISIBLE_CLASSES:
            self.class_filter.addItem(c)

        self.date_from = QDateEdit()
        self.date_to = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_to.setCalendarPopup(True)
        today = QDate.currentDate()
        self.date_to.setDate(today)
        self.date_from.setDate(today.addDays(-7))

        self.search_text = QLineEdit()
        self.search_text.setPlaceholderText("Szukaj po nazwie pliku...")

        self.refresh_btn = QPushButton("Odśwież")
        self.delete_btn = QPushButton("Usuń zaznaczone")
        self.select_all_btn = QPushButton("Zaznacz wszystko")
        self.clear_selection_btn = QPushButton("Odznacz wszystko")

        filters.addWidget(QLabel("Kamera:"))
        filters.addWidget(self.camera_filter)
        filters.addSpacing(10)
        filters.addWidget(QLabel("Klasa:"))
        filters.addWidget(self.class_filter)
        filters.addSpacing(10)
        filters.addWidget(QLabel("Od:"))
        filters.addWidget(self.date_from)
        filters.addWidget(QLabel("Do:"))
        filters.addWidget(self.date_to)
        filters.addStretch(1)
        filters.addWidget(self.search_text, 2)
        filters.addWidget(self.refresh_btn)
        filters.addWidget(self.delete_btn)
        filters.addWidget(self.select_all_btn)
        filters.addWidget(self.clear_selection_btn)

        layout.addLayout(filters)

        # LISTA NAGRAŃ
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setMovement(QListWidget.Static)
        self.list.setSpacing(12)
        self.list.setUniformItemSizes(False)
        self.list.setIconSize(QSize(256, 144))
        self.list.setSelectionMode(QListWidget.ExtendedSelection)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        layout.addWidget(self.list, 1)

        # sygnały
        self.refresh_btn.clicked.connect(self.refresh)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.select_all_btn.clicked.connect(self.list.selectAll)
        self.clear_selection_btn.clicked.connect(self._clear_selection)
        self.camera_filter.currentIndexChanged.connect(self.apply_filters)
        self.class_filter.currentIndexChanged.connect(self.apply_filters)
        self.date_from.dateChanged.connect(self.apply_filters)
        self.date_to.dateChanged.connect(self.apply_filters)
        self.search_text.textChanged.connect(self.apply_filters)
        self.list.itemDoubleClicked.connect(self.open_selected)
        self.list.customContextMenuRequested.connect(self._context_menu)
        self.list.itemSelectionChanged.connect(self._sync_selection_with_checkboxes)

        # stan
        self.all_items = []
        # Opóźnione wczytywanie listy nagrań, aby okno otwierało się szybciej
        QTimer.singleShot(0, self._initial_load)

    def _initial_load(self):
        """Uruchamia asynchroniczne skanowanie katalogu."""
        self.refresh()

    def refresh(self):
        self._start_scan_worker()

    def _start_scan_worker(self):
        if self._scan_worker is not None:
            self._scan_worker.stop()
            with suppress(TypeError):
                self._scan_worker.recordFound.disconnect(self._on_record_found)
            with suppress(TypeError):
                self._scan_worker.scanFinished.disconnect(self._on_scan_finished)
        self.all_items.clear()
        self._visible_paths.clear()
        self.list.clear()
        self.refresh_btn.setEnabled(False)
        worker = RecordingsScanWorker(self.base_dir, self.cameras)
        worker.recordFound.connect(self._on_record_found)
        worker.scanFinished.connect(self._on_scan_finished)
        self._scan_worker = worker
        self.scan_pool.start(worker)

    def _on_record_found(self, meta: dict):
        self.all_items.append(meta)
        if self._record_matches_filters(meta):
            self._add_list_item(meta)

    def _on_scan_finished(self):
        self.refresh_btn.setEnabled(True)
        self._scan_worker = None

    def _clear_selection(self):
        self.list.clearSelection()

    def _current_filters(self):
        cam_sel = self.camera_filter.currentText()
        if cam_sel == "Wszystkie kamery":
            cam_sel = None
        cls_sel = self.class_filter.currentText()
        if cls_sel == "Wszystkie klasy":
            cls_sel = None
        qfrom = self.date_from.date()
        qto = self.date_to.date()
        text = self.search_text.text().strip().lower()
        return cam_sel, cls_sel, qfrom, qto, text

    def meta_in_date_range(self, meta, qfrom: QDate, qto: QDate):
        try:
            dt = datetime.datetime.strptime(meta.get("time", ""), "%Y-%m-%d %H:%M:%S")
            d = QDate(dt.year, dt.month, dt.day)
            return (d >= qfrom) and (d <= qto)
        except Exception:
            return True

    def _record_matches_filters(self, meta, filters=None):
        if filters is None:
            filters = self._current_filters()
        cam_sel, cls_sel, qfrom, qto, text = filters
        if cam_sel and meta.get("camera") != cam_sel:
            return False
        if cls_sel and meta.get("label") != cls_sel:
            return False
        if not self.meta_in_date_range(meta, qfrom, qto):
            return False
        if text and text not in os.path.basename(meta.get("file", "")).lower():
            return False
        return True

    def apply_filters(self):
        selected_paths = self._collect_checked_filepaths()
        filters = self._current_filters()
        self.list.clear()
        self._visible_paths.clear()
        for meta in list(self.all_items):
            if not self._record_matches_filters(meta, filters):
                continue
            preselect = meta.get("file") in selected_paths
            self._add_list_item(meta, preselect)

    def _add_list_item(self, meta, preselect=False):
        path = meta.get("file")
        if not path or path in self._visible_paths:
            return
        widget = RecordingItemWidget(meta, thread_pool=self.thumbnail_pool)
        item = QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        item.setData(Qt.UserRole, path)
        self.list.addItem(item)
        self.list.setItemWidget(item, widget)
        self._visible_paths.add(path)
        widget.selectionToggled.connect(lambda checked, it=item: self._on_widget_toggled(it, checked))
        if preselect:
            item.setSelected(True)
        self._update_widget_selection_from_item(item, widget)

    def _update_widget_selection_from_item(self, item: QListWidgetItem, widget: RecordingItemWidget = None):
        if widget is None:
            widget = self.list.itemWidget(item)
        if widget is not None:
            widget.set_checked(item.isSelected())

    def _on_widget_toggled(self, item: QListWidgetItem, checked: bool):
        if item is None:
            return
        item.setSelected(checked)

    def _sync_selection_with_checkboxes(self):
        for idx in range(self.list.count()):
            item = self.list.item(idx)
            self._update_widget_selection_from_item(item)

    def open_selected(self, item: QListWidgetItem):
        fp = item.data(Qt.UserRole)
        if fp and os.path.exists(fp):
            self.open_video.emit(fp)

    def _selected_filepaths(self):
        return [i.data(Qt.UserRole) for i in self.list.selectedItems() if i.data(Qt.UserRole)]

    def _collect_checked_filepaths(self):
        paths = set(self._selected_filepaths())
        for idx in range(self.list.count()):
            item = self.list.item(idx)
            path = item.data(Qt.UserRole)
            if not path:
                continue
            widget = self.list.itemWidget(item)
            if widget and widget.is_checked():
                paths.add(path)
        return list(paths)

    def delete_selected(self):
        paths = self._collect_checked_filepaths()
        if not paths:
            QMessageBox.information(self, "Usuń nagrania", "Nie wybrano żadnych nagrań.")
            return
        if len(paths) == 1:
            msg = f"Czy na pewno usunąć nagranie?\n\n{os.path.basename(paths[0])}"
        else:
            msg = f"Czy na pewno usunąć {len(paths)} nagrań?"
        if QMessageBox.question(self, "Potwierdzenie", msg,
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        errors = []
        deleted = 0
        for fp in paths:
            for p in [fp, fp + ".json", fp + ".jpg"]:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception as e:
                    errors.append(f"{os.path.basename(p)}: {e}")
            deleted += 1
        remaining = set(paths)
        self.all_items = [m for m in self.all_items if m.get("file") not in remaining]
        self.apply_filters()
        if errors:
            QMessageBox.warning(self, "Usunięto z błędami",
                                f"Usunięto: {deleted}, ale wystąpiły błędy:\n- " + "\n- ".join(errors))
        else:
            QMessageBox.information(self, "Usunięto", f"Usunięto {deleted} nagrań.")

    def _context_menu(self, pos: QPoint):
        item = self.list.itemAt(pos)
        menu = QMenu(self)
        act_open = menu.addAction("Otwórz")
        act_del = menu.addAction("Usuń")
        action = menu.exec_(self.list.mapToGlobal(pos))
        if action == act_open and item is not None:
            self.open_selected(item)
        elif action == act_del:
            self.delete_selected()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Delete:
            self.delete_selected()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, event):
        if self._scan_worker is not None:
            self._scan_worker.stop()
            with suppress(TypeError):
                self._scan_worker.recordFound.disconnect(self._on_record_found)
            with suppress(TypeError):
                self._scan_worker.scanFinished.disconnect(self._on_scan_finished)
            self.scan_pool.waitForDone()
            self._scan_worker = None
        self.thumbnail_pool.waitForDone()
        super().closeEvent(event)


# --- Kreator dodawania/edycji kamery (RTSP krok-po-kroku) ---
class AddCameraWizard(QDialog):
    def __init__(self, parent=None, existing=None):
        super().__init__(parent)
        self.setWindowTitle("Kamera – kreator RTSP")
        self.resize(520, 360)

        self._editing = existing is not None

        self.stack = QStackedWidget()
        self.btn_prev = QPushButton("Wstecz")
        self.btn_next = QPushButton("Zakończ" if self._editing else "Dalej")
        self.btn_cancel = QPushButton("Anuluj")

        # --- Krok 1: nazwa, IP, port
        p1 = QWidget()
        f1 = QFormLayout(p1)
        self.name_edit = QLineEdit()
        self.ip_edit = QLineEdit()
        self.ip_edit.setPlaceholderText("np. 192.168.1.10")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(554)
        f1.addRow("Nazwa kamery*", self.name_edit)
        f1.addRow("Adres IP*", self.ip_edit)
        f1.addRow("Port", self.port_spin)

        # --- Krok 2: uwierzytelnianie
        p2 = QWidget()
        f2 = QFormLayout(p2)
        self.user_edit = QLineEdit()
        self.pass_edit = QLineEdit()
        self.pass_edit.setEchoMode(QLineEdit.Password)
        self.show_pass = QCheckBox("Pokaż hasło")
        self.show_pass.toggled.connect(lambda v: self.pass_edit.setEchoMode(QLineEdit.Normal if v else QLineEdit.Password))
        f2.addRow("Użytkownik", self.user_edit)
        f2.addRow("Hasło", self.pass_edit)
        f2.addRow("", self.show_pass)

        # --- Krok 3: ścieżka strumienia
        p3 = QWidget()
        f3 = QFormLayout(p3)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("np. /Streaming/Channels/101 lub /h264")
        f3.addRow("Ścieżka (opcjonalnie)", self.path_edit)

        # --- Krok 4: podsumowanie + test
        p4 = QWidget()
        v4 = QVBoxLayout(p4)
        self.url_preview = QLabel("rtsp://...")
        self.url_preview.setStyleSheet("background:#111; color:#0f0; padding:8px;")
        self.test_btn = QPushButton("Test połączenia (szybki)")
        self.test_status = QLabel("")
        v4.addWidget(QLabel("Podgląd adresu RTSP:"))
        v4.addWidget(self.url_preview)
        v4.addWidget(self.test_btn)
        v4.addWidget(self.test_status)
        v4.addStretch(1)

        self.stack.addWidget(p1)
        self.stack.addWidget(p2)
        self.stack.addWidget(p3)
        self.stack.addWidget(p4)

        # układ
        main = QVBoxLayout(self)
        main.addWidget(self.stack)
        nav = QHBoxLayout()
        nav.addWidget(self.btn_cancel)
        nav.addStretch(1)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        main.addLayout(nav)

        self.btn_prev.setEnabled(False)
        if self._editing:
            self.btn_next.setText("Zakończ")

        # sygnały
        self.btn_prev.clicked.connect(self.prev_step)
        self.btn_next.clicked.connect(self.next_step)
        self.btn_cancel.clicked.connect(self.reject)
        for w in [self.name_edit, self.ip_edit, self.port_spin, self.user_edit, self.pass_edit, self.path_edit]:
            if isinstance(w, QLineEdit):
                w.textChanged.connect(self.update_preview)
            else:
                w.valueChanged.connect(self.update_preview)
        self.test_btn.clicked.connect(self.quick_test)

        # wypełnij istniejące
        if existing:
            self.name_edit.setText(existing.get("name", ""))
            url = existing.get("rtsp", "")
            try:
                rest = url.replace("rtsp://", "")
                auth_part, host_path = (rest.split("@", 1) + [""])[:2] if "@" in rest else ("", rest)
                user, pwd = ("", "")
                if auth_part:
                    if ":" in auth_part:
                        user, pwd = auth_part.split(":", 1)
                    else:
                        user = auth_part
                host, path = (host_path.split("/", 1) + [""])[:2]
                ip, port = (host.split(":", 1) + ["554"])[:2]
                self.ip_edit.setText(ip)
                self.port_spin.setValue(int(port) if port.isdigit() else 554)
                self.user_edit.setText(user)
                self.pass_edit.setText(pwd)
                self.path_edit.setText("/" + path if path else "")
            except Exception:
                self.ip_edit.setText("")
                self.path_edit.setText(url)
        self.update_preview()

    def build_rtsp(self):
        name = self.name_edit.text().strip()
        ip = self.ip_edit.text().strip()
        port = int(self.port_spin.value())
        user = self.user_edit.text().strip()
        pwd = self.pass_edit.text()
        path = self.path_edit.text().strip()
        auth = ""
        if user and pwd:
            auth = f"{user}:{pwd}@"
        elif user and not pwd:
            auth = f"{user}@"
        p = f":{port}" if port else ""
        if path and not path.startswith("/"):
            path = "/" + path
        url = f"rtsp://{auth}{ip}{p}{path}"
        return name, url

    def update_preview(self):
        _, url = self.build_rtsp()
        self.url_preview.setText(url)

    def validate_step(self, idx):
        if idx == 0:
            if not self.name_edit.text().strip():
                QMessageBox.warning(self, "Brak nazwy", "Podaj nazwę kamery.")
                return False
            ip = self.ip_edit.text().strip()
            if not ip:
                QMessageBox.warning(self, "Brak adresu IP", "Podaj adres IP kamery.")
                return False
        return True

    def next_step(self):
        i = self.stack.currentIndex()
        if not self._editing:
            if not self.validate_step(i):
                return
            if i < self.stack.count() - 1:
                self.stack.setCurrentIndex(i + 1)
                self.btn_prev.setEnabled(True)
                if i + 1 == self.stack.count() - 1:
                    self.btn_next.setText("Zakończ")
                else:
                    self.btn_next.setText("Dalej")
                return
        name, url = self.build_rtsp()
        self.result_data = {"name": name, "rtsp": url, "type": "rtsp"}
        self.accept()

    def prev_step(self):
        i = self.stack.currentIndex()
        if i > 0:
            self.stack.setCurrentIndex(i - 1)
            self.btn_next.setText("Dalej")
            self.btn_prev.setEnabled(i - 1 > 0)

    def quick_test(self):
        self.test_status.setText("Testuję...")
        self.test_status.setStyleSheet("color:#ccc;")
        _, url = self.build_rtsp()
        cap = cv2.VideoCapture(url)
        ok, _ = cap.read()
        cap.release()
        if ok:
            self.test_status.setText("✅ Połączenie OK (pierwsza klatka odczytana).")
            self.test_status.setStyleSheet("color:#0f0;")
        else:
            self.test_status.setText("⚠️ Nie udało się odczytać klatki. Adres/poświadczenia/ścieżka?")
            self.test_status.setStyleSheet("color:#f80;")


# --- Dodawanie kamery USB ---
class AddUsbCameraDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dodaj kamerę USB")
        self.resize(400, 200)

        form = QFormLayout(self)
        self.name_edit = QLineEdit()
        self.device_combo = QComboBox()
        for idx, name in list_usb_cameras():
            self.device_combo.addItem(f"{name} ({idx})", idx)
        if self.device_combo.count() == 0:
            self.device_combo.addItem("Brak kamer", -1)

        self.test_btn = QPushButton("Testuj")
        self.test_status = QLabel("")
        test_layout = QHBoxLayout()
        test_layout.addWidget(self.test_btn)
        test_layout.addWidget(self.test_status)

        btns = QHBoxLayout()
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Anuluj")
        btns.addStretch(1)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)

        form.addRow("Nazwa", self.name_edit)
        form.addRow("Urządzenie", self.device_combo)
        form.addRow(test_layout)
        form.addRow(btns)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.test_btn.clicked.connect(self._test_device)

        self.result_data = None

    def _test_device(self):
        idx = self.device_combo.currentData()
        if idx is None or idx < 0:
            return
        self.test_status.setText("Testuję...")
        self.test_status.setStyleSheet("color:#ccc;")
        cap = cv2.VideoCapture(int(idx))
        ok, _ = cap.read()
        cap.release()
        if ok:
            self.test_status.setText("✅ OK")
            self.test_status.setStyleSheet("color:#0f0;")
        else:
            self.test_status.setText("⚠️ Błąd")
            self.test_status.setStyleSheet("color:#f80;")

    def accept(self):
        name = self.name_edit.text().strip()
        idx = self.device_combo.currentData()
        if not name or idx is None or idx < 0:
            QMessageBox.warning(self, "Błąd", "Podaj nazwę i wybierz urządzenie.")
            return
        self.result_data = {"name": name, "rtsp": int(idx), "type": "usb"}
        super().accept()


# --- Ustawienia pojedynczej kamery ---
class SingleCameraDialog(QDialog):
    def __init__(self, parent=None, camera=None):
        super().__init__(parent)
        self.setWindowTitle("Ustawienia kamery")
        self.resize(480, 520)

        form = QFormLayout(self)

        self.name_edit = QLineEdit()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["rtsp", "usb"])
        self.rtsp_edit = QLineEdit()
        self.device_combo = QComboBox()
        for idx, name in list_usb_cameras():
            self.device_combo.addItem(f"{name} ({idx})", idx)
        self.source_stack = QStackedWidget()
        self.source_stack.addWidget(self.rtsp_edit)
        self.source_stack.addWidget(self.device_combo)
        self.type_combo.currentTextChanged.connect(self._on_type_change)

        self.model_combo = QComboBox()
        try:
            models = [d for d in os.listdir(MODELS_PATH) if os.path.isdir(os.path.join(MODELS_PATH, d))]
        except Exception:
            models = []
        if not models:
            models = [camera.get("model", DEFAULT_MODEL) if camera else DEFAULT_MODEL]
        self.model_combo.addItems(models)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.draw_chk = QCheckBox()
        self.detect_chk = QCheckBox()
        self.record_chk = QCheckBox()
        self.hours_edit = QLineEdit()
        self.visible_edit = QLineEdit()
        self.record_edit = QLineEdit()
        self.path_edit = QLineEdit()
        self.btn_path = QPushButton("Wybierz")
        self.pre_spin = QSpinBox()
        self.pre_spin.setRange(0, 60)
        self.lost_spin = QSpinBox()
        self.lost_spin.setRange(0, 60)
        self.post_spin = QSpinBox()
        self.post_spin.setRange(0, 60)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.btn_path)

        form.addRow("Nazwa", self.name_edit)
        form.addRow("Typ źródła", self.type_combo)
        form.addRow("Adres/Urządzenie", self.source_stack)
        form.addRow("Model detekcji", self.model_combo)
        form.addRow("FPS", self.fps_spin)
        form.addRow("Próg pewności", self.conf_spin)
        form.addRow("Rysuj nakładki", self.draw_chk)
        form.addRow("Wykrywaj obiekty", self.detect_chk)
        form.addRow("Nagrywaj detekcje", self.record_chk)
        form.addRow("Godziny detekcji", self.hours_edit)
        form.addRow("Widoczne klasy", self.visible_edit)
        form.addRow("Klasy nagrywane", self.record_edit)
        form.addRow("Folder nagrań", path_layout)
        form.addRow("Pre seconds", self.pre_spin)
        form.addRow("Lost seconds", self.lost_spin)
        form.addRow("Post seconds", self.post_spin)

        # test rtsp
        self.test_btn = QPushButton("Test połączenia")
        self.test_status = QLabel("")
        form.addRow(self.test_btn, self.test_status)

        btns = QHBoxLayout()
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Anuluj")
        btns.addStretch(1)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)
        form.addRow(btns)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_path.clicked.connect(self._choose_path)
        self.test_btn.clicked.connect(self._test_source)

        self.result_camera = None
        if camera:
            self.load_camera(camera)
        else:
            self._on_type_change(self.type_combo.currentText())

    def _choose_path(self):
        d = QFileDialog.getExistingDirectory(self, "Wybierz folder nagrań", self.path_edit.text() or DEFAULT_RECORD_PATH)
        if d:
            self.path_edit.setText(d)

    def _on_type_change(self, value):
        if value == "usb":
            self.source_stack.setCurrentWidget(self.device_combo)
        else:
            self.source_stack.setCurrentWidget(self.rtsp_edit)

    def _test_source(self):
        self.test_status.setText("Testuję...")
        self.test_status.setStyleSheet("color:#ccc;")
        if self.type_combo.currentText() == "usb":
            idx = self.device_combo.currentData()
            cap = cv2.VideoCapture(int(idx))
        else:
            url = self.rtsp_edit.text().strip()
            cap = cv2.VideoCapture(url)
        ok, _ = cap.read()
        cap.release()
        if ok:
            self.test_status.setText("✅ OK")
            self.test_status.setStyleSheet("color:#0f0;")
        else:
            self.test_status.setText("⚠️ Błąd")
            self.test_status.setStyleSheet("color:#f80;")

    def load_camera(self, cam):
        cam = cam or {}
        self.name_edit.setText(cam.get("name", ""))
        src_type = cam.get("type", "rtsp")
        self.type_combo.setCurrentText(src_type)
        if src_type == "usb":
            idx = int(cam.get("rtsp", 0))
            # select matching device if present
            i = self.device_combo.findData(idx)
            if i >= 0:
                self.device_combo.setCurrentIndex(i)
            self.source_stack.setCurrentWidget(self.device_combo)
        else:
            self.rtsp_edit.setText(str(cam.get("rtsp", "")))
            self.source_stack.setCurrentWidget(self.rtsp_edit)
        self.model_combo.setCurrentText(cam.get("model", DEFAULT_MODEL))
        self.fps_spin.setValue(int(cam.get("fps", DEFAULT_FPS)))
        self.conf_spin.setValue(float(cam.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)))
        self.draw_chk.setChecked(bool(cam.get("draw_overlays", DEFAULT_DRAW_OVERLAYS)))
        self.detect_chk.setChecked(bool(cam.get("enable_detection", DEFAULT_ENABLE_DETECTION)))
        self.record_chk.setChecked(bool(cam.get("enable_recording", DEFAULT_ENABLE_RECORDING)))
        self.hours_edit.setText(cam.get("detection_hours", DEFAULT_DETECTION_HOURS))
        self.visible_edit.setText(",".join(cam.get("visible_classes", VISIBLE_CLASSES)))
        self.record_edit.setText(",".join(cam.get("record_classes", RECORD_CLASSES)))
        self.path_edit.setText(cam.get("record_path", DEFAULT_RECORD_PATH))
        self.pre_spin.setValue(int(cam.get("pre_seconds", DEFAULT_PRE_SECONDS)))
        self.lost_spin.setValue(int(cam.get("lost_seconds", DEFAULT_LOST_SECONDS)))
        self.post_spin.setValue(int(cam.get("post_seconds", DEFAULT_POST_SECONDS)))

    def accept(self):
        name = self.name_edit.text().strip()
        if self.type_combo.currentText() == "usb":
            url = int(self.device_combo.currentData())
            if not name:
                QMessageBox.warning(self, "Błąd", "Nazwa jest wymagana")
                return
        else:
            url = self.rtsp_edit.text().strip()
            if not name or not url:
                QMessageBox.warning(self, "Błąd", "Nazwa i adres RTSP są wymagane")
                return
        cam = {
            "name": name,
            "rtsp": url,
            "type": self.type_combo.currentText(),
            "model": self.model_combo.currentText(),
            "fps": int(self.fps_spin.value()),
            "confidence_threshold": float(self.conf_spin.value()),
            "draw_overlays": self.draw_chk.isChecked(),
            "enable_detection": self.detect_chk.isChecked(),
            "enable_recording": self.record_chk.isChecked(),
            "detection_hours": self.hours_edit.text().strip() or DEFAULT_DETECTION_HOURS,
            "visible_classes": [c.strip() for c in self.visible_edit.text().split(",") if c.strip()],
            "record_classes": [c.strip() for c in self.record_edit.text().split(",") if c.strip()],
            "record_path": self.path_edit.text().strip() or DEFAULT_RECORD_PATH,
            "pre_seconds": int(self.pre_spin.value()),
            "lost_seconds": int(self.lost_spin.value()),
            "post_seconds": int(self.post_spin.value()),
        }
        self.result_camera = cam
        super().accept()

# --- Dialog usuwania kamer ---
class RemoveCameraDialog(QDialog):
    def __init__(self, cameras, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Usuń kamerę")
        self.resize(420, 360)
        self.cameras = cameras

        v = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.ExtendedSelection)
        for cam in cameras:
            item = QListWidgetItem(cam["name"])
            item.setData(Qt.UserRole, cam["name"])
            self.list.addItem(item)
        v.addWidget(QLabel("Wybierz kamery do usunięcia:"))
        v.addWidget(self.list)

        h = QHBoxLayout()
        self.btn_cancel = QPushButton("Anuluj")
        self.btn_ok = QPushButton("Usuń")
        h.addStretch(1)
        h.addWidget(self.btn_cancel)
        h.addWidget(self.btn_ok)
        v.addLayout(h)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.do_remove)

    def do_remove(self):
        names = [i.data(Qt.UserRole) for i in self.list.selectedItems()]
        if not names:
            QMessageBox.information(self, "Usuń kamerę", "Nie wybrano kamer.")
            return
        msg = "Czy na pewno usunąć: " + ", ".join(names) + " ?"
        if QMessageBox.question(self, "Potwierdzenie", msg,
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.removed = names
        self.accept()


# --- Dialog zarządzania kamerami ---
class CameraSettingsDialog(QDialog):
    def __init__(self, cameras, start_cb, stop_cb, test_cb, settings_cb, delete_cb, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zarządzanie kamerami")
        self.cameras = cameras
        self.start_cb = start_cb
        self.stop_cb = stop_cb
        self.test_cb = test_cb
        self.settings_cb = settings_cb
        self.delete_cb = delete_cb

        v = QVBoxLayout(self)
        self.combo = QComboBox()
        for cam in cameras:
            self.combo.addItem(cam.get("name", ""))
        v.addWidget(self.combo)

        btns = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_test = QPushButton("Test połączenia")
        self.btn_copy = QPushButton("Kopiuj RTSP")
        self.btn_settings = QPushButton("Ustawienia…")
        self.btn_delete = QPushButton("Usuń")
        for b in (self.btn_start, self.btn_stop, self.btn_test, self.btn_copy, self.btn_settings, self.btn_delete):
            btns.addWidget(b)
        v.addLayout(btns)

        self.form = SingleCameraDialog(self, cameras[0] if cameras else None)
        self.form.setWindowFlags(Qt.Widget)
        self.form.btn_ok.hide()
        self.form.btn_cancel.hide()
        v.addWidget(self.form)

        self.combo.currentIndexChanged.connect(self._on_idx_change)
        self.btn_start.clicked.connect(lambda: self.start_cb(self.combo.currentIndex()))
        self.btn_stop.clicked.connect(lambda: self.stop_cb(self.combo.currentIndex()))
        self.btn_test.clicked.connect(lambda: self.test_cb(self.combo.currentIndex()))
        self.btn_copy.clicked.connect(self._copy_rtsp)
        self.btn_settings.clicked.connect(lambda: self.settings_cb(self.combo.currentIndex()))
        self.btn_delete.clicked.connect(lambda: self.delete_cb(self.combo.currentIndex()))

        self._on_idx_change(self.combo.currentIndex())

    def _on_idx_change(self, idx):
        if 0 <= idx < len(self.cameras):
            self.form.load_camera(self.cameras[idx])

    def _copy_rtsp(self):
        idx = self.combo.currentIndex()
        if 0 <= idx < len(self.cameras):
            QApplication.clipboard().setText(str(self.cameras[idx]["rtsp"]), QClipboard.Clipboard)
            QMessageBox.information(self, "Skopiowano", "Adres RTSP skopiowany do schowka.")

# --- Dialog listy kamer ---
class CameraListDialog(QDialog):
    camera_selected = pyqtSignal(int)

    def __init__(self, grid_widget: CameraGridWidget, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowState(self.windowState() | Qt.WindowFullScreen)
        self.setStyleSheet("background:rgba(0,0,0,0.6);")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setAlignment(Qt.AlignCenter)
        self.grid = grid_widget
        self.grid.setParent(self)
        self.grid.show()
        layout.addWidget(self.grid)
        self.grid.camera_clicked.connect(self._on_item_clicked)

    def _on_item_clicked(self, index):
        self.camera_selected.emit(index)
        self.accept()

# --- GŁÓWNE OKNO ---
class MainWindow(QMainWindow):
    def __init__(self, cameras):
        super().__init__()
        self.setWindowTitle("AI Monitoring – PyQt5 (pełne GUI)")
        self.resize(1400, 900)

        # Pamięć alertów
        self.alert_mem = AlertMemory(ALERTS_HISTORY_PATH, max_items=5000)
        self.last_detected_label = ""
        self.sound_enabled = True
        self.last_detection_ids = {}
        self.active_recording_ids = {}
        # Starts that arrive before their log entry is created
        self.pending_record_starts = {}

        # Precompute alert sound once
        self.alert_sound = QSoundEffect()
        try:
            fs = 44100
            t = np.linspace(0, 1, fs, False)
            tone = np.sin(2 * np.pi * 880 * t)
            pulse = (np.sin(2 * np.pi * 5 * t) > 0).astype(float)
            envelope = np.linspace(1, 0, fs)
            audio = (tone * pulse * envelope * 0.5 * 32767).astype(np.int16)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(fs)
                f.writeframes(audio.tobytes())
            data = base64.b64encode(buf.getvalue()).decode()
            self.alert_sound.setSource(QUrl.fromEncoded(f"data:audio/wav;base64,{data}".encode()))
            self.alert_sound.setLoopCount(1)
            self.alert_sound.setVolume(1.0)
        except Exception as e:
            print(f"Failed to initialize alert sound: {e}")

        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: black;")
        main_vlayout = QVBoxLayout(main_widget)
        main_vlayout.setContentsMargins(10,10,10,10)

        main_hlayout = QHBoxLayout()
        main_hlayout.setContentsMargins(10,10,10,10)
        main_hlayout.setSpacing(10)

        self.cameras = list(cameras)
        self.output_dir = self.cameras[0].get("record_path", DEFAULT_RECORD_PATH) if self.cameras else DEFAULT_RECORD_PATH

        self.camera_list = CameraListWidget(self.cameras)
        self.camera_list.hide()

        self.camera_grid = CameraGridWidget(self.cameras)
        self.camera_grid.hide()

        self.log_window = LogWindow(LOG_HISTORY_PATH, LOG_RETENTION_HOURS)
        self.log_window.load_history()
        main_hlayout.addWidget(self.log_window)

        # Centrum: panel z obrazem
        self.center_panel = QWidget()
        center_v = QVBoxLayout(self.center_panel)
        center_v.setContentsMargins(0,0,0,0)
        center_v.setSpacing(10)
        self.camera_view = QLabel("")
        self.camera_view.setMinimumSize(800, 600)
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet("background:#000; color:#fff; border: 1px solid red;")
        center_v.addWidget(self.camera_view, stretch=1)
        self.camera_view.mouseDoubleClickEvent = lambda e: self.toggle_fullscreen()

        controls_widget = QWidget()
        controls_widget.setStyleSheet("background: transparent; border: 1px solid red;")
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0,50,0,50)
        controls_layout.setSpacing(20)
        controls_layout.setAlignment(Qt.AlignCenter)

        btn_cameras = QToolButton()
        btn_cameras.setIcon(QIcon(str(ICON_DIR / "camera-video.svg")))
        btn_cameras.setIconSize(QSize(50, 50))
        btn_cameras.clicked.connect(self.open_camera_list_dialog)

        btn_recordings = QToolButton()
        btn_recordings.setIcon(QIcon(str(ICON_DIR / "folder.svg")))
        if btn_recordings.icon().isNull():
            print("Ostrzeżenie: nie udało się załadować ikony folder.svg")
        btn_recordings.setIconSize(QSize(50, 50))
        btn_recordings.clicked.connect(self.open_recordings_browser)

        btn_settings = QToolButton()
        btn_settings.setIcon(QIcon(str(ICON_DIR / "gear.svg")))
        btn_settings.setIconSize(QSize(50, 50))
        btn_settings.clicked.connect(self.open_settings)

        btn_cam_ctrl = QToolButton()
        btn_cam_ctrl.setIcon(QIcon(str(ICON_DIR / "sliders.svg")))
        btn_cam_ctrl.setIconSize(QSize(50, 50))
        btn_cam_ctrl.clicked.connect(self.open_camera_settings)

        btn_alerts = QToolButton()
        btn_alerts.setIcon(QIcon(str(ICON_DIR / "exclamation-square.svg")))
        btn_alerts.setIconSize(QSize(50, 50))
        btn_alerts.clicked.connect(self.open_alert_dialog)

        self.btn_sound = QToolButton()
        self.btn_sound.setIcon(QIcon(str(ICON_DIR / "volume-up.svg")))
        self.btn_sound.setIconSize(QSize(50, 50))
        self.btn_sound.clicked.connect(self.toggle_sound)

        btn_fullscreen = QToolButton()
        btn_fullscreen.setIcon(QIcon(str(ICON_DIR / "window-fullscreen.svg")))
        btn_fullscreen.setIconSize(QSize(50, 50))
        btn_fullscreen.clicked.connect(self.toggle_fullscreen)

        btn_style = """
QToolButton {
    background: transparent;
    border: none;
    padding: 0px;
    color: white;
}
QToolButton:hover { background: #ff6666; }  # jasnoczerwone tło po najechaniu
QToolButton:focus { outline: none; }
        """

        for btn in (btn_cameras, btn_recordings, btn_settings, btn_cam_ctrl, btn_alerts, self.btn_sound, btn_fullscreen):
            btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
            btn.setAutoRaise(True)
            btn.setStyleSheet(btn_style)

        controls_layout.addStretch()
        controls_layout.addWidget(btn_cameras)
        controls_layout.addWidget(btn_recordings)
        controls_layout.addWidget(btn_settings)
        controls_layout.addWidget(btn_cam_ctrl)
        controls_layout.addWidget(btn_alerts)
        controls_layout.addWidget(self.btn_sound)
        controls_layout.addWidget(btn_fullscreen)
        controls_layout.addStretch()

        center_v.addWidget(controls_widget)

        main_hlayout.addWidget(self.center_panel, stretch=1)

        self.alert_list = AlertListWidget(self.alert_mem)
        main_hlayout.addWidget(self.alert_list)

        main_vlayout.addLayout(main_hlayout)

        self.setCentralWidget(main_widget)

        self.log_window.add_entry("application", "aplikacja uruchomiona")

        # backend
        self.workers = []
        self.camera_list.currentRowChanged.connect(self.switch_camera)

        self.alert_list.open_video.connect(self.open_video_file)

        # FPS liczniki i HUD stan
        self._fps_times = {}
        self._last_frame = {}
        self._last_status = {}
        self._last_error = {}
        self._last_fps_text = {}

        # zacznij od startu wszystkich, ale z niewielkim opóźnieniem aby GUI
        # mogło się pojawić bez czekania na inicjalizację kamer
        QTimer.singleShot(0, self.start_all)

    def restart_app(self):
        if QMessageBox.question(
            self,
            "Restart aplikacji",
            "Czy na pewno zrestartować aplikację?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        ) != QMessageBox.Yes:
            return
        self.log_window.add_entry("application", "restart aplikacji")
        try:
            self.stop_all()
        except Exception:
            pass
        python = sys.executable
        os.execl(python, python, *sys.argv)

    def toggle_fullscreen(self):
        if getattr(self, "_is_fullscreen", False):
            self.showNormal()
            self._is_fullscreen = False
        else:
            self.showFullScreen()
            self._is_fullscreen = True

    def toggle_sound(self):
        self.sound_enabled = not self.sound_enabled
        icon = "volume-up.svg" if self.sound_enabled else "volume-mute.svg"
        self.btn_sound.setIcon(QIcon(str(ICON_DIR / icon)))
        if self.alert_sound:
            self.alert_sound.setVolume(1.0 if self.sound_enabled else 0.0)
        state = "włączono" if self.sound_enabled else "wyłączono"
        self.log_window.add_entry("application", f"{state} powiadomienia dźwiękowe")

    def play_alert_sound(self):
        if self.alert_sound:
            try:
                self.alert_sound.play()
            except Exception as e:
                print(f"Alert sound playback failed: {e}")

    def open_alert_dialog(self):
        dlg = AlertDialog(self)
        dlg.exec_()

    def open_camera_settings(self):
        self.log_window.add_entry("settings", "otwarto ustawienia kamer")
        dlg = CameraSettingsDialog(
            self.cameras,
            start_cb=self.start_camera,
            stop_cb=self.stop_camera,
            test_cb=self.test_camera,
            settings_cb=self.camera_settings,
            delete_cb=self.delete_camera,
            parent=self,
        )
        dlg.exec_()

    # --- Alerty ---
    def on_new_alert(self, alert: dict):
        """Handle an incoming detection alert and align recording state.

        A start signal for the same camera may have been received earlier and
        is stored in ``pending_record_starts`` until a log ID is available."""
        self.alert_list.add_alert(alert)
        self.alert_mem.add(alert)
        cam = alert.get("camera", "kamera")
        label = alert.get("label", "obiekt")
        self.last_detected_label = label
        log_id = self.log_window.add_entry("detection", cam, "", label)
        self.last_detection_ids[cam] = log_id
        # If a recording start came before this alert, finalize the association now
        if cam in self.pending_record_starts:
            self.active_recording_ids[cam] = log_id
            self.log_window.update_recording_by_id(log_id, "started")
            self.pending_record_starts.pop(cam, None)
        cam_cfg = next((c for c in self.cameras if c.get("name") == cam), {})
        if not cam_cfg.get("enable_recording", True):
            self.log_window.update_recording_by_id(log_id, "det_started")
        if self.sound_enabled:
            self.play_alert_sound()

    def on_record_event(self, event: str, filepath: str, cam_name: str):
        """Process recording start/stop signals.

        The worker may emit a start before :meth:`on_new_alert` creates the
        corresponding log entry. Such starts are stored in
        ``pending_record_starts`` and resolved when the alert arrives.
        """
        if event == "start":
            log_id = self.last_detection_ids.get(cam_name)
            if log_id:
                self.active_recording_ids[cam_name] = log_id
                self.log_window.update_recording_by_id(log_id, "started")
            else:
                # Start arrived before alert; remember it until log ID exists
                self.pending_record_starts[cam_name] = filepath
        elif event == "stop":
            log_id = self.active_recording_ids.pop(cam_name, None)
            if log_id:
                self.log_window.update_recording_by_id(log_id, "finished")
            # Clear any pending start for this camera
            self.pending_record_starts.pop(cam_name, None)

    # --- Zarządzanie kamerami ---

    def start_camera(self, idx: int):
        if idx < 0 or idx >= len(self.cameras):
            return
        if idx < len(self.workers) and isinstance(self.workers[idx], CameraWorker) and self.workers[idx].isRunning():
            return
        while len(self.workers) < len(self.cameras):
            self.workers.append(None)
        cam = self.cameras[idx]
        model_name = cam.get("model", DEFAULT_MODEL)
        try:
            model = dg.load_model(
                model_name=model_name,
                inference_host_address="@local",
                zoo_url=os.path.join(MODELS_PATH, model_name),
            )
        except Exception as e:
            QMessageBox.warning(self, "Model", f"Nie udało się załadować modelu '{model_name}': {e}")
            self.log_window.add_entry("error", f"model {model_name}: {e}")
            return
        w = CameraWorker(camera=cam, model=model, index=idx)
        w.frame_signal.connect(self.update_frame)
        w.alert_signal.connect(self.on_new_alert)
        w.error_signal.connect(self._worker_error)
        w.status_signal.connect(self._worker_status)
        w.record_signal.connect(lambda event, fp, cam_name=cam.get("name", idx): self.on_record_event(event, fp, cam_name))
        w.start()
        self.workers[idx] = w
        self.log_window.add_entry("application", f"uruchomiono kamerę {cam.get('name', idx)}")

    def stop_camera(self, idx: int):
        if 0 <= idx < len(self.workers):
            w = self.workers[idx]
            if isinstance(w, CameraWorker):
                w.stop()
                self.workers[idx] = None
                cam = self.cameras[idx]
                self.log_window.add_entry("application", f"zatrzymano kamerę {cam.get('name', idx)}")


    def _worker_status(self, text: str, idx: int):
        self._last_status[idx] = text
        if idx == self.camera_list.currentRow():
            self._render_current()

    def _worker_error(self, msg: str, idx: int):
        # map known causes to crisp labels (already mapped in worker, but double safety)
        m = str(msg).lower()
        cause = None
        if "401" in m or "unauthorized" in m or "auth" in m:
            cause = "Auth/401"
        elif "timed out" in m or "timeout" in m:
            cause = "Timeout"
        elif "name or service not known" in m or "getaddrinfo" in m or "dns" in m:
            cause = "DNS"
        elif "connection refused" in m:
            cause = "Connection refused"
        elif "no route to host" in m:
            cause = "No route to host"
        elif "pusta klatka" in m or "empty frame" in m:
            cause = "Brak sygnału (pusta klatka)"
        else:
            cause = str(msg)
        self._last_error[idx] = cause
        cam_name = self.cameras[idx]["name"] if idx < len(self.cameras) else str(idx)
        self.log_window.add_entry("error", f"{cam_name}: {cause}")
        if "Brak sygnału" in cause:
            self.log_window.add_entry("application", "brak sygnału RTSP")
        if idx == self.camera_list.currentRow():
            self._render_current()
        print(msg)

    # --- Zarządzanie kamerami (global) ---
    def add_camera_wizard(self):
        dlg = AddCameraWizard(self)
        if dlg.exec_():
            data = dlg.result_data
            _fill_camera_defaults(data)
            cfg = load_config()
            if any(c["name"] == data["name"] for c in self.cameras):
                QMessageBox.warning(self, "Duplikat", f"Kamera o nazwie '{data['name']}' już istnieje.")
                return
            self.cameras.append(data)
            cfg["cameras"] = self.cameras
            save_config(cfg)
            self.restart_workers_and_ui()
            self.log_window.add_entry("settings", f"dodano kamerę {data.get('name')}")

    def add_usb_camera(self):
        dlg = AddUsbCameraDialog(self)
        if dlg.exec_():
            data = dlg.result_data
            _fill_camera_defaults(data)
            cfg = load_config()
            if any(c["name"] == data["name"] for c in self.cameras):
                QMessageBox.warning(self, "Duplikat", f"Kamera o nazwie '{data['name']}' już istnieje.")
                return
            self.cameras.append(data)
            cfg["cameras"] = self.cameras
            save_config(cfg)
            self.restart_workers_and_ui()
            self.log_window.add_entry("settings", f"dodano kamerę {data.get('name')}")

    def camera_settings(self, idx: int):
        cam = self.cameras[idx]
        dlg = SingleCameraDialog(self, cam)
        if dlg.exec_():
            new_data = dlg.result_camera
            if new_data["name"] != cam["name"] and any(c["name"] == new_data["name"] for c in self.cameras):
                QMessageBox.warning(self, "Duplikat", f"Kamera o nazwie '{new_data['name']}' już istnieje.")
                return
            model_changed = new_data.get("model") != cam.get("model")
            self.cameras[idx] = new_data
            cfg = load_config()
            cfg["cameras"] = self.cameras
            save_config(cfg)
            self.camera_list.rebuild(self.cameras)
            self.camera_grid.rebuild(self.cameras)
            self.camera_list.setCurrentRow(idx)
            self.log_window.add_entry("settings", f"zapisano ustawienia kamery {new_data.get('name')}")
            if model_changed:
                self.log_window.add_entry("settings", "zmieniono model")

            w = self.workers[idx] if idx < len(self.workers) else None
            if isinstance(w, CameraWorker) and w.isRunning():
                if model_changed:
                    self.stop_camera(idx)
                    self.start_camera(idx)
                else:
                    w.set_confidence(new_data.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
                    w.set_draw_overlays(new_data.get("draw_overlays", DEFAULT_DRAW_OVERLAYS))
                    w.set_enable_detection(new_data.get("enable_detection", DEFAULT_ENABLE_DETECTION))
                    w.set_enable_recording(new_data.get("enable_recording", DEFAULT_ENABLE_RECORDING))
                    w.set_detection_schedule(new_data.get("detection_hours", DEFAULT_DETECTION_HOURS))
                    w.visible_classes = list(new_data.get("visible_classes", VISIBLE_CLASSES))
                    w.record_classes = list(new_data.get("record_classes", RECORD_CLASSES))
                    w.pre_seconds = int(new_data.get("pre_seconds", DEFAULT_PRE_SECONDS))
                    w.lost_seconds = int(new_data.get("lost_seconds", DEFAULT_LOST_SECONDS))
                    w.post_seconds = int(new_data.get("post_seconds", DEFAULT_POST_SECONDS))
                    w.prerecord_buffer = deque(maxlen=int(w.pre_seconds * w.fps))
                    w.output_dir = os.path.join(new_data.get("record_path", DEFAULT_RECORD_PATH), new_data.get("name"))
                    os.makedirs(w.output_dir, exist_ok=True)
                    w.camera.update(new_data)
                    if new_data.get("rtsp") != cam.get("rtsp"):
                        w.restart_requested = True
                    if new_data.get("fps") != cam.get("fps"):
                        w.set_fps(new_data.get("fps", w.fps))
            else:
                self.stop_camera(idx)
                self.start_camera(idx)

    def delete_camera(self, idx: int):
        name = self.cameras[idx]["name"]
        if QMessageBox.question(self, "Usuń kamerę",
                                f"Czy na pewno usunąć '{name}'?",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.stop_camera(idx)
        del self.cameras[idx]
        cfg = load_config()
        cfg["cameras"] = self.cameras
        save_config(cfg)
        self.restart_workers_and_ui()

    def remove_camera_dialog(self):
        if not self.cameras:
            QMessageBox.information(self, "Usuń kamerę", "Brak zdefiniowanych kamer.")
            return
        dlg = RemoveCameraDialog(self.cameras, self)
        if dlg.exec_():
            names = set(dlg.removed)
            for i in reversed(range(len(self.cameras))):
                if self.cameras[i]["name"] in names:
                    self.stop_camera(i)
                    del self.cameras[i]
            cfg = load_config()
            cfg["cameras"] = self.cameras
            save_config(cfg)
            self.restart_workers_and_ui()

    def restart_workers_and_ui(self):
        self.stop_all()
        self.camera_list.rebuild(self.cameras)
        self.camera_grid.rebuild(self.cameras)
        self.workers = [None] * len(self.cameras)
        self.start_all()

    def test_camera(self, idx: int):
        url = self.cameras[idx]["rtsp"]
        if self.cameras[idx].get("type") == "usb":
            try:
                url = int(url)
            except Exception:
                pass
        cap = cv2.VideoCapture(url)
        ok, _ = cap.read()
        cap.release()
        if ok:
            QMessageBox.information(self, "Test połączenia", f"✅ Połączenie OK dla: {self.cameras[idx]['name']}")
        else:
            QMessageBox.warning(self, "Test połączenia", f"⚠️ Nie udało się odczytać klatki:\n{url}")

    def start_all(self):
        self.stop_all()
        self.workers = [None] * len(self.cameras)
        for idx in range(len(self.cameras)):
            self.start_camera(idx)
        if self.camera_list.currentRow() < 0 and self.cameras:
            self.camera_list.setCurrentRow(0)
        # przy starcie — brak klatki jeszcze: narysuj HUD "Łączenie…"
        self._last_status[self.camera_list.currentRow()] = "Łączenie…"
        self._render_current()

    def stop_all(self):
        for w in self.workers:
            if isinstance(w, CameraWorker):
                w.stop()
        self.workers = []

    def switch_camera(self, idx):
        # odśwież HUD dla nowej kamery
        self._render_current()

    def update_frame(self, frame, index):
        self.camera_list.update_thumbnail(index, frame)
        self.camera_grid.update_frame(index, frame)

        # FPS liczenie dla tej kamery
        from time import perf_counter
        t = perf_counter()
        dq = self._fps_times.setdefault(index, [])
        dq.append(t)
        if len(dq) > 60:
            del dq[0:len(dq)-60]
        fps_txt = ""
        if len(dq) >= 2:
            dt = dq[-1] - dq[0]
            if dt > 0:
                fps_now = (len(dq)-1) / dt
                fps_txt = f"{fps_now:.1f} fps"

        # zapisz stan
        self._last_frame[index] = frame
        self._last_fps_text[index] = fps_txt
        self._last_status[index] = "Połączono"
        self._last_error.pop(index, None)

        if index == self.camera_list.currentRow():
            self._render_current()

    def _compose_letterboxed(self, frame, top_lines):
        # Create canvas matching label size, paste scaled frame centered
        w_label = max(1, self.camera_view.width())
        h_label = max(1, self.camera_view.height())
        canvas = np.zeros((h_label, w_label, 3), dtype=np.uint8)

        top_bar_h = 0
        y0 = 0
        if frame is not None:
            fh, fw = frame.shape[:2]
            if fh > 0 and fw > 0:
                scale = min(w_label / fw, h_label / fh)
                new_w = max(1, int(fw * scale))
                new_h = max(1, int(fh * scale))
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                x0 = (w_label - new_w) // 2
                y0 = (h_label - new_h) // 2
                top_bar_h = max(40, y0)  # wysokość górnego paska (min 40px)
                canvas[y0:y0+new_h, x0:x0+new_w] = resized
        else:
            top_bar_h = 60

        # Convert to QImage and draw text (UTF-8 capable), centered
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w_label, h_label, rgb.strides[0], QImage.Format_RGB888).copy()
        painter = QPainter(qimg)
        try:
            # dynamic font size vs width
            base_size = 14
            if w_label > 1200:
                base_size = 20
            elif w_label > 900:
                base_size = 18
            elif w_label > 700:
                base_size = 16

            font1 = QFont("DejaVu Sans", base_size)
            font1.setBold(False)
            font2 = QFont("DejaVu Sans", max(12, base_size - 2))
            font2.setBold(False)

            painter.setPen(QColor(255, 255, 255))

            # Rects for lines on the top bar
            line1_rect = QRect(0, 6, w_label, max(24, top_bar_h//2))
            line2_rect = QRect(0, line1_rect.bottom() + 2, w_label, max(22, top_bar_h - line1_rect.height()))

            if top_lines and len(top_lines) > 0 and top_lines[0]:
                painter.setFont(font1)
                painter.drawText(line1_rect, Qt.AlignHCenter | Qt.AlignVCenter, top_lines[0])
            if top_lines and len(top_lines) > 1 and top_lines[1]:
                painter.setFont(font2)
                painter.drawText(line2_rect, Qt.AlignHCenter | Qt.AlignVCenter, top_lines[1])
        finally:
            painter.end()

        return qimg

    def _render_current(self):
        idx = self.camera_list.currentRow()
        if idx < 0:
            return
        frame = self._last_frame.get(idx)
        name = self.cameras[idx]["name"] if (0 <= idx < len(self.cameras)) else "-"
        fps_txt = self._last_fps_text.get(idx, "")
        status = self._last_status.get(idx, "")
        err = self._last_error.get(idx, "")
        # Prepare lines: name, fps, status/error
        line1 = name
        line2_parts = []
        if fps_txt:
            line2_parts.append(fps_txt)
        if err:
            line2_parts.append(f"Błąd: {err}")
        elif status:
            line2_parts.append(status)
        line2 = "  |  ".join(line2_parts) if line2_parts else ""

        composed_qimg = self._compose_letterboxed(frame if frame is not None else np.zeros((720,1280,3), dtype=np.uint8), [line1, line2])
        self.camera_view.setPixmap(QPixmap.fromImage(composed_qimg))


    def open_video_file(self, filepath: str):
        self.log_window.add_entry("application", f"odtworzono nagranie {os.path.basename(filepath)}")
        dlg = VideoPlayerDialog(filepath, self)
        dlg.exec_()

    def open_recordings_browser(self):
        self.log_window.add_entry("application", "otwarto przeglądarkę nagrań")
        dlg = RecordingsBrowserDialog(self.output_dir, self.cameras, self)
        dlg.open_video.connect(self.open_video_file)
        dlg.exec_()

    def open_camera_list_dialog(self):
        self.log_window.add_entry("application", "otwarto listę kamer")
        dlg = CameraListDialog(self.camera_grid, self)
        dlg.camera_selected.connect(lambda idx: self.camera_list.setCurrentRow(idx))
        dlg.exec_()
        self.camera_grid.setParent(None)
        self.camera_grid.hide()

    def closeEvent(self, event):
        self.stop_all()
        event.accept()

    def open_settings(self):
        self.log_window.add_entry("settings", "otworzono ustawienia")
        dlg = SettingsHub(self)
        dlg.exec_()


# --- Centrum ustawień ---
class SettingsHub(QDialog):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        self.setWindowTitle("Menu ustawień")
        self.resize(300, 200)

        layout = QVBoxLayout(self)

        btn_add_cam = QPushButton("Dodaj kamerę RTSP")
        btn_add_usb = QPushButton("Dodaj kamerę USB")
        btn_remove_cam = QPushButton("Usuń kamerę")
        btn_restart = QPushButton("Restart aplikacji")
        btn_close = QPushButton("Zamknij")

        for b in [btn_add_cam, btn_add_usb, btn_remove_cam, btn_restart, btn_close]:
            layout.addWidget(b)

        btn_add_cam.clicked.connect(parent.add_camera_wizard)
        btn_add_usb.clicked.connect(parent.add_usb_camera)
        btn_remove_cam.clicked.connect(parent.remove_camera_dialog)
        btn_restart.clicked.connect(parent.restart_app)
        btn_close.clicked.connect(self.accept)

# --- START ---
def main(windowed: bool = False):
    cfg = load_config()
    app = QApplication(sys.argv)
    win = MainWindow(cameras=cfg.get("cameras", []))
    if windowed:
        win.show()
    else:
        win.showFullScreen()
    sys.exit(app.exec_())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APP-MONITORING launcher")
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Run application in a window instead of full screen",
    )
    args = parser.parse_args()
    main(windowed=args.windowed)

