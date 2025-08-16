
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
from glob import glob
import argparse

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QListWidget, QListWidgetItem,
    QVBoxLayout, QHBoxLayout, QPushButton, QSlider,
    QMenu, QFrame, QFileDialog, QDialog, QFormLayout,
    QComboBox, QMessageBox, QDateEdit, QLineEdit, QCheckBox, QStackedWidget,
    QSpinBox, QDoubleSpinBox, QToolButton, QStyle
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QDate, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QClipboard, QPainter, QFont, QColor, QIcon
from PyQt5 import QtSvg
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
    }
    for k, v in defaults.items():
        cam.setdefault(k, v)
    return cam


def load_config():
    if not os.path.exists(CONFIG_PATH):
        cfg = {
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

    for cam in cfg.get("cameras", []):
        _fill_camera_defaults(cam)
    return cfg


def save_config(cfg):
    for cam in cfg.get("cameras", []):
        _fill_camera_defaults(cam)
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
class CameraWorker(QThread):
    frame_signal = pyqtSignal(object, int)  # (np.ndarray BGR, index)
    alert_signal = pyqtSignal(object)       # dict z klatką i metadanymi
    error_signal = pyqtSignal(str, int)     # komunikat, index
    status_signal = pyqtSignal(str, int)    # status tekstowy, index

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
        self.video_writer = None
        self.output_file = None
        self.frames_since_last_detection = 0
        self.stop_signal = False

        self.prerecord_buffer = deque(maxlen=int(self.pre_seconds * self.fps))
        self.frame = None

        # hot-reload
        self.restart_requested = False

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
                for inference_result in degirum_tools.predict_stream(
                    self.model,
                    self.camera["rtsp"],
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

                    # Nagrywanie (pre + post)
                    if detected and self.enable_recording:
                        if not self.recording:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            self.output_file = os.path.join(self.output_dir, f"nagranie_{self.camera['name']}_{timestamp}.mp4")
                            h, w = self.frame.shape[:2]
                            self.video_writer = degirum_tools.VideoWriter(self.output_file, w, h, self.fps)
                            # pre-bufor
                            for bf in list(self.prerecord_buffer):
                                self.video_writer.write(bf)
                            self.recording = True
                            self.frames_since_last_detection = 0

                            thumb_path = self.output_file + ".jpg"
                            try:
                                cv2.imwrite(thumb_path, self.frame)
                            except Exception as ex:
                                print("Nie zapisano miniatury:", ex)

                            alert = {
                                "camera": self.camera["name"],
                                "label": best_label or "object",
                                "confidence": float(best_score),
                                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "frame": self.frame.copy(),
                                "filepath": self.output_file,
                                "thumb": thumb_path,
                            }
                            self.alert_signal.emit(alert)

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

                        else:
                            self.frames_since_last_detection = 0
                    else:
                        if self.recording:
                            self.frames_since_last_detection += 1
                            if self.frames_since_last_detection >= int(self.post_seconds * self.fps):
                                if self.video_writer:
                                    self.video_writer.release()
                                    self.video_writer = None
                                self.recording = False

                    if self.recording and self.video_writer:
                        self.video_writer.write(self.frame)

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

            if self.stop_signal:
                break
            # krótka pauza zanim spróbujemy ponownie (autoreconnect)
            QThread.msleep(300)

        # sprzątanie
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def stop(self):
        self.stop_signal = True
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.wait(2000)


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
        # Czytelne tło i biały tekst zamiast całkowitej przezroczystości
        self.setStyleSheet("background: #222; color: white;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setFixedWidth(300)
        self.list.setFrameShape(QFrame.NoFrame)
        self.list.setStyleSheet(
            "QListWidget{background: #222; border: none; color: white;}"
        )
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.list)

        self.menu_btn = QToolButton(self)
        self.menu_btn.setAutoRaise(True)
        self.menu_btn.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxWarning))
        self.menu_btn.setIconSize(QSize(24,24))
        self.menu_btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.menu_btn)
        menu.addAction("Wczytaj ponownie", self.reload)
        menu.addAction("Eksport do CSV", self.export_csv)
        menu.addAction("Wyczyść pamięć", self.clear)
        self.menu_btn.setMenu(menu)
        self.menu_btn.setFixedSize(32,32)
        self.menu_btn.raise_()

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

    def resizeEvent(self, e):
        super().resizeEvent(e)
        x = self.list.x() + self.list.width() - self.menu_btn.width() - 8
        y = self.list.y() + (self.list.height() - self.menu_btn.height()) // 2
        self.menu_btn.move(x, y)


class LogEntryWidget(QFrame):
    def __init__(self, group: str, text: str):
        super().__init__()
        colors = {
            "application": "#4aa3ff",
            "detection": "#4caf50",
            "settings": "#ff8800",
        }
        self.setStyleSheet(
            "QFrame{border:1px solid white; background:rgba(0,0,0,0.4);}" 
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setAlignment(Qt.AlignCenter)
        title = QLabel(group.capitalize())
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color:{colors.get(group, '#fff')}; font-weight:600;")
        msg = QLabel(text)
        msg.setAlignment(Qt.AlignCenter)
        msg.setStyleSheet("color:white;")
        layout.addWidget(title)
        layout.addWidget(msg)


class LogWindow(QListWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(300)
        self.setFrameShape(QFrame.NoFrame)
        self.setSpacing(8)
        self.setStyleSheet("QListWidget{background:transparent; border:none;}")

    def add_entry(self, group: str, text: str):
        widget = LogEntryWidget(group, text)
        item = QListWidgetItem(self)
        item.setSizeHint(widget.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, widget)
        self.scrollToItem(item)
        if self.count() > 200:
            self.takeItem(0)


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
        except Exception as e:
            QMessageBox.warning(self, "Błąd", str(e))

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
class RecordingItemWidget(QWidget):
    def __init__(self, meta: dict, thumb_size=(256, 144)):
        super().__init__()
        self.meta = meta
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setAlignment(Qt.AlignCenter)

        self.thumb = QLabel()
        self.thumb.setFixedSize(*thumb_size)
        self.thumb.setStyleSheet("border:1px solid #555; background:#111;")
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

        self.load_thumbnail()

    def load_thumbnail(self):
        file = self.meta.get("file", "")
        jpg = file + ".jpg"
        pix = None
        if os.path.exists(jpg):
            img = cv2.imread(jpg)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
        if pix is None and os.path.exists(file):
            cap = cv2.VideoCapture(file)
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
        if pix is None:
            p = QPixmap(self.thumb.size())
            p.fill(Qt.black)
            pix = p
        self.thumb.setPixmap(pix.scaled(self.thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class RecordingsBrowserDialog(QDialog):
    open_video = pyqtSignal(str)

    def __init__(self, base_dir, cameras, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nagrania – przeglądarka")
        self.resize(1100, 700)
        self.base_dir = base_dir
        self.cameras = cameras

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
        self.refresh_btn.clicked.connect(self.apply_filters)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.camera_filter.currentIndexChanged.connect(self.apply_filters)
        self.class_filter.currentIndexChanged.connect(self.apply_filters)
        self.date_from.dateChanged.connect(self.apply_filters)
        self.date_to.dateChanged.connect(self.apply_filters)
        self.search_text.textChanged.connect(self.apply_filters)
        self.list.itemDoubleClicked.connect(self.open_selected)
        self.list.customContextMenuRequested.connect(self._context_menu)

        # stan
        self.all_items = []
        self.scan_files()
        self.apply_filters()

    def scan_files(self):
        self.all_items.clear()
        if not os.path.isdir(self.base_dir):
            return
        for cam in self.cameras:
            cam_dir = os.path.join(self.base_dir, cam["name"])
            if not os.path.isdir(cam_dir):
                continue
            for mp4 in sorted(glob(os.path.join(cam_dir, "nagranie_*.mp4")), reverse=True):
                meta_path = mp4 + ".json"
                meta = {
                    "camera": cam["name"],
                    "label": "unknown",
                    "confidence": 0.0,
                    "time": None,
                    "file": mp4
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
                self.all_items.append(meta)

    def meta_in_date_range(self, meta, qfrom: QDate, qto: QDate):
        try:
            dt = datetime.datetime.strptime(meta.get("time", ""), "%Y-%m-%d %H:%M:%S")
            d = QDate(dt.year, dt.month, dt.day)
            return (d >= qfrom) and (d <= qto)
        except Exception:
            return True

    def apply_filters(self):
        cam_sel = self.camera_filter.currentText()
        if cam_sel == "Wszystkie kamery":
            cam_sel = None
        cls_sel = self.class_filter.currentText()
        if cls_sel == "Wszystkie klasy":
            cls_sel = None
        qfrom = self.date_from.date()
        qto = self.date_to.date()
        text = self.search_text.text().strip().lower()

        self.list.clear()
        for meta in self.all_items:
            if cam_sel and meta.get("camera") != cam_sel:
                continue
            if cls_sel and meta.get("label") != cls_sel:
                continue
            if not self.meta_in_date_range(meta, qfrom, qto):
                continue
            if text and text not in os.path.basename(meta.get("file", "")).lower():
                continue

            widget = RecordingItemWidget(meta)
            item = QListWidgetItem()
            item.setSizeHint(widget.sizeHint())
            item.setData(Qt.UserRole, meta.get("file"))
            self.list.addItem(item)
            self.list.setItemWidget(item, widget)

    def open_selected(self, item: QListWidgetItem):
        fp = item.data(Qt.UserRole)
        if fp and os.path.exists(fp):
            self.open_video.emit(fp)

    def _selected_filepaths(self):
        return [i.data(Qt.UserRole) for i in self.list.selectedItems() if i.data(Qt.UserRole)]

    def delete_selected(self):
        paths = self._selected_filepaths()
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
        self.scan_files()
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
        self.result_data = {"name": name, "rtsp": url}
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


# --- Ustawienia pojedynczej kamery ---
class CameraSettingsDialog(QDialog):
    def __init__(self, parent=None, camera=None):
        super().__init__(parent)
        self.setWindowTitle("Ustawienia kamery")
        self.resize(480, 520)

        cam = camera or {}

        form = QFormLayout(self)

        self.name_edit = QLineEdit(cam.get("name", ""))
        self.rtsp_edit = QLineEdit(cam.get("rtsp", ""))
        self.model_combo = QComboBox()
        try:
            models = [d for d in os.listdir(MODELS_PATH) if os.path.isdir(os.path.join(MODELS_PATH, d))]
        except Exception:
            models = []
        if not models:
            models = [cam.get("model", DEFAULT_MODEL)]
        self.model_combo.addItems(models)
        self.model_combo.setCurrentText(cam.get("model", DEFAULT_MODEL))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(int(cam.get("fps", DEFAULT_FPS)))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(float(cam.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)))
        self.draw_chk = QCheckBox()
        self.draw_chk.setChecked(bool(cam.get("draw_overlays", DEFAULT_DRAW_OVERLAYS)))
        self.detect_chk = QCheckBox()
        self.detect_chk.setChecked(bool(cam.get("enable_detection", DEFAULT_ENABLE_DETECTION)))
        self.record_chk = QCheckBox()
        self.record_chk.setChecked(bool(cam.get("enable_recording", DEFAULT_ENABLE_RECORDING)))
        self.hours_edit = QLineEdit(cam.get("detection_hours", DEFAULT_DETECTION_HOURS))
        self.visible_edit = QLineEdit(",".join(cam.get("visible_classes", VISIBLE_CLASSES)))
        self.record_edit = QLineEdit(",".join(cam.get("record_classes", RECORD_CLASSES)))
        self.path_edit = QLineEdit(cam.get("record_path", DEFAULT_RECORD_PATH))
        self.btn_path = QPushButton("Wybierz")
        self.pre_spin = QSpinBox()
        self.pre_spin.setRange(0, 60)
        self.pre_spin.setValue(int(cam.get("pre_seconds", DEFAULT_PRE_SECONDS)))
        self.post_spin = QSpinBox()
        self.post_spin.setRange(0, 60)
        self.post_spin.setValue(int(cam.get("post_seconds", DEFAULT_POST_SECONDS)))

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.btn_path)

        form.addRow("Nazwa", self.name_edit)
        form.addRow("RTSP", self.rtsp_edit)
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
        self.test_btn.clicked.connect(self._test_rtsp)

        self.result_camera = None

    def _choose_path(self):
        d = QFileDialog.getExistingDirectory(self, "Wybierz folder nagrań", self.path_edit.text() or DEFAULT_RECORD_PATH)
        if d:
            self.path_edit.setText(d)

    def _test_rtsp(self):
        url = self.rtsp_edit.text().strip()
        self.test_status.setText("Testuję...")
        self.test_status.setStyleSheet("color:#ccc;")
        cap = cv2.VideoCapture(url)
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
        url = self.rtsp_edit.text().strip()
        if not name or not url:
            QMessageBox.warning(self, "Błąd", "Nazwa i adres RTSP są wymagane")
            return
        cam = {
            "name": name,
            "rtsp": url,
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


# --- Dialog listy kamer ---
class CameraListDialog(QDialog):
    camera_selected = pyqtSignal(int)

    def __init__(self, list_widget: CameraListWidget, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(500, 300)
        self.setStyleSheet(
            "background:rgba(0,0,0,0.6); border:1px solid white;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setAlignment(Qt.AlignCenter)
        self.list = list_widget
        self.list.setParent(self)
        self.list.show()
        layout.addWidget(self.list)
        self.list.itemClicked.connect(self._on_item_clicked)

    def _on_item_clicked(self, item):
        row = self.list.row(item)
        self.camera_selected.emit(row)
        self.accept()


# --- Dialog listy alertów ---
class AlertListDialog(QDialog):
    def __init__(self, alert_widget: AlertListWidget, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(320, 500)
        # półprzezroczyste tło z jasnym obramowaniem
        self.setStyleSheet(
            "background:rgba(0,0,0,0.8); border:1px solid white; color:white;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setAlignment(Qt.AlignCenter)
        self.list = alert_widget
        self.list.setParent(self)
        self.list.show()
        layout.addWidget(self.list)

# --- GŁÓWNE OKNO ---
class MainWindow(QMainWindow):
    def __init__(self, cameras):
        super().__init__()
        self.setWindowTitle("AI Monitoring – PyQt5 (pełne GUI)")
        self.resize(1400, 900)

        # Pamięć alertów
        self.alert_mem = AlertMemory(ALERTS_HISTORY_PATH, max_items=5000)

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
        self.camera_list.request_context.connect(self._show_camera_context_menu)
        self.camera_list.hide()

        self.log_window = LogWindow()
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
        btn_cameras.setIcon(QIcon(str(ICON_DIR / "pc-display-horizontal.svg")))
        btn_cameras.setIconSize(QSize(50, 50))
        btn_cameras.clicked.connect(self.open_camera_list_dialog)

        btn_alerts = QToolButton()
        btn_alerts.setIcon(QIcon(str(ICON_DIR / "exclamation-square.svg")))
        btn_alerts.setIconSize(QSize(50, 50))
        btn_alerts.clicked.connect(self.open_alerts_dialog)

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

        for btn in (btn_cameras, btn_recordings, btn_alerts, btn_settings, btn_fullscreen):
            btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
            btn.setAutoRaise(True)
            btn.setStyleSheet(btn_style)

        controls_layout.addStretch()
        controls_layout.addWidget(btn_cameras)
        controls_layout.addWidget(btn_recordings)
        controls_layout.addWidget(btn_alerts)
        controls_layout.addWidget(btn_settings)
        controls_layout.addWidget(btn_fullscreen)
        controls_layout.addStretch()

        center_v.addWidget(controls_widget)

        main_hlayout.addWidget(self.center_panel, stretch=1)

        self.alert_list = AlertListWidget(self.alert_mem)
        self.alert_list.hide()

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

        # zacznij od startu wszystkich (możesz zmienić na start tylko bieżącej, jeśli wolisz)
        self.start_all()

    def restart_app(self):
        if QMessageBox.question(
            self,
            "Restart aplikacji",
            "Czy na pewno zrestartować aplikację?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        ) != QMessageBox.Yes:
            return
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

    # --- Alerty ---
    def on_new_alert(self, alert: dict):
        self.alert_list.add_alert(alert)
        self.alert_mem.add(alert)
        label = alert.get("label", "obiekt")
        self.log_window.add_entry("detection", f"wykryto obiekt ({label})")
        self.log_window.add_entry("detection", "rozpoczęto nagrywanie")


    # --- MENU KONTEKSTOWE KAMERY ---
    def _show_camera_context_menu(self, row: int, global_pos: QPoint):
        if row < 0 or row >= len(self.cameras):
            return
        cam = self.cameras[row]
        menu = QMenu(self)

        running = row < len(self.workers) and isinstance(self.workers[row], CameraWorker) and self.workers[row].isRunning()
        act_start = menu.addAction("Start") if not running else None
        act_stop = menu.addAction("Stop") if running else None
        menu.addSeparator()
        act_test = menu.addAction("Test połączenia")
        act_copy = menu.addAction("Kopiuj RTSP")
        menu.addSeparator()
        act_settings = menu.addAction("Ustawienia…")
        act_del = menu.addAction("Usuń…")

        action = menu.exec_(global_pos)
        if action is None:
            return
        if action == act_start:
            self.start_camera(row)
        elif action == act_stop:
            self.stop_camera(row)
        elif action == act_test:
            self.test_camera(row)
        elif action == act_copy:
            QApplication.clipboard().setText(cam["rtsp"], QClipboard.Clipboard)
            QMessageBox.information(self, "Skopiowano", "Adres RTSP skopiowany do schowka.")
        elif action == act_settings:
            self.camera_settings(row)
        elif action == act_del:
            self.delete_camera(row)

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
            return
        w = CameraWorker(camera=cam, model=model, index=idx)
        w.frame_signal.connect(self.update_frame)
        w.alert_signal.connect(self.on_new_alert)
        w.error_signal.connect(self._worker_error)
        w.status_signal.connect(self._worker_status)
        w.start()
        self.workers[idx] = w

    def stop_camera(self, idx: int):
        if 0 <= idx < len(self.workers):
            w = self.workers[idx]
            if isinstance(w, CameraWorker):
                w.stop()
                self.workers[idx] = None


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

    def camera_settings(self, idx: int):
        cam = self.cameras[idx]
        dlg = CameraSettingsDialog(self, cam)
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
            self.camera_list.setCurrentRow(idx)
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
        self.workers = [None] * len(self.cameras)
        self.start_all()

    def test_camera(self, idx: int):
        url = self.cameras[idx]["rtsp"]
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
        dlg = VideoPlayerDialog(filepath, self)
        dlg.exec_()

    def open_recordings_browser(self):
        dlg = RecordingsBrowserDialog(self.output_dir, self.cameras, self)
        dlg.open_video.connect(self.open_video_file)
        dlg.exec_()

    def open_alerts_dialog(self):
        dlg = AlertListDialog(self.alert_list, self)
        dlg.exec_()
        self.alert_list.setParent(None)
        self.alert_list.hide()

    def open_camera_list_dialog(self):
        dlg = CameraListDialog(self.camera_list, self)
        dlg.camera_selected.connect(lambda idx: self.camera_list.setCurrentRow(idx))
        dlg.exec_()
        self.camera_list.setParent(None)
        self.camera_list.hide()

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

        btn_add_cam = QPushButton("Dodaj kamerę")
        btn_remove_cam = QPushButton("Usuń kamerę")
        btn_restart = QPushButton("Restart aplikacji")
        btn_close = QPushButton("Zamknij")

        for b in [btn_add_cam, btn_remove_cam, btn_restart, btn_close]:
            layout.addWidget(b)

        btn_add_cam.clicked.connect(parent.add_camera_wizard)
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

