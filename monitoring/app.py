
# -*- coding: utf-8 -*-
import base64
import csv
from collections import deque
import datetime
import io
import json
import os
import re
import sys
import uuid
import wave
from bisect import bisect_left
from glob import glob
from pathlib import Path

import cv2
import degirum as dg
import degirum_tools
import numpy as np
from PyQt5 import QtSvg
from PyQt5.QtCore import (
    QDate,
    QPoint,
    QRect,
    QRunnable,
    QObject,
    QSignalBlocker,
    QThreadPool,
    QUrl,
    Qt,
    QTimer,
    QSize,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPixmap, QClipboard
from PyQt5.QtMultimedia import QSoundEffect
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .config import (
    ALERTS_HISTORY_PATH,
    CONFIG_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DETECTION_HOURS,
    DEFAULT_DRAW_OVERLAYS,
    DEFAULT_ENABLE_DETECTION,
    DEFAULT_ENABLE_RECORDING,
    DEFAULT_FPS,
    DEFAULT_LOST_SECONDS,
    DEFAULT_MODEL,
    DEFAULT_POST_SECONDS,
    DEFAULT_PRE_SECONDS,
    DEFAULT_RECORD_PATH,
    ICON_DIR,
    LOG_HISTORY_PATH,
    LOG_RETENTION_HOURS,
    MODELS_PATH,
    RECORDINGS_CATALOG_PATH,
    RECORD_CLASSES,
    VISIBLE_CLASSES,
    fill_camera_defaults,
    list_usb_cameras,
    load_config,
    save_config,
)
from .storage import (
    AlertMemory,
    load_recordings_catalog,
    remove_from_recordings_catalog,
    save_recordings_catalog,
    update_recordings_catalog,
)
from .workers import CameraWorker
from .widgets.alerts import AlertDialog, AlertListWidget
from .widgets.camera_grid import CameraGridWidget
from .widgets.camera_list import CameraListWidget
from .widgets.logs import LogWindow

# Qt platform plugin path (Linux)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

# --- Alert z miniaturką (karta) ---
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

    def __init__(self, filepath: str, thumb_path: str = ""):
        QObject.__init__(self)
        QRunnable.__init__(self)
        self._filepath = filepath or ""
        self._thumb_path = thumb_path or ""

    def run(self):
        qimg = self._load_thumbnail()
        if qimg is None:
            qimg = QImage()
        self.thumbnailReady.emit(qimg)

    def _load_thumbnail(self):
        if not self._filepath and not self._thumb_path:
            return None

        thumb_candidates = []
        if self._thumb_path:
            thumb_candidates.append(self._thumb_path)
        if self._filepath:
            thumb_candidates.append(self._filepath + ".jpg")

        for candidate in thumb_candidates:
            if not candidate:
                continue
            try:
                if os.path.exists(candidate):
                    img = cv2.imread(candidate)
                    if img is not None:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb.shape
                        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            except Exception:
                continue
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

    def __init__(self, camera_dirs, history_path=ALERTS_HISTORY_PATH):
        QObject.__init__(self)
        QRunnable.__init__(self)
        self._camera_dirs = list(camera_dirs)
        self._abort = False
        self._history_meta = self._load_history_metadata(history_path)
        self._seen_abs_paths = set()

    def stop(self):
        self._abort = True

    def run(self):
        try:
            self._scan()
        finally:
            self.scanFinished.emit()

    def _scan(self):
        self._seen_abs_paths.clear()
        pattern = re.compile(r"^nagranie_.*\.mp4$", re.IGNORECASE)
        for cam_name, cam_dir in self._camera_dirs:
            if self._abort:
                break
            if not cam_dir or not os.path.isdir(cam_dir):
                continue
            try:
                for root, _dirs, files in os.walk(cam_dir):
                    if self._abort:
                        break
                    for name in files:
                        if self._abort:
                            break
                        if not pattern.match(name):
                            continue
                        path = os.path.join(root, name)
                        if not os.path.isfile(path):
                            continue
                        abs_path = os.path.abspath(path)
                        if abs_path in self._seen_abs_paths:
                            continue
                        self._seen_abs_paths.add(abs_path)
                        meta = self._build_meta(cam_name, path)
                        if self._abort:
                            break
                        self.recordFound.emit(meta)
            except FileNotFoundError:
                continue
        if self._abort:
            return
        self._emit_catalog_entries()

    def _emit_catalog_entries(self):
        for item in load_recordings_catalog():
            if self._abort:
                break
            if not isinstance(item, dict):
                continue
            path = item.get("filepath") or item.get("file")
            if not path:
                continue
            abs_path = os.path.abspath(path)
            if abs_path in self._seen_abs_paths:
                continue
            cam_name = item.get("camera") or self._camera_name_for_path(path)
            meta = self._build_meta(cam_name, path, overrides=item)
            self._seen_abs_paths.add(abs_path)
            self.recordFound.emit(meta)

    def _camera_name_for_path(self, path):
        try:
            abs_path = os.path.abspath(path)
        except Exception:
            return ""
        for name, directory in self._camera_dirs:
            if not directory:
                continue
            try:
                abs_dir = os.path.abspath(directory)
            except Exception:
                continue
            if abs_path == abs_dir:
                return name
            prefix = abs_dir.rstrip(os.sep) + os.sep
            if abs_path.startswith(prefix):
                return name
        return ""

    def _build_meta(self, cam_name, mp4, overrides=None):
        meta_path = mp4 + ".json"
        meta = {
            "camera": cam_name,
            "label": "unknown",
            "confidence": 0.0,
            "time": None,
            "file": mp4,
        }
        hist_meta = self._history_meta.get(os.path.abspath(mp4))
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    m = json.load(f)
                    meta.update(m)
            except Exception:
                pass
        elif hist_meta:
            meta.update(hist_meta)
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
        if hist_meta:
            meta.setdefault("thumb", hist_meta.get("thumb", ""))
            meta.setdefault("camera", hist_meta.get("camera", cam_name))
            if not meta.get("label") or meta.get("label") == "unknown":
                meta["label"] = hist_meta.get("label", meta.get("label"))
            if not meta.get("time"):
                meta["time"] = hist_meta.get("time")
            if not meta.get("confidence"):
                try:
                    meta["confidence"] = float(hist_meta.get("confidence", meta.get("confidence", 0.0)))
                except Exception:
                    pass
        overrides = overrides or {}
        path_override = overrides.get("filepath") or overrides.get("file")
        if path_override:
            meta["file"] = path_override
        camera_override = overrides.get("camera")
        if camera_override:
            meta["camera"] = camera_override
        label_override = overrides.get("label")
        if label_override:
            meta["label"] = label_override
        if "confidence" in overrides and overrides.get("confidence") is not None:
            try:
                meta["confidence"] = float(overrides.get("confidence"))
            except Exception:
                pass
        time_override = overrides.get("time")
        if time_override:
            meta["time"] = time_override
        thumb_override = overrides.get("thumb")
        if thumb_override:
            meta["thumb"] = thumb_override
        if "timestamp" in overrides and overrides.get("timestamp") is not None:
            try:
                meta["timestamp"] = float(overrides.get("timestamp"))
            except Exception:
                pass

        meta.setdefault("camera", cam_name)
        meta["file"] = meta.get("file") or mp4
        meta["filepath"] = meta.get("file")
        meta["label"] = meta.get("label") or "unknown"
        try:
            meta["confidence"] = float(meta.get("confidence", 0.0))
        except Exception:
            meta["confidence"] = 0.0

        timestamp = meta.get("timestamp")
        if timestamp is not None:
            try:
                timestamp = float(timestamp)
            except Exception:
                timestamp = None
        if timestamp is None:
            time_value = meta.get("time")
            if time_value:
                try:
                    timestamp = datetime.datetime.strptime(time_value, "%Y-%m-%d %H:%M:%S").timestamp()
                except Exception:
                    timestamp = None
            if timestamp is None:
                try:
                    file_ts = float(os.path.getmtime(mp4))
                except Exception:
                    file_ts = 0.0
                timestamp = file_ts
                if not meta.get("time"):
                    try:
                        ts = datetime.datetime.fromtimestamp(file_ts)
                        meta["time"] = ts.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        meta["time"] = ""
        meta["timestamp"] = float(timestamp)
        return meta

    def _load_history_metadata(self, history_path):
        if not history_path:
            return {}
        meta = {}
        try:
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    items = json.load(f)
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        fp = item.get("filepath") or item.get("file")
                        if not fp:
                            continue
                        meta[os.path.abspath(fp)] = {
                            "camera": item.get("camera", ""),
                            "label": item.get("label", "unknown"),
                            "confidence": item.get("confidence", 0.0),
                            "time": item.get("time", ""),
                            "thumb": item.get("thumb", ""),
                        }
        except Exception as e:
            print("Nie udało się wczytać historii alertów dla nagrań:", e)
        for item in load_recordings_catalog():
            if not isinstance(item, dict):
                continue
            fp = item.get("filepath") or item.get("file")
            if not fp:
                continue
            key = os.path.abspath(fp)
            existing = meta.get(key, {})
            combined = dict(existing)
            camera = item.get("camera")
            if camera:
                combined["camera"] = camera
            label = item.get("label")
            if label:
                combined["label"] = label
            if "confidence" in item:
                try:
                    combined["confidence"] = float(item.get("confidence", combined.get("confidence", 0.0)))
                except Exception:
                    pass
            time_value = item.get("time")
            if time_value:
                combined["time"] = time_value
            thumb = item.get("thumb")
            if thumb:
                combined["thumb"] = thumb
            if "timestamp" in item:
                try:
                    combined["timestamp"] = float(item.get("timestamp"))
                except Exception:
                    pass
            meta[key] = combined
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

        self._loader = ThumbnailLoader(meta.get("file", ""), meta.get("thumb", ""))
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

    def __init__(self, camera_dirs, parent=None, history_path=ALERTS_HISTORY_PATH):
        super().__init__(parent)
        self.setWindowTitle("Nagrania – przeglądarka")
        self.resize(1100, 700)
        self.camera_dirs = list(camera_dirs)
        self.history_path = history_path

        self.scan_pool = QThreadPool()
        self.thumbnail_pool = QThreadPool()
        self._scan_worker = None
        self._visible_paths = set()

        layout = QVBoxLayout(self)

        # FILTRY
        filters = QHBoxLayout()
        self.camera_filter = QComboBox()
        self.camera_filter.addItem("Wszystkie kamery")
        for name, _ in self.camera_dirs:
            self.camera_filter.addItem(name)

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
        self._all_keys = []
        self._known_paths = set()
        self._closing = False
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
        self._closing = False
        self.all_items.clear()
        self._all_keys.clear()
        self._known_paths.clear()
        self._visible_paths.clear()
        self.list.clear()
        self.refresh_btn.setEnabled(False)
        worker = RecordingsScanWorker(self.camera_dirs, history_path=self.history_path)
        worker.recordFound.connect(self._on_record_found)
        worker.scanFinished.connect(self._on_scan_finished)
        self._scan_worker = worker
        self.scan_pool.start(worker)

    def _on_record_found(self, meta: dict):
        idx = self._insert_sorted_meta(meta)
        if idx is None:
            return
        filters = self._current_filters()
        if not self._record_matches_filters(meta, filters):
            return
        row = self._count_visible_before(idx, filters)
        self._add_list_item(meta, row=row)

    def _on_scan_finished(self):
        self._update_date_filters_from_items()
        if not self._closing:
            self.refresh_btn.setEnabled(True)
        self._scan_worker = None

    def _clear_selection(self):
        self.list.clearSelection()

    def _meta_sort_key(self, meta):
        try:
            ts = float(meta.get("timestamp", 0.0))
        except (TypeError, ValueError):
            ts = 0.0
        return (-ts, meta.get("file", ""))

    def _insert_sorted_meta(self, meta):
        path = meta.get("file")
        if path and path in self._known_paths:
            return None
        key = self._meta_sort_key(meta)
        idx = bisect_left(self._all_keys, key)
        self._all_keys.insert(idx, key)
        self.all_items.insert(idx, meta)
        if path:
            self._known_paths.add(path)
        return idx

    def _count_visible_before(self, idx, filters):
        if idx is None:
            return 0
        count = 0
        for meta in self.all_items[:idx]:
            if self._record_matches_filters(meta, filters):
                count += 1
        return count

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

    def _meta_to_qdate(self, meta):
        dt = None
        try:
            ts = meta.get("timestamp")
            if ts is not None:
                dt = datetime.datetime.fromtimestamp(float(ts))
        except Exception:
            dt = None
        if dt is None:
            try:
                dt = datetime.datetime.strptime(meta.get("time", ""), "%Y-%m-%d %H:%M:%S")
            except Exception:
                return None
        return QDate(dt.year, dt.month, dt.day)

    def meta_in_date_range(self, meta, qfrom: QDate, qto: QDate):
        d = self._meta_to_qdate(meta)
        if d is None:
            return True
        return (d >= qfrom) and (d <= qto)

    def _update_date_filters_from_items(self):
        if not self.all_items:
            return
        dates = [self._meta_to_qdate(meta) for meta in self.all_items]
        dates = [d for d in dates if d is not None]
        if not dates:
            return
        min_date = min(dates)
        max_date = max(dates)
        current_from = self.date_from.date()
        current_to = self.date_to.date()
        if current_from == min_date and current_to == max_date:
            return
        blocker_from = QSignalBlocker(self.date_from)
        blocker_to = QSignalBlocker(self.date_to)
        try:
            self.date_from.setDate(min_date)
            self.date_to.setDate(max_date)
        finally:
            del blocker_from
            del blocker_to
        self.apply_filters()

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

    def _add_list_item(self, meta, preselect=False, row=None):
        path = meta.get("file")
        if not path or path in self._visible_paths:
            return
        widget = RecordingItemWidget(meta, thread_pool=self.thumbnail_pool)
        item = QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        item.setData(Qt.UserRole, path)
        if row is None or row >= self.list.count():
            self.list.addItem(item)
        else:
            self.list.insertItem(row, item)
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
        remove_from_recordings_catalog(paths)
        remaining = set(paths)
        if remaining:
            self.all_items = [m for m in self.all_items if m.get("file") not in remaining]
            self._all_keys = [self._meta_sort_key(m) for m in self.all_items]
            self._known_paths = {m.get("file") for m in self.all_items if m.get("file")}
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
        self._closing = True
        worker = self._scan_worker
        if worker is not None:
            worker.stop()
            with suppress(TypeError):
                worker.recordFound.disconnect(self._on_record_found)
        self._scan_worker = None
        self.scan_pool.clear()
        self.thumbnail_pool.clear()
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
            models = [d.name for d in MODELS_PATH.iterdir() if d.is_dir()]
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
        initial_dir = self.path_edit.text() or str(DEFAULT_RECORD_PATH)
        d = QFileDialog.getExistingDirectory(self, "Wybierz folder nagrań", initial_dir)
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
        self.path_edit.setText(str(cam.get("record_path", DEFAULT_RECORD_PATH)))
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
            "record_path": self.path_edit.text().strip() or str(DEFAULT_RECORD_PATH),
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
        if self.cameras:
            self.output_dir = str(self.cameras[0].get("record_path", DEFAULT_RECORD_PATH))
        else:
            self.output_dir = str(DEFAULT_RECORD_PATH)

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
                zoo_url=str(MODELS_PATH / model_name),
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
            fill_camera_defaults(data)
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
            fill_camera_defaults(data)
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
                    record_base = Path(new_data.get("record_path", DEFAULT_RECORD_PATH))
                    w.output_dir = str(record_base / new_data.get("name"))
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
        is_valid = (
            isinstance(frame, np.ndarray)
            and frame.size > 0
            and frame.ndim >= 2
            and frame.shape[0] > 0
            and frame.shape[1] > 0
        )
        if not is_valid:
            self._last_status[index] = "Brak sygnału (pusta klatka)"
            self._last_error[index] = "Brak sygnału (pusta klatka)"
            self._last_frame.pop(index, None)
            self._last_fps_text[index] = ""
            if index == self.camera_list.currentRow():
                self._render_current()
            return

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
        camera_dirs = []
        for cam in self.cameras:
            name = cam.get("name") or "camera"
            record_root = str(cam.get("record_path") or DEFAULT_RECORD_PATH)
            full_dir = os.path.join(record_root, name)
            camera_dirs.append((name, full_dir))
        dlg = RecordingsBrowserDialog(camera_dirs, self, history_path=ALERTS_HISTORY_PATH)
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

