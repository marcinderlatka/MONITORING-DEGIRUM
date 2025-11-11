
# -*- coding: utf-8 -*-
import base64
from collections import deque
from contextlib import suppress
import datetime
import io
import json
import logging
import os
import re
import sys
import uuid
import wave
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
    QObject,
    QSignalBlocker,
    QUrl,
    Qt,
    QTimer,
    QSize,
    QEvent,
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
from .widgets.recordings_browser import RecordingsBrowserDialog

# Qt platform plugin path (Linux)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

# --- Alert z miniaturką (karta) ---
class VideoPlayerDialog(QDialog):
    def __init__(self, filepath, parent=None):
        # QDialog refuses to enter fullscreen mode when it has a parent.
        # Store the reference manually for logging purposes and detach the
        # widget from the hierarchy so that the window manager treats it as a
        # standalone top-level window.
        self._owner = parent
        super().__init__(None)

        # Ensure the dialog behaves like a top-level window so that the
        # window manager allows switching to the fullscreen state.
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )
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
        self._normal_geometry = None
        self._is_fullscreen = False
        self.load_video(self.file_list[self.file_index])

    def showEvent(self, event):
        super().showEvent(event)
        # Zapamiętaj faktyczny rozmiar dopiero po wyrenderowaniu okna,
        # w przeciwnym wypadku geometry() zwraca wartości domyślne i
        # późniejsze przywracanie z pełnego ekranu nie działa poprawnie.
        self._is_fullscreen = self.isFullScreen()
        if self._normal_geometry is None and not self._is_fullscreen:
            self._normal_geometry = self.geometry()

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
            owner = self._owner
            if owner is not None and hasattr(owner, "log_window"):
                owner.log_window.add_entry("application", f"wyeksportowano kadr {os.path.basename(out)}")
        except Exception as e:
            QMessageBox.warning(self, "Błąd", str(e))
            owner = self._owner
            if owner is not None and hasattr(owner, "log_window"):
                owner.log_window.add_entry("error", f"kadr: {e}")

    def toggle_fullscreen(self):
        if self._is_fullscreen:
            self.showNormal()
            if self._normal_geometry is not None:
                self.setGeometry(self._normal_geometry)
            self._is_fullscreen = False
            self.btn_full.setText("Pełny ekran")
        else:
            self._normal_geometry = self.geometry()
            self.showFullScreen()
            self._is_fullscreen = True
            self.btn_full.setText("Zamknij pełny ekran")

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            self._is_fullscreen = self.isFullScreen()
            self.btn_full.setText(
                "Zamknij pełny ekran" if self._is_fullscreen else "Pełny ekran"
            )
        super().changeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self.toggle_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, e):
        self.pause()
        if self.cap:
            self.cap.release()
        super().closeEvent(e)


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
logger = logging.getLogger(__name__)


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
        try:
            idx = int(index)
        except (TypeError, ValueError):
            logger.warning("Ignoring frame with invalid index %r", index)
            return
        is_valid = (
            isinstance(frame, np.ndarray)
            and frame.size > 0
            and frame.ndim >= 2
            and frame.shape[0] > 0
            and frame.shape[1] > 0
        )
        if not is_valid:
            self._last_status[idx] = "Brak sygnału (pusta klatka)"
            self._last_error[idx] = "Brak sygnału (pusta klatka)"
            self._last_frame.pop(idx, None)
            self._last_fps_text[idx] = ""
            if idx == self.camera_list.currentRow():
                self._render_current()
            return

        self.camera_list.update_thumbnail(idx, frame)
        self.camera_grid.update_frame(idx, frame)

        # FPS liczenie dla tej kamery
        from time import perf_counter
        t = perf_counter()
        dq = self._fps_times.setdefault(idx, [])
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
        self._last_frame[idx] = frame
        self._last_fps_text[idx] = fps_txt
        self._last_status[idx] = "Połączono"
        self._last_error.pop(idx, None)

        if idx == self.camera_list.currentRow():
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
        history_items = [dict(item) for item in self.alert_mem.items]
        dlg = RecordingsBrowserDialog(
            camera_dirs,
            self,
            history_path=ALERTS_HISTORY_PATH,
            history_items=history_items,
        )
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

