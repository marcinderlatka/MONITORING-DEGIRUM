
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
import time
import traceback
import threading
import faulthandler
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
    qInstallMessageHandler,
    QtMsgType,
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
    QAction,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from . import config as config_module
from .config import (
    ALERTS_HISTORY_PATH,
    CONFIG_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD_DRAW,
    DEFAULT_CONFIDENCE_THRESHOLD_RECORD,
    DEFAULT_DETECTION_HOURS,
    DEFAULT_DRAW_OVERLAYS,
    DEFAULT_ENABLE_DETECTION,
    DEFAULT_ENABLE_RECORDING,
    DEFAULT_FPS,
    DEFAULT_LOST_SECONDS,
    DEFAULT_MODEL,
    DEFAULT_POST_SECONDS,
    DEFAULT_PREVIEW_FPS_MAIN,
    DEFAULT_PREVIEW_FPS_THUMB,
    DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN,
    DEFAULT_SHOW_CAMERA_INFO_OVERLAY,
    DEFAULT_OVERLOAD_PROTECTION_ENABLED,
    DEFAULT_OVERLOAD_MIN_CAMERA_COUNT,
    DEFAULT_OVERLOAD_CAMERA_COUNT_THRESHOLD,
    DEFAULT_OVERLOAD_REDUCE_THUMB_PREVIEW_FPS,
    DEFAULT_OVERLOAD_REDUCE_DETECT_FPS_FACTOR,
    DEFAULT_OVERLOAD_DISABLE_NONESSENTIAL_OVERLAYS,
    DEFAULT_OVERLOAD_ENTER_DEBOUNCE_SECONDS,
    DEFAULT_OVERLOAD_EXIT_DEBOUNCE_SECONDS,
    DEFAULT_PRE_SECONDS,
    DEFAULT_RECORD_PATH,
    DEFAULT_RECORD_START_MODE,
    DEFAULT_REQUIRED_HITS_TO_START_RECORDING,
    DEFAULT_REQUIRED_MISSES_TO_END_DETECTION,
    DEFAULT_MIN_RECORD_SECONDS,
    DEFAULT_RTSP_FPS,
    DEFAULT_THUMBNAIL_MODE,
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
    flush_storage,
    load_recordings_catalog,
    remove_from_recordings_catalog,
    save_recordings_catalog,
    update_recordings_catalog,
)
from .workers import CameraWorker
from .runtime_helpers import (
    app_log,
    camera_overlay_anchor,
    classify_camera_setting_changes,
    compute_letterboxed_rect,
    evaluate_heartbeat_health,
    evaluate_overload_transition,
    register_app_logger,
)
from .widgets.alerts import AlertDialog, AlertListWidget
from .widgets.camera_grid import CameraGridWidget
from .widgets.camera_list import CameraListWidget
from .widgets.logs import LogSettingsDialog, LogWindow
from .widgets.recordings_browser import RecordingsBrowserDialog

# Qt platform plugin path (Linux)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

CAMERA_RESTART_REQUIRED_FIELDS = {"rtsp", "type", "model"}
CAMERA_RUNTIME_APPLY_FIELDS = {
    "fps", "rtsp_fps", "confidence_threshold", "confidence_threshold_draw", "confidence_threshold_record",
    "draw_overlays", "enable_detection", "enable_recording", "visible_classes", "record_classes",
    "detection_hours", "record_path", "pre_seconds", "lost_seconds", "post_seconds",
    "required_hits_to_start_recording", "required_misses_to_end_detection", "min_record_seconds",
    "thumbnail_mode", "record_start_mode", "preview_fps_main", "preview_fps_thumb", "preview_pause_when_hidden", "show_camera_info_overlay",
}

CAMERA_SETTING_TOOLTIPS = {
    "name": (
        "Nazwa kamery widoczna w aplikacji, alertach, nagraniach i przeglądarce nagrań.\n\n"
        "To pole nie zmienia obrazu ani detekcji, ale ułatwia rozpoznanie źródła zdarzenia.\n"
        "Warto używać krótkich i jednoznacznych nazw, np. 'Brama', 'Podjazd', 'Magazyn'."
    ),
    "type": (
        "Typ źródła obrazu dla tej kamery: RTSP (kamera sieciowa) albo USB (kamera lokalna).\n\n"
        "To ustawienie decyduje, jak aplikacja otwiera strumień i jakiej ścieżki inicjalizacji używa.\n"
        "Zmiana typu jest krytyczna dla strumienia i zwykle wymaga automatycznego restartu tej kamery."
    ),
    "url": (
        "Adres źródła obrazu, najczęściej strumienia RTSP.\n\n"
        "To najważniejsze ustawienie połączenia z kamerą. Jeśli adres jest błędny, kamera nie połączy się "
        "albo obraz nie będzie dostępny.\n\n"
        "Zmiana tego pola zwykle wymaga przeładowania lub restartu tylko tej kamery, ponieważ zmienia sposób "
        "inicjalizacji strumienia."
    ),
    "rtsp": (
        "Adres źródła obrazu, najczęściej strumienia RTSP.\n\n"
        "To najważniejsze ustawienie połączenia z kamerą. Jeśli adres jest błędny, kamera nie połączy się "
        "albo obraz nie będzie dostępny.\n\n"
        "Zmiana tego pola zwykle wymaga przeładowania lub restartu tylko tej kamery, ponieważ zmienia sposób "
        "inicjalizacji strumienia."
    ),
    "model_name": (
        "Model AI używany do detekcji obiektów dla tej kamery.\n\n"
        "Różne modele mogą różnić się szybkością działania, dokładnością oraz listą obsługiwanych klas.\n"
        "Lżejszy model zwykle działa szybciej, ale może gorzej wykrywać trudniejsze obiekty.\n"
        "Zmiana modelu zwykle wymaga przeładowania kamery."
    ),
    "model": (
        "Model AI używany do detekcji obiektów dla tej kamery.\n\n"
        "Różne modele mogą różnić się szybkością działania, dokładnością oraz listą obsługiwanych klas.\n"
        "Lżejszy model zwykle działa szybciej, ale może gorzej wykrywać trudniejsze obiekty.\n"
        "Zmiana modelu zwykle wymaga przeładowania kamery."
    ),
    "fps": (
        "Docelowa częstotliwość wykonywania detekcji AI.\n\n"
        "Wyższa wartość oznacza częstsze sprawdzanie klatek, większą szansę wykrycia krótkiego zdarzenia, "
        "ale też większe obciążenie CPU/GPU.\n"
        "Niższa wartość zmniejsza obciążenie, ale może powodować pomijanie krótkich pojawień się obiektu.\n\n"
        "To ustawienie wpływa głównie na detekcję, a nie bezpośrednio na sam podgląd."
    ),
    "rtsp_fps": (
        "Ograniczenie liczby klatek pobieranych ze strumienia RTSP do przetwarzania.\n\n"
        "Pozwala zmniejszyć obciążenie systemu przy wielu kamerach. Zbyt niska wartość może jednak pogorszyć "
        "płynność ruchu i obniżyć skuteczność wykrywania szybkich zdarzeń.\n\n"
        "W praktyce: wyższe rtsp_fps = lepsza płynność i więcej danych, niższe rtsp_fps = mniejsze obciążenie."
    ),
    "confidence_threshold": (
        "Starszy, zgodnościowy próg pewności detekcji.\n\n"
        "Jeżeli aplikacja używa osobnych progów rysowania i nagrywania, to to pole może pełnić rolę ustawienia "
        "zgodności ze starszym configiem.\n\n"
        "Im wyższy próg, tym mniej słabych wykryć. Im niższy próg, tym większa czułość, ale także większe ryzyko "
        "fałszywych trafień."
    ),
    "confidence_threshold_draw": (
        "Minimalna pewność detekcji potrzebna do narysowania ramki i etykiety na podglądzie.\n\n"
        "Niższa wartość pokaże więcej wykryć, także słabszych. Wyższa wartość ograniczy liczbę ramek do bardziej "
        "pewnych trafień.\n\n"
        "To ustawienie wpływa na to, co widzisz na ekranie, ale nie musi samo w sobie uruchamiać nagrywania."
    ),
    "confidence_threshold_record": (
        "Minimalna pewność detekcji wymagana do uruchomienia logiki zdarzenia i nagrania.\n\n"
        "Zwiększenie tej wartości zmniejsza liczbę fałszywych alarmów, ale może powodować pomijanie trudniejszych "
        "wykryć.\n"
        "Zmniejszenie zwiększa czułość systemu, ale może prowadzić do częstszych niepotrzebnych nagrań."
    ),
    "draw_overlays": (
        "Włącza rysowanie ramek, etykiet i opisów detekcji na podglądzie na żywo.\n\n"
        "Wyłączenie tej opcji może trochę zmniejszyć obciążenie systemu, szczególnie przy wielu kamerach.\n"
        "Detekcja może nadal działać nawet wtedy, gdy ramki nie są rysowane.\n\n"
        "Uwaga: miniatura nagrania może nadal zawierać zaznaczony obiekt, aby łatwiej było rozpoznać zdarzenie."
    ),
    "show_camera_info_overlay": (
        "Pokazuje na obrazie kamery scalone informacje o statusie połączenia, FPS, stanie nagrywania i diagnostyce.\n\n"
        "Wyłączenie ukrywa okno informacyjne, ale nie wyłącza detekcji ani nagrywania."
    ),
    "enable_detection": (
        "Włącza analizę AI dla tej kamery.\n\n"
        "Gdy opcja jest wyłączona, obraz z kamery może nadal być wyświetlany, ale aplikacja nie będzie "
        "wykrywać obiektów, tworzyć zdarzeń ani reagować logiką detekcji."
    ),
    "enable_recording": (
        "Pozwala zapisywać nagrania zdarzeń dla tej kamery.\n\n"
        "Po wyłączeniu tej opcji detekcja może nadal działać i obiekty mogą być widoczne na podglądzie, "
        "ale aplikacja nie będzie tworzyć plików MP4 dla wykrytych zdarzeń."
    ),
    "visible_classes": (
        "Lista klas obiektów, które mogą być pokazywane na podglądzie kamery.\n\n"
        "Jeśli obiekt nie znajduje się na tej liście, może nie być rysowany na ekranie nawet wtedy, gdy model go wykryje.\n"
        "To ustawienie dotyczy warstwy wizualnej i pomaga ograniczyć bałagan na obrazie."
    ),
    "record_classes": (
        "Lista klas obiektów, które mogą uruchamiać nagrywanie zdarzenia.\n\n"
        "Pozwala zawęzić nagrania tylko do ważnych obiektów, np. 'person' albo 'car'.\n"
        "Dzięki temu można ograniczyć liczbę niepotrzebnych plików i lepiej kontrolować logikę alarmów."
    ),
    "detection_hours": (
        "Zakres godzin, w których detekcja ma być aktywna.\n\n"
        "Przydaje się, gdy kamera ma reagować tylko o określonych porach, np. nocą lub poza godzinami pracy.\n"
        "Poza wskazanym zakresem system może nie wykonywać detekcji albo ignorować zdarzenia — zależnie od implementacji."
    ),
    "record_path": (
        "Folder zapisu nagrań, miniaturek JPG i metadanych JSON dla tej kamery.\n\n"
        "Jeśli ścieżka jest błędna, niedostępna albo bez uprawnień zapisu, nagrania mogą się nie tworzyć.\n"
        "Warto używać stabilnej lokalizacji z odpowiednią ilością miejsca na dysku."
    ),
    "pre_seconds": (
        "Liczba sekund materiału zachowywana przed wykryciem obiektu.\n\n"
        "Większa wartość daje lepszy kontekst zdarzenia, bo nagranie może pokazać, co działo się chwilę wcześniej.\n"
        "Zwiększa jednak użycie pamięci, ponieważ aplikacja musi buforować więcej klatek."
    ),
    "lost_seconds": (
        "Czas oczekiwania po zniknięciu obiektu, zanim system uzna, że zdarzenie rzeczywiście się skończyło.\n\n"
        "Pomaga uniknąć nerwowego przerywania nagrania, gdy obiekt znika tylko na chwilę albo detekcja chwilowo go nie widzi."
    ),
    "post_seconds": (
        "Dodatkowy czas nagrywania po ostatnim wykryciu obiektu.\n\n"
        "Dzięki temu końcówka zdarzenia nie zostaje ucięta zbyt wcześnie i nagranie jest bardziej naturalne."
    ),
    "required_hits_to_start_recording": (
        "Liczba kolejnych lub potwierdzonych trafień wymagana do uruchomienia nagrania.\n\n"
        "Wartość 1 daje najszybszą reakcję.\n"
        "Wyższa wartość poprawia stabilność i ogranicza fałszywe uruchomienia, ale może minimalnie opóźnić start nagrania."
    ),
    "required_misses_to_end_detection": (
        "Liczba kolejnych braków detekcji pomagająca potwierdzić zakończenie zdarzenia.\n\n"
        "Wyższa wartość sprawia, że system mniej nerwowo kończy zdarzenie przy chwilowym zaniku obiektu."
    ),
    "min_record_seconds": (
        "Minimalna długość pojedynczego nagrania po jego uruchomieniu.\n\n"
        "Zapobiega tworzeniu bardzo krótkich, mało użytecznych klipów przy pojedynczym krótkim wykryciu."
    ),
    "thumbnail_mode": (
        "Sposób wyboru miniatury nagrania widocznej w przeglądarce.\n\n"
        "first_detection — miniatura z pierwszego potwierdzonego wykrycia,\n"
        "best_detection — miniatura z najlepiej ocenionego wykrycia,\n"
        "first_frame — miniatura z pierwszej klatki zdarzenia.\n\n"
        "Najczęściej najlepszy efekt daje first_detection albo best_detection."
    ),
    "record_start_mode": (
        "Sposób organizacji początku nagrania.\n\n"
        "detection_first — nagranie zaczyna się od momentu wykrycia,\n"
        "include_prerecord_first — nagranie może zawierać także materiał sprzed wykrycia.\n\n"
        "To ustawienie wpływa na to, co użytkownik zobaczy na początku klipu i jak interpretowany jest start zdarzenia."
    ),
    "preview_fps_main": (
        "Maksymalna częstotliwość odświeżania głównego podglądu wybranej kamery.\n\n"
        "Wyższa wartość daje płynniejszy obraz, ale zwiększa obciążenie GUI.\n"
        "Niższa wartość zmniejsza zużycie CPU, ale może pogorszyć płynność."
    ),
    "preview_fps_thumb": (
        "Maksymalna częstotliwość odświeżania miniaturek lub widoków pobocznych kamer.\n\n"
        "To ważne ustawienie przy wielu kamerach, bo ogranicza obciążenie interfejsu bez wyłączania samej detekcji."
    ),
    "preview_pause_when_hidden": (
        "Ogranicza lub zatrzymuje aktualizację podglądu, gdy kamera nie jest widoczna w aktywnym widoku.\n\n"
        "Zmniejsza obciążenie GUI, zachowując działanie logiki detekcji i nagrywania, jeśli system wspiera taki tryb."
    ),
}

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
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_frame is not None:
            # Przerysuj bieżącą klatkę, aby dopasować ją do nowego rozmiaru.
            self._show_frame(self.current_frame)

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

        screen = QApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            self.resize(int(geom.width() * 0.9), int(geom.height() * 0.86))
        else:
            self.resize(1400, 900)

        self.name_edit = QLineEdit()
        self.type_combo = QComboBox(); self.type_combo.addItems(["rtsp", "usb"])
        self.rtsp_edit = QLineEdit()
        self.device_combo = QComboBox()
        for idx, name in list_usb_cameras():
            self.device_combo.addItem(f"{name} ({idx})", idx)
        self.source_stack = QStackedWidget(); self.source_stack.addWidget(self.rtsp_edit); self.source_stack.addWidget(self.device_combo)
        self.type_combo.currentTextChanged.connect(self._on_type_change)

        self.model_combo = QComboBox()
        try:
            models = [d.name for d in MODELS_PATH.iterdir() if d.is_dir()]
        except Exception:
            models = []
        if not models:
            models = [camera.get("model", DEFAULT_MODEL) if camera else DEFAULT_MODEL]
        self.model_combo.addItems(models)

        self.fps_spin = QSpinBox(); self.fps_spin.setRange(1, 60)
        self.rtsp_fps_spin = QSpinBox(); self.rtsp_fps_spin.setRange(0, 60); self.rtsp_fps_spin.setSpecialValueText("Auto")
        self.conf_spin = QDoubleSpinBox(); self.conf_spin.setRange(0.0, 1.0); self.conf_spin.setSingleStep(0.05)
        self.conf_draw_spin = QDoubleSpinBox(); self.conf_draw_spin.setRange(0.0, 1.0); self.conf_draw_spin.setSingleStep(0.05)
        self.conf_record_spin = QDoubleSpinBox(); self.conf_record_spin.setRange(0.0, 1.0); self.conf_record_spin.setSingleStep(0.05)
        self.draw_chk = QCheckBox(); self.detect_chk = QCheckBox(); self.record_chk = QCheckBox()
        self.info_overlay_chk = QCheckBox()
        self.hours_edit = QLineEdit(); self.visible_edit = QLineEdit(); self.record_edit = QLineEdit()
        self.path_edit = QLineEdit(); self.btn_path = QPushButton("Wybierz")
        self.pre_spin = QSpinBox(); self.pre_spin.setRange(0, 60)
        self.lost_spin = QSpinBox(); self.lost_spin.setRange(0, 60)
        self.post_spin = QSpinBox(); self.post_spin.setRange(0, 60)
        self.thumbnail_mode_combo = QComboBox(); self.thumbnail_mode_combo.addItems(["first_detection", "best_detection", "first_frame"])
        self.record_start_mode_combo = QComboBox(); self.record_start_mode_combo.addItems(["detection_first", "include_prerecord_first"])
        self.required_hits_spin = QSpinBox(); self.required_hits_spin.setRange(1, 10)
        self.required_misses_spin = QSpinBox(); self.required_misses_spin.setRange(1, 10)
        self.min_record_seconds_spin = QSpinBox(); self.min_record_seconds_spin.setRange(0, 120)
        self.preview_fps_main_spin = QDoubleSpinBox(); self.preview_fps_main_spin.setRange(1.0, 60.0); self.preview_fps_main_spin.setSingleStep(1.0)
        self.preview_fps_thumb_spin = QDoubleSpinBox(); self.preview_fps_thumb_spin.setRange(0.5, 30.0); self.preview_fps_thumb_spin.setSingleStep(0.5)
        self.preview_pause_chk = QCheckBox()

        path_layout = QHBoxLayout(); path_layout.addWidget(self.path_edit); path_layout.addWidget(self.btn_path)

        self._field_rows = {}
        self._focus_help_widgets = {}

        root = QVBoxLayout(self)
        cols = QHBoxLayout()
        root.addLayout(cols, stretch=1)

        left_box = QFrame(); left_layout = QFormLayout(left_box)
        middle_box = QFrame(); middle_layout = QFormLayout(middle_box)
        right_box = QFrame(); right_layout = QFormLayout(right_box)
        cols.addWidget(left_box, 1); cols.addWidget(middle_box, 1); cols.addWidget(right_box, 1)

        self._add_field_row(left_layout, "name", "Nazwa", self.name_edit)
        self._add_field_row(left_layout, "type", "Typ źródła", self.type_combo)
        self._add_field_row(left_layout, "rtsp", "Adres/Urządzenie", self.source_stack, input_widget=self.rtsp_edit, focus_widgets=[self.rtsp_edit, self.device_combo, self.source_stack])
        self._add_field_row(left_layout, "model", "Model detekcji", self.model_combo)
        self._add_field_row(left_layout, "fps", "FPS/S DETECT", self.fps_spin)
        self._add_field_row(left_layout, "rtsp_fps", "FPS/S RTSP", self.rtsp_fps_spin)
        self._add_field_row(left_layout, "show_camera_info_overlay", "Pokaż okno info na obrazie", self.info_overlay_chk)

        self._add_field_row(middle_layout, "confidence_threshold", "Próg pewności (legacy)", self.conf_spin)
        self._add_field_row(middle_layout, "confidence_threshold_draw", "Próg rysowania", self.conf_draw_spin)
        self._add_field_row(middle_layout, "confidence_threshold_record", "Próg nagrania", self.conf_record_spin)
        self._add_field_row(middle_layout, "draw_overlays", "Rysuj nakładki", self.draw_chk)
        self._add_field_row(middle_layout, "enable_detection", "Wykrywaj obiekty", self.detect_chk)
        self._add_field_row(middle_layout, "enable_recording", "Nagrywaj detekcje", self.record_chk)
        self._add_field_row(middle_layout, "detection_hours", "Godziny detekcji", self.hours_edit)
        self._add_field_row(middle_layout, "visible_classes", "Widoczne klasy", self.visible_edit)
        self._add_field_row(middle_layout, "record_classes", "Klasy nagrywane", self.record_edit)

        self._add_field_row(right_layout, "record_path", "Folder nagrań", path_layout, input_widget=self.path_edit, focus_widgets=[self.path_edit, self.btn_path])
        self._add_field_row(right_layout, "pre_seconds", "Pre seconds", self.pre_spin)
        self._add_field_row(right_layout, "lost_seconds", "Lost seconds", self.lost_spin)
        self._add_field_row(right_layout, "post_seconds", "Post seconds", self.post_spin)
        self._add_field_row(right_layout, "thumbnail_mode", "Tryb miniatury", self.thumbnail_mode_combo)
        self._add_field_row(right_layout, "record_start_mode", "Tryb startu nagrania", self.record_start_mode_combo)
        self._add_field_row(right_layout, "required_hits_to_start_recording", "Wymagane trafienia start", self.required_hits_spin)
        self._add_field_row(right_layout, "required_misses_to_end_detection", "Wymagane pudła stop", self.required_misses_spin)
        self._add_field_row(right_layout, "min_record_seconds", "Minimalny czas nagrania", self.min_record_seconds_spin)
        self._add_field_row(right_layout, "preview_fps_main", "Preview FPS main", self.preview_fps_main_spin)
        self._add_field_row(right_layout, "preview_fps_thumb", "Preview FPS thumb", self.preview_fps_thumb_spin)
        self._add_field_row(right_layout, "preview_pause_when_hidden", "Pauzuj preview gdy ukryta", self.preview_pause_chk)

        self._apply_all_tooltips()
        self._apply_field_tooltip("record_path", None, self.btn_path)
        self._apply_field_tooltip("rtsp", None, self.device_combo)
        self._apply_field_tooltip("rtsp", None, self.source_stack)

        self.help_panel = QTextEdit()
        self.help_panel.setReadOnly(True)
        self.help_panel.setMinimumHeight(130)
        self.help_panel.setPlaceholderText("Kliknij lub zaznacz pole ustawień, aby zobaczyć szczegółowy opis.")
        root.addWidget(self.help_panel)
        self._set_help_panel_text("name")

        controls = QHBoxLayout()
        self.test_btn = QPushButton("Test połączenia")
        self.test_status = QLabel("")
        self.btn_ok = QPushButton("Zapisz")
        self.btn_cancel = QPushButton("Anuluj")
        controls.addWidget(self.test_btn)
        controls.addWidget(self.test_status)
        controls.addStretch(1)
        controls.addWidget(self.btn_cancel)
        controls.addWidget(self.btn_ok)
        root.addLayout(controls)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_path.clicked.connect(self._choose_path)
        self.test_btn.clicked.connect(self._test_source)

        self.result_camera = None
        if camera:
            self.load_camera(camera)
        else:
            self._on_type_change(self.type_combo.currentText())
            self.preview_fps_main_spin.setValue(float(DEFAULT_PREVIEW_FPS_MAIN))
            self.preview_fps_thumb_spin.setValue(float(DEFAULT_PREVIEW_FPS_THUMB))
            self.preview_pause_chk.setChecked(bool(DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN))
            self.info_overlay_chk.setChecked(bool(DEFAULT_SHOW_CAMERA_INFO_OVERLAY))

    def _add_field_row(self, form: QFormLayout, field_key: str, label_text: str, widget, input_widget=None, focus_widgets=None):
        label = QLabel(label_text)
        form.addRow(label, widget)
        field_input = input_widget or widget
        self._field_rows[field_key] = (label, field_input)
        self._register_focus_help(field_key, label)
        widgets_for_focus = focus_widgets or [field_input]
        for focus_widget in widgets_for_focus:
            self._register_focus_help(field_key, focus_widget)

    def _apply_field_tooltip(self, key: str, label_widget, input_widget):
        text = CAMERA_SETTING_TOOLTIPS.get(key, "")
        if text:
            if label_widget is not None:
                label_widget.setToolTip(text)
                label_widget.setWhatsThis(text)
            if input_widget is not None:
                input_widget.setToolTip(text)
                input_widget.setWhatsThis(text)

    def _apply_all_tooltips(self):
        for key, (label, input_widget) in self._field_rows.items():
            self._apply_field_tooltip(key, label, input_widget)

    def _register_focus_help(self, key: str, widget):
        if widget is None:
            return
        self._focus_help_widgets[widget] = key
        widget.installEventFilter(self)

    def _set_help_panel_text(self, key: str):
        if not hasattr(self, "help_panel"):
            return
        self.help_panel.setPlainText(CAMERA_SETTING_TOOLTIPS.get(key, ""))

    def eventFilter(self, watched, event):
        if event.type() == QEvent.FocusIn:
            key = self._focus_help_widgets.get(watched)
            if key:
                self._set_help_panel_text(key)
        return super().eventFilter(watched, event)

    def _choose_path(self):
        path = QFileDialog.getExistingDirectory(self, "Wybierz folder nagrań", self.path_edit.text() or str(DEFAULT_RECORD_PATH))
        if path:
            self.path_edit.setText(path)

    def _on_type_change(self, t):
        self.source_stack.setCurrentWidget(self.device_combo if t == "usb" else self.rtsp_edit)

    def _test_source(self):
        self.test_status.setText("Testuję…")
        self.test_status.setStyleSheet("color:#ccc;")
        if self.type_combo.currentText() == "usb":
            cap = cv2.VideoCapture(int(self.device_combo.currentData()))
        else:
            url = self.rtsp_edit.text().strip()
            cap = cv2.VideoCapture(url)
        ok, _ = cap.read(); cap.release()
        if ok:
            self.test_status.setText("✅ OK"); self.test_status.setStyleSheet("color:#0f0;")
        else:
            self.test_status.setText("⚠️ Błąd"); self.test_status.setStyleSheet("color:#f80;")

    def load_camera(self, cam):
        cam = cam or {}
        self.name_edit.setText(cam.get("name", ""))
        src_type = cam.get("type", "rtsp")
        self.type_combo.setCurrentText(src_type)
        if src_type == "usb":
            idx = int(cam.get("rtsp", 0))
            i = self.device_combo.findData(idx)
            if i >= 0:
                self.device_combo.setCurrentIndex(i)
            self.source_stack.setCurrentWidget(self.device_combo)
        else:
            self.rtsp_edit.setText(str(cam.get("rtsp", "")))
            self.source_stack.setCurrentWidget(self.rtsp_edit)
        self.model_combo.setCurrentText(cam.get("model", DEFAULT_MODEL))
        self.fps_spin.setValue(int(cam.get("fps", DEFAULT_FPS)))
        self.rtsp_fps_spin.setValue(int(cam.get("rtsp_fps", DEFAULT_RTSP_FPS)))
        legacy_conf = float(cam.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
        self.conf_spin.setValue(legacy_conf)
        self.conf_draw_spin.setValue(float(cam.get("confidence_threshold_draw", legacy_conf)))
        self.conf_record_spin.setValue(float(cam.get("confidence_threshold_record", legacy_conf)))
        self.draw_chk.setChecked(bool(cam.get("draw_overlays", DEFAULT_DRAW_OVERLAYS)))
        self.detect_chk.setChecked(bool(cam.get("enable_detection", DEFAULT_ENABLE_DETECTION)))
        self.record_chk.setChecked(bool(cam.get("enable_recording", DEFAULT_ENABLE_RECORDING)))
        self.info_overlay_chk.setChecked(bool(cam.get("show_camera_info_overlay", DEFAULT_SHOW_CAMERA_INFO_OVERLAY)))
        self.hours_edit.setText(cam.get("detection_hours", DEFAULT_DETECTION_HOURS))
        self.visible_edit.setText(",".join(cam.get("visible_classes", VISIBLE_CLASSES)))
        self.record_edit.setText(",".join(cam.get("record_classes", RECORD_CLASSES)))
        self.path_edit.setText(str(cam.get("record_path", DEFAULT_RECORD_PATH)))
        self.pre_spin.setValue(int(cam.get("pre_seconds", DEFAULT_PRE_SECONDS)))
        self.lost_spin.setValue(int(cam.get("lost_seconds", DEFAULT_LOST_SECONDS)))
        self.post_spin.setValue(int(cam.get("post_seconds", DEFAULT_POST_SECONDS)))
        self.thumbnail_mode_combo.setCurrentText(str(cam.get("thumbnail_mode", DEFAULT_THUMBNAIL_MODE)))
        self.record_start_mode_combo.setCurrentText(str(cam.get("record_start_mode", DEFAULT_RECORD_START_MODE)))
        self.required_hits_spin.setValue(int(cam.get("required_hits_to_start_recording", DEFAULT_REQUIRED_HITS_TO_START_RECORDING)))
        self.required_misses_spin.setValue(int(cam.get("required_misses_to_end_detection", DEFAULT_REQUIRED_MISSES_TO_END_DETECTION)))
        self.min_record_seconds_spin.setValue(int(cam.get("min_record_seconds", DEFAULT_MIN_RECORD_SECONDS)))
        self.preview_fps_main_spin.setValue(float(cam.get("preview_fps_main", DEFAULT_PREVIEW_FPS_MAIN)))
        self.preview_fps_thumb_spin.setValue(float(cam.get("preview_fps_thumb", DEFAULT_PREVIEW_FPS_THUMB)))
        self.preview_pause_chk.setChecked(bool(cam.get("preview_pause_when_hidden", DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN)))

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
            "confidence_threshold_draw": float(self.conf_draw_spin.value()),
            "confidence_threshold_record": float(self.conf_record_spin.value()),
            "draw_overlays": self.draw_chk.isChecked(),
            "enable_detection": self.detect_chk.isChecked(),
            "enable_recording": self.record_chk.isChecked(),
            "show_camera_info_overlay": self.info_overlay_chk.isChecked(),
            "rtsp_fps": int(self.rtsp_fps_spin.value()),
            "detection_hours": self.hours_edit.text().strip() or DEFAULT_DETECTION_HOURS,
            "visible_classes": [c.strip() for c in self.visible_edit.text().split(",") if c.strip()],
            "record_classes": [c.strip() for c in self.record_edit.text().split(",") if c.strip()],
            "record_path": self.path_edit.text().strip() or str(DEFAULT_RECORD_PATH),
            "pre_seconds": int(self.pre_spin.value()),
            "lost_seconds": int(self.lost_spin.value()),
            "post_seconds": int(self.post_spin.value()),
            "thumbnail_mode": self.thumbnail_mode_combo.currentText(),
            "record_start_mode": self.record_start_mode_combo.currentText(),
            "required_hits_to_start_recording": int(self.required_hits_spin.value()),
            "required_misses_to_end_detection": int(self.required_misses_spin.value()),
            "min_record_seconds": int(self.min_record_seconds_spin.value()),
            "preview_fps_main": float(self.preview_fps_main_spin.value()),
            "preview_fps_thumb": float(self.preview_fps_thumb_spin.value()),
            "preview_pause_when_hidden": self.preview_pause_chk.isChecked(),
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


class AppLogBridge(QObject):
    entry_signal = pyqtSignal(object)

    def __init__(self, target_window=None) -> None:
        super().__init__()
        self._target_window = target_window
        self.entry_signal.connect(self._deliver, Qt.QueuedConnection)

    def set_target(self, target_window) -> None:
        self._target_window = target_window

    def log(self, group: str, message: str, camera: str = "", source: str = "", level: str = "INFO", details: str = "", traceback_text: str = "", action: str = "") -> None:
        payload = {
            "group": group,
            "camera": camera,
            "source": source,
            "level": level,
            "action": action or message,
            "details": details,
            "traceback": traceback_text,
        }
        self.entry_signal.emit(payload)

    def info(self, group: str, message: str, **kwargs) -> None:
        self.log(group=group, message=message, level="INFO", **kwargs)

    def warning(self, group: str, message: str, **kwargs) -> None:
        self.log(group=group, message=message, level="WARNING", **kwargs)

    def error(self, group: str, message: str, **kwargs) -> None:
        self.log(group=group, message=message, level="ERROR", **kwargs)

    def exception(self, group: str, message: str, exc: BaseException | None = None, **kwargs) -> None:
        tb_text = kwargs.pop("traceback_text", "")
        if not tb_text:
            if exc is not None:
                tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            else:
                tb_text = traceback.format_exc()
        self.log(group=group, message=message, level="CRITICAL", traceback_text=tb_text, details=tb_text, **kwargs)

    def _deliver(self, payload: object) -> None:
        if self._target_window is None or not isinstance(payload, dict):
            return
        try:
            self._target_window.log_window.add_structured_entry(payload)
        except Exception:
            print("AppLogBridge delivery failed", file=sys.stderr)


class UILoggingHandler(logging.Handler):
    def __init__(self, bridge: AppLogBridge) -> None:
        super().__init__(level=logging.INFO)
        self.bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("PyQt5"):
            return
        level = record.levelname.upper()
        group = "application"
        if level in {"ERROR", "CRITICAL"}:
            group = "error"
        elif level == "WARNING":
            group = "warning"
        tb_text = ""
        if record.exc_info:
            tb_text = "".join(traceback.format_exception(*record.exc_info))
        self.bridge.log(
            group=group,
            message=record.getMessage(),
            source=record.name,
            level=level,
            details=tb_text,
            traceback_text=tb_text,
        )


APP_LOG_BRIDGE = AppLogBridge()
_QT_HANDLER_INSTALLED = False


def install_global_exception_hooks() -> None:
    def _sys_hook(exc_type, exc, tb):
        tb_text = "".join(traceback.format_exception(exc_type, exc, tb))
        logger.critical("Unhandled exception", exc_info=(exc_type, exc, tb))
        APP_LOG_BRIDGE.log(group="error", message=str(exc), source="global-exception", level="CRITICAL", details=tb_text, traceback_text=tb_text, action="Unhandled application exception")
        sys.__excepthook__(exc_type, exc, tb)

    def _thread_hook(args):
        tb_text = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        logger.critical("Unhandled thread exception", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        APP_LOG_BRIDGE.log(group="error", message=str(args.exc_value), source="thread-exception", level="CRITICAL", details=tb_text, traceback_text=tb_text, action=f"Unhandled exception in thread {args.thread.name}")

    sys.excepthook = _sys_hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_hook


def install_qt_message_handler() -> None:
    global _QT_HANDLER_INSTALLED
    if _QT_HANDLER_INSTALLED:
        return

    def _handler(mode, context, message):
        msg = str(message)
        source = "qt"
        if mode == QtMsgType.QtWarningMsg:
            APP_LOG_BRIDGE.warning("warning", msg, source=source)
        elif mode in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
            APP_LOG_BRIDGE.log("error", msg, source=source, level="CRITICAL" if mode == QtMsgType.QtFatalMsg else "ERROR")
        else:
            APP_LOG_BRIDGE.info("ui", msg, source=source)

    qInstallMessageHandler(_handler)
    _QT_HANDLER_INSTALLED = True


class MainWindow(QMainWindow):
    def __init__(self, cameras):
        super().__init__()
        self.setWindowTitle("AI Monitoring – PyQt5 (pełne GUI)")
        self.resize(1400, 900)

        # Pamięć alertów
        self.alert_mem = AlertMemory(ALERTS_HISTORY_PATH, max_items=5000)
        self.last_detected_label = ""
        self.sound_enabled = True
        self.sound_volume = 1.0
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
            self.alert_sound.setVolume(self.sound_volume)
        except Exception as e:
            self._log_exception("error", f"Failed to initialize alert sound: {e}", exc=e, source="audio")

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
        APP_LOG_BRIDGE.set_target(self)
        register_app_logger(APP_LOG_BRIDGE.log)
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
            self._log_warning("ui", "nie udało się załadować ikony folder.svg", source="ui")
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

        btn_logs = QToolButton()
        btn_logs.setIcon(QIcon(str(ICON_DIR / "terminal.svg")))
        btn_logs.setIconSize(QSize(50, 50))
        btn_logs.clicked.connect(self.open_log_settings_dialog)

        self.btn_sound = QToolButton()
        self.btn_sound.setIcon(QIcon(str(ICON_DIR / "volume-up.svg")))
        self.btn_sound.setIconSize(QSize(50, 50))
        self._setup_sound_menu()

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

        for btn in (
            btn_cameras,
            btn_recordings,
            btn_settings,
            btn_cam_ctrl,
            btn_alerts,
            btn_logs,
            self.btn_sound,
            btn_fullscreen,
        ):
            btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
            btn.setAutoRaise(True)
            btn.setStyleSheet(btn_style)

        controls_layout.addStretch()
        controls_layout.addWidget(btn_cameras)
        controls_layout.addWidget(btn_recordings)
        controls_layout.addWidget(btn_settings)
        controls_layout.addWidget(btn_cam_ctrl)
        controls_layout.addWidget(btn_alerts)
        controls_layout.addWidget(btn_logs)
        controls_layout.addWidget(self.btn_sound)
        controls_layout.addWidget(btn_fullscreen)
        controls_layout.addStretch()

        center_v.addWidget(controls_widget)

        main_hlayout.addWidget(self.center_panel, stretch=1)

        self.alert_list = AlertListWidget(self.alert_mem)
        main_hlayout.addWidget(self.alert_list)

        main_vlayout.addLayout(main_hlayout)

        self.setCentralWidget(main_widget)

        self._heartbeat_last_seen: dict[str, float] = {}
        self._heartbeat_alerted: set[str] = set()
        self._watchdog_flags: set[str] = set()
        self._last_frame_update_ts: dict[int, float] = {}
        self._log_info("application", "aplikacja uruchomiona", source="startup")

        # Track windowed geometry so fullscreen toggle can restore it reliably
        self._is_fullscreen = False
        self._normal_geometry = None

        # backend
        self.workers = []
        self.model_cache: dict[str, object] = {}
        self.camera_list.currentRowChanged.connect(self.switch_camera)

        self.alert_list.open_video.connect(self.open_video_file)

        # FPS liczniki i HUD stan
        self._fps_times = {}
        self._last_frame = {}
        self._last_status = {}
        self._last_error = {}
        self._last_fps_text = {}
        self._worker_diag: dict[str, dict[str, object]] = {}
        self.worker_status: dict[str, dict[str, object]] = {}
        self.last_render_time = 0.0
        self._render_interval_s = 1 / 15
        self.preview_fps_main = float(self.config.get("preview_fps_main", DEFAULT_PREVIEW_FPS_MAIN)) if hasattr(self, "config") else DEFAULT_PREVIEW_FPS_MAIN
        self.preview_fps_thumb = float(self.config.get("preview_fps_thumb", DEFAULT_PREVIEW_FPS_THUMB)) if hasattr(self, "config") else DEFAULT_PREVIEW_FPS_THUMB
        self.preview_pause_when_hidden = bool(self.config.get("preview_pause_when_hidden", DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN)) if hasattr(self, "config") else DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN
        self.overload_protection_enabled = bool(self.config.get("overload_protection_enabled", DEFAULT_OVERLOAD_PROTECTION_ENABLED)) if hasattr(self, "config") else DEFAULT_OVERLOAD_PROTECTION_ENABLED
        self.overload_min_camera_count = int(self.config.get("overload_min_camera_count", DEFAULT_OVERLOAD_MIN_CAMERA_COUNT)) if hasattr(self, "config") else DEFAULT_OVERLOAD_MIN_CAMERA_COUNT
        self.overload_camera_count_threshold = int(self.config.get("overload_camera_count_threshold", DEFAULT_OVERLOAD_CAMERA_COUNT_THRESHOLD)) if hasattr(self, "config") else DEFAULT_OVERLOAD_CAMERA_COUNT_THRESHOLD
        self.overload_reduce_thumb_preview_fps = float(self.config.get("overload_reduce_thumb_preview_fps", DEFAULT_OVERLOAD_REDUCE_THUMB_PREVIEW_FPS)) if hasattr(self, "config") else DEFAULT_OVERLOAD_REDUCE_THUMB_PREVIEW_FPS
        self.overload_reduce_detect_fps_factor = float(self.config.get("overload_reduce_detect_fps_factor", DEFAULT_OVERLOAD_REDUCE_DETECT_FPS_FACTOR)) if hasattr(self, "config") else DEFAULT_OVERLOAD_REDUCE_DETECT_FPS_FACTOR
        self.overload_disable_nonessential_overlays = bool(self.config.get("overload_disable_nonessential_overlays", DEFAULT_OVERLOAD_DISABLE_NONESSENTIAL_OVERLAYS)) if hasattr(self, "config") else DEFAULT_OVERLOAD_DISABLE_NONESSENTIAL_OVERLAYS
        self.overload_enter_debounce_seconds = float(self.config.get("overload_enter_debounce_seconds", DEFAULT_OVERLOAD_ENTER_DEBOUNCE_SECONDS)) if hasattr(self, "config") else DEFAULT_OVERLOAD_ENTER_DEBOUNCE_SECONDS
        self.overload_exit_debounce_seconds = float(self.config.get("overload_exit_debounce_seconds", DEFAULT_OVERLOAD_EXIT_DEBOUNCE_SECONDS)) if hasattr(self, "config") else DEFAULT_OVERLOAD_EXIT_DEBOUNCE_SECONDS
        self.overload_mode_active = False
        self._overload_last_change_ts = 0.0

        self.diag_panel = QLabel("Diagnostyka (debug): brak danych")
        self.diag_panel.setStyleSheet("color: #dddddd; background: #111; padding: 8px; border: 1px solid #333;")
        self.diag_panel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.diag_panel.setMinimumHeight(90)
        self.diag_panel.setVisible(False)
        self.diag_timer = QTimer(self)
        self.diag_timer.timeout.connect(self._update_diagnostics_panel)
        self.diag_timer.start(1000)

        self.watchdog_timer = QTimer(self)
        self.watchdog_timer.timeout.connect(self._run_watchdogs)
        self.watchdog_timer.start(5000)

        # zacznij od startu wszystkich, ale z niewielkim opóźnieniem aby GUI
        # mogło się pojawić bez czekania na inicjalizację kamer
        QTimer.singleShot(0, self.start_all)

    def _log_info(self, group: str, message: str, **kwargs) -> None:
        APP_LOG_BRIDGE.info(group, message, **kwargs)

    def _log_warning(self, group: str, message: str, **kwargs) -> None:
        APP_LOG_BRIDGE.warning(group, message, **kwargs)

    def _log_error(self, group: str, message: str, **kwargs) -> None:
        APP_LOG_BRIDGE.error(group, message, **kwargs)

    def _log_exception(self, group: str, message: str, exc: BaseException | None = None, **kwargs) -> None:
        APP_LOG_BRIDGE.exception(group, message, exc=exc, **kwargs)

    def _run_watchdogs(self) -> None:
        active = {}
        for idx, cam in enumerate(self.cameras):
            name = str(cam.get("name", idx))
            worker = self.workers[idx] if idx < len(self.workers) else None
            active[name] = isinstance(worker, CameraWorker) and worker.isRunning()
        stale = evaluate_heartbeat_health(active, self._heartbeat_last_seen, timeout_seconds=15.0)
        stale_set = set(stale)
        for name in stale:
            if name not in self._heartbeat_alerted:
                self._heartbeat_alerted.add(name)
                age = time.monotonic() - float(self._heartbeat_last_seen.get(name, 0.0) or 0.0)
                self._log_warning("performance", f"worker heartbeat timeout", source="heartbeat-watchdog", camera=name, details=f"last_seen_age_s={age:.1f} timeout_s=15")
        recovered = [name for name in list(self._heartbeat_alerted) if name not in stale_set]
        for name in recovered:
            self._log_info("performance", "worker heartbeat recovered", source="heartbeat-watchdog", camera=name)
        self._heartbeat_alerted.intersection_update(stale_set)

        now = time.monotonic()
        for idx, cam in enumerate(self.cameras):
            name = str(cam.get("name", idx))
            worker = self.workers[idx] if idx < len(self.workers) else None
            is_running = isinstance(worker, CameraWorker) and worker.isRunning()
            frame_age = now - float(self._last_frame_update_ts.get(idx, 0.0) or 0.0)
            key = f"{name}:noframe"
            if is_running and frame_age > 20.0:
                if key not in self._watchdog_flags:
                    self._watchdog_flags.add(key)
                    self._log_warning("performance", f"Brak aktualizacji klatki od {frame_age:.1f}s", source="gui-watchdog", camera=name)
            else:
                self._watchdog_flags.discard(key)

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
        if self._is_fullscreen:
            self.showNormal()
            if self._normal_geometry is not None:
                self.setGeometry(self._normal_geometry)
            self._is_fullscreen = False
        else:
            self._normal_geometry = self.geometry()
            self.showFullScreen()
            self._is_fullscreen = True

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            self._is_fullscreen = self.isFullScreen()
            if not self._is_fullscreen and self._normal_geometry is not None:
                self.setGeometry(self._normal_geometry)
        super().changeEvent(event)

    def toggle_sound(self):
        self.sound_enabled = not self.sound_enabled
        self._apply_sound_state()
        state = "włączono" if self.sound_enabled else "wyłączono"
        self.log_window.add_entry("application", f"{state} powiadomienia dźwiękowe")

    def _setup_sound_menu(self) -> None:
        self.sound_menu = QMenu(self)
        self.action_mute = QAction("Wycisz powiadomienia", self)
        self.action_resume = QAction("Wznów powiadomienia", self)
        self.action_mute.triggered.connect(lambda: self._set_sound_enabled(False))
        self.action_resume.triggered.connect(lambda: self._set_sound_enabled(True))
        self.sound_menu.addAction(self.action_mute)
        self.sound_menu.addAction(self.action_resume)

        slider_widget = QWidget()
        slider_layout = QVBoxLayout(slider_widget)
        slider_layout.setContentsMargins(12, 8, 12, 8)
        slider_header = QHBoxLayout()
        slider_label = QLabel("Głośność")
        self.sound_percent_label = QLabel()
        slider_header.addWidget(slider_label)
        slider_header.addStretch(1)
        slider_header.addWidget(self.sound_percent_label)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(self.sound_volume * 100))
        slider.valueChanged.connect(self._on_sound_volume_changed)
        slider_layout.addLayout(slider_header)
        slider_layout.addWidget(slider)

        volume_action = QWidgetAction(self)
        volume_action.setDefaultWidget(slider_widget)
        self.sound_menu.addAction(volume_action)

        self.btn_sound.setMenu(self.sound_menu)
        self.btn_sound.setPopupMode(QToolButton.InstantPopup)
        self._apply_sound_state()

    def _set_sound_enabled(self, enabled: bool) -> None:
        if self.sound_enabled == enabled:
            return
        self.sound_enabled = enabled
        self._apply_sound_state()
        state = "włączono" if self.sound_enabled else "wyłączono"
        self.log_window.add_entry("application", f"{state} powiadomienia dźwiękowe")

    def _apply_sound_state(self) -> None:
        icon = "volume-up.svg" if self.sound_enabled else "volume-mute.svg"
        self.btn_sound.setIcon(QIcon(str(ICON_DIR / icon)))
        if self.alert_sound:
            volume = self.sound_volume if self.sound_enabled else 0.0
            self.alert_sound.setVolume(volume)
        if hasattr(self, "action_mute"):
            self.action_mute.setEnabled(self.sound_enabled)
            self.action_resume.setEnabled(not self.sound_enabled)
        if hasattr(self, "sound_percent_label"):
            self.sound_percent_label.setText(f"{int(round(self.sound_volume * 100))}%")

    def _on_sound_volume_changed(self, value: int) -> None:
        self.sound_volume = max(0.0, min(1.0, value / 100.0))
        if self.alert_sound and self.sound_enabled:
            self.alert_sound.setVolume(self.sound_volume)
        if hasattr(self, "sound_percent_label"):
            self.sound_percent_label.setText(f"{int(round(self.sound_volume * 100))}%")

    def play_alert_sound(self):
        if self.alert_sound:
            try:
                self.alert_sound.play()
            except Exception as e:
                self._log_exception("error", f"Alert sound playback failed: {e}", exc=e, source="audio")

    def open_alert_dialog(self):
        dlg = AlertDialog(self)
        dlg.exec_()

    def open_log_settings_dialog(self):
        self.log_window.add_entry("settings", "otwarto ustawienia logów")
        dlg = LogSettingsDialog(self)
        dlg.exec_()

    def update_log_retention_hours(self, hours: int) -> None:
        hours = max(1, min(24 * 7, int(hours)))
        if self.log_window.retention_hours == hours:
            return
        self.log_window.set_retention_hours(hours)
        cfg = load_config()
        cfg["cameras"] = self.cameras
        cfg["log_retention_hours"] = hours
        save_config(cfg)
        config_module.LOG_RETENTION_HOURS = hours

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

    def _get_model(self, model_name: str):
        if model_name in self.model_cache:
            self.log_window.add_entry("application", f"model z cache: {model_name}")
            return self.model_cache[model_name]
        model = dg.load_model(
            model_name=model_name,
            inference_host_address="@local",
            zoo_url=str(MODELS_PATH / model_name),
        )
        self.model_cache[model_name] = model
        self.log_window.add_entry("application", f"model załadowany: {model_name}")
        return model

    def _apply_worker_preview_roles(self) -> None:
        selected_idx = self.camera_list.currentRow()
        for idx, worker in enumerate(self.workers):
            if not isinstance(worker, CameraWorker):
                continue
            role = "main" if idx == selected_idx else "thumb"
            worker.preview_fps_main = self.preview_fps_main
            worker.preview_fps_thumb = self.preview_fps_thumb
            worker.preview_pause_when_hidden = self.preview_pause_when_hidden
            worker.set_preview_role(role)

    def _evaluate_overload_mode(self) -> None:
        active_workers = [w for w in self.workers if isinstance(w, CameraWorker) and w.isRunning()]
        active_count = len(active_workers)
        recording_count = sum(1 for w in active_workers if w.recording)
        gui_load = sum(max(0.0, float(st.get("stream_fps", 0.0))) for st in self.worker_status.values())
        now_ts = time.monotonic()
        overload_active, change_ts, reason = evaluate_overload_transition(
            now_ts=now_ts,
            active_camera_count=active_count,
            gui_load_fps=gui_load,
            recording_count=recording_count,
            currently_active=self.overload_mode_active,
            last_change_ts=self._overload_last_change_ts,
            protection_enabled=self.overload_protection_enabled,
            min_camera_count=self.overload_min_camera_count,
            camera_threshold=self.overload_camera_count_threshold,
            load_per_camera_threshold=10.0,
            enter_debounce_seconds=self.overload_enter_debounce_seconds,
            exit_debounce_seconds=self.overload_exit_debounce_seconds,
        )
        self._overload_last_change_ts = change_ts

        if overload_active != self.overload_mode_active:
            self.overload_mode_active = overload_active
            mode = "enter" if overload_active else "exit"
            self._log_info(
                "application",
                f"overload {mode}",
                source="app",
                details=(
                    f"reason={reason} active_cameras={active_count} min_cameras={self.overload_min_camera_count} "
                    f"camera_threshold={self.overload_camera_count_threshold} gui_load={gui_load:.2f} "
                    f"enter_debounce_s={self.overload_enter_debounce_seconds} exit_debounce_s={self.overload_exit_debounce_seconds}"
                ),
            )

        selected_idx = self.camera_list.currentRow()
        for idx, worker in enumerate(self.workers):
            if not isinstance(worker, CameraWorker) or not worker.isRunning():
                continue
            is_main = idx == selected_idx
            detect_factor = 1.0 if is_main or worker.recording else (self.overload_reduce_detect_fps_factor if overload_active else 1.0)
            thumb_fps = self.overload_reduce_thumb_preview_fps if overload_active else self.preview_fps_thumb
            worker.set_overload_state(
                overload_active=overload_active and not is_main,
                detect_fps_factor=detect_factor,
                thumb_preview_fps=thumb_fps,
                disable_overlays=self.overload_disable_nonessential_overlays,
            )

    def _refresh_camera_status_indicators(self) -> None:
        now = time.monotonic()
        for idx, cam in enumerate(self.cameras):
            name = str(cam.get("name", idx))
            stat = self.worker_status.get(name, {})
            flags = []
            if bool(stat.get("recording_active", False)):
                flags.append("REC")
            det_seconds = float(stat.get("last_detection_seconds", -1.0))
            if 0 <= det_seconds <= 10.0:
                flags.append("DET")
            if bool(stat.get("overload_degraded", False)):
                flags.append("DEG")
            if bool(stat.get("stream_error_active", False)):
                flags.append("ERR")
            suffix = f" [{' '.join(flags)}]" if flags else ""
            if idx < len(self.camera_list.widgets):
                self.camera_list.widgets[idx].text_label.setText(f"{name}{suffix}")
            if idx < len(self.camera_grid.items):
                self.camera_grid.items[idx].name_label.setText(f"{name}{suffix}")

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
            model = self._get_model(model_name)
        except Exception as e:
            QMessageBox.warning(self, "Model", f"Nie udało się załadować modelu '{model_name}': {e}")
            self._log_error("error", f"model {model_name}: {e}", source="app", camera=str(cam.get("name", idx)))
            return
        w = CameraWorker(camera=cam, model=model, index=idx)
        w.preview_fps_main = self.preview_fps_main
        w.preview_fps_thumb = self.preview_fps_thumb
        w.preview_pause_when_hidden = self.preview_pause_when_hidden
        w.frame_signal.connect(self.update_frame)
        w.alert_signal.connect(self.on_new_alert)
        w.error_signal.connect(self._worker_error)
        w.status_signal.connect(self._worker_status)
        w.record_signal.connect(lambda event, fp, cam_name=cam.get("name", idx): self.on_record_event(event, fp, cam_name))
        w.worker_status_signal.connect(self._on_worker_heartbeat)
        self._log_info("worker", "camera start requested", source="app", camera=str(cam.get("name", idx)))
        w.start()
        self.workers[idx] = w
        self._apply_worker_preview_roles()
        self._evaluate_overload_mode()
        self._log_info("worker", f"uruchomiono kamerę {cam.get('name', idx)}", source="app", camera=str(cam.get("name", idx)))

    def stop_camera(self, idx: int):
        if not (0 <= idx < len(self.workers)):
            return
        if 0 <= idx < len(self.cameras):
            self._log_info("worker", "camera stop requested", source="app", camera=str(self.cameras[idx].get("name", idx)))
        w = self.workers[idx]
        if not isinstance(w, CameraWorker):
            return

        cam = self.cameras[idx]
        stopped = w.stop()
        if not stopped:
            self._log_warning("worker", "stop_camera timeout", source="app", camera=str(cam.get("name", idx)), details="worker did not stop in timeout window")

        with suppress(Exception):
            w.frame_signal.disconnect(self.update_frame)
        with suppress(Exception):
            w.alert_signal.disconnect(self.on_new_alert)
        with suppress(Exception):
            w.error_signal.disconnect(self._worker_error)
        with suppress(Exception):
            w.status_signal.disconnect(self._worker_status)

        self.workers[idx] = None
        self._last_frame.pop(idx, None)
        self._last_fps_text[idx] = ""
        self._last_status[idx] = "Zatrzymano"
        self._last_error.pop(idx, None)
        cam_name = str(cam.get("name", idx))
        self.worker_status.pop(cam_name, None)
        self._worker_diag.pop(cam_name, None)
        if hasattr(self.camera_grid, "update_frame"):
            blank = np.zeros((180, 320, 3), dtype=np.uint8)
            with suppress(Exception):
                self.camera_grid.update_frame(idx, blank)
        if hasattr(self.camera_list, "update_thumbnail"):
            blank = np.zeros((180, 320, 3), dtype=np.uint8)
            with suppress(Exception):
                self.camera_list.update_thumbnail(idx, blank)
        if idx == self.camera_list.currentRow():
            self._render_current()

        self._log_info("worker", "stop_camera completed", source="app", camera=str(cam.get("name", idx)), details=f"stopped={stopped}")
        self._evaluate_overload_mode()
        self._refresh_camera_status_indicators()


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
        self._log_error("error", f"{cam_name}: {cause}", source="worker", camera=str(cam_name))
        if "Brak sygnału" in cause:
            self._log_warning("worker", "brak sygnału RTSP", source="worker", camera=str(cam_name))
        if idx == self.camera_list.currentRow():
            self._render_current()

    def _on_worker_heartbeat(self, camera_name: str, status: dict):
        payload = dict(status or {})
        cam_name = str(camera_name)
        self._worker_diag[cam_name] = payload
        self.worker_status[cam_name] = payload
        self._heartbeat_last_seen[cam_name] = time.monotonic()
        if cam_name in self._heartbeat_alerted:
            self._log_info("performance", "worker heartbeat recovered", source="heartbeat-watchdog", camera=cam_name)
        self._heartbeat_alerted.discard(cam_name)
        self._evaluate_overload_mode()
        self._refresh_camera_status_indicators()

    def _update_diagnostics_panel(self):
        if not self.diag_panel.isVisible():
            return
        idx = self.camera_list.currentRow()
        if idx < 0 or idx >= len(self.cameras):
            self.diag_panel.setText("Diagnostyka: brak wybranej kamery")
            return
        name = str(self.cameras[idx].get("name", idx))
        stat = self.worker_status.get(name, {})
        if not stat:
            self.diag_panel.setText(f"Diagnostyka [{name}]: oczekiwanie na heartbeat")
            return
        self.diag_panel.setText(
            "\n".join(
                [
                    f"[{name}]",
                    f"stream fps: {float(stat.get('stream_fps', 0.0)):.2f}",
                    f"detect fps: {float(stat.get('detect_fps', 0.0)):.2f}",
                    f"writer fps: {float(stat.get('writer_fps', 0.0)):.2f}",
                    f"recording queue size: {int(stat.get('queue_size', 0))}",
                    f"dropped frames: {int(stat.get('dropped_frames', 0))}",
                    f"preview role: {stat.get('preview_role', '-')}",
                    f"overload degraded: {bool(stat.get('overload_degraded', False))}",
                    f"last detection seconds: {float(stat.get('last_detection_seconds', -1.0)):.1f}",
                ]
            )
        )

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

    def _requires_worker_restart(self, changed_keys: list[str], old_camera: dict, new_camera: dict) -> tuple[bool, list[str]]:
        del old_camera, new_camera
        restart_keys = [key for key in changed_keys if key in CAMERA_RESTART_REQUIRED_FIELDS]
        return bool(restart_keys), restart_keys

    def _restart_camera_with_new_settings(self, idx: int, was_running: bool) -> bool:
        if not was_running:
            return False
        camera_name = str(self.cameras[idx].get("name", idx)) if idx < len(self.cameras) else str(idx)
        self._log_info("settings", "automatic camera restart due to settings change", source="app", camera=camera_name)
        try:
            self.stop_camera(idx)
            self.start_camera(idx)
            self._log_info("settings", "camera restart success", source="app", camera=camera_name)
            return True
        except Exception as exc:
            self._log_exception("error", "camera restart failure", exc=exc, source="app", camera=camera_name, details=traceback.format_exc())
            return False

    def _apply_camera_settings_change(self, idx: int, old_camera: dict, new_camera: dict) -> dict:
        changed_keys, _ = classify_camera_setting_changes(old_camera, new_camera, CAMERA_RESTART_REQUIRED_FIELDS)
        requires_restart, restart_reason_keys = self._requires_worker_restart(changed_keys, old_camera, new_camera)
        worker = self.workers[idx] if idx < len(self.workers) else None
        was_running = isinstance(worker, CameraWorker) and worker.isRunning()

        result = {
            "saved": True,
            "changed_keys": changed_keys,
            "restart_reason_keys": restart_reason_keys,
            "applied_live": False,
            "restarted": False,
            "was_running": was_running,
        }

        if not changed_keys:
            return result

        if requires_restart:
            result["restarted"] = self._restart_camera_with_new_settings(idx, was_running)
            return result

        if was_running:
            worker.apply_runtime_settings(new_camera)
            result["applied_live"] = True
        return result

    def _show_camera_settings_result_message(self, camera_name: str, result: dict) -> None:
        if result.get("restarted"):
            message = f"Ustawienia kamery „{camera_name}” zostały zapisane. Zmiany wymagające restartu zostały zastosowane automatycznie."
        elif result.get("applied_live"):
            message = f"Ustawienia kamery „{camera_name}” zostały zapisane i zastosowane bez restartu."
        elif result.get("was_running"):
            message = f"Ustawienia kamery „{camera_name}” zostały zapisane."
        else:
            message = f"Ustawienia kamery „{camera_name}” zostały zapisane."
        if self.statusBar() is not None:
            self.statusBar().showMessage(message, 8000)
        QMessageBox.information(self, "Ustawienia kamery", message)

    def camera_settings(self, idx: int):
        cam = dict(self.cameras[idx])
        dlg = SingleCameraDialog(self, cam)
        if not dlg.exec_():
            return
        new_data = dlg.result_camera
        try:
            if new_data["name"] != cam["name"] and any(c["name"] == new_data["name"] for i, c in enumerate(self.cameras) if i != idx):
                QMessageBox.warning(self, "Duplikat", f"Kamera o nazwie '{new_data['name']}' już istnieje.")
                return

            fill_camera_defaults(new_data)
            self.cameras[idx] = new_data
            cfg = load_config()
            cfg["cameras"] = self.cameras
            save_config(cfg)

            self.camera_list.rebuild(self.cameras)
            self.camera_grid.rebuild(self.cameras)
            self.camera_list.setCurrentRow(idx)

            result = self._apply_camera_settings_change(idx, cam, new_data)
            self.log_window.add_entry(
                "settings",
                f"zapisano ustawienia kamery {new_data.get('name')} changed={result.get('changed_keys', [])} restart={result.get('restart_reason_keys', [])}",
            )
            self._show_camera_settings_result_message(new_data.get("name", "kamera"), result)
        except Exception as exc:
            self._log_exception("error", f"błąd zapisu ustawień kamery {new_data.get('name', 'unknown')}: {exc}", exc=exc, source="ui", camera=str(new_data.get("name", "unknown")))
            QMessageBox.critical(self, "Błąd ustawień", f"Nie udało się zapisać ustawień kamery: {exc}")

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

    def start_cameras_staggered(self, camera_names):
        names = list(camera_names or [])
        for offset, name in enumerate(names):
            def _start_one(cam_name=name):
                for idx, cam in enumerate(self.cameras):
                    if cam.get("name") == cam_name:
                        self.start_camera(idx)
                        return
            QTimer.singleShot(int(offset * 200), _start_one)

    def start_all(self):
        self._log_info("worker", "start_all requested", source="app")
        self.stop_all()
        self.workers = [None] * len(self.cameras)
        self.start_cameras_staggered([str(c.get("name", "")) for c in self.cameras])
        if self.camera_list.currentRow() < 0 and self.cameras:
            self.camera_list.setCurrentRow(0)
        # przy starcie — brak klatki jeszcze: narysuj HUD "Łączenie…"
        self._last_status[self.camera_list.currentRow()] = "Łączenie…"
        self._apply_worker_preview_roles()
        self._render_current()

    def stop_all(self):
        self._log_info("worker", "stop_all requested", source="app")
        for w in self.workers:
            if isinstance(w, CameraWorker):
                w.stop()
        self.workers = []
        self.worker_status.clear()
        self.overload_mode_active = False

    def switch_camera(self, idx):
        # odśwież HUD dla nowej kamery
        self.last_render_time = 0.0
        self._apply_worker_preview_roles()
        self._evaluate_overload_mode()
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
        self._last_frame_update_ts[idx] = time.monotonic()

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

    def _build_camera_hud_lines(self, idx: int) -> list[str]:
        if idx < 0 or idx >= len(self.cameras):
            return []
        cam = self.cameras[idx]
        name = str(cam.get("name", idx))
        status = self._last_status.get(idx, "")
        err = self._last_error.get(idx, "")
        fps_txt = self._last_fps_text.get(idx, "")
        stat = self.worker_status.get(name, {})

        flags = []
        if bool(stat.get("recording_active", False)):
            flags.append("REC")
        if 0 <= float(stat.get("last_detection_seconds", -1.0)) <= 10.0:
            flags.append("DET")
        if bool(stat.get("stream_error_active", False)):
            flags.append("ERR")
        if bool(stat.get("overload_degraded", False)):
            flags.append("DEG")

        lines = [name]
        state_line = f"Status: {err if err else (status or 'Brak danych')}"
        if flags:
            state_line += f" [{' '.join(flags)}]"
        lines.append(state_line)
        metric_parts = []
        if fps_txt:
            metric_parts.append(f"preview {fps_txt}")
        metric_parts.extend([
            f"stream {float(stat.get('stream_fps', 0.0)):.1f}",
            f"detect {float(stat.get('detect_fps', 0.0)):.1f}",
            f"writer {float(stat.get('writer_fps', 0.0)):.1f}",
            f"q {int(stat.get('queue_size', 0))}",
            f"drop {int(stat.get('dropped_frames', 0))}",
        ])
        lines.append(" | ".join(metric_parts))
        return lines


    @staticmethod
    def _compute_letterboxed_rect(frame_width: int, frame_height: int, canvas_width: int, canvas_height: int) -> tuple[int, int, int, int]:
        return compute_letterboxed_rect(frame_width, frame_height, canvas_width, canvas_height)

    @staticmethod
    def _camera_info_overlay_anchor(image_rect: tuple[int, int, int, int], box_size: tuple[int, int], padding: int = 10) -> tuple[int, int]:
        return camera_overlay_anchor(image_rect, box_size, padding)

    def _draw_camera_info_overlay(self, qimg: QImage, idx: int, image_rect: tuple[int, int, int, int]) -> QImage:
        if qimg.isNull():
            return qimg

        cam = self.cameras[idx] if 0 <= idx < len(self.cameras) else {}
        if not bool(cam.get("show_camera_info_overlay", DEFAULT_SHOW_CAMERA_INFO_OVERLAY)):
            return qimg

        lines = self._build_camera_hud_lines(idx)
        if not lines:
            return qimg

        w_label = qimg.width()
        font_size = 10 if w_label < 900 else 12 if w_label < 1400 else 14
        painter = QPainter(qimg)
        try:
            font = QFont("DejaVu Sans", font_size)
            painter.setFont(font)
            fm = painter.fontMetrics()
            pad = 8
            line_h = fm.height() + 2
            text_w = max(fm.horizontalAdvance(line) for line in lines)
            box_w = text_w + 2 * pad
            box_h = line_h * len(lines) + 2 * pad
            x, y = self._camera_info_overlay_anchor(image_rect, (box_w, box_h), padding=10)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 128))
            painter.drawRoundedRect(x, y, box_w, box_h, 8, 8)

            painter.setPen(QColor(255, 255, 255))
            for i, line in enumerate(lines):
                painter.drawText(x + pad, y + pad + (i + 1) * line_h - 4, line)
        finally:
            painter.end()

        return qimg

    def _compose_letterboxed(self, frame, idx: int):
        w_label = max(1, self.camera_view.width())
        h_label = max(1, self.camera_view.height())
        canvas = np.zeros((h_label, w_label, 3), dtype=np.uint8)

        image_rect = (0, 0, w_label, h_label)
        if frame is not None:
            fh, fw = frame.shape[:2]
            if fh > 0 and fw > 0:
                x0, y0, new_w, new_h = self._compute_letterboxed_rect(fw, fh, w_label, h_label)
                image_rect = (x0, y0, new_w, new_h)
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                canvas[y0:y0+new_h, x0:x0+new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w_label, h_label, rgb.strides[0], QImage.Format_RGB888).copy()

        return self._draw_camera_info_overlay(qimg, idx, image_rect)

    def _render_current(self):
        now = time.time()
        if now - self.last_render_time < self._render_interval_s:
            return
        self.last_render_time = now

        idx = self.camera_list.currentRow()
        if idx < 0:
            return
        frame = self._last_frame.get(idx)
        composed_qimg = self._compose_letterboxed(
            frame if frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8),
            idx,
        )
        self.camera_view.setPixmap(QPixmap.fromImage(composed_qimg))


    def open_video_file(self, filepath: str):
        self._log_info("browser", f"odtworzono nagranie {os.path.basename(filepath)}", source="ui")
        dlg = VideoPlayerDialog(filepath, self)
        dlg.exec_()

    def open_recordings_browser(self):
        self._log_info("browser", "otwarto przeglądarkę nagrań", source="ui")
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
        self.alert_mem.flush()
        flush_storage()
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
    faulthandler.enable()
    install_global_exception_hooks()
    install_qt_message_handler()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    cfg = load_config()
    app = QApplication(sys.argv)
    win = MainWindow(cameras=cfg.get("cameras", []))
    ui_handler = UILoggingHandler(APP_LOG_BRIDGE)
    ui_handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    root_logger = logging.getLogger()
    if not any(isinstance(h, UILoggingHandler) for h in root_logger.handlers):
        root_logger.addHandler(ui_handler)
    APP_LOG_BRIDGE.info("application", "startup completed", source="startup")
    if windowed:
        win.show()
    else:
        win.showFullScreen()
    code = app.exec_()
    APP_LOG_BRIDGE.info("application", "app shutdown", source="shutdown")
    sys.exit(code)
