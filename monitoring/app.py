
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
    QThread,
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
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QAction,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QTableWidget,
    QTableWidgetItem,
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
    DEFAULT_DETECTION_FPS_LIMIT,
    DEFAULT_LOST_SECONDS,
    DEFAULT_MODEL,
    DEFAULT_POST_SECONDS,
    DEFAULT_PREVIEW_FPS_MAIN,
    DEFAULT_PREVIEW_FPS_GRID,
    DEFAULT_PREVIEW_FPS_THUMB,
    DEFAULT_PREVIEW_MAIN_MAX_HEIGHT,
    DEFAULT_PREVIEW_GRID_MAX_HEIGHT,
    DEFAULT_GRID_PREVIEW_QUALITY,
    DEFAULT_PREVIEW_MAIN_MAX_WIDTH,
    DEFAULT_PREVIEW_GRID_MAX_WIDTH,
    DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN,
    DEFAULT_PREVIEW_THUMB_MAX_HEIGHT,
    DEFAULT_PREVIEW_THUMB_MAX_WIDTH,
    DEFAULT_SHOW_CAMERA_INFO_OVERLAY,
    DEFAULT_QUALITY_PERFORMANCE_PRESET,
    DEFAULT_CONFIG_WATCHDOG_ENABLED,
    DEFAULT_CONFIG_WATCHDOG_EVAL_SECONDS,
    DEFAULT_CONFIG_WATCHDOG_DROP_DELTA_THRESHOLD,
    DEFAULT_CONFIG_WATCHDOG_QUEUE_DELTA_THRESHOLD,
    QUALITY_PERFORMANCE_PRESETS,
    DEFAULT_CAMERA_INFO_OVERLAY_ALPHA,
    CAMERA_PRIORITIES,
    DEFAULT_OVERLOAD_PROTECTION_ENABLED,
    DEFAULT_OVERLOAD_MIN_CAMERA_COUNT,
    DEFAULT_OVERLOAD_CAMERA_COUNT_THRESHOLD,
    DEFAULT_OVERLOAD_REDUCE_THUMB_PREVIEW_FPS,
    DEFAULT_OVERLOAD_REDUCE_DETECT_FPS_FACTOR,
    DEFAULT_OVERLOAD_DISABLE_NONESSENTIAL_OVERLAYS,
    DEFAULT_OVERLOAD_ENTER_DEBOUNCE_SECONDS,
    DEFAULT_OVERLOAD_EXIT_DEBOUNCE_SECONDS,
    DEFAULT_OVERLOAD_MAX_UI_RENDER_MS,
    DEFAULT_OVERLOAD_MAX_QUEUE_SIZE,
    DEFAULT_OVERLOAD_MAX_PREVIEW_BANDWIDTH_MBPS,
    DEFAULT_PERFORMANCE_DIAGNOSTICS_ENABLED,
    DEFAULT_PERFORMANCE_LOG_INTERVAL_S,
    DEFAULT_LOG_FILTER_GROUPS,
    DEFAULT_LOG_FILTER_LEVELS,
    DEFAULT_LOG_FILTER_SOURCES,
    DEFAULT_PRE_SECONDS,
    DEFAULT_RECORD_PATH,
    DEFAULT_RECORD_START_MODE,
    DEFAULT_REQUIRED_HITS_TO_START_RECORDING,
    DEFAULT_REQUIRED_MISSES_TO_END_DETECTION,
    DEFAULT_MIN_RECORD_SECONDS,
    DEFAULT_RTSP_FPS,
    DEFAULT_SENSITIVITY_PROFILE,
    DEFAULT_THUMBNAIL_MODE,
    ICON_DIR,
    LOG_HISTORY_PATH,
    LOG_RETENTION_HOURS,
    MODELS_PATH,
    RECORDINGS_CATALOG_PATH,
    RECORD_CLASSES,
    SENSITIVITY_PROFILES,
    VISIBLE_CLASSES,
    apply_sensitivity_profile,
    fill_camera_defaults,
    normalize_log_filters,
    is_log_entry_enabled,
    infer_sensitivity_profile,
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
from .log_messages import PERFORMANCE_PARAM_LABELS, format_dict_multiline, msg
from .runtime_helpers import (
    app_log,
    build_root_cause_summary,
    camera_overlay_anchor,
    classify_camera_setting_changes,
    compute_letterboxed_rect,
    evaluate_heartbeat_health,
    evaluate_overload_transition,
    overload_level_profile,
    register_app_logger,
)
from .widgets.alerts import AlertDialog, AlertListWidget
from .widgets.camera_grid import CameraGridWidget
from .widgets.camera_list import CameraListWidget
from .widgets.logs import LogSettingsDialog, LogWindow
from .widgets.recordings_browser import RecordingsBrowserDialog
from .system_metrics import SystemMetricsSampler
from .degirum_devices import (
    benchmark_device_candidates,
    build_degirum_load_model_kwargs,
    detect_degirum_devices,
    get_model_supported_device_types,
    load_model_with_timeout,
    resolve_effective_degirum_selection,
    resolve_degirum_runtime_target,
    sanitize_degirum_load_model_kwargs,
)

# Qt platform plugin path (Linux)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

CAMERA_RESTART_REQUIRED_FIELDS = {
    "rtsp",
    "type",
    "model",
    "degirum_device_override_enabled",
    "degirum_device_override",
}
CAMERA_RUNTIME_APPLY_FIELDS = {
    "fps", "detection_fps_limit", "rtsp_fps", "confidence_threshold", "confidence_threshold_draw", "confidence_threshold_record",
    "draw_overlays", "enable_detection", "enable_recording", "visible_classes", "record_classes",
    "detection_hours", "record_path", "pre_seconds", "lost_seconds", "post_seconds",
    "required_hits_to_start_recording", "required_misses_to_end_detection", "min_record_seconds", "sensitivity_profile",
    "thumbnail_mode", "record_start_mode", "preview_fps_main", "preview_fps_grid", "preview_fps_thumb",
    "preview_pause_when_hidden", "preview_grid_max_width", "preview_grid_max_height", "camera_priority",
    "show_camera_info_overlay",
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
    "detection_fps_limit": (
        "Maksymalna liczba inferencji AI na sekundę dla tej kamery.\n\n"
        "Gdy brak tego pola w starszej konfiguracji, aplikacja używa wartości z pola FPS detekcji.\n"
        "Niższa wartość zmniejsza obciążenie, wyższa poprawia responsywność wykrywania."
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
    "sensitivity_profile": (
        "Profil czułości mapuje zestaw kluczowych parametrów detekcji i nagrywania.\n\n"
        "high_recall — większa czułość (więcej wykryć i potencjalnie więcej fałszywych alarmów),\n"
        "balanced — ustawienie domyślne z kompromisem skuteczności,\n"
        "high_precision — mniej fałszywych alarmów kosztem ryzyka pominięcia trudnych przypadków,\n"
        "custom — ręczne wartości bez automatycznego mapowania."
    ),
    "draw_overlays": (
        "Włącza rysowanie ramek, etykiet i opisów detekcji na podglądzie na żywo.\n\n"
        "Wyłączenie tej opcji może trochę zmniejszyć obciążenie systemu, szczególnie przy wielu kamerach.\n"
        "Detekcja może nadal działać nawet wtedy, gdy ramki nie są rysowane.\n\n"
        "Uwaga: miniatura nagrania może nadal zawierać zaznaczony obiekt, aby łatwiej było rozpoznać zdarzenie."
    ),
    "show_camera_info_overlay": (
        "Pokazuje na obrazie kamery informacje o statusie połączenia, FPS, nagrywaniu i diagnostyce.\n\n"
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
        self.setMinimumSize(840, 520)

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
        self.btn_prev = QPushButton("⏮ Poprzednie")
        self.btn_next = QPushButton("Następne ⏭")
        self.btn_snap = QPushButton("📷")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTracking(True)
        self.speed_combo = QComboBox()
        self.speed_combo.addItem("0.5x", 0.5)
        self.speed_combo.addItem("1.0x", 1.0)
        self.speed_combo.addItem("1.5x", 1.5)
        self.speed_combo.addItem("2.0x", 2.0)
        self.speed_combo.setCurrentIndex(1)
        self.loop_chk = QCheckBox("Zapętl")
        self.autoplay_next_chk = QCheckBox("Auto następne")
        self.autoplay_next_chk.setChecked(True)
        self.position_label = QLabel("0:00:00 / 0:00:00")
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
        ctrl.addWidget(self.position_label)
        ctrl.addWidget(self.speed_combo)
        ctrl.addWidget(self.loop_chk)
        ctrl.addWidget(self.autoplay_next_chk)
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
        self.slider.valueChanged.connect(self._seek_preview)
        self.slider.sliderReleased.connect(self.seek_to_slider)
        self.speed_combo.currentIndexChanged.connect(self._apply_playback_speed)

        self.video_label.mouseDoubleClickEvent = lambda e: self.toggle_fullscreen()

        self.cap = None
        self.current_index = 0
        self.current_frame = None
        self._normal_geometry = None
        self._is_fullscreen = False
        self.playback_speed = 1.0
        try:
            self.timer.setTimerType(Qt.PreciseTimer)
        except Exception:
            pass
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
            self._update_position_label()

    def _next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            if self.loop_chk.isChecked():
                self._show_frame_at(0)
                self.play()
                return
            if self.autoplay_next_chk.isChecked() and self.file_index < len(self.file_list) - 1:
                was_playing = self.timer.isActive()
                self.next_video()
                if was_playing:
                    self.play()
                return
            self.pause()
            return
        self.current_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_index)
        self.slider.blockSignals(False)
        self._show_frame(frame)
        self._update_position_label()

    def play(self):
        if not self.timer.isActive():
            interval_ms = int(1000 / max(self.fps * self.playback_speed, 0.01))
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

    def _seek_preview(self, value: int):
        if self.slider.isSliderDown():
            self._show_frame_at(int(value))

    def _apply_playback_speed(self):
        self.playback_speed = float(self.speed_combo.currentData() or 1.0)
        if self.timer.isActive():
            self.pause()
            self.play()

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
        self._update_position_label()

    def next_video(self):
        if self.file_index < len(self.file_list) - 1:
            self.file_index += 1
            self.load_video(self.file_list[self.file_index])
        elif self.loop_chk.isChecked() and self.file_list:
            self.file_index = 0
            self.load_video(self.file_list[self.file_index])

    def prev_video(self):
        if self.file_index > 0:
            self.file_index -= 1
            self.load_video(self.file_list[self.file_index])
        elif self.loop_chk.isChecked() and self.file_list:
            self.file_index = len(self.file_list) - 1
            self.load_video(self.file_list[self.file_index])

    def _update_position_label(self):
        total_seconds = (self.frame_count / max(self.fps, 1e-3)) if self.frame_count else 0.0
        current_seconds = (self.current_index / max(self.fps, 1e-3)) if self.frame_count else 0.0
        self.position_label.setText(
            f"{datetime.timedelta(seconds=int(max(0, current_seconds)))} / {datetime.timedelta(seconds=int(max(0, total_seconds)))}"
        )

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
        if event.key() == Qt.Key_Space:
            if self.timer.isActive():
                self.pause()
            else:
                self.play()
            event.accept()
            return
        if event.key() in (Qt.Key_Right, Qt.Key_D):
            self.step_forward()
            event.accept()
            return
        if event.key() in (Qt.Key_Left, Qt.Key_A):
            self.step_back()
            event.accept()
            return
        if event.key() == Qt.Key_N:
            self.next_video()
            event.accept()
            return
        if event.key() == Qt.Key_P:
            self.prev_video()
            event.accept()
            return
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
        self._degirum_device_label_map: dict[str, str] = {
            "inherit": "Dziedzicz (globalne)",
            "cpu": "CPU (procesor)",
            "gpu": "GPU (karta graficzna)",
        }

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
        self.detection_fps_limit_spin = QSpinBox(); self.detection_fps_limit_spin.setRange(1, 60)
        self.priority_combo = QComboBox(); self.priority_combo.addItems(list(CAMERA_PRIORITIES))
        self.rtsp_fps_spin = QSpinBox(); self.rtsp_fps_spin.setRange(0, 60); self.rtsp_fps_spin.setSpecialValueText("Auto")
        self.conf_spin = QDoubleSpinBox(); self.conf_spin.setRange(0.0, 1.0); self.conf_spin.setSingleStep(0.05)
        self.conf_draw_spin = QDoubleSpinBox(); self.conf_draw_spin.setRange(0.0, 1.0); self.conf_draw_spin.setSingleStep(0.05)
        self.conf_record_spin = QDoubleSpinBox(); self.conf_record_spin.setRange(0.0, 1.0); self.conf_record_spin.setSingleStep(0.05)
        self.show_legacy_conf_chk = QCheckBox("Pokaż ustawienia legacy (zaawansowane)")
        self.show_legacy_conf_chk.setChecked(False)
        self.sensitivity_profile_combo = QComboBox(); self.sensitivity_profile_combo.addItems(["balanced", "high_recall", "high_precision", "custom"])
        self.draw_chk = QCheckBox(); self.detect_chk = QCheckBox(); self.record_chk = QCheckBox()
        self.info_overlay_chk = QCheckBox()
        self.hours_edit = QLineEdit(); self.visible_edit = QLineEdit(); self.record_edit = QLineEdit()
        self.record_classes_hint = QLabel("")
        self.record_classes_hint.setWordWrap(True)
        self.record_classes_hint.setStyleSheet("color:#f7d26a;")
        self._model_labels_cache: dict[str, set[str] | None] = {}
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
        self.degirum_device_override_enabled_chk = QCheckBox("Nadpisz ustawienie globalne")
        self.degirum_device_override_combo = QComboBox()
        self._refresh_degirum_override_options()
        self.degirum_device_override_enabled_chk.toggled.connect(self.degirum_device_override_combo.setEnabled)

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
        self._add_field_row(left_layout, "detection_fps_limit", "FPS/S DETECT LIMIT", self.detection_fps_limit_spin)
        self._add_field_row(left_layout, "camera_priority", "Priorytet kamery", self.priority_combo)
        self._add_field_row(left_layout, "rtsp_fps", "FPS/S RTSP", self.rtsp_fps_spin)
        self._add_field_row(left_layout, "show_camera_info_overlay", "Pokaż informacje na obrazie", self.info_overlay_chk)

        self._add_field_row(middle_layout, "confidence_threshold", "Próg pewności (legacy, kompatybilność)", self.conf_spin)
        self._add_field_row(middle_layout, "confidence_threshold_draw", "Próg wizualizacji (DRAW)", self.conf_draw_spin)
        self._add_field_row(middle_layout, "confidence_threshold_record", "Próg zdarzenia/nagrania (RECORD)", self.conf_record_spin)
        middle_layout.addRow("", self.show_legacy_conf_chk)
        self._add_field_row(middle_layout, "sensitivity_profile", "Profil sensitivity", self.sensitivity_profile_combo)
        self._add_field_row(middle_layout, "draw_overlays", "Rysuj nakładki", self.draw_chk)
        self._add_field_row(middle_layout, "enable_detection", "Wykrywaj obiekty", self.detect_chk)
        self._add_field_row(middle_layout, "enable_recording", "Nagrywaj detekcje", self.record_chk)
        self._add_field_row(middle_layout, "detection_hours", "Godziny detekcji", self.hours_edit)
        self._add_field_row(middle_layout, "visible_classes", "Widoczne klasy", self.visible_edit)
        self._add_field_row(middle_layout, "record_classes", "Klasy nagrywane", self.record_edit)
        middle_layout.addRow("", self.record_classes_hint)

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
        self._add_field_row(right_layout, "degirum_device_override_enabled", "", self.degirum_device_override_enabled_chk)
        self._add_field_row(right_layout, "degirum_device_override", "Urządzenie DeGirum", self.degirum_device_override_combo)

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
        self.calibration_btn = QPushButton("Kreator kalibracji 24h")
        self.btn_ok = QPushButton("Zapisz")
        self.btn_cancel = QPushButton("Anuluj")
        controls.addWidget(self.test_btn)
        controls.addWidget(self.test_status)
        controls.addWidget(self.calibration_btn)
        controls.addStretch(1)
        controls.addWidget(self.btn_cancel)
        controls.addWidget(self.btn_ok)
        root.addLayout(controls)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_path.clicked.connect(self._choose_path)
        self.test_btn.clicked.connect(self._test_source)
        self.sensitivity_profile_combo.currentTextChanged.connect(self._on_sensitivity_profile_changed)
        self.show_legacy_conf_chk.toggled.connect(self._set_legacy_conf_visible)
        self.record_edit.textChanged.connect(self._update_record_classes_hint)
        self.model_combo.currentTextChanged.connect(self._update_record_classes_hint)
        self.calibration_btn.clicked.connect(self._open_calibration_wizard)

        self.result_camera = None
        if camera:
            self.load_camera(camera)
        else:
            self._on_type_change(self.type_combo.currentText())
            self.preview_fps_main_spin.setValue(float(DEFAULT_PREVIEW_FPS_MAIN))
            self.preview_fps_thumb_spin.setValue(float(DEFAULT_PREVIEW_FPS_THUMB))
            self.preview_pause_chk.setChecked(bool(DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN))
            self.info_overlay_chk.setChecked(bool(DEFAULT_SHOW_CAMERA_INFO_OVERLAY))
            self.sensitivity_profile_combo.setCurrentText(DEFAULT_SENSITIVITY_PROFILE)
        self._set_legacy_conf_visible(False)
        self._update_record_classes_hint()

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

    def _set_legacy_conf_visible(self, visible: bool) -> None:
        row = self._field_rows.get("confidence_threshold")
        if not row:
            return
        label, input_widget = row
        label.setVisible(bool(visible))
        input_widget.setVisible(bool(visible))

    @staticmethod
    def _csv_to_lower_set(raw_text: str) -> set[str]:
        return {part.strip().lower() for part in str(raw_text).split(",") if part.strip()}

    def _load_model_labels(self, model_name: str) -> set[str] | None:
        name = str(model_name or "").strip()
        if not name:
            return None
        if name in self._model_labels_cache:
            return self._model_labels_cache[name]
        model_dir = MODELS_PATH / name
        labels_data = None
        try:
            for labels_file in sorted(model_dir.glob("labels*.json")):
                with labels_file.open("r", encoding="utf-8") as fh:
                    labels_data = json.load(fh)
                break
        except Exception:
            labels_data = None
        labels: set[str] = set()
        if isinstance(labels_data, dict):
            labels = {str(v).strip().lower() for v in labels_data.values() if str(v).strip()}
        elif isinstance(labels_data, list):
            labels = {str(v).strip().lower() for v in labels_data if str(v).strip()}
        loaded = labels or None
        self._model_labels_cache[name] = loaded
        return loaded

    def _update_record_classes_hint(self) -> None:
        record_classes = self._csv_to_lower_set(self.record_edit.text())
        model_labels = self._load_model_labels(self.model_combo.currentText())
        if not record_classes or not model_labels:
            self.record_classes_hint.clear()
            return
        overlap = sorted(record_classes.intersection(model_labels))
        if overlap:
            self.record_classes_hint.setText(f"Dopasowane klasy modelu: {', '.join(overlap)}")
            self.record_classes_hint.setStyleSheet("color:#7fd18c;")
            return
        preview = ", ".join(sorted(model_labels)[:8])
        suffix = " …" if len(model_labels) > 8 else ""
        self.record_classes_hint.setText(
            "⚠️ Brak przecięcia record_classes z etykietami modelu. "
            f"Przykładowe etykiety modelu: {preview}{suffix}"
        )
        self.record_classes_hint.setStyleSheet("color:#f7d26a;")

    def _set_help_panel_text(self, key: str):
        if not hasattr(self, "help_panel"):
            return
        self.help_panel.setPlainText(CAMERA_SETTING_TOOLTIPS.get(key, ""))

    def _detected_degirum_devices(self) -> list[str]:
        parent = self.parent()
        candidates = []
        if parent is not None:
            candidates.extend(getattr(parent, "degirum_available_devices", []) or [])
            parent_cfg = getattr(parent, "config", None)
            if isinstance(parent_cfg, dict):
                candidates.extend(parent_cfg.get("degirum_available_devices", []) or [])
        normalized = []
        seen = set()
        for item in candidates:
            value = config_module.normalize_degirum_device_selection(item)
            if not value or value in {"inherit", "auto", "cpu", "gpu"} or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def _refresh_degirum_override_options(self) -> None:
        current_value = config_module.normalize_degirum_device_selection(
            self.degirum_device_override_combo.currentData() or "inherit",
            allow_inherit=True,
        )
        options: list[tuple[str, str]] = [
            (self._degirum_device_label_map["inherit"], "inherit"),
            (self._degirum_device_label_map["cpu"], "cpu"),
            (self._degirum_device_label_map["gpu"], "gpu"),
        ]
        for device_id in self._detected_degirum_devices():
            options.append((f"Urządzenie: {device_id}", device_id))
        self.degirum_device_override_combo.clear()
        for label, value in options:
            self.degirum_device_override_combo.addItem(label, value)
        idx = self.degirum_device_override_combo.findData(current_value)
        self.degirum_device_override_combo.setCurrentIndex(idx if idx >= 0 else 0)

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

    def _set_sensitivity_inputs_enabled(self, enabled: bool) -> None:
        for widget in (
            self.conf_draw_spin,
            self.conf_record_spin,
            self.required_hits_spin,
            self.required_misses_spin,
            self.min_record_seconds_spin,
        ):
            widget.setEnabled(bool(enabled))

    def _on_sensitivity_profile_changed(self, profile_name: str) -> None:
        profile = str(profile_name or "custom")
        if profile == "custom":
            self._set_sensitivity_inputs_enabled(True)
            return
        values = SENSITIVITY_PROFILES.get(profile, {})
        if values:
            self.conf_draw_spin.setValue(float(values.get("confidence_threshold_draw", self.conf_draw_spin.value())))
            self.conf_record_spin.setValue(float(values.get("confidence_threshold_record", self.conf_record_spin.value())))
            self.required_hits_spin.setValue(int(values.get("required_hits_to_start_recording", self.required_hits_spin.value())))
            self.required_misses_spin.setValue(int(values.get("required_misses_to_end_detection", self.required_misses_spin.value())))
            self.min_record_seconds_spin.setValue(int(values.get("min_record_seconds", self.min_record_seconds_spin.value())))
        self._set_sensitivity_inputs_enabled(False)

    def _open_calibration_wizard(self) -> None:
        telemetry = dict(getattr(self, "_camera_runtime_telemetry", {}) or {})
        samples = int(telemetry.get("calibration_sample_count", 0))
        duration_h = float(telemetry.get("calibration_duration_hours", 0.0))
        suggestion = telemetry.get("suggested_record_threshold")
        if duration_h < 24.0 or suggestion is None:
            QMessageBox.information(
                self,
                "Kreator kalibracji",
                "Kalibracja wymaga minimum 24h danych telemetrycznych.\n"
                f"Aktualnie: {duration_h:.1f}h, próbek: {samples}.",
            )
            return
        suggestion_f = float(suggestion)
        delta = suggestion_f - float(self.conf_record_spin.value())
        action = "podnieść" if delta > 0 else "obniżyć"
        msg = (
            "Na podstawie 24h statystyk sugerowany jest nowy próg nagrywania.\n\n"
            f"Obecny próg: {float(self.conf_record_spin.value()):.2f}\n"
            f"Sugerowany próg: {suggestion_f:.2f} ({action} o {abs(delta):.2f})\n"
            f"Średnie confidence: {float(telemetry.get('avg_confidence', 0.0)):.3f}\n"
            f"False-positive proxy: {float(telemetry.get('false_positive_proxy_rate', 0.0)):.1%}"
        )
        if QMessageBox.question(
            self,
            "Kreator kalibracji",
            msg + "\n\nZastosować sugestię?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        ) == QMessageBox.Yes:
            self.conf_record_spin.setValue(suggestion_f)
            self.sensitivity_profile_combo.setCurrentText("custom")

    def load_camera(self, cam):
        cam = cam or {}
        self._refresh_degirum_override_options()
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
        self.detection_fps_limit_spin.setValue(int(cam.get("detection_fps_limit", cam.get("fps", DEFAULT_DETECTION_FPS_LIMIT))))
        self.priority_combo.setCurrentText(str(cam.get("camera_priority", "normal")))
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
        override_enabled = bool(cam.get("degirum_device_override_enabled", False))
        self.degirum_device_override_enabled_chk.setChecked(override_enabled)
        override_value = config_module.normalize_degirum_device_selection(
            cam.get("degirum_device_override", "inherit"),
            allow_inherit=True,
        )
        idx = self.degirum_device_override_combo.findData(override_value)
        if idx < 0 and override_value not in {"inherit", "cpu", "gpu"}:
            self.degirum_device_override_combo.addItem(f"Urządzenie: {override_value}", override_value)
            idx = self.degirum_device_override_combo.findData(override_value)
        self.degirum_device_override_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.degirum_device_override_combo.setEnabled(override_enabled)
        self._camera_runtime_telemetry = dict(cam.get("runtime_telemetry", {}) or {})
        profile_name = str(cam.get("sensitivity_profile", infer_sensitivity_profile(cam)) or DEFAULT_SENSITIVITY_PROFILE)
        if profile_name not in {"balanced", "high_recall", "high_precision", "custom"}:
            profile_name = "custom"
        self.sensitivity_profile_combo.setCurrentText(profile_name)
        self._on_sensitivity_profile_changed(profile_name)
        self._update_record_classes_hint()

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
            "detection_fps_limit": int(self.detection_fps_limit_spin.value()),
            "camera_priority": self.priority_combo.currentText(),
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
            "sensitivity_profile": self.sensitivity_profile_combo.currentText(),
            "preview_fps_main": float(self.preview_fps_main_spin.value()),
            "preview_fps_thumb": float(self.preview_fps_thumb_spin.value()),
            "preview_pause_when_hidden": self.preview_pause_chk.isChecked(),
            "degirum_device_override_enabled": self.degirum_device_override_enabled_chk.isChecked(),
            "degirum_device_override": config_module.normalize_degirum_device_selection(
                self.degirum_device_override_combo.currentData() or "inherit",
                allow_inherit=True,
            ),
        }
        profile_name = str(cam.get("sensitivity_profile", "custom") or "custom")
        if profile_name != "custom":
            apply_sensitivity_profile(cam, profile_name, force=True)
        model_labels = self._load_model_labels(cam.get("model", ""))
        record_classes_lower = {str(c).strip().lower() for c in cam.get("record_classes", []) if str(c).strip()}
        if model_labels and record_classes_lower and not record_classes_lower.intersection(model_labels):
            QMessageBox.warning(
                self,
                "Klasy nagrywania",
                "Brak przecięcia klas nagrywanych z etykietami modelu.\n"
                "Nagrywanie zdarzeń może się nie uruchamiać dla tej konfiguracji.",
            )
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
    def __init__(self, cameras, start_cb, stop_cb, test_cb, settings_cb, delete_cb, load_balancer_cb=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zarządzanie kamerami")
        self.cameras = cameras
        self.start_cb = start_cb
        self.stop_cb = stop_cb
        self.test_cb = test_cb
        self.settings_cb = settings_cb
        self.delete_cb = delete_cb
        self.load_balancer_cb = load_balancer_cb

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
        self.btn_load_balancer = QPushButton("Auto balans")
        self.btn_delete = QPushButton("Usuń")
        for b in (self.btn_start, self.btn_stop, self.btn_test, self.btn_copy, self.btn_settings, self.btn_load_balancer, self.btn_delete):
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
        self.btn_load_balancer.clicked.connect(lambda: self.load_balancer_cb() if callable(self.load_balancer_cb) else None)
        self.btn_delete.clicked.connect(lambda: self.delete_cb(self.combo.currentIndex()))
        self.btn_load_balancer.setEnabled(callable(self.load_balancer_cb))

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
        self._dedupe_window_seconds = 5.0
        self._last_message_ts: dict[tuple[str, str, str], float] = {}
        self.entry_signal.connect(self._deliver, Qt.QueuedConnection)

    def set_target(self, target_window) -> None:
        self._target_window = target_window

    def _should_suppress_duplicate(self, group: str, source: str, message: str) -> bool:
        if group != "performance" or source not in {"worker", "ui", "heartbeat-watchdog"}:
            return False
        if "metryk" not in message.lower() and "metrics" not in message.lower():
            return False
        now = time.monotonic()
        key = (group, source, message)
        last_ts = self._last_message_ts.get(key)
        self._last_message_ts[key] = now
        if last_ts is None:
            return False
        return (now - last_ts) < self._dedupe_window_seconds

    def log(self, group: str, message: str, camera: str = "", source: str = "", level: str = "INFO", details: str = "", traceback_text: str = "", action: str = "", **kwargs) -> None:
        legacy_traceback = kwargs.pop("traceback", "")
        if legacy_traceback and not traceback_text:
            traceback_text = str(legacy_traceback)
        if traceback_text and traceback_text not in details:
            details = f"{details}\n\n{traceback_text}".strip() if details else traceback_text
        if not is_log_entry_enabled(group=group, level=level, source=source):
            return
        if self._should_suppress_duplicate(group=group, source=source, message=message):
            return
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


class ModelLoadThread(QThread):
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, *, kwargs: dict[str, object], timeout_s: float = 25.0, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.kwargs = dict(kwargs)
        self.timeout_s = float(timeout_s)

    def run(self) -> None:
        try:
            model = load_model_with_timeout(dg, timeout_s=self.timeout_s, **self.kwargs)
            self.loaded.emit(model)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, cameras):
        super().__init__()
        self.setWindowTitle("AI Monitoring – PyQt5 (pełne GUI)")
        self.resize(1400, 900)
        self.config = load_config()

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

        self.cameras = list(cameras or self.config.get("cameras", []))
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
        btn_settings.setToolTip("Ustawienia główne")

        btn_auto_balance = QToolButton()
        btn_auto_balance.setIcon(QIcon(str(ICON_DIR / "sliders.svg")))
        btn_auto_balance.setIconSize(QSize(50, 50))
        btn_auto_balance.clicked.connect(self.open_system_load_balancer_dialog)
        btn_auto_balance.setToolTip("Auto-balans obciążenia kamer")

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
            btn_auto_balance,
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
        controls_layout.addWidget(btn_auto_balance)
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
        self.model_cache: dict[tuple[str, str], object] = {}
        self.camera_list.currentRowChanged.connect(self.switch_camera)

        self.alert_list.open_video.connect(self.open_video_file)

        # FPS liczniki i HUD stan
        self._fps_times = {}
        self._last_main_frame = {}
        self._last_thumb_frame = {}
        self._last_status = {}
        self._last_error = {}
        self._last_fps_text = {}
        self._worker_diag: dict[str, dict[str, object]] = {}
        self.worker_status: dict[str, dict[str, object]] = {}
        self.last_render_time = 0.0
        self.preview_fps_main = float(self.config.get("preview_fps_main", DEFAULT_PREVIEW_FPS_MAIN)) if hasattr(self, "config") else DEFAULT_PREVIEW_FPS_MAIN
        self.preview_fps_grid = float(self.config.get("preview_fps_grid", DEFAULT_PREVIEW_FPS_GRID)) if hasattr(self, "config") else DEFAULT_PREVIEW_FPS_GRID
        self.preview_fps_thumb = float(self.config.get("preview_fps_thumb", DEFAULT_PREVIEW_FPS_THUMB)) if hasattr(self, "config") else DEFAULT_PREVIEW_FPS_THUMB
        self.preview_pause_when_hidden = bool(self.config.get("preview_pause_when_hidden", DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN)) if hasattr(self, "config") else DEFAULT_PREVIEW_PAUSE_WHEN_HIDDEN
        self.preview_main_max_width = int(self.config.get("preview_main_max_width", DEFAULT_PREVIEW_MAIN_MAX_WIDTH)) if hasattr(self, "config") else DEFAULT_PREVIEW_MAIN_MAX_WIDTH
        self.preview_main_max_height = int(self.config.get("preview_main_max_height", DEFAULT_PREVIEW_MAIN_MAX_HEIGHT)) if hasattr(self, "config") else DEFAULT_PREVIEW_MAIN_MAX_HEIGHT
        self.preview_grid_max_width = int(self.config.get("preview_grid_max_width", DEFAULT_PREVIEW_GRID_MAX_WIDTH)) if hasattr(self, "config") else DEFAULT_PREVIEW_GRID_MAX_WIDTH
        self.preview_grid_max_height = int(self.config.get("preview_grid_max_height", DEFAULT_PREVIEW_GRID_MAX_HEIGHT)) if hasattr(self, "config") else DEFAULT_PREVIEW_GRID_MAX_HEIGHT
        self.preview_thumb_max_width = int(self.config.get("preview_thumb_max_width", DEFAULT_PREVIEW_THUMB_MAX_WIDTH)) if hasattr(self, "config") else DEFAULT_PREVIEW_THUMB_MAX_WIDTH
        self.preview_thumb_max_height = int(self.config.get("preview_thumb_max_height", DEFAULT_PREVIEW_THUMB_MAX_HEIGHT)) if hasattr(self, "config") else DEFAULT_PREVIEW_THUMB_MAX_HEIGHT
        self.quality_performance_preset = str(self.config.get("quality_performance_preset", DEFAULT_QUALITY_PERFORMANCE_PRESET))
        self.overload_protection_enabled = bool(self.config.get("overload_protection_enabled", DEFAULT_OVERLOAD_PROTECTION_ENABLED)) if hasattr(self, "config") else DEFAULT_OVERLOAD_PROTECTION_ENABLED
        self.overload_min_camera_count = int(self.config.get("overload_min_camera_count", DEFAULT_OVERLOAD_MIN_CAMERA_COUNT)) if hasattr(self, "config") else DEFAULT_OVERLOAD_MIN_CAMERA_COUNT
        self.overload_camera_count_threshold = int(self.config.get("overload_camera_count_threshold", DEFAULT_OVERLOAD_CAMERA_COUNT_THRESHOLD)) if hasattr(self, "config") else DEFAULT_OVERLOAD_CAMERA_COUNT_THRESHOLD
        self.overload_reduce_thumb_preview_fps = float(self.config.get("overload_reduce_thumb_preview_fps", DEFAULT_OVERLOAD_REDUCE_THUMB_PREVIEW_FPS)) if hasattr(self, "config") else DEFAULT_OVERLOAD_REDUCE_THUMB_PREVIEW_FPS
        self.overload_reduce_detect_fps_factor = float(self.config.get("overload_reduce_detect_fps_factor", DEFAULT_OVERLOAD_REDUCE_DETECT_FPS_FACTOR)) if hasattr(self, "config") else DEFAULT_OVERLOAD_REDUCE_DETECT_FPS_FACTOR
        self.overload_disable_nonessential_overlays = bool(self.config.get("overload_disable_nonessential_overlays", DEFAULT_OVERLOAD_DISABLE_NONESSENTIAL_OVERLAYS)) if hasattr(self, "config") else DEFAULT_OVERLOAD_DISABLE_NONESSENTIAL_OVERLAYS
        self.overload_enter_debounce_seconds = float(self.config.get("overload_enter_debounce_seconds", DEFAULT_OVERLOAD_ENTER_DEBOUNCE_SECONDS)) if hasattr(self, "config") else DEFAULT_OVERLOAD_ENTER_DEBOUNCE_SECONDS
        self.overload_exit_debounce_seconds = float(self.config.get("overload_exit_debounce_seconds", DEFAULT_OVERLOAD_EXIT_DEBOUNCE_SECONDS)) if hasattr(self, "config") else DEFAULT_OVERLOAD_EXIT_DEBOUNCE_SECONDS
        self.overload_max_ui_render_ms = float(self.config.get("overload_max_ui_render_ms", DEFAULT_OVERLOAD_MAX_UI_RENDER_MS)) if hasattr(self, "config") else DEFAULT_OVERLOAD_MAX_UI_RENDER_MS
        self.overload_max_queue_size = int(self.config.get("overload_max_queue_size", DEFAULT_OVERLOAD_MAX_QUEUE_SIZE)) if hasattr(self, "config") else DEFAULT_OVERLOAD_MAX_QUEUE_SIZE
        self.overload_max_preview_bandwidth_mbps = float(self.config.get("overload_max_preview_bandwidth_mbps", DEFAULT_OVERLOAD_MAX_PREVIEW_BANDWIDTH_MBPS)) if hasattr(self, "config") else DEFAULT_OVERLOAD_MAX_PREVIEW_BANDWIDTH_MBPS
        self.overload_safety_threshold_pct = int(self.config.get("overload_safety_threshold_pct", 85)) if hasattr(self, "config") else 85
        self.degirum_device_mode = config_module.normalize_degirum_device_selection(
            self.config.get("degirum_device_mode", "auto")
        )
        self.degirum_preferred_device = config_module.normalize_degirum_device_selection(
            self.config.get("degirum_preferred_device", "auto")
        )
        self.degirum_auto_select_best = bool(self.config.get("degirum_auto_select_best", True))
        raw_devices = self.config.get("degirum_available_devices", []) if hasattr(self, "config") else []
        if not isinstance(raw_devices, list):
            raw_devices = [raw_devices]
        self.degirum_available_devices = [
            normalized
            for item in raw_devices
            if (normalized := config_module.normalize_degirum_device_selection(item)) not in {"auto", "inherit"}
        ]
        self.degirum_last_benchmark = dict(self.config.get("degirum_last_benchmark", {})) if hasattr(self, "config") else {}
        self._degirum_supported_types_cache: dict[tuple[str, str], list[str]] = {}
        self.overload_level = 0
        self.overload_mode_active = False
        self._overload_last_change_ts = 0.0
        self._ui_render_ms_by_camera: dict[str, float] = {}
        self._ui_render_stage_ms_by_camera: dict[str, dict[str, float]] = {}
        self._performance_log_last_ts_by_camera: dict[str, float] = {}
        self._base_performance_log_interval_s = float(
            self.config.get("performance_log_interval_s", DEFAULT_PERFORMANCE_LOG_INTERVAL_S)
        )
        self._performance_log_interval_s = self._base_performance_log_interval_s
        self.performance_diagnostics_enabled = bool(
            self.config.get("performance_diagnostics_enabled", DEFAULT_PERFORMANCE_DIAGNOSTICS_ENABLED)
        )
        self._performance_log_delta_thresholds = {
            "capture_fps": 1.0,
            "infer_fps": 1.0,
            "cpu_percent": 6.0,
            "queue_size": 3.0,
            "dropped_frames": 2.0,
        }
        self._performance_log_snapshot_by_camera: dict[str, dict[str, float]] = {}
        self.grid_preview_quality = str(self.config.get("grid_preview_quality", DEFAULT_GRID_PREVIEW_QUALITY)) if hasattr(self, "config") else DEFAULT_GRID_PREVIEW_QUALITY
        self.config_watchdog_enabled = bool(self.config.get("config_watchdog_enabled", DEFAULT_CONFIG_WATCHDOG_ENABLED))
        self.config_watchdog_eval_seconds = float(self.config.get("config_watchdog_eval_seconds", DEFAULT_CONFIG_WATCHDOG_EVAL_SECONDS))
        self.config_watchdog_drop_delta_threshold = int(self.config.get("config_watchdog_drop_delta_threshold", DEFAULT_CONFIG_WATCHDOG_DROP_DELTA_THRESHOLD))
        self.config_watchdog_queue_delta_threshold = int(self.config.get("config_watchdog_queue_delta_threshold", DEFAULT_CONFIG_WATCHDOG_QUEUE_DELTA_THRESHOLD))
        self._config_watchdog_state: dict[str, object] | None = None
        self._config_watchdog_rollback_running = False
        self._stable_config_snapshot = self._build_runtime_config()
        self._preview_cache: dict[tuple[int, str, int, int, int, int, str], QPixmap] = {}
        self._last_thumb_update_ts: dict[int, float] = {}
        self._thumb_update_interval_s = 1.0
        self._last_grid_update_ts: dict[int, float] = {}
        self._grid_update_interval_s = 1.0
        self._refresh_preview_intervals()
        self._recordings_browser_open = False
        self._hud_interval_s = 0.35
        self._last_hud_render_ts = 0.0
        self._hud_cache_qimg: QImage | None = None
        self._hud_cache_key: tuple | None = None
        self._letterbox_geometry_cache: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {}
        self._canvas_bg_cache: dict[tuple[int, int], np.ndarray] = {}
        self._main_view_geometry_cache: tuple[int, int] | None = None
        self._ui_grid_ms_history_by_camera: dict[str, deque[float]] = {}
        self._ui_grid_frame_times_by_camera: dict[str, deque[float]] = {}

        self.diag_panel = QLabel("Diagnostyka (debug): brak danych")
        self.diag_panel.setStyleSheet("color: #dddddd; background: #111; padding: 8px; border: 1px solid #333;")
        self.diag_panel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.diag_panel.setMinimumHeight(90)
        self.diag_panel.setVisible(False)
        self.diag_timer = QTimer(self)
        self.diag_timer.timeout.connect(self._refresh_diag_panel)
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

    def _refresh_preview_intervals(self) -> None:
        self._render_interval_s = 1.0 / max(1.0, float(self.preview_fps_main))
        self._thumb_update_interval_s = 1.0 / max(0.5, float(self.preview_fps_thumb))
        self._grid_update_interval_s = 1.0 / max(0.5, float(self.preview_fps_grid))

    def _log_exception(self, group: str, message: str, exc: BaseException | None = None, **kwargs) -> None:
        APP_LOG_BRIDGE.exception(group, message, exc=exc, **kwargs)

    def _build_runtime_config(self) -> dict:
        cfg = load_config()
        cfg["cameras"] = list(self.cameras)
        cfg["preview_fps_main"] = float(self.preview_fps_main)
        cfg["preview_fps_grid"] = float(self.preview_fps_grid)
        cfg["preview_fps_thumb"] = float(self.preview_fps_thumb)
        cfg["preview_main_max_width"] = int(self.preview_main_max_width)
        cfg["preview_main_max_height"] = int(self.preview_main_max_height)
        cfg["preview_grid_max_width"] = int(self.preview_grid_max_width)
        cfg["preview_grid_max_height"] = int(self.preview_grid_max_height)
        cfg["preview_thumb_max_width"] = int(self.preview_thumb_max_width)
        cfg["preview_thumb_max_height"] = int(self.preview_thumb_max_height)
        cfg["quality_performance_preset"] = str(self.quality_performance_preset)
        cfg["grid_preview_quality"] = str(self.grid_preview_quality)
        cfg["config_watchdog_enabled"] = bool(self.config_watchdog_enabled)
        cfg["config_watchdog_eval_seconds"] = float(self.config_watchdog_eval_seconds)
        cfg["config_watchdog_drop_delta_threshold"] = int(self.config_watchdog_drop_delta_threshold)
        cfg["config_watchdog_queue_delta_threshold"] = int(self.config_watchdog_queue_delta_threshold)
        cfg["performance_log_interval_s"] = float(self._base_performance_log_interval_s)
        cfg["performance_diagnostics_enabled"] = bool(self.performance_diagnostics_enabled)
        cfg["overload_protection_enabled"] = bool(self.overload_protection_enabled)
        cfg["overload_min_camera_count"] = int(self.overload_min_camera_count)
        cfg["overload_camera_count_threshold"] = int(self.overload_camera_count_threshold)
        cfg["overload_reduce_thumb_preview_fps"] = float(self.overload_reduce_thumb_preview_fps)
        cfg["overload_reduce_detect_fps_factor"] = float(self.overload_reduce_detect_fps_factor)
        cfg["overload_disable_nonessential_overlays"] = bool(self.overload_disable_nonessential_overlays)
        cfg["overload_enter_debounce_seconds"] = float(self.overload_enter_debounce_seconds)
        cfg["overload_exit_debounce_seconds"] = float(self.overload_exit_debounce_seconds)
        cfg["overload_max_ui_render_ms"] = float(self.overload_max_ui_render_ms)
        cfg["overload_max_queue_size"] = int(self.overload_max_queue_size)
        cfg["overload_max_preview_bandwidth_mbps"] = float(self.overload_max_preview_bandwidth_mbps)
        cfg["overload_safety_threshold_pct"] = int(self.overload_safety_threshold_pct)
        cfg["degirum_device_mode"] = str(self.degirum_device_mode)
        cfg["degirum_preferred_device"] = str(self.degirum_preferred_device)
        cfg["degirum_auto_select_best"] = bool(self.degirum_auto_select_best)
        cfg["degirum_available_devices"] = list(self.degirum_available_devices)
        cfg["degirum_last_benchmark"] = dict(self.degirum_last_benchmark)
        return cfg

    def _heartbeat_perf_changed(self, camera_name: str, payload: dict) -> bool:
        current = {
            "capture_fps": float(payload.get("capture_fps", 0.0)),
            "infer_fps": float(payload.get("infer_fps", 0.0)),
            "cpu_percent": float(payload.get("cpu_percent", 0.0)),
            "queue_size": float(payload.get("queue_size", 0.0)),
            "dropped_frames": float(payload.get("dropped_frames", 0.0)),
        }
        previous = self._performance_log_snapshot_by_camera.get(camera_name)
        self._performance_log_snapshot_by_camera[camera_name] = current
        if not previous:
            return True
        for key, threshold in self._performance_log_delta_thresholds.items():
            if abs(current[key] - float(previous.get(key, 0.0))) >= threshold:
                return True
        return False

    def _save_runtime_config(self) -> dict:
        cfg = self._build_runtime_config()
        save_config(cfg)
        self.config = cfg
        return cfg

    def _metrics_baseline(self) -> dict[str, float]:
        queue_total = sum(int(st.get("queue_size", 0)) for st in self.worker_status.values())
        dropped_total = sum(int(st.get("dropped_frames", 0)) for st in self.worker_status.values())
        cameras = max(1, len(self.worker_status))
        return {
            "queue_avg": float(queue_total / cameras),
            "dropped_avg": float(dropped_total / cameras),
        }

    def _start_config_watchdog(self, previous_cfg: dict, candidate_cfg: dict, reason: str) -> None:
        if not self.config_watchdog_enabled or self._config_watchdog_rollback_running:
            self._stable_config_snapshot = candidate_cfg
            return
        self._config_watchdog_state = {
            "started_ts": time.monotonic(),
            "reason": str(reason),
            "previous_cfg": dict(previous_cfg),
            "candidate_cfg": dict(candidate_cfg),
            "baseline": self._metrics_baseline(),
        }

    def _run_config_change_watchdog(self) -> None:
        state = self._config_watchdog_state
        if not state or self._config_watchdog_rollback_running:
            return
        elapsed = time.monotonic() - float(state.get("started_ts", 0.0))
        baseline = dict(state.get("baseline", {}) or {})
        current = self._metrics_baseline()
        queue_delta = current["queue_avg"] - float(baseline.get("queue_avg", 0.0))
        dropped_delta = current["dropped_avg"] - float(baseline.get("dropped_avg", 0.0))
        if queue_delta >= self.config_watchdog_queue_delta_threshold or dropped_delta >= self.config_watchdog_drop_delta_threshold:
            previous_cfg = dict(state.get("previous_cfg", {}) or {})
            if previous_cfg:
                self._config_watchdog_rollback_running = True
                self._log_warning(
                    "settings",
                    "config watchdog rollback",
                    source="config-watchdog",
                    details=f"queue_delta={queue_delta:.2f} dropped_delta={dropped_delta:.2f}",
                )
                save_config(previous_cfg)
                self.config = previous_cfg
                self.cameras = list(previous_cfg.get("cameras", []))
                self.restart_workers_and_ui()
                self._config_watchdog_rollback_running = False
                self._stable_config_snapshot = previous_cfg
            self._config_watchdog_state = None
            return
        if elapsed >= float(self.config_watchdog_eval_seconds):
            self._stable_config_snapshot = dict(state.get("candidate_cfg", {}) or {})
            self._config_watchdog_state = None

    def _run_watchdogs(self) -> None:
        self._run_config_change_watchdog()
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

    def update_log_filters(self, filters: dict[str, list[str]]) -> None:
        normalized = normalize_log_filters(filters)
        config_module.LOG_FILTERS = dict(normalized)
        cfg = load_config()
        cfg["cameras"] = self.cameras
        cfg["log_filters"] = dict(normalized)
        save_config(cfg)

    def current_log_filters(self) -> dict[str, list[str]]:
        active = normalize_log_filters(getattr(config_module, "LOG_FILTERS", {}))
        return {
            "groups": list(active.get("groups", DEFAULT_LOG_FILTER_GROUPS)),
            "levels": list(active.get("levels", DEFAULT_LOG_FILTER_LEVELS)),
            "sources": list(active.get("sources", DEFAULT_LOG_FILTER_SOURCES)),
        }

    def open_camera_settings(self):
        self.log_window.add_entry("settings", "otwarto ustawienia kamer")
        dlg = CameraSettingsDialog(
            self.cameras,
            start_cb=self.start_camera,
            stop_cb=self.stop_camera,
            test_cb=self.test_camera,
            settings_cb=self.camera_settings,
            delete_cb=self.delete_camera,
            load_balancer_cb=self.open_system_load_balancer_dialog,
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

    def _build_model_cache_key(self, model_name: str, device_config: object = None) -> tuple[str, str]:
        model_key = str(model_name).strip()
        effective_device = str(device_config or "").strip() or "CPU"
        return model_key, effective_device

    def _load_model_with_progress(self, *, load_kwargs: dict[str, object], camera_name: str, model_name: str):
        progress = QProgressDialog(
            f"Ładowanie modelu '{model_name}' dla kamery '{camera_name}'...",
            "",
            0,
            0,
            self,
        )
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(True)
        progress.setCancelButton(None)
        progress.setValue(0)
        progress.show()

        holder: dict[str, object] = {}
        canceled_by_user = False

        def _on_cancel() -> None:
            nonlocal canceled_by_user
            canceled_by_user = True

        progress.canceled.connect(_on_cancel)

        thread = ModelLoadThread(kwargs=load_kwargs, timeout_s=25.0, parent=self)
        thread.loaded.connect(lambda model: holder.update({"model": model}))
        thread.failed.connect(lambda error: holder.update({"error": error}))
        thread.start()

        deadline = time.monotonic() + 25.5
        while thread.isRunning() and time.monotonic() < deadline:
            QApplication.processEvents()
            if canceled_by_user:
                break
            QThread.msleep(20)

        if thread.isRunning():
            thread.requestInterruption()
        thread.wait(100)
        progress.close()

        if canceled_by_user:
            raise TimeoutError("Ładowanie modelu anulowane przez użytkownika.")
        if "error" in holder:
            raise RuntimeError(str(holder["error"]))
        model = holder.get("model")
        if model is None:
            raise TimeoutError("Przekroczono limit czasu ładowania modelu (25s).")
        return model

    def _get_model(self, camera: dict):
        model_name = str(camera.get("model", DEFAULT_MODEL))
        logical = camera.get("degirum_device_override", self.config.get("degirum_preferred_device", "auto"))
        supported = get_model_supported_device_types(
            dg,
            model_name=model_name,
            zoo_url=MODELS_PATH / model_name,
            cache=self._degirum_supported_types_cache,
        )
        if not supported:
            self._log_warning(
                "warning",
                "supported_device_types is empty",
                source="detection",
                details=f"model={model_name} logical_device={logical}",
            )
        resolution = resolve_degirum_runtime_target(
            logical_selection=logical,
            supported_device_types=supported,
        )
        final_device = str(resolution.get("final_device_type") or "").strip()
        cache_key = None
        if final_device:
            cache_key = self._build_model_cache_key(model_name, final_device)
            if cache_key in self.model_cache:
                self.log_window.add_entry("application", f"model cache-hit: {model_name} ({cache_key[1]})")
                camera["effective_device_type"] = final_device
                return self.model_cache[cache_key]

        try:
            raw_kwargs = build_degirum_load_model_kwargs(
                model_name=model_name,
                inference_host_address="@local",
                zoo_url=MODELS_PATH / model_name,
                device_type=final_device or None,
            )
            load_kwargs = sanitize_degirum_load_model_kwargs(raw_kwargs)
            logger.debug("degirum _get_model raw kwargs=%s", raw_kwargs)
            logger.debug("degirum _get_model sanitized types=%s", {k: type(v).__name__ for k, v in load_kwargs.items()})
            logger.debug("degirum _get_model load attempt camera=%s model=%s", camera.get("name"), model_name)
            model = self._load_model_with_progress(
                load_kwargs=load_kwargs,
                camera_name=str(camera.get("name", "")),
                model_name=model_name,
            )
            logger.debug("degirum _get_model load success camera=%s model=%s", camera.get("name"), model_name)
            camera["effective_device_type"] = final_device
            self._log_info("detection", f"model loaded → {final_device}", camera=camera.get("name"))
            cache_variant = final_device
            if cache_variant:
                success_key = self._build_model_cache_key(model_name, cache_variant)
                self.model_cache[success_key] = model
            self.log_window.add_entry("application", f"model load: {model_name} ({cache_variant})")
            return model
        except Exception as init_error:
            self._log_error(
                "error",
                "degirum backend model init failure",
                source="detection",
                details=(
                    f"model={model_name} logical_device={logical} "
                    f"device_type={final_device or '<none>'} error={init_error}"
                ),
            )
        cpu_resolution = resolve_degirum_runtime_target(
            logical_selection="cpu",
            supported_device_types=supported,
        )
        cpu_type = str(cpu_resolution.get("final_device_type") or "").strip()

        cpu_candidates = [cpu_type] if cpu_type else ["CPU", "TFLITE/CPU", "OPENVINO/CPU"]
        if not cpu_type:
            self._log_warning(
                "warning",
                "cpu resolution returned empty final_device_type; using explicit cpu fallback chain",
                source="detection",
                details=f"model={model_name} requested={logical}",
            )

        self._log_warning(
            "warning",
            "fallback to cpu after backend error",
            source="detection",
            details=f"model={model_name} requested={logical} cpu_candidates={cpu_candidates}",
        )
        last_cpu_error = None
        for candidate in cpu_candidates:
            cpu_key = self._build_model_cache_key(model_name, candidate)
            if cpu_key in self.model_cache:
                self.log_window.add_entry("application", f"model cache-hit: {model_name} ({candidate})")
                camera["effective_device_type"] = candidate
                return self.model_cache[cpu_key]

            raw_kwargs = build_degirum_load_model_kwargs(
                model_name=model_name,
                inference_host_address="@local",
                zoo_url=MODELS_PATH / model_name,
                device_type=candidate,
            )
            load_kwargs = sanitize_degirum_load_model_kwargs(raw_kwargs)
            try:
                logger.debug("degirum _get_model cpu raw kwargs=%s", raw_kwargs)
                logger.debug("degirum _get_model cpu sanitized types=%s", {k: type(v).__name__ for k, v in load_kwargs.items()})
                logger.debug(
                    "degirum _get_model cpu fallback attempt camera=%s model=%s candidate=%s",
                    camera.get("name"),
                    model_name,
                    candidate,
                )
                model = self._load_model_with_progress(
                    load_kwargs=load_kwargs,
                    camera_name=str(camera.get("name", "")),
                    model_name=model_name,
                )
                logger.debug(
                    "degirum _get_model cpu fallback success camera=%s model=%s candidate=%s",
                    camera.get("name"),
                    model_name,
                    candidate,
                )
                camera["effective_device_type"] = candidate
                self.model_cache[cpu_key] = model
                self.log_window.add_entry("application", f"model load: {model_name} ({candidate})")
                return model
            except Exception as cpu_error:
                last_cpu_error = cpu_error
                logger.debug(
                    "degirum _get_model cpu fallback failure camera=%s model=%s candidate=%s error=%s",
                    camera.get("name"),
                    model_name,
                    candidate,
                    cpu_error,
                )
                continue

        self._log_error(
            "error",
            "degirum backend cpu fallback failed",
            source="detection",
            details=f"model={model_name} cpu_candidates={cpu_candidates} error={last_cpu_error}",
        )
        if last_cpu_error is not None:
            raise last_cpu_error
        raise RuntimeError("CPU fallback chain exhausted without explicit error.")

    def _apply_worker_preview_roles(self) -> None:
        selected_idx = self.camera_list.currentRow()
        grid_visible = bool(self.camera_grid.isVisible())
        list_visible = bool(self.camera_list.isVisible()) if hasattr(self.camera_list, "isVisible") else True
        for idx, worker in enumerate(self.workers):
            if not isinstance(worker, CameraWorker):
                continue
            if grid_visible:
                role = "grid"
            elif idx == selected_idx:
                role = "main"
            elif list_visible:
                role = "thumb"
            else:
                role = "hidden"
            worker.preview_fps_main = self.preview_fps_main
            worker.preview_fps_grid = self.preview_fps_grid
            worker.preview_fps_thumb = self.preview_fps_thumb
            worker.preview_pause_when_hidden = self.preview_pause_when_hidden
            worker.preview_main_max_width = self.preview_main_max_width
            worker.preview_main_max_height = self.preview_main_max_height
            worker.preview_grid_max_width = self.preview_grid_max_width
            worker.preview_thumb_max_width = self.preview_thumb_max_width
            worker.preview_grid_max_height = self.preview_grid_max_height
            worker.preview_thumb_max_height = self.preview_thumb_max_height
            worker.set_preview_role(role)

    def _evaluate_overload_mode(self) -> None:
        active_workers = [w for w in self.workers if isinstance(w, CameraWorker) and w.isRunning()]
        active_count = len(active_workers)
        recording_count = sum(1 for w in active_workers if w.recording)
        gui_load = sum(max(0.0, float(st.get("stream_fps", 0.0))) for st in self.worker_status.values())
        avg_ui_render_ms = 0.0
        if self.worker_status:
            avg_ui_render_ms = sum(max(0.0, float(st.get("ui_render_ms", 0.0))) for st in self.worker_status.values()) / max(1, len(self.worker_status))
        max_queue_size = max((int(st.get("queue_size", 0)) for st in self.worker_status.values()), default=0)
        preview_bandwidth_mbps = 0.0
        for stat in self.worker_status.values():
            emit_fps = max(0.0, float(stat.get("preview_emit_fps", 0.0)))
            stream_fps = max(1.0, float(stat.get("stream_fps", 1.0)))
            width = int(self.preview_thumb_max_width)
            height = int(self.preview_thumb_max_height)
            role = str(stat.get("preview_role", "thumb")).lower()
            if role == "main":
                width = int(self.preview_main_max_width)
                height = int(self.preview_main_max_height)
            elif role == "grid":
                width = int(self.preview_grid_max_width)
                height = int(self.preview_grid_max_height)
            compression_ratio = 0.15
            preview_bandwidth_mbps += (emit_fps / stream_fps) * width * height * 3.0 * compression_ratio * 8.0 / 1_000_000.0

        now_ts = time.monotonic()
        safety_factor = max(0.6, min(1.2, float(self.overload_safety_threshold_pct) / 100.0))
        effective_camera_threshold = max(1, int(round(self.overload_camera_count_threshold * safety_factor)))
        effective_max_ui_render_ms = max(4.0, float(self.overload_max_ui_render_ms) * safety_factor)
        effective_max_queue_size = max(2, int(round(self.overload_max_queue_size * safety_factor)))
        effective_max_preview_bandwidth_mbps = max(2.0, float(self.overload_max_preview_bandwidth_mbps) * safety_factor)
        overload_level, change_ts, reason = evaluate_overload_transition(
            now_ts=now_ts,
            active_camera_count=active_count,
            gui_load_fps=gui_load,
            recording_count=recording_count,
            currently_level=self.overload_level,
            last_change_ts=self._overload_last_change_ts,
            protection_enabled=self.overload_protection_enabled,
            min_camera_count=self.overload_min_camera_count,
            camera_threshold=effective_camera_threshold,
            load_per_camera_threshold=10.0,
            enter_debounce_seconds=self.overload_enter_debounce_seconds,
            exit_debounce_seconds=self.overload_exit_debounce_seconds,
            ui_render_ms=avg_ui_render_ms,
            max_ui_render_ms=effective_max_ui_render_ms,
            queue_size=max_queue_size,
            max_queue_size=effective_max_queue_size,
            preview_bandwidth_mbps=preview_bandwidth_mbps,
            max_preview_bandwidth_mbps=effective_max_preview_bandwidth_mbps,
        )
        self._overload_last_change_ts = change_ts

        if overload_level != self.overload_level:
            self.overload_level = overload_level
            self.overload_mode_active = overload_level > 0
            direction = "enter" if overload_level > 0 else "exit"
            self._log_info(
                "application",
                f"overload {direction}",
                source="app",
                details=(
                    f"reason={reason} level=L{overload_level} active_cameras={active_count} min_cameras={self.overload_min_camera_count} "
                    f"camera_threshold={effective_camera_threshold} gui_load={gui_load:.2f} ui_render_ms={avg_ui_render_ms:.2f} "
                    f"queue_size={max_queue_size} preview_bandwidth_mbps={preview_bandwidth_mbps:.2f} "
                    f"enter_debounce_s={self.overload_enter_debounce_seconds} exit_debounce_s={self.overload_exit_debounce_seconds} "
                    f"safety_threshold_pct={self.overload_safety_threshold_pct}"
                ),
            )

        profile = overload_level_profile(self.overload_level)
        self._performance_log_interval_s = max(self._base_performance_log_interval_s, float(profile.performance_log_interval_s))

        for idx, worker in enumerate(self.workers):
            if not isinstance(worker, CameraWorker) or not worker.isRunning():
                continue
            is_main = str(getattr(worker, "preview_role", "thumb")) == "main"
            camera_cfg = self.cameras[idx] if idx < len(self.cameras) else {}
            camera_priority = str(camera_cfg.get("camera_priority", "normal")).lower()
            priority_detect_scale = 1.0
            priority_thumb_scale = 1.0
            priority_resolution_floor = 0.3
            if camera_priority == "high":
                priority_detect_scale = 1.0
                priority_thumb_scale = 1.0
                priority_resolution_floor = 0.9
            elif camera_priority == "low":
                extra = {0: 1.0, 1: 0.8, 2: 0.65, 3: 0.5}.get(int(self.overload_level), 0.5)
                priority_detect_scale = extra
                priority_thumb_scale = extra
                priority_resolution_floor = 0.45
            role_level = int(self.overload_level)
            main_fps_factor = profile.main_preview_fps_factor if role_level >= 2 else 1.0
            thumb_fps_factor = profile.thumb_preview_fps_factor if role_level >= 1 else 1.0
            grid_fps_factor = profile.grid_preview_fps_factor if role_level >= 1 else 1.0

            worker.preview_fps_main = max(1.0, self.preview_fps_main * main_fps_factor)
            thumb_fps = max(0.5, self.preview_fps_thumb * thumb_fps_factor * priority_thumb_scale)
            grid_fps = max(0.5, self.preview_fps_grid * grid_fps_factor * priority_thumb_scale)

            preview_resolution_factor = profile.preview_resolution_factor
            if camera_priority == "high":
                preview_resolution_factor = max(0.9, preview_resolution_factor)
            else:
                preview_resolution_factor = max(priority_resolution_floor, preview_resolution_factor)

            detect_factor = 1.0
            if role_level >= 3 and not is_main and not worker.recording and camera_priority != "high":
                detect_factor = max(0.2, profile.detect_fps_factor * priority_detect_scale)

            worker.set_overload_state(
                overload_level=role_level,
                detect_fps_factor=detect_factor,
                thumb_preview_fps=thumb_fps,
                grid_preview_fps=grid_fps,
                disable_overlays=(role_level >= 2 and (self.overload_disable_nonessential_overlays or profile.disable_nonessential_overlays)),
                overlay_stride=(profile.overlay_stride if role_level >= 2 else 1),
                preview_resolution_factor=preview_resolution_factor,
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
        selection = resolve_effective_degirum_selection(cam, self.config)
        cam["degirum_device_override"] = selection.get("logical_selection", "auto")
        try:
            model = self._get_model(cam)
        except Exception as e:
            QMessageBox.warning(self, "Model", f"Nie udało się załadować modelu '{model_name}': {e}")
            self._log_error("error", f"model {model_name}: {e}", source="app", camera=str(cam.get("name", idx)))
            return
        cam_runtime = dict(cam)
        cam_runtime["performance_log_interval_s"] = float(self._base_performance_log_interval_s)
        cam_runtime["performance_diagnostics_enabled"] = bool(self.performance_diagnostics_enabled)
        w = CameraWorker(camera=cam_runtime, model=model, index=idx)
        w.preview_fps_main = self.preview_fps_main
        w.preview_fps_grid = self.preview_fps_grid
        w.preview_fps_thumb = self.preview_fps_thumb
        w.preview_pause_when_hidden = self.preview_pause_when_hidden
        w.preview_main_max_width = self.preview_main_max_width
        w.preview_main_max_height = self.preview_main_max_height
        w.preview_grid_max_width = self.preview_grid_max_width
        w.preview_thumb_max_width = self.preview_thumb_max_width
        w.preview_grid_max_height = self.preview_grid_max_height
        w.preview_thumb_max_height = self.preview_thumb_max_height
        w.main_preview_signal.connect(lambda frame, cam_idx: self.update_frame(frame, cam_idx, "main"))
        w.thumb_preview_signal.connect(lambda frame, cam_idx: self.update_frame(frame, cam_idx, "thumb"))
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
            w.main_preview_signal.disconnect()
        with suppress(Exception):
            w.thumb_preview_signal.disconnect()
        with suppress(Exception):
            w.alert_signal.disconnect(self.on_new_alert)
        with suppress(Exception):
            w.error_signal.disconnect(self._worker_error)
        with suppress(Exception):
            w.status_signal.disconnect(self._worker_status)

        self.workers[idx] = None
        self._last_main_frame.pop(idx, None)
        self._last_thumb_frame.pop(idx, None)
        self._last_fps_text[idx] = ""
        self._last_status[idx] = "Zatrzymano"
        self._last_error.pop(idx, None)
        cam_name = str(cam.get("name", idx))
        self.worker_status.pop(cam_name, None)
        self._worker_diag.pop(cam_name, None)
        self._invalidate_preview_cache(idx)
        self._last_thumb_update_ts.pop(idx, None)
        if hasattr(self.camera_grid, "update_pixmap"):
            with suppress(Exception):
                self.camera_grid.update_pixmap(idx, None)
        if hasattr(self.camera_list, "update_thumbnail_pixmap"):
            with suppress(Exception):
                self.camera_list.update_thumbnail_pixmap(idx, None)
        if idx == self.camera_list.currentRow():
            self._render_current()

        self._log_info("worker", "stop_camera completed", source="app", camera=str(cam.get("name", idx)), details=f"stopped={stopped}")
        self._evaluate_overload_mode()
        self._refresh_camera_status_indicators()
        idx = self.camera_list.currentRow()
        if 0 <= idx < len(self.cameras) and str(self.cameras[idx].get("name", idx)) == cam_name:
            self.last_render_time = 0.0
            self._render_current()


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
        payload.setdefault("inference_ms", 0.0)
        payload.setdefault("average_inference_ms", 0.0)
        payload.setdefault("preview_processing_ms", 0.0)
        payload.setdefault("recording_enqueue_ms", 0.0)
        payload.setdefault("recording_frames_written", 0)
        payload.setdefault("recording_queue_size", int(payload.get("queue_size", 0) or 0))
        payload.setdefault("recording_queue_peak", 0)
        payload.setdefault("dropped_frames", 0)
        payload.setdefault("overload_level", int(self.overload_level))
        payload["ui_render_ms"] = float(self._ui_render_ms_by_camera.get(cam_name, 0.0))
        stage_stats = self._ui_render_stage_ms_by_camera.get(cam_name, {})
        payload["ui_thumb_ms"] = float(stage_stats.get("thumb", 0.0))
        payload["ui_grid_ms"] = float(stage_stats.get("grid", 0.0))
        payload["ui_grid_avg_ms"] = float(stage_stats.get("grid_avg_ms", payload["ui_grid_ms"]))
        payload["ui_grid_fps"] = float(stage_stats.get("grid_fps", 0.0))
        payload["ui_grid_target_fps"] = float(stage_stats.get("grid_target_fps", self._grid_target_fps()))
        payload["ui_main_ms"] = float(stage_stats.get("main", 0.0))
        self._worker_diag[cam_name] = payload
        self.worker_status[cam_name] = payload
        self._heartbeat_last_seen[cam_name] = time.monotonic()
        if cam_name in self._heartbeat_alerted:
            self._log_info("performance", "worker heartbeat recovered", source="heartbeat-watchdog", camera=cam_name)
        self._heartbeat_alerted.discard(cam_name)

        now = time.monotonic()
        last_perf = float(self._performance_log_last_ts_by_camera.get(cam_name, 0.0))
        significant_change = self._heartbeat_perf_changed(cam_name, payload)
        if self.performance_diagnostics_enabled and now - last_perf >= self._performance_log_interval_s and significant_change:
            mode = str(payload.get("preview_role", "thumb"))
            overload = "on" if bool(payload.get("overload_degraded", False) or self.overload_mode_active) else "off"
            self._log_info(
                "performance",
                msg("ui_worker_metrics_summary_action"),
                source="ui",
                camera=cam_name,
                details=format_dict_multiline(
                    {
                        "mode": mode,
                        "overload": overload,
                        "overload_level": int(payload.get("overload_level", self.overload_level)),
                        "capture_fps": f"{float(payload.get('capture_fps', 0.0)):.2f}",
                        "infer_fps": f"{float(payload.get('infer_fps', 0.0)):.2f}",
                        "preview_emit_fps": f"{float(payload.get('preview_emit_fps', 0.0)):.2f}",
                        "preview_target_fps": f"{float(payload.get('preview_target_fps', 0.0)):.2f}",
                        "inference_ms": f"{float(payload.get('inference_ms', payload.get('last_inference_ms', 0.0))):.2f}",
                        "average_inference_ms": f"{float(payload.get('average_inference_ms', 0.0)):.2f}",
                        "preview_processing_ms": f"{float(payload.get('preview_processing_ms', 0.0)):.2f}",
                        "recording_enqueue_ms": f"{float(payload.get('recording_enqueue_ms', 0.0)):.2f}",
                        "ui_render_ms": f"{float(payload.get('ui_render_ms', 0.0)):.2f}",
                        "thumb_ms": f"{float(payload.get('ui_thumb_ms', 0.0)):.2f}",
                        "grid_ms": f"{float(payload.get('ui_grid_ms', 0.0)):.2f}",
                        "grid_avg_ms": f"{float(payload.get('ui_grid_avg_ms', 0.0)):.2f}",
                        "grid_fps": f"{float(payload.get('ui_grid_fps', 0.0)):.2f}/{float(payload.get('ui_grid_target_fps', 0.0)):.2f}",
                        "main_ms": f"{float(payload.get('ui_main_ms', 0.0)):.2f}",
                        "queue_size": int(payload.get("queue_size", 0)),
                        "recording_queue_size": int(payload.get("recording_queue_size", payload.get("queue_size", 0))),
                        "recording_queue_peak": int(payload.get("recording_queue_peak", 0)),
                        "recording_frames_written": int(payload.get("recording_frames_written", 0)),
                        "dropped_frames": int(payload.get("dropped_frames", 0)),
                        "cpu_percent": f"{float(payload.get('cpu_percent', 0.0)):.1f}",
                        "rss_mb": f"{float(payload.get('rss_mb', 0.0)):.1f}",
                        "fp_proxy": f"{float(payload.get('false_positive_proxy_rate', 0.0)):.3f}",
                        "avg_conf": f"{float(payload.get('avg_confidence', 0.0)):.3f}",
                        "trigger_h": f"{float(payload.get('trigger_frequency_per_hour', 0.0)):.2f}",
                    },
                    PERFORMANCE_PARAM_LABELS,
                ),
            )
            self._performance_log_last_ts_by_camera[cam_name] = now

        for idx, cam in enumerate(self.cameras):
            if str(cam.get("name", idx)) == cam_name:
                cam["runtime_telemetry"] = {
                    "false_positive_proxy_rate": float(payload.get("false_positive_proxy_rate", 0.0)),
                    "avg_confidence": float(payload.get("avg_confidence", 0.0)),
                    "trigger_frequency_per_hour": float(payload.get("trigger_frequency_per_hour", 0.0)),
                    "calibration_sample_count": int(payload.get("calibration_sample_count", 0)),
                    "calibration_duration_hours": float(payload.get("calibration_duration_hours", 0.0)),
                    "suggested_record_threshold": payload.get("suggested_record_threshold"),
                }
                break

        self._evaluate_overload_mode()
        self._refresh_camera_status_indicators()
        idx = self.camera_list.currentRow()
        if 0 <= idx < len(self.cameras) and str(self.cameras[idx].get("name", idx)) == cam_name:
            self.last_render_time = 0.0
            self._render_current()


    def _is_heartbeat_stale(self, camera_name: str, timeout_seconds: float = 15.0) -> bool:
        last_seen = float(self._heartbeat_last_seen.get(str(camera_name), 0.0) or 0.0)
        if last_seen <= 0.0:
            return True
        return (time.monotonic() - last_seen) > float(timeout_seconds)

    @staticmethod
    def _fmt_metric_or_stale(value: object, *, stale: bool, fmt: str, suffix: str = "") -> str:
        if stale:
            return f"--{suffix}"
        try:
            return f"{fmt.format(float(value))}{suffix}"
        except (TypeError, ValueError):
            return f"--{suffix}"

    def _refresh_diag_panel(self):
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
        stale = self._is_heartbeat_stale(name)
        cpu_text = self._fmt_metric_or_stale(stat.get('cpu_percent', 0.0), stale=stale, fmt='{:.1f}')
        rejection = dict(stat.get("rejection_counters", {}) or {})
        self.diag_panel.setText(
            "\n".join(
                [
                    f"[{name}]" + (" (heartbeat przeterminowany)" if stale else ""),
                    f"stream fps: {float(stat.get('stream_fps', 0.0)):.2f}",
                    f"detect fps: {float(stat.get('detect_fps', 0.0)):.2f}",
                    f"capture fps: {float(stat.get('capture_fps', 0.0)):.2f}",
                    f"infer fps: {float(stat.get('infer_fps', 0.0)):.2f}",
                    f"preview emit fps: {float(stat.get('preview_emit_fps', 0.0)):.2f}",
                    f"ui render ms: {float(stat.get('ui_render_ms', 0.0)):.2f}",
                    f"ui grid avg ms: {float(stat.get('ui_grid_avg_ms', 0.0)):.2f}",
                    f"ui grid fps/target: {float(stat.get('ui_grid_fps', 0.0)):.2f}/{float(stat.get('ui_grid_target_fps', self._grid_target_fps())):.2f}",
                    f"writer fps: {float(stat.get('writer_fps', 0.0)):.2f}",
                    f"recording queue size: {int(stat.get('queue_size', 0))}",
                    f"dropped frames: {int(stat.get('dropped_frames', 0))}",
                    f"cpu %: {cpu_text}",
                    f"rss mb: {float(stat.get('rss_mb', 0.0)):.1f}",
                    f"preview role: {stat.get('preview_role', '-')}",
                    f"overload degraded: {bool(stat.get('overload_degraded', False))}",
                    f"last detection seconds: {float(stat.get('last_detection_seconds', -1.0)):.1f}",
                    (
                        "rejections: "
                        f"below_record_threshold={int(rejection.get('below_record_threshold', 0))}, "
                        f"class_not_in_record_classes={int(rejection.get('class_not_in_record_classes', 0))}, "
                        f"outside_schedule={int(rejection.get('outside_schedule', 0))}, "
                        f"detection_disabled={int(rejection.get('detection_disabled', 0))}"
                    ),
                ]
            )
        )

    def _update_diagnostics_panel(self):
        self._refresh_diag_panel()

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
            self._refresh_camera_hud(idx)
            return result

        if requires_restart:
            result["restarted"] = self._maybe_restart_camera_after_settings(
                idx,
                was_running,
                requires_restart,
                restart_reason_keys,
            )
            self._refresh_camera_hud(idx)
            return result

        if was_running and isinstance(worker, CameraWorker):
            worker.apply_runtime_settings(new_camera)
            result["applied_live"] = True
            self._log_info("settings", "ustawienia zastosowane bez restartu", source="settings", camera=str(new_camera.get("name", idx)))

        if "show_camera_info_overlay" in changed_keys:
            widocznosc = "włączono" if bool(new_camera.get("show_camera_info_overlay", True)) else "wyłączono"
            self._log_info("settings", f"widoczność HUD: {widocznosc}", source="camera-hud", camera=str(new_camera.get("name", idx)))

        self._refresh_camera_hud(idx)
        return result


    def _refresh_camera_hud(self, idx: int) -> None:
        try:
            if not (0 <= idx < len(self.cameras)):
                return
            camera_name = str(self.cameras[idx].get("name", idx))
            self._last_status[idx] = self._last_status.get(idx, "Połączono") or "Połączono"
            if idx == self.camera_list.currentRow():
                self.last_render_time = 0.0
                self._render_current()
            self._log_info("settings", "odświeżono HUD kamery po zmianie ustawień", source="camera-hud", camera=camera_name)
        except Exception as exc:
            self._log_warning("settings", f"nie udało się odświeżyć HUD: {exc}", source="camera-hud")

    def _maybe_restart_camera_after_settings(
        self,
        idx: int,
        was_running: bool,
        requires_restart: bool,
        restart_reason_keys: list[str] | None = None,
    ) -> bool:
        if not requires_restart:
            return False
        restart_reason_keys = restart_reason_keys or []
        if any(key in {"degirum_device_override_enabled", "degirum_device_override"} for key in restart_reason_keys):
            self._log_info(
                "settings",
                "restart kamery po zmianie urządzenia",
                source="settings",
                camera=str(self.cameras[idx].get("name", idx)),
                details=f"reason_keys={restart_reason_keys}",
            )
        restarted = self._restart_camera_with_new_settings(idx, was_running)
        if restarted:
            self._log_info("settings", "kamera została automatycznie zrestartowana po zmianie ustawień", source="settings", camera=str(self.cameras[idx].get("name", idx)))
        return restarted

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
            previous_cfg = self._build_runtime_config()
            self.cameras[idx] = new_data
            candidate_cfg = self._save_runtime_config()

            self.camera_list.rebuild(self.cameras)
            self.camera_grid.rebuild(self.cameras)
            self.camera_list.setCurrentRow(idx)

            result = self._apply_camera_settings_change(idx, cam, new_data)
            profile_name = str(new_data.get("sensitivity_profile", "custom") or "custom")
            details = (
                f"profile={profile_name} "
                f"confidence_threshold_draw={float(new_data.get('confidence_threshold_draw', 0.0)):.2f} "
                f"confidence_threshold_record={float(new_data.get('confidence_threshold_record', 0.0)):.2f} "
                f"required_hits_to_start_recording={int(new_data.get('required_hits_to_start_recording', 1))} "
                f"required_misses_to_end_detection={int(new_data.get('required_misses_to_end_detection', 1))} "
                f"min_record_seconds={int(new_data.get('min_record_seconds', 0))}"
            )
            self._log_info("settings", "camera settings persisted", source="settings", camera=str(new_data.get("name", idx)), details=details)
            self.log_window.add_entry(
                "settings",
                f"zapisano ustawienia kamery {new_data.get('name')} changed={result.get('changed_keys', [])} restart={result.get('restart_reason_keys', [])} {details}",
            )
            self._start_config_watchdog(previous_cfg, candidate_cfg, reason=f"camera-settings:{new_data.get('name', idx)}")
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
        self.overload_level = 0
        self._invalidate_preview_cache()

    def switch_camera(self, idx):
        # odśwież HUD dla nowej kamery
        self.last_render_time = 0.0
        self._apply_worker_preview_roles()
        self._evaluate_overload_mode()
        self._render_current()

    def update_frame(self, frame, index, quality: str = "main"):
        try:
            idx = int(index)
        except (TypeError, ValueError):
            logger.warning("Ignoring frame with invalid index %r", index)
            return
        quality_mode = "thumb" if str(quality).lower() == "thumb" else "main"
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
            self._last_main_frame.pop(idx, None)
            self._last_thumb_frame.pop(idx, None)
            self._last_fps_text[idx] = ""
            self._invalidate_preview_cache(idx)
            if hasattr(self.camera_list, "update_thumbnail_pixmap"):
                self.camera_list.update_thumbnail_pixmap(idx, None)
            if hasattr(self.camera_grid, "update_pixmap"):
                self.camera_grid.update_pixmap(idx, None)
            if idx == self.camera_list.currentRow():
                self._render_current()
            return

        self._last_frame_update_ts[idx] = time.monotonic()
        if quality_mode == "main":
            self._invalidate_preview_cache(idx, channels={"main"})
        else:
            self._invalidate_preview_cache(idx, channels={"thumb", "grid"})
        now_mono = time.monotonic()
        if quality_mode == "main":
            self._last_main_frame[idx] = frame
        else:
            self._last_thumb_frame[idx] = frame

        # Main preview rendering path stays isolated from thumb/grid work.
        if quality_mode != "main":
            self._update_thumbnail_view(idx, now_mono)
            self._update_grid_view(idx, now_mono)

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

        self._last_fps_text[idx] = fps_txt
        self._last_status[idx] = "Połączono"
        self._last_error.pop(idx, None)

        if quality_mode == "main" and idx == self.camera_list.currentRow():
            self._render_current()

    def _update_thumbnail_view(self, idx: int, now_mono: float) -> None:
        stage_started = time.perf_counter()
        thumb_source = self._last_thumb_frame.get(idx)
        if thumb_source is None:
            thumb_source = self._last_main_frame.get(idx)
        if thumb_source is None:
            return
        if now_mono - float(self._last_thumb_update_ts.get(idx, 0.0)) < self._thumb_update_interval_s:
            return
        thumb_pm = self._get_scaled_preview_pixmap(idx, "thumb", thumb_source, 192, 108)
        if hasattr(self.camera_list, "update_thumbnail_pixmap"):
            self.camera_list.update_thumbnail_pixmap(idx, thumb_pm)
        self._last_thumb_update_ts[idx] = now_mono
        self._record_render_stage(idx, "thumb", (time.perf_counter() - stage_started) * 1000.0)

    def _update_grid_view(self, idx: int, now_mono: float) -> None:
        if not self.camera_grid.isVisible():
            return
        interval = self._grid_update_interval_s * (2.0 if self._recordings_browser_open else 1.0)
        if now_mono - float(self._last_grid_update_ts.get(idx, 0.0)) < interval:
            return
        grid_started = time.perf_counter()
        grid_source, grid_w, grid_h, grid_dpr, source_tag = self._resolve_grid_render_params(idx)
        if grid_source is not None:
            grid_pm = self._get_scaled_preview_pixmap(
                idx,
                "grid",
                grid_source,
                grid_w,
                grid_h,
                tile_width=grid_w,
                tile_height=grid_h,
                dpr=grid_dpr,
                source_tag=source_tag,
            )
        else:
            grid_pm = None
        if hasattr(self.camera_grid, "update_pixmap"):
            self.camera_grid.update_pixmap(idx, grid_pm)
        self._last_grid_update_ts[idx] = now_mono
        self._record_render_stage(idx, "grid", (time.perf_counter() - grid_started) * 1000.0)
    def _invalidate_preview_cache(self, idx: int | None = None, channels: set[str] | None = None) -> None:
        if idx is None:
            self._preview_cache.clear()
            return
        normalized_channels = {str(ch).lower() for ch in (channels or set()) if str(ch).strip()}
        keys = [
            key
            for key in self._preview_cache
            if key[0] == int(idx) and (not normalized_channels or str(key[1]).lower() in normalized_channels)
        ]
        for key in keys:
            self._preview_cache.pop(key, None)

    def _grid_target_fps(self) -> float:
        target_fps = float(self.preview_fps_grid)
        if self.camera_grid.isVisible() and str(self.grid_preview_quality).lower() == "high-quality":
            target_fps = max(target_fps, float(self.preview_fps_main))
        if self.overload_mode_active and int(self.overload_level) >= 2:
            target_fps *= max(0.4, float(overload_level_profile(self.overload_level).thumb_preview_fps_factor))
        return max(0.5, target_fps)

    def _resolve_grid_render_params(self, idx: int) -> tuple[np.ndarray | None, int, int, float, str]:
        thumb_source = self._last_thumb_frame.get(idx)
        main_source = self._last_main_frame.get(idx)
        source = thumb_source if thumb_source is not None else main_source
        source_tag = "thumb"
        camera_name = str(self.cameras[idx].get("name", idx)) if 0 <= idx < len(self.cameras) else str(idx)
        camera_overload_state = self.worker_status.get(camera_name, {}) if isinstance(self.worker_status, dict) else {}
        camera_critical_overload = self.overload_mode_active and (
            int(self.overload_level) >= 2 or bool(camera_overload_state.get("overload_degraded", False))
        )

        tile_width = 320
        tile_height = 180
        dpr = 1.0
        if 0 <= idx < len(self.camera_grid.items):
            label = self.camera_grid.items[idx].frame_label
            tile_size = label.size()
            tile_width = max(1, int(tile_size.width()))
            tile_height = max(1, int(tile_size.height()))
            try:
                dpr = max(1.0, float(label.devicePixelRatioF()))
            except Exception:
                dpr = 1.0
        if dpr <= 0:
            dpr = 1.0

        if self.camera_grid.isVisible() and str(self.grid_preview_quality).lower() == "high-quality":
            if main_source is not None and not camera_critical_overload:
                source = main_source
                source_tag = "main"
            else:
                tile_width = max(tile_width, int(self.preview_grid_max_width))
                tile_height = max(tile_height, int(self.preview_grid_max_height))
                source_tag = "thumb-hq"

        if camera_critical_overload and thumb_source is not None:
            source = thumb_source
            source_tag = "thumb"

        if self.overload_mode_active:
            overload_scale = 0.85 if int(self.overload_level) <= 1 else 0.65
            tile_width = max(1, int(round(tile_width * overload_scale)))
            tile_height = max(1, int(round(tile_height * overload_scale)))
            source_tag = f"{source_tag}-overload"

        tile_width = min(tile_width, int(self.preview_grid_max_width))
        tile_height = min(tile_height, int(self.preview_grid_max_height))

        return source, tile_width, tile_height, dpr, source_tag

    def _get_scaled_preview_pixmap(
        self,
        idx: int,
        channel: str,
        frame: np.ndarray,
        width: int,
        height: int,
        *,
        tile_width: int | None = None,
        tile_height: int | None = None,
        dpr: float = 1.0,
        source_tag: str = "thumb",
    ) -> QPixmap:
        if (
            not isinstance(frame, np.ndarray)
            or frame.size <= 0
            or frame.ndim < 2
            or frame.shape[0] <= 0
            or frame.shape[1] <= 0
        ):
            return QPixmap()
        logical_w = max(1, int(width))
        logical_h = max(1, int(height))
        tile_w = max(1, int(tile_width if tile_width is not None else logical_w))
        tile_h = max(1, int(tile_height if tile_height is not None else logical_h))
        dpr_scaled = max(100, int(round(max(1.0, float(dpr)) * 100.0)))
        key = (int(idx), str(channel), logical_w, logical_h, tile_w, tile_h, f"{dpr_scaled}:{source_tag}")
        cached = self._preview_cache.get(key)
        if cached is not None and not cached.isNull():
            return cached
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        physical_w = max(1, int(round(logical_w * (dpr_scaled / 100.0))))
        physical_h = max(1, int(round(logical_h * (dpr_scaled / 100.0))))
        interpolation = cv2.INTER_CUBIC if "hq" in source_tag or source_tag.startswith("main") else cv2.INTER_AREA
        resized = cv2.resize(rgb, (physical_w, physical_h), interpolation=interpolation)
        qimg = QImage(resized.data, physical_w, physical_h, resized.strides[0], QImage.Format_RGB888).copy()
        qimg.setDevicePixelRatio(dpr_scaled / 100.0)
        pixmap = QPixmap.fromImage(qimg)
        self._preview_cache[key] = pixmap
        return pixmap

    def _record_render_stage(self, idx: int, stage: str, render_ms: float) -> None:
        cam_name = str(self.cameras[idx].get("name", idx)) if 0 <= idx < len(self.cameras) else str(idx)
        stage_stats = self._ui_render_stage_ms_by_camera.setdefault(cam_name, {})
        stage_stats[str(stage)] = float(render_ms)
        if str(stage) == "grid":
            grid_hist = self._ui_grid_ms_history_by_camera.setdefault(cam_name, deque(maxlen=90))
            grid_hist.append(float(render_ms))
            stage_stats["grid_avg_ms"] = float(sum(grid_hist) / max(1, len(grid_hist)))
            fps_times = self._ui_grid_frame_times_by_camera.setdefault(cam_name, deque(maxlen=180))
            now_ts = time.monotonic()
            fps_times.append(now_ts)
            if len(fps_times) >= 2:
                dt = max(1e-6, fps_times[-1] - fps_times[0])
                stage_stats["grid_fps"] = float((len(fps_times) - 1) / dt)
            else:
                stage_stats["grid_fps"] = 0.0
            stage_stats["grid_target_fps"] = float(self._grid_target_fps())
        stat = self.worker_status.get(cam_name)
        if isinstance(stat, dict):
            stat[f"ui_{stage}_ms"] = float(render_ms)
            if str(stage) == "grid":
                stat["ui_grid_avg_ms"] = float(stage_stats.get("grid_avg_ms", render_ms))
                stat["ui_grid_fps"] = float(stage_stats.get("grid_fps", 0.0))
                stat["ui_grid_target_fps"] = float(stage_stats.get("grid_target_fps", self._grid_target_fps()))

    def _build_camera_hud_lines(self, idx: int) -> list[str]:
        if idx < 0 or idx >= len(self.cameras):
            return []
        cam = self.cameras[idx]
        name = str(cam.get("name", idx))
        status = self._last_status.get(idx, "")
        err = self._last_error.get(idx, "")
        stat = self.worker_status.get(name, {})
        preview_fps = self._last_fps_text.get(idx, "")
        root_cause = build_root_cause_summary(
            ui_render_ms=float(stat.get("ui_render_ms", 0.0)),
            ui_render_limit_ms=float(self.overload_max_ui_render_ms),
            queue_size=int(stat.get("queue_size", 0)),
            queue_limit=int(self.overload_max_queue_size),
            infer_fps=float(stat.get("infer_fps", 0.0)),
            detect_fps_target=float(stat.get("detect_fps", 0.0)),
            stream_fps=float(stat.get("stream_fps", 0.0)),
            writer_fps=float(stat.get("writer_fps", 0.0)),
        )

        status_text = err if err else (status or "brak danych")
        tryb = "nagrywanie" if bool(stat.get("recording_active", False)) else "podgląd"
        polaczenie = "zatrzymane" if "zatrzym" in status_text.lower() else "aktywne"

        return [
            f"Kamera: {name}",
            f"Tryb odciążenia: poziom {int(self.overload_level)}",
            f"Status: {status_text}",
            f"Podgląd FPS: {preview_fps or '0.0 fps'}",
            f"Strumień FPS: {float(stat.get('stream_fps', 0.0)):.1f}",
            f"Capture FPS: {float(stat.get('capture_fps', 0.0)):.1f}",
            f"Infer FPS: {float(stat.get('infer_fps', 0.0)):.1f}",
            f"Preview emit FPS: {float(stat.get('preview_emit_fps', 0.0)):.1f}",
            f"UI render: {float(stat.get('ui_render_ms', 0.0)):.1f} ms",
            f"UI grid avg: {float(stat.get('ui_grid_avg_ms', 0.0)):.1f} ms",
            f"UI grid FPS: {float(stat.get('ui_grid_fps', 0.0)):.1f}/{float(stat.get('ui_grid_target_fps', self._grid_target_fps())):.1f}",
            f"Detekcja FPS: {float(stat.get('detect_fps', 0.0)):.1f}",
            f"Zapis FPS: {float(stat.get('writer_fps', 0.0)):.1f}",
            f"Kolejka: {int(stat.get('queue_size', 0))}",
            f"Pominięte klatki: {int(stat.get('dropped_frames', 0))}",
            f"CPU: {self._fmt_metric_or_stale(stat.get('cpu_percent', 0.0), stale=self._is_heartbeat_stale(name), fmt='{:.1f}', suffix='%')}",
            f"RSS: {float(stat.get('rss_mb', 0.0)):.1f} MB",
            f"Tryb: {tryb}",
            f"Połączenie: {polaczenie}",
            f"Błąd: {err or 'brak'}",
            f"Wąskie gardło: {root_cause}",
        ]


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
            pad_x = 12
            pad_y = 10
            line_gap = 2
            line_h = fm.height() + line_gap
            text_w = max(fm.horizontalAdvance(line) for line in lines)
            box_w = text_w + 2 * pad_x
            box_h = (line_h * len(lines)) + (2 * pad_y)
            x, y = self._camera_info_overlay_anchor(image_rect, (box_w, box_h), padding=10)

            overlay_alpha = int(cam.get("camera_info_overlay_alpha", DEFAULT_CAMERA_INFO_OVERLAY_ALPHA))
            overlay_alpha = max(0, min(255, overlay_alpha))
            overlay_color = QColor(0, 0, 0, overlay_alpha)
            border_color = QColor(255, 255, 255, 40)
            painter.setPen(border_color)
            painter.setBrush(overlay_color)
            painter.drawRoundedRect(x, y, box_w, box_h, 8, 8)

            baseline = y + pad_y + fm.ascent()
            text_color = QColor(255, 255, 255, 178)
            for i, line in enumerate(lines):
                ty = baseline + (i * line_h)
                painter.setPen(QColor(0, 0, 0, 230))
                painter.drawText(x + pad_x + 2, ty + 2, line)
                painter.setPen(QColor(0, 0, 0, 180))
                painter.drawText(x + pad_x + 1, ty + 1, line)
                painter.setPen(text_color)
                painter.drawText(x + pad_x, ty, line)
        finally:
            painter.end()

        return qimg

    def _hud_signature(self, idx: int) -> tuple:
        cam_name = str(self.cameras[idx].get("name", idx)) if 0 <= idx < len(self.cameras) else str(idx)
        stat = self.worker_status.get(cam_name, {}) if isinstance(self.worker_status, dict) else {}
        return (
            str(self._last_status.get(idx, "") or ""),
            str(self._last_error.get(idx, "") or ""),
            str(self._last_fps_text.get(idx, "") or ""),
            int(stat.get("queue_size", 0) or 0),
            int(stat.get("dropped_frames", 0) or 0),
            bool(stat.get("recording_active", False)),
            int(stat.get("overload_level", self.overload_level) or 0),
        )

    def _compose_letterboxed(self, frame, idx: int):
        w_label = max(1, self.camera_view.width())
        h_label = max(1, self.camera_view.height())
        current_geometry = (w_label, h_label)
        if current_geometry != self._main_view_geometry_cache:
            self._main_view_geometry_cache = current_geometry
            self._hud_cache_qimg = None
            self._hud_cache_key = None
        canvas_key = (w_label, h_label)
        canvas_template = self._canvas_bg_cache.get(canvas_key)
        if canvas_template is None:
            canvas_template = np.zeros((h_label, w_label, 3), dtype=np.uint8)
            self._canvas_bg_cache = {canvas_key: canvas_template}
        canvas = canvas_template.copy()

        image_rect = (0, 0, w_label, h_label)
        if (
            isinstance(frame, np.ndarray)
            and frame.size > 0
            and frame.ndim >= 2
            and frame.shape[0] > 0
            and frame.shape[1] > 0
        ):
            fh, fw = frame.shape[:2]
            rect_key = (fw, fh, w_label, h_label)
            cached_rect = self._letterbox_geometry_cache.get(rect_key)
            if cached_rect is None:
                cached_rect = self._compute_letterboxed_rect(fw, fh, w_label, h_label)
                self._letterbox_geometry_cache[rect_key] = cached_rect
                if len(self._letterbox_geometry_cache) > 24:
                    self._letterbox_geometry_cache.clear()
                    self._letterbox_geometry_cache[rect_key] = cached_rect
            x0, y0, new_w, new_h = cached_rect
            image_rect = (x0, y0, new_w, new_h)
            resize_started = time.perf_counter()
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas[y0:y0+new_h, x0:x0+new_w] = resized
            resize_ms = (time.perf_counter() - resize_started) * 1000.0
        else:
            resize_ms = 0.0

        cvt_started = time.perf_counter()
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        cvtcolor_ms = (time.perf_counter() - cvt_started) * 1000.0
        qimg_started = time.perf_counter()
        qimg = QImage(rgb.data, w_label, h_label, rgb.strides[0], QImage.Format_RGB888).copy()
        qimage_ms = (time.perf_counter() - qimg_started) * 1000.0

        return qimg, image_rect, {
            "resize_ms": float(resize_ms),
            "cvtcolor_ms": float(cvtcolor_ms),
            "qimage_ms": float(qimage_ms),
        }

    def _render_current(self):
        now = time.monotonic()
        if now - self.last_render_time < self._render_interval_s:
            return
        self.last_render_time = now

        idx = self.camera_list.currentRow()
        if idx < 0:
            return
        cam_name = str(self.cameras[idx].get("name", idx)) if 0 <= idx < len(self.cameras) else str(idx)
        render_started = time.perf_counter()
        frame = self._last_main_frame.get(idx)
        composed_qimg, image_rect, compose_timing = self._compose_letterboxed(frame, idx)
        hud_key = (
            idx,
            composed_qimg.width(),
            composed_qimg.height(),
            int(now / max(1e-3, self._hud_interval_s)),
            self._hud_signature(idx),
        )
        hud_started = time.perf_counter()
        if now - self._last_hud_render_ts >= self._hud_interval_s or self._hud_cache_qimg is None or self._hud_cache_key != hud_key:
            composed_qimg = self._draw_camera_info_overlay(composed_qimg, idx, image_rect)
            self._hud_cache_qimg = composed_qimg.copy()
            self._hud_cache_key = hud_key
            self._last_hud_render_ts = now
        elif self._hud_cache_qimg is not None:
            composed_qimg = self._hud_cache_qimg
        hud_ms = (time.perf_counter() - hud_started) * 1000.0
        qpix_started = time.perf_counter()
        self.camera_view.setPixmap(QPixmap.fromImage(composed_qimg))
        qpixmap_ms = (time.perf_counter() - qpix_started) * 1000.0
        render_ms = (time.perf_counter() - render_started) * 1000.0
        self._record_render_stage(idx, "main", render_ms)
        self._record_render_stage(idx, "resize_ms", compose_timing.get("resize_ms", 0.0))
        self._record_render_stage(idx, "cvtcolor_ms", compose_timing.get("cvtcolor_ms", 0.0))
        self._record_render_stage(idx, "qimage_ms", compose_timing.get("qimage_ms", 0.0))
        self._record_render_stage(idx, "qpixmap_ms", qpixmap_ms)
        self._record_render_stage(idx, "hud_ms", hud_ms)
        self._record_render_stage(idx, "total_ui_render_ms", render_ms)
        self._ui_render_ms_by_camera[cam_name] = float(render_ms)
        stat = self.worker_status.get(cam_name)
        if isinstance(stat, dict):
            stat["ui_render_ms"] = float(render_ms)


    def open_video_file(self, filepath: str):
        self._log_info("browser", f"odtworzono nagranie {os.path.basename(filepath)}", source="ui")
        dlg = VideoPlayerDialog(filepath, self)
        dlg.exec_()

    def open_recordings_browser(self):
        self._log_info("browser", "otwarto przeglądarkę nagrań", source="ui")
        self._recordings_browser_open = True
        self._thumb_update_interval_s = max(self._thumb_update_interval_s, 1.0 / 2.0)
        self._grid_update_interval_s = max(self._grid_update_interval_s, 1.0 / 1.0)
        self._apply_worker_preview_roles()
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
        self._recordings_browser_open = False
        self._refresh_preview_intervals()
        self._apply_worker_preview_roles()

    def open_camera_list_dialog(self):
        self.log_window.add_entry("application", "otwarto listę kamer")
        dlg = CameraListDialog(self.camera_grid, self)
        dlg.camera_selected.connect(lambda idx: self.camera_list.setCurrentRow(idx))
        self._apply_worker_preview_roles()
        dlg.exec_()
        self.camera_grid.setParent(None)
        self.camera_grid.hide()
        self._apply_worker_preview_roles()

    def closeEvent(self, event):
        self.stop_all()
        self.alert_mem.flush()
        flush_storage()
        event.accept()

    def open_settings(self):
        self.log_window.add_entry("settings", "otworzono ustawienia")
        dlg = SettingsHub(self)
        dlg.exec_()

    def open_system_load_balancer_dialog(self):
        self.log_window.add_entry("settings", "otworzono auto-balans obciążenia")
        dlg = SystemLoadBalancerDialog(self)
        dlg.exec_()

    def apply_system_load_balancer_settings(self, settings: dict) -> None:
        previous_cfg = self._build_runtime_config()
        self.overload_protection_enabled = bool(settings.get("enabled", self.overload_protection_enabled))
        self.overload_safety_threshold_pct = int(settings.get("safety_threshold_pct", self.overload_safety_threshold_pct))
        self.overload_min_camera_count = int(settings.get("min_camera_count", self.overload_min_camera_count))
        self.overload_camera_count_threshold = int(settings.get("camera_count_threshold", self.overload_camera_count_threshold))
        self.overload_max_ui_render_ms = float(settings.get("max_ui_render_ms", self.overload_max_ui_render_ms))
        self.overload_max_queue_size = int(settings.get("max_queue_size", self.overload_max_queue_size))
        self.overload_max_preview_bandwidth_mbps = float(settings.get("max_preview_bandwidth_mbps", self.overload_max_preview_bandwidth_mbps))
        self.overload_enter_debounce_seconds = float(settings.get("enter_debounce_seconds", self.overload_enter_debounce_seconds))
        self.overload_exit_debounce_seconds = float(settings.get("exit_debounce_seconds", self.overload_exit_debounce_seconds))
        self.overload_disable_nonessential_overlays = bool(settings.get("disable_nonessential_overlays", self.overload_disable_nonessential_overlays))
        candidate_cfg = self._save_runtime_config()
        self._evaluate_overload_mode()
        self._log_info(
            "settings",
            "zastosowano auto-balans obciążenia",
            source="settings",
            details=(
                f"enabled={self.overload_protection_enabled} safety_pct={self.overload_safety_threshold_pct} "
                f"min_camera_count={self.overload_min_camera_count} camera_count_threshold={self.overload_camera_count_threshold}"
            ),
        )
        self._start_config_watchdog(previous_cfg, candidate_cfg, reason="system-load-balancer")

    def open_quality_performance_panel(self):
        dlg = QualityPerformanceDialog(self)
        dlg.exec_()

    def open_degirum_device_settings_dialog(self):
        self.log_window.add_entry("settings", "otworzono ustawienia urządzeń DeGirum")
        dlg = DeGirumDeviceSettingsDialog(self)
        dlg.exec_()

    def apply_degirum_device_settings(self, settings: dict) -> None:
        previous_cfg = self._build_runtime_config()
        self.degirum_auto_select_best = bool(settings.get("degirum_auto_select_best", self.degirum_auto_select_best))
        preferred = config_module.normalize_degirum_device_selection(
            settings.get("degirum_preferred_device", self.degirum_preferred_device)
        )
        available = [
            config_module.normalize_degirum_device_selection(item)
            for item in settings.get("degirum_available_devices", self.degirum_available_devices)
        ]
        unique_available = [item for item in dict.fromkeys([item for item in available if item and item not in {"auto"}])]
        self.degirum_preferred_device = preferred
        self.degirum_device_mode = "auto"
        self.degirum_available_devices = unique_available
        self.degirum_last_benchmark = dict(settings.get("degirum_last_benchmark", self.degirum_last_benchmark))
        candidate_cfg = self._save_runtime_config()
        self.model_cache.clear()
        self.restart_workers_and_ui()
        self._refresh_camera_status_indicators()
        self._log_info(
            "settings",
            "zastosowano ustawienia urządzeń DeGirum",
            source="settings",
            details=f"mode={self.degirum_device_mode} preferred={self.degirum_preferred_device} auto_select={self.degirum_auto_select_best}",
        )
        self._start_config_watchdog(previous_cfg, candidate_cfg, reason="degirum-device-settings")

    def apply_quality_performance_preset(self, preset_key: str) -> None:
        preset = QUALITY_PERFORMANCE_PRESETS.get(str(preset_key))
        if not preset:
            return
        previous_cfg = self._build_runtime_config()
        self.quality_performance_preset = str(preset_key)
        self.preview_fps_main = float(preset.get("preview_fps_main", self.preview_fps_main))
        self.preview_fps_grid = float(preset.get("preview_fps_grid", self.preview_fps_grid))
        self.preview_fps_thumb = float(preset.get("preview_fps_thumb", self.preview_fps_thumb))
        self.preview_main_max_width = int(preset.get("preview_main_max_width", self.preview_main_max_width))
        self.preview_main_max_height = int(preset.get("preview_main_max_height", self.preview_main_max_height))
        self.preview_grid_max_width = int(preset.get("preview_grid_max_width", self.preview_grid_max_width))
        self.preview_grid_max_height = int(preset.get("preview_grid_max_height", self.preview_grid_max_height))
        self.preview_thumb_max_width = int(preset.get("preview_thumb_max_width", self.preview_thumb_max_width))
        self.preview_thumb_max_height = int(preset.get("preview_thumb_max_height", self.preview_thumb_max_height))
        self._refresh_preview_intervals()
        for cam in self.cameras:
            if not isinstance(cam, dict):
                continue
            cam["preview_fps_main"] = float(self.preview_fps_main)
            cam["preview_fps_grid"] = float(self.preview_fps_grid)
            cam["preview_fps_thumb"] = float(self.preview_fps_thumb)
            cam["preview_main_max_width"] = int(self.preview_main_max_width)
            cam["preview_main_max_height"] = int(self.preview_main_max_height)
            cam["preview_grid_max_width"] = int(self.preview_grid_max_width)
            cam["preview_grid_max_height"] = int(self.preview_grid_max_height)
            cam["preview_thumb_max_width"] = int(self.preview_thumb_max_width)
            cam["preview_thumb_max_height"] = int(self.preview_thumb_max_height)
            cam["preview_channel_policies"] = {
                "main": {"fps": float(self.preview_fps_main), "max_width": int(self.preview_main_max_width), "max_height": int(self.preview_main_max_height)},
                "grid": {"fps": float(self.preview_fps_grid), "max_width": int(self.preview_grid_max_width), "max_height": int(self.preview_grid_max_height)},
                "thumb": {"fps": float(self.preview_fps_thumb), "max_width": int(self.preview_thumb_max_width), "max_height": int(self.preview_thumb_max_height)},
            }
        candidate_cfg = self._save_runtime_config()
        self._apply_worker_preview_roles()
        self._evaluate_overload_mode()
        self._invalidate_preview_cache()
        self._start_config_watchdog(previous_cfg, candidate_cfg, reason=f"quality-preset:{preset_key}")


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
        btn_auto_balance = QPushButton("Auto-balans obciążenia")
        btn_quality_perf = QPushButton("Jakość/Wydajność")
        btn_degirum_device = QPushButton("Urządzenia DeGirum")
        btn_restart = QPushButton("Restart aplikacji")
        btn_close = QPushButton("Zamknij")

        for b in [btn_add_cam, btn_add_usb, btn_remove_cam, btn_auto_balance, btn_quality_perf, btn_degirum_device, btn_restart, btn_close]:
            layout.addWidget(b)

        btn_add_cam.clicked.connect(parent.add_camera_wizard)
        btn_add_usb.clicked.connect(parent.add_usb_camera)
        btn_remove_cam.clicked.connect(parent.remove_camera_dialog)
        btn_auto_balance.clicked.connect(parent.open_system_load_balancer_dialog)
        btn_quality_perf.clicked.connect(parent.open_quality_performance_panel)
        btn_degirum_device.clicked.connect(parent.open_degirum_device_settings_dialog)
        btn_restart.clicked.connect(parent.restart_app)
        btn_close.clicked.connect(self.accept)


class DeGirumDeviceSettingsDialog(QDialog):
    _worker_finished = pyqtSignal(str, object)

    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        self.setWindowTitle("Ustawienia urządzeń DeGirum")
        self.resize(840, 540)
        self.parent_window = parent
        self._task_running = False
        self._detected_devices = list(parent.degirum_available_devices)
        self._benchmark_data = dict(parent.degirum_last_benchmark)
        self._worker_finished.connect(self._on_worker_finished, Qt.QueuedConnection)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.auto_select_chk = QCheckBox("Automatycznie wybieraj najlepsze urządzenie")
        self.auto_select_chk.setChecked(bool(parent.degirum_auto_select_best))
        self.auto_select_chk.setToolTip("Po testach aplikacja sama wskaże najszybsze urządzenie i zapisze je w konfiguracji.")
        form.addRow(self.auto_select_chk)

        self.default_device_combo = QComboBox()
        self.default_device_combo.setToolTip("Automatycznie wybiera najlepszy wspierany device type modelu.")
        form.addRow("Domyślne urządzenie", self.default_device_combo)

        self.benchmark_model_combo = QComboBox()
        self.benchmark_model_combo.setToolTip("Model używany do wykrywania urządzeń i benchmarku DeGirum.")
        form.addRow("Model benchmarku", self.benchmark_model_combo)
        layout.addLayout(form)

        buttons_row = QHBoxLayout()
        self.detect_btn = QPushButton("Wykryj urządzenia")
        self.detect_btn.setToolTip("Wykrywa urządzenia wspierane przez lokalny runtime DeGirum bez blokowania interfejsu.")
        self.benchmark_btn = QPushButton("Przetestuj i wybierz najlepsze")
        self.benchmark_btn.setToolTip("Uruchamia test czasu inicjalizacji modelu na urządzeniach i wybiera rekomendację.")
        buttons_row.addWidget(self.detect_btn)
        buttons_row.addWidget(self.benchmark_btn)
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

        self.status_label = QLabel("Gotowe.")
        self.status_label.setToolTip("Stan ostatniej operacji wykrywania/testu urządzeń.")
        layout.addWidget(self.status_label)
        self.benchmark_context_label = QLabel("")
        self.benchmark_context_label.setToolTip("Informacja, dla jakiego modelu i urządzenia pokazano wyniki.")
        layout.addWidget(self.benchmark_context_label)

        self.table = QTableWidget(0, 6, self)
        self.table.setHorizontalHeaderLabels(["Nazwa", "Typ", "Dostępność", "Czas testu", "Rekomendacja", "Szczegóły"])
        self.table.setToolTip("Lista urządzeń wykrytych i przetestowanych. Czas testu to czas inicjalizacji modelu.")
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        layout.addWidget(self.table, stretch=1)

        action_row = QHBoxLayout()
        self.btn_cancel = QPushButton("Anuluj")
        self.btn_apply = QPushButton("Zapisz")
        self.btn_apply.setToolTip("Zapisuje ustawienia urządzeń do globalnego config i odświeża działanie aplikacji.")
        action_row.addStretch(1)
        action_row.addWidget(self.btn_cancel)
        action_row.addWidget(self.btn_apply)
        layout.addLayout(action_row)

        self.detect_btn.clicked.connect(self._on_detect_clicked)
        self.benchmark_btn.clicked.connect(self._on_benchmark_clicked)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_apply.clicked.connect(self._apply)

        initial_device = str(parent.degirum_preferred_device or "auto")
        self._refresh_device_combo(initial_device)
        self._refresh_benchmark_model_combo()
        self._refresh_table()

    def _collect_available_benchmark_models(self) -> list[str]:
        models: list[str] = []
        for cam in self.parent_window.cameras:
            model_name = str(cam.get("model", "")).strip()
            if model_name:
                models.append(model_name)
        last_model = str(self._benchmark_data.get("model_name", "")).strip() if isinstance(self._benchmark_data, dict) else ""
        if last_model:
            models.append(last_model)
        if not models:
            models = [DEFAULT_MODEL]
        return list(dict.fromkeys(models))

    def _preferred_benchmark_model(self) -> str:
        active_idx = self.parent_window.camera_list.currentRow() if hasattr(self.parent_window, "camera_list") else -1
        if isinstance(active_idx, int) and 0 <= active_idx < len(self.parent_window.cameras):
            active_model = str(self.parent_window.cameras[active_idx].get("model", "")).strip()
            if active_model:
                return active_model
        if self.parent_window.cameras:
            global_model = str(self.parent_window.cameras[0].get("model", "")).strip()
            if global_model:
                return global_model
        return str(self._benchmark_data.get("model_name", "")).strip() or DEFAULT_MODEL

    def _refresh_benchmark_model_combo(self) -> None:
        selected_model = self._preferred_benchmark_model()
        models = self._collect_available_benchmark_models()
        self.benchmark_model_combo.clear()
        self.benchmark_model_combo.addItems(models)
        idx = self.benchmark_model_combo.findText(selected_model)
        self.benchmark_model_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _selected_benchmark_model(self) -> str:
        model_name = str(self.benchmark_model_combo.currentText() or "").strip()
        return model_name or DEFAULT_MODEL

    def _effective_device_type_label(self, model_name: str) -> str:
        logical = config_module.normalize_degirum_device_selection(self.default_device_combo.currentData() or "auto")
        supported_types = []
        if isinstance(self._benchmark_data, dict):
            supported_types = list(self._benchmark_data.get("supported_device_types", []))
        if not supported_types:
            try:
                supported_types = list(
                    get_model_supported_device_types(
                        dg,
                        model_name=model_name,
                        zoo_url=MODELS_PATH / model_name,
                        supported_cache=self.parent_window._degirum_supported_types_cache,
                    )
                )
            except Exception:
                supported_types = []
        resolution = resolve_degirum_runtime_target(
            logical_selection=logical,
            supported_device_types=supported_types,
        )
        return str(resolution.get("final_device_type") or "n/a")

    def _refresh_device_combo(self, selected_device: str | None = None) -> None:
        options: list[tuple[str, str]] = [
            ("Auto (zalecane)", "auto"),
            ("CPU (stabilny)", "cpu"),
            ("GPU (szybki)", "gpu"),
        ]
        for dev in self._detected_devices:
            val = config_module.normalize_degirum_device_selection(dev)
            if not val or val in {"auto", "cpu", "gpu"}:
                continue
            options.append((val, val))
        unique_options = []
        seen_values = set()
        for label, value in options:
            if value in seen_values:
                continue
            seen_values.add(value)
            unique_options.append((label, value))
        self.default_device_combo.clear()
        for label, value in unique_options:
            self.default_device_combo.addItem(label, value)
        if selected_device is None:
            selected_device = config_module.normalize_degirum_device_selection(self.default_device_combo.currentData() or "auto")
        idx = self.default_device_combo.findData(config_module.normalize_degirum_device_selection(selected_device))
        self.default_device_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _rows_for_table(self) -> list[dict]:
        benchmark_by_device = self._benchmark_data.get("by_device", {}) if isinstance(self._benchmark_data, dict) else {}
        recommended = str(self._benchmark_data.get("recommended_device", "")).strip().lower() if isinstance(self._benchmark_data, dict) else ""
        supported_types = self._benchmark_data.get("supported_device_types", []) if isinstance(self._benchmark_data, dict) else []
        known_devices = ["auto", "cpu", "gpu"] + list(self._detected_devices)
        rows = []
        for device in dict.fromkeys([str(item).strip().lower() for item in known_devices if str(item).strip()]):
            if not device:
                continue
            bench = benchmark_by_device.get(device, {}) if isinstance(benchmark_by_device, dict) else {}
            if device == "auto":
                device_type = "AUTO"
            elif device == "cpu":
                device_type = "CPU"
            elif device == "gpu":
                device_type = "GPU"
            else:
                device_type = "device_id"
            resolution = resolve_degirum_runtime_target(
                logical_selection=device,
                supported_device_types=supported_types,
            )
            available = "Tak" if bool(resolution.get("final_device_type")) else "Nie"
            elapsed_ms = bench.get("elapsed_ms")
            if isinstance(elapsed_ms, (int, float)) and elapsed_ms > 0:
                time_label = f"{float(elapsed_ms):.1f} ms"
            else:
                time_label = "—"
            recommendation = "Tak" if device == recommended else "—"
            details = str(
                bench.get(
                    "details",
                    f"logical={device} -> final={resolution.get('final_device_type') or '-'}",
                )
            )
            rows.append(
                {
                    "name": device,
                    "type": device_type,
                    "availability": available,
                    "test_time": time_label,
                    "recommendation": recommendation,
                    "details": details,
                }
            )
        return rows

    def _refresh_table(self) -> None:
        model_name = str(self._benchmark_data.get("model_name", "")).strip() if isinstance(self._benchmark_data, dict) else ""
        if not model_name:
            model_name = self._selected_benchmark_model()
        camera_idx = self.parent_window.camera_list.currentRow() if hasattr(self.parent_window, "camera_list") else -1
        camera_name = "-"
        if isinstance(camera_idx, int) and 0 <= camera_idx < len(self.parent_window.cameras):
            camera_name = str(self.parent_window.cameras[camera_idx].get("name", f"#{camera_idx + 1}"))
        effective_device_type = self._effective_device_type_label(model_name)
        self.benchmark_context_label.setText(
            f"Benchmark dla modelu: {model_name} | Załadowany model (kamera {camera_name}): {model_name}, device_type: {effective_device_type}"
        )
        rows = self._rows_for_table()
        self.table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            self.table.setItem(row_idx, 0, QTableWidgetItem(row["name"]))
            self.table.setItem(row_idx, 1, QTableWidgetItem(row["type"]))
            self.table.setItem(row_idx, 2, QTableWidgetItem(row["availability"]))
            self.table.setItem(row_idx, 3, QTableWidgetItem(row["test_time"]))
            self.table.setItem(row_idx, 4, QTableWidgetItem(row["recommendation"]))
            self.table.setItem(row_idx, 5, QTableWidgetItem(row["details"]))

    def _set_busy(self, busy: bool, text: str) -> None:
        self._task_running = bool(busy)
        self.detect_btn.setEnabled(not busy)
        self.benchmark_btn.setEnabled(not busy)
        self.btn_apply.setEnabled(not busy)
        self.status_label.setText(text)

    def _run_task_async(self, task_name: str, fn) -> None:
        if self._task_running:
            return
        self._set_busy(True, "Przetwarzanie…")

        def _runner():
            try:
                payload = fn()
            except Exception as exc:
                payload = {"error": str(exc)}
            self._worker_finished.emit(task_name, payload)

        threading.Thread(target=_runner, daemon=True).start()

    def _discover_devices_payload(self) -> dict:
        model_name = self._selected_benchmark_model()
        try:
            records = detect_degirum_devices(
                dg,
                model_name=model_name,
                zoo_url=MODELS_PATH / model_name,
                supported_cache=self.parent_window._degirum_supported_types_cache,
            )
            discovered = [str(row.get("id")) for row in records if str(row.get("id")) not in {"auto", "cpu", "gpu"}]
            supported = [str(row.get("id")) for row in records if config_module.is_valid_degirum_device_type(row.get("id"))]
            self.parent_window._log_info(
                "detection",
                "wykryto urządzenia degirum",
                source="settings-dialog",
                details=f"model={model_name} supported={supported}",
            )
        except Exception as exc:
            discovered = []
            supported = []
            self.parent_window._log_exception(
                "error",
                "degirum device detection failed; using cpu fallback",
                exc=exc,
                source="settings-dialog",
                details=traceback.format_exc(),
            )
        return {"devices": discovered, "supported_device_types": supported}

    def _benchmark_payload(self) -> dict:
        discovery = self._discover_devices_payload()
        devices = discovery.get("devices", [])
        model_name = self._selected_benchmark_model()
        candidates = list(dict.fromkeys(["auto", "cpu", "gpu"] + list(devices)))
        benchmark = benchmark_device_candidates(
            dg,
            model_name=model_name,
            candidates=candidates,
            zoo_url=MODELS_PATH / model_name,
            config=None,
            supported_cache=self.parent_window._degirum_supported_types_cache,
        )
        benchmark_by_device = {
            str(row.get("device")): {
                "elapsed_ms": row.get("load_time_ms"),
                "details": (
                    f"logical={row.get('device')} -> final={row.get('final_device_type') or '-'}; "
                    f"{row.get('error') or 'OK'}"
                ),
            }
            for row in benchmark.get("results", [])
        }
        recommended = next(
            (str(row.get("device")) for row in benchmark.get("results", []) if row.get("available")),
            "cpu",
        )
        ranking_summary = ", ".join(f"{key}:{val.get('elapsed_ms')}" for key, val in benchmark_by_device.items())
        self.parent_window._log_info(
            "detection",
            "wynik rankingu i rekomendacja degirum",
            source="settings-dialog",
            details=f"ranking={ranking_summary} recommended={recommended}",
        )
        return {
            "devices": list(devices),
            "benchmark": {
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "model_name": model_name,
                "recommended_device": recommended,
                "by_device": benchmark_by_device,
                "supported_device_types": list(benchmark.get("supported_device_types", [])),
            },
        }

    def _on_detect_clicked(self) -> None:
        self._run_task_async("detect", self._discover_devices_payload)

    def _on_benchmark_clicked(self) -> None:
        self._run_task_async("benchmark", self._benchmark_payload)

    def _on_worker_finished(self, task_name: str, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        if data.get("error"):
            self.parent_window._log_error(
                "error",
                "degirum settings background task failed",
                source="settings-dialog",
                details=f"task={task_name} error={data.get('error')}",
            )
            self._set_busy(False, f"Błąd: {data.get('error')}")
            return
        if task_name in {"detect", "benchmark"}:
            self._detected_devices = list(data.get("devices", self._detected_devices))
            if task_name == "benchmark" and isinstance(data.get("benchmark"), dict):
                self._benchmark_data = dict(data["benchmark"])
                recommended = str(self._benchmark_data.get("recommended_device", "")).strip().lower()
                if self.auto_select_chk.isChecked() and recommended:
                    idx = self.default_device_combo.findData(recommended)
                    if idx >= 0:
                        self.default_device_combo.setCurrentIndex(idx)
        self._refresh_device_combo(self.default_device_combo.currentData())
        self._refresh_table()
        if task_name == "detect":
            self._set_busy(False, "Wykrywanie zakończone.")
        else:
            self._set_busy(False, "Test zakończony.")

    def _apply(self) -> None:
        selected = config_module.normalize_degirum_device_selection(self.default_device_combo.currentData() or "auto")
        payload = {
            "degirum_auto_select_best": self.auto_select_chk.isChecked(),
            "degirum_preferred_device": selected,
            "degirum_available_devices": list(self._detected_devices),
            "degirum_last_benchmark": dict(self._benchmark_data),
        }
        self.parent_window.apply_degirum_device_settings(payload)
        self.accept()


class SystemLoadBalancerDialog(QDialog):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        self.setWindowTitle("Auto-balans obciążenia systemu")
        self.parent_window = parent
        screen = parent.screen() if parent else None
        if screen is None:
            screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen else QRect(0, 0, 1280, 800)
        self.resize(int(available.width() * 0.92), int(available.height() * 0.92))
        self.setMinimumSize(640, 520)
        self._metrics_sampler = SystemMetricsSampler()
        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.setInterval(1000)
        self._telemetry_timer.timeout.connect(self._refresh_system_telemetry)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Funkcja stale monitoruje wydajność i automatycznie reguluje parametry kamer, "
            "aby utrzymać stabilną pracę aplikacji i uniknąć zawieszania."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.enabled_chk = QCheckBox("Włącz automatyczne równoważenie obciążenia")
        self.enabled_chk.setChecked(bool(parent.overload_protection_enabled))
        form.addRow(self.enabled_chk)

        self.safety_threshold_spin = QSpinBox()
        self.safety_threshold_spin.setRange(60, 120)
        self.safety_threshold_spin.setSuffix(" %")
        self.safety_threshold_spin.setValue(int(parent.overload_safety_threshold_pct))
        self.safety_threshold_spin.setToolTip("Niższa wartość reaguje wcześniej (bezpieczniej), wyższa później.")
        form.addRow("Próg bezpieczeństwa", self.safety_threshold_spin)

        self.min_camera_spin = QSpinBox()
        self.min_camera_spin.setRange(1, 128)
        self.min_camera_spin.setValue(int(parent.overload_min_camera_count))
        form.addRow("Min. liczba kamer", self.min_camera_spin)

        self.camera_threshold_spin = QSpinBox()
        self.camera_threshold_spin.setRange(1, 256)
        self.camera_threshold_spin.setValue(int(parent.overload_camera_count_threshold))
        form.addRow("Próg liczby kamer", self.camera_threshold_spin)

        self.max_ui_render_spin = QDoubleSpinBox()
        self.max_ui_render_spin.setRange(4.0, 50.0)
        self.max_ui_render_spin.setDecimals(1)
        self.max_ui_render_spin.setSingleStep(0.5)
        self.max_ui_render_spin.setSuffix(" ms")
        self.max_ui_render_spin.setValue(float(parent.overload_max_ui_render_ms))
        form.addRow("Limit renderowania UI", self.max_ui_render_spin)

        self.max_queue_spin = QSpinBox()
        self.max_queue_spin.setRange(2, 500)
        self.max_queue_spin.setValue(int(parent.overload_max_queue_size))
        form.addRow("Limit kolejki klatek", self.max_queue_spin)

        self.max_bandwidth_spin = QDoubleSpinBox()
        self.max_bandwidth_spin.setRange(2.0, 200.0)
        self.max_bandwidth_spin.setDecimals(1)
        self.max_bandwidth_spin.setSingleStep(0.5)
        self.max_bandwidth_spin.setSuffix(" Mbps")
        self.max_bandwidth_spin.setValue(float(parent.overload_max_preview_bandwidth_mbps))
        form.addRow("Limit pasma podglądu", self.max_bandwidth_spin)

        self.enter_debounce_spin = QDoubleSpinBox()
        self.enter_debounce_spin.setRange(0.5, 30.0)
        self.enter_debounce_spin.setDecimals(1)
        self.enter_debounce_spin.setSingleStep(0.5)
        self.enter_debounce_spin.setSuffix(" s")
        self.enter_debounce_spin.setValue(float(parent.overload_enter_debounce_seconds))
        form.addRow("Wejście w tryb odciążenia", self.enter_debounce_spin)

        self.exit_debounce_spin = QDoubleSpinBox()
        self.exit_debounce_spin.setRange(0.5, 60.0)
        self.exit_debounce_spin.setDecimals(1)
        self.exit_debounce_spin.setSingleStep(0.5)
        self.exit_debounce_spin.setSuffix(" s")
        self.exit_debounce_spin.setValue(float(parent.overload_exit_debounce_seconds))
        form.addRow("Wyjście z trybu odciążenia", self.exit_debounce_spin)

        self.disable_overlay_chk = QCheckBox("Wyłączaj mniej istotne nakładki przy wysokim obciążeniu")
        self.disable_overlay_chk.setChecked(bool(parent.overload_disable_nonessential_overlays))
        form.addRow(self.disable_overlay_chk)
        layout.addLayout(form)

        telemetry_group = QGroupBox("Live system telemetry")
        telemetry_layout = QVBoxLayout(telemetry_group)
        self.telemetry_table = QTableWidget(0, 3, telemetry_group)
        self.telemetry_table.setHorizontalHeaderLabels(["Metric", "Value", "Status"])
        header = self.telemetry_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.telemetry_table.verticalHeader().setVisible(False)
        self.telemetry_table.setEditTriggers(self.telemetry_table.NoEditTriggers)
        self.telemetry_table.setSelectionMode(self.telemetry_table.NoSelection)
        telemetry_layout.addWidget(self.telemetry_table)
        self.telemetry_hint = QLabel("Odświeżanie metryk co 1 sekundę.")
        self.telemetry_hint.setStyleSheet("color: #888;")
        telemetry_layout.addWidget(self.telemetry_hint)
        layout.addWidget(telemetry_group)

        buttons = QHBoxLayout()
        btn_cancel = QPushButton("Anuluj")
        btn_apply = QPushButton("Zastosuj")
        buttons.addStretch(1)
        buttons.addWidget(btn_cancel)
        buttons.addWidget(btn_apply)
        layout.addLayout(buttons)
        btn_cancel.clicked.connect(self.reject)
        btn_apply.clicked.connect(self._apply)
        self._refresh_system_telemetry()
        self._telemetry_timer.start()

    def closeEvent(self, event):
        if self._telemetry_timer.isActive():
            self._telemetry_timer.stop()
        super().closeEvent(event)

    def _status_from_percent(self, value):
        if value is None:
            return "N/A", QColor("#8d99ae")
        if value < 65.0:
            return "OK", QColor("#2a9d8f")
        if value < 85.0:
            return "Warning", QColor("#f4a261")
        return "Critical", QColor("#e63946")

    def _format_mb_pair(self, used, other):
        if used is None or other is None:
            return "N/A"
        return f"{used:.1f} / {other:.1f} MB"

    def _refresh_system_telemetry(self):
        data = self._metrics_sampler.collect()
        rows = []

        cpu_total = data.get("cpu_total_percent")
        cpu_status, cpu_color = self._status_from_percent(cpu_total)
        cpu_total_value = "N/A" if cpu_total is None else f"{cpu_total:.1f}%"
        rows.append(("CPU total", cpu_total_value, cpu_status, cpu_color))

        cores = data.get("cpu_per_core_percent") or []
        if cores:
            core_parts = []
            for idx, value in enumerate(cores, start=1):
                if value is None:
                    core_parts.append(f"C{idx}: N/A")
                else:
                    core_parts.append(f"C{idx}: {value:.0f}%")
            rows.append(("CPU per-core", ", ".join(core_parts), cpu_status, cpu_color))
        else:
            rows.append(("CPU per-core", "N/A", "N/A", QColor("#8d99ae")))

        mem_used = data.get("memory_used_mb")
        mem_avail = data.get("memory_available_mb")
        mem_used_pct = data.get("memory_used_percent")
        mem_status, mem_color = self._status_from_percent(mem_used_pct)
        rows.append(("RAM used/available", self._format_mb_pair(mem_used, mem_avail), mem_status, mem_color))

        swap_used = data.get("swap_used_mb")
        swap_total = data.get("swap_total_mb")
        swap_pct = data.get("swap_used_percent")
        swap_status, swap_color = self._status_from_percent(swap_pct)
        rows.append(("Swap used/total", self._format_mb_pair(swap_used, swap_total), swap_status, swap_color))

        load_avg = data.get("load_average")
        if load_avg:
            rows.append(("Load average (1/5/15m)", f"{load_avg[0]:.2f} / {load_avg[1]:.2f} / {load_avg[2]:.2f}", "Info", QColor("#457b9d")))

        io_read = data.get("io_read_mbps")
        io_write = data.get("io_write_mbps")
        if io_read is not None and io_write is not None:
            rows.append(("Disk I/O (read/write)", f"{io_read:.2f} / {io_write:.2f} MB/s", "Info", QColor("#457b9d")))

        self.telemetry_table.setRowCount(len(rows))
        for row_idx, (metric, value, status, color) in enumerate(rows):
            metric_item = QTableWidgetItem(metric)
            value_item = QTableWidgetItem(value)
            status_item = QTableWidgetItem(status)
            status_item.setForeground(color)
            self.telemetry_table.setItem(row_idx, 0, metric_item)
            self.telemetry_table.setItem(row_idx, 1, value_item)
            self.telemetry_table.setItem(row_idx, 2, status_item)

    def _apply(self):
        self.parent_window.apply_system_load_balancer_settings(
            {
                "enabled": self.enabled_chk.isChecked(),
                "safety_threshold_pct": int(self.safety_threshold_spin.value()),
                "min_camera_count": int(self.min_camera_spin.value()),
                "camera_count_threshold": int(self.camera_threshold_spin.value()),
                "max_ui_render_ms": float(self.max_ui_render_spin.value()),
                "max_queue_size": int(self.max_queue_spin.value()),
                "max_preview_bandwidth_mbps": float(self.max_bandwidth_spin.value()),
                "enter_debounce_seconds": float(self.enter_debounce_spin.value()),
                "exit_debounce_seconds": float(self.exit_debounce_spin.value()),
                "disable_nonessential_overlays": self.disable_overlay_chk.isChecked(),
            }
        )
        self.accept()


class QualityPerformanceDialog(QDialog):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        self.setWindowTitle("Jakość/Wydajność")
        self.resize(420, 220)
        self.parent_window = parent

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.preset_combo = QComboBox()
        items = [
            ("Monitoring jakościowy", "quality_monitoring"),
            ("Zbalansowany", "balanced"),
            ("Monitoring oszczędny", "economy_monitoring"),
        ]
        for label, key in items:
            self.preset_combo.addItem(label, key)
        idx = self.preset_combo.findData(parent.quality_performance_preset)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        form.addRow("Preset globalny", self.preset_combo)
        self.grid_quality_combo = QComboBox()
        self.grid_quality_combo.addItem("Normalna", "normal")
        self.grid_quality_combo.addItem("Najwyższa (main source)", "high-quality")
        grid_idx = self.grid_quality_combo.findData(str(parent.grid_preview_quality).lower())
        if grid_idx >= 0:
            self.grid_quality_combo.setCurrentIndex(grid_idx)
        form.addRow("Jakość podglądu siatki", self.grid_quality_combo)
        layout.addLayout(form)
        hint = QLabel("Preset aktualizuje limity kanałów: main, grid, thumb (FPS + rozdzielczość).")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QHBoxLayout()
        btn_cancel = QPushButton("Anuluj")
        btn_apply = QPushButton("Zastosuj")
        buttons.addStretch(1)
        buttons.addWidget(btn_cancel)
        buttons.addWidget(btn_apply)
        layout.addLayout(buttons)
        btn_cancel.clicked.connect(self.reject)
        btn_apply.clicked.connect(self._apply)

    def _apply(self):
        key = str(self.preset_combo.currentData() or "balanced")
        self.parent_window.grid_preview_quality = str(self.grid_quality_combo.currentData() or DEFAULT_GRID_PREVIEW_QUALITY)
        self.parent_window.apply_quality_performance_preset(key)
        self.accept()

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
