from __future__ import annotations

import datetime as _dt
import os
import traceback
from typing import Dict, List, Mapping, Sequence

import cv2
import numpy as np
from PyQt5.QtCore import QDate, QObject, QPoint, QRunnable, QSize, Qt, QThreadPool, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPalette, QPixmap, QColor, QPainter
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import ALERTS_HISTORY_PATH, VISIBLE_CLASSES
from ..recordings import (
    CameraDirectory,
    RecordingMetadata,
    default_filter_bounds,
    iter_recording_entries_progressive,
    thumbnail_candidates_for_entry,
)
from ..storage import remove_from_recordings_catalog
from ..runtime_helpers import app_log


class ThumbnailTaskSignals(QObject):
    ready = pyqtSignal(str, object, str)
    failed = pyqtSignal(str, str)


class ThumbnailTask(QRunnable):
    """Background thumbnail loader for QThreadPool.

    Signals live on a dedicated QObject to avoid fragile QObject+QRunnable
    multiple-inheritance lifetime issues.
    """

    def __init__(self, entry: RecordingMetadata, allow_mp4_fallback: bool = False):
        super().__init__()
        self._entry = entry
        self._allow_mp4_fallback = bool(allow_mp4_fallback)
        self.signals = ThumbnailTaskSignals()
        self.setAutoDelete(True)

    def run(self) -> None:  # pragma: no cover - async GUI path
        try:
            image, source = self._load_image()
            if image is None or image.isNull():
                self.signals.failed.emit(self._entry.filepath, source or "brak poprawnej miniatury")
                return
            self.signals.ready.emit(self._entry.filepath, image, source)
        except Exception as exc:  # pragma: no cover - defensive async path
            app_log("error", "thumbnail task crash", source="recordings-browser", level="ERROR", details=f"filepath={self._entry.filepath}; error={exc}", traceback=traceback.format_exc())
            self.signals.failed.emit(self._entry.filepath, f"thumbnail task crash: {exc}")

    def _load_image(self) -> tuple[QImage | None, str]:
        candidates = thumbnail_candidates_for_entry(self._entry)
        for idx, candidate in enumerate(candidates):
            if not os.path.exists(candidate):
                continue
            image = QImage(candidate)
            if not image.isNull():
                if idx == 0:
                    return image, "jpg-explicit"
                return image, "jpg"
            cv_img = cv2.imread(candidate, cv2.IMREAD_COLOR)
            if cv_img is None:
                continue
            return self._qimage_from_bgr(cv_img), "jpg"

        if not self._allow_mp4_fallback:
            return QImage(), "jpg-missing"

        if os.path.exists(self._entry.filepath):
            cap = cv2.VideoCapture(self._entry.filepath)
            try:
                if hasattr(cap, "isOpened") and not cap.isOpened():
                    app_log("warning", "thumbnail mp4 open failed", source="recordings-browser", level="WARNING", details=self._entry.filepath)
                    return QImage(), "mp4-open-failed"
                ok, frame = cap.read()
            except Exception as exc:
                app_log("error", "thumbnail worker exception", source="recordings-browser", level="ERROR", details=str(exc), traceback=traceback.format_exc())
                ok, frame = False, None
            finally:
                cap.release()
            if ok and frame is not None:
                return self._qimage_from_bgr(frame), "mp4-fallback"
            app_log("warning", "thumbnail mp4 read failed", source="recordings-browser", level="WARNING", details=self._entry.filepath)
            return QImage(), "mp4-read-failed"
        return QImage(), "mp4-fallback-failed"

    @staticmethod
    def _qimage_from_bgr(frame: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


class RefreshTaskSignals(QObject):
    started = pyqtSignal(int)
    chunk = pyqtSignal(int, object, object)
    ready = pyqtSignal(int, object, object)
    failed = pyqtSignal(int, str)


class RefreshTask(QRunnable):
    def __init__(
        self,
        request_id: int,
        camera_dirs: Sequence[CameraDirectory],
        history_source: object,
    ) -> None:
        super().__init__()
        self._request_id = int(request_id)
        self._camera_dirs = list(camera_dirs)
        self._history_source = history_source
        self.signals = RefreshTaskSignals()
        self.setAutoDelete(True)

    def run(self) -> None:  # pragma: no cover - async GUI path
        try:
            self.signals.started.emit(self._request_id)
            preview_entries: List[RecordingMetadata] = []
            preview_index: Dict[str, RecordingMetadata] = {}
            entries: List[RecordingMetadata] = []
            diagnostics: Dict[str, object] = {}
            for chunk, progress in iter_recording_entries_progressive(
                self._camera_dirs,
                self._history_source,
                prefer_catalog=True,
                allow_disk_fallback=True,
                heal_catalog=True,
                chunk_size=120,
            ):
                phase = str(progress.get("phase", ""))
                if phase == "final":
                    entries.extend(chunk)
                    diagnostics = dict(progress.get("diagnostics") or diagnostics)
                    self.signals.chunk.emit(self._request_id, list(chunk), dict(progress))
                    continue
                for item in chunk:
                    preview_index[item.filepath] = item
                preview_entries = sorted(preview_index.values(), key=lambda item: item.timestamp, reverse=True)
                self.signals.ready.emit(
                    self._request_id,
                    list(preview_entries),
                    {"partial": True, "phase": phase, "progress": dict(progress)},
                )
                self.signals.chunk.emit(self._request_id, list(chunk), dict(progress))
            self.signals.ready.emit(self._request_id, entries, {"partial": False, "diagnostics": diagnostics})
        except Exception as exc:
            self.signals.failed.emit(self._request_id, str(exc))


class RecordingCardWidget(QWidget):
    def __init__(self, entry: RecordingMetadata, thumb_size: QSize, parent: QWidget | None = None):
        super().__init__(parent)
        self._thumb_size = thumb_size
        self._entry = entry
        self.setObjectName("recordingCard")
        self.thumb = QLabel()
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setFixedSize(self._thumb_size)

        self.thumb_status = QLabel("Ładowanie miniatury...")
        self.thumb_status.setAlignment(Qt.AlignCenter)
        self.thumb_status.setStyleSheet("font-size:12px; color:#6b7280;")

        self.title = QLabel(f"{entry.display_time}")
        self.title.setWordWrap(True)
        self.meta = QLabel(f"{entry.camera} • {entry.label}")
        duration = float(entry.extra.get("duration", entry.extra.get("recording_duration", 0.0)) or 0.0)
        confidence = float(entry.confidence or 0.0)
        self.extra = QLabel(f"conf: {confidence:.2f} • dur: {duration:.1f}s")
        writer_fps = float(entry.extra.get("writer_fps", 0.0) or 0.0)
        dropped = int(entry.extra.get("dropped_frames", 0) or 0)
        self.diag = QLabel(f"writer: {writer_fps:.2f} • drop: {dropped}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.thumb)
        layout.addWidget(self.thumb_status)
        layout.addWidget(self.title)
        layout.addWidget(self.meta)
        layout.addWidget(self.extra)
        layout.addWidget(self.diag)

        self._set_selected(False)

    def set_loading_state(self, pixmap: QPixmap) -> None:
        self.thumb.setPixmap(pixmap)
        self.thumb_status.setText("Ładowanie miniatury...")

    def set_thumbnail_success(self, pixmap: QPixmap) -> None:
        self.thumb.setPixmap(pixmap)
        self.thumb_status.setText("")

    def set_thumbnail_failure(self, pixmap: QPixmap, message: str = "Brak miniatury") -> None:
        self.thumb.setPixmap(pixmap)
        self.thumb_status.setText(message)

    def _set_selected(self, selected: bool) -> None:
        border = "#1d5fd1" if selected else "#d0d0d0"
        bg = "#eaf1ff" if selected else "#ffffff"
        self.setStyleSheet(
            f"#recordingCard {{background: {bg}; border: 1px solid {border}; border-radius: 8px;}}"
            "QLabel { color: #000000; }"
        )


class RecordingsBrowserDialog(QDialog):
    open_video = pyqtSignal(str)
    refresh_started = pyqtSignal(int)
    refresh_ready = pyqtSignal(int, object, object)
    refresh_failed = pyqtSignal(int, str)

    def __init__(
        self,
        camera_dirs: Sequence[CameraDirectory],
        parent: QObject | None = None,
        history_path: str | os.PathLike[str] = ALERTS_HISTORY_PATH,
        history_items: Sequence[Mapping[str, object]] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Nagrania – przeglądarka")
        self.resize(1280, 760)
        self._thumb_size = QSize(320, 180)
        self._camera_dirs = list(camera_dirs)
        self._history_path = str(history_path)
        self._history_items = [dict(item) for item in history_items] if history_items is not None else None
        self._entries: List[RecordingMetadata] = []
        self._filtered_entries: List[RecordingMetadata] = []
        self.thumbnail_cache: Dict[str, QPixmap] = {}
        self._pending_thumbnails: set[str] = set()
        self._failed_thumbnails: set[str] = set()
        self._thumbnail_tasks: Dict[str, ThumbnailTask] = {}
        self._thumbnail_entries: Dict[str, RecordingMetadata] = {}
        self._mp4_fallback_requested: set[str] = set()
        self._tile_items: Dict[str, QListWidgetItem] = {}
        self._tile_cards: Dict[str, RecordingCardWidget] = {}
        self._table_rows: Dict[str, int] = {}
        self._load_diagnostics: Dict[str, object] = {}
        self.thumbnail_pool = QThreadPool(self)
        self.thumbnail_pool.setMaxThreadCount(2)
        self._tile_view_dirty = True
        self._table_view_dirty = True
        self._refresh_request_seq = 0
        self._active_refresh_request_id = 0
        self._refresh_tasks: Dict[int, RefreshTask] = {}
        self._refresh_active = False
        self._refresh_seen_count = 0
        self._refresh_loading_hint = ""
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(140)
        self._spinner_phase = 0
        self._spinner_timer.timeout.connect(self._update_loading_status)

        self.refresh_started.connect(self._on_refresh_started, Qt.QueuedConnection)
        self.refresh_ready.connect(self._on_refresh_ready, Qt.QueuedConnection)
        self.refresh_failed.connect(self._on_refresh_failed, Qt.QueuedConnection)

        self._class_options: Dict[str, str] = {str(c).casefold(): str(c) for c in VISIBLE_CLASSES}
        self._apply_light_theme()

        root = QVBoxLayout(self)
        root.addLayout(self._build_filters())
        root.addLayout(self._build_mode_row())
        root.addLayout(self._build_views())

        self.refresh(retry_failed=False)

    def _apply_light_theme(self) -> None:
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#ffffff"))
        palette.setColor(QPalette.Base, QColor("#ffffff"))
        palette.setColor(QPalette.Text, QColor("#000000"))
        self.setPalette(palette)

    def _build_filters(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Kamera:"))
        self.camera_filter = QComboBox()
        self.camera_filter.addItem("Wszystkie kamery")
        for name, _ in self._camera_dirs:
            self.camera_filter.addItem(name)
        layout.addWidget(self.camera_filter)

        layout.addWidget(QLabel("Klasa:"))
        self.class_filter = QComboBox()
        self.class_filter.addItem("Wszystkie klasy")
        for cls in sorted(self._class_options.values(), key=str.casefold):
            self.class_filter.addItem(cls)
        layout.addWidget(self.class_filter)

        layout.addWidget(QLabel("Od:"))
        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        layout.addWidget(self.date_from)

        layout.addWidget(QLabel("Do:"))
        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        layout.addWidget(self.date_to)

        layout.addWidget(QLabel("Zakres:"))
        self.quick_range = QComboBox()
        self.quick_range.addItems(["Wszystkie", "Dzisiaj", "7 dni", "30 dni"])
        layout.addWidget(self.quick_range)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Filtruj po nazwie pliku lub etykiecie...")
        layout.addWidget(self.search_edit, stretch=1)

        self.refresh_btn = QPushButton("Odśwież")
        self.delete_btn = QPushButton("Usuń zaznaczone")
        self.select_all_checkbox = QCheckBox("Zaznacz wszystko")
        layout.addWidget(self.refresh_btn)
        layout.addWidget(self.delete_btn)
        layout.addWidget(self.select_all_checkbox)

        self.camera_filter.currentTextChanged.connect(self._apply_filters)
        self.class_filter.currentTextChanged.connect(self._apply_filters)
        self.date_from.dateChanged.connect(self._apply_filters)
        self.date_to.dateChanged.connect(self._apply_filters)
        self.search_edit.textChanged.connect(self._apply_filters)
        self.quick_range.currentTextChanged.connect(self._apply_quick_range)
        self.refresh_btn.clicked.connect(lambda: self.refresh(retry_failed=True))
        self.delete_btn.clicked.connect(self.delete_selected)
        self.select_all_checkbox.stateChanged.connect(self._select_all_changed)
        return layout

    def _build_mode_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.view_mode = QComboBox()
        self.view_mode.addItems(["Kafelki", "Lista"])
        self.view_mode.setCurrentText("Kafelki")
        self.view_mode.currentTextChanged.connect(self._switch_view)
        self.status_label = QLabel("Wczytuję nagrania...")
        layout.addWidget(QLabel("Widok:"))
        layout.addWidget(self.view_mode)
        layout.addStretch(1)
        layout.addWidget(self.status_label)
        return layout

    def _build_views(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self._view_stack = QStackedLayout()

        self.tile_list = QListWidget()
        self.tile_list.setViewMode(QListWidget.IconMode)
        self.tile_list.setResizeMode(QListWidget.Adjust)
        self.tile_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tile_list.setSpacing(12)
        self.tile_list.setGridSize(QSize(360, 330))
        self.tile_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tile_list.customContextMenuRequested.connect(self._context_menu)
        self.tile_list.itemSelectionChanged.connect(self._sync_card_selection_state)
        self.tile_list.itemDoubleClicked.connect(self._tile_double_clicked)
        self._view_stack.addWidget(self.tile_list)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Usuń", "Miniatura", "Czas", "Kamera", "Klasa", "Pewność", "Czas trwania", "Plik"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.cellDoubleClicked.connect(self._table_double_clicked)
        self._view_stack.addWidget(self.table)

        holder = QWidget()
        holder.setLayout(self._view_stack)
        self._view_stack.setCurrentWidget(self.tile_list)
        layout.addWidget(holder)

        self.empty_label = QLabel("")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("font-size: 16px; color: #555;")
        layout.addWidget(self.empty_label)
        self.empty_label.hide()
        return layout

    def refresh(self, retry_failed: bool = True) -> None:
        history_source = self._history_items if self._history_items is not None else self._history_path
        if retry_failed:
            for failed_key in list(self._failed_thumbnails):
                self.thumbnail_cache.pop(failed_key, None)
            self._failed_thumbnails.clear()

        self._refresh_request_seq += 1
        request_id = self._refresh_request_seq
        self._active_refresh_request_id = request_id
        app_log("browser", "refresh recordings browser", source="recordings-browser", level="INFO", details=f"request_id={request_id}")

        task = RefreshTask(request_id, self._camera_dirs, history_source)
        task.signals.started.connect(self.refresh_started.emit, Qt.QueuedConnection)
        task.signals.ready.connect(self.refresh_ready.emit, Qt.QueuedConnection)
        task.signals.failed.connect(self.refresh_failed.emit, Qt.QueuedConnection)
        task.signals.chunk.connect(self._on_refresh_chunk, Qt.QueuedConnection)
        self._refresh_tasks[request_id] = task
        self.thumbnail_pool.start(task)

    @pyqtSlot(int)
    def _on_refresh_started(self, request_id: int) -> None:
        if request_id != self._active_refresh_request_id:
            return
        self._refresh_active = True
        self._refresh_seen_count = 0
        self._refresh_loading_hint = ""
        self._spinner_phase = 0
        self._render_loading_status()
        if not self._spinner_timer.isActive():
            self._spinner_timer.start()

    @pyqtSlot(int, object, object)
    def _on_refresh_ready(self, request_id: int, entries: object, diagnostics: object) -> None:
        self._refresh_tasks.pop(request_id, None)
        if request_id != self._active_refresh_request_id:
            return
        payload = dict(diagnostics or {}) if isinstance(diagnostics, dict) else {}
        if bool(payload.get("partial")):
            self._entries = list(entries) if isinstance(entries, list) else []
            self._ensure_class_filter_entries(self._entries)
            self._set_default_date_bounds(self._entries)
            self._apply_filters()
            return
        self._refresh_active = False
        if self._spinner_timer.isActive():
            self._spinner_timer.stop()
        self._entries = list(entries) if isinstance(entries, list) else []
        self._load_diagnostics = dict(payload.get("diagnostics") or {}) if isinstance(payload.get("diagnostics"), dict) else {}
        self._ensure_class_filter_entries(self._entries)
        self._set_default_date_bounds(self._entries)
        self._apply_filters()

    @pyqtSlot(int, str)
    def _on_refresh_failed(self, request_id: int, reason: str) -> None:
        self._refresh_tasks.pop(request_id, None)
        if request_id != self._active_refresh_request_id:
            return
        self._refresh_active = False
        if self._spinner_timer.isActive():
            self._spinner_timer.stop()
        app_log("error", "browser refresh failure", source="recordings-browser", level="ERROR", details=reason)
        QMessageBox.warning(self, "Nagrania", f"Nie udało się odczytać nagrań: {reason}")

    @pyqtSlot(int, object, object)
    def _on_refresh_chunk(self, request_id: int, chunk: object, progress: object) -> None:
        if request_id != self._active_refresh_request_id:
            return
        chunk_len = len(chunk) if isinstance(chunk, list) else 0
        total = max(self._refresh_seen_count, chunk_len)
        hint = ""
        if isinstance(progress, dict):
            phase = str(progress.get("phase", ""))
            if phase == "final":
                total = int(progress.get("offset", 0) or 0) + chunk_len
            elif phase == "catalog":
                total = max(total, int(progress.get("valid_catalog_entries", 0) or 0))
                hint = "(katalog)"
            elif phase == "disk_scan":
                total = max(total, self._refresh_seen_count + chunk_len)
                hint = "(skan dysku)"
        self._refresh_seen_count = max(self._refresh_seen_count, total)
        self._refresh_loading_hint = hint
        self._render_loading_status()

    def _render_loading_status(self) -> None:
        if not self._refresh_active:
            return
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        marker = frames[self._spinner_phase % len(frames)]
        suffix = f" {self._refresh_loading_hint}" if self._refresh_loading_hint else ""
        if self._refresh_seen_count > 0:
            self.status_label.setText(f"{marker} Wczytywanie nagrań... {self._refresh_seen_count} {suffix}".rstrip())
        else:
            self.status_label.setText(f"{marker} Wczytywanie nagrań...{suffix}".rstrip())

    def _update_loading_status(self) -> None:
        if not self._refresh_active:
            return
        self._spinner_phase += 1
        self._render_loading_status()

    def _set_default_date_bounds(self, entries: Sequence[RecordingMetadata]) -> None:
        dfrom, dto = default_filter_bounds(entries)
        qfrom = QDate(dfrom.year, dfrom.month, dfrom.day)
        qto = QDate(dto.year, dto.month, dto.day)
        self.date_from.setDate(qfrom)
        self.date_to.setDate(qto)
        self.date_from.setMinimumDate(qfrom)
        self.date_to.setMinimumDate(qfrom)
        self.date_from.setMaximumDate(qto)
        self.date_to.setMaximumDate(qto)
        self.quick_range.setCurrentText("Wszystkie")

    def _apply_quick_range(self, value: str) -> None:
        if value == "Wszystkie":
            if self._entries:
                self._set_default_date_bounds(self._entries)
            return
        today = QDate.currentDate()
        self.date_to.setDate(today)
        if value == "Dzisiaj":
            self.date_from.setDate(today)
        elif value == "7 dni":
            self.date_from.setDate(today.addDays(-6))
        elif value == "30 dni":
            self.date_from.setDate(today.addDays(-29))

    def _matches_filters(self, entry: RecordingMetadata) -> bool:
        camera_sel = self.camera_filter.currentText()
        if camera_sel != "Wszystkie kamery" and entry.camera.casefold() != camera_sel.casefold():
            return False
        class_sel = self.class_filter.currentText()
        if class_sel != "Wszystkie klasy" and entry.label.casefold() != class_sel.casefold():
            return False

        dt = _dt.datetime.fromtimestamp(entry.timestamp)
        qdate = QDate(dt.year, dt.month, dt.day)
        if qdate < self.date_from.date() or qdate > self.date_to.date():
            return False

        needle = self.search_edit.text().strip().lower()
        if needle:
            if needle not in f"{entry.filename} {entry.label}".lower():
                return False
        return True

    def _apply_filters(self) -> None:
        self._filtered_entries = [entry for entry in self._entries if self._matches_filters(entry)]
        self._tile_view_dirty = True
        self._table_view_dirty = True
        self._rebuild_active_view()
        self._update_empty_state()
        self.status_label.setText(f"Wczytano {len(self._entries)} nagrań, widoczne {len(self._filtered_entries)}")

    def _rebuild_active_view(self) -> None:
        if self.view_mode.currentText() == "Lista":
            self._rebuild_table_view(self._filtered_entries)
            self._table_view_dirty = False
        else:
            self._rebuild_tile_view(self._filtered_entries)
            self._tile_view_dirty = False


    @staticmethod
    def _thumb_cache_key(filepath: str) -> str:
        return os.path.abspath(str(filepath))

    def _rebuild_tile_view(self, entries: Sequence[RecordingMetadata]) -> None:
        self.tile_list.clear()
        self._tile_items.clear()
        self._tile_cards.clear()
        for entry in entries:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, entry.filepath)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Unchecked)
            item.setSizeHint(QSize(350, 320))
            self.tile_list.addItem(item)
            card = RecordingCardWidget(entry, self._thumb_size)
            card.set_loading_state(self._loading_pixmap())
            self.tile_list.setItemWidget(item, card)
            key = self._thumb_cache_key(entry.filepath)
            self._tile_items[key] = item
            self._tile_cards[key] = card
            cached = self.thumbnail_cache.get(key)
            if cached is not None:
                if key in self._failed_thumbnails:
                    card.set_thumbnail_failure(cached, "Brak miniatury")
                else:
                    card.set_thumbnail_success(cached)
            else:
                self._start_thumbnail_request(entry, allow_mp4_fallback=False)

    def _rebuild_table_view(self, entries: Sequence[RecordingMetadata]) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self._table_rows.clear()
        for row, entry in enumerate(entries):
            self.table.insertRow(row)
            key = self._thumb_cache_key(entry.filepath)
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Unchecked)
            chk.setData(Qt.UserRole, entry.filepath)
            self.table.setItem(row, 0, chk)

            thumb_lbl = QLabel()
            thumb_lbl.setAlignment(Qt.AlignCenter)
            if key in self.thumbnail_cache:
                thumb_lbl.setPixmap(self.thumbnail_cache[key])
            else:
                thumb_lbl.setPixmap(self._loading_pixmap())
            self.table.setCellWidget(row, 1, thumb_lbl)

            self.table.setItem(row, 2, QTableWidgetItem(entry.display_time))
            self.table.setItem(row, 3, QTableWidgetItem(entry.camera))
            self.table.setItem(row, 4, QTableWidgetItem(entry.label))
            self.table.setItem(row, 5, QTableWidgetItem("-" if entry.confidence <= 0 else f"{entry.confidence:.2f}"))
            duration = float(entry.extra.get("duration", entry.extra.get("recording_duration", 0.0)) or 0.0)
            self.table.setItem(row, 6, QTableWidgetItem("-" if duration <= 0 else f"{duration:.1f}s"))
            file_item = QTableWidgetItem(entry.filepath)
            file_item.setData(Qt.UserRole, entry.filepath)
            self.table.setItem(row, 7, file_item)
            self._table_rows[key] = row
            if key not in self.thumbnail_cache:
                self._start_thumbnail_request(entry, allow_mp4_fallback=False)
        self.table.setSortingEnabled(True)

    def _update_empty_state(self) -> None:
        message = ""
        if not self._entries:
            dirs_available = any(os.path.isdir(path) for _name, path in self._camera_dirs)
            message = "Nie znaleziono żadnych nagrań." if dirs_available else "Folder z nagraniami jest niedostępny."
        elif not self._filtered_entries:
            message = "Brak nagrań dla bieżących filtrów."
        elif self._load_diagnostics.get("used_disk_fallback") and self._load_diagnostics.get("catalog_entries", 0) == 0:
            message = "Katalog był pusty lub niepełny — wczytano nagrania bezpośrednio z dysku."

        if message:
            self.empty_label.setText(message)
            self.empty_label.show()
        else:
            self.empty_label.hide()

    def _start_thumbnail_request(self, entry: RecordingMetadata, allow_mp4_fallback: bool = False) -> None:
        key = self._thumb_cache_key(entry.filepath)
        self._thumbnail_entries[key] = entry
        if key in self._pending_thumbnails:
            return
        if key in self._thumbnail_tasks:
            return
        if key in self.thumbnail_cache:
            if key in self._failed_thumbnails:
                self._apply_thumbnail_failure_to_card(entry.filepath, "Brak miniatury")
                self._apply_thumbnail_to_table(entry.filepath, self.thumbnail_cache[key])
            else:
                self._apply_thumbnail_to_card(entry.filepath, self.thumbnail_cache[key])
                self._apply_thumbnail_to_table(entry.filepath, self.thumbnail_cache[key])
            return
        app_log("browser", "thumbnail task started", source="recordings-browser", level="INFO", details=f"filepath={entry.filepath}; mp4_fallback={allow_mp4_fallback}")
        task = ThumbnailTask(entry, allow_mp4_fallback=allow_mp4_fallback)
        task.signals.ready.connect(self._on_thumbnail_ready, Qt.QueuedConnection)
        task.signals.failed.connect(self._on_thumbnail_failed, Qt.QueuedConnection)
        self._thumbnail_tasks[key] = task
        self._pending_thumbnails.add(key)
        self.thumbnail_pool.start(task)

    @pyqtSlot(str, object, str)
    def _on_thumbnail_ready(self, filepath: str, image: object, source: str) -> None:
        key = self._thumb_cache_key(filepath)
        self._pending_thumbnails.discard(key)
        self._thumbnail_tasks.pop(key, None)
        if isinstance(image, QImage) and not image.isNull():
            pixmap = QPixmap.fromImage(image).scaled(self._thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._apply_thumbnail_to_card(filepath, pixmap)
            self._apply_thumbnail_to_table(filepath, pixmap)
            if source in {"jpg", "jpg-explicit"}:
                app_log("browser", "wczytano miniaturę JPG", source="recordings-browser", level="INFO", details=filepath)
            elif source == "mp4-fallback":
                app_log("browser", "użyto klatki MP4 jako miniatury", source="recordings-browser", level="INFO", details=filepath)
            return
        self._on_thumbnail_failed(filepath, f"niepoprawny obraz miniatury, źródło={source}")

    @pyqtSlot(str, str)
    def _on_thumbnail_failed(self, filepath: str, reason: str) -> None:
        key = self._thumb_cache_key(filepath)
        self._pending_thumbnails.discard(key)
        self._thumbnail_tasks.pop(key, None)
        if reason == "jpg-missing":
            app_log("warning", "brak pliku miniatury JPG", source="recordings-browser", level="WARNING", details=filepath)
            self._apply_thumbnail_failed(filepath, reason, level="WARNING", group="warning")
            if self._is_tile_visible(filepath) and key not in self._mp4_fallback_requested:
                self._mp4_fallback_requested.add(key)
                entry = self._thumbnail_entries.get(key)
                if entry is not None:
                    self._start_thumbnail_request(entry, allow_mp4_fallback=True)
            return
        self._apply_thumbnail_failed(filepath, reason)

    def _apply_thumbnail_failed(self, filepath: str, reason: str, level: str = "ERROR", group: str = "error") -> None:
        key = self._thumb_cache_key(filepath)
        self._failed_thumbnails.add(key)
        pixmap = self._failure_pixmap()
        self.thumbnail_cache[key] = pixmap
        self._apply_thumbnail_failure_to_card(filepath, "Brak miniatury")
        self._apply_thumbnail_to_table(filepath, pixmap)
        app_log(
            group,
            "błąd ładowania miniatury",
            source="recordings-browser",
            level=level,
            details=f"filepath={filepath}; reason={reason}",
        )

    def _apply_thumbnail_to_card(self, filepath: str, pixmap: QPixmap) -> None:
        key = self._thumb_cache_key(filepath)
        self.thumbnail_cache[key] = pixmap
        self._failed_thumbnails.discard(key)
        card = self._tile_cards.get(key)
        if card is None:
            if self.view_mode.currentText() == "Kafelki":
                app_log(
                    "warning",
                    "nie można przypisać miniatury do widocznej karty",
                    source="recordings-browser",
                    level="WARNING",
                    details=f"filepath={filepath}",
                )
            return
        card.set_thumbnail_success(pixmap)

    def _apply_thumbnail_failure_to_card(self, filepath: str, message: str) -> None:
        key = self._thumb_cache_key(filepath)
        card = self._tile_cards.get(key)
        if card is None:
            return
        card.set_thumbnail_failure(self._failure_pixmap(), message)

    def _apply_thumbnail_to_table(self, filepath: str, pixmap: QPixmap) -> None:
        key = self._thumb_cache_key(filepath)
        row = self._table_rows.get(key)
        if row is not None:
            try:
                widget = self.table.cellWidget(row, 1)
                if isinstance(widget, QLabel):
                    widget.setPixmap(pixmap)
            except Exception as exc:
                app_log("error", "thumbnail apply-to-widget failure", source="recordings-browser", level="ERROR", details=str(exc), traceback=traceback.format_exc())

    def _loading_pixmap(self) -> QPixmap:
        pix = QPixmap(self._thumb_size)
        pix.fill(QColor("#e9edf3"))
        painter = QPainter(pix)
        try:
            painter.setPen(QColor("#6b7280"))
            painter.drawText(pix.rect(), Qt.AlignCenter, "Ładowanie miniatury...")
        finally:
            painter.end()
        return pix

    def _failure_pixmap(self) -> QPixmap:
        pix = QPixmap(self._thumb_size)
        pix.fill(QColor("#f3f4f6"))
        painter = QPainter(pix)
        try:
            painter.setPen(QColor("#6b7280"))
            painter.drawText(pix.rect(), Qt.AlignCenter, "Brak miniatury")
        finally:
            painter.end()
        return pix

    def _switch_view(self, mode: str) -> None:
        current = self._current_selected_path()
        if mode == "Kafelki":
            if self._tile_view_dirty:
                self._rebuild_tile_view(self._filtered_entries)
                self._tile_view_dirty = False
            self._view_stack.setCurrentWidget(self.tile_list)
        else:
            if self._table_view_dirty:
                self._rebuild_table_view(self._filtered_entries)
                self._table_view_dirty = False
            self._view_stack.setCurrentWidget(self.table)
        if current:
            self._restore_selection(current)

    def _is_tile_visible(self, filepath: str) -> bool:
        key = self._thumb_cache_key(filepath)
        item = self._tile_items.get(key)
        if item is None:
            return False
        rect = self.tile_list.visualItemRect(item)
        if not rect.isValid():
            return False
        return self.tile_list.viewport().rect().intersects(rect)

    def _current_selected_path(self) -> str:
        if self._view_stack.currentWidget() is self.tile_list:
            item = self.tile_list.currentItem()
            return str(item.data(Qt.UserRole)) if item else ""
        row = self.table.currentRow()
        if row < 0:
            return ""
        item = self.table.item(row, 7)
        return str(item.data(Qt.UserRole)) if item else ""

    def _restore_selection(self, filepath: str) -> None:
        item = self._tile_items.get(self._thumb_cache_key(filepath))
        if item is not None:
            self.tile_list.setCurrentItem(item)
        row = self._table_rows.get(self._thumb_cache_key(filepath))
        if row is not None:
            self.table.selectRow(row)

    def _selected_paths(self) -> List[str]:
        paths: List[str] = []
        if self._view_stack.currentWidget() is self.tile_list:
            for i in range(self.tile_list.count()):
                item = self.tile_list.item(i)
                if item.checkState() == Qt.Checked or item.isSelected():
                    paths.append(str(item.data(Qt.UserRole)))
        else:
            for row in range(self.table.rowCount()):
                chk = self.table.item(row, 0)
                if chk and chk.checkState() == Qt.Checked:
                    paths.append(str(chk.data(Qt.UserRole)))
            for idx in self.table.selectionModel().selectedRows():
                item = self.table.item(idx.row(), 7)
                if item:
                    paths.append(str(item.data(Qt.UserRole)))
        # de-duplicate preserving order
        seen = set()
        out: List[str] = []
        for p in paths:
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out

    def delete_selected(self) -> None:
        paths = self._selected_paths()
        if not paths:
            QMessageBox.information(self, "Usuń nagrania", "Nie wybrano żadnych nagrań.")
            return
        if QMessageBox.question(self, "Potwierdzenie", f"Usunąć {len(paths)} nagrań?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return

        errors: List[str] = []
        for fp in paths:
            for candidate in (fp, fp + ".json", fp + ".mp4.json", fp + ".jpg"):
                if not os.path.exists(candidate):
                    continue
                try:
                    os.remove(candidate)
                except Exception as exc:
                    errors.append(f"{os.path.basename(candidate)}: {exc}")

        remove_from_recordings_catalog(paths)
        removed = set(paths)
        self._entries = [entry for entry in self._entries if entry.filepath not in removed]
        for path in removed:
            key = self._thumb_cache_key(path)
            self.thumbnail_cache.pop(key, None)
            self._failed_thumbnails.discard(key)
        self._apply_filters()
        if errors:
            QMessageBox.warning(self, "Usuń nagrania", "Częściowo usunięto pliki:\n" + "\n".join(errors))

    def _open_selected(self) -> None:
        paths = self._selected_paths()
        if paths:
            self.open_video.emit(paths[0])

    def _context_menu(self, pos: QPoint) -> None:
        menu = QMenu(self)
        open_action = menu.addAction("Otwórz")
        del_action = menu.addAction("Usuń")
        show_action = menu.addAction("Pokaż w folderze")
        action = menu.exec_(self.sender().mapToGlobal(pos))
        if action == open_action:
            self._open_selected()
        elif action == del_action:
            self.delete_selected()
        elif action == show_action:
            paths = self._selected_paths()
            if paths:
                QMessageBox.information(self, "Folder", os.path.dirname(paths[0]))

    def _tile_double_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path:
            self.open_video.emit(str(path))

    def _table_double_clicked(self, row: int, _col: int) -> None:
        item = self.table.item(row, 7)
        if item:
            path = item.data(Qt.UserRole)
            if path:
                self.open_video.emit(str(path))

    def _select_all_changed(self, state: int) -> None:
        checked = state == Qt.Checked
        if self._view_stack.currentWidget() is self.tile_list:
            for i in range(self.tile_list.count()):
                self.tile_list.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)
        else:
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                if item:
                    item.setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def _ensure_class_filter_entries(self, entries: Sequence[RecordingMetadata]) -> None:
        for entry in entries:
            key = entry.label.casefold()
            if not key or key in self._class_options:
                continue
            self._class_options[key] = entry.label
            self.class_filter.addItem(entry.label)

    def _sync_card_selection_state(self) -> None:
        selected = {self.tile_list.row(item) for item in self.tile_list.selectedItems()}
        for idx in range(self.tile_list.count()):
            item = self.tile_list.item(idx)
            path = self._thumb_cache_key(str(item.data(Qt.UserRole)))
            card = self._tile_cards.get(path)
            if card:
                card._set_selected(idx in selected)

    def closeEvent(self, event):  # noqa: D401
        self.thumbnail_pool.clear()
        super().closeEvent(event)
