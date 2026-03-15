from __future__ import annotations

import datetime as _dt
import os
from typing import Dict, List, Mapping, Sequence

import cv2
import numpy as np
from PyQt5.QtCore import QDate, QObject, QPoint, QRunnable, QSize, Qt, QThreadPool, pyqtSignal, pyqtSlot
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
    load_recording_entries,
    thumbnail_candidates_for_entry,
)
from ..storage import remove_from_recordings_catalog
from ..runtime_helpers import thumbnail_load_outcome


class ThumbnailWorker(QObject, QRunnable):
    thumbnail_ready = pyqtSignal(str, object)

    def __init__(self, entry: RecordingMetadata):
        super().__init__()
        QRunnable.__init__(self)
        self._entry = entry

    def run(self) -> None:  # pragma: no cover - async GUI path
        image = self._load_image()
        self.thumbnail_ready.emit(self._entry.filepath, image)

    def _load_image(self) -> QImage:
        for candidate in thumbnail_candidates_for_entry(self._entry):
            if not os.path.exists(candidate):
                continue
            image = QImage(candidate)
            if not image.isNull():
                return image
            cv_img = cv2.imread(candidate, cv2.IMREAD_COLOR)
            if cv_img is None:
                continue
            return self._qimage_from_bgr(cv_img)

        if os.path.exists(self._entry.filepath):
            cap = cv2.VideoCapture(self._entry.filepath)
            try:
                ok, frame = cap.read()
            finally:
                cap.release()
            if ok and frame is not None:
                return self._qimage_from_bgr(frame)
        return QImage()

    @staticmethod
    def _qimage_from_bgr(frame: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


class RecordingCardWidget(QWidget):
    def __init__(self, entry: RecordingMetadata, thumb_size: QSize, parent: QWidget | None = None):
        super().__init__(parent)
        self._thumb_size = thumb_size
        self._entry = entry
        self.setObjectName("recordingCard")
        self.thumb = QLabel()
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setFixedSize(self._thumb_size)

        self.thumb_status = QLabel("trwa wczytywanie")
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

    def set_loading_placeholder(self, pixmap: QPixmap) -> None:
        self.thumb.setPixmap(pixmap)
        self.thumb_status.setText("trwa wczytywanie")

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        self.thumb.setPixmap(pixmap)
        self.thumb_status.setText("")

    def set_thumbnail_failed(self, pixmap: QPixmap) -> None:
        self.thumb.setPixmap(pixmap)
        self.thumb_status.setText("brak miniatury")

    def _set_selected(self, selected: bool) -> None:
        border = "#1d5fd1" if selected else "#d0d0d0"
        bg = "#eaf1ff" if selected else "#ffffff"
        self.setStyleSheet(
            f"#recordingCard {{background: {bg}; border: 1px solid {border}; border-radius: 8px;}}"
            "QLabel { color: #000000; }"
        )


class RecordingsBrowserDialog(QDialog):
    open_video = pyqtSignal(str)

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
        self._thumbnail_workers: Dict[str, ThumbnailWorker] = {}
        self._tile_items: Dict[str, QListWidgetItem] = {}
        self._tile_cards: Dict[str, RecordingCardWidget] = {}
        self._table_rows: Dict[str, int] = {}
        self._load_diagnostics: Dict[str, object] = {}
        self.thumbnail_pool = QThreadPool(self)

        self._class_options: Dict[str, str] = {str(c).casefold(): str(c) for c in VISIBLE_CLASSES}
        self._apply_light_theme()

        root = QVBoxLayout(self)
        root.addLayout(self._build_filters())
        root.addLayout(self._build_mode_row())
        root.addLayout(self._build_views())

        self.refresh()

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
        self.refresh_btn.clicked.connect(self.refresh)
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

    def refresh(self) -> None:
        self.refresh_btn.setEnabled(False)
        try:
            history_source = self._history_items if self._history_items is not None else self._history_path
            entries, diag = load_recording_entries(self._camera_dirs, history_source, prefer_catalog=True, allow_disk_fallback=True, heal_catalog=True)
            self._entries = entries
            self._load_diagnostics = diag

            self._ensure_class_filter_entries(entries)
            self._set_default_date_bounds(entries)
            self._apply_filters()
        finally:
            self.refresh_btn.setEnabled(True)

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
        self._rebuild_tile_view(self._filtered_entries)
        self._rebuild_table_view(self._filtered_entries)
        self._update_empty_state()
        self.status_label.setText(f"Wczytano {len(self._entries)} nagrań, widoczne {len(self._filtered_entries)}")


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
            card.set_loading_placeholder(self._placeholder_pixmap())
            self.tile_list.setItemWidget(item, card)
            key = self._thumb_cache_key(entry.filepath)
            self._tile_items[key] = item
            self._tile_cards[key] = card
            cached = self.thumbnail_cache.get(key)
            if cached is not None:
                if key in self._failed_thumbnails:
                    card.set_thumbnail_failed(cached)
                else:
                    card.set_thumbnail(cached)
            else:
                self._request_thumbnail(entry)

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
            thumb_lbl.setPixmap(self.thumbnail_cache.get(key, self._placeholder_pixmap()))
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
                self._request_thumbnail(entry)
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

    def _request_thumbnail(self, entry: RecordingMetadata) -> None:
        key = self._thumb_cache_key(entry.filepath)
        if key in self.thumbnail_cache or key in self._pending_thumbnails:
            return
        worker = ThumbnailWorker(entry)
        worker.thumbnail_ready.connect(self._apply_thumbnail, Qt.QueuedConnection)
        self._thumbnail_workers[key] = worker
        self._pending_thumbnails.add(key)
        self.thumbnail_pool.start(worker)

    @pyqtSlot(str, object)
    def _apply_thumbnail(self, filepath: str, image: object) -> None:
        key = self._thumb_cache_key(filepath)
        self._pending_thumbnails.discard(key)
        self._thumbnail_workers.pop(key, None)
        outcome = thumbnail_load_outcome(image)
        success = outcome == "success"
        pixmap = self._placeholder_pixmap()
        if success:
            pixmap = QPixmap.fromImage(image).scaled(self._thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._failed_thumbnails.discard(key)
        else:
            self._failed_thumbnails.add(key)
        self.thumbnail_cache[key] = pixmap

        card = self._tile_cards.get(key)
        if card is not None:
            if success:
                card.set_thumbnail(pixmap)
            else:
                card.set_thumbnail_failed(pixmap)

        row = self._table_rows.get(key)
        if row is not None:
            widget = self.table.cellWidget(row, 1)
            if isinstance(widget, QLabel):
                widget.setPixmap(pixmap)

    def _placeholder_pixmap(self) -> QPixmap:
        pix = QPixmap(self._thumb_size)
        pix.fill(QColor("#e9edf3"))
        painter = QPainter(pix)
        try:
            painter.setPen(QColor("#6b7280"))
            painter.drawText(pix.rect(), Qt.AlignCenter, "Ładowanie miniatury...")
        finally:
            painter.end()
        return pix

    def _switch_view(self, mode: str) -> None:
        current = self._current_selected_path()
        self._view_stack.setCurrentWidget(self.tile_list if mode == "Kafelki" else self.table)
        if current:
            self._restore_selection(current)

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
