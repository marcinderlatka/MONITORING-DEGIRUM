# Naprawa: Widoczne miniaturki + brak białego tła (obsługa błędów ładowania)
from __future__ import annotations

import datetime as _dt
import os
from contextlib import suppress
from typing import Any, Dict, List, Mapping, Sequence

import cv2
import numpy as np
from PyQt5.QtCore import (
    QDate,
    QPoint,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    pyqtSignal,
    QObject,
)
from PyQt5.QtGui import QBrush, QColor, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import ALERTS_HISTORY_PATH, RECORDINGS_CATALOG_PATH, VISIBLE_CLASSES
from ..recordings import (
    CameraDirectory,
    RecordingMetadata,
    iter_catalog_entries,
    load_history_metadata,
)
from ..storage import remove_from_recordings_catalog


def _thumbnail_candidates_for_entry(entry: RecordingMetadata) -> List[str]:
    """Return possible thumbnail paths for a given recording entry."""

    def _resolve(path: str) -> List[str]:
        if not path:
            return []
        resolved: List[str] = [path]
        if not os.path.isabs(path):
            resolved.append(os.path.join(os.path.dirname(entry.filepath), path))
        return [os.path.abspath(p) for p in resolved]

    candidates: List[str] = []
    if entry.thumb_path:
        candidates.extend(_resolve(entry.thumb_path))

    base, _ext = os.path.splitext(entry.filepath)
    for suffix in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        candidates.append(os.path.abspath(f"{base}{suffix}"))

    for suffix in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        candidates.append(os.path.abspath(f"{entry.filepath}{suffix}"))

    stem, ext = os.path.splitext(entry.filepath)
    for replacement in (
        "_thumb.jpg",
        "_thumb.jpeg",
        "_preview.jpg",
        "_preview.jpeg",
        "_THUMB.JPG",
        "_THUMB.JPEG",
        "_PREVIEW.JPG",
        "_PREVIEW.JPEG",
    ):
        if ext:
            candidates.append(os.path.abspath(f"{stem}{replacement}"))

    seen: set[str] = set()
    ordered: List[str] = []
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


class ThumbnailWorker(QObject, QRunnable):
    """Asynchronously prepares preview images for recordings."""

    thumbnail_ready = pyqtSignal(str, object)

    def __init__(self, entry: RecordingMetadata, target_size: QSize):
        super().__init__()
        QRunnable.__init__(self)
        self._entry = entry
        self._size = target_size

    def run(self) -> None:  # pragma: no cover - exercised via GUI
        frame = self._load_image()
        self.thumbnail_ready.emit(self._entry.filepath, frame)

    def _load_image(self) -> np.ndarray | None:
        for candidate in self._thumbnail_candidates():
            image = self._load_thumbnail_file(candidate)
            if image is not None:
                return image

        if os.path.exists(self._entry.filepath):
            cap = cv2.VideoCapture(self._entry.filepath)
            try:
                frame = self._extract_preview_frame(cap)
            finally:
                cap.release()
            if frame is not None:
                return frame

        return None

    def _thumbnail_candidates(self) -> List[str]:
        return _thumbnail_candidates_for_entry(self._entry)

    def _load_thumbnail_file(self, path: str) -> object | None:
        """Najprostsze i najpewniejsze ładowanie miniatury (bez QImageReader)."""
        from PyQt5.QtGui import QPixmap

        if not os.path.exists(path):
            print("⚠️  Miniatura nie istnieje:", path)
            return None

        pix = QPixmap(path)
        if pix.isNull():
            print("❌ Nie udało się wczytać (pusty QPixmap):", path)
            return None
        print("✅ Wczytano miniaturę:", path)
        return pix

    def _extract_preview_frame(self, cap: cv2.VideoCapture) -> Any:
        """Pick a representative non-dark frame from the video if possible."""

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_indices: List[int] = []
        if frame_count > 0:
            sample_indices.extend(
                sorted(
                    {
                        max(0, min(frame_count - 1, int(frame_count * ratio)))
                        for ratio in (0.05, 0.15, 0.3, 0.5)
                    }
                )
            )
        sample_indices.append(0)

        fallback: Any | None = None

        for index in sample_indices:
            if index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if not self._is_dark(frame):
                return frame
            if fallback is None:
                fallback = frame

        # Fallback: scan first few frames sequentially in case seeking failed.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(30):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if not self._is_dark(frame):
                return frame
            if fallback is None:
                fallback = frame

        return fallback

    @staticmethod
    def _is_dark(frame: Any) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.mean(gray)[0]) < 12.0


class RecordingsBrowserDialog(QDialog):
    """Interactive browser for reviewing, filtering and deleting recordings."""

    open_video = pyqtSignal(str)

    CHECK_COLUMN = 0
    THUMB_COLUMN = 1
    TIME_COLUMN = 2
    CAMERA_COLUMN = 3
    CLASS_COLUMN = 4
    CONF_COLUMN = 5
    FILE_COLUMN = 6

    def __init__(
        self,
        camera_dirs: Sequence[CameraDirectory],
        parent: QObject | None = None,
        history_path: str | os.PathLike[str] = ALERTS_HISTORY_PATH,
        history_items: Sequence[Mapping[str, object]] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Nagrania – przeglądarka")
        self.resize(1200, 720)
        self._apply_light_theme()

        self._block_item_changed = False
        self._syncing_select_all = False

        self._camera_dirs = list(camera_dirs)
        self._history_path = str(history_path)
        self._history_items = [dict(item) for item in history_items] if history_items is not None else None
        self._entries: List[RecordingMetadata] = []
        self._row_lookup: Dict[str, int] = {}
        self._thumbnail_cache: Dict[str, QPixmap] = {}
        self._pending_thumbnails: set[str] = set()
        self._thumbnail_workers: Dict[str, ThumbnailWorker] = {}
        self._thumbnail_labels: Dict[str, QLabel] = {}
        self._class_options: Dict[str, str] = {
            cls.casefold(): cls for cls in VISIBLE_CLASSES
        }
        self._min_date: QDate | None = None
        self._max_date: QDate | None = None

        self._thumb_size = QSize(200, 150)
        self._placeholder_pixmap_obj = self._create_placeholder_pixmap()
        self.thumbnail_pool = QThreadPool()

        layout = QVBoxLayout(self)
        layout.addLayout(self._build_filters())
        layout.addWidget(self._build_table())

        QTimer.singleShot(0, self.refresh)

    # ------------------------------------------------------------------ UI --
    def _apply_light_theme(self) -> None:
        palette = QPalette()
        background = QColor("#ffffff")
        surface = QColor("#f6f6f6")
        accent = QColor("#1d5fd1")
        text = QColor("#000000")
        palette.setColor(QPalette.Window, background)
        palette.setColor(QPalette.Base, QColor("#ffffff"))
        palette.setColor(QPalette.AlternateBase, surface)
        palette.setColor(QPalette.Text, text)
        palette.setColor(QPalette.Button, QColor("#ffffff"))
        palette.setColor(QPalette.ButtonText, text)
        palette.setColor(QPalette.Highlight, accent)
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        palette.setColor(QPalette.WindowText, text)
        palette.setColor(QPalette.ToolTipBase, QColor("#ffffdc"))
        palette.setColor(QPalette.ToolTipText, text)
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.setStyleSheet(
            """
            QDialog { background-color: #ffffff; color: #000000; }
            QLabel { color: #000000; }
            QLineEdit, QComboBox, QDateEdit {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #c6c6c6;
                padding: 4px 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #000000;
                selection-background-color: #1d5fd1;
                selection-color: #ffffff;
            }
            QPushButton {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #bdbdbd;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #f5f5f5;
                color: #000000;
                gridline-color: #d0d0d0;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                color: #000000;
                padding: 4px;
                border: 0px;
                border-right: 1px solid #d0d0d0;
            }
        """
        )

    def _configure_table_palette(self) -> None:
        if not hasattr(self, "table"):
            return
        palette = self.table.palette()
        palette.setColor(QPalette.Base, QColor("#ffffff"))
        palette.setColor(QPalette.AlternateBase, QColor("#f5f5f5"))
        palette.setColor(QPalette.Text, QColor("#000000"))
        palette.setColor(QPalette.Highlight, QColor("#1d5fd1"))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        palette.setColor(QPalette.Button, QColor("#ffffff"))
        palette.setColor(QPalette.ButtonText, QColor("#000000"))
        self.table.setPalette(palette)

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

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Filtruj po nazwie pliku lub etykiecie...")
        layout.addWidget(self.search_edit, stretch=1)

        self.refresh_btn = QPushButton("Odśwież")
        layout.addWidget(self.refresh_btn)

        self.delete_btn = QPushButton("Usuń zaznaczone")
        layout.addWidget(self.delete_btn)

        self.select_all_checkbox = QCheckBox("Zaznacz wszystko")
        self.select_all_checkbox.setTristate(True)
        self.select_all_checkbox.setCheckState(Qt.Unchecked)
        layout.addWidget(self.select_all_checkbox)

        layout.addStretch(1)

        today = QDate.currentDate()
        self.date_to.setDate(today)
        self.date_from.setDate(today.addDays(-1))

        self.camera_filter.currentTextChanged.connect(self._apply_filters)
        self.class_filter.currentTextChanged.connect(self._apply_filters)
        self.date_from.dateChanged.connect(self._apply_filters)
        self.date_to.dateChanged.connect(self._apply_filters)
        self.search_edit.textChanged.connect(self._apply_filters)
        self.refresh_btn.clicked.connect(self.refresh)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.select_all_checkbox.stateChanged.connect(self._handle_select_all_changed)

        return layout

    def _build_table(self) -> QWidget:
        container = QWidget()
        self._table_stack = QStackedLayout(container)

        self.loading_label = QLabel("Wczytuję Twoje nagrania...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setWordWrap(True)
        self.loading_label.setStyleSheet(
            "color: #3a3a3a; font-size: 18px; background-color: #ffffff;"
        )
        self.loading_label.setMargin(40)
        self._table_stack.addWidget(self.loading_label)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "Usuń",
                "Miniatura",
                "Czas",
                "Kamera",
                "Klasa",
                "Pewność",
                "Plik",
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(False)
        self.table.setIconSize(self._thumb_size)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setStyleSheet(
            "QTableWidget::item{selection-background-color: transparent;"
            " background: transparent; border:0.5px solid transparent;}"
            "QTableWidget::item:selected{selection-background-color: transparent;"
            " background: transparent; color: inherit; border:0.5px solid transparent;}"
        )
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setDefaultSectionSize(160)
        self.table.setColumnWidth(self.CHECK_COLUMN, 90)
        self.table.setColumnWidth(self.THUMB_COLUMN, self._thumb_size.width())
        self.table.verticalHeader().setDefaultSectionSize(self._thumb_size.height() + 20)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.cellDoubleClicked.connect(self._cell_double_clicked)
        self.table.itemChanged.connect(self._handle_item_changed)
        self.table.itemSelectionChanged.connect(self._update_row_highlight)

        self._configure_table_palette()

        self._table_stack.addWidget(self.table)
        self._table_stack.setCurrentWidget(self.loading_label)
        return container

    # --------------------------------------------------------------- actions --
    def load_recordings(self) -> List[RecordingMetadata]:
        catalog_path = os.path.abspath(str(RECORDINGS_CATALOG_PATH))
        print(f"[RecordingsBrowser] Ładowanie katalogu nagrań z: {catalog_path}")
        print(f"[RecordingsBrowser] Historia alertów: {self._history_path}")
        try:
            entries = self._load_entries_from_catalog()
        except Exception as exc:  # pragma: no cover - diagnostyka GUI
            print(f"[RecordingsBrowser] Błąd odczytu katalogu: {exc}")
            return []

        if not entries:
            print("[RecordingsBrowser] Nie znaleziono żadnych wpisów w katalogu.")
            return []

        print(f"[RecordingsBrowser] Wczytano {len(entries)} wpisów.")
        for entry in entries:
            thumb_path = self._resolve_thumbnail_path(entry)
            if thumb_path and os.path.exists(thumb_path):
                print(f"  • {entry.filepath} | thumb: {thumb_path}")
            else:
                missing = thumb_path or "(brak informacji)"
                print(f"  • {entry.filepath} | thumb brak ({missing})")
        return entries

    def refresh(self) -> None:
        self.refresh_btn.setEnabled(False)
        self._set_loading_state(True)

        try:
            self._entries = []
            self._row_lookup.clear()
            self._thumbnail_cache.clear()
            self._pending_thumbnails.clear()
            self._thumbnail_workers.clear()
            self._thumbnail_labels.clear()
            self.table.setRowCount(0)
            self._min_date = None
            self._max_date = None
            self._set_select_all_state(Qt.Unchecked)

            entries = self.load_recordings()
            self._entries = list(entries)

            for entry in self._entries:
                self._update_class_options(entry.label)
                self._update_date_range(entry)

            self._apply_filters()
        finally:
            self.refresh_btn.setEnabled(True)
            self._set_loading_state(False)

    def _set_loading_state(self, loading: bool) -> None:
        if not hasattr(self, "_table_stack"):
            return
        target = self.loading_label if loading else self.table
        if self._table_stack.currentWidget() is not target:
            self._table_stack.setCurrentWidget(target)
        self.loading_label.setVisible(loading)

    def delete_selected(self) -> None:
        paths = self._selected_paths()
        if not paths:
            QMessageBox.information(self, "Usuń nagrania", "Nie wybrano żadnych nagrań.")
            return

        if len(paths) == 1:
            msg = f"Czy na pewno usunąć nagranie?\n\n{os.path.basename(paths[0])}"
        else:
            msg = f"Czy na pewno usunąć {len(paths)} nagrań?"

        if (
            QMessageBox.question(
                self,
                "Potwierdzenie",
                msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return

        errors: List[str] = []
        deleted = len(paths)
        for fp in paths:
            for candidate in (fp, fp + ".json", fp + ".jpg"):
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
            self._thumbnail_cache.pop(path, None)
            self._pending_thumbnails.discard(path)

        self._apply_filters()

        if errors:
            QMessageBox.warning(
                self,
                "Usunięto z błędami",
                "Usunięto: {} (część z błędami):\n- {}".format(deleted, "\n- ".join(errors)),
            )
        else:
            QMessageBox.information(self, "Usunięto", f"Usunięto {deleted} nagrań.")

    # -------------------------------------------------------------- handlers --
    def _load_entries_from_catalog(self) -> List[RecordingMetadata]:
        history_source: (
            Mapping[str, Mapping[str, object]]
            | Sequence[Mapping[str, object]]
            | str
        )
        if self._history_items is not None:
            history_source = self._history_items
        else:
            history_source = self._history_path

        history = load_history_metadata(history_source)
        try:
            entries = list(iter_catalog_entries(self._camera_dirs, history_meta=history))
        except Exception:
            entries = []

        entries.sort(key=lambda item: (-item.timestamp, item.filename))
        return entries

    def _cell_double_clicked(self, row: int, column: int) -> None:
        path = self._row_filepath(row)
        if path and os.path.exists(path):
            self.open_video.emit(path)

    def _context_menu(self, pos: QPoint) -> None:
        menu = QMenu(self)
        open_action = menu.addAction("Otwórz")
        delete_action = menu.addAction("Usuń")
        selected_action = menu.exec_(self.table.mapToGlobal(pos))
        if selected_action == open_action:
            selected = self.table.currentRow()
            if selected >= 0:
                self._cell_double_clicked(selected, self.THUMB_COLUMN)
        elif selected_action == delete_action:
            self.delete_selected()

    # --------------------------------------------------------------- helpers --
    def _row_filepath(self, row: int) -> str | None:
        for column in range(self.table.columnCount()):
            item = self.table.item(row, column)
            if not item:
                continue
            path = item.data(Qt.UserRole)
            if path:
                return str(path)
        return None

    def _handle_item_changed(self, item: QTableWidgetItem) -> None:
        if self._block_item_changed:
            return
        if item.column() != self.CHECK_COLUMN:
            return
        self._sync_select_all_checkbox()

    def _handle_select_all_changed(self, state: int) -> None:
        if self._syncing_select_all:
            return
        if state == Qt.PartiallyChecked:
            return
        target = Qt.Checked if state == Qt.Checked else Qt.Unchecked
        self._set_all_checkboxes(target)

    def _set_all_checkboxes(self, state: Qt.CheckState) -> None:
        self._block_item_changed = True
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.CHECK_COLUMN)
            if item:
                item.setCheckState(state)
        self._block_item_changed = False
        self._sync_select_all_checkbox()

    def _set_select_all_state(self, state: Qt.CheckState) -> None:
        if not hasattr(self, "select_all_checkbox"):
            return
        self._syncing_select_all = True
        self.select_all_checkbox.setCheckState(state)
        self._syncing_select_all = False

    def _sync_select_all_checkbox(self) -> None:
        if not hasattr(self, "select_all_checkbox"):
            return
        total = self.table.rowCount()
        checked = 0
        for row in range(total):
            item = self.table.item(row, self.CHECK_COLUMN)
            if item and item.checkState() == Qt.Checked:
                checked += 1
        if total == 0 or checked == 0:
            state = Qt.Unchecked
        elif checked == total:
            state = Qt.Checked
        else:
            state = Qt.PartiallyChecked
        self._set_select_all_state(state)

    def _selected_paths(self) -> List[str]:
        paths: List[str] = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.CHECK_COLUMN)
            if not item:
                continue
            if item.checkState() == Qt.Checked:
                path = item.data(Qt.UserRole)
                if path:
                    paths.append(str(path))
        if paths:
            return paths

        for item in self.table.selectedItems():
            if item.column() != self.THUMB_COLUMN:
                continue
            path = item.data(Qt.UserRole)
            if path:
                paths.append(str(path))
        if not paths and self.table.currentRow() >= 0:
            item = self.table.item(self.table.currentRow(), self.THUMB_COLUMN)
            if item:
                path = item.data(Qt.UserRole)
                if path:
                    paths.append(str(path))
        return paths

    def _update_class_options(self, label: str) -> None:
        if not label:
            return
        key = label.casefold()
        if key in self._class_options:
            return
        self._class_options[key] = label
        self.class_filter.addItem(label)

    def _update_date_range(self, entry: RecordingMetadata) -> None:
        dt = _dt.datetime.fromtimestamp(entry.timestamp)
        qdate = QDate(dt.year, dt.month, dt.day)
        changed = False
        if self._min_date is None or qdate < self._min_date:
            self._min_date = qdate
            changed = True
        if self._max_date is None or qdate > self._max_date:
            self._max_date = qdate
            changed = True
        if changed:
            with suppress(Exception):
                if self._min_date:
                    self.date_from.setMinimumDate(self._min_date)
                    self.date_to.setMinimumDate(self._min_date)
                if self._max_date:
                    self.date_from.setMaximumDate(self._max_date)
                    self.date_to.setMaximumDate(self._max_date)

    def _matches_filters(self, entry: RecordingMetadata) -> bool:
        camera_sel = self.camera_filter.currentText()
        if (
            camera_sel
            and camera_sel != "Wszystkie kamery"
            and entry.camera.casefold() != camera_sel.casefold()
        ):
            return False
        class_sel = self.class_filter.currentText()
        if (
            class_sel
            and class_sel != "Wszystkie klasy"
            and entry.label.casefold() != class_sel.casefold()
        ):
            return False

        qfrom = self.date_from.date()
        qto = self.date_to.date()
        dt = _dt.datetime.fromtimestamp(entry.timestamp)
        qdate = QDate(dt.year, dt.month, dt.day)
        if qdate < qfrom or qdate > qto:
            return False

        needle = self.search_edit.text().strip().lower()
        if needle:
            haystack = f"{entry.filename} {entry.label}".lower()
            if needle not in haystack:
                return False
        return True

    def _insert_row(self, entry: RecordingMetadata, row: int | None = None) -> None:
        previous_block = self._block_item_changed
        self._block_item_changed = True
        if row is None:
            row = self.table.rowCount()
        self.table.insertRow(row)
        for path, current_row in list(self._row_lookup.items()):
            if current_row >= row:
                self._row_lookup[path] = current_row + 1

        check_item = QTableWidgetItem()
        check_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        check_item.setCheckState(Qt.Unchecked)
        check_item.setData(Qt.UserRole, entry.filepath)
        check_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, self.CHECK_COLUMN, check_item)

        thumb_item = QTableWidgetItem()
        thumb_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        thumb_item.setData(Qt.UserRole, entry.filepath)
        thumb_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, self.THUMB_COLUMN, thumb_item)

        thumb_label = QLabel()
        thumb_label.setFixedSize(self._thumb_size)
        thumb_label.setAlignment(Qt.AlignCenter)
        thumb_label.setStyleSheet(
            "border: 0.5px solid #d0d0d0; background-color: {};".format(
                self._thumbnail_background_color().name()
            )
        )
        thumb_label.setPixmap(self._placeholder_pixmap())
        self.table.setCellWidget(row, self.THUMB_COLUMN, thumb_label)
        self._thumbnail_labels[entry.filepath] = thumb_label

        time_item = QTableWidgetItem(entry.display_time)
        time_item.setData(Qt.UserRole, entry.filepath)
        time_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, self.TIME_COLUMN, time_item)

        cam_item = QTableWidgetItem(entry.camera)
        cam_item.setData(Qt.UserRole, entry.filepath)
        cam_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, self.CAMERA_COLUMN, cam_item)

        label_item = QTableWidgetItem(entry.label)
        label_item.setData(Qt.UserRole, entry.filepath)
        label_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, self.CLASS_COLUMN, label_item)

        conf_item = QTableWidgetItem("-" if entry.confidence <= 0 else f"{entry.confidence:.2f}")
        conf_item.setData(Qt.UserRole, entry.filepath)
        conf_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, self.CONF_COLUMN, conf_item)

        file_text = self._format_file_cell_text(entry)
        file_item = QTableWidgetItem(file_text)
        file_item.setData(Qt.UserRole, entry.filepath)
        if "\n" in file_text:
            file_item.setToolTip(file_text)
        file_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, self.FILE_COLUMN, file_item)

        self._row_lookup[entry.filepath] = row
        if entry.filepath in self._thumbnail_cache:
            self._apply_thumbnail(entry.filepath, self._thumbnail_cache[entry.filepath])
        else:
            self._request_thumbnail(entry)
        self._block_item_changed = previous_block
        self._sync_select_all_checkbox()
        self._update_row_highlight()

    def _create_placeholder_pixmap(self) -> QPixmap:
        pixmap = QPixmap(self._thumb_size)
        pixmap.fill(QColor("#f0f0f0"))
        return pixmap

    def _placeholder_pixmap(self) -> QPixmap:
        if not hasattr(self, "_placeholder_pixmap_obj"):
            self._placeholder_pixmap_obj = self._create_placeholder_pixmap()
        return self._placeholder_pixmap_obj

    def _thumbnail_background_color(self) -> QColor:
        return QColor("#f8f8f8")

    def _request_thumbnail(self, entry: RecordingMetadata) -> None:
        if entry.filepath in self._pending_thumbnails or entry.filepath in self._thumbnail_cache:
            return
        thumb_path = self._resolve_thumbnail_path(entry)
        direct_pixmap = self._load_pixmap_from_disk(thumb_path)
        if direct_pixmap is not None:
            self._apply_thumbnail(entry.filepath, direct_pixmap)
            return
        worker = ThumbnailWorker(entry, self._thumb_size)
        worker.thumbnail_ready.connect(self._apply_thumbnail)
        self._pending_thumbnails.add(entry.filepath)
        self._thumbnail_workers[entry.filepath] = worker
        self.thumbnail_pool.start(worker)

    def _scale_pixmap(self, pixmap: QPixmap) -> QPixmap:
        if pixmap.isNull():
            return self._placeholder_pixmap()

        scaled = pixmap.scaled(
            self._thumb_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        if scaled.isNull() or not scaled.width() or not scaled.height():
            return self._placeholder_pixmap()

        return scaled

    def _normalise_qimage(self, image: QImage) -> QImage:
        if image.isNull():
            return image

        fmt = image.format()
        if fmt == QImage.Format_Invalid:
            return QImage()

        def formats(*names: str) -> set[int]:
            return {getattr(QImage, name) for name in names if hasattr(QImage, name)}

        grayscale_formats = formats(
            "Format_Indexed8",
            "Format_Alpha8",
            "Format_Grayscale8",
            "Format_Grayscale16",
            "Format_Mono",
            "Format_MonoLSB",
        )

        rgb_like = formats(
            "Format_RGB888",
            "Format_BGR888",
            "Format_RGB16",
            "Format_RGB555",
            "Format_RGB666",
            "Format_RGB444",
            "Format_RGB30",
            "Format_BGR30",
        )

        if fmt in grayscale_formats:
            target = QImage.Format_Grayscale8
        elif fmt in rgb_like:
            target = QImage.Format_RGB888
        else:
            # Always drop alpha to avoid fully transparent thumbnails rendering black.
            target = QImage.Format_RGB888

        if fmt == target and not image.hasAlphaChannel():
            return image.copy()

        converted = image.convertToFormat(target)
        if converted.isNull():
            return QImage()
        return converted.copy()

    def _qimage_from_frame(self, frame: np.ndarray) -> QImage:
        if frame.size == 0:
            return QImage()

        array = frame
        if array.dtype != np.uint8:
            try:
                array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX)
            except cv2.error:
                return QImage()
            array = array.astype(np.uint8)

        if array.ndim == 2:
            gray = np.ascontiguousarray(array)
            height, width = gray.shape
            return QImage(
                gray.data,
                width,
                height,
                int(gray.strides[0]),
                QImage.Format_Grayscale8,
            ).copy()

        if array.ndim != 3:
            return QImage()

        height, width, channels = array.shape

        try:
            if channels == 1:
                gray = np.ascontiguousarray(array.reshape(height, width))
                return QImage(
                    gray.data,
                    width,
                    height,
                    int(gray.strides[0]),
                    QImage.Format_Grayscale8,
                ).copy()

            if channels == 3:
                rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                return QImage(
                    rgb.data,
                    width,
                    height,
                    int(rgb.strides[0]),
                    QImage.Format_RGB888,
                ).copy()

            if channels == 4:
                rgba = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
                rgba = np.ascontiguousarray(rgba)
                return QImage(
                    rgba.data,
                    width,
                    height,
                    int(rgba.strides[0]),
                    QImage.Format_RGBA8888,
                ).copy()
        except cv2.error:
            return QImage()

        return QImage()


    def _pixmap_from_frame(self, frame: np.ndarray) -> QPixmap:
        image = self._qimage_from_frame(frame)
        if image.isNull():
            return self._placeholder_pixmap()
        return self._scale_pixmap(QPixmap.fromImage(image))

    def _compose_thumbnail(self, source: object) -> QPixmap:
        if isinstance(source, QPixmap):
            return self._scale_pixmap(source)
        if isinstance(source, QImage):
            return self._scale_pixmap(QPixmap.fromImage(source))
        if isinstance(source, np.ndarray):
            return self._pixmap_from_frame(source)
        return self._placeholder_pixmap()

    def _apply_thumbnail(self, filepath: str, image: object) -> None:
        # 🔹 Jeśli już mamy gotowy QPixmap – nie konwertujemy
        if isinstance(image, QPixmap) and not image.isNull():
            pixmap = image.scaled(self._thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        elif isinstance(image, np.ndarray):
            pixmap = self._pixmap_from_frame(image)
        elif isinstance(image, QImage) and not image.isNull():
            pixmap = QPixmap.fromImage(image)
            pixmap = self._scale_pixmap(pixmap)
        else:
            pixmap = self._placeholder_pixmap()

        # 🔹 Cache + aktualizacja widoku
        self._thumbnail_cache[filepath] = pixmap
        self._pending_thumbnails.discard(filepath)
        self._thumbnail_workers.pop(filepath, None)

        row = self._row_lookup.get(filepath)
        if row is None:
            return

        label = self._thumbnail_labels.get(filepath)
        if label is not None:
            label.setPixmap(pixmap)

    def _apply_filters(self) -> None:
        if not self._entries:
            self.table.setRowCount(0)
            self._row_lookup.clear()
            self._thumbnail_labels.clear()
            self._sync_select_all_checkbox()
            return

        self.table.setRowCount(0)
        self._row_lookup.clear()
        self._thumbnail_labels.clear()
        for entry in self._entries:
            if self._matches_filters(entry):
                self._insert_row(entry)
        self._sync_select_all_checkbox()
        self._update_row_highlight()

    def _format_file_cell_text(self, entry: RecordingMetadata) -> str:
        mp4_path = entry.filepath
        thumb_path = self._resolve_thumbnail_path(entry)
        if thumb_path and thumb_path != mp4_path:
            return "\n".join(
                [self._shorten_path(mp4_path), self._shorten_path(thumb_path)]
            )
        return self._shorten_path(mp4_path)

    def _shorten_path(self, path: str, max_length: int = 60) -> str:
        if not path:
            return ""
        normalized = os.path.normpath(str(path))
        if len(normalized) <= max_length:
            return normalized
        ellipsis = "…"
        keep = max(max_length - len(ellipsis), 4)
        head = keep // 2
        tail = keep - head
        return f"{normalized[:head]}{ellipsis}{normalized[-tail:]}"

    def _resolve_thumbnail_path(self, entry: RecordingMetadata) -> str:
        candidates = _thumbnail_candidates_for_entry(entry)
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0] if candidates else ""

    def _load_pixmap_from_disk(self, thumb_path: str) -> QPixmap | None:
        if not thumb_path:
            return None
        if not os.path.exists(thumb_path):
            print(f"[RecordingsBrowser] Brak miniatury na dysku: {thumb_path}")
            return None
        pixmap = QPixmap(thumb_path)
        if pixmap.isNull():
            print(f"[RecordingsBrowser] Nieprawidłowy plik miniatury: {thumb_path}")
            return None
        return pixmap

    def _update_row_highlight(self) -> None:
        if not hasattr(self, "table"):
            return
        selection = self.table.selectionModel()
        if selection is None:
            return
        selected_rows = {index.row() for index in selection.selectedRows()}
        transparent_brush = QBrush()
        for row in range(self.table.rowCount()):
            is_selected = row in selected_rows
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is not None:
                    item.setBackground(transparent_brush)
            check_item = self.table.item(row, self.CHECK_COLUMN)
            filepath = check_item.data(Qt.UserRole) if check_item else ""
            thumb_label = self._thumbnail_labels.get(filepath)
            if thumb_label:
                thumb_label.setStyleSheet(
                    "border: 0.5px solid {border}; background-color: {bg};".format(
                        border="#ff3333" if is_selected else "#d0d0d0",
                        bg="rgba(255,0,0,0.05)"
                        if is_selected
                        else self._thumbnail_background_color().name(),
                    )
                )

    # ------------------------------------------------------------ lifecycle --
    def closeEvent(self, event):  # noqa: D401
        self.thumbnail_pool.clear()
        super().closeEvent(event)
