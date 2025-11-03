from __future__ import annotations

import datetime as _dt
import os
from bisect import bisect_left
from contextlib import suppress
from typing import Any, Dict, List, Mapping, Sequence

import cv2
from PyQt5.QtCore import (
    QDate,
    QPoint,
    QRunnable,
    QSize,
    Qt,
    QThread,
    QThreadPool,
    QTimer,
    pyqtSignal,
    QObject,
)
from PyQt5.QtGui import QIcon, QImage, QImageReader, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..config import ALERTS_HISTORY_PATH, VISIBLE_CLASSES
from ..recordings import (
    CameraDirectory,
    RecordingMetadata,
    build_recording_metadata,
    iter_catalog_entries,
    load_history_metadata,
    walk_recordings,
)
from ..storage import remove_from_recordings_catalog


class RecordingsScanWorker(QObject):
    """Background worker that discovers recordings on disk."""

    record_discovered = pyqtSignal(RecordingMetadata)
    finished = pyqtSignal()

    def __init__(
        self,
        camera_dirs: Sequence[CameraDirectory],
        history_path: str | os.PathLike[str],
        history_items: Sequence[Mapping[str, object]] | Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        super().__init__()
        self._camera_dirs = list(camera_dirs)
        self._history_path = history_path
        self._history_items = history_items
        self._abort = False

    def stop(self) -> None:
        self._abort = True

    def run(self) -> None:  # pragma: no cover - exercised via GUI
        try:
            history_source = self._history_items if self._history_items is not None else self._history_path
            history = load_history_metadata(history_source)
            seen: set[str] = set()
            for path in walk_recordings(self._camera_dirs):
                if self._abort:
                    break
                entry = build_recording_metadata(str(path), self._camera_dirs, history_meta=history)
                seen.add(entry.filepath)
                self.record_discovered.emit(entry)
            if not self._abort:
                for entry in iter_catalog_entries(self._camera_dirs, history_meta=history):
                    if self._abort:
                        break
                    if entry.filepath in seen:
                        continue
                    seen.add(entry.filepath)
                    self.record_discovered.emit(entry)
        finally:
            self.finished.emit()


class ThumbnailWorker(QObject, QRunnable):
    """Asynchronously prepares preview images for recordings."""

    thumbnail_ready = pyqtSignal(str, QImage)

    def __init__(self, entry: RecordingMetadata, target_size: QSize):
        super().__init__()
        QRunnable.__init__(self)
        self._entry = entry
        self._size = target_size

    def run(self) -> None:  # pragma: no cover - exercised via GUI
        image = self._load_image()
        self.thumbnail_ready.emit(self._entry.filepath, image)

    def _load_image(self) -> QImage | QPixmap:
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
                return self._to_qimage(frame)

        placeholder = QImage(self._size.width(), self._size.height(), QImage.Format_RGB32)
        placeholder.fill(Qt.black)
        return placeholder

    def _thumbnail_candidates(self) -> List[str]:
        def _resolve(path: str) -> List[str]:
            if not path:
                return []
            resolved: List[str] = [path]
            if not os.path.isabs(path):
                resolved.append(os.path.join(os.path.dirname(self._entry.filepath), path))
            return [os.path.abspath(p) for p in resolved]

        candidates: List[str] = []
        if self._entry.thumb_path:
            candidates.extend(_resolve(self._entry.thumb_path))

        base, _ext = os.path.splitext(self._entry.filepath)
        for suffix in (".jpg", ".jpeg", ".JPG", ".JPEG"):
            candidates.append(os.path.abspath(f"{base}{suffix}"))

        for suffix in (".jpg", ".jpeg", ".JPG", ".JPEG"):
            candidates.append(os.path.abspath(f"{self._entry.filepath}{suffix}"))

        stem, ext = os.path.splitext(self._entry.filepath)
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

    def _load_thumbnail_file(self, path: str) -> QImage | QPixmap | None:
        if not os.path.exists(path):
            return None

        img = cv2.imread(path)
        if img is not None:
            return self._to_qimage(img)

        reader = QImageReader(path)
        reader.setAutoTransform(True)
        image = reader.read()
        if not image.isNull():
            return image

        pixmap = QPixmap()
        if pixmap.load(path):
            return pixmap

        return None

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

        for index in sample_indices:
            if index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if not self._is_dark(frame):
                return frame

        # Fallback: scan first few frames sequentially in case seeking failed.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(30):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if not self._is_dark(frame):
                return frame

        return None

    @staticmethod
    def _is_dark(frame: Any) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.mean(gray)[0]) < 12.0

    def _to_qimage(self, frame: Any) -> QImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format_RGB888,
        ).copy()
        return qimg.scaled(
            self._size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )


class RecordingsBrowserDialog(QDialog):
    """Interactive browser for reviewing, filtering and deleting recordings."""

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
        self.resize(1200, 720)

        self._camera_dirs = list(camera_dirs)
        self._history_path = str(history_path)
        self._history_items = [dict(item) for item in history_items] if history_items is not None else None
        self._entries: List[RecordingMetadata] = []
        self._entry_keys: List[tuple[float, str]] = []
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

        self._thumb_size = QSize(256, 144)
        self.scan_thread: QThread | None = None
        self._scan_worker: RecordingsScanWorker | None = None
        self.thumbnail_pool = QThreadPool()

        layout = QVBoxLayout(self)
        layout.addLayout(self._build_filters())
        layout.addWidget(self._build_table())

        QTimer.singleShot(0, self.refresh)

    # ------------------------------------------------------------------ UI --
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

        layout.addStretch(1)

        today = QDate.currentDate()
        self.date_to.setDate(today)
        self.date_from.setDate(today.addDays(-7))

        self.camera_filter.currentTextChanged.connect(self._apply_filters)
        self.class_filter.currentTextChanged.connect(self._apply_filters)
        self.date_from.dateChanged.connect(self._apply_filters)
        self.date_to.dateChanged.connect(self._apply_filters)
        self.search_edit.textChanged.connect(self._apply_filters)
        self.refresh_btn.clicked.connect(self.refresh)
        self.delete_btn.clicked.connect(self.delete_selected)

        return layout

    def _build_table(self) -> QTableWidget:
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Miniatura", "Czas", "Kamera", "Klasa", "Pewność", "Plik"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setIconSize(self._thumb_size)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setDefaultSectionSize(160)
        self.table.setColumnWidth(0, self._thumb_size.width())
        self.table.verticalHeader().setDefaultSectionSize(self._thumb_size.height() + 20)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.cellDoubleClicked.connect(self._cell_double_clicked)

        return self.table

    # --------------------------------------------------------------- actions --
    def refresh(self) -> None:
        self._stop_scan_worker()
        self._entries.clear()
        self._entry_keys.clear()
        self._row_lookup.clear()
        self._thumbnail_cache.clear()
        self._pending_thumbnails.clear()
        self._thumbnail_workers.clear()
        self._thumbnail_labels.clear()
        self.table.setRowCount(0)
        self.refresh_btn.setEnabled(False)
        self._min_date = None
        self._max_date = None

        worker = RecordingsScanWorker(
            self._camera_dirs,
            self._history_path,
            history_items=self._history_items,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.record_discovered.connect(self._on_entry_discovered)
        worker.finished.connect(self._handle_scan_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)
        thread.start()
        self._scan_worker = worker
        self.scan_thread = thread

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
        self._entry_keys = [(-entry.timestamp, entry.filename) for entry in self._entries]
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
    def _on_entry_discovered(self, entry: RecordingMetadata) -> None:
        key = (-entry.timestamp, entry.filename)
        idx = bisect_left(self._entry_keys, key)
        self._entry_keys.insert(idx, key)
        self._entries.insert(idx, entry)
        self._update_class_options(entry.label)
        self._update_date_range(entry)
        self._maybe_insert_row(entry, idx)

    def _on_scan_finished(self) -> None:
        self.refresh_btn.setEnabled(True)
        self._apply_filters()  # Ensure the latest data respects filters

    def _handle_scan_finished(self) -> None:
        try:
            self._on_scan_finished()
        finally:
            self._clear_scan_worker()

    def _cell_double_clicked(self, row: int, column: int) -> None:
        item = self.table.item(row, 0)
        if not item:
            return
        path = item.data(Qt.UserRole)
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
                self._cell_double_clicked(selected, 0)
        elif selected_action == delete_action:
            self.delete_selected()

    # --------------------------------------------------------------- helpers --
    def _selected_paths(self) -> List[str]:
        paths: List[str] = []
        for item in self.table.selectedItems():
            if item.column() != 0:
                continue
            path = item.data(Qt.UserRole)
            if path:
                paths.append(str(path))
        if not paths and self.table.currentRow() >= 0:
            item = self.table.item(self.table.currentRow(), 0)
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
                self.date_from.blockSignals(True)
                self.date_to.blockSignals(True)
                if self._min_date:
                    self.date_from.setDate(self._min_date)
                if self._max_date:
                    self.date_to.setDate(self._max_date)
            self.date_from.blockSignals(False)
            self.date_to.blockSignals(False)

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

    def _maybe_insert_row(self, entry: RecordingMetadata, index: int) -> None:
        if not self._matches_filters(entry):
            return
        row = self._count_visible_before(index)
        self._insert_row(entry, row)

    def _count_visible_before(self, index: int) -> int:
        visible = 0
        for entry in self._entries[:index]:
            if self._matches_filters(entry):
                visible += 1
        return visible

    def _insert_row(self, entry: RecordingMetadata, row: int | None = None) -> None:
        if row is None:
            row = self.table.rowCount()
        self.table.insertRow(row)
        for path, current_row in list(self._row_lookup.items()):
            if current_row >= row:
                self._row_lookup[path] = current_row + 1

        thumb_item = QTableWidgetItem()
        thumb_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        thumb_item.setData(Qt.UserRole, entry.filepath)
        self.table.setItem(row, 0, thumb_item)

        thumb_label = QLabel()
        thumb_label.setFixedSize(self._thumb_size)
        thumb_label.setAlignment(Qt.AlignCenter)
        thumb_label.setStyleSheet("border:1px solid #555; background:#111;")
        thumb_label.setPixmap(self._placeholder_pixmap())
        self.table.setCellWidget(row, 0, thumb_label)
        self._thumbnail_labels[entry.filepath] = thumb_label

        time_item = QTableWidgetItem(entry.display_time)
        time_item.setData(Qt.UserRole, entry.filepath)
        self.table.setItem(row, 1, time_item)

        cam_item = QTableWidgetItem(entry.camera)
        cam_item.setData(Qt.UserRole, entry.filepath)
        self.table.setItem(row, 2, cam_item)

        label_item = QTableWidgetItem(entry.label)
        label_item.setData(Qt.UserRole, entry.filepath)
        self.table.setItem(row, 3, label_item)

        conf_item = QTableWidgetItem("-" if entry.confidence <= 0 else f"{entry.confidence:.2f}")
        conf_item.setData(Qt.UserRole, entry.filepath)
        self.table.setItem(row, 4, conf_item)

        file_item = QTableWidgetItem(entry.filename)
        file_item.setData(Qt.UserRole, entry.filepath)
        self.table.setItem(row, 5, file_item)

        self._row_lookup[entry.filepath] = row
        if entry.filepath in self._thumbnail_cache:
            self._apply_thumbnail(entry.filepath, self._thumbnail_cache[entry.filepath])
        else:
            self._request_thumbnail(entry)

    def _placeholder_pixmap(self) -> QPixmap:
        if not hasattr(self, "_placeholder_pix"):
            pixmap = QPixmap(self._thumb_size)
            pixmap.fill(QColor("#111111"))
            setattr(self, "_placeholder_pix", pixmap)
        return getattr(self, "_placeholder_pix")

    def _compose_thumbnail(self, source: QImage | QPixmap) -> QPixmap:
        if isinstance(source, QImage):
            pixmap = QPixmap.fromImage(source)
        else:
            pixmap = source
        if pixmap.isNull():
            return self._placeholder_pixmap()

        scaled = pixmap.scaled(
            self._thumb_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        canvas = QPixmap(self._thumb_size)
        canvas.fill(QColor("#111111"))
        painter = QPainter(canvas)
        try:
            x = (self._thumb_size.width() - scaled.width()) // 2
            y = (self._thumb_size.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        finally:
            painter.end()
        return canvas

    def _request_thumbnail(self, entry: RecordingMetadata) -> None:
        if entry.filepath in self._pending_thumbnails or entry.filepath in self._thumbnail_cache:
            return
        worker = ThumbnailWorker(entry, self._thumb_size)
        worker.thumbnail_ready.connect(self._apply_thumbnail)
        self._pending_thumbnails.add(entry.filepath)
        self._thumbnail_workers[entry.filepath] = worker
        self.thumbnail_pool.start(worker)

    def _apply_thumbnail(self, filepath: str, image: QImage | QPixmap) -> None:
        pixmap = self._compose_thumbnail(image)
        self._thumbnail_cache[filepath] = pixmap
        self._pending_thumbnails.discard(filepath)
        self._thumbnail_workers.pop(filepath, None)
        row = self._row_lookup.get(filepath)
        if row is None:
            return
        item = self.table.item(row, 0)
        if item is not None:
            item.setIcon(QIcon(pixmap))
        label = self._thumbnail_labels.get(filepath)
        if label is not None:
            label.setPixmap(pixmap)

    def _apply_filters(self) -> None:
        if not self._entries:
            self.table.setRowCount(0)
            self._row_lookup.clear()
            return

        self.table.setRowCount(0)
        self._row_lookup.clear()
        self._thumbnail_labels.clear()
        for entry in self._entries:
            if self._matches_filters(entry):
                self._insert_row(entry)

    # ------------------------------------------------------------ lifecycle --
    def closeEvent(self, event):  # noqa: D401
        self._stop_scan_worker(wait=True)
        self.thumbnail_pool.clear()
        super().closeEvent(event)

    # --------------------------------------------------------- scan control --
    def _stop_scan_worker(self, wait: bool = False) -> None:
        worker = self._scan_worker
        thread = self.scan_thread
        if worker is None or thread is None:
            return
        worker.stop()
        thread.quit()
        if wait:
            thread.wait()
        self._clear_scan_worker()

    def _clear_scan_worker(self) -> None:
        self._scan_worker = None
        self.scan_thread = None
