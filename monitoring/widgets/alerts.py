"""Alert related widgets."""

from __future__ import annotations

import datetime
import os
from typing import List

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QEvent, QPoint, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..storage import AlertMemory


class AlertItemWidget(QWidget):
    BASE_STYLE = "QWidget{border:1px solid transparent; border-radius:10px; background:rgba(0,0,0,0.35);}"  # noqa: E501
    SELECTED_STYLE = "QWidget{border:2px solid #ff3333; border-radius:10px; background:rgba(255,0,0,0.08);}"  # noqa: E501
    COLOR_PALETTE = [
        "#f94144",
        "#f3722c",
        "#f8961e",
        "#f9844a",
        "#90be6d",
        "#43aa8b",
        "#577590",
        "#277da1",
        "#9b5de5",
        "#e07a5f",
    ]
    _label_color_map: dict[str, str] = {}

    def __init__(self, alert: dict, thumb_size: tuple[int, int] = (256, 144)) -> None:
        super().__init__()
        self.alert = alert
        self.setStyleSheet(self.BASE_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setAlignment(Qt.AlignTop)

        self.thumb = QLabel()
        self.thumb.setFixedSize(*thumb_size)
        self.thumb.setStyleSheet("border:1px solid #555; background:#111;")
        layout.addWidget(self.thumb, alignment=Qt.AlignCenter)

        camera = alert.get("camera", "?")
        label = alert.get("label", "object")
        confidence = float(alert.get("confidence", 0.0)) * 100.0
        timestamp = alert.get("time", "--:--:--")
        date_text, time_text = self._split_timestamp(timestamp)
        object_line = QLabel(f"OBIEKT: {label.upper()} ({confidence:.1f}%)")
        object_color = self._get_label_color(label)
        object_line.setStyleSheet(
            f"padding-top:6px; font-weight:bold; color:{object_color};"
        )
        camera_line = QLabel(f"KAMERA: {camera}")
        camera_line.setStyleSheet("color:#ddd;")
        date_line = QLabel(f"DATA: {date_text}")
        date_line.setStyleSheet("color:#ddd;")
        time_line = QLabel(f"GODZINA: {time_text}")
        time_line.setStyleSheet("color:#ddd;")

        for lbl in (object_line, camera_line, date_line, time_line):
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl, alignment=Qt.AlignCenter)

        frame = alert.get("frame")
        if frame is not None:
            self.set_frame(frame)
        else:
            thumb = alert.get("thumb")
            if not thumb:
                filepath = alert.get("filepath") or alert.get("file") or ""
                thumb = filepath + ".jpg" if filepath else ""
            if thumb and os.path.exists(thumb):
                image = cv2.imread(thumb)
                if image is not None:
                    self.set_frame(image)

    @classmethod
    def _get_label_color(cls, label: str) -> str:
        key = label.lower().strip() or "unknown"
        if key not in cls._label_color_map:
            color = cls.COLOR_PALETTE[len(cls._label_color_map) % len(cls.COLOR_PALETTE)]
            cls._label_color_map[key] = color
        return cls._label_color_map[key]

    @staticmethod
    def _split_timestamp(timestamp: str) -> tuple[str, str]:
        try:
            dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
        except Exception:
            if " " in timestamp:
                parts = timestamp.split(" ", 1)
                return parts[0].strip(), parts[1].strip()
            return timestamp, "--:--:--"

    def set_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimg = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.thumb.setPixmap(pixmap)

    def set_selected(self, selected: bool) -> None:
        self.setStyleSheet(self.SELECTED_STYLE if selected else self.BASE_STYLE)


class AlertListWidget(QWidget):
    open_video = pyqtSignal(str)

    def __init__(self, alert_memory: AlertMemory) -> None:
        super().__init__()
        self.mem = alert_memory
        self.setFixedWidth(300)
        self.setStyleSheet("background: transparent;")
        self._time_range_days = 1
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setFixedWidth(300)
        self.list.setFrameShape(QFrame.NoFrame)
        self.list.setStyleSheet(
            "\n".join(
                [
                    "QListWidget{background: transparent; border: none;}",
                    "QListWidget::item:selected{background: transparent; color: inherit;}",
                    "QListWidget::item:selected:active{background: transparent; color: inherit;}",
                    "QListWidget::item:selected:!active{background: transparent; color: inherit;}",
                ]
            )
        )
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.list)

        self.list.itemDoubleClicked.connect(self._open_selected)
        self.list.itemSelectionChanged.connect(self._update_selection_highlight)

        self.load_from_history(self.mem.items)

    @property
    def time_range_days(self) -> int:
        return self._time_range_days

    def set_time_range_days(self, days: int) -> None:
        days = max(1, min(7, int(days)))
        if days == self._time_range_days:
            return
        self._time_range_days = days
        self.load_from_history(self.mem.items)

    def add_alert(self, alert: dict) -> None:
        widget = AlertItemWidget(alert)
        item = QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        self.list.insertItem(0, item)
        self.list.setItemWidget(item, widget)
        self.list.scrollToItem(item, hint=QListWidget.PositionAtTop)
        self._update_selection_highlight()

    def load_from_history(self, items: List[dict]) -> None:
        self.list.clear()

        def parse_dt(alert: dict) -> datetime.datetime:
            text = alert.get("time", "")
            try:
                return datetime.datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
            except Exception:
                filepath = alert.get("filepath") or alert.get("file") or ""
                try:
                    return (
                        datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
                        if filepath and os.path.exists(filepath)
                        else datetime.datetime.min
                    )
                except Exception:
                    return datetime.datetime.min

        now = datetime.datetime.now()
        cutoff = now - datetime.timedelta(days=self._time_range_days)
        entries = [(alert, parse_dt(alert)) for alert in items[-300:]]
        entries.sort(key=lambda pair: pair[1], reverse=True)
        for alert, alert_dt in entries:
            if (
                alert_dt is not None
                and alert_dt != datetime.datetime.min
                and alert_dt < cutoff
            ):
                continue
            widget = AlertItemWidget(alert)
            item = QListWidgetItem()
            item.setSizeHint(widget.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, widget)
        if self.list.count():
            self.list.scrollToTop()
        self._update_selection_highlight()

    def _open_selected(self, item) -> None:
        widget = self.list.itemWidget(item)
        if isinstance(widget, AlertItemWidget):
            filepath = widget.alert.get("filepath") or widget.alert.get("file")
            if filepath and os.path.exists(filepath):
                self.open_video.emit(filepath)

    def _update_selection_highlight(self) -> None:
        selected_items = self.list.selectedItems()
        for index in range(self.list.count()):
            item = self.list.item(index)
            widget = self.list.itemWidget(item)
            if isinstance(widget, AlertItemWidget):
                widget.set_selected(item in selected_items)

    def reload(self) -> None:
        self.mem.load()
        self.load_from_history(self.mem.items)

    def export_csv(self) -> None:
        dialog = QFileDialog(self, "Eksport alertów do CSV")
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilter("CSV (*.csv)")
        dialog.selectFile("alerts.csv")
        dialog.setOption(QFileDialog.DontUseNativeDialog, False)
        dialog.setStyleSheet("")
        dialog.setPalette(QApplication.style().standardPalette())
        if dialog.exec_() != QDialog.Accepted:
            return
        paths = dialog.selectedFiles()
        if not paths:
            return
        path = paths[0]
        ok, err = self.mem.export_csv(path)
        if ok:
            QMessageBox.information(self, "Eksport", f"Zapisano: {path}")
        else:
            QMessageBox.warning(self, "Eksport", f"Błąd: {err}")

    def clear(self) -> None:
        if QMessageBox.question(
            self,
            "Pamięć alertów",
            "Czy na pewno wyczyścić pamięć alertów?",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self.mem.clear()
        self.list.clear()

    def delete_history_file(self) -> None:
        path = self.mem.path
        if QMessageBox.question(
            self,
            "Historia alertów",
            "Usunąć zapisany plik historii alertów?",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        try:
            if path.exists():
                path.unlink()
            self.mem.items = []
            self.list.clear()
            QMessageBox.information(
                self,
                "Historia alertów",
                "Plik historii został usunięty.",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Historia alertów",
                f"Nie udało się usunąć pliku historii: {exc}",
            )


class AlertDialog(QDialog):
    def __init__(self, main_window) -> None:
        super().__init__(main_window)
        self.mw = main_window
        self.setWindowTitle("Alerty")
        self.setPalette(QApplication.palette())
        self.setMinimumSize(600, 400)
        self.resize(600, 400)
        self._drag_offset: QPoint | None = None

        layout = QVBoxLayout(self)

        self.header_label = QLabel("Ustawienia alertów")
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet("font-size:16px; font-weight:bold;")
        self.header_label.setCursor(Qt.OpenHandCursor)
        self.header_label.installEventFilter(self)
        layout.addWidget(self.header_label)

        self.chk_visible = QCheckBox("Pokaż listę alertów")
        self.chk_visible.setChecked(self.mw.alert_list.isVisible())
        self.chk_visible.toggled.connect(self.mw.alert_list.setVisible)
        layout.addWidget(self.chk_visible)

        range_box = QVBoxLayout()
        range_title = QLabel("Zakres wczytywanych alertów")
        range_title.setStyleSheet("font-weight:bold;")
        range_box.addWidget(range_title)

        range_row = QHBoxLayout()
        self.range_slider = QSlider(Qt.Horizontal)
        self.range_slider.setMinimum(1)
        self.range_slider.setMaximum(7)
        self.range_slider.setPageStep(1)
        self.range_slider.setTickInterval(1)
        self.range_slider.setTickPosition(QSlider.TicksBelow)
        self.range_slider.setValue(self.mw.alert_list.time_range_days)
        range_row.addWidget(self.range_slider)
        self.range_label = QLabel()
        self.range_label.setMinimumWidth(150)
        range_row.addWidget(self.range_label)
        range_box.addLayout(range_row)
        layout.addLayout(range_box)

        self.range_slider.valueChanged.connect(self._update_time_range)
        self._update_time_range(self.range_slider.value())

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

        btn_delete = QPushButton("Usuń plik alertów")
        btn_delete.clicked.connect(self.mw.alert_list.delete_history_file)
        btn_layout.addWidget(btn_delete)

        layout.addLayout(btn_layout)
        layout.addStretch(1)

    def _update_time_range(self, days: int) -> None:
        days = max(1, min(7, int(days)))
        if days == 1:
            text = "ostatnie 24 godziny"
        else:
            text = f"ostatnie {days} dni"
        self.range_label.setText(text)
        self.mw.alert_list.set_time_range_days(days)

    def eventFilter(self, obj, event):
        if obj is self.header_label:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
                self.header_label.setCursor(Qt.ClosedHandCursor)
                return True
            if (
                event.type() == QEvent.MouseMove
                and event.buttons() & Qt.LeftButton
                and self._drag_offset is not None
            ):
                self.move(event.globalPos() - self._drag_offset)
                return True
            if event.type() == QEvent.MouseButtonRelease and self._drag_offset is not None:
                self._drag_offset = None
                self.header_label.setCursor(Qt.OpenHandCursor)
                return True
        return super().eventFilter(obj, event)


__all__ = ["AlertDialog", "AlertItemWidget", "AlertListWidget"]
