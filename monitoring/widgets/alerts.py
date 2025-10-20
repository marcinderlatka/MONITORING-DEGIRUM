"""Alert related widgets."""

from __future__ import annotations

import datetime
import os
from typing import List

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
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
    QVBoxLayout,
    QWidget,
)

from ..storage import AlertMemory


class AlertItemWidget(QWidget):
    def __init__(self, alert: dict, thumb_size: tuple[int, int] = (256, 144)) -> None:
        super().__init__()
        self.alert = alert
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
        self.meta = QLabel(f"{camera}\n{timestamp} — {label} ({confidence:.1f}%)")
        self.meta.setStyleSheet("padding-top:6px; color:#ddd;")
        self.meta.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.meta, alignment=Qt.AlignCenter)

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

    def set_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimg = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.thumb.setPixmap(pixmap)


class AlertListWidget(QWidget):
    open_video = pyqtSignal(str)

    def __init__(self, alert_memory: AlertMemory) -> None:
        super().__init__()
        self.mem = alert_memory
        self.setFixedWidth(300)
        self.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setFixedWidth(300)
        self.list.setFrameShape(QFrame.NoFrame)
        self.list.setStyleSheet("QListWidget{background: transparent; border: none;}")
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.list)

        self.list.itemDoubleClicked.connect(self._open_selected)

        self.load_from_history(self.mem.items)

    def add_alert(self, alert: dict) -> None:
        widget = AlertItemWidget(alert)
        item = QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        self.list.insertItem(0, item)
        self.list.setItemWidget(item, widget)
        self.list.scrollToItem(item, hint=QListWidget.PositionAtTop)

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

        sorted_items = sorted(items[-300:], key=parse_dt, reverse=True)
        for alert in sorted_items:
            widget = AlertItemWidget(alert)
            item = QListWidgetItem()
            item.setSizeHint(widget.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, widget)
        if self.list.count():
            self.list.scrollToTop()

    def _open_selected(self, item) -> None:
        widget = self.list.itemWidget(item)
        if isinstance(widget, AlertItemWidget):
            filepath = widget.alert.get("filepath") or widget.alert.get("file")
            if filepath and os.path.exists(filepath):
                self.open_video.emit(filepath)

    def reload(self) -> None:
        self.mem.load()
        self.load_from_history(self.mem.items)

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Eksport alertów do CSV", "alerts.csv", "CSV (*.csv)"
        )
        if not path:
            return
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


class AlertDialog(QDialog):
    def __init__(self, main_window) -> None:
        super().__init__(main_window)
        self.mw = main_window
        self.setWindowTitle("Alerty")
        self.setPalette(QApplication.palette())

        layout = QVBoxLayout(self)

        self.chk_visible = QCheckBox("Pokaż listę alertów")
        self.chk_visible.setChecked(self.mw.alert_list.isVisible())
        self.chk_visible.toggled.connect(self.mw.alert_list.setVisible)
        layout.addWidget(self.chk_visible)

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

        layout.addLayout(btn_layout)


__all__ = ["AlertDialog", "AlertItemWidget", "AlertListWidget"]
