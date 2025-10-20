"""Camera list widgets."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFrame, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget


class CameraListWidgetItem(QWidget):
    def __init__(self, camera_name: str) -> None:
        super().__init__()
        self.setStyleSheet(
            """
            QWidget#CameraCard {
                background: transparent;
                border: none;
            }
            QLabel#CamName {
                font-weight: 600;
                color: #e6e6e6;
                padding: 6px 8px 2px 8px;
            }
            """
        )
        self.setObjectName("CameraCard")

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self.text_label = QLabel(camera_name)
        self.text_label.setObjectName("CamName")
        self.text_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.text_label)

        self.icon_label = QLabel()
        self.thumb_w, self.thumb_h = 192, 108
        self.icon_label.setFixedSize(self.thumb_w, self.thumb_h)
        self.icon_label.setFrameShape(QFrame.NoFrame)
        self.icon_label.setStyleSheet("background: transparent; border: none;")
        self._placeholder = QPixmap(self.thumb_w, self.thumb_h)
        self._placeholder.fill(Qt.black)
        self.icon_label.setPixmap(self._placeholder)
        root.addWidget(self.icon_label, alignment=Qt.AlignCenter)

    def set_thumbnail(self, frame: np.ndarray | None) -> None:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            self.icon_label.setPixmap(self._placeholder)
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(rgb, (self.thumb_w, self.thumb_h), interpolation=cv2.INTER_AREA)
            qimg = QImage(
                image.data,
                self.thumb_w,
                self.thumb_h,
                image.strides[0],
                QImage.Format_RGB888,
            )
            self.icon_label.setPixmap(QPixmap.fromImage(qimg))
        except (cv2.error, ValueError, TypeError):
            self.icon_label.setPixmap(self._placeholder)


class CameraListWidget(QListWidget):
    request_context = pyqtSignal(int, QPoint)

    def __init__(self, cameras: list[dict]):
        super().__init__()
        self.setFixedWidth(300)
        self.setSpacing(12)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("QListWidget{ background: transparent; border: none; padding:8px; }")
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.widgets: list[CameraListWidgetItem] = []
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
        for cam in cameras:
            widget = CameraListWidgetItem(cam["name"])
            item = QListWidgetItem(self)
            item.setSizeHint(widget.sizeHint())
            self.addItem(item)
            self.setItemWidget(item, widget)
            self.widgets.append(widget)

    def update_thumbnail(self, index: int, frame: np.ndarray | None) -> None:
        if 0 <= index < len(self.widgets):
            self.widgets[index].set_thumbnail(frame)

    def rebuild(self, cameras: list[dict]) -> None:
        self.clear()
        self.widgets = []
        for cam in cameras:
            widget = CameraListWidgetItem(cam["name"])
            item = QListWidgetItem(self)
            item.setSizeHint(widget.sizeHint())
            self.addItem(item)
            self.setItemWidget(item, widget)
            self.widgets.append(widget)

    def _on_context_menu(self, pos: QPoint) -> None:
        row = self.currentRow()
        item = self.itemAt(pos)
        if item is not None:
            row = self.row(item)
        if row < 0:
            return
        self.request_context.emit(row, self.mapToGlobal(pos))


__all__ = ["CameraListWidget", "CameraListWidgetItem"]
