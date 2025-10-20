"""Camera grid widget for multi-camera preview."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class CameraGridItem(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, index: int, name: str) -> None:
        super().__init__()
        self.index = index
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setStyleSheet("background:#000;")
        self.frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.frame_label, 1)

        self.name_label = QLabel(name)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("color:white; background:rgba(0,0,0,0.5);")
        layout.addWidget(self.name_label)

        self._pixmap: QPixmap | None = None
        self._placeholder = QPixmap(320, 180)
        self._placeholder.fill(Qt.black)
        self.frame_label.setPixmap(self._placeholder)

    def set_frame(self, frame: np.ndarray | None) -> None:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            self._pixmap = self._placeholder
            self._update_pixmap()
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self._pixmap = QPixmap.fromImage(qimg)
        except (cv2.error, ValueError, TypeError):
            self._pixmap = self._placeholder
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._pixmap is not None:
            self.frame_label.setPixmap(
                self._pixmap.scaled(
                    self.frame_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        self._update_pixmap()

    def mousePressEvent(self, event):  # type: ignore[override]
        self.clicked.emit(self.index)


class CameraGridWidget(QWidget):
    camera_clicked = pyqtSignal(int)

    def __init__(self, cameras: list[dict]):
        super().__init__()
        self.cameras = list(cameras)
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        self.items: list[CameraGridItem] = []
        self._build()

    def _build(self) -> None:
        for item in self.items:
            item.setParent(None)
        self.items = []
        for idx, cam in enumerate(self.cameras):
            item = CameraGridItem(idx, cam["name"])
            item.clicked.connect(self.camera_clicked.emit)
            self.items.append(item)
        self._reflow()

    def _reflow(self) -> None:
        while self.layout.count():
            self.layout.takeAt(0)
        n = len(self.items)
        if n == 0:
            return
        cols = int(np.ceil(np.sqrt(n)))
        for idx, item in enumerate(self.items):
            r = idx // cols
            c = idx % cols
            self.layout.addWidget(item, r, c)
            self.layout.setRowStretch(r, 1)
            self.layout.setColumnStretch(c, 1)

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        self._reflow()

    def rebuild(self, cameras: list[dict]) -> None:
        self.cameras = list(cameras)
        self._build()

    def update_frame(self, index: int, frame: np.ndarray | None) -> None:
        if 0 <= index < len(self.items):
            self.items[index].set_frame(frame)


__all__ = ["CameraGridItem", "CameraGridWidget"]
