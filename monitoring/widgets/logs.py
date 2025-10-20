"""Logging widgets."""

from __future__ import annotations

import datetime
import json
import os
import uuid
from typing import List

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..config import LOG_HISTORY_PATH, LOG_RETENTION_HOURS


class LogEntryWidget(QFrame):
    def __init__(
        self,
        entry_id: str,
        group: str,
        ts: str,
        camera: str = "",
        action: str = "",
        detected: str = "",
        recording: str = "",
    ) -> None:
        super().__init__()
        self.group = group
        self.entry_id = entry_id
        colors = {
            "application": "#4aa3ff",
            "detection": "#4caf50",
            "error": "#ff4444",
        }
        self.setStyleSheet("QFrame{border:none; background:rgba(0,0,0,0.4);}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setAlignment(Qt.AlignLeft)

        color = colors.get(group, "#fff")

        dt = None
        try:
            dt = datetime.datetime.strptime(ts, "%A %H:%M:%S %Y-%m-%d")
        except Exception:
            pass

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setAlignment(Qt.AlignLeft)
        header_widget.setStyleSheet(f"border-bottom:1px solid {color};")

        self.group_label = QLabel(group.upper())
        self.group_label.setAlignment(Qt.AlignLeft)
        self.group_label.setStyleSheet(f"color:{color}; font-size:15px; font-weight:600;")
        self.group_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        header_layout.addWidget(self.group_label)

        date_str = dt.strftime("%Y-%m-%d") if dt else ""
        self.date_label = QLabel(date_str)
        self.date_label.setAlignment(Qt.AlignLeft)
        self.date_label.setStyleSheet(f"color:{color}; font-size:15px;")
        header_layout.addWidget(self.date_label)
        header_layout.addStretch()

        layout.addWidget(header_widget)

        def add_line(text: str, text_color: str) -> None:
            label = QLabel(text)
            label.setAlignment(Qt.AlignLeft)
            label.setWordWrap(True)
            label.setStyleSheet(f"color:{text_color}; font-size:15px;")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            layout.addWidget(label)

        time_weekday_layout = QHBoxLayout()
        time_label = QLabel(dt.strftime("%H:%M:%S") if dt else ts)
        time_label.setAlignment(Qt.AlignLeft)
        time_label.setStyleSheet("color:#ff8800; font-size:15px;")
        weekday_str = dt.strftime("%A").capitalize() if dt else ""
        weekday_label = QLabel(weekday_str)
        weekday_label.setAlignment(Qt.AlignRight)
        weekday_label.setStyleSheet("color:#ff8800; font-size:15px;")
        time_weekday_layout.addWidget(time_label)
        time_weekday_layout.addStretch()
        time_weekday_layout.addWidget(weekday_label)
        layout.addLayout(time_weekday_layout)

        if camera:
            add_line(camera, "#4aa3ff")
        if detected:
            add_line(detected.upper(), "#4caf50")
        if group != "detection" and action:
            add_line(action, "#ff8800")

        if group == "detection":
            action_row = QHBoxLayout()
            self.rec_dot = QLabel()
            self.rec_dot.setFixedSize(10, 10)
            self.rec_text = QLabel()
            action_row.addWidget(self.rec_dot)
            action_row.addWidget(self.rec_text)
            layout.addLayout(action_row)
            self.rec_dot.hide()
            self.rec_text.hide()

            self._blink_timer = QTimer(self)
            self._blink_timer.timeout.connect(
                lambda: self.rec_dot.setVisible(not self.rec_dot.isVisible())
            )

            if recording == "started":
                self.start_recording()
            elif recording == "finished":
                self.finish_recording()
            elif recording == "det_started":
                self.start_detection()
            elif recording == "det_finished":
                self.finish_detection()
        else:
            self.rec_dot = QLabel()
            self.rec_text = QLabel()
            self._blink_timer = QTimer(self)

    def start_recording(self) -> None:
        self.rec_text.setText("Recording started")
        self.rec_text.setStyleSheet("color:red; font-size:15px;")
        self.rec_dot.setStyleSheet("background:red; border-radius:5px;")
        self.rec_dot.show()
        self.rec_text.show()
        self.rec_dot.setVisible(True)
        self._blink_timer.start(500)

    def finish_recording(self) -> None:
        self.rec_text.setText("Recording finished")
        self.rec_text.setStyleSheet("color:red; font-size:15px;")
        self.rec_dot.setStyleSheet("background:red; border-radius:5px;")
        self.rec_dot.show()
        self.rec_text.show()
        self._blink_timer.stop()
        self.rec_dot.setVisible(True)

    def start_detection(self) -> None:
        self.rec_text.setText("Detection started")
        self.rec_text.setStyleSheet("color:green; font-size:15px;")
        self.rec_dot.setStyleSheet("background:green; border-radius:5px;")
        self.rec_dot.show()
        self.rec_text.show()
        self.rec_dot.setVisible(True)
        self._blink_timer.start(500)

    def finish_detection(self) -> None:
        self.rec_text.setText("Detection finished")
        self.rec_text.setStyleSheet("color:green; font-size:15px;")
        self.rec_dot.setStyleSheet("background:green; border-radius:5px;")
        self.rec_dot.show()
        self.rec_text.show()
        self._blink_timer.stop()
        self.rec_dot.setVisible(True)


class LogWindow(QListWidget):
    """Widget prezentujący logi oraz zapisujący je do pliku."""

    def __init__(
        self,
        log_path: str = str(LOG_HISTORY_PATH),
        retention_hours: int = LOG_RETENTION_HOURS,
    ) -> None:
        super().__init__()
        self.setFixedWidth(300)
        self.setFrameShape(QFrame.NoFrame)
        self.setSpacing(8)
        self.setStyleSheet("QListWidget{background:transparent; border:none;}")
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.log_path = log_path
        self.retention_hours = retention_hours
        self.history: List[dict] = []

    def _add_widget_entry(self, entry: dict) -> None:
        widget = LogEntryWidget(
            entry.get("id", ""),
            entry.get("group", ""),
            entry.get("timestamp", ""),
            entry.get("camera", ""),
            entry.get("action", ""),
            entry.get("detected", ""),
            entry.get("recording", ""),
        )
        item = QListWidgetItem(self)
        self.addItem(item)
        self.setItemWidget(item, widget)
        item.setSizeHint(widget.sizeHint())

    def _refresh_widget(self) -> None:
        self.clear()
        for entry in self.history[-200:]:
            self._add_widget_entry(entry)
        if self.count():
            self.scrollToItem(self.item(self.count() - 1))

    def load_history(self) -> None:
        self.history = []
        try:
            if os.path.exists(self.log_path):
                with open(self.log_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    if isinstance(data, list):
                        self.history = data
                    else:
                        self.history = []
        except Exception:
            self.history = []

        cutoff = datetime.datetime.now() - datetime.timedelta(hours=self.retention_hours)
        filtered: List[dict] = []
        allowed = {"detection", "error", "application"}
        for entry in self.history:
            if entry.get("group") not in allowed:
                continue
            ts_str = entry.get("timestamp")
            try:
                ts_dt = datetime.datetime.strptime(ts_str, "%A %H:%M:%S %Y-%m-%d")
            except Exception:
                continue
            if ts_dt >= cutoff:
                filtered.append(entry)
        self.history = filtered
        self._refresh_widget()

    def add_entry(
        self,
        group: str,
        camera: str = "",
        action: str = "",
        detected: str = "",
    ) -> str:
        if group == "detection object":
            group = "detection"
        allowed = {"detection", "error", "application"}
        if group not in allowed:
            return ""

        ts = datetime.datetime.now().strftime("%A %H:%M:%S %Y-%m-%d")
        entry_id = uuid.uuid4().hex
        entry = {
            "id": entry_id,
            "group": group,
            "camera": camera,
            "action": action,
            "detected": detected,
            "timestamp": ts,
            "recording": "",
        }
        self.history.append(entry)

        cutoff = datetime.datetime.now() - datetime.timedelta(hours=self.retention_hours)
        filtered: List[dict] = []
        for item in self.history:
            try:
                ts_dt = datetime.datetime.strptime(
                    item.get("timestamp", ""), "%A %H:%M:%S %Y-%m-%d"
                )
            except Exception:
                continue
            if ts_dt >= cutoff:
                filtered.append(item)
        self.history = filtered

        self._refresh_widget()

        try:
            with open(self.log_path, "w", encoding="utf-8") as handle:
                json.dump(self.history, handle, indent=2)
        except Exception as exc:
            print("Nie udało się zapisać historii logów:", exc)
        return entry_id

    def update_recording_by_id(self, entry_id: str, status: str) -> None:
        for entry in self.history:
            if entry.get("id") == entry_id:
                entry["recording"] = status
                break

        try:
            with open(self.log_path, "w", encoding="utf-8") as handle:
                json.dump(self.history, handle, indent=2)
        except Exception as exc:
            print("Nie udało się zaktualizować logów:", exc)

    def get_recent_detections(self, limit: int = 10) -> List[dict]:
        detections = [e for e in self.history if e.get("group") == "detection"]
        return detections[-limit:]


__all__ = ["LogEntryWidget", "LogWindow"]
