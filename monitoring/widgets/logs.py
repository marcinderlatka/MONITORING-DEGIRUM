"""Logging widgets."""

from __future__ import annotations

import datetime
import json
import os
import uuid
from typing import List

from PyQt5.QtCore import QEvent, QPoint, Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..config import LOG_HISTORY_PATH, LOG_RETENTION_HOURS


class LogEntryWidget(QFrame):
    BASE_STYLE = (
        "#logEntry{border:0.5px solid transparent; border-radius:10px;"
        " background:rgba(0,0,0,0.4);}"  # noqa: E501
    )
    SELECTED_STYLE = (
        "#logEntry{border:0.5px solid #ff3333; border-radius:10px;"
        " background:rgba(255,0,0,0.05);}"  # noqa: E501
    )
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
        self.setObjectName("logEntry")
        self.setStyleSheet(self.BASE_STYLE)
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

    def set_selected(self, selected: bool) -> None:
        self.setStyleSheet(self.SELECTED_STYLE if selected else self.BASE_STYLE)

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
        self.setStyleSheet(
            "\n".join(
                [
                    "QListWidget{background:transparent; border:none;}",
                    "QListWidget::item:selected{background: transparent; color: inherit;}",
                    "QListWidget::item:selected:active{background: transparent; color: inherit;}",
                    "QListWidget::item:selected:!active{background: transparent; color: inherit;}",
                ]
            )
        )
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.itemSelectionChanged.connect(self._update_selection_highlight)
        self._selected_rows: set[int] = set()

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
        self._update_selection_highlight()

    def _refresh_widget(self) -> None:
        self.clear()
        for entry in self.history[-200:]:
            self._add_widget_entry(entry)
        if self.count():
            self.scrollToItem(self.item(self.count() - 1))
        self._update_selection_highlight()

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
        allowed = {"detection", "error", "application", "settings"}
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
        allowed = {"detection", "error", "application", "settings"}
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

    def set_retention_hours(self, hours: int) -> None:
        hours = max(1, int(hours))
        if self.retention_hours == hours:
            return
        self.retention_hours = hours
        self.load_history()

    def clear_history(self) -> None:
        self.history = []
        self._refresh_widget()
        try:
            with open(self.log_path, "w", encoding="utf-8") as handle:
                json.dump(self.history, handle, indent=2)
        except Exception as exc:
            print("Nie udało się wyczyścić historii logów:", exc)

    def delete_history_file(self) -> None:
        self.history = []
        self._refresh_widget()
        try:
            if os.path.exists(self.log_path):
                os.remove(self.log_path)
        except Exception as exc:
            print("Nie udało się usunąć pliku logów:", exc)

    def reload(self) -> None:
        self.load_history()

    def _update_selection_highlight(self) -> None:
        selected_rows = {index.row() for index in self.selectedIndexes()}
        changed_rows = selected_rows.symmetric_difference(self._selected_rows)
        if not changed_rows:
            return
        self._selected_rows = selected_rows
        for row in changed_rows:
            if row < 0 or row >= self.count():
                continue
            item = self.item(row)
            widget = self.itemWidget(item)
            if isinstance(widget, LogEntryWidget):
                widget.set_selected(row in selected_rows)

    def get_recent_detections(self, limit: int = 10) -> List[dict]:
        detections = [e for e in self.history if e.get("group") == "detection"]
        return detections[-limit:]


class LogSettingsDialog(QDialog):
    def __init__(self, main_window) -> None:
        super().__init__(main_window)
        self.mw = main_window
        self.setWindowTitle("Logi")
        self.setPalette(QApplication.palette())
        self.setMinimumSize(400, 250)
        self.resize(450, 260)
        self._drag_offset: QPoint | None = None

        layout = QVBoxLayout(self)

        self.header_label = QLabel("Ustawienia logów")
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet("font-size:16px; font-weight:bold;")
        self.header_label.setCursor(Qt.OpenHandCursor)
        self.header_label.installEventFilter(self)
        layout.addWidget(self.header_label)

        self.chk_visible = QCheckBox("Pokaż panel logów")
        self.chk_visible.setChecked(self.mw.log_window.isVisible())
        self.chk_visible.toggled.connect(self.mw.log_window.setVisible)
        layout.addWidget(self.chk_visible)

        retention_box = QVBoxLayout()
        retention_title = QLabel("Retencja logów (w godzinach)")
        retention_title.setStyleSheet("font-weight:bold;")
        retention_box.addWidget(retention_title)

        retention_row = QHBoxLayout()
        self.retention_slider = QSlider(Qt.Horizontal)
        self.retention_slider.setMinimum(1)
        self.retention_slider.setMaximum(24 * 7)
        self.retention_slider.setPageStep(6)
        self.retention_slider.setTickInterval(6)
        self.retention_slider.setTickPosition(QSlider.TicksBelow)
        slider_value = max(
            self.retention_slider.minimum(),
            min(self.retention_slider.maximum(), self.mw.log_window.retention_hours),
        )
        self.retention_slider.setValue(slider_value)
        retention_row.addWidget(self.retention_slider)

        self.retention_label = QLabel()
        self.retention_label.setMinimumWidth(160)
        retention_row.addWidget(self.retention_label)
        retention_box.addLayout(retention_row)
        layout.addLayout(retention_box)

        self.retention_slider.valueChanged.connect(self._update_retention)
        self._update_retention(self.retention_slider.value())

        btn_layout = QHBoxLayout()

        btn_reload = QPushButton("Wczytaj ponownie")
        btn_reload.clicked.connect(self.mw.log_window.reload)
        btn_layout.addWidget(btn_reload)

        btn_clear = QPushButton("Wyczyść")
        btn_clear.clicked.connect(self._clear_logs)
        btn_layout.addWidget(btn_clear)

        btn_delete = QPushButton("Usuń plik")
        btn_delete.clicked.connect(self._delete_logs)
        btn_layout.addWidget(btn_delete)

        layout.addLayout(btn_layout)
        layout.addStretch(1)

    def _update_retention(self, hours: int) -> None:
        hours = max(1, int(hours))
        if hours < 24:
            text = f"ostatnie {hours}h"
        else:
            days = hours / 24.0
            rounded = int(days) if days.is_integer() else round(days, 1)
            suffix = "dzień" if rounded == 1 else "dni"
            text = f"ostatnie {rounded} {suffix}"
        self.retention_label.setText(text)
        self.mw.update_log_retention_hours(hours)

    def _clear_logs(self) -> None:
        if (
            QMessageBox.question(
                self,
                "Logi",
                "Czy na pewno wyczyścić bieżącą historię logów?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        self.mw.log_window.clear_history()

    def _delete_logs(self) -> None:
        if (
            QMessageBox.question(
                self,
                "Logi",
                "Usunąć plik logów z dysku?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        self.mw.log_window.delete_history_file()

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


__all__ = ["LogEntryWidget", "LogWindow", "LogSettingsDialog"]
