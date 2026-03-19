from __future__ import annotations

import datetime
import json
import os
import sys
import uuid
from threading import Lock, Timer
from typing import Any, List

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
from ..log_messages import msg, summarize_performance_details

SUPPORTED_LOG_GROUPS = {
    "detection",
    "error",
    "warning",
    "application",
    "settings",
    "worker",
    "ui",
    "browser",
    "recording",
    "performance",
}


class LogEntryWidget(QFrame):
    BASE_STYLE = (
        "#logEntry{border:0.5px solid transparent; border-radius:10px;"
        " background:rgba(0,0,0,0.4);}"
    )
    SELECTED_STYLE = (
        "#logEntry{border:0.5px solid #ff3333; border-radius:10px;"
        " background:rgba(255,0,0,0.05);}"
    )

    def __init__(self, entry: dict[str, Any]) -> None:
        super().__init__()
        self.group = str(entry.get("group", "application"))
        self.entry_id = str(entry.get("id", ""))
        level = str(entry.get("level", "INFO")).upper()
        source = str(entry.get("source", "")).strip()
        camera = str(entry.get("camera", "")).strip()
        action = str(entry.get("action", "")).strip()
        details = str(entry.get("details", "")).strip()
        traceback_text = str(entry.get("traceback", "")).strip()
        detected = str(entry.get("detected", "")).strip()
        recording = str(entry.get("recording", "")).strip()
        ts = str(entry.get("timestamp", ""))

        colors = {
            "application": "#4aa3ff",
            "detection": "#4caf50",
            "error": "#ff4444",
            "warning": "#ffb300",
            "worker": "#8bc34a",
            "ui": "#8e44ad",
            "browser": "#8bc6ff",
            "recording": "#29b6f6",
            "performance": "#ffd54f",
            "settings": "#90caf9",
        }
        if level in {"ERROR", "CRITICAL"}:
            color = "#ff4444"
        elif level == "WARNING" or self.group == "warning":
            color = "#ffb300"
        else:
            color = colors.get(self.group, "#fff")

        self.setObjectName("logEntry")
        self.setStyleSheet(self.BASE_STYLE)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setAlignment(Qt.AlignLeft)

        dt = None
        try:
            dt = datetime.datetime.strptime(ts, "%A %H:%M:%S %Y-%m-%d")
        except Exception:
            pass

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_widget.setStyleSheet(f"border-bottom:1px solid {color};")

        group_title = self.group.upper()
        if source:
            group_title += f" • {source}"
        self.group_label = QLabel(group_title)
        self.group_label.setWordWrap(True)
        self.group_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.group_label.setStyleSheet(f"color:{color}; font-size:15px; font-weight:600;")
        header_layout.addWidget(self.group_label)

        date_str = dt.strftime("%Y-%m-%d") if dt else ""
        self.date_label = QLabel(date_str)
        self.date_label.setWordWrap(True)
        self.date_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.date_label.setStyleSheet(f"color:{color}; font-size:15px;")
        header_layout.addStretch()
        header_layout.addWidget(self.date_label)
        layout.addWidget(header_widget)

        def add_line(text: str, text_color: str = "#ddd", font_size: int = 14) -> QLabel:
            label = QLabel(text)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignLeft)
            label.setStyleSheet(f"color:{text_color}; font-size:{font_size}px;")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            layout.addWidget(label)
            return label

        time_weekday_layout = QHBoxLayout()
        time_label = QLabel(dt.strftime("%H:%M:%S") if dt else ts)
        time_label.setWordWrap(True)
        time_label.setStyleSheet("color:#ddd; font-size:14px;")
        time_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        time_weekday_layout.addWidget(time_label)
        time_weekday_layout.addStretch()
        weekday_label = QLabel(dt.strftime("%A").capitalize() if dt else "")
        weekday_label.setWordWrap(True)
        weekday_label.setStyleSheet("color:#ddd; font-size:14px;")
        weekday_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        weekday_label.setAlignment(Qt.AlignRight)
        time_weekday_layout.addWidget(weekday_label)
        layout.addLayout(time_weekday_layout)

        if camera:
            add_line(f"{msg('camera_label')}: {camera}", "#4aa3ff")
        if detected:
            add_line(f"{msg('detection_label')}: {detected.upper()}", "#4caf50")
        if action:
            add_line(action, "#ff8800" if self.group == "detection" else "#ddd")

        preview = details or traceback_text
        if preview:
            self.details_label = add_line("", "#cfcfcf")
            if self.group == "performance":
                self._details_short_text = summarize_performance_details(preview)
                self._details_full_text = f"{msg('details_technical_prefix')}: {preview}"
                self._details_expanded = False
                self.details_toggle_btn = QPushButton(msg("show_more"))
                self.details_toggle_btn.setStyleSheet("font-size:13px; color:#8bc6ff; text-align:left;")
                self.details_toggle_btn.setFlat(True)
                self.details_toggle_btn.setCursor(Qt.PointingHandCursor)
                self.details_toggle_btn.clicked.connect(self._toggle_details)
                layout.addWidget(self.details_toggle_btn)
            else:
                self._details_full_text = f"{msg('details_prefix')}: {preview}"
                self._details_short_text = self._details_full_text
                self._details_expanded = True
                if len(self._details_full_text) > 900:
                    self._details_short_text = f"{self._details_full_text[:900]}…"
                    self._details_expanded = False
                    self.details_toggle_btn = QPushButton(msg("show_more"))
                    self.details_toggle_btn.setStyleSheet("font-size:13px; color:#8bc6ff; text-align:left;")
                    self.details_toggle_btn.setFlat(True)
                    self.details_toggle_btn.setCursor(Qt.PointingHandCursor)
                    self.details_toggle_btn.clicked.connect(self._toggle_details)
                    layout.addWidget(self.details_toggle_btn)
                else:
                    self.details_toggle_btn = None
            self._update_details_text()
        else:
            self.details_label = None
            self.details_toggle_btn = None

        if self.group == "detection":
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
            self._blink_timer.timeout.connect(lambda: self.rec_dot.setVisible(not self.rec_dot.isVisible()))
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

    def _update_details_text(self) -> None:
        if self.details_label is None:
            return
        self.details_label.setText(self._details_full_text if self._details_expanded else self._details_short_text)
        if self.details_toggle_btn is not None:
            self.details_toggle_btn.setText(msg("show_less") if self._details_expanded else msg("show_more"))

    def _toggle_details(self) -> None:
        self._details_expanded = not self._details_expanded
        self._update_details_text()

    def set_selected(self, selected: bool) -> None:
        self.setStyleSheet(self.SELECTED_STYLE if selected else self.BASE_STYLE)

    def start_recording(self) -> None:
        self.rec_text.setText(msg("recording_started"))
        self.rec_text.setStyleSheet("color:red; font-size:15px;")
        self.rec_dot.setStyleSheet("background:red; border-radius:5px;")
        self.rec_dot.show(); self.rec_text.show(); self.rec_dot.setVisible(True)
        self._blink_timer.start(500)

    def finish_recording(self) -> None:
        self.rec_text.setText(msg("recording_finished"))
        self.rec_text.setStyleSheet("color:red; font-size:15px;")
        self.rec_dot.setStyleSheet("background:red; border-radius:5px;")
        self.rec_dot.show(); self.rec_text.show(); self._blink_timer.stop(); self.rec_dot.setVisible(True)

    def start_detection(self) -> None:
        self.rec_text.setText(msg("detection_started"))
        self.rec_text.setStyleSheet("color:green; font-size:15px;")
        self.rec_dot.setStyleSheet("background:green; border-radius:5px;")
        self.rec_dot.show(); self.rec_text.show(); self.rec_dot.setVisible(True)
        self._blink_timer.start(500)

    def finish_detection(self) -> None:
        self.rec_text.setText(msg("detection_finished"))
        self.rec_text.setStyleSheet("color:green; font-size:15px;")
        self.rec_dot.setStyleSheet("background:green; border-radius:5px;")
        self.rec_dot.show(); self.rec_text.show(); self._blink_timer.stop(); self.rec_dot.setVisible(True)


class _DebouncedHistoryWriter:
    def __init__(self, path: str, debounce_seconds: float = 1.0) -> None:
        self.path = path
        self.debounce_seconds = debounce_seconds
        self._timer: Timer | None = None
        self._lock = Lock()
        self._pending: list[dict[str, Any]] | None = None

    def schedule(self, payload: list[dict[str, Any]]) -> None:
        with self._lock:
            self._pending = list(payload)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = Timer(self.debounce_seconds, self.flush)
            self._timer.daemon = True
            self._timer.start()

    def flush(self) -> None:
        with self._lock:
            payload = self._pending
            self._pending = None
            timer = self._timer
            self._timer = None
        if timer is not None:
            timer.cancel()
        if payload is None:
            return
        try:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:
            print(f"Log history save failed: {exc}", file=sys.stderr)


class LogWindow(QListWidget):
    """Widget prezentujący logi oraz zapisujący je do pliku."""

    VISIBLE_HISTORY_LIMIT = 200
    RETENTION_CHECK_INTERVAL_MS = 60_000

    def __init__(self, log_path: str = str(LOG_HISTORY_PATH), retention_hours: int = LOG_RETENTION_HOURS) -> None:
        super().__init__()
        self.setMinimumWidth(280)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setFrameShape(QFrame.NoFrame)
        self.setSpacing(8)
        self.setStyleSheet("\n".join([
            "QListWidget{background:transparent; border:none;}",
            "QListWidget::item:selected{background: transparent; color: inherit;}",
            "QListWidget::item:selected:active{background: transparent; color: inherit;}",
            "QListWidget::item:selected:!active{background: transparent; color: inherit;}",
        ]))
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.itemSelectionChanged.connect(self._update_selection_highlight)
        self._selected_rows: set[int] = set()
        self.log_path = log_path
        self.retention_hours = retention_hours
        self.history: List[dict[str, Any]] = []
        self._history_writer = _DebouncedHistoryWriter(self.log_path, debounce_seconds=1.0)
        self._retention_timer = QTimer(self)
        self._retention_timer.setInterval(self.RETENTION_CHECK_INTERVAL_MS)
        self._retention_timer.timeout.connect(self._run_retention_cycle)
        self._retention_timer.start()

    @staticmethod
    def normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            "id": str(entry.get("id") or uuid.uuid4().hex),
            "group": str(entry.get("group") or "application"),
            "camera": str(entry.get("camera") or ""),
            "action": str(entry.get("action") or ""),
            "detected": str(entry.get("detected") or ""),
            "timestamp": str(entry.get("timestamp") or datetime.datetime.now().strftime("%A %H:%M:%S %Y-%m-%d")),
            "recording": str(entry.get("recording") or ""),
            "level": str(entry.get("level") or "INFO"),
            "source": str(entry.get("source") or ""),
            "details": str(entry.get("details") or ""),
            "traceback": str(entry.get("traceback") or ""),
        }
        if normalized["group"] == "detection object":
            normalized["group"] = "detection"
        if normalized["group"] not in SUPPORTED_LOG_GROUPS:
            normalized["group"] = "application"
        return normalized

    def _schedule_history_persist(self) -> None:
        self._history_writer.schedule(self.history)

    def flush_history_persist(self) -> None:
        self._history_writer.flush()

    def _add_widget_entry(self, entry: dict[str, Any]) -> None:
        widget = LogEntryWidget(entry)
        item = QListWidgetItem(self)
        self.addItem(item)
        self.setItemWidget(item, widget)
        self._sync_item_size(item, widget)
        self._update_selection_highlight()

    def _sync_item_size(self, item: QListWidgetItem, widget: QWidget) -> None:
        width = max(1, self.viewport().width() - 2 * self.frameWidth() - self.spacing())
        hinted = widget.sizeHint()
        hinted.setWidth(width)
        item.setSizeHint(hinted)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        for row in range(self.count()):
            item = self.item(row)
            widget = self.itemWidget(item)
            if item is None or widget is None:
                continue
            self._sync_item_size(item, widget)

    def _refresh_widget(self) -> None:
        self.clear()
        for entry in self.history[-self.VISIBLE_HISTORY_LIMIT :]:
            self._add_widget_entry(entry)
        if self.count():
            self.scrollToItem(self.item(self.count() - 1))
        self._update_selection_highlight()

    def _append_incremental(self, entry: dict[str, Any]) -> None:
        self._add_widget_entry(entry)
        while self.count() > self.VISIBLE_HISTORY_LIMIT:
            self.takeItem(0)
        if self.count():
            self.scrollToItem(self.item(self.count() - 1))

    def _run_retention_cycle(self) -> None:
        before = len(self.history)
        if before == 0:
            return
        filtered = self._retention_filtered(self.history)
        if len(filtered) == before:
            return
        self.history = filtered
        self._refresh_widget()
        self._schedule_history_persist()

    def _retention_filtered(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cutoff = datetime.datetime.now() - datetime.timedelta(hours=self.retention_hours)
        filtered: list[dict[str, Any]] = []
        for raw in entries:
            entry = self.normalize_entry(raw)
            try:
                ts_dt = datetime.datetime.strptime(entry["timestamp"], "%A %H:%M:%S %Y-%m-%d")
            except Exception:
                continue
            if ts_dt >= cutoff:
                filtered.append(entry)
        return filtered

    def load_history(self) -> None:
        self.history = []
        try:
            if os.path.exists(self.log_path):
                with open(self.log_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    if isinstance(data, list):
                        self.history = self._retention_filtered(data)
        except Exception:
            self.history = []
        self._refresh_widget()

    def add_structured_entry(self, entry: dict[str, Any]) -> str:
        normalized = self.normalize_entry(entry)
        self.history.append(normalized)
        self._append_incremental(normalized)
        self._schedule_history_persist()
        return normalized["id"]

    def add_entry(self, group: str, camera: str = "", action: str = "", detected: str = "") -> str:
        return self.add_structured_entry({"group": group, "camera": camera, "action": action, "detected": detected})

    def update_recording_by_id(self, entry_id: str, status: str) -> None:
        for entry in self.history:
            if entry.get("id") == entry_id:
                entry["recording"] = status
                break
        self._schedule_history_persist()

    def set_retention_hours(self, hours: int) -> None:
        hours = max(1, int(hours))
        if self.retention_hours == hours:
            return
        self.retention_hours = hours
        self._run_retention_cycle()

    def clear_history(self) -> None:
        self.history = []
        self._refresh_widget()
        self._schedule_history_persist()

    def delete_history_file(self) -> None:
        self.history = []
        self._refresh_widget()
        try:
            if os.path.exists(self.log_path):
                os.remove(self.log_path)
        except Exception as exc:
            print(f"Log history delete failed: {exc}", file=sys.stderr)

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


# Keep existing dialog class from original file appended later.

class LogSettingsDialog(QDialog):
    GROUP_OPTIONS = [
        "detection",
        "error",
        "warning",
        "worker",
        "ui",
        "recording",
        "performance",
        "application",
        "settings",
        "browser",
    ]
    LEVEL_OPTIONS = ["INFO", "WARNING", "ERROR", "CRITICAL"]
    SOURCE_OPTIONS = ["worker", "ui", "app"]

    def __init__(self, main_window) -> None:
        super().__init__(main_window)
        self.mw = main_window
        self.setWindowTitle("Logi")
        self.setPalette(QApplication.palette())
        self.setMinimumSize(400, 250)
        self.resize(450, 260)
        self._drag_offset: QPoint | None = None
        self._group_checks: dict[str, QCheckBox] = {}
        self._level_checks: dict[str, QCheckBox] = {}
        self._source_checks: dict[str, QCheckBox] = {}

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
        self._build_filters_section(layout)

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

    def _build_filters_section(self, root_layout: QVBoxLayout) -> None:
        filters = self.mw.current_log_filters()
        container = QWidget()
        box_layout = QVBoxLayout(container)
        box_layout.setContentsMargins(10, 8, 10, 8)
        title = QLabel("Filtry logów")
        title.setStyleSheet("font-size:14px; font-weight:bold;")
        box_layout.addWidget(title)

        box_layout.addWidget(self._create_checkbox_group(
            "Grupy",
            self.GROUP_OPTIONS,
            set(filters.get("groups", [])),
            self._group_checks,
        ))
        box_layout.addWidget(self._create_checkbox_group(
            "Poziomy",
            self.LEVEL_OPTIONS,
            set(filters.get("levels", [])),
            self._level_checks,
        ))
        box_layout.addWidget(self._create_checkbox_group(
            "Źródła",
            self.SOURCE_OPTIONS,
            set(filters.get("sources", [])),
            self._source_checks,
        ))
        root_layout.addWidget(container)

    def _create_checkbox_group(
        self,
        title: str,
        options: list[str],
        enabled: set[str],
        target: dict[str, QCheckBox],
    ) -> QWidget:
        group_widget = QWidget()
        group_layout = QVBoxLayout(group_widget)
        group_layout.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight:bold;")
        group_layout.addWidget(title_label)

        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(14)
        for option in options:
            chk = QCheckBox(option)
            chk.setChecked(option in enabled)
            chk.toggled.connect(self._save_filters)
            target[option] = chk
            row_layout.addWidget(chk)
        row_layout.addStretch(1)
        group_layout.addLayout(row_layout)
        return group_widget

    def _save_filters(self) -> None:
        filters = {
            "groups": [name for name, box in self._group_checks.items() if box.isChecked()],
            "levels": [name for name, box in self._level_checks.items() if box.isChecked()],
            "sources": [name for name, box in self._source_checks.items() if box.isChecked()],
        }
        self.mw.update_log_filters(filters)

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
