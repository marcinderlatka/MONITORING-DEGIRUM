"""Persistent storage helpers for alerts and recordings."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock, Timer
from typing import Iterable, List

from .config import ALERTS_HISTORY_PATH, BASE_DIR, RECORDINGS_CATALOG_PATH


class _DebouncedJsonWriter:
    def __init__(self, path: Path | str, debounce_seconds: float = 1.0) -> None:
        self.path = Path(path)
        self.debounce_seconds = debounce_seconds
        self._timer: Timer | None = None
        self._lock = Lock()
        self._pending: object | None = None

    def schedule(self, payload: object) -> None:
        with self._lock:
            self._pending = payload
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
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - I/O errors
            print("Nie udało się zapisać JSON:", exc)


class AlertMemory:
    """Persistent alert storage backed by a JSON file."""

    def __init__(self, path: Path | str = ALERTS_HISTORY_PATH, max_items: int = 5000) -> None:
        self.path = Path(path)
        self.max_items = max_items
        self.items: List[dict] = []
        self._writer = _DebouncedJsonWriter(self.path, debounce_seconds=2.0)
        self.last_save_time = 0.0
        self.load()

    def load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.items = data[-self.max_items :]
                else:
                    self.items = []
            else:
                self.items = []
        except Exception:
            self.items = []

    def save(self) -> None:
        now = time.monotonic()
        if now - self.last_save_time < 2.0:
            return
        self.last_save_time = now
        self._writer.schedule(self.items[-self.max_items :])

    def flush(self) -> None:
        self.last_save_time = time.monotonic()
        self._writer.schedule(self.items[-self.max_items :])
        self._writer.flush()

    def add(self, alert_meta: dict) -> None:
        slim = {
            "camera": alert_meta.get("camera", ""),
            "label": alert_meta.get("label", ""),
            "confidence": float(alert_meta.get("confidence", 0.0)),
            "time": alert_meta.get("time", ""),
            "filepath": alert_meta.get("filepath", ""),
            "thumb": alert_meta.get("thumb", ""),
        }
        self.items.append(slim)
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items :]
        self.save()

    def clear(self) -> None:
        self.items = []
        self.save()

    def export_csv(self, csv_path: Path | str) -> tuple[bool, str | None]:
        fields = ["time", "camera", "label", "confidence", "filepath"]
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                import csv

                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for item in self.items:
                    row = {key: item.get(key, "") for key in fields}
                    writer.writerow(row)
            return True, None
        except Exception as exc:
            return False, str(exc)


def _normalise_catalog_entry(entry: dict) -> dict | None:
    if not isinstance(entry, dict):
        return None
    filepath = entry.get("filepath") or entry.get("file")
    if not filepath:
        return None
    item = dict(entry)

    path = Path(filepath)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    item["filepath"] = str(path)
    item.setdefault("file", str(path))
    item.setdefault("time", item.get("event_time", ""))
    if "timestamp" not in item and item.get("event_start_ts") not in (None, ""):
        item["timestamp"] = item.get("event_start_ts")
    return item


class _RecordingsCatalog:
    def __init__(self, path: Path | str = RECORDINGS_CATALOG_PATH) -> None:
        self.path = Path(path)
        self._writer = _DebouncedJsonWriter(self.path, debounce_seconds=1.0)
        self._lock = Lock()
        self._loaded = False
        self._entries: List[dict] = []

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._entries = _load_recordings_catalog_sync(self.path)
        self._loaded = True

    def load(self) -> List[dict]:
        with self._lock:
            self._ensure_loaded()
            return [dict(item) for item in self._entries]

    def update(self, entry: dict) -> None:
        filepath = entry.get("file") or entry.get("filepath")
        if not filepath:
            return
        abs_target = os.path.abspath(filepath)
        with self._lock:
            self._ensure_loaded()
            filtered: List[dict] = []
            for item in self._entries:
                fp = item.get("filepath") or item.get("file")
                if fp and os.path.abspath(fp) == abs_target:
                    continue
                filtered.append(item)
            new_entry = dict(entry)
            new_entry.setdefault("filepath", filepath)
            filtered.append(new_entry)
            self._entries = filtered
            self._writer.schedule(self._entries)

    def remove(self, paths: Iterable[str]) -> bool:
        targets = {os.path.abspath(p) for p in paths if p}
        if not targets:
            return False
        with self._lock:
            self._ensure_loaded()
            remaining: List[dict] = []
            removed = False
            for item in self._entries:
                fp = item.get("filepath") or item.get("file")
                if fp and os.path.abspath(fp) in targets:
                    removed = True
                    continue
                remaining.append(item)
            if removed:
                self._entries = remaining
                self._writer.schedule(self._entries)
            return removed

    def save(self, entries: Iterable[dict]) -> None:
        with self._lock:
            self._entries = list(entries or [])
            self._loaded = True
            self._writer.schedule(self._entries)

    def flush(self) -> None:
        with self._lock:
            self._ensure_loaded()
            self._writer.schedule(self._entries)
        self._writer.flush()


def _load_recordings_catalog_sync(path: Path | str = RECORDINGS_CATALOG_PATH) -> List[dict]:
    catalog_path = Path(path)
    if not catalog_path.exists():
        return []
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("Nie udało się wczytać katalogu nagrań:", exc)
        return []
    if not isinstance(data, list):
        return []
    cleaned: List[dict] = []
    for entry in data:
        normalised = _normalise_catalog_entry(entry)
        if normalised:
            cleaned.append(normalised)
    return cleaned


_RECORDINGS_CATALOG = _RecordingsCatalog(RECORDINGS_CATALOG_PATH)


def load_recordings_catalog(path: Path | str = RECORDINGS_CATALOG_PATH) -> List[dict]:
    if Path(path) != RECORDINGS_CATALOG_PATH:
        return _load_recordings_catalog_sync(path)
    return _RECORDINGS_CATALOG.load()


def save_recordings_catalog(entries: Iterable[dict], path: Path | str = RECORDINGS_CATALOG_PATH) -> None:
    if Path(path) != RECORDINGS_CATALOG_PATH:
        try:
            payload = json.dumps(list(entries or []), indent=2)
            Path(path).write_text(payload, encoding="utf-8")
        except Exception as exc:
            print("Nie udało się zapisać katalogu nagrań:", exc)
        return
    _RECORDINGS_CATALOG.save(entries)


def update_recordings_catalog(entry: dict, path: Path | str = RECORDINGS_CATALOG_PATH) -> None:
    if Path(path) != RECORDINGS_CATALOG_PATH:
        catalog_path = Path(path)
        filepath = entry.get("file") or entry.get("filepath")
        if not filepath:
            return
        try:
            catalog = _load_recordings_catalog_sync(catalog_path)
            abs_target = os.path.abspath(filepath)
            filtered: List[dict] = []
            for item in catalog:
                fp = item.get("filepath") or item.get("file")
                if fp and os.path.abspath(fp) == abs_target:
                    continue
                filtered.append(item)
            new_entry = dict(entry)
            new_entry.setdefault("filepath", filepath)
            filtered.append(new_entry)
            catalog_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
        except Exception as exc:
            print("Nie udało się zaktualizować katalogu nagrań:", exc)
        return
    _RECORDINGS_CATALOG.update(entry)


def remove_from_recordings_catalog(paths: Iterable[str], path: Path | str = RECORDINGS_CATALOG_PATH) -> bool:
    if Path(path) != RECORDINGS_CATALOG_PATH:
        catalog_path = Path(path)
        targets = {os.path.abspath(p) for p in paths if p}
        if not targets:
            return False
        try:
            catalog = _load_recordings_catalog_sync(catalog_path)
            if not catalog:
                return False
            remaining: List[dict] = []
            removed = False
            for item in catalog:
                fp = item.get("filepath") or item.get("file")
                if fp and os.path.abspath(fp) in targets:
                    removed = True
                    continue
                remaining.append(item)
            if removed:
                catalog_path.write_text(json.dumps(remaining, indent=2), encoding="utf-8")
            return removed
        except Exception as exc:
            print("Nie udało się zaktualizować katalogu nagrań przy usuwaniu:", exc)
            return False
    return _RECORDINGS_CATALOG.remove(paths)


def flush_storage() -> None:
    _RECORDINGS_CATALOG.flush()


__all__ = [
    "AlertMemory",
    "load_recordings_catalog",
    "save_recordings_catalog",
    "update_recordings_catalog",
    "remove_from_recordings_catalog",
    "flush_storage",
]
