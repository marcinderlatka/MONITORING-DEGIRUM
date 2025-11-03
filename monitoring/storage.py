"""Persistent storage helpers for alerts and recordings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List

from .config import ALERTS_HISTORY_PATH, RECORDINGS_CATALOG_PATH, BASE_DIR


class AlertMemory:
    """Persistent alert storage backed by a JSON file."""

    def __init__(self, path: Path | str = ALERTS_HISTORY_PATH, max_items: int = 5000) -> None:
        self.path = Path(path)
        self.max_items = max_items
        self.items: List[dict] = []
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
        try:
            payload = json.dumps(self.items[-self.max_items :], indent=2)
            self.path.write_text(payload, encoding="utf-8")
        except Exception as exc:  # pragma: no cover - log to stdout for now
            print("Nie udało się zapisać historii alertów:", exc)

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
    if "file" in item:
        item["file"] = str(path)
    return item


def load_recordings_catalog(path: Path | str = RECORDINGS_CATALOG_PATH) -> List[dict]:
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


def save_recordings_catalog(entries: Iterable[dict], path: Path | str = RECORDINGS_CATALOG_PATH) -> None:
    catalog_path = Path(path)
    try:
        payload = json.dumps(list(entries or []), indent=2)
        catalog_path.write_text(payload, encoding="utf-8")
    except Exception as exc:
        print("Nie udało się zapisać katalogu nagrań:", exc)


def update_recordings_catalog(entry: dict, path: Path | str = RECORDINGS_CATALOG_PATH) -> None:
    catalog_path = Path(path)
    filepath = entry.get("file") or entry.get("filepath")
    if not filepath:
        return
    try:
        catalog = load_recordings_catalog(catalog_path)
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
        save_recordings_catalog(filtered, catalog_path)
    except Exception as exc:
        print("Nie udało się zaktualizować katalogu nagrań:", exc)


def remove_from_recordings_catalog(paths: Iterable[str], path: Path | str = RECORDINGS_CATALOG_PATH) -> bool:
    catalog_path = Path(path)
    targets = {os.path.abspath(p) for p in paths if p}
    if not targets:
        return False
    try:
        catalog = load_recordings_catalog(catalog_path)
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
            save_recordings_catalog(remaining, catalog_path)
        return removed
    except Exception as exc:
        print("Nie udało się zaktualizować katalogu nagrań przy usuwaniu:", exc)
        return False


__all__ = [
    "AlertMemory",
    "load_recordings_catalog",
    "save_recordings_catalog",
    "update_recordings_catalog",
    "remove_from_recordings_catalog",
]
