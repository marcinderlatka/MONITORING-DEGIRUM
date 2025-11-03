"""Utilities for working with recorded alert videos.

This module contains pure-Python helpers that prepare metadata for recorded
video files.  The logic is intentionally independent from Qt so it can be
tested
without requiring a GUI environment.  The :mod:`monitoring.widgets` package
uses these helpers to power the interactive recordings browser dialog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .storage import load_recordings_catalog


@dataclass(slots=True)
class RecordingMetadata:
    """Describes a single recording discovered on disk."""

    filepath: str
    camera: str
    label: str
    confidence: float
    timestamp: float
    display_time: str
    thumb_path: str = ""
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return os.path.basename(self.filepath)


CameraDirectory = Tuple[str, str]


HistorySource = Path | str | Sequence[Mapping[str, object]] | Mapping[str, Mapping[str, object]]


def _iter_history_items(payload: object) -> Iterable[Mapping[str, object]]:
    if isinstance(payload, Mapping):
        return (item for item in payload.values() if isinstance(item, Mapping))
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return (item for item in payload if isinstance(item, Mapping))
    return ()


def load_history_metadata(history_path: HistorySource) -> Dict[str, Dict[str, object]]:
    """Load alert history metadata indexed by absolute file path."""

    metadata: Dict[str, Dict[str, object]] = {}

    if isinstance(history_path, (str, os.PathLike)):
        path = Path(history_path)
        if not path.exists():
            return metadata
        try:
            payload: object = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return metadata
    else:
        payload = history_path

    for item in _iter_history_items(payload):
        fp = item.get("filepath") or item.get("file")
        if not fp:
            continue
        metadata[os.path.abspath(str(fp))] = {
            "camera": item.get("camera", ""),
            "label": item.get("label", "unknown"),
            "confidence": item.get("confidence", 0.0),
            "time": item.get("time", ""),
            "thumb": item.get("thumb", ""),
        }
    return metadata


def _normalise_dirs(camera_dirs: Sequence[CameraDirectory]) -> List[CameraDirectory]:
    normalised: List[CameraDirectory] = []
    for name, directory in camera_dirs:
        if not name or not directory:
            continue
        normalised.append((name, os.path.abspath(directory)))
    return normalised


def camera_name_for_path(camera_dirs: Sequence[CameraDirectory], filepath: str) -> str:
    """Return the logical camera name for the given recording path."""

    abs_path = os.path.abspath(filepath)
    for name, directory in _normalise_dirs(camera_dirs):
        if abs_path.startswith(directory.rstrip(os.sep) + os.sep) or abs_path == directory:
            return name
    return ""


def _parse_timestamp_from_name(filename: str) -> Optional[_dt.datetime]:
    stem = Path(filename).stem
    # Expect format: something_YYYYMMDD_HHMMSS
    parts = stem.rsplit("_", maxsplit=2)
    if len(parts) < 2:
        return None
    date_str, time_str = parts[-2:]
    try:
        return _dt.datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    except Exception:
        return None


def _read_json_metadata(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _merge_dict(base: MutableMapping[str, object], overrides: Mapping[str, object]) -> None:
    for key, value in overrides.items():
        if value in (None, ""):
            continue
        if key == "confidence":
            try:
                base[key] = float(value)
            except Exception:
                continue
        else:
            base[key] = value


def build_recording_metadata(
    filepath: str,
    camera_dirs: Sequence[CameraDirectory],
    history_meta: Mapping[str, Mapping[str, object]] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> RecordingMetadata:
    """Create :class:`RecordingMetadata` for ``filepath``.

    The function combines information from the on-disk JSON sidecar (if
    present), alert history and any explicit overrides (for instance entries
    stored in the recordings catalog).
    """

    mp4_path = Path(filepath)
    abs_path = mp4_path.resolve()
    info: Dict[str, object] = {
        "camera": camera_name_for_path(camera_dirs, str(abs_path)),
        "label": "unknown",
        "confidence": 0.0,
        "time": "",
        "thumb": "",
        "timestamp": None,
    }

    sidecar = _read_json_metadata(mp4_path.with_suffix(".mp4.json"))
    if not sidecar:
        sidecar = _read_json_metadata(mp4_path.with_suffix(".json"))
    _merge_dict(info, sidecar)

    history = history_meta or {}
    history_item = history.get(str(abs_path))
    if history_item:
        _merge_dict(info, history_item)

    catalog_item = overrides or {}
    _merge_dict(info, catalog_item)

    timestamp = info.get("timestamp")
    dt_value: Optional[_dt.datetime]
    if timestamp not in (None, ""):
        try:
            dt_value = _dt.datetime.fromtimestamp(float(timestamp))
        except Exception:
            dt_value = None
    else:
        dt_value = None

    if dt_value is None and info.get("time"):
        try:
            dt_value = _dt.datetime.strptime(str(info["time"]), "%Y-%m-%d %H:%M:%S")
        except Exception:
            dt_value = None

    if dt_value is None:
        dt_value = _parse_timestamp_from_name(mp4_path.name)

    if dt_value is None:
        try:
            dt_value = _dt.datetime.fromtimestamp(mp4_path.stat().st_mtime)
        except Exception:
            dt_value = _dt.datetime.fromtimestamp(0)

    timestamp_float = dt_value.timestamp()
    info["timestamp"] = timestamp_float
    info.setdefault("time", dt_value.strftime("%Y-%m-%d %H:%M:%S"))

    return RecordingMetadata(
        filepath=str(abs_path),
        camera=str(info.get("camera", "")),
        label=str(info.get("label", "unknown")),
        confidence=float(info.get("confidence", 0.0) or 0.0),
        timestamp=timestamp_float,
        display_time=str(info.get("time", "")),
        thumb_path=str(info.get("thumb", "")),
        extra={k: v for k, v in info.items() if k not in {"camera", "label", "confidence", "time", "thumb", "timestamp"}},
    )


def iter_catalog_entries(
    camera_dirs: Sequence[CameraDirectory],
    history_meta: Mapping[str, Mapping[str, object]] | None = None,
) -> Iterator[RecordingMetadata]:
    """Yield metadata for recordings listed in the catalog file."""

    history = dict(history_meta or {})
    for raw_entry in load_recordings_catalog():
        if not isinstance(raw_entry, Mapping):
            continue
        filepath = raw_entry.get("filepath") or raw_entry.get("file")
        if not filepath:
            continue
        yield build_recording_metadata(filepath, camera_dirs, history_meta=history, overrides=raw_entry)


def walk_recordings(camera_dirs: Sequence[CameraDirectory]) -> Iterator[Path]:
    """Iterate over MP4 files discovered under the provided directories."""

    seen: set[str] = set()
    for _name, directory in _normalise_dirs(camera_dirs):
        if not os.path.isdir(directory):
            continue
        for root, _dirs, files in os.walk(directory):
            for filename in files:
                if not filename.lower().endswith(".mp4"):
                    continue
                resolved = os.path.abspath(os.path.join(root, filename))
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield Path(resolved)


def discover_recordings(
    camera_dirs: Sequence[CameraDirectory],
    history_path: Path | str,
) -> Iterator[RecordingMetadata]:
    """Iterate over :class:`RecordingMetadata` instances for on-disk files."""

    history = load_history_metadata(history_path)
    for path in walk_recordings(camera_dirs):
        yield build_recording_metadata(str(path), camera_dirs, history_meta=history)


__all__ = [
    "CameraDirectory",
    "RecordingMetadata",
    "build_recording_metadata",
    "camera_name_for_path",
    "discover_recordings",
    "iter_catalog_entries",
    "load_history_metadata",
    "walk_recordings",
]
