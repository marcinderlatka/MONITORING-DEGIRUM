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

from .storage import load_recordings_catalog, update_recordings_catalog


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


def build_recording_sidecar_metadata(
    *,
    camera: str,
    label: str,
    confidence: float,
    event_time: str,
    filepath: str,
    thumb: str,
    source_fps: float,
    writer_fps: float,
    detect_fps: float,
    event_start_ts: float,
    thumbnail_ts: float,
    frames_written: int,
    dropped_frames: int,
    thumbnail_mode: str,
    inference_count: int,
    positive_detection_count: int,
    record_start_mode: str = "detection_first",
    min_record_seconds: int = 0,
    required_hits_to_start_recording: int = 1,
    required_misses_to_end_detection: int = 1,
    event_end_ts: float = 0.0,
    recording_duration: float = 0.0,
    detection_count: int = 0,
    max_confidence: float = 0.0,
    avg_confidence: float = 0.0,
    stream_fps: float = 0.0,
    preview_role_at_start: str = "",
    overload_degraded_at_start: bool = False,
    measured_capture_fps: float = 0.0,
    effective_detect_fps: float = 0.0,
    preview_frames_dropped: int = 0,
    skipped_inference_cycles: int = 0,
    app_overload_mode: bool | None = None,
    recorder_queue_peak: int = 0,
) -> Dict[str, object]:
    """Build backward-compatible recording metadata payload."""
    return {
        "camera": camera,
        "label": label,
        "confidence": float(confidence),
        "time": event_time,
        "timestamp": float(event_start_ts),
        "file": filepath,
        "filepath": filepath,
        "thumb": thumb,
        "source_fps": float(source_fps),
        "writer_fps": float(writer_fps),
        "detect_fps": float(detect_fps),
        "event_start_ts": float(event_start_ts),
        "thumbnail_ts": float(thumbnail_ts),
        "thumbnail_mode": thumbnail_mode,
        "frames_written": int(frames_written),
        "dropped_frames": int(dropped_frames),
        "inference_count": int(inference_count),
        "positive_detection_count": int(positive_detection_count),
        "record_start_mode": record_start_mode,
        "min_record_seconds": int(min_record_seconds),
        "required_hits_to_start_recording": int(required_hits_to_start_recording),
        "required_misses_to_end_detection": int(required_misses_to_end_detection),
        "event_end_ts": float(event_end_ts),
        "recording_duration": float(recording_duration),
        "detection_count": int(detection_count),
        "max_confidence": float(max_confidence),
        "avg_confidence": float(avg_confidence),
        "duration": float(recording_duration),
        "stream_fps": float(stream_fps),
        "preview_role_at_start": str(preview_role_at_start),
        "overload_degraded_at_start": bool(overload_degraded_at_start),
        "measured_capture_fps": float(measured_capture_fps),
        "effective_detect_fps": float(effective_detect_fps),
        "preview_frames_dropped": int(preview_frames_dropped),
        "skipped_inference_cycles": int(skipped_inference_cycles),
        "app_overload_mode": bool(app_overload_mode) if app_overload_mode is not None else False,
        "recorder_queue_peak": int(recorder_queue_peak),
    }


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


def thumbnail_candidates_for_entry(entry: RecordingMetadata) -> List[str]:
    """Return possible thumbnail paths for a given recording entry."""

    def _resolve(path: str) -> List[str]:
        if not path:
            return []
        resolved: List[str] = [path]
        if not os.path.isabs(path):
            resolved.append(os.path.join(os.path.dirname(entry.filepath), path))
        return [os.path.abspath(p) for p in resolved]

    candidates: List[str] = []
    explicit_thumb = entry.thumb_path or str(entry.extra.get("thumb", ""))
    if explicit_thumb:
        candidates.extend(_resolve(explicit_thumb))

    base, _ext = os.path.splitext(entry.filepath)
    for suffix in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        candidates.append(os.path.abspath(f"{base}{suffix}"))

    for suffix in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        candidates.append(os.path.abspath(f"{entry.filepath}{suffix}"))

    stem, ext = os.path.splitext(entry.filepath)
    for replacement in (
        "_thumb.jpg",
        "_thumb.jpeg",
        "_preview.jpg",
        "_preview.jpeg",
        "_THUMB.JPG",
        "_THUMB.JPEG",
        "_PREVIEW.JPG",
        "_PREVIEW.JPEG",
    ):
        if ext:
            candidates.append(os.path.abspath(f"{stem}{replacement}"))

    seen: set[str] = set()
    ordered: List[str] = []
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


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
    history_path: HistorySource,
) -> Iterator[RecordingMetadata]:
    """Iterate over :class:`RecordingMetadata` instances for on-disk files."""

    history = load_history_metadata(history_path)
    for path in walk_recordings(camera_dirs):
        yield build_recording_metadata(str(path), camera_dirs, history_meta=history)


def merge_recording_entries(
    catalog_entries: Sequence[RecordingMetadata],
    disk_entries: Sequence[RecordingMetadata],
    *,
    hide_missing_files: bool = True,
) -> tuple[List[RecordingMetadata], List[RecordingMetadata]]:
    """Merge catalog + disk entries by absolute filepath.

    Returns ``(merged_entries, disk_only_entries)``.
    """

    disk_map = {os.path.abspath(entry.filepath): entry for entry in disk_entries}
    merged_map: Dict[str, RecordingMetadata] = {}

    for entry in catalog_entries:
        path = os.path.abspath(entry.filepath)
        disk_entry = disk_map.get(path)
        if hide_missing_files and disk_entry is None and not os.path.exists(path):
            continue
        if disk_entry is None:
            merged_map[path] = entry
            continue

        extra = dict(disk_entry.extra)
        extra.update(entry.extra)
        merged_map[path] = RecordingMetadata(
            filepath=path,
            camera=entry.camera or disk_entry.camera,
            label=entry.label or disk_entry.label,
            confidence=entry.confidence or disk_entry.confidence,
            timestamp=entry.timestamp or disk_entry.timestamp,
            display_time=entry.display_time or disk_entry.display_time,
            thumb_path=entry.thumb_path or disk_entry.thumb_path,
            extra=extra,
        )

    disk_only: List[RecordingMetadata] = []
    for path, entry in disk_map.items():
        if path in merged_map:
            continue
        merged_map[path] = entry
        disk_only.append(entry)

    merged = sorted(merged_map.values(), key=lambda item: item.timestamp, reverse=True)
    return merged, disk_only


def default_filter_bounds(entries: Sequence[RecordingMetadata], now: Optional[_dt.datetime] = None) -> tuple[_dt.date, _dt.date]:
    """Return a broad, data-driven default date range for recordings filters."""

    if entries:
        dates = [_dt.datetime.fromtimestamp(item.timestamp).date() for item in entries]
        return min(dates), max(dates)
    anchor = (now or _dt.datetime.now()).date()
    return anchor - _dt.timedelta(days=30), anchor


def load_recording_entries(
    camera_dirs: Sequence[CameraDirectory],
    history_source: HistorySource,
    *,
    prefer_catalog: bool = True,
    allow_disk_fallback: bool = True,
    heal_catalog: bool = True,
) -> tuple[List[RecordingMetadata], Dict[str, object]]:
    """Load recordings reliably using catalog first with disk fallback.

    Returns ``(entries, diagnostics)``.
    """

    history = load_history_metadata(history_source)
    catalog_entries = list(iter_catalog_entries(camera_dirs, history_meta=history)) if prefer_catalog else []
    valid_catalog_entries = [entry for entry in catalog_entries if os.path.exists(entry.filepath)]
    should_scan_disk = allow_disk_fallback and (
        not valid_catalog_entries
        or len(valid_catalog_entries) < len(catalog_entries)
    )

    disk_entries: List[RecordingMetadata] = []
    if should_scan_disk:
        disk_entries = list(discover_recordings(camera_dirs, history))

    merged, disk_only = merge_recording_entries(valid_catalog_entries, disk_entries, hide_missing_files=True)

    if heal_catalog and disk_only:
        for entry in disk_only:
            payload: Dict[str, object] = {
                "filepath": entry.filepath,
                "file": entry.filepath,
                "camera": entry.camera,
                "label": entry.label,
                "confidence": entry.confidence,
                "time": entry.display_time,
                "timestamp": entry.timestamp,
                "thumb": entry.thumb_path,
            }
            payload.update(entry.extra)
            update_recordings_catalog(payload)

    diagnostics: Dict[str, object] = {
        "catalog_entries": len(catalog_entries),
        "valid_catalog_entries": len(valid_catalog_entries),
        "disk_entries": len(disk_entries),
        "disk_only_entries": len(disk_only),
        "used_disk_fallback": bool(should_scan_disk and disk_entries),
        "catalog_incomplete": len(valid_catalog_entries) < len(catalog_entries),
    }
    return merged, diagnostics


__all__ = [
    "CameraDirectory",
    "RecordingMetadata",
    "build_recording_metadata",
    "build_recording_sidecar_metadata",
    "camera_name_for_path",
    "discover_recordings",
    "default_filter_bounds",
    "iter_catalog_entries",
    "load_recording_entries",
    "merge_recording_entries",
    "load_history_metadata",
    "walk_recordings",
]
