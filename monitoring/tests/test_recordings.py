from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import types

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace()

sys.path.append(str(Path(__file__).resolve().parents[2]))

from monitoring.recordings import (
    build_recording_metadata,
    camera_name_for_path,
    load_history_metadata,
    build_recording_sidecar_metadata,
)
from monitoring.runtime_helpers import compute_effective_writer_fps
from monitoring.recordings import thumbnail_candidates_for_entry


def test_build_recording_metadata_merges_sources(tmp_path):
    camera_root = tmp_path / "Cam1"
    camera_root.mkdir()
    video_path = camera_root / "alert_20240102_030405.mp4"
    video_path.write_bytes(b"")

    sidecar = video_path.with_suffix(".mp4.json")
    sidecar.write_text(json.dumps({"label": "vehicle", "custom": "value"}), encoding="utf-8")

    history = {
        os.path.abspath(str(video_path)): {
            "thumb": str(video_path) + ".jpg",
            "camera": "Cam1",
            "time": "2024-01-02 03:04:05",
        }
    }

    overrides = {"label": "person", "confidence": 0.42, "timestamp": 123.0}

    metadata = build_recording_metadata(
        str(video_path),
        [("Cam1", str(camera_root))],
        history_meta=history,
        overrides=overrides,
    )

    assert metadata.filepath == os.path.abspath(str(video_path))
    assert metadata.camera == "Cam1"
    assert metadata.label == "person"
    assert metadata.confidence == 0.42
    assert metadata.thumb_path.endswith(".jpg")
    assert metadata.display_time == "2024-01-02 03:04:05"
    assert metadata.timestamp == 123.0
    assert metadata.extra.get("custom") == "value"


def test_build_recording_metadata_preserves_extended_fields(tmp_path):
    root = tmp_path / "Cam1"
    root.mkdir()
    video = root / "clip_20240102_030405.mp4"
    video.write_bytes(b"")
    video.with_suffix(".mp4.json").write_text(
        json.dumps({"writer_fps": 5.0, "source_fps": 25.0, "thumbnail_mode": "best_detection", "frames_written": 12}),
        encoding="utf-8",
    )

    metadata = build_recording_metadata(str(video), [("Cam1", str(root))])

    assert metadata.extra["writer_fps"] == 5.0
    assert metadata.extra["source_fps"] == 25.0
    assert metadata.extra["thumbnail_mode"] == "best_detection"
    assert metadata.extra["frames_written"] == 12


def test_camera_name_for_path_handles_unknown(tmp_path):
    other = tmp_path / "Other"
    other.mkdir()
    file_path = other / "clip.mp4"
    file_path.write_bytes(b"")

    camera_dirs = [("CamA", str(tmp_path / "CamA"))]
    assert camera_name_for_path(camera_dirs, str(file_path)) == ""


def test_load_history_metadata_accepts_preloaded_items(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    snapshot = [
        {
            "filepath": str(video),
            "camera": "Cam1",
            "label": "car",
            "confidence": 0.9,
            "time": "2024-01-02 03:04:05",
            "thumb": "thumb.jpg",
        }
    ]

    metadata = load_history_metadata(snapshot)
    key = os.path.abspath(str(video))
    assert key in metadata
    assert metadata[key]["camera"] == "Cam1"
    assert metadata[key]["label"] == "car"


def test_build_recording_sidecar_metadata_contains_reliability_fields():
    payload = build_recording_sidecar_metadata(
        camera="Cam1",
        label="person",
        confidence=0.9,
        event_time="2024-01-01 00:00:00",
        filepath="/tmp/a.mp4",
        thumb="/tmp/a.mp4.jpg",
        source_fps=25.0,
        writer_fps=5.0,
        detect_fps=3.0,
        event_start_ts=1704067200.0,
        thumbnail_ts=1704067200.1,
        frames_written=12,
        dropped_frames=1,
        thumbnail_mode="first_detection",
        inference_count=30,
        positive_detection_count=8,
        event_end_ts=1704067204.0,
        recording_duration=4.0,
        detection_count=12,
        max_confidence=0.87,
        avg_confidence=0.54,
        stream_fps=24.0,
    )

    assert payload["file"] == "/tmp/a.mp4"
    assert payload["filepath"] == "/tmp/a.mp4"
    assert payload["writer_fps"] == 5.0
    assert payload["thumbnail_mode"] == "first_detection"
    assert payload["frames_written"] == 12
    assert payload["event_end_ts"] == 1704067204.0
    assert payload["recording_duration"] == 4.0
    assert payload["duration"] == 4.0
    assert payload["detection_count"] == 12


def test_metadata_duration_fields(tmp_path):
    root = tmp_path / "Cam1"
    root.mkdir()
    video = root / "clip_20240102_030405.mp4"
    video.write_bytes(b"")
    video.with_suffix(".mp4.json").write_text(
        json.dumps(
            {
                "label": "person",
                "recording_duration": 6.5,
                "duration": 6.5,
                "event_end_ts": 1704067206.5,
                "detection_count": 7,
                "max_confidence": 0.9,
                "avg_confidence": 0.6,
            }
        ),
        encoding="utf-8",
    )

    metadata = build_recording_metadata(str(video), [("Cam1", str(root))])
    assert metadata.extra["recording_duration"] == 6.5
    assert metadata.extra["duration"] == 6.5
    assert metadata.extra["event_end_ts"] == 1704067206.5


def test_thumbnail_candidates_prefer_explicit_thumb_path(tmp_path):
    video = tmp_path / "x.mp4"
    video.write_bytes(b"")
    explicit = tmp_path / "explicit.jpg"
    explicit.write_bytes(b"x")
    metadata = build_recording_metadata(str(video), [("cam", str(tmp_path))], overrides={"thumb": str(explicit)})

    candidates = thumbnail_candidates_for_entry(metadata)

    assert candidates[0] == str(explicit.resolve())
    assert str((tmp_path / "x.jpg").resolve()) in candidates


def test_effective_writer_fps_helper():
    assert compute_effective_writer_fps(5, 3.0, 25.0) == 5.0
    assert compute_effective_writer_fps(0, 3.0, 25.0) == 3.0


def test_old_metadata_without_new_fields_still_loads(tmp_path):
    root = tmp_path / "Cam1"
    root.mkdir()
    video = root / "alert_20240102_030405.mp4"
    video.write_bytes(b"")
    video.with_suffix(".mp4.json").write_text(json.dumps({"label": "person", "time": "2024-01-02 03:04:05"}), encoding="utf-8")

    metadata = build_recording_metadata(str(video), [("Cam1", str(root))])

    assert metadata.label == "person"
    assert metadata.display_time == "2024-01-02 03:04:05"
