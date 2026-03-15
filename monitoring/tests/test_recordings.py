from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from monitoring.recordings import (
    build_recording_metadata,
    camera_name_for_path,
    load_history_metadata,
    build_recording_sidecar_metadata,
)


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
    )

    assert payload["file"] == "/tmp/a.mp4"
    assert payload["filepath"] == "/tmp/a.mp4"
    assert payload["writer_fps"] == 5.0
    assert payload["thumbnail_mode"] == "first_detection"
    assert payload["frames_written"] == 12
