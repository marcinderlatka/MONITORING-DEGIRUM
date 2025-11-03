from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from monitoring.recordings import build_recording_metadata, camera_name_for_path


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
